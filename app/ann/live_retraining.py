from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from app.ann.family_ann_trainer import default_family_specs, train_family_ann_model
from app.ann.features import build_ann_feature_map
from app.ann.predictor import AnnPredictor
from app.ann.rect_patch_trainer import train_rect_patch_inverse_ann
from app.antenna.recipes import generate_recipe, resolve_patch_shape
from app.core.schemas import AnnPrediction, OptimizeRequest
from app.data.family_feedback import (
    append_amc_patch_feedback_row,
    append_wban_patch_feedback_row,
    build_amc_patch_datasets,
    build_wban_patch_datasets,
)
from app.data.rect_patch_feedback import build_rect_patch_datasets
from app.data.rect_patch_feedback_logger import append_rect_patch_feedback_row
from config import (
    AMC_PATCH_ANN_SETTINGS,
    AMC_PATCH_DATA_SETTINGS,
    DATA_DIR,
    RECT_PATCH_ANN_SETTINGS,
    RECT_PATCH_DATA_SETTINGS,
    WBAN_PATCH_ANN_SETTINGS,
    WBAN_PATCH_DATA_SETTINGS,
)


_SUPPORTED_LIVE_RETRAIN_FAMILIES: tuple[str, ...] = ("microstrip_patch", "amc_patch", "wban_patch")


def _train_family_live_ann(
    family: str,
    *,
    validated_csv: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    min_rows_for_training: int,
) -> Any:
    spec = replace(
        default_family_specs()[family],
        dataset_path=validated_csv,
        model_dir=checkpoint_path.parent,
        min_rows=max(1, int(min_rows_for_training)),
    )
    return train_family_ann_model(spec)


def _train_amc_patch_live_ann(
    *,
    validated_csv: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    min_rows_for_training: int,
) -> Any:
    return _train_family_live_ann(
        "amc_patch",
        validated_csv=validated_csv,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        min_rows_for_training=min_rows_for_training,
    )


def _train_wban_patch_live_ann(
    *,
    validated_csv: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    min_rows_for_training: int,
) -> Any:
    return _train_family_live_ann(
        "wban_patch",
        validated_csv=validated_csv,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        min_rows_for_training=min_rows_for_training,
    )


class OnlineRetrainingManager:
    def __init__(
        self,
        *,
        predictor: AnnPredictor,
        raw_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.raw_feedback_path,
        validated_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.validated_feedback_path,
        rejected_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.rejected_feedback_path,
        inverse_train_path: Path = RECT_PATCH_DATA_SETTINGS.inverse_train_path,
        forward_train_path: Path = RECT_PATCH_DATA_SETTINGS.forward_train_path,
        amc_raw_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.raw_feedback_path,
        amc_validated_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.validated_feedback_path,
        amc_rejected_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.rejected_feedback_path,
        wban_raw_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.raw_feedback_path,
        wban_validated_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.validated_feedback_path,
        wban_rejected_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.rejected_feedback_path,
        results_ledger_path: Path = DATA_DIR / "raw" / "live_results_v1.jsonl",
        state_path: Path = RECT_PATCH_ANN_SETTINGS.model_dir / "live_retrain_state.json",
        retrain_trigger_row_count: int = 50,
        min_valid_rows_for_training: int = 50,
        trainer: Callable[..., Any] = train_rect_patch_inverse_ann,
        amc_trainer: Callable[..., Any] = _train_amc_patch_live_ann,
        wban_trainer: Callable[..., Any] = _train_wban_patch_live_ann,
        async_retraining: bool = True,
    ) -> None:
        self.predictor = predictor
        self.raw_feedback_path = raw_feedback_path
        self.validated_feedback_path = validated_feedback_path
        self.rejected_feedback_path = rejected_feedback_path
        self.inverse_train_path = inverse_train_path
        self.forward_train_path = forward_train_path
        self.amc_raw_feedback_path = amc_raw_feedback_path
        self.amc_validated_feedback_path = amc_validated_feedback_path
        self.amc_rejected_feedback_path = amc_rejected_feedback_path
        self.wban_raw_feedback_path = wban_raw_feedback_path
        self.wban_validated_feedback_path = wban_validated_feedback_path
        self.wban_rejected_feedback_path = wban_rejected_feedback_path
        self.results_ledger_path = results_ledger_path
        self.state_path = state_path
        self.retrain_trigger_row_count = max(1, int(retrain_trigger_row_count))
        self.min_valid_rows_for_training = max(1, int(min_valid_rows_for_training))
        self.trainer = trainer
        self.amc_trainer = amc_trainer
        self.wban_trainer = wban_trainer
        self.async_retraining = async_retraining
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        self._state = self._load_state()

    @staticmethod
    def _default_family_state() -> dict[str, Any]:
        return {
            "last_trained_valid_rows": 0,
            "last_retrained_at": None,
            "last_error": None,
            "last_reload_ok": None,
            "retraining_in_progress": False,
        }

    @classmethod
    def _sync_legacy_state_fields(cls, state: dict[str, Any]) -> None:
        families = state.get("families", {}) if isinstance(state.get("families"), dict) else {}
        rect_state = families.get("microstrip_patch", cls._default_family_state())
        for key in ("last_trained_valid_rows", "last_retrained_at", "last_error", "last_reload_ok", "retraining_in_progress"):
            state[key] = rect_state.get(key)

    def _load_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state = loaded
            except json.JSONDecodeError:
                state = {}

        families = state.get("families", {}) if isinstance(state.get("families"), dict) else {}
        for family in _SUPPORTED_LIVE_RETRAIN_FAMILIES:
            family_state = families.get(family, {}) if isinstance(families.get(family), dict) else {}
            defaults = self._default_family_state()
            for key, value in defaults.items():
                family_state.setdefault(key, value)
            families[family] = family_state
        state["families"] = families
        self._sync_legacy_state_fields(state)
        return state

    def _save_state(self) -> None:
        self._sync_legacy_state_fields(self._state)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _family_state(self, family: str) -> dict[str, Any]:
        families = self._state.setdefault("families", {})
        family_state = families.get(family)
        if not isinstance(family_state, dict):
            family_state = self._default_family_state()
            families[family] = family_state
        return cast(dict[str, Any], family_state)

    def status(self) -> dict[str, Any]:
        with self._lock:
            families: dict[str, dict[str, Any]] = {}
            any_in_progress = False
            for family in _SUPPORTED_LIVE_RETRAIN_FAMILIES:
                worker = self._workers.get(family)
                worker_alive = worker is not None and worker.is_alive()
                family_state = dict(self._family_state(family))
                family_state["retraining_in_progress"] = bool(family_state.get("retraining_in_progress", False) or worker_alive)
                family_state["retrain_trigger_row_count"] = self.retrain_trigger_row_count
                family_state["min_valid_rows_for_training"] = self.min_valid_rows_for_training
                families[family] = family_state
                any_in_progress = any_in_progress or bool(family_state["retraining_in_progress"])

            rect_state = families["microstrip_patch"]
            return {
                "retraining_in_progress": any_in_progress,
                "last_trained_valid_rows": int(rect_state.get("last_trained_valid_rows", 0)),
                "last_retrained_at": rect_state.get("last_retrained_at"),
                "last_error": rect_state.get("last_error"),
                "last_reload_ok": rect_state.get("last_reload_ok"),
                "retrain_trigger_row_count": self.retrain_trigger_row_count,
                "min_valid_rows_for_training": self.min_valid_rows_for_training,
                "families": families,
            }

    @staticmethod
    def _normalize_feed_type(request: OptimizeRequest, patch_shape: str) -> str:
        feed_type = str(getattr(request.target_spec, "feed_type", "auto") or "auto").strip().lower()
        if feed_type == "auto" and patch_shape == "rectangular":
            return "edge"
        return feed_type

    def _supports_rect_patch_training(self, request: OptimizeRequest) -> bool:
        family = str(request.target_spec.antenna_family).strip().lower()
        patch_shape = resolve_patch_shape(request)
        feed_type = self._normalize_feed_type(request, patch_shape)
        return family == "microstrip_patch" and patch_shape == "rectangular" and feed_type == "edge"

    @staticmethod
    def _result_key(payload: dict[str, Any]) -> str:
        artifacts = cast(dict[str, Any], payload.get("artifacts", {})) if isinstance(payload.get("artifacts"), dict) else {}
        return "|".join(
            [
                str(payload.get("session_id", "")),
                str(payload.get("design_id", "")),
                str(payload.get("iteration_index", 0)),
                str(artifacts.get("s11_trace_ref", "")),
                str(artifacts.get("summary_metrics_ref", "")),
            ]
        )

    def _ledger_entry_already_logged(self, result_key: str) -> bool:
        if not self.results_ledger_path.exists() or self.results_ledger_path.stat().st_size == 0:
            return False
        with self.results_ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    existing = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if str(existing.get("result_key", "")) == result_key:
                    return True
        return False

    def _append_results_ledger(
        self,
        *,
        request: OptimizeRequest,
        payload: dict[str, Any],
        evaluation: dict[str, Any],
        dataset_family_supported: bool,
    ) -> bool:
        result_key = self._result_key(payload)
        self.results_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self._ledger_entry_already_logged(result_key):
            return False

        entry: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "result_key": result_key,
            "session_id": payload.get("session_id"),
            "trace_id": payload.get("trace_id"),
            "design_id": payload.get("design_id"),
            "iteration_index": int(payload.get("iteration_index", 0)),
            "simulation_status": payload.get("simulation_status"),
            "accepted": bool(evaluation.get("accepted", False)),
            "antenna_family": str(request.target_spec.antenna_family),
            "patch_shape": resolve_patch_shape(request),
            "dataset_family_supported": dataset_family_supported,
            "payload": payload,
        }
        with self.results_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return True

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _build_rect_patch_row(
        self,
        *,
        request: OptimizeRequest,
        ann_prediction: AnnPrediction,
        payload: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        recipe = generate_recipe(request)
        feature_map = build_ann_feature_map(request, recipe)
        dims = ann_prediction.dimensions
        artifacts = cast(dict[str, Any], payload.get("artifacts", {})) if isinstance(payload.get("artifacts"), dict) else {}
        acceptance = request.optimization_policy.acceptance
        efficiency_pct = self._safe_float(payload.get("actual_efficiency"), 0.0)
        if efficiency_pct <= 1.0:
            efficiency_pct *= 100.0

        return {
            "run_id": f"{payload.get('session_id', 'unknown')}-{payload.get('design_id', 'design')}-iter{int(payload.get('iteration_index', 0))}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "antenna_family": "microstrip_patch",
            "patch_shape": "rectangular",
            "feed_type": "edge",
            "polarization": str(getattr(request.target_spec, "polarization", "linear") or "linear"),
            "substrate_name": (request.design_constraints.allowed_substrates or ["unspecified_substrate"])[0],
            "conductor_name": (request.design_constraints.allowed_materials or ["unspecified_conductor"])[0],
            "target_frequency_ghz": float(request.target_spec.frequency_ghz),
            "target_bandwidth_mhz": float(request.target_spec.bandwidth_mhz),
            "target_minimum_gain_dbi": float(acceptance.minimum_gain_dbi),
            "target_maximum_vswr": float(acceptance.maximum_vswr),
            "target_minimum_return_loss_db": float(acceptance.minimum_return_loss_db),
            "substrate_epsilon_r": float(feature_map.get("substrate_epsilon_r", 2.2)),
            "substrate_height_mm": float(dims.substrate_height_mm),
            "patch_length_mm": float(dims.patch_length_mm),
            "patch_width_mm": float(dims.patch_width_mm),
            "patch_height_mm": float(dims.patch_height_mm),
            "substrate_length_mm": float(dims.substrate_length_mm),
            "substrate_width_mm": float(dims.substrate_width_mm),
            "feed_length_mm": float(dims.feed_length_mm),
            "feed_width_mm": float(dims.feed_width_mm),
            "feed_offset_x_mm": float(dims.feed_offset_x_mm),
            "feed_offset_y_mm": float(dims.feed_offset_y_mm),
            "actual_center_frequency_ghz": self._safe_float(payload.get("actual_center_frequency_ghz"), float(request.target_spec.frequency_ghz)),
            "actual_bandwidth_mhz": self._safe_float(payload.get("actual_bandwidth_mhz"), 0.0),
            "actual_return_loss_db": self._safe_float(payload.get("actual_return_loss_db"), 0.0),
            "actual_vswr": self._safe_float(payload.get("actual_vswr"), float(acceptance.maximum_vswr)),
            "actual_gain_dbi": self._safe_float(payload.get("actual_gain_dbi"), float(acceptance.minimum_gain_dbi)),
            "actual_radiation_efficiency_pct": efficiency_pct,
            "actual_total_efficiency_pct": efficiency_pct,
            "actual_directivity_dbi": self._safe_float(payload.get("actual_directivity_dbi"), payload.get("actual_gain_dbi", 0.0)),
            "actual_peak_theta_deg": self._safe_float(payload.get("actual_peak_theta_deg"), 0.0),
            "actual_peak_phi_deg": self._safe_float(payload.get("actual_peak_phi_deg"), 0.0),
            "actual_front_to_back_db": self._safe_float(payload.get("actual_front_to_back_db"), 0.0),
            "actual_axial_ratio_db": self._safe_float(payload.get("actual_axial_ratio_db"), 30.0),
            "accepted": bool(evaluation.get("accepted", False)),
            "solver_status": str(payload.get("simulation_status", "completed") or "completed").strip().lower(),
            "simulation_time_sec": self._safe_float(payload.get("simulation_time_sec"), 0.0),
            "notes": str(payload.get("notes") or ""),
            "farfield_artifact_path": str(artifacts.get("farfield_ref") or ""),
            "s11_artifact_path": str(artifacts.get("s11_trace_ref") or ""),
        }

    def _build_amc_patch_row(
        self,
        *,
        request: OptimizeRequest,
        ann_prediction: AnnPrediction,
        payload: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        recipe = generate_recipe(request)
        feature_map = build_ann_feature_map(request, recipe)
        dims = ann_prediction.dimensions
        family_params = dict(getattr(ann_prediction, "family_parameters", {}) or {})
        acceptance = request.optimization_policy.acceptance
        base_period = self._safe_float(family_params.get("amc_unit_cell_period_mm"), max(6.0, (float(dims.patch_width_mm) * 0.48)))
        patch_size = self._safe_float(family_params.get("amc_patch_size_mm"), max(4.0, base_period * 0.82))
        gap_mm = self._safe_float(family_params.get("amc_gap_mm"), max(0.2, base_period - patch_size))
        actual_gain = self._safe_float(payload.get("actual_gain_dbi"), float(acceptance.minimum_gain_dbi))
        actual_center = self._safe_float(payload.get("actual_center_frequency_ghz"), float(request.target_spec.frequency_ghz))
        efficiency_pct = self._safe_float(payload.get("actual_efficiency"), 0.0)
        if efficiency_pct <= 1.0:
            efficiency_pct *= 100.0

        return {
            "run_id": f"{payload.get('session_id', 'unknown')}-{payload.get('design_id', 'design')}-iter{int(payload.get('iteration_index', 0))}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "antenna_family": "amc_patch",
            "patch_shape": resolve_patch_shape(request),
            "feed_type": str(getattr(request.target_spec, 'feed_type', 'auto') or 'auto'),
            "polarization": str(getattr(request.target_spec, 'polarization', 'linear') or 'linear'),
            "substrate_name": (request.design_constraints.allowed_substrates or ['unspecified_substrate'])[0],
            "substrate_epsilon_r": float(feature_map.get('substrate_epsilon_r', 4.4)),
            "substrate_loss_tangent": self._safe_float(payload.get('substrate_loss_tangent'), 0.02),
            "substrate_height_mm": float(dims.substrate_height_mm),
            "conductor_name": (request.design_constraints.allowed_materials or ['unspecified_conductor'])[0],
            "conductor_conductivity_s_per_m": self._safe_float(payload.get('conductor_conductivity_s_per_m'), 5.8e7),
            "target_frequency_ghz": float(request.target_spec.frequency_ghz),
            "target_bandwidth_mhz": float(request.target_spec.bandwidth_mhz),
            "target_minimum_gain_dbi": float(acceptance.minimum_gain_dbi),
            "target_maximum_vswr": float(acceptance.maximum_vswr),
            "target_minimum_return_loss_db": float(acceptance.minimum_return_loss_db),
            "amc_unit_cell_period_mm": round(base_period, 4),
            "amc_patch_size_mm": round(patch_size, 4),
            "amc_gap_mm": round(gap_mm, 4),
            "amc_via_radius_mm": round(self._safe_float(family_params.get("amc_via_radius_mm"), max(0.1, base_period * 0.03)), 4),
            "amc_via_height_mm": self._safe_float(family_params.get("amc_via_height_mm"), float(dims.substrate_height_mm)),
            "amc_ground_size_mm": round(self._safe_float(family_params.get("amc_ground_size_mm"), max(float(dims.substrate_width_mm), base_period * 1.15)), 4),
            "amc_array_rows": max(1, int(round(self._safe_float(family_params.get("amc_array_rows"), float(dims.substrate_length_mm) / max(base_period, 1e-6))))),
            "amc_array_cols": max(1, int(round(self._safe_float(family_params.get("amc_array_cols"), float(dims.substrate_width_mm) / max(base_period, 1e-6))))),
            "amc_air_gap_mm": round(self._safe_float(family_params.get("amc_air_gap_mm"), max(0.0, base_period * 0.18)), 4),
            "actual_reflection_phase_center_ghz": actual_center,
            "actual_reflection_phase_bandwidth_mhz": self._safe_float(payload.get('actual_bandwidth_mhz'), 0.0),
            "actual_gain_improvement_dbi": round(max(0.0, actual_gain - float(acceptance.minimum_gain_dbi)), 4),
            "actual_back_lobe_reduction_db": self._safe_float(payload.get('actual_front_to_back_db'), 0.0),
            "actual_center_frequency_ghz": actual_center,
            "actual_bandwidth_mhz": self._safe_float(payload.get('actual_bandwidth_mhz'), 0.0),
            "actual_return_loss_db": self._safe_float(payload.get('actual_return_loss_db'), 0.0),
            "actual_vswr": self._safe_float(payload.get('actual_vswr'), float(acceptance.maximum_vswr)),
            "actual_gain_dbi": actual_gain,
            "actual_radiation_efficiency_pct": efficiency_pct,
            "actual_total_efficiency_pct": efficiency_pct,
            "actual_directivity_dbi": self._safe_float(payload.get('actual_directivity_dbi'), actual_gain),
            "accepted": bool(evaluation.get('accepted', False)),
            "solver_status": str(payload.get('simulation_status', 'completed') or 'completed').strip().lower(),
            "simulation_time_sec": self._safe_float(payload.get('simulation_time_sec'), 0.0),
            "notes": str(payload.get('notes') or ''),
        }

    def _build_wban_patch_row(
        self,
        *,
        request: OptimizeRequest,
        ann_prediction: AnnPrediction,
        payload: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        recipe = generate_recipe(request)
        feature_map = build_ann_feature_map(request, recipe)
        dims = ann_prediction.dimensions
        family_params = dict(getattr(ann_prediction, 'family_parameters', {}) or {})
        acceptance = request.optimization_policy.acceptance
        efficiency_pct = self._safe_float(payload.get('actual_efficiency'), 0.0)
        if efficiency_pct <= 1.0:
            efficiency_pct *= 100.0
        actual_center = self._safe_float(payload.get('actual_center_frequency_ghz'), float(request.target_spec.frequency_ghz))
        actual_gain = self._safe_float(payload.get('actual_gain_dbi'), float(acceptance.minimum_gain_dbi))
        detuning_mhz = abs(actual_center - float(request.target_spec.frequency_ghz)) * 1000.0

        return {
            "run_id": f"{payload.get('session_id', 'unknown')}-{payload.get('design_id', 'design')}-iter{int(payload.get('iteration_index', 0))}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "antenna_family": "wban_patch",
            "patch_shape": resolve_patch_shape(request),
            "feed_type": str(getattr(request.target_spec, 'feed_type', 'edge') or 'edge'),
            "polarization": str(getattr(request.target_spec, 'polarization', 'linear') or 'linear'),
            "substrate_name": (request.design_constraints.allowed_substrates or ['unspecified_substrate'])[0],
            "substrate_epsilon_r": float(feature_map.get('substrate_epsilon_r', 3.0)),
            "substrate_loss_tangent": self._safe_float(payload.get('substrate_loss_tangent'), 0.0013),
            "substrate_height_mm": float(dims.substrate_height_mm),
            "conductor_name": (request.design_constraints.allowed_materials or ['unspecified_conductor'])[0],
            "conductor_conductivity_s_per_m": self._safe_float(payload.get('conductor_conductivity_s_per_m'), 5.8e7),
            "target_frequency_ghz": float(request.target_spec.frequency_ghz),
            "target_bandwidth_mhz": float(request.target_spec.bandwidth_mhz),
            "target_minimum_gain_dbi": float(acceptance.minimum_gain_dbi),
            "target_maximum_vswr": float(acceptance.maximum_vswr),
            "target_minimum_return_loss_db": float(acceptance.minimum_return_loss_db),
            "patch_length_mm": float(dims.patch_length_mm),
            "patch_width_mm": float(dims.patch_width_mm),
            "patch_height_mm": float(dims.patch_height_mm),
            "substrate_length_mm": float(dims.substrate_length_mm),
            "substrate_width_mm": float(dims.substrate_width_mm),
            "feed_length_mm": float(dims.feed_length_mm),
            "feed_width_mm": float(dims.feed_width_mm),
            "feed_offset_x_mm": float(dims.feed_offset_x_mm),
            "feed_offset_y_mm": float(dims.feed_offset_y_mm),
            "body_distance_mm": self._safe_float(payload.get('body_distance_mm'), self._safe_float(family_params.get('body_distance_mm'), 4.0)),
            "bending_radius_mm": self._safe_float(payload.get('bending_radius_mm'), self._safe_float(family_params.get('bending_radius_mm'), 65.0)),
            "ground_slot_length_mm": round(self._safe_float(family_params.get('ground_slot_length_mm'), max(0.0, float(dims.patch_length_mm) * 0.28)), 4),
            "ground_slot_width_mm": round(self._safe_float(family_params.get('ground_slot_width_mm'), max(0.0, float(dims.patch_width_mm) * 0.08)), 4),
            "notch_length_mm": round(self._safe_float(family_params.get('notch_length_mm'), max(0.0, float(dims.patch_length_mm) * 0.09)), 4),
            "notch_width_mm": round(self._safe_float(family_params.get('notch_width_mm'), max(0.0, float(dims.patch_width_mm) * 0.04)), 4),
            "actual_on_body_gain_dbi": self._safe_float(payload.get('actual_on_body_gain_dbi'), actual_gain),
            "actual_off_body_gain_dbi": self._safe_float(payload.get('actual_off_body_gain_dbi'), actual_gain + 0.5),
            "actual_sar_1g_wkg": self._safe_float(payload.get('actual_sar_1g_wkg'), 0.0),
            "actual_sar_10g_wkg": self._safe_float(payload.get('actual_sar_10g_wkg'), 0.0),
            "actual_detuning_mhz": round(detuning_mhz, 4),
            "actual_efficiency_on_body_pct": efficiency_pct,
            "actual_center_frequency_ghz": actual_center,
            "actual_bandwidth_mhz": self._safe_float(payload.get('actual_bandwidth_mhz'), 0.0),
            "actual_return_loss_db": self._safe_float(payload.get('actual_return_loss_db'), 0.0),
            "actual_vswr": self._safe_float(payload.get('actual_vswr'), float(acceptance.maximum_vswr)),
            "actual_gain_dbi": actual_gain,
            "actual_radiation_efficiency_pct": efficiency_pct,
            "actual_total_efficiency_pct": efficiency_pct,
            "actual_directivity_dbi": self._safe_float(payload.get('actual_directivity_dbi'), actual_gain),
            "accepted": bool(evaluation.get('accepted', False)),
            "solver_status": str(payload.get('simulation_status', 'completed') or 'completed').strip().lower(),
            "simulation_time_sec": self._safe_float(payload.get('simulation_time_sec'), 0.0),
            "notes": str(payload.get('notes') or ''),
        }

    def _run_retraining(self, family: str, valid_rows: int) -> None:
        with self._lock:
            family_state = self._family_state(family)
            family_state["retraining_in_progress"] = True
            family_state["last_error"] = None
            self._save_state()

        try:
            if family == "microstrip_patch":
                self.trainer(
                    inverse_csv=self.inverse_train_path,
                    checkpoint_path=RECT_PATCH_ANN_SETTINGS.checkpoint_path,
                    metadata_path=RECT_PATCH_ANN_SETTINGS.metadata_path,
                    min_rows_for_training=self.min_valid_rows_for_training,
                )
            elif family == "amc_patch":
                self.amc_trainer(
                    validated_csv=self.amc_validated_feedback_path,
                    checkpoint_path=AMC_PATCH_ANN_SETTINGS.checkpoint_path,
                    metadata_path=AMC_PATCH_ANN_SETTINGS.metadata_path,
                    min_rows_for_training=self.min_valid_rows_for_training,
                )
            elif family == "wban_patch":
                self.wban_trainer(
                    validated_csv=self.wban_validated_feedback_path,
                    checkpoint_path=WBAN_PATCH_ANN_SETTINGS.checkpoint_path,
                    metadata_path=WBAN_PATCH_ANN_SETTINGS.metadata_path,
                    min_rows_for_training=self.min_valid_rows_for_training,
                )
            else:
                raise ValueError(f"Unsupported live retraining family: {family}")

            reload_ok = self.predictor.reload_artifacts()
            with self._lock:
                family_state = self._family_state(family)
                family_state["last_trained_valid_rows"] = int(valid_rows)
                family_state["last_retrained_at"] = datetime.now(timezone.utc).isoformat()
                family_state["last_reload_ok"] = bool(reload_ok)
                family_state["last_error"] = None if reload_ok else "ann_reload_failed"
        except Exception as exc:
            with self._lock:
                family_state = self._family_state(family)
                family_state["last_error"] = str(exc)
                family_state["last_reload_ok"] = False
        finally:
            with self._lock:
                family_state = self._family_state(family)
                family_state["retraining_in_progress"] = False
                self._save_state()

    def _trigger_retraining_if_needed(self, family: str, valid_rows: int) -> bool:
        with self._lock:
            if valid_rows < self.min_valid_rows_for_training:
                return False
            last_trained = int(self._family_state(family).get("last_trained_valid_rows", 0))
            if valid_rows < last_trained + self.retrain_trigger_row_count:
                return False
            worker = self._workers.get(family)
            if worker is not None and worker.is_alive():
                return False
            if self.async_retraining:
                self._workers[family] = threading.Thread(
                    target=self._run_retraining,
                    args=(family, int(valid_rows)),
                    name=f"{family.replace('_', '-')}-live-retrain",
                    daemon=True,
                )
                self._workers[family].start()
            else:
                self._run_retraining(family, int(valid_rows))
            return True

    def ingest_result(
        self,
        *,
        request: OptimizeRequest,
        ann_prediction: AnnPrediction,
        payload: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        family = str(request.target_spec.antenna_family).strip().lower()
        supported = self._supports_rect_patch_training(request) or family in {"amc_patch", "wban_patch"}
        ledger_stored = self._append_results_ledger(
            request=request,
            payload=payload,
            evaluation=evaluation,
            dataset_family_supported=supported,
        )

        if family == "amc_patch":
            try:
                row = self._build_amc_patch_row(
                    request=request,
                    ann_prediction=ann_prediction,
                    payload=payload,
                    evaluation=evaluation,
                )
                append_amc_patch_feedback_row(row, csv_path=self.amc_raw_feedback_path)
                artifacts = build_amc_patch_datasets(
                    raw_feedback_path=self.amc_raw_feedback_path,
                    validated_feedback_path=self.amc_validated_feedback_path,
                    rejected_feedback_path=self.amc_rejected_feedback_path,
                )
                retrain_triggered = self._trigger_retraining_if_needed("amc_patch", artifacts.valid_rows)
                return {
                    "stored": True,
                    "ledger_stored": ledger_stored,
                    "reason": "stored_for_future_retraining",
                    "storage_family": "amc_patch",
                    "dataset_path": str(self.amc_raw_feedback_path),
                    "valid_rows": int(artifacts.valid_rows),
                    "rejected_rows": int(artifacts.rejected_rows),
                    "retrain_triggered": retrain_triggered,
                    **self.status(),
                }
            except Exception as exc:
                return {
                    "stored": False,
                    "ledger_stored": ledger_stored,
                    "reason": f"result_ingest_failed: {exc}",
                    "storage_family": "amc_patch",
                    "dataset_path": str(self.amc_raw_feedback_path),
                    "valid_rows": 0,
                    "rejected_rows": 0,
                    "retrain_triggered": False,
                    **self.status(),
                }

        if family == "wban_patch":
            try:
                row = self._build_wban_patch_row(
                    request=request,
                    ann_prediction=ann_prediction,
                    payload=payload,
                    evaluation=evaluation,
                )
                append_wban_patch_feedback_row(row, csv_path=self.wban_raw_feedback_path)
                artifacts = build_wban_patch_datasets(
                    raw_feedback_path=self.wban_raw_feedback_path,
                    validated_feedback_path=self.wban_validated_feedback_path,
                    rejected_feedback_path=self.wban_rejected_feedback_path,
                )
                retrain_triggered = self._trigger_retraining_if_needed("wban_patch", artifacts.valid_rows)
                return {
                    "stored": True,
                    "ledger_stored": ledger_stored,
                    "reason": "stored_for_future_retraining",
                    "storage_family": "wban_patch",
                    "dataset_path": str(self.wban_raw_feedback_path),
                    "valid_rows": int(artifacts.valid_rows),
                    "rejected_rows": int(artifacts.rejected_rows),
                    "retrain_triggered": retrain_triggered,
                    **self.status(),
                }
            except Exception as exc:
                return {
                    "stored": False,
                    "ledger_stored": ledger_stored,
                    "reason": f"result_ingest_failed: {exc}",
                    "storage_family": "wban_patch",
                    "dataset_path": str(self.wban_raw_feedback_path),
                    "valid_rows": 0,
                    "rejected_rows": 0,
                    "retrain_triggered": False,
                    **self.status(),
                }

        if not self._supports_rect_patch_training(request):
            return {
                "stored": False,
                "ledger_stored": ledger_stored,
                "reason": "family_not_supported_for_live_ann_retraining",
                "valid_rows": 0,
                "rejected_rows": 0,
                "retrain_triggered": False,
                **self.status(),
            }

        try:
            row = self._build_rect_patch_row(
                request=request,
                ann_prediction=ann_prediction,
                payload=payload,
                evaluation=evaluation,
            )
            append_rect_patch_feedback_row(row, csv_path=self.raw_feedback_path)
            artifacts = build_rect_patch_datasets(
                raw_feedback_path=self.raw_feedback_path,
                validated_feedback_path=self.validated_feedback_path,
                rejected_feedback_path=self.rejected_feedback_path,
                inverse_train_path=self.inverse_train_path,
                forward_train_path=self.forward_train_path,
            )
            retrain_triggered = self._trigger_retraining_if_needed("microstrip_patch", artifacts.valid_rows)
            return {
                "stored": True,
                "ledger_stored": ledger_stored,
                "reason": "stored_for_future_retraining",
                "storage_family": "microstrip_patch",
                "dataset_path": str(self.raw_feedback_path),
                "valid_rows": int(artifacts.valid_rows),
                "rejected_rows": int(artifacts.rejected_rows),
                "retrain_triggered": retrain_triggered,
                **self.status(),
            }
        except Exception as exc:
            return {
                "stored": False,
                "ledger_stored": ledger_stored,
                "reason": f"result_ingest_failed: {exc}",
                "storage_family": "microstrip_patch",
                "dataset_path": str(self.raw_feedback_path),
                "valid_rows": 0,
                "rejected_rows": 0,
                "retrain_triggered": False,
                **self.status(),
            }
