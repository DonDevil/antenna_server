from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ann.features import build_ann_feature_map
from app.ann.model import InverseAnnRegressor
from app.antenna.recipes import generate_recipe
from app.antenna.recipes import resolve_patch_shape
from app.core.schemas import (
    AnnPrediction,
    ClientCapabilities,
    DesignConstraints,
    DimensionPrediction,
    OptimizationPolicy,
    OptimizeRequest,
    RuntimePreferences,
    TargetSpec,
)
from config import AMC_PATCH_ANN_SETTINGS, ANN_SETTINGS, BOUNDS, RECT_PATCH_ANN_SETTINGS, WBAN_PATCH_ANN_SETTINGS


class AnnPredictor:
    def __init__(self, checkpoint_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self._dynamic_artifacts_enabled = checkpoint_path is None and metadata_path is None
        self.checkpoint_path = checkpoint_path or ANN_SETTINGS.checkpoint_path
        self.metadata_path = metadata_path or ANN_SETTINGS.metadata_path
        self._model = None
        self._meta: dict[str, Any] | None = None
        self._loaded_artifacts: dict[tuple[Path, Path], tuple[InverseAnnRegressor, dict[str, Any]]] = {}
        self._last_error: str | None = None
        self._load_lock = threading.RLock()

    def is_ready(self) -> bool:
        if self._dynamic_artifacts_enabled:
            return self._legacy_artifacts_ready() or self._rect_patch_artifacts_ready()
        return self.checkpoint_path.exists() and self.metadata_path.exists()

    def is_loaded(self) -> bool:
        if self._dynamic_artifacts_enabled:
            return bool(self._loaded_artifacts)
        return self._model is not None and self._meta is not None

    def last_error(self) -> str | None:
        return self._last_error

    def reload_artifacts(self) -> bool:
        with self._load_lock:
            self._model = None
            self._meta = None
            self._loaded_artifacts.clear()
            self._last_error = None
        return self.warm_up()

    def warm_up(self) -> bool:
        candidates = self._artifact_candidates_for_warmup()
        if not candidates:
            self._last_error = "ann_artifacts_missing"
            return False

        load_errors: list[str] = []
        loaded_any = False
        for checkpoint_path, metadata_path in candidates:
            try:
                self._load(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
                loaded_any = True
            except Exception as exc:
                load_errors.append(f"{checkpoint_path.parent.name}: {exc}")

        if not loaded_any:
            self._model = None
            self._meta = None
            self._loaded_artifacts.clear()
            self._last_error = " | ".join(load_errors) if load_errors else "ann_model_unavailable"
            return False

        self._last_error = None
        return True

    @staticmethod
    def _coerce_hidden_dims(raw_value: Any) -> tuple[int, ...] | None:
        if not isinstance(raw_value, (list, tuple)) or not raw_value:
            return None
        if not all(isinstance(value, (int, float)) for value in raw_value):
            return None
        return tuple(int(value) for value in raw_value)

    @classmethod
    def _infer_hidden_dims_from_state_dict(cls, state_dict: dict[str, Any]) -> tuple[int, ...] | None:
        layer_widths: list[tuple[int, int]] = []
        for name, value in state_dict.items():
            if not name.startswith("net.") or not name.endswith(".weight") or getattr(value, "ndim", None) != 2:
                continue
            try:
                layer_index = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            layer_widths.append((layer_index, int(value.shape[0])))
        if len(layer_widths) <= 1:
            return None
        layer_widths.sort(key=lambda item: item[0])
        return tuple(width for _, width in layer_widths[:-1])

    def _load(self, *, checkpoint_path: Path, metadata_path: Path) -> tuple[InverseAnnRegressor, dict[str, Any]]:
        with self._load_lock:
            cache_key = (checkpoint_path, metadata_path)
            if cache_key in self._loaded_artifacts:
                model, meta = self._loaded_artifacts[cache_key]
                self._model = model
                self._meta = meta
                return model, meta

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            state_dict = checkpoint["state_dict"]
            hidden_dims = (
                self._coerce_hidden_dims(checkpoint.get("hidden_dims"))
                or self._coerce_hidden_dims(meta.get("hidden_dims") if isinstance(meta, dict) else None)
                or self._infer_hidden_dims_from_state_dict(state_dict)
            )
            model = InverseAnnRegressor(
                input_dim=int(checkpoint["input_dim"]),
                output_dim=int(checkpoint["output_dim"]),
                hidden_dims=hidden_dims or (64, 128, 64),
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self._meta = meta
            self._loaded_artifacts[cache_key] = (model, meta)
            self._last_error = None
            return model, meta

    def _legacy_artifacts_ready(self) -> bool:
        return self.checkpoint_path.exists() and self.metadata_path.exists()

    @staticmethod
    def _family_artifact_settings() -> tuple[Any, ...]:
        return (RECT_PATCH_ANN_SETTINGS, AMC_PATCH_ANN_SETTINGS, WBAN_PATCH_ANN_SETTINGS)

    @classmethod
    def _family_artifacts_ready(cls, settings: Any) -> bool:
        return settings.checkpoint_path.exists() and settings.metadata_path.exists()

    def _artifact_candidates_for_warmup(self) -> list[tuple[Path, Path]]:
        if not self._dynamic_artifacts_enabled:
            return [(self.checkpoint_path, self.metadata_path)] if self._legacy_artifacts_ready() else []

        candidates: list[tuple[Path, Path]] = []
        for settings in self._family_artifact_settings():
            if self._family_artifacts_ready(settings):
                candidates.append((settings.checkpoint_path, settings.metadata_path))
        if self._legacy_artifacts_ready():
            candidates.append((self.checkpoint_path, self.metadata_path))
        return candidates

    def _artifact_candidates_for_request(self, request: OptimizeRequest) -> list[tuple[Path, Path]]:
        if not self._dynamic_artifacts_enabled:
            return [(self.checkpoint_path, self.metadata_path)] if self._legacy_artifacts_ready() else []

        candidates: list[tuple[Path, Path]] = []
        family_name = str(request.target_spec.antenna_family).strip().lower()
        patch_shape = resolve_patch_shape(request)
        feed_type = str(getattr(request.target_spec, "feed_type", "auto") or "auto").strip().lower()

        for settings in self._family_artifact_settings():
            if family_name != settings.family or patch_shape != settings.patch_shape or not self._family_artifacts_ready(settings):
                continue
            if getattr(settings, "feed_type", None) is not None and feed_type not in {"auto", str(settings.feed_type)}:
                continue
            candidates.append((settings.checkpoint_path, settings.metadata_path))

        if self._legacy_artifacts_ready():
            candidates.append((self.checkpoint_path, self.metadata_path))
        return candidates

    @staticmethod
    def _coerce_request(payload: OptimizeRequest | TargetSpec) -> OptimizeRequest:
        if isinstance(payload, OptimizeRequest):
            return payload
        return OptimizeRequest(
            schema_version="optimize_request.v1",
            user_request=f"Design a {payload.antenna_family} antenna at {payload.frequency_ghz} GHz",
            target_spec=payload,
            design_constraints=DesignConstraints(),
            optimization_policy=OptimizationPolicy(),
            runtime_preferences=RuntimePreferences(),
            client_capabilities=ClientCapabilities(),
        )

    @staticmethod
    def _recipe_family_parameters(recipe: dict[str, Any]) -> dict[str, int | float | str | bool]:
        raw = recipe.get("family_parameters", {}) if isinstance(recipe.get("family_parameters"), dict) else {}
        clean: dict[str, int | float | str | bool] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                clean[str(key)] = value
            elif isinstance(value, int):
                clean[str(key)] = int(value)
            elif isinstance(value, float):
                clean[str(key)] = float(value)
            elif isinstance(value, str):
                clean[str(key)] = value
        return clean

    @classmethod
    def _merge_family_parameters(
        cls,
        recipe: dict[str, Any],
        model_values: dict[str, float],
        metadata_bounds: dict[str, Any] | None = None,
    ) -> dict[str, int | float | str | bool]:
        family_parameters = cls._recipe_family_parameters(recipe)
        for name, predicted in model_values.items():
            if name in ANN_SETTINGS.output_columns:
                continue
            value = float(predicted)
            bounds = cls._resolve_output_bounds(name, metadata_bounds)
            if bounds is not None:
                value = max(bounds[0], min(bounds[1], value))
            template = family_parameters.get(name)
            if isinstance(template, int) and not isinstance(template, bool):
                family_parameters[name] = int(round(value))
            else:
                family_parameters[name] = float(value)
        return family_parameters

    @staticmethod
    def _baseline_result(request: OptimizeRequest, recipe: dict[str, Any], *, model_version: str, confidence: float, hint: str) -> AnnPrediction:
        dims = dict(recipe["dimensions"])
        return AnnPrediction(
            ann_model_version=model_version,
            confidence=confidence,
            dimensions=DimensionPrediction(**dims),
            recipe_name=str(recipe.get("recipe_name")),
            patch_shape=str(recipe.get("patch_shape")),
            optimizer_hint=hint,
            family_parameters=AnnPredictor._recipe_family_parameters(recipe),
        )

    @staticmethod
    def _resolve_output_bounds(
        name: str,
        metadata_bounds: dict[str, Any] | None,
    ) -> tuple[float, float] | None:
        if metadata_bounds and name in metadata_bounds:
            raw_bounds = metadata_bounds[name]
            if isinstance(raw_bounds, (list, tuple)):
                bounds_values: list[Any] = list(raw_bounds)
                if len(bounds_values) == 2 and all(isinstance(value, (int, float)) for value in bounds_values):
                    lower_bound = float(bounds_values[0])
                    upper_bound = float(bounds_values[1])
                    return lower_bound, upper_bound

        built_in_bounds: dict[str, tuple[float, float]] = {
            "patch_length_mm": BOUNDS.patch_length_mm,
            "patch_width_mm": BOUNDS.patch_width_mm,
            "patch_height_mm": BOUNDS.patch_height_mm,
            "substrate_length_mm": BOUNDS.substrate_length_mm,
            "substrate_width_mm": BOUNDS.substrate_width_mm,
            "substrate_height_mm": BOUNDS.substrate_height_mm,
            "feed_length_mm": BOUNDS.feed_length_mm,
            "feed_width_mm": BOUNDS.feed_width_mm,
            "feed_offset_x_mm": BOUNDS.feed_offset_x_mm,
            "feed_offset_y_mm": BOUNDS.feed_offset_y_mm,
        }
        return built_in_bounds.get(name)

    @classmethod
    def merge_recipe_and_model_outputs(
        cls,
        recipe: dict[str, Any],
        model_values: dict[str, float],
        mode: str,
        metadata_bounds: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        combined = {key: float(value) for key, value in dict(recipe["dimensions"]).items() if value is not None}
        for name in ANN_SETTINGS.output_columns:
            if name not in model_values:
                continue
            base = float(combined.get(name, 0.0))
            predicted = float(model_values[name])
            if mode.startswith("residual"):
                value = base + predicted
            elif mode == "selected_output_override":
                value = predicted
            else:
                value = (0.65 * base) + (0.35 * predicted)
            bounds = cls._resolve_output_bounds(name, metadata_bounds)
            if bounds is not None:
                value = max(bounds[0], min(bounds[1], value))
            elif name.endswith("_mm") and not name.startswith("feed_offset"):
                value = max(0.01, value)
            combined[name] = value

        combined["patch_radius_mm"] = max(0.01, float(combined.get("patch_width_mm", 1.0)) / 2.0)
        return combined

    def predict(self, payload: OptimizeRequest | TargetSpec) -> AnnPrediction:
        request = self._coerce_request(payload)
        recipe = generate_recipe(request)

        candidates = self._artifact_candidates_for_request(request)
        if not candidates:
            return self._baseline_result(
                request,
                recipe,
                model_version="recipe_cold_start",
                confidence=0.55,
                hint="recipe_only",
            )

        feature_map = build_ann_feature_map(request, recipe)
        for checkpoint_path, metadata_path in candidates:
            try:
                model, meta = self._load(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
                input_columns = list(meta.get("input_columns", ANN_SETTINGS.input_columns))
                output_columns = list(meta.get("output_columns", ANN_SETTINGS.output_columns))

                x = np.array([[float(feature_map.get(name, 0.0)) for name in input_columns]], dtype=np.float32)
                x_mean = np.array(meta.get("x_mean", [0.0] * len(input_columns)), dtype=np.float32)
                x_std = np.array(meta.get("x_std", [1.0] * len(input_columns)), dtype=np.float32)
                y_mean = np.array(meta.get("y_mean", [0.0] * len(output_columns)), dtype=np.float32)
                y_std = np.array(meta.get("y_std", [1.0] * len(output_columns)), dtype=np.float32)
                x_std[x_std == 0] = 1.0
                y_std[y_std == 0] = 1.0
                x_scaled = (x - x_mean) / x_std

                with torch.no_grad():
                    y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
                y = (y_scaled * y_std) + y_mean

                model_values = {name: float(y[0][idx]) for idx, name in enumerate(output_columns)}
                prediction_mode = str(meta.get("prediction_mode", "absolute_blend"))
                values = self.merge_recipe_and_model_outputs(
                    recipe,
                    model_values,
                    prediction_mode,
                    meta.get("safe_output_bounds") if isinstance(meta.get("safe_output_bounds"), dict) else None,
                )
                hint = "recipe_plus_ann_blend" if prediction_mode != "selected_output_override" else "recipe_plus_family_ann"
                return AnnPrediction(
                    ann_model_version=str(meta.get("model_version", ANN_SETTINGS.model_version)),
                    confidence=0.82,
                    dimensions=DimensionPrediction(**values),
                    recipe_name=str(recipe.get("recipe_name")),
                    patch_shape=str(recipe.get("patch_shape")),
                    optimizer_hint=hint,
                    family_parameters=self._merge_family_parameters(
                        recipe,
                        model_values,
                        meta.get("safe_output_bounds") if isinstance(meta.get("safe_output_bounds"), dict) else None,
                    ),
                )
            except Exception as exc:
                self._last_error = str(exc)

        return self._baseline_result(
            request,
            recipe,
            model_version="recipe_fallback_after_ann_error",
            confidence=0.48,
            hint="recipe_fallback",
        )
