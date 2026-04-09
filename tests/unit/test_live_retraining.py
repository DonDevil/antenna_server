from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from app.ann.live_retraining import OnlineRetrainingManager
from app.core.schemas import AnnPrediction, ClientCapabilities, DesignConstraints, DimensionPrediction, OptimizationPolicy, OptimizeRequest, RuntimePreferences, TargetSpec


class _DummyPredictor:
    def __init__(self) -> None:
        self.reload_calls = 0

    def reload_artifacts(self) -> bool:
        self.reload_calls += 1
        return True


def _request() -> OptimizeRequest:
    return OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request="Design a rectangular microstrip patch antenna at 2.45 GHz",
        target_spec=TargetSpec(
            frequency_ghz=2.45,
            bandwidth_mhz=90.0,
            antenna_family="microstrip_patch",
            patch_shape="rectangular",
            feed_type="edge",
            polarization="linear",
        ),
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=["Rogers RT/duroid 5880"],
        ),
        optimization_policy=OptimizationPolicy(),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )


def _ann() -> AnnPrediction:
    return AnnPrediction(
        ann_model_version="rect_patch_v1",
        confidence=0.8,
        recipe_name="rectangular_microstrip_patch",
        patch_shape="rectangular",
        optimizer_hint="recipe_plus_family_ann",
        dimensions=DimensionPrediction(
            patch_length_mm=38.0,
            patch_width_mm=47.0,
            patch_height_mm=0.035,
            patch_radius_mm=23.5,
            substrate_length_mm=54.0,
            substrate_width_mm=63.0,
            substrate_height_mm=1.6,
            feed_length_mm=14.0,
            feed_width_mm=2.1,
            feed_offset_x_mm=0.0,
            feed_offset_y_mm=-7.5,
        ),
    )


def _payload() -> dict[str, object]:
    return {
        "schema_version": "client_feedback.v1",
        "session_id": "sess-001",
        "trace_id": "trace-001",
        "design_id": "design-001",
        "iteration_index": 0,
        "simulation_status": "completed",
        "actual_center_frequency_ghz": 2.44,
        "actual_bandwidth_mhz": 92.0,
        "actual_return_loss_db": -18.5,
        "actual_vswr": 1.45,
        "actual_gain_dbi": 4.8,
        "actual_efficiency": 0.78,
        "actual_axial_ratio_db": 28.0,
        "actual_front_to_back_db": 14.0,
        "notes": "initial CST run",
        "artifacts": {
            "s11_trace_ref": "artifacts/s11_iter0.csv",
            "summary_metrics_ref": "artifacts/summary_iter0.json",
            "farfield_ref": "artifacts/farfield_iter0.json",
            "current_distribution_ref": None,
        },
    }


def test_online_retraining_manager_stores_result_row_and_can_trigger_retraining(tmp_path: Path) -> None:
    train_calls: list[int] = []

    def _train_stub(**_: object) -> object:
        train_calls.append(1)
        return None

    manager = OnlineRetrainingManager(
        predictor=cast(Any, _DummyPredictor()),
        raw_feedback_path=tmp_path / "raw.csv",
        validated_feedback_path=tmp_path / "validated.csv",
        rejected_feedback_path=tmp_path / "rejected.csv",
        inverse_train_path=tmp_path / "inverse.csv",
        forward_train_path=tmp_path / "forward.csv",
        results_ledger_path=tmp_path / "live_results_v1.jsonl",
        state_path=tmp_path / "live_retrain_state.json",
        retrain_trigger_row_count=1,
        min_valid_rows_for_training=1,
        trainer=_train_stub,
        async_retraining=False,
    )

    info = manager.ingest_result(
        request=_request(),
        ann_prediction=_ann(),
        payload=_payload(),
        evaluation={"accepted": True},
    )

    assert info["stored"] is True
    assert info["valid_rows"] == 1
    assert info["retrain_triggered"] is True
    assert len(train_calls) == 1


def _noop_trainer(**_: object) -> None:
    return None


def test_online_retraining_manager_deduplicates_ledger_entries_for_same_result(tmp_path: Path) -> None:
    manager = OnlineRetrainingManager(
        predictor=cast(Any, _DummyPredictor()),
        raw_feedback_path=tmp_path / "raw.csv",
        validated_feedback_path=tmp_path / "validated.csv",
        rejected_feedback_path=tmp_path / "rejected.csv",
        inverse_train_path=tmp_path / "inverse.csv",
        forward_train_path=tmp_path / "forward.csv",
        results_ledger_path=tmp_path / "live_results_v1.jsonl",
        state_path=tmp_path / "live_retrain_state.json",
        retrain_trigger_row_count=999,
        min_valid_rows_for_training=999,
        trainer=_noop_trainer,
        async_retraining=False,
    )

    first = manager.ingest_result(
        request=_request(),
        ann_prediction=_ann(),
        payload=_payload(),
        evaluation={"accepted": True},
    )
    second = manager.ingest_result(
        request=_request(),
        ann_prediction=_ann(),
        payload=_payload(),
        evaluation={"accepted": True},
    )

    ledger_lines = (tmp_path / "live_results_v1.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert first["ledger_stored"] is True
    assert second["ledger_stored"] is False
    assert len(ledger_lines) == 1
