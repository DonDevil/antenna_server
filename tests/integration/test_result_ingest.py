from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import server
from app.ann.live_retraining import OnlineRetrainingManager
from app.core.session_store import SessionStore


def _optimize_payload(family: str = "microstrip_patch") -> dict[str, Any]:
    substrate = "Rogers RT/duroid 5880"
    if family == "amc_patch":
        substrate = "FR-4 (lossy)"
    elif family == "wban_patch":
        substrate = "Rogers RO3003"

    return {
        "schema_version": "optimize_request.v1",
        "user_request": f"Design a rectangular {family} antenna at 2.45 GHz with 90 MHz bandwidth.",
        "target_spec": {
            "frequency_ghz": 2.45,
            "bandwidth_mhz": 90.0,
            "antenna_family": family,
            "patch_shape": "rectangular",
            "feed_type": "edge",
            "polarization": "linear",
        },
        "design_constraints": {
            "allowed_materials": ["Copper (annealed)"],
            "allowed_substrates": [substrate],
        },
        "optimization_policy": {
            "mode": "auto_iterate",
            "max_iterations": 3,
            "stop_on_first_valid": True,
            "acceptance": {
              "center_tolerance_mhz": 20.0,
              "minimum_bandwidth_mhz": 80.0,
              "maximum_vswr": 2.0,
              "minimum_gain_dbi": 4.0,
              "minimum_return_loss_db": -15.0
            },
            "fallback_behavior": "best_effort"
        },
        "runtime_preferences": {
            "require_explanations": False,
            "persist_artifacts": True,
            "llm_temperature": 0.0,
            "timeout_budget_sec": 300,
            "priority": "normal"
        },
        "client_capabilities": {
            "supports_farfield_export": True,
            "supports_current_distribution_export": False,
            "supports_parameter_sweep": False,
            "max_simulation_timeout_sec": 600,
            "export_formats": ["json"]
        }
    }


def _feedback_payload(*, session_id: str, trace_id: str, design_id: str) -> dict[str, Any]:
    return {
        "schema_version": "client_feedback.v1",
        "session_id": session_id,
        "trace_id": trace_id,
        "design_id": design_id,
        "iteration_index": 0,
        "simulation_status": "completed",
        "actual_center_frequency_ghz": 2.451,
        "actual_bandwidth_mhz": 91.0,
        "actual_return_loss_db": -22.0,
        "actual_vswr": 1.3,
        "actual_gain_dbi": 5.1,
        "actual_efficiency": 0.81,
        "actual_axial_ratio_db": 25.0,
        "actual_front_to_back_db": 13.2,
        "notes": "good result",
        "artifacts": {
            "s11_trace_ref": "s11_iter0.csv",
            "summary_metrics_ref": "summary_iter0.json",
            "farfield_ref": "farfield_iter0.json",
            "current_distribution_ref": None,
        },
    }


def _build_test_client(tmp_path: Path) -> tuple[TestClient, dict[str, Path]]:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store

    paths = {
        "microstrip_patch": tmp_path / "data" / "raw" / "rect_patch_feedback_v1.csv",
        "amc_patch": tmp_path / "data" / "raw" / "amc_patch_feedback_v1.csv",
        "wban_patch": tmp_path / "data" / "raw" / "wban_patch_feedback_v1.csv",
    }
    server.brain.live_retraining = OnlineRetrainingManager(
        predictor=server.brain.ann_predictor,
        raw_feedback_path=paths["microstrip_patch"],
        validated_feedback_path=tmp_path / "data" / "validated" / "rect_patch_feedback_validated_v1.csv",
        rejected_feedback_path=tmp_path / "data" / "rejected" / "rect_patch_feedback_rejected_v1.csv",
        inverse_train_path=tmp_path / "data" / "validated" / "rect_patch_inverse_train_v1.csv",
        forward_train_path=tmp_path / "data" / "validated" / "rect_patch_forward_train_v1.csv",
        amc_raw_feedback_path=paths["amc_patch"],
        amc_validated_feedback_path=tmp_path / "data" / "validated" / "amc_patch_feedback_validated_v1.csv",
        amc_rejected_feedback_path=tmp_path / "data" / "rejected" / "amc_patch_feedback_rejected_v1.csv",
        wban_raw_feedback_path=paths["wban_patch"],
        wban_validated_feedback_path=tmp_path / "data" / "validated" / "wban_patch_feedback_validated_v1.csv",
        wban_rejected_feedback_path=tmp_path / "data" / "rejected" / "wban_patch_feedback_rejected_v1.csv",
        results_ledger_path=tmp_path / "data" / "raw" / "live_results_v1.jsonl",
        state_path=tmp_path / "models" / "live_retrain_state.json",
        retrain_trigger_row_count=999,
        min_valid_rows_for_training=999,
        async_retraining=False,
    )
    return TestClient(server.app), paths


def test_result_endpoint_stores_feedback_for_future_retraining(tmp_path: Path) -> None:
    client, paths = _build_test_client(tmp_path)

    optimize_response = client.post("/api/v1/optimize", json=_optimize_payload())
    assert optimize_response.status_code == 200
    optimize_data = optimize_response.json()

    response = client.post(
        "/api/v1/result",
        json=_feedback_payload(
            session_id=optimize_data["session_id"],
            trace_id=optimize_data["trace_id"],
            design_id=optimize_data["command_package"]["design_id"],
        ),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_feedback"]["stored"] is True
    assert data["dataset_feedback"]["storage_family"] == "microstrip_patch"
    assert data["dataset_feedback"]["valid_rows"] == 1
    assert paths["microstrip_patch"].exists() is True


def test_result_endpoint_routes_amc_and_wban_results_to_family_specific_files(tmp_path: Path) -> None:
    client, paths = _build_test_client(tmp_path)

    for family in ("amc_patch", "wban_patch"):
        optimize_response = client.post("/api/v1/optimize", json=_optimize_payload(family))
        assert optimize_response.status_code == 200
        optimize_data = optimize_response.json()

        response = client.post(
            "/api/v1/result",
            json=_feedback_payload(
                session_id=optimize_data["session_id"],
                trace_id=optimize_data["trace_id"],
                design_id=optimize_data["command_package"]["design_id"],
            ),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dataset_feedback"]["stored"] is True
        assert data["dataset_feedback"]["storage_family"] == family
        assert data["dataset_feedback"]["valid_rows"] == 1
        assert paths[family].exists() is True
