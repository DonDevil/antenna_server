from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import server
from app.core.session_store import SessionStore


def _optimize_payload() -> dict[str, Any]:
    return {
        "schema_version": "optimize_request.v1",
        "user_request": "Design an AMC patch antenna with 2.45 GHz center frequency and 80 MHz bandwidth.",
        "target_spec": {
            "frequency_ghz": 2.45,
            "bandwidth_mhz": 80.0,
            "antenna_family": "amc_patch",
        },
        "design_constraints": {
            "allowed_materials": ["Copper (annealed)"],
            "allowed_substrates": ["FR-4 (lossy)"],
        },
        "optimization_policy": {
            "mode": "auto_iterate",
            "max_iterations": 3,
            "stop_on_first_valid": True,
            "acceptance": {
                "center_tolerance_mhz": 20.0,
                "minimum_bandwidth_mhz": 80.0,
                "maximum_vswr": 2.0,
                "minimum_gain_dbi": 5.0,
            },
            "fallback_behavior": "best_effort",
        },
        "runtime_preferences": {
            "require_explanations": False,
            "persist_artifacts": True,
            "llm_temperature": 0.0,
            "timeout_budget_sec": 300,
            "priority": "normal",
        },
        "client_capabilities": {
            "supports_farfield_export": True,
            "supports_current_distribution_export": False,
            "supports_parameter_sweep": False,
            "max_simulation_timeout_sec": 600,
            "export_formats": ["json"],
        },
    }


def _feedback_payload(
    *,
    session_id: str,
    trace_id: str,
    design_id: str,
    iteration_index: int,
    actual_center_frequency_ghz: float,
    actual_bandwidth_mhz: float,
    actual_vswr: float,
    actual_gain_dbi: float,
    completion_requested: bool = False,
) -> dict[str, Any]:
    payload = {
        "schema_version": "client_feedback.v1",
        "session_id": session_id,
        "trace_id": trace_id,
        "design_id": design_id,
        "iteration_index": iteration_index,
        "simulation_status": "completed",
        "actual_center_frequency_ghz": actual_center_frequency_ghz,
        "actual_bandwidth_mhz": actual_bandwidth_mhz,
        "actual_return_loss_db": -18.0,
        "actual_vswr": actual_vswr,
        "actual_gain_dbi": actual_gain_dbi,
        "artifacts": {
            "s11_trace_ref": f"s11_iter{iteration_index}.json",
            "summary_metrics_ref": f"summary_iter{iteration_index}.json",
            "farfield_ref": None,
            "current_distribution_ref": None,
        },
    }
    if completion_requested:
        payload["completion_requested"] = True
    return payload


def _build_test_client(tmp_path: Path) -> TestClient:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def test_done_request_completes_session_for_qml_client_even_with_restore_offset(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    optimize_response = client.post("/api/v1/optimize", json=_optimize_payload())
    assert optimize_response.status_code == 200
    optimize_data = optimize_response.json()

    session_id = optimize_data["session_id"]
    trace_id = optimize_data["trace_id"]
    design_id = optimize_data["command_package"]["design_id"]

    # The QML client can restore a locally saved session after CST execution,
    # which leaves the local iteration one step ahead of the server until
    # feedback is submitted. A Done request should still complete the session.
    done_response = client.post(
        "/api/v1/client-feedback",
        json=_feedback_payload(
            session_id=session_id,
            trace_id=trace_id,
            design_id=design_id,
            iteration_index=1,
            actual_center_frequency_ghz=2.20,
            actual_bandwidth_mhz=40.0,
            actual_vswr=3.2,
            actual_gain_dbi=2.0,
            completion_requested=True,
        ),
    )
    assert done_response.status_code == 200
    done_data = done_response.json()
    assert done_data["status"] == "completed"
    assert done_data["accepted"] is False
    assert done_data["decision_reason"] == "user_marked_done"
    assert done_data["stop_reason"] == "user_marked_done"

    final_session = client.get(f"/api/v1/sessions/{session_id}")
    assert final_session.status_code == 200
    final_session_data = final_session.json()
    assert final_session_data["status"] == "completed"
    assert final_session_data["stop_reason"] == "user_marked_done"
    assert final_session_data["current_iteration"] == 0


def test_optimize_feedback_refine_complete_and_query(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    optimize_response = client.post("/api/v1/optimize", json=_optimize_payload())
    assert optimize_response.status_code == 200
    optimize_data = optimize_response.json()
    assert isinstance(optimize_data["objective_state"], dict)
    assert optimize_data["objective_state"]["primary"]["s11"]["status"] == "pending"
    assert optimize_data["ann_prediction"]["ann_model_version"] == "amc_patch_formula_bootstrap_v1"
    assert optimize_data["ann_prediction"]["family_parameters"]["amc_unit_cell_period_mm"] > 0.0
    assert len(optimize_data["warnings"]) >= 3
    assert "surrogate_confidence=" in optimize_data["warnings"][0]
    assert "surrogate_residuals:" in optimize_data["warnings"][1]

    session_id = optimize_data["session_id"]
    trace_id = optimize_data["trace_id"]
    design_id = optimize_data["command_package"]["design_id"]

    initial_session = client.get(f"/api/v1/sessions/{session_id}")
    assert initial_session.status_code == 200
    assert initial_session.json()["status"] == "accepted"
    assert initial_session.json()["surrogate_summary"] is not None
    assert isinstance(initial_session.json()["surrogate_validation"], dict)
    assert initial_session.json()["history_count"] == 1

    # First feedback intentionally fails acceptance to force refinement.
    feedback_1 = _feedback_payload(
        session_id=session_id,
        trace_id=trace_id,
        design_id=design_id,
        iteration_index=0,
        actual_center_frequency_ghz=2.20,
        actual_bandwidth_mhz=40.0,
        actual_vswr=3.2,
        actual_gain_dbi=2.0,
    )
    feedback_1_response = client.post("/api/v1/client-feedback", json=feedback_1)
    assert feedback_1_response.status_code == 200
    feedback_1_data = feedback_1_response.json()
    assert feedback_1_data["status"] == "refining"
    assert feedback_1_data["accepted"] is False
    assert feedback_1_data["iteration_index"] == 1
    assert feedback_1_data["decision_reason"] == "apply_refinement_strategy_due_to_unmet_acceptance"
    assert feedback_1_data["stop_reason"] is None
    assert isinstance(feedback_1_data["objective_state"], dict)
    assert feedback_1_data["objective_state"]["overall_status"] == "needs_refinement"
    assert isinstance(feedback_1_data["planning_summary"], dict)
    assert isinstance(feedback_1_data["planning_summary"]["selected_action"], str)
    assert isinstance(feedback_1_data["planning_summary"]["decision_source"], str)
    assert feedback_1_data["next_command_package"]["iteration_index"] == 1

    next_commands = [item["command"] for item in feedback_1_data["next_command_package"]["commands"]]
    assert next_commands[0] == "update_parameter"
    assert "rebuild_model" in next_commands
    assert next_commands.index("rebuild_model") < next_commands.index("run_simulation")
    assert "define_brick" not in next_commands
    assert "define_cylinder" not in next_commands

    # Second feedback satisfies acceptance to complete the session.
    feedback_2 = _feedback_payload(
        session_id=session_id,
        trace_id=trace_id,
        design_id=design_id,
        iteration_index=1,
        actual_center_frequency_ghz=2.451,
        actual_bandwidth_mhz=90.0,
        actual_vswr=1.4,
        actual_gain_dbi=5.8,
    )
    feedback_2_response = client.post("/api/v1/client-feedback", json=feedback_2)
    assert feedback_2_response.status_code == 200
    feedback_2_data = feedback_2_response.json()
    assert feedback_2_data["status"] == "completed"
    assert feedback_2_data["accepted"] is True
    assert feedback_2_data["iteration_index"] == 1
    assert feedback_2_data["decision_reason"] == "acceptance_criteria_met"
    assert feedback_2_data["stop_reason"] == "acceptance_criteria_met"

    final_session = client.get(f"/api/v1/sessions/{session_id}")
    assert final_session.status_code == 200
    final_session_data = final_session.json()
    assert final_session_data["status"] == "completed"
    assert final_session_data["stop_reason"] == "acceptance_criteria_met"
    assert final_session_data["surrogate_summary"] is not None
    assert isinstance(final_session_data["policy_runtime"], dict)
    assert final_session_data["current_iteration"] == 1
    assert final_session_data["history_count"] == 4

    stored_session = server.session_store.load(session_id)
    manifest = stored_session["artifact_manifest"]
    assert manifest["manifest_version"] == "artifact_manifest.v1"
    assert manifest["request_schema_version"] == "optimize_request.v1"
    assert manifest["command_schema_version"] == "cst_command_package.v2"
    assert manifest["ann_model_version"] == "amc_patch_formula_bootstrap_v1"
    assert manifest["latest_iteration_index"] == 1
    assert isinstance(manifest["latest_command_package_checksum_sha256"], str)
    assert len(manifest["latest_command_package_checksum_sha256"]) == 64
    assert len(manifest["history"]) == 3
    assert isinstance(manifest["latest_planning_decision"], dict)

    for entry in stored_session["history"]:
        assert "timestamp" in entry
        assert isinstance(entry["timestamp"], str)
        assert "decision_reason" in entry
        assert "stop_reason" in entry
