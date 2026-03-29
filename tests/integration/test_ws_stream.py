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
) -> dict[str, Any]:
    return {
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


def _build_test_client(tmp_path: Path) -> TestClient:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def test_websocket_stream_emits_iteration_and_terminal_events(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    optimize_response = client.post("/api/v1/optimize", json=_optimize_payload())
    assert optimize_response.status_code == 200
    optimize_data = optimize_response.json()

    session_id = optimize_data["session_id"]
    trace_id = optimize_data["trace_id"]
    design_id = optimize_data["command_package"]["design_id"]

    with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as websocket:
        # First event reflects existing initial_plan history.
        first_event = websocket.receive_json()
        assert first_event["schema_version"] == "session_event.v1"
        assert first_event["event_type"] == "iteration.completed"

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
        response_1 = client.post("/api/v1/client-feedback", json=feedback_1)
        assert response_1.status_code == 200

        # feedback_evaluation + refinement_plan entries are streamed.
        second_event = websocket.receive_json()
        third_event = websocket.receive_json()
        assert second_event["event_type"] == "iteration.completed"
        assert third_event["event_type"] == "iteration.completed"

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
        response_2 = client.post("/api/v1/client-feedback", json=feedback_2)
        assert response_2.status_code == 200

        fourth_event = websocket.receive_json()
        terminal_event = websocket.receive_json()
        assert fourth_event["event_type"] == "iteration.completed"
        assert terminal_event["event_type"] == "session.completed"
        assert terminal_event["payload"]["accepted"] is True
