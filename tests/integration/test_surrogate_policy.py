from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import central_brain
import server
from app.core.session_store import SessionStore


def _optimize_payload(fallback_behavior: str) -> dict[str, Any]:
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
            "fallback_behavior": fallback_behavior,
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


def _build_test_client(tmp_path: Path) -> TestClient:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def _low_confidence_surrogate(_: Any, __: Any) -> dict[str, Any]:
    return {
        "surrogate_model_version": "heuristic_forward.v1",
        "confidence": 0.12,
        "threshold": 0.45,
        "accepted": False,
        "decision_reason": "surrogate_confidence_below_threshold",
        "estimated_metrics": {"center_frequency_ghz": 1.1, "bandwidth_mhz": 35.0},
        "target_metrics": {"center_frequency_ghz": 2.45, "bandwidth_mhz": 80.0},
        "residual": {
            "center_frequency_abs_error_ghz": 1.35,
            "bandwidth_abs_error_mhz": 45.0,
        },
        "component_scores": {
            "ann_score": 0.8,
            "freq_score": 0.0,
            "bandwidth_score": 0.1,
            "domain_support_score": 0.2,
        },
    }


def test_surrogate_low_confidence_requires_clarification(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(central_brain, "validate_with_surrogate", _low_confidence_surrogate)
    client = _build_test_client(tmp_path)

    response = client.post("/api/v1/optimize", json=_optimize_payload("require_user_confirmation"))
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "clarification_required"
    assert body["current_stage"] == "clarification_required"
    assert body["command_package"] is None
    assert len(body["warnings"]) >= 3
    assert "surrogate_confidence=" in body["warnings"][0]
    assert body["clarification"]["reason"]

    session_id = body["session_id"]
    session = server.session_store.load(session_id)
    assert session["status"] == "clarification_required"
    assert session["stop_reason"] == "requires_user_confirmation"
    assert session["current_command_package"] is None
    assert session["current_surrogate_validation"]["accepted"] is False

    queried = client.get(f"/api/v1/sessions/{session_id}")
    assert queried.status_code == 200
    queried_body = queried.json()
    assert queried_body["surrogate_summary"]["accepted"] is False
    assert queried_body["surrogate_summary"]["decision_reason"] == "surrogate_confidence_below_threshold"


def test_surrogate_low_confidence_returns_error(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(central_brain, "validate_with_surrogate", _low_confidence_surrogate)
    client = _build_test_client(tmp_path)

    response = client.post("/api/v1/optimize", json=_optimize_payload("return_error"))
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "error"
    assert body["current_stage"] == "failed"
    assert body["command_package"] is None
    assert len(body["warnings"]) >= 3
    assert "surrogate_confidence=" in body["warnings"][0]
    assert body["error"]["code"] == "LOW_SURROGATE_CONFIDENCE"

    session_id = body["session_id"]
    session = server.session_store.load(session_id)
    assert session["status"] == "error"
    assert session["stop_reason"] == "surrogate_rejected_by_policy"
    assert session["artifact_manifest"]["latest_command_package_checksum_sha256"] is None

    queried = client.get(f"/api/v1/sessions/{session_id}")
    assert queried.status_code == 200
    queried_body = queried.json()
    assert queried_body["surrogate_summary"]["accepted"] is False
    assert queried_body["surrogate_summary"]["decision_reason"] == "surrogate_confidence_below_threshold"
