from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import server
from app.core.session_store import SessionStore


def _base_payload() -> dict[str, Any]:
    return {
        "schema_version": "optimize_request.v1",
        "user_request": "Design antenna for 2.45 GHz and 80 MHz bandwidth.",
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


def _build_test_client(tmp_path: Path) -> TestClient:
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def _force_high_surrogate(_: Any, __: Any) -> dict[str, Any]:
    return {
        "surrogate_model_version": "heuristic_forward.v1",
        "confidence": 0.91,
        "threshold": 0.45,
        "accepted": True,
        "decision_reason": "surrogate_confidence_sufficient",
        "estimated_metrics": {"center_frequency_ghz": 2.44, "bandwidth_mhz": 82.0},
        "target_metrics": {"center_frequency_ghz": 2.45, "bandwidth_mhz": 80.0},
        "residual": {
            "center_frequency_abs_error_ghz": 0.01,
            "bandwidth_abs_error_mhz": 2.0,
        },
        "component_scores": {
            "ann_score": 0.8,
            "freq_score": 0.95,
            "bandwidth_score": 0.90,
            "domain_support_score": 0.80,
        },
    }


def _noop_validate_contract(_schema_key: str, _payload: dict[str, Any]) -> None:
    return None


def test_supported_family_honors_requested_materials(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("central_brain.validate_with_surrogate", _force_high_surrogate)
    client = _build_test_client(tmp_path)

    payload = _base_payload()
    payload["target_spec"]["antenna_family"] = "microstrip_patch"
    payload["design_constraints"]["allowed_substrates"] = ["FR-4 (lossy)"]

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"

    commands = body["command_package"]["commands"]
    substrate_defs = [
        c for c in commands if c["command"] == "define_material" and c["params"].get("kind") == "substrate"
    ]
    assert len(substrate_defs) == 1
    assert substrate_defs[0]["params"]["name"] == "FR-4 (lossy)"


def test_unsupported_family_returns_schema_safe_error(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Bypass schema contract to test runtime family validation path.
    monkeypatch.setattr(server, "validate_contract", _noop_validate_contract)
    monkeypatch.setattr("central_brain.validate_with_surrogate", _force_high_surrogate)
    client = _build_test_client(tmp_path)

    payload = _base_payload()
    payload["target_spec"]["antenna_family"] = "banana_patch"

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error_code"] == "FAMILY_NOT_SUPPORTED"


def test_family_constraint_violation_returns_schema_safe_error(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("central_brain.validate_with_surrogate", _force_high_surrogate)
    client = _build_test_client(tmp_path)

    payload = _base_payload()
    payload["target_spec"]["antenna_family"] = "wban_patch"
    payload["design_constraints"]["allowed_substrates"] = ["CeramicX-123"]

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error_code"] == "FAMILY_PROFILE_CONSTRAINT_FAILED"


def test_supported_non_default_materials_are_accepted_and_echoed(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("central_brain.validate_with_surrogate", _force_high_surrogate)
    client = _build_test_client(tmp_path)

    payload = _base_payload()
    payload["target_spec"]["antenna_family"] = "microstrip_patch"
    payload["design_constraints"]["allowed_materials"] = ["Silver"]
    payload["design_constraints"]["allowed_substrates"] = ["Rogers RO4350B"]

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    body = response.json()

    conductor_defs = [
        c for c in body["command_package"]["commands"] if c["command"] == "define_material" and c["params"].get("kind") == "conductor"
    ]
    substrate_defs = [
        c for c in body["command_package"]["commands"] if c["command"] == "define_material" and c["params"].get("kind") == "substrate"
    ]

    assert body["command_package"]["design_recipe"]["selected_materials"]["conductor"] == "Silver"
    assert body["command_package"]["design_recipe"]["selected_materials"]["substrate"] == "Rogers RO4350B"
    assert conductor_defs[0]["params"]["name"] == "Silver"
    assert substrate_defs[0]["params"]["name"] == "Rogers RO4350B"
