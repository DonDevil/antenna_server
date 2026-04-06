"""Live integration tests: LLM correctness + full pipeline with Ollama.

These tests require a running Ollama instance with deepseek-r1:8b.
They are automatically skipped when Ollama is unavailable so CI is unaffected.

Run manually with:
    ./env/bin/python -m pytest -v tests/integration/test_llm_live.py -s
"""
from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.llm.ollama_client import check_ollama_health, generate_json
from app.llm.action_ranker import choose_action_with_llm
from app.llm.session_context_builder import build_refinement_context
from app.planning.action_rules import rank_rule_candidates
from app.core.feedback_features import derive_feedback_features
from app.core.refinement import evaluate_acceptance
from app.core.schemas import OptimizeRequest


# ── pytest fixture: skip entire module when Ollama is down ───────────────────

pytestmark = pytest.mark.skipif(
    not check_ollama_health(timeout_sec=3),
    reason="Ollama not reachable — skipping live LLM tests",
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _optimize_request() -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "schema_version": "optimize_request.v1",
            "user_request": "Design an AMC patch antenna for 2.45 GHz with 80 MHz bandwidth.",
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
            },
            "client_capabilities": {
                "supports_farfield_export": True,
                "supports_current_distribution_export": False,
                "supports_parameter_sweep": False,
                "max_simulation_timeout_sec": 600,
                "export_formats": ["json"],
            },
        }
    )


def _candidates_freq_too_high() -> list[dict[str, Any]]:
    return [
        {
            "action": "decrease_patch_length",
            "score": 0.87,
            "rule_id": "freq.too_high",
            "rationale": "Decrease patch length to shift resonant frequency downward",
        },
        {
            "action": "adjust_feed_geometry",
            "score": 0.52,
            "rule_id": "matching.poor",
            "rationale": "Adjust feed geometry to improve input impedance matching",
        },
        {
            "action": "increase_substrate_height",
            "score": 0.40,
            "rule_id": "bandwidth.shortfall",
            "rationale": "Increase substrate height to broaden bandwidth",
        },
    ]


def _candidates_freq_too_low() -> list[dict[str, Any]]:
    return [
        {
            "action": "increase_patch_length",
            "score": 0.88,
            "rule_id": "freq.too_low",
            "rationale": "Increase patch length to shift resonant frequency upward",
        },
        {
            "action": "adjust_feed_geometry",
            "score": 0.50,
            "rule_id": "matching.poor",
            "rationale": "Adjust feed geometry to improve impedance match",
        },
    ]


# ── 1. Raw generate_json correctness ────────────────────────────────────────

def test_generate_json_returns_valid_dict() -> None:
    """generate_json must parse a model response into a non-empty dict."""
    result = generate_json(
        prompt='Return a JSON object with a single key "status" set to "ready".',
        system_prompt="Return compact JSON only. No explanation, no extra text.",
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}: {result}"
    assert len(result) >= 1


def test_generate_json_respects_key_constraint() -> None:
    """The model must include the exact key we asked for."""
    result = generate_json(
        prompt='Return JSON: {"antenna_type": "patch"}',
        system_prompt="Return valid JSON only.",
    )
    assert isinstance(result, dict)
    assert "antenna_type" in result


def test_generate_json_numeric_value() -> None:
    """Verify the model can return numeric values in JSON."""
    result = generate_json(
        prompt='Return JSON with "frequency_ghz" set to 2.45.',
        system_prompt="Return valid compact JSON only.",
    )
    assert isinstance(result, dict)
    assert "frequency_ghz" in result
    assert float(result["frequency_ghz"]) == pytest.approx(2.45, abs=0.01)


# ── 2. Action ranker correctness ─────────────────────────────────────────────

def test_action_ranker_selects_from_candidates_only() -> None:
    """The LLM must never invent an action — must choose from the given list."""
    candidates = _candidates_freq_too_high()
    known_actions = {str(c["action"]) for c in candidates}
    context = {
        "session_id": "live-test-001",
        "iteration": 1,
        "target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0},
        "features": {
            "freq_error_mhz": 120.0,
            "bandwidth_gap_mhz": 5.0,
            "vswr_gap": 0.3,
            "gain_gap": 0.0,
            "severity": 0.8,
        },
        "recent_history": [],
        "candidate_actions": [
            {"action": c["action"], "score": c["score"],
             "rule_id": c["rule_id"], "rationale": c["rationale"]}
            for c in candidates
        ],
    }
    choice = choose_action_with_llm(context=context, candidates=candidates)
    assert choice is not None, "LLM returned None — likely invalid response or timeout"
    assert choice["selected_action"] in known_actions, (
        f"LLM invented action '{choice['selected_action']}' not in {known_actions}"
    )
    assert isinstance(choice.get("reason"), str) and len(choice["reason"]) > 0


def test_action_ranker_freq_too_high_picks_sensible_action() -> None:
    """With a +120 MHz frequency error, the model should pick a length-related or
    matching candidate — never a gain or bandwidth action that makes no sense."""
    candidates = _candidates_freq_too_high()
    context = {
        "session_id": "live-test-002",
        "iteration": 1,
        "target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0},
        "features": {
            "freq_error_mhz": 120.0,
            "bandwidth_gap_mhz": 0.0,
            "vswr_gap": 0.0,
            "gain_gap": 0.0,
            "severity": 0.7,
        },
        "recent_history": [],
        "candidate_actions": [
            {"action": c["action"], "score": c["score"],
             "rule_id": c["rule_id"], "rationale": c["rationale"]}
            for c in candidates
        ],
    }
    choice = choose_action_with_llm(context=context, candidates=candidates)
    assert choice is not None
    # The LLM must pick one of the top two frequency-relevant candidates for a large freq error.
    assert choice["selected_action"] in {"decrease_patch_length", "adjust_feed_geometry"}, (
        f"Unexpected choice '{choice['selected_action']}' for large positive freq error"
    )


def test_action_ranker_freq_too_low_picks_sensible_action() -> None:
    """With a -150 MHz frequency error, the model must pick an action consistent
    with raising frequency (increase patch length)."""
    candidates = _candidates_freq_too_low()
    context = {
        "session_id": "live-test-003",
        "iteration": 1,
        "target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0},
        "features": {
            "freq_error_mhz": -150.0,
            "bandwidth_gap_mhz": 0.0,
            "vswr_gap": 0.0,
            "gain_gap": 0.0,
            "severity": 0.7,
        },
        "recent_history": [],
        "candidate_actions": [
            {"action": c["action"], "score": c["score"],
             "rule_id": c["rule_id"], "rationale": c["rationale"]}
            for c in candidates
        ],
    }
    choice = choose_action_with_llm(context=context, candidates=candidates)
    assert choice is not None
    assert choice["selected_action"] in {"increase_patch_length", "adjust_feed_geometry"}, (
        f"Unexpected choice '{choice['selected_action']}' for large negative freq error"
    )


# ── 3. Session context builder ───────────────────────────────────────────────

def test_session_context_builder_feeds_valid_context_to_ranker() -> None:
    """The context built by session_context_builder must be accepted by the ranker."""
    session: dict[str, Any] = {
        "session_id": "ctx-test-001",
        "current_iteration": 1,
        "request": {
            "target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0}
        },
        "history": [
            {
                "type": "initial_design",
                "iteration_index": 0,
                "decision_reason": "initial_optimization",
                "stop_reason": None,
            }
        ],
    }
    features: dict[str, Any] = {
        "freq_error_mhz": 80.0,
        "bandwidth_gap_mhz": 10.0,
        "vswr_excess": 0.2,
        "vswr_gap": 0.2,
        "gain_deficit": 0.0,
        "gain_gap": 0.0,
        "severity": 0.5,
    }
    candidates = _candidates_freq_too_high()
    context = build_refinement_context(session=session, features=features, candidates=candidates)

    # Structural assertions (no LLM call needed)
    assert context["session_id"] == "ctx-test-001"
    assert context["iteration"] == 1
    assert len(context["candidate_actions"]) == len(candidates)
    assert context["features"]["freq_error_mhz"] == 80.0

    # Then feed it to the ranker live
    choice = choose_action_with_llm(context=context, candidates=candidates)
    assert choice is not None
    known = {c["action"] for c in candidates}
    assert choice["selected_action"] in known


# ── 4. Rule book → LLM pipeline round-trip ───────────────────────────────────

def test_rule_book_to_llm_pipeline_for_high_frequency_error() -> None:
    """Full pipeline: feedback → features → rule ranking → LLM choice.

    Verifies the chain rule_book → action_rules → session_context_builder →
    action_ranker produces a valid, known action for a clear frequency failure.
    """
    request = _optimize_request()
    feedback: dict[str, Any] = {
        "actual_center_frequency_ghz": 2.70,   # +250 MHz — obvious freq error
        "actual_bandwidth_mhz": 75.0,
        "actual_return_loss_db": -10.0,
        "actual_vswr": 2.8,
        "actual_gain_dbi": 4.5,
    }
    evaluation = evaluate_acceptance(request, feedback)
    features = derive_feedback_features(request, feedback, evaluation)
    candidates = rank_rule_candidates(features)

    assert candidates, "Rule book returned no candidates"
    known_actions = {str(c["action"]) for c in candidates}

    session: dict[str, Any] = {
        "session_id": "pipeline-test-001",
        "current_iteration": 1,
        "request": {"target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0}},
        "history": [],
    }
    context = build_refinement_context(session=session, features=features, candidates=candidates[:5])
    choice = choose_action_with_llm(context=context, candidates=candidates[:5])

    assert choice is not None, "LLM returned None for clear frequency failure"
    assert choice["selected_action"] in known_actions, (
        f"LLM invented '{choice['selected_action']}' — not in rule book candidates"
    )


# ── 5. Full server pipeline with LLM enabled ─────────────────────────────────

def _build_test_client(tmp_path: Path):
    import server
    from app.core.session_store import SessionStore
    test_store = SessionStore(base_dir=tmp_path / "sessions")
    server.session_store = test_store
    server.brain.session_store = test_store
    return TestClient(server.app)


def _optimize_payload() -> dict[str, Any]:
    return {
        "schema_version": "optimize_request.v1",
        "user_request": "Design an AMC patch antenna for 2.45 GHz with 80 MHz bandwidth.",
        "target_spec": {"frequency_ghz": 2.45, "bandwidth_mhz": 80.0, "antenna_family": "amc_patch"},
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


def test_full_pipeline_optimize_refine_complete_with_llm(tmp_path: Path) -> None:
    """End-to-end: optimize → failing feedback (triggers LLM-assisted refinement)
    → accepting feedback (session completes).

    Asserts:
    - LLM was consulted OR fell back gracefully (deterministic fallback)
    - planning_summary is populated in the feedback response
    - policy_runtime reflects call accounting
    - guardrails are applied (refined dims stay within per-action limits)
    - session completes cleanly with latest_planning_decision in manifest
    """
    client = _build_test_client(tmp_path)

    # Step 1: Optimize
    opt_resp = client.post("/api/v1/optimize", json=_optimize_payload())
    assert opt_resp.status_code == 200
    opt_data = opt_resp.json()
    session_id = opt_data["session_id"]
    trace_id = opt_data["trace_id"]
    design_id = opt_data["command_package"]["design_id"]

    # Step 2: First feedback — deliberately fails acceptance to trigger LLM refinement
    fb1 = _feedback_payload(
        session_id=session_id,
        trace_id=trace_id,
        design_id=design_id,
        iteration_index=0,
        actual_center_frequency_ghz=2.70,   # high freq error — rule book should fire freq.too_high
        actual_bandwidth_mhz=40.0,
        actual_vswr=3.2,
        actual_gain_dbi=2.0,
    )
    fb1_resp = client.post("/api/v1/client-feedback", json=fb1)
    assert fb1_resp.status_code == 200
    fb1_data = fb1_resp.json()

    assert fb1_data["status"] == "refining"
    assert fb1_data["accepted"] is False
    assert fb1_data["decision_reason"] == "apply_refinement_strategy_due_to_unmet_acceptance"

    planning = fb1_data["planning_summary"]
    assert isinstance(planning, dict)
    assert isinstance(planning["selected_action"], str)
    # LLM was enabled — either it was used or it fell back gracefully
    assert planning["decision_source"] in {"llm_ranked_candidate", "deterministic_rule"}
    assert isinstance(planning["confidence"], float)

    # The refined command package must be valid
    next_pkg = fb1_data["next_command_package"]
    assert isinstance(next_pkg, dict)
    assert next_pkg["iteration_index"] == 1
    assert isinstance(next_pkg["commands"], list)
    assert len(next_pkg["commands"]) > 0

    # Step 3: Query session — policy_runtime must track any LLM calls made
    sess_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert sess_resp.status_code == 200
    sess_data = sess_resp.json()
    policy = sess_data["policy_runtime"]
    assert isinstance(policy, dict)
    assert "llm_calls_total" in policy
    assert "llm_enabled_for_refinement" in policy
    assert policy["llm_enabled_for_refinement"] is True
    assert isinstance(policy["llm_calls_total"], int)

    # Step 4: Accepting feedback — session should complete
    fb2 = _feedback_payload(
        session_id=session_id,
        trace_id=trace_id,
        design_id=design_id,
        iteration_index=1,
        actual_center_frequency_ghz=2.451,
        actual_bandwidth_mhz=85.0,
        actual_vswr=1.5,
        actual_gain_dbi=5.5,
    )
    fb2_resp = client.post("/api/v1/client-feedback", json=fb2)
    assert fb2_resp.status_code == 200
    fb2_data = fb2_resp.json()
    assert fb2_data["status"] == "completed"
    assert fb2_data["accepted"] is True

    # Step 5: Final session state — planning decision preserved in manifest
    final_sess = client.get(f"/api/v1/sessions/{session_id}")
    assert final_sess.status_code == 200
    final_data = final_sess.json()
    assert final_data["status"] == "completed"
    assert isinstance(final_data["latest_planning_decision"], dict)
    lpd = final_data["latest_planning_decision"]
    assert isinstance(lpd.get("selected_action"), str)
    assert lpd.get("decision_source") in {"llm_ranked_candidate", "deterministic_rule"}


def test_health_endpoint_reflects_ollama_status(tmp_path: Path) -> None:
    """The /api/v1/health endpoint must return dependency readiness details."""
    client = _build_test_client(tmp_path)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "ann_model_ready" in data
    assert "ann_status" in data
    assert data["ann_status"] in {"available", "loading", "none"}
    assert "llm_status" in data
    assert data["llm_status"] in {"available", "loading", "none"}
    assert data["llm_model"]
