from __future__ import annotations

from app.core.feedback_features import derive_feedback_features
from app.core.policy_runtime import (
    ensure_policy_runtime_state,
    record_llm_call,
    should_call_llm_for_refinement,
)
from app.core.refinement import evaluate_acceptance
from app.core.schemas import OptimizeRequest
from app.planning.action_rules import rank_rule_candidates


def _request() -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "schema_version": "optimize_request.v1",
            "user_request": "Design a 2.45 GHz AMC patch with 80 MHz bandwidth.",
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


def test_rule_ranking_prefers_frequency_correction_when_freq_error_high() -> None:
    request = _request()
    feedback = {
        "actual_center_frequency_ghz": 2.60,
        "actual_bandwidth_mhz": 70.0,
        "actual_return_loss_db": -12.0,
        "actual_vswr": 2.6,
        "actual_gain_dbi": 4.2,
    }
    evaluation = evaluate_acceptance(request, feedback)
    features = derive_feedback_features(request, feedback, evaluation)
    candidates = rank_rule_candidates(features)

    assert candidates
    assert candidates[0]["rule_id"] in {"freq.too_high", "matching.poor", "bandwidth.shortfall"}
    assert 0.0 <= float(candidates[0]["score"]) <= 1.0


def test_policy_runtime_enforces_iteration_call_budget() -> None:
    session: dict[str, object] = {}
    state = ensure_policy_runtime_state(session)
    state["llm_enabled_for_refinement"] = True
    state["max_llm_calls_per_session"] = 2
    state["max_llm_calls_per_iteration"] = 1
    state["llm_refinement_confidence_threshold"] = 0.8

    allowed_first, reason_first = should_call_llm_for_refinement(
        session,
        iteration_index=1,
        deterministic_confidence=0.4,
    )
    assert allowed_first is True
    assert reason_first == "llm_budget_available"

    record_llm_call(session, iteration_index=1)
    allowed_second, reason_second = should_call_llm_for_refinement(
        session,
        iteration_index=1,
        deterministic_confidence=0.4,
    )
    assert allowed_second is False
    assert reason_second == "llm_iteration_budget_exhausted"
