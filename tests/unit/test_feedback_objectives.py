from __future__ import annotations

from app.core.feedback_features import derive_feedback_features
from app.core.objectives import build_initial_objective_state, evaluate_objective_state
from app.core.refinement import evaluate_acceptance
from app.core.schemas import OptimizeRequest
from app.llm.session_context_builder import build_refinement_context


def _request() -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "schema_version": "optimize_request.v1",
            "user_request": "Design a circular microstrip patch antenna around 2.45 GHz.",
            "target_spec": {
                "frequency_ghz": 2.45,
                "bandwidth_mhz": 90.0,
                "antenna_family": "microstrip_patch",
                "patch_shape": "circular",
            },
            "design_constraints": {
                "allowed_materials": ["Copper (annealed)"],
                "allowed_substrates": ["Rogers RT/duroid 5880"],
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
                    "minimum_return_loss_db": -15.0,
                },
                "fallback_behavior": "best_effort",
            },
            "optimization_targets": {
                "primary": {
                    "s11": "minimize",
                    "bandwidth": "maximize",
                    "gain": "maximize",
                    "efficiency": "maximize",
                },
                "secondary": {
                    "zin_target": "50+j0",
                    "axial_ratio": "<3 dB",
                    "sll": "minimize",
                    "front_to_back": "maximize",
                },
            },
            "runtime_preferences": {
                "require_explanations": True,
                "persist_artifacts": True,
                "llm_temperature": 0.0,
                "timeout_budget_sec": 300,
                "priority": "research",
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


def test_initial_objective_state_tracks_primary_and_secondary_targets() -> None:
    state = build_initial_objective_state(_request())

    assert state["primary"]["s11"]["goal"] == "minimize"
    assert state["primary"]["bandwidth"]["goal"] == "maximize"
    assert state["secondary"]["zin"]["target"] == "50+j0"
    assert state["secondary"]["axial_ratio"]["target"] == "<3 dB"


def test_feedback_objective_state_and_refinement_context_capture_status() -> None:
    request = _request()
    feedback = {
        "actual_center_frequency_ghz": 2.50,
        "actual_bandwidth_mhz": 82.0,
        "actual_return_loss_db": -8.0,
        "actual_vswr": 2.6,
        "actual_gain_dbi": 3.2,
    }
    evaluation = evaluate_acceptance(request, feedback)
    features = derive_feedback_features(request, feedback, evaluation)
    objective_state = evaluate_objective_state(request, feedback, evaluation)

    assert objective_state["primary"]["s11"]["status"] == "not_met"
    assert objective_state["primary"]["gain"]["status"] == "not_met"
    assert objective_state["secondary"]["zin"]["status"] == "unknown"

    context = build_refinement_context(
        session={
            "session_id": "sess-obj",
            "current_iteration": 1,
            "request": request.model_dump(mode="json"),
            "history": [],
            "objective_state": objective_state,
        },
        features=features,
        candidates=[{"action": "feedline_matching_adjustment", "score": 0.8, "rule_id": "matching.poor", "rationale": "improve match"}],
    )

    assert context["objective_targets"]["primary"]["s11"] == "minimize"
    assert context["objective_state"]["primary"]["s11"]["status"] == "not_met"
    assert context["feedback_state"]["matching_state"] in {"off_target", "poor"}
