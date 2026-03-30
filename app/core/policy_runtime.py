from __future__ import annotations

from typing import Any

from config import PLANNER_SETTINGS


def default_policy_runtime_state() -> dict[str, Any]:
    return {
        "llm_enabled_for_intent": bool(PLANNER_SETTINGS.llm_enabled_for_intent),
        "llm_enabled_for_refinement": bool(PLANNER_SETTINGS.llm_enabled_for_refinement),
        "llm_refinement_confidence_threshold": float(PLANNER_SETTINGS.llm_refinement_confidence_threshold),
        "max_llm_calls_per_session": int(PLANNER_SETTINGS.llm_max_calls_per_session),
        "max_llm_calls_per_iteration": int(PLANNER_SETTINGS.llm_max_calls_per_iteration),
        "llm_calls_total": 0,
        "llm_calls_by_iteration": {},
    }


def ensure_policy_runtime_state(session: dict[str, Any]) -> dict[str, Any]:
    state = session.get("policy_runtime")
    if isinstance(state, dict):
        return state
    state = default_policy_runtime_state()
    session["policy_runtime"] = state
    return state


def should_call_llm_for_refinement(
    session: dict[str, Any],
    *,
    iteration_index: int,
    deterministic_confidence: float,
) -> tuple[bool, str]:
    state = ensure_policy_runtime_state(session)
    if not bool(state.get("llm_enabled_for_refinement", False)):
        return False, "llm_refinement_disabled"

    threshold = float(state.get("llm_refinement_confidence_threshold", 1.0))
    if deterministic_confidence >= threshold:
        return False, "deterministic_confidence_sufficient"

    max_session = int(state.get("max_llm_calls_per_session", 0))
    total_calls = int(state.get("llm_calls_total", 0))
    if total_calls >= max_session:
        return False, "llm_session_budget_exhausted"

    by_iteration = state.get("llm_calls_by_iteration", {})
    if not isinstance(by_iteration, dict):
        by_iteration = {}
        state["llm_calls_by_iteration"] = by_iteration
    iteration_key = str(iteration_index)
    used_iteration_calls = int(by_iteration.get(iteration_key, 0))
    max_iteration = int(state.get("max_llm_calls_per_iteration", 0))
    if used_iteration_calls >= max_iteration:
        return False, "llm_iteration_budget_exhausted"

    return True, "llm_budget_available"


def record_llm_call(session: dict[str, Any], *, iteration_index: int) -> None:
    state = ensure_policy_runtime_state(session)
    state["llm_calls_total"] = int(state.get("llm_calls_total", 0)) + 1
    by_iteration = state.get("llm_calls_by_iteration", {})
    if not isinstance(by_iteration, dict):
        by_iteration = {}
        state["llm_calls_by_iteration"] = by_iteration
    iteration_key = str(iteration_index)
    by_iteration[iteration_key] = int(by_iteration.get(iteration_key, 0)) + 1