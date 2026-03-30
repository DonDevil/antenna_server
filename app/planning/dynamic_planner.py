from __future__ import annotations

from typing import Any

from app.core.policy_runtime import record_llm_call, should_call_llm_for_refinement
from app.llm.action_ranker import choose_action_with_llm
from app.llm.ollama_client import check_ollama_health
from app.llm.session_context_builder import build_refinement_context
from app.planning.action_rules import rank_rule_candidates


def _select_candidate(candidates: list[dict[str, Any]], action_name: str) -> dict[str, Any] | None:
    for c in candidates:
        if str(c.get("action")) == action_name:
            return c
    return None


def plan_refinement_strategy(
    *,
    session: dict[str, Any],
    features: dict[str, Any],
    iteration_index: int,
) -> dict[str, Any]:
    candidates = rank_rule_candidates(features)
    top_candidate = candidates[0]
    deterministic_confidence = float(top_candidate.get("score", 0.0))

    should_call, llm_gate_reason = should_call_llm_for_refinement(
        session,
        iteration_index=iteration_index,
        deterministic_confidence=deterministic_confidence,
    )

    selected = top_candidate
    source = "deterministic_rule"
    llm_reason = llm_gate_reason
    llm_used = False

    if should_call:
        if not check_ollama_health():
            llm_reason = "ollama_unavailable_fast_fallback"
        else:
            context = build_refinement_context(session=session, features=features, candidates=candidates[:5])
            llm_choice = choose_action_with_llm(context=context, candidates=candidates[:5])
            if llm_choice is not None:
                chosen = _select_candidate(candidates, str(llm_choice["selected_action"]))
                if chosen is not None:
                    selected = chosen
                    source = "llm_ranked_candidate"
                    llm_reason = str(llm_choice.get("reason", "llm_selected_candidate"))
                    llm_used = True
            if llm_choice is None:
                llm_reason = "llm_unavailable_or_invalid_response"
        record_llm_call(session, iteration_index=iteration_index)

    strategy = selected.get("strategy", {})
    if not isinstance(strategy, dict):
        strategy = {}

    return {
        "selected_action": str(selected.get("action", "generic_refinement")),
        "strategy": strategy,
        "confidence": float(selected.get("score", 0.0)),
        "decision_source": source,
        "rule_id": str(selected.get("rule_id", "fallback.default_refinement")),
        "reason": str(selected.get("rationale", "rule_selected")),
        "llm_used": llm_used,
        "llm_reason": llm_reason,
        "candidate_count": len(candidates),
        "candidates": candidates[:5],
    }