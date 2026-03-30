from __future__ import annotations

from typing import Any

from config import PLANNER_SETTINGS


def build_refinement_context(
    *,
    session: dict[str, Any],
    features: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    history = session.get("history", [])
    if not isinstance(history, list):
        history = []
    max_items = max(1, int(PLANNER_SETTINGS.session_context_max_history))
    short_history = history[-max_items:]

    compact_history = []
    for entry in short_history:
        if not isinstance(entry, dict):
            continue
        compact_history.append(
            {
                "type": entry.get("type"),
                "iteration_index": entry.get("iteration_index"),
                "decision_reason": entry.get("decision_reason"),
                "stop_reason": entry.get("stop_reason"),
            }
        )

    return {
        "session_id": session.get("session_id"),
        "iteration": session.get("current_iteration"),
        "target_spec": session.get("request", {}).get("target_spec", {}),
        "features": {
            "freq_error_mhz": features.get("freq_error_mhz"),
            "bandwidth_gap_mhz": features.get("bandwidth_gap_mhz"),
            "vswr_gap": features.get("vswr_gap"),
            "gain_gap": features.get("gain_gap"),
            "severity": features.get("severity"),
        },
        "recent_history": compact_history,
        "candidate_actions": [
            {
                "action": c.get("action"),
                "score": c.get("score"),
                "rule_id": c.get("rule_id"),
                "rationale": c.get("rationale"),
            }
            for c in candidates
        ],
    }