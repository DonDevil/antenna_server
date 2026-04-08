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

    request_payload = session.get("request", {})
    if not isinstance(request_payload, dict):
        request_payload = {}

    return {
        "session_id": session.get("session_id"),
        "iteration": session.get("current_iteration"),
        "target_spec": request_payload.get("target_spec", {}),
        "objective_targets": request_payload.get("optimization_targets", {}),
        "objective_state": session.get("objective_state", features.get("objective_state", {})),
        "current_dimensions": session.get("current_ann_prediction", {}).get("dimensions", {}),
        "feedback_state": {
            "frequency_state": features.get("frequency_state"),
            "bandwidth_state": features.get("bandwidth_state"),
            "matching_state": features.get("matching_state"),
            "gain_state": features.get("gain_state"),
        },
        "features": {
            "freq_error_mhz": features.get("freq_error_mhz"),
            "bandwidth_gap_mhz": features.get("bandwidth_gap_mhz"),
            "return_loss_gap_db": features.get("return_loss_gap_db"),
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