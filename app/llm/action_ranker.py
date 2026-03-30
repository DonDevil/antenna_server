from __future__ import annotations

from typing import Any

from app.llm.ollama_client import generate_json


ACTION_RANKER_SYSTEM_PROMPT = (
    "You are selecting the safest next antenna refinement action. "
    "Never invent actions. Return strict JSON: "
    '{"selected_action":"...","reason":"..."}'
)


def choose_action_with_llm(
    *,
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    prompt = (
        "Select one action from candidate_actions only. "
        "Prefer safest option with strongest evidence from current errors.\n"
        f"context={context}"
    )
    parsed = generate_json(prompt=prompt, system_prompt=ACTION_RANKER_SYSTEM_PROMPT)
    if parsed is None:
        return None

    selected_action = parsed.get("selected_action")
    if not isinstance(selected_action, str):
        return None

    for candidate in candidates:
        if str(candidate.get("action")) == selected_action:
            return {
                "selected_action": selected_action,
                "reason": str(parsed.get("reason", "llm_selected_candidate")),
            }
    return None