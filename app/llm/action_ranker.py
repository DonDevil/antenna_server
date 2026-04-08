from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.llm.ollama_client import generate_json
from config import CONTEXT_DIR


@lru_cache(maxsize=1)
def _load_optimization_guide() -> str:
    path = CONTEXT_DIR / "optimization_guide.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


ACTION_RANKER_SYSTEM_PROMPT = (
    "You are selecting the safest next antenna refinement action. "
    "Respect primary objectives first (S11/matching, frequency alignment, bandwidth, then gain/efficiency). "
    "Never invent actions. Return strict JSON: "
    '{"selected_action":"...","reason":"..."}'
)


def _fallback_choice(candidates: list[dict[str, Any]], reason: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))
    return {
        "selected_action": str(best.get("action")),
        "reason": reason,
    }


def choose_action_with_llm(
    *,
    context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    prompt = (
        "Select one action from candidate_actions only. "
        "Prefer the safest option with the strongest evidence from the current feedback state and objective priorities.\n"
        f"optimization_guide={_load_optimization_guide()}\n"
        f"context={context}"
    )
    parsed = generate_json(prompt=prompt, system_prompt=ACTION_RANKER_SYSTEM_PROMPT)
    if parsed is None:
        return _fallback_choice(candidates, "fallback_top_scored_candidate")

    selected_action = parsed.get("selected_action")
    if not isinstance(selected_action, str):
        return _fallback_choice(candidates, "fallback_invalid_llm_action")

    for candidate in candidates:
        if str(candidate.get("action")) == selected_action:
            return {
                "selected_action": selected_action,
                "reason": str(parsed.get("reason", "llm_selected_candidate")),
            }
    return _fallback_choice(candidates, "fallback_unknown_llm_action")