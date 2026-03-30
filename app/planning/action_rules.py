from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import CONTEXT_DIR


RULE_BOOK_PATH = CONTEXT_DIR / "rule_book" / "s11_refinement_rules.v1.json"
PRIORS_PATH = CONTEXT_DIR / "rule_book" / "action_effect_priors.v1.json"


def _evaluate_when(features: dict[str, Any], when: dict[str, Any]) -> bool:
    all_of = when.get("all_of", [])
    if not isinstance(all_of, list):
        return False
    for cond in all_of:
        if not isinstance(cond, dict):
            return False
        feature = str(cond.get("feature", ""))
        op = str(cond.get("op", ""))
        threshold = cond.get("value")
        current = features.get(feature)
        if current is None:
            return False
        if op == ">" and not (float(current) > float(threshold)):
            return False
        if op == ">=" and not (float(current) >= float(threshold)):
            return False
        if op == "<" and not (float(current) < float(threshold)):
            return False
        if op == "<=" and not (float(current) <= float(threshold)):
            return False
        if op == "==" and not (current == threshold):
            return False
    return True


@lru_cache(maxsize=1)
def load_rule_book() -> dict[str, Any]:
    if not RULE_BOOK_PATH.exists():
        raise FileNotFoundError(f"Missing rule book: {RULE_BOOK_PATH}")
    return json.loads(RULE_BOOK_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_action_priors() -> dict[str, Any]:
    if not PRIORS_PATH.exists():
        return {}
    payload = json.loads(PRIORS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload.get("priors", {}) if isinstance(payload.get("priors"), dict) else {}


def rank_rule_candidates(features: dict[str, Any]) -> list[dict[str, Any]]:
    book = load_rule_book()
    priors = load_action_priors()
    rules = book.get("rules", [])
    candidates: list[dict[str, Any]] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        when = rule.get("when", {})
        if not isinstance(when, dict) or not _evaluate_when(features, when):
            continue

        for cand in rule.get("candidates", []):
            if not isinstance(cand, dict):
                continue
            action = str(cand.get("action", ""))
            if not action:
                continue
            prior = float(priors.get(action, 0.0))
            base = float(cand.get("base_score", 0.0))
            score = max(0.0, min(1.0, base + prior))
            candidates.append(
                {
                    "rule_id": str(rule.get("id", "unknown_rule")),
                    "action": action,
                    "score": score,
                    "rationale": str(cand.get("rationale", "rule_selected")),
                    "strategy": cand.get("strategy", {}),
                }
            )

    if not candidates:
        candidates.append(
            {
                "rule_id": "fallback.default_refinement",
                "action": "generic_refinement",
                "score": 0.3,
                "rationale": "No specific rule matched; using conservative fallback refinement.",
                "strategy": {
                    "scale": {
                        "patch_length_mm": 1.01,
                        "feed_width_mm": 1.01,
                    },
                    "offset": {},
                },
            }
        )

    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    return candidates