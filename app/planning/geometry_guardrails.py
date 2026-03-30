"""Hard per-action geometry guardrails.

Each refinement action is allowed to move a dimension by at most a defined
absolute delta (mm) per iteration.  Dimensions not listed in an action's
table fall back to the action's ``_default`` limit.  Actions not listed in
``_GUARDRAIL_TABLE`` fall back to ``_GLOBAL_DEFAULT_MM``.

These limits are the *last* safety gate before the command package is built,
ensuring that no single LLM-driven or rule-driven refinement step can push the
design geometry by an amount a human engineer would consider unsafe to simulate.
"""
from __future__ import annotations

from typing import Any

# ── per-action guardrail tables ──────────────────────────────────────────────
# All values are maximum absolute change in mm for that dimension.
# "_default" is used for any dimension key not explicitly listed.

_GUARDRAIL_TABLE: dict[str, dict[str, float]] = {
    "increase_patch_length": {
        "patch_length_mm": 6.0,
        "_default": 1.0,
    },
    "decrease_patch_length": {
        "patch_length_mm": 6.0,
        "_default": 1.0,
    },
    "increase_patch_width": {
        "patch_width_mm": 6.0,
        "_default": 1.0,
    },
    "decrease_patch_width": {
        "patch_width_mm": 6.0,
        "_default": 1.0,
    },
    "increase_substrate_height": {
        "substrate_height_mm": 0.5,
        "substrate_length_mm": 4.0,
        "substrate_width_mm": 4.0,
        "_default": 1.0,
    },
    "decrease_substrate_height": {
        "substrate_height_mm": 0.5,
        "_default": 1.0,
    },
    "adjust_feed_geometry": {
        "feed_width_mm": 3.0,
        "feed_length_mm": 3.0,
        "feed_offset_x_mm": 2.0,
        "feed_offset_y_mm": 2.0,
        "_default": 1.0,
    },
    "generic_refinement": {
        "_default": 4.0,
    },
}

# Fallback when the action_name is not in _GUARDRAIL_TABLE at all.
_GLOBAL_DEFAULT_MM: float = 3.0


def apply_geometry_guardrails(
    before: dict[str, Any],
    after: dict[str, Any],
    action_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Clamp per-dimension change to the max allowed delta for *action_name*.

    Args:
        before: Dimension dict before refinement (original ANN output).
        after:  Dimension dict after strategy/heuristic modifications.
        action_name: The selected refinement action (e.g. ``"increase_patch_length"``).

    Returns:
        A ``(clamped_dims, clamped_fields)`` tuple where *clamped_dims* is the
        safe dimension dict and *clamped_fields* is the list of keys that were
        limited by the guardrail.
    """
    table = _GUARDRAIL_TABLE.get(action_name)
    default_max_delta: float = (
        float(table["_default"]) if isinstance(table, dict) else _GLOBAL_DEFAULT_MM
    )

    clamped: dict[str, Any] = dict(after)
    clamped_fields: list[str] = []

    for key in after:
        prev = before.get(key)
        if prev is None:
            continue
        prev_f = float(prev)
        next_f = float(after[key])
        max_delta: float = (
            float(table[key]) if isinstance(table, dict) and key in table else default_max_delta
        )
        delta = next_f - prev_f
        if abs(delta) > max_delta:
            sign = 1.0 if delta > 0.0 else -1.0
            clamped[key] = prev_f + sign * max_delta
            clamped_fields.append(key)

    return clamped, clamped_fields
