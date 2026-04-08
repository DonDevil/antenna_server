from __future__ import annotations

import re
from typing import Any

AliasMap = dict[str, tuple[str, ...]]

from app.llm.ollama_client import generate_json
from config import OLLAMA_SETTINGS, PLANNER_SETTINGS


_NUMERIC_WITH_UNIT_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ghz|gigahertz|giga\s*hertz|mhz|megahertz|mega\s*hertz|megs?)\b",
    re.IGNORECASE,
)
_BANDWIDTH_HINT_PATTERN = re.compile(r"\b(bandwidth|bw|wide|span)\b", re.IGNORECASE)
_FREQUENCY_HINT_PATTERN = re.compile(
    r"\b(freq(?:uency)?|center(?:ed)?|resonan(?:ce|t)|operate(?:s|d|ing)?|target)\b",
    re.IGNORECASE,
)
_ALLOWED_FAMILIES = {"amc_patch", "microstrip_patch", "wban_patch"}
_ALLOWED_PATCH_SHAPES = {"auto", "rectangular", "circular"}
_SUBSTRATE_ALIASES: AliasMap = {
    "FR-4 (lossy)": ("fr-4", "fr4", "fr 4"),
    "Rogers RT/duroid 5880": ("rogers rt/duroid 5880", "rogers 5880", "duroid 5880", "rt duroid 5880"),
    "Rogers RO3003": ("rogers ro3003", "ro3003", "ro 3003"),
}
_CONDUCTOR_ALIASES: AliasMap = {
    "Copper (annealed)": ("copper", "annealed copper"),
}
_FAMILY_VALUE_ALIASES: AliasMap = {
    "amc_patch": ("amc_patch", "amc patch", "amc", "artificial magnetic conductor"),
    "microstrip_patch": ("microstrip_patch", "microstrip patch", "microstrip", "rectangular patch"),
    "wban_patch": (
        "wban_patch",
        "wban patch",
        "wban",
        "wearable patch",
        "wearable antenna",
        "wearable",
        "body worn",
        "body-worn",
        "on body",
        "on-body",
    ),
}
_PATCH_SHAPE_ALIASES: AliasMap = {
    "rectangular": ("rectangular", "rectangle"),
    "circular": ("circular", "round", "disc", "disk"),
    "auto": ("auto",),
}


def _normalize_free_text(value: str) -> str:
    return re.sub(r"[\s_\-/]+", " ", value.strip().lower())


def _detect_named_option(user_request: str, alias_map: AliasMap) -> str | None:
    text = _normalize_free_text(user_request)
    for canonical_name, aliases in alias_map.items():
        if any(_normalize_free_text(alias) in text for alias in aliases):
            return canonical_name
    return None


def _normalize_family_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalize_free_text(value)
    for canonical_name, aliases in _FAMILY_VALUE_ALIASES.items():
        if normalized == canonical_name or normalized in {_normalize_free_text(alias) for alias in aliases}:
            return canonical_name
    return None


def _normalize_patch_shape_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _normalize_free_text(value)
    for canonical_name, aliases in _PATCH_SHAPE_ALIASES.items():
        if normalized == canonical_name or normalized in {_normalize_free_text(alias) for alias in aliases}:
            return canonical_name
    return None


def _extract_frequency_and_bandwidth(user_request: str) -> tuple[float | None, float | None]:
    parsed_freq = None
    parsed_bw = None

    for match in _NUMERIC_WITH_UNIT_PATTERN.finditer(user_request):
        value = float(match.group("value"))
        unit = _normalize_free_text(match.group("unit"))
        start, end = match.span()
        context_window = user_request[max(0, start - 24): min(len(user_request), end + 24)]
        has_bw_hint = bool(_BANDWIDTH_HINT_PATTERN.search(context_window))
        has_freq_hint = bool(_FREQUENCY_HINT_PATTERN.search(context_window))

        if unit in {"ghz", "gigahertz", "giga hertz"}:
            if parsed_freq is None:
                parsed_freq = value
            continue

        if parsed_bw is None and has_bw_hint:
            parsed_bw = value
            continue
        if parsed_freq is None and (value > 1000.0 or has_freq_hint):
            parsed_freq = value / 1000.0
            continue
        if parsed_bw is None:
            parsed_bw = value

    return parsed_freq, parsed_bw


def _detect_family_from_text(user_request: str) -> str | None:
    text = _normalize_free_text(user_request)
    if any(token in text for token in ("wban", "wearable", "body worn", "on body")):
        return "wban_patch"
    if "amc" in text or "artificial magnetic conductor" in text:
        return "amc_patch"
    if "rectangular patch" in text or "microstrip" in text or "patch antenna" in text:
        return "microstrip_patch"
    return None


def _detect_patch_shape_from_text(user_request: str) -> str | None:
    text = _normalize_free_text(user_request)
    if "rectangular" in text:
        return "rectangular"
    if any(token in text for token in ("circular", "round patch", "round antenna", "round", "disc", "disk")):
        return "circular"
    return None


def summarize_user_intent(user_request: str) -> dict[str, Any]:
    """Parse user intent for antenna specs. Tries LLM if enabled, falls back to regex."""
    parsed_freq, parsed_bw = _extract_frequency_and_bandwidth(user_request)
    parsed_family = _detect_family_from_text(user_request)
    parsed_patch_shape = _detect_patch_shape_from_text(user_request)
    parsed_substrate_material = _detect_named_option(user_request, _SUBSTRATE_ALIASES)
    parsed_conductor_material = _detect_named_option(user_request, _CONDUCTOR_ALIASES)

    llm_intent: dict[str, Any] | None = None
    if bool(PLANNER_SETTINGS.llm_enabled_for_intent):
        system_prompt = (
            "Extract antenna design specifications from the user request. "
            "Return JSON with keys: "
            "frequency_ghz, bandwidth_mhz, antenna_family, patch_shape, substrate_material, conductor_material. "
            "Use null for unknown values. antenna_family must be one of amc_patch, microstrip_patch, wban_patch. "
            "Interpret wearable/on-body/body-worn requests as wban_patch, generic rectangular patch requests as microstrip_patch, "
            "and AMC requests as amc_patch. Interpret gigahertz as GHz and megs/megahertz as MHz. "
            "patch_shape must be one of rectangular, circular, auto, or null."
        )
        prompt = f"User request: {user_request}"
        llm_result = generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            timeout_sec=30,
            model_name=OLLAMA_SETTINGS.fast_model_name,
        )
        if isinstance(llm_result, dict):
            try:
                llm_freq = llm_result.get("frequency_ghz")
                llm_bw = llm_result.get("bandwidth_mhz")
                llm_family = _normalize_family_value(llm_result.get("antenna_family"))
                llm_patch_shape = _normalize_patch_shape_value(llm_result.get("patch_shape"))
                llm_substrate = llm_result.get("substrate_material")
                llm_conductor = llm_result.get("conductor_material")

                if parsed_freq is None and llm_freq is not None:
                    parsed_freq = float(llm_freq)
                if parsed_bw is None and llm_bw is not None:
                    parsed_bw = float(llm_bw)
                if parsed_family is None and llm_family in _ALLOWED_FAMILIES:
                    parsed_family = llm_family
                if parsed_patch_shape is None and llm_patch_shape in _ALLOWED_PATCH_SHAPES:
                    parsed_patch_shape = None if llm_patch_shape == "auto" else llm_patch_shape

                normalized_substrate = None
                if isinstance(llm_substrate, str) and llm_substrate.strip():
                    normalized_substrate = _detect_named_option(llm_substrate, _SUBSTRATE_ALIASES)
                    if normalized_substrate is None and llm_substrate.strip() in _SUBSTRATE_ALIASES:
                        normalized_substrate = llm_substrate.strip()
                if parsed_substrate_material is None and normalized_substrate is not None:
                    parsed_substrate_material = normalized_substrate

                normalized_conductor = None
                if isinstance(llm_conductor, str) and llm_conductor.strip():
                    normalized_conductor = _detect_named_option(llm_conductor, _CONDUCTOR_ALIASES)
                    if normalized_conductor is None and llm_conductor.strip() in _CONDUCTOR_ALIASES:
                        normalized_conductor = llm_conductor.strip()
                if parsed_conductor_material is None and normalized_conductor is not None:
                    parsed_conductor_material = normalized_conductor

                llm_intent = {"source": "llm", "llm_response": llm_result}
            except (ValueError, TypeError):
                pass

    return {
        "summary_version": "intent_summary.v1",
        "raw_request": user_request,
        "parsed_frequency_ghz": parsed_freq,
        "parsed_bandwidth_mhz": parsed_bw,
        "parsed_antenna_family": parsed_family,
        "parsed_patch_shape": parsed_patch_shape,
        "parsed_substrate_material": parsed_substrate_material,
        "parsed_conductor_material": parsed_conductor_material,
        "missing_fields": [
            key
            for key, value in {
                "frequency_ghz": parsed_freq,
                "bandwidth_mhz": parsed_bw,
            }.items()
            if value is None
        ],
        "llm_intent": llm_intent,
    }