from __future__ import annotations

import re
from typing import Any

from config import OLLAMA_SETTINGS, PLANNER_SETTINGS
from app.llm.ollama_client import generate_json


_FREQ_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*ghz", re.IGNORECASE)
_BW_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*mhz", re.IGNORECASE)


def _detect_family_from_text(user_request: str) -> str | None:
    text = user_request.lower()
    if "rectangular patch" in text or "microstrip" in text:
        return "microstrip_patch"
    if "wban" in text:
        return "wban_patch"
    if "amc" in text:
        return "amc_patch"
    return None


def summarize_user_intent(user_request: str) -> dict[str, Any]:
    """Parse user intent for antenna specs. Tries LLM if enabled, falls back to regex."""
    frequency_match = _FREQ_PATTERN.search(user_request)
    bandwidth_match = _BW_PATTERN.search(user_request)
    
    parsed_freq = float(frequency_match.group(1)) if frequency_match else None
    parsed_bw = float(bandwidth_match.group(1)) if bandwidth_match else None
    parsed_family = _detect_family_from_text(user_request)
    
    llm_intent = None
    if bool(PLANNER_SETTINGS.llm_enabled_for_intent):
        system_prompt = (
            "Extract antenna design specifications from user request. "
            "Return JSON: {\"frequency_ghz\": <number or null>, \"bandwidth_mhz\": <number or null>, "
            "\"antenna_family\": <amc_patch|microstrip_patch|wban_patch|null>}"
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
                llm_family = llm_result.get("antenna_family")
                if llm_freq is not None:
                    parsed_freq = float(llm_freq)
                if llm_bw is not None:
                    parsed_bw = float(llm_bw)
                if isinstance(llm_family, str) and llm_family in {"amc_patch", "microstrip_patch", "wban_patch"}:
                    parsed_family = llm_family
                llm_intent = {"source": "llm", "llm_response": llm_result}
            except (ValueError, TypeError):
                pass
    
    return {
        "summary_version": "intent_summary.v1",
        "raw_request": user_request,
        "parsed_frequency_ghz": parsed_freq,
        "parsed_bandwidth_mhz": parsed_bw,
        "parsed_antenna_family": parsed_family,
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