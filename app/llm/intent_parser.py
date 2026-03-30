from __future__ import annotations

import re
from typing import Any


_FREQ_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*ghz", re.IGNORECASE)
_BW_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*mhz", re.IGNORECASE)


def summarize_user_intent(user_request: str) -> dict[str, Any]:
    frequency_match = _FREQ_PATTERN.search(user_request)
    bandwidth_match = _BW_PATTERN.search(user_request)
    return {
        "summary_version": "intent_summary.v1",
        "raw_request": user_request,
        "parsed_frequency_ghz": float(frequency_match.group(1)) if frequency_match else None,
        "parsed_bandwidth_mhz": float(bandwidth_match.group(1)) if bandwidth_match else None,
        "missing_fields": [
            key
            for key, value in {
                "frequency_ghz": frequency_match,
                "bandwidth_mhz": bandwidth_match,
            }.items()
            if value is None
        ],
    }