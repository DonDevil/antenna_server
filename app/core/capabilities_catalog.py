from __future__ import annotations

import json
from typing import Any

from app.core.family_registry import list_supported_families
from config import BOUNDS, CONTEXT_DIR


CAPABILITIES_PATH = CONTEXT_DIR / "capabilities" / "antenna_capabilities.v1.json"


def _default_capabilities() -> dict[str, Any]:
    return {
        "schema_version": "antenna_capabilities.v1",
        "frequency_range_ghz": {
            "min": float(BOUNDS.frequency_ghz[0]),
            "max": float(BOUNDS.frequency_ghz[1]),
        },
        "bandwidth_range_mhz": {
            "min": float(BOUNDS.bandwidth_mhz[0]),
            "max": float(BOUNDS.bandwidth_mhz[1]),
        },
        "supported_families": list_supported_families(),
        "available_conductor_materials": ["Copper (annealed)", "Aluminum", "Silver", "Gold"],
        "available_substrate_materials": ["FR-4 (lossy)", "Rogers RT/duroid 5880", "Rogers RO3003", "Rogers RO4350B"],
        "notes": "Default capabilities fallback.",
    }


def load_capabilities_catalog() -> dict[str, Any]:
    if not CAPABILITIES_PATH.exists():
        return _default_capabilities()

    try:
        payload = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_capabilities()

    if not isinstance(payload, dict):
        return _default_capabilities()

    base = _default_capabilities()
    base.update(payload)

    if not isinstance(base.get("supported_families"), list):
        base["supported_families"] = list_supported_families()
    if not isinstance(base.get("available_conductor_materials"), list):
        base["available_conductor_materials"] = ["Copper (annealed)"]
    if not isinstance(base.get("available_substrate_materials"), list):
        base["available_substrate_materials"] = ["FR-4 (lossy)"]

    return base
