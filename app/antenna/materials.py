from __future__ import annotations

from typing import Any


_SUBSTRATE_LIBRARY: dict[str, dict[str, float]] = {
    "FR-4 (lossy)": {"epsilon_r": 4.4, "loss_tangent": 0.02},
    "Rogers RT/duroid 5880": {"epsilon_r": 2.2, "loss_tangent": 0.0009},
    "Rogers RO3003": {"epsilon_r": 3.0, "loss_tangent": 0.0013},
    "Rogers RO4350B": {"epsilon_r": 3.48, "loss_tangent": 0.0037},
}

_CONDUCTOR_LIBRARY: dict[str, dict[str, float]] = {
    "Copper (annealed)": {"conductivity_s_per_m": 5.8e7},
    "Aluminum": {"conductivity_s_per_m": 3.56e7},
    "Silver": {"conductivity_s_per_m": 6.3e7},
    "Gold": {"conductivity_s_per_m": 4.1e7},
}


def get_substrate_properties(name: str | None) -> dict[str, Any]:
    substrate_name = (name or "FR-4 (lossy)").strip() or "FR-4 (lossy)"
    props = _SUBSTRATE_LIBRARY.get(substrate_name, _SUBSTRATE_LIBRARY["FR-4 (lossy)"])
    return {
        "name": substrate_name,
        "epsilon_r": float(props["epsilon_r"]),
        "loss_tangent": float(props["loss_tangent"]),
    }


def get_conductor_properties(name: str | None) -> dict[str, Any]:
    conductor_name = (name or "Copper (annealed)").strip() or "Copper (annealed)"
    props = _CONDUCTOR_LIBRARY.get(conductor_name, _CONDUCTOR_LIBRARY["Copper (annealed)"])
    return {
        "name": conductor_name,
        "conductivity_s_per_m": float(props["conductivity_s_per_m"]),
    }
