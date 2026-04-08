from __future__ import annotations

from typing import Any


_SUBSTRATE_LIBRARY: dict[str, dict[str, float]] = {
    "FR-4 (lossy)": {"epsilon_r": 4.4, "loss_tangent": 0.02},
    "Rogers RT/duroid 5880": {"epsilon_r": 2.2, "loss_tangent": 0.0009},
    "Rogers RO3003": {"epsilon_r": 3.0, "loss_tangent": 0.0013},
}


def get_substrate_properties(name: str | None) -> dict[str, Any]:
    substrate_name = (name or "FR-4 (lossy)").strip() or "FR-4 (lossy)"
    props = _SUBSTRATE_LIBRARY.get(substrate_name, _SUBSTRATE_LIBRARY["FR-4 (lossy)"])
    return {
        "name": substrate_name,
        "epsilon_r": float(props["epsilon_r"]),
        "loss_tangent": float(props["loss_tangent"]),
    }
