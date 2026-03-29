from __future__ import annotations

import math


def baseline_dimensions(frequency_ghz: float, bandwidth_mhz: float) -> dict:
    # Basic microstrip-inspired deterministic fallback for cold-start.
    eps_eff = 4.4
    c = 3e8
    f_hz = frequency_ghz * 1e9
    patch_length_m = c / (2 * f_hz * math.sqrt(eps_eff))
    patch_width_m = patch_length_m * 1.2

    patch_length_mm = patch_length_m * 1e3
    patch_width_mm = patch_width_m * 1e3
    substrate_height_mm = max(0.8, min(2.4, 0.03 * patch_length_mm))
    substrate_length_mm = patch_length_mm + 6 * substrate_height_mm
    substrate_width_mm = patch_width_mm + 6 * substrate_height_mm
    feed_length_mm = max(4.0, patch_length_mm * 0.4)
    feed_width_mm = max(0.8, min(4.0, 0.04 * patch_width_mm + bandwidth_mhz / 400.0))

    return {
        "patch_length_mm": patch_length_mm,
        "patch_width_mm": patch_width_mm,
        "patch_height_mm": 0.035,
        "substrate_length_mm": substrate_length_mm,
        "substrate_width_mm": substrate_width_mm,
        "substrate_height_mm": substrate_height_mm,
        "feed_length_mm": feed_length_mm,
        "feed_width_mm": feed_width_mm,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": -patch_length_mm / 4,
    }
