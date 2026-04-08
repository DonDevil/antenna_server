from __future__ import annotations

import math
from typing import Any, Literal

from app.antenna.materials import get_substrate_properties
from app.core.schemas import OptimizeRequest

PatchShape = Literal["rectangular", "circular"]


def _midpoint_from_constraint(request: OptimizeRequest, field_name: str, default: float) -> float:
    maybe = getattr(request.design_constraints, field_name, None)
    if maybe is None:
        return float(default)
    return (float(maybe.min) + float(maybe.max)) / 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _microstrip_feed_width_mm(epsilon_r: float, height_mm: float) -> float:
    h = max(height_mm, 0.1)
    er = max(epsilon_r, 1.1)
    target_ratio = 2.2 / math.sqrt(er)
    return _clamp(target_ratio * h, 0.6, 6.0)


def resolve_patch_shape(request: OptimizeRequest) -> PatchShape:
    requested = str(getattr(request.target_spec, "patch_shape", "auto") or "auto").strip().lower()
    if requested in {"rectangular", "circular"}:
        return requested  # type: ignore[return-value]

    polarization = str(getattr(request.target_spec, "polarization", "unspecified") or "unspecified").lower()
    if polarization == "circular":
        return "circular"
    return "rectangular"


def _rectangular_recipe(request: OptimizeRequest, substrate: dict[str, Any]) -> dict[str, Any]:
    f_ghz = float(request.target_spec.frequency_ghz)
    bw_mhz = float(request.target_spec.bandwidth_mhz)
    epsilon_r = float(substrate["epsilon_r"])
    height_mm = _midpoint_from_constraint(request, "substrate_height_mm", max(0.8, min(2.4, 30.0 / max(f_ghz, 0.5))))

    c = 299_792_458.0
    f_hz = f_ghz * 1e9
    h_m = height_mm / 1e3
    width_m = c / (2.0 * f_hz) * math.sqrt(2.0 / (epsilon_r + 1.0))
    eps_eff = ((epsilon_r + 1.0) / 2.0) + ((epsilon_r - 1.0) / 2.0) * (1.0 / math.sqrt(1.0 + 12.0 * h_m / max(width_m, 1e-9)))
    delta_l_m = 0.412 * h_m * (((eps_eff + 0.3) * ((width_m / h_m) + 0.264)) / ((eps_eff - 0.258) * ((width_m / h_m) + 0.8)))
    effective_length_m = c / (2.0 * f_hz * math.sqrt(max(eps_eff, 1.0)))
    length_m = max(effective_length_m - (2.0 * delta_l_m), h_m)

    bandwidth_boost = 1.0 + _clamp((bw_mhz - 80.0) / 800.0, 0.0, 0.18)
    patch_length_mm = length_m * 1e3
    patch_width_mm = width_m * 1e3 * bandwidth_boost
    patch_height_mm = 0.035
    substrate_length_mm = patch_length_mm + max(6.0 * height_mm, 8.0)
    substrate_width_mm = patch_width_mm + max(6.0 * height_mm, 8.0)
    feed_length_mm = max(4.0, patch_length_mm * 0.38)
    feed_width_mm = max(_microstrip_feed_width_mm(epsilon_r, height_mm), min(6.0, 0.025 * patch_width_mm + (bw_mhz / 500.0)))

    return {
        "recipe_name": "rectangular_microstrip_patch",
        "patch_shape": "rectangular",
        "substrate": substrate,
        "dimensions": {
            "patch_length_mm": patch_length_mm,
            "patch_width_mm": patch_width_mm,
            "patch_height_mm": patch_height_mm,
            "patch_radius_mm": patch_width_mm / 2.0,
            "substrate_length_mm": substrate_length_mm,
            "substrate_width_mm": substrate_width_mm,
            "substrate_height_mm": height_mm,
            "feed_length_mm": feed_length_mm,
            "feed_width_mm": feed_width_mm,
            "feed_offset_x_mm": 0.0,
            "feed_offset_y_mm": -(patch_length_mm * 0.28),
        },
        "notes": [
            "Rectangular patch uses cavity-model-inspired width and effective length equations.",
            "Feed width is initialized near a 50-ohm microstrip estimate and widened for larger bandwidth goals.",
        ],
    }


def _circular_recipe(request: OptimizeRequest, substrate: dict[str, Any]) -> dict[str, Any]:
    f_ghz = float(request.target_spec.frequency_ghz)
    bw_mhz = float(request.target_spec.bandwidth_mhz)
    epsilon_r = float(substrate["epsilon_r"])
    height_mm = _midpoint_from_constraint(request, "substrate_height_mm", max(0.8, min(2.2, 24.0 / max(f_ghz, 0.5))))

    height_cm = height_mm / 10.0
    f_term_cm = 8.791 / (f_ghz * math.sqrt(max(epsilon_r, 1.0)))
    correction = 1.0 + ((2.0 * height_cm) / (math.pi * epsilon_r * max(f_term_cm, 1e-9))) * (
        math.log((math.pi * f_term_cm) / max(2.0 * height_cm, 1e-9)) + 1.7726
    )
    radius_cm = f_term_cm / math.sqrt(max(correction, 1.0))
    radius_mm = max(radius_cm * 10.0, 1.0)
    diameter_mm = 2.0 * radius_mm

    bandwidth_boost = 1.0 + _clamp((bw_mhz - 60.0) / 700.0, 0.0, 0.15)
    patch_height_mm = 0.035
    substrate_length_mm = diameter_mm * bandwidth_boost + max(8.0 * height_mm, 10.0)
    substrate_width_mm = diameter_mm * bandwidth_boost + max(8.0 * height_mm, 10.0)
    feed_length_mm = max(4.0, radius_mm * 1.15)
    feed_width_mm = max(_microstrip_feed_width_mm(epsilon_r, height_mm), min(5.0, 0.18 * radius_mm))

    return {
        "recipe_name": "circular_microstrip_patch",
        "patch_shape": "circular",
        "substrate": substrate,
        "dimensions": {
            "patch_length_mm": diameter_mm,
            "patch_width_mm": diameter_mm,
            "patch_height_mm": patch_height_mm,
            "patch_radius_mm": radius_mm,
            "substrate_length_mm": substrate_length_mm,
            "substrate_width_mm": substrate_width_mm,
            "substrate_height_mm": height_mm,
            "feed_length_mm": feed_length_mm,
            "feed_width_mm": feed_width_mm,
            "feed_offset_x_mm": 0.0,
            "feed_offset_y_mm": -(radius_mm * 0.42),
        },
        "notes": [
            "Circular patch uses a standard corrected radius estimate based on substrate height and permittivity.",
            "Equivalent diameter is exposed so current downstream planners can stay compatible while shape metadata is preserved.",
        ],
    }


def generate_recipe(request: OptimizeRequest) -> dict[str, Any]:
    substrate_name = request.design_constraints.allowed_substrates[0] if request.design_constraints.allowed_substrates else None
    substrate = get_substrate_properties(substrate_name)
    patch_shape = resolve_patch_shape(request)

    if patch_shape == "circular":
        return _circular_recipe(request, substrate)
    return _rectangular_recipe(request, substrate)
