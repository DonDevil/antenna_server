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


def _microstrip_patch_geometry(
    *,
    frequency_ghz: float,
    bandwidth_mhz: float,
    epsilon_r: float,
    height_mm: float,
    feed_offset_ratio: float = 0.28,
    substrate_margin_factor: float = 6.0,
    minimum_margin_mm: float = 8.0,
) -> dict[str, float]:
    c = 299_792_458.0
    f_hz = max(frequency_ghz, 0.1) * 1e9
    h_m = height_mm / 1e3
    width_m = c / (2.0 * f_hz) * math.sqrt(2.0 / (epsilon_r + 1.0))
    eps_eff = ((epsilon_r + 1.0) / 2.0) + ((epsilon_r - 1.0) / 2.0) * (1.0 / math.sqrt(1.0 + 12.0 * h_m / max(width_m, 1e-9)))
    delta_l_m = 0.412 * h_m * (((eps_eff + 0.3) * ((width_m / h_m) + 0.264)) / ((eps_eff - 0.258) * ((width_m / h_m) + 0.8)))
    effective_length_m = c / (2.0 * f_hz * math.sqrt(max(eps_eff, 1.0)))
    length_m = max(effective_length_m - (2.0 * delta_l_m), h_m)

    bandwidth_boost = 1.0 + _clamp((bandwidth_mhz - 80.0) / 800.0, 0.0, 0.18)
    patch_length_mm = length_m * 1e3
    patch_width_mm = width_m * 1e3 * bandwidth_boost
    patch_height_mm = 0.035
    margin_mm = max(substrate_margin_factor * height_mm, minimum_margin_mm)
    substrate_length_mm = patch_length_mm + margin_mm
    substrate_width_mm = patch_width_mm + margin_mm
    feed_length_mm = max(4.0, patch_length_mm * 0.38)
    feed_width_mm = max(_microstrip_feed_width_mm(epsilon_r, height_mm), min(6.0, 0.025 * patch_width_mm + (bandwidth_mhz / 500.0)))

    return {
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
        "feed_offset_y_mm": -(patch_length_mm * feed_offset_ratio),
    }


def resolve_patch_shape(request: OptimizeRequest) -> PatchShape:
    family_name = str(getattr(request.target_spec, "antenna_family", "microstrip_patch") or "microstrip_patch").strip().lower()
    if family_name in {"amc_patch", "wban_patch"}:
        return "rectangular"

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
    dimensions = _microstrip_patch_geometry(
        frequency_ghz=f_ghz,
        bandwidth_mhz=bw_mhz,
        epsilon_r=epsilon_r,
        height_mm=height_mm,
    )

    return {
        "recipe_name": "rectangular_microstrip_patch",
        "patch_shape": "rectangular",
        "substrate": substrate,
        "dimensions": dimensions,
        "family_parameters": {},
        "notes": [
            "Rectangular patch uses transmission-line / cavity-model equations for W, εeff, ΔL, Leff, and L.",
            "Substrate outline is initialized with about 6h edge margin and the feedline is seeded near a 50 Ω microstrip width.",
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
        "family_parameters": {},
        "notes": [
            "Circular patch uses a corrected radius estimate based on substrate height and permittivity.",
            "Equivalent diameter is exposed so downstream planners and the client remain compatible.",
        ],
    }


def _amc_recipe(request: OptimizeRequest, substrate: dict[str, Any]) -> dict[str, Any]:
    f_ghz = float(request.target_spec.frequency_ghz)
    bw_mhz = float(request.target_spec.bandwidth_mhz)
    epsilon_r = float(substrate["epsilon_r"])
    height_mm = _midpoint_from_constraint(request, "substrate_height_mm", max(0.8, min(2.4, 28.0 / max(f_ghz, 0.5))))
    dimensions = _microstrip_patch_geometry(
        frequency_ghz=f_ghz,
        bandwidth_mhz=bw_mhz,
        epsilon_r=epsilon_r,
        height_mm=height_mm,
    )

    wavelength_mm = (299_792_458.0 / (max(f_ghz, 0.1) * 1e9)) * 1000.0
    target_gain = getattr(request.target_spec, "target_gain_dbi", None)
    if target_gain is None:
        target_gain = request.optimization_policy.acceptance.minimum_gain_dbi
    bandwidth_norm = _clamp((bw_mhz - 60.0) / 390.0, 0.0, 1.0)
    gain_norm = _clamp((float(target_gain) - 4.0) / 6.0, 0.0, 1.0)

    period_ratio = _clamp(0.18 + (0.07 * bandwidth_norm) + (0.03 * gain_norm), 0.15, 0.30)
    patch_ratio = _clamp(0.84 - (0.12 * bandwidth_norm) + (0.05 * (1.0 - gain_norm)), 0.60, 0.90)
    via_ratio = _clamp(0.025 + (0.012 * gain_norm), 0.02, 0.05)
    default_air_gap_mm = wavelength_mm * _clamp(0.025 + (0.025 * gain_norm) + (0.015 * bandwidth_norm), 0.02, 0.08)
    air_gap_mm = _midpoint_from_constraint(request, "amc_air_gap_mm", default_air_gap_mm)

    amc_unit_cell_period_mm = round(wavelength_mm * period_ratio, 4)
    amc_patch_size_mm = round(amc_unit_cell_period_mm * patch_ratio, 4)
    amc_gap_mm = round(max(amc_unit_cell_period_mm - amc_patch_size_mm, 0.2), 4)
    amc_via_radius_mm = round(amc_unit_cell_period_mm * via_ratio, 4)
    amc_via_height_mm = round(height_mm, 4)
    amc_array_rows = max(3, int(round(_clamp(4.0 + (3.0 * gain_norm) + (2.0 * bandwidth_norm), 3.0, 10.0))))
    amc_array_cols = max(3, int(round(_clamp(4.0 + (3.4 * gain_norm) + (1.6 * bandwidth_norm), 3.0, 10.0))))
    amc_ground_size_mm = round(max(amc_array_rows, amc_array_cols) * amc_unit_cell_period_mm, 4)

    dimensions["substrate_length_mm"] = max(float(dimensions["substrate_length_mm"]), amc_ground_size_mm)
    dimensions["substrate_width_mm"] = max(float(dimensions["substrate_width_mm"]), amc_ground_size_mm)

    return {
        "recipe_name": "amc_backed_rectangular_patch",
        "patch_shape": "rectangular",
        "substrate": substrate,
        "dimensions": dimensions,
        "family_parameters": {
            "amc_unit_cell_period_mm": amc_unit_cell_period_mm,
            "amc_patch_size_mm": amc_patch_size_mm,
            "amc_gap_mm": amc_gap_mm,
            "amc_via_radius_mm": amc_via_radius_mm,
            "amc_via_height_mm": amc_via_height_mm,
            "amc_ground_size_mm": amc_ground_size_mm,
            "amc_array_rows": amc_array_rows,
            "amc_array_cols": amc_array_cols,
            "amc_air_gap_mm": round(air_gap_mm, 4),
            "reflection_phase_target_deg": 0.0,
        },
        "notes": [
            "AMC unit-cell sizing follows λ0-scaled rules: p≈0.15–0.30λ0, a≈0.60–0.90p, and g=p−a.",
            "The server keeps the AMC resonance aligned with the patch target frequency and returns unit-cell, array, and air-gap guidance to the client.",
        ],
    }


def _wban_recipe(request: OptimizeRequest, substrate: dict[str, Any]) -> dict[str, Any]:
    f_ghz = float(request.target_spec.frequency_ghz)
    bw_mhz = float(request.target_spec.bandwidth_mhz)
    epsilon_r = float(substrate["epsilon_r"])
    height_mm = _midpoint_from_constraint(request, "substrate_height_mm", max(0.8, min(2.4, 24.0 / max(f_ghz, 0.5))))
    body_distance_mm = _midpoint_from_constraint(request, "body_distance_mm", 6.0)
    bending_radius_mm = _midpoint_from_constraint(request, "bending_radius_mm", 65.0)

    body_factor = _clamp((10.0 - body_distance_mm) / 8.0, 0.0, 1.0)
    bend_factor = _clamp((100.0 - bending_radius_mm) / 70.0, 0.0, 1.0)
    detuning_compensation_ratio = _clamp(1.04 + (0.03 * body_factor) + (0.02 * bend_factor), 1.03, 1.10)
    design_frequency_ghz = f_ghz * detuning_compensation_ratio

    dimensions = _microstrip_patch_geometry(
        frequency_ghz=design_frequency_ghz,
        bandwidth_mhz=bw_mhz,
        epsilon_r=epsilon_r,
        height_mm=height_mm,
        feed_offset_ratio=0.32,
        substrate_margin_factor=7.0,
        minimum_margin_mm=10.0,
    )

    bandwidth_norm = _clamp((bw_mhz - 40.0) / 240.0, 0.0, 1.0)
    ground_slot_length_ratio = _clamp(0.22 + (0.20 * bandwidth_norm) + (0.06 * body_factor), 0.20, 0.50)
    ground_slot_width_ratio = _clamp(0.03 + (0.04 * bandwidth_norm) + (0.02 * bend_factor), 0.02, 0.10)
    notch_length_ratio = _clamp(0.06 + (0.06 * body_factor) + (0.05 * bandwidth_norm), 0.05, 0.20)
    notch_width_ratio = _clamp(0.015 + (0.02 * bandwidth_norm) + (0.015 * bend_factor), 0.01, 0.06)

    patch_length_mm = float(dimensions["patch_length_mm"])
    patch_width_mm = float(dimensions["patch_width_mm"])

    return {
        "recipe_name": "wban_detuned_rectangular_patch",
        "patch_shape": "rectangular",
        "substrate": substrate,
        "dimensions": dimensions,
        "family_parameters": {
            "body_distance_mm": round(body_distance_mm, 4),
            "bending_radius_mm": round(bending_radius_mm, 4),
            "design_frequency_ghz": round(design_frequency_ghz, 4),
            "ground_slot_length_mm": round(ground_slot_length_ratio * patch_length_mm, 4),
            "ground_slot_width_mm": round(ground_slot_width_ratio * patch_width_mm, 4),
            "notch_length_mm": round(notch_length_ratio * patch_length_mm, 4),
            "notch_width_mm": round(notch_width_ratio * patch_width_mm, 4),
            "detuning_compensation_ratio": round(detuning_compensation_ratio, 4),
        },
        "notes": [
            "WBAN geometry uses an elevated design frequency (about 1.03–1.10× target) to offset on-body detuning.",
            "The server returns body-distance, bending, slot, and notch guidance so the client can build CST commands consistently.",
        ],
    }


def generate_recipe(request: OptimizeRequest) -> dict[str, Any]:
    substrate_name = request.design_constraints.allowed_substrates[0] if request.design_constraints.allowed_substrates else None
    substrate = get_substrate_properties(substrate_name)
    family_name = str(getattr(request.target_spec, "antenna_family", "microstrip_patch") or "microstrip_patch").strip().lower()
    patch_shape = resolve_patch_shape(request)

    if family_name == "amc_patch":
        return _amc_recipe(request, substrate)
    if family_name == "wban_patch":
        return _wban_recipe(request, substrate)
    if patch_shape == "circular":
        return _circular_recipe(request, substrate)
    return _rectangular_recipe(request, substrate)
