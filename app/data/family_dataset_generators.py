from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import DATA_DIR


C0 = 299_792_458.0

_COMMON_INPUT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "antenna_family",
    "patch_shape",
    "feed_type",
    "polarization",
    "substrate_name",
    "substrate_epsilon_r",
    "substrate_loss_tangent",
    "substrate_height_mm",
    "conductor_name",
    "conductor_conductivity_s_per_m",
    "target_frequency_ghz",
    "target_bandwidth_mhz",
    "target_minimum_gain_dbi",
    "target_maximum_vswr",
    "target_minimum_return_loss_db",
)

_COMMON_OUTPUT_COLUMNS: tuple[str, ...] = (
    "actual_center_frequency_ghz",
    "actual_bandwidth_mhz",
    "actual_return_loss_db",
    "actual_vswr",
    "actual_gain_dbi",
    "actual_radiation_efficiency_pct",
    "actual_total_efficiency_pct",
    "actual_directivity_dbi",
    "accepted",
    "solver_status",
    "simulation_time_sec",
    "notes",
)

RECT_PATCH_SYNTH_COLUMNS: tuple[str, ...] = (
    *_COMMON_INPUT_COLUMNS,
    "patch_length_mm",
    "patch_width_mm",
    "patch_height_mm",
    "substrate_length_mm",
    "substrate_width_mm",
    "feed_length_mm",
    "feed_width_mm",
    "feed_offset_x_mm",
    "feed_offset_y_mm",
    "actual_peak_theta_deg",
    "actual_peak_phi_deg",
    "actual_front_to_back_db",
    "actual_axial_ratio_db",
    *_COMMON_OUTPUT_COLUMNS,
)

AMC_SYNTH_COLUMNS: tuple[str, ...] = (
    *_COMMON_INPUT_COLUMNS,
    "amc_unit_cell_period_mm",
    "amc_patch_size_mm",
    "amc_gap_mm",
    "amc_via_radius_mm",
    "amc_via_height_mm",
    "amc_ground_size_mm",
    "amc_array_rows",
    "amc_array_cols",
    "amc_air_gap_mm",
    "actual_reflection_phase_center_ghz",
    "actual_reflection_phase_bandwidth_mhz",
    "actual_gain_improvement_dbi",
    "actual_back_lobe_reduction_db",
    *_COMMON_OUTPUT_COLUMNS,
)

WBAN_SYNTH_COLUMNS: tuple[str, ...] = (
    *_COMMON_INPUT_COLUMNS,
    "patch_length_mm",
    "patch_width_mm",
    "patch_height_mm",
    "substrate_length_mm",
    "substrate_width_mm",
    "feed_length_mm",
    "feed_width_mm",
    "feed_offset_x_mm",
    "feed_offset_y_mm",
    "body_distance_mm",
    "bending_radius_mm",
    "ground_slot_length_mm",
    "ground_slot_width_mm",
    "notch_length_mm",
    "notch_width_mm",
    "actual_on_body_gain_dbi",
    "actual_off_body_gain_dbi",
    "actual_sar_1g_wkg",
    "actual_sar_10g_wkg",
    "actual_detuning_mhz",
    "actual_efficiency_on_body_pct",
    *_COMMON_OUTPUT_COLUMNS,
)

_PATCH_SUBSTRATES: tuple[dict[str, float | str], ...] = (
    {"name": "Rogers RT/duroid 5880", "epsilon_r": 2.2, "loss_tangent": 0.0009},
    {"name": "Rogers RO3003", "epsilon_r": 3.0, "loss_tangent": 0.0013},
    {"name": "FR-4 (lossy)", "epsilon_r": 4.4, "loss_tangent": 0.02},
)

_AMC_SUBSTRATES: tuple[dict[str, float | str], ...] = (
    {"name": "FR-4 (lossy)", "epsilon_r": 4.4, "loss_tangent": 0.02},
    {"name": "Rogers RT/duroid 5880", "epsilon_r": 2.2, "loss_tangent": 0.0009},
)

_WBAN_SUBSTRATES: tuple[dict[str, float | str], ...] = (
    {"name": "Rogers RO3003", "epsilon_r": 3.0, "loss_tangent": 0.0013},
    {"name": "Rogers RT/duroid 5880", "epsilon_r": 2.2, "loss_tangent": 0.0009},
)

_CONDUCTORS: tuple[dict[str, float | str], ...] = (
    {"name": "Copper (annealed)", "conductivity_s_per_m": 5.8e7},
    {"name": "Silver", "conductivity_s_per_m": 6.3e7},
    {"name": "Aluminum", "conductivity_s_per_m": 3.56e7},
)

_DEFAULT_OUTPUT_PATHS: dict[str, Path] = {
    "microstrip_patch": DATA_DIR / "raw" / "rect_patch_formula_synth_v1.csv",
    "amc_patch": DATA_DIR / "raw" / "amc_patch_formula_synth_v1.csv",
    "wban_patch": DATA_DIR / "raw" / "wban_patch_formula_synth_v1.csv",
}


@dataclass(frozen=True)
class FamilySynthArtifacts:
    family: str
    csv_path: Path
    rows: int
    accepted_rows: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _jitter(rng: np.random.Generator, value: float, pct: float) -> float:
    return float(value) * float(1.0 + rng.uniform(-pct, pct))


def _pick(rng: np.random.Generator, options: tuple[dict[str, float | str], ...]) -> dict[str, float | str]:
    return dict(options[int(rng.integers(0, len(options)))])


def _stable_run_id(family: str, seed: int, index: int) -> str:
    payload = f"{family}-{seed}-{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, payload))


def _timestamp_for(index: int) -> str:
    base = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=index * 17)).isoformat()


def _is_accepted(target_frequency_ghz: float, actual_center_frequency_ghz: float, actual_vswr: float, actual_return_loss_db: float) -> bool:
    frequency_close = abs(actual_center_frequency_ghz - target_frequency_ghz) <= (0.05 * target_frequency_ghz)
    return bool(frequency_close and actual_vswr < 2.0 and actual_return_loss_db < -10.0)


def _microstrip_geometry(
    *,
    target_frequency_ghz: float,
    substrate_epsilon_r: float,
    substrate_height_mm: float,
    target_bandwidth_mhz: float,
    rng: np.random.Generator,
    feed_offset_span: tuple[float, float],
    feed_width_span: tuple[float, float],
) -> dict[str, float]:
    f_hz = target_frequency_ghz * 1e9
    h_m = substrate_height_mm / 1000.0

    width_m = C0 / (2.0 * f_hz) * math.sqrt(2.0 / (substrate_epsilon_r + 1.0))
    eps_eff = ((substrate_epsilon_r + 1.0) / 2.0) + ((substrate_epsilon_r - 1.0) / 2.0) * (
        1.0 + 12.0 * h_m / max(width_m, 1e-9)
    ) ** -0.5
    delta_l_m = 0.412 * h_m * (
        ((eps_eff + 0.3) * ((width_m / max(h_m, 1e-9)) + 0.264))
        / ((eps_eff - 0.258) * ((width_m / max(h_m, 1e-9)) + 0.8))
    )
    effective_length_m = C0 / (2.0 * f_hz * math.sqrt(max(eps_eff, 1e-9)))
    length_m = max(effective_length_m - (2.0 * delta_l_m), h_m)

    ideal_length_mm = length_m * 1000.0
    ideal_width_mm = width_m * 1000.0
    bandwidth_boost = 1.0 + 0.12 * _clamp((target_bandwidth_mhz - 60.0) / 240.0, 0.0, 1.0)

    patch_length_mm = _clamp(_jitter(rng, ideal_length_mm, 0.05), 5.0, 120.0)
    patch_width_mm = _clamp(_jitter(rng, ideal_width_mm * bandwidth_boost, 0.06), 5.0, 120.0)

    bandwidth_norm = _clamp((target_bandwidth_mhz - 30.0) / 270.0, 0.0, 1.0)
    height_norm = _clamp((substrate_height_mm - 0.8) / 2.4, 0.0, 1.0)
    width_low, width_high = feed_width_span
    offset_low, offset_high = feed_offset_span
    base_width_scale = ((width_low + width_high) / 2.0) + ((width_high - width_low) * 0.35 * (bandwidth_norm - 0.5)) + ((width_high - width_low) * 0.15 * (height_norm - 0.5))
    feed_width_scale = _clamp(base_width_scale + float(rng.normal(0.0, (width_high - width_low) * 0.04)), width_low, width_high)
    base_offset_ratio = offset_low + ((offset_high - offset_low) * (0.35 + (0.45 * bandwidth_norm)))
    feed_offset_ratio = _clamp(base_offset_ratio + float(rng.normal(0.0, 0.012)), offset_low, offset_high)

    half_wavelength_mm = (C0 / (2.0 * f_hz)) * 1000.0 / math.sqrt(max(substrate_epsilon_r, 1.0))
    feed_width_mm = _clamp(half_wavelength_mm * feed_width_scale, 0.4, 8.0)
    feed_offset_y_mm = -feed_offset_ratio * patch_length_mm

    return {
        "patch_length_mm": patch_length_mm,
        "patch_width_mm": patch_width_mm,
        "patch_height_mm": 0.035,
        "substrate_length_mm": patch_length_mm + (6.0 * substrate_height_mm),
        "substrate_width_mm": patch_width_mm + (6.0 * substrate_height_mm),
        "feed_length_mm": max(4.0, 0.33 * (patch_length_mm + (6.0 * substrate_height_mm))),
        "feed_width_mm": feed_width_mm,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": feed_offset_y_mm,
    }


def _common_inputs(
    *,
    family: str,
    patch_shape: str,
    feed_type: str,
    polarization: str,
    substrate: dict[str, float | str],
    conductor: dict[str, float | str],
    target_frequency_ghz: float,
    target_bandwidth_mhz: float,
    target_minimum_gain_dbi: float,
    target_maximum_vswr: float,
    index: int,
    seed: int,
) -> dict[str, object]:
    return {
        "run_id": _stable_run_id(family, seed, index),
        "timestamp_utc": _timestamp_for(index),
        "antenna_family": family,
        "patch_shape": patch_shape,
        "feed_type": feed_type,
        "polarization": polarization,
        "substrate_name": str(substrate["name"]),
        "substrate_epsilon_r": float(substrate["epsilon_r"]),
        "substrate_loss_tangent": float(substrate["loss_tangent"]),
        "substrate_height_mm": float(substrate["height_mm"]),
        "conductor_name": str(conductor["name"]),
        "conductor_conductivity_s_per_m": float(conductor["conductivity_s_per_m"]),
        "target_frequency_ghz": target_frequency_ghz,
        "target_bandwidth_mhz": target_bandwidth_mhz,
        "target_minimum_gain_dbi": target_minimum_gain_dbi,
        "target_maximum_vswr": target_maximum_vswr,
        "target_minimum_return_loss_db": -float(np.random.default_rng(seed + index).uniform(10.0, 20.0)),
    }


def generate_rect_patch_synth_dataset(*, rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, object]] = []

    for index in range(rows):
        substrate = _pick(rng, _PATCH_SUBSTRATES)
        conductor = _pick(rng, _CONDUCTORS)
        substrate["height_mm"] = round(float(rng.uniform(0.8, 3.0)), 3)
        target_frequency_ghz = round(float(rng.uniform(1.0, 6.0)), 4)
        target_bandwidth_mhz = round(float(rng.uniform(30.0, 300.0)), 3)
        geometry = _microstrip_geometry(
            target_frequency_ghz=target_frequency_ghz,
            substrate_epsilon_r=float(substrate["epsilon_r"]),
            substrate_height_mm=float(substrate["height_mm"]),
            target_bandwidth_mhz=target_bandwidth_mhz,
            rng=rng,
            feed_offset_span=(0.20, 0.40),
            feed_width_span=(0.05, 0.10),
        )

        height_factor = _clamp((float(substrate["height_mm"]) - 0.8) / 2.2, 0.0, 1.0)
        loss_tangent = float(substrate["loss_tangent"])
        detune_ratio = float(rng.normal(0.0, 0.012))
        actual_center_frequency_ghz = round(target_frequency_ghz * (1.0 + detune_ratio), 4)
        aspect_ratio = geometry["patch_width_mm"] / max(geometry["patch_length_mm"], 1e-6)
        actual_bandwidth_mhz = round(
            _clamp(
                target_bandwidth_mhz
                * (0.88 + (0.20 * height_factor) + (0.07 * (aspect_ratio - 1.0)) - (2.0 * loss_tangent) + float(rng.uniform(-0.04, 0.04))),
                8.0,
                target_bandwidth_mhz * 1.45,
            ),
            3,
        )
        match_quality = _clamp(1.18 - (abs(detune_ratio) * 8.0) + (0.05 * height_factor) - (4.0 * loss_tangent), 0.25, 1.2)
        actual_return_loss_db = round(-_clamp(8.0 + (18.0 * match_quality) + float(rng.uniform(-2.0, 2.0)), 6.0, 35.0), 3)
        actual_vswr = round(_clamp(1.05 + max(0.0, 2.1 - (abs(actual_return_loss_db) / 10.0)) + (abs(detune_ratio) * 5.0), 1.02, 4.5), 4)
        actual_radiation_efficiency_pct = round(
            _clamp(94.0 - (520.0 * loss_tangent) + (4.0 * height_factor) - (120.0 * abs(detune_ratio)) + float(rng.uniform(-3.0, 3.0)), 45.0, 98.5),
            3,
        )
        actual_total_efficiency_pct = round(_clamp(actual_radiation_efficiency_pct * float(rng.uniform(0.92, 0.99)), 35.0, actual_radiation_efficiency_pct), 3)
        aperture_factor = (geometry["patch_length_mm"] * geometry["patch_width_mm"]) / 900.0
        actual_gain_dbi = round(_clamp(1.8 + (1.7 * aperture_factor) + (actual_total_efficiency_pct / 100.0) * 1.6 + float(rng.uniform(-0.6, 0.8)), -2.0, 11.0), 3)
        actual_directivity_dbi = round(_clamp(actual_gain_dbi + float(rng.uniform(1.4, 2.4)), actual_gain_dbi, 14.0), 3)
        actual_front_to_back_db = round(_clamp(10.0 + (3.0 * height_factor) + float(rng.uniform(-2.0, 3.0)), 6.0, 22.0), 3)
        actual_axial_ratio_db = round(float(rng.uniform(18.0, 40.0)), 3)
        accepted = _is_accepted(target_frequency_ghz, actual_center_frequency_ghz, actual_vswr, actual_return_loss_db)

        row: dict[str, object] = {
            **_common_inputs(
                family="microstrip_patch",
                patch_shape="rectangular",
                feed_type="edge",
                polarization="linear",
                substrate=substrate,
                conductor=conductor,
                target_frequency_ghz=target_frequency_ghz,
                target_bandwidth_mhz=target_bandwidth_mhz,
                target_minimum_gain_dbi=round(float(rng.uniform(2.0, 8.0)), 3),
                target_maximum_vswr=round(float(rng.uniform(1.5, 2.2)), 3),
                index=index,
                seed=seed,
            ),
            **geometry,
            "actual_peak_theta_deg": round(float(rng.uniform(0.0, 45.0)), 3),
            "actual_peak_phi_deg": round(float(rng.uniform(0.0, 180.0)), 3),
            "actual_front_to_back_db": actual_front_to_back_db,
            "actual_axial_ratio_db": actual_axial_ratio_db,
            "actual_center_frequency_ghz": actual_center_frequency_ghz,
            "actual_bandwidth_mhz": actual_bandwidth_mhz,
            "actual_return_loss_db": actual_return_loss_db,
            "actual_vswr": actual_vswr,
            "actual_gain_dbi": actual_gain_dbi,
            "actual_radiation_efficiency_pct": actual_radiation_efficiency_pct,
            "actual_total_efficiency_pct": actual_total_efficiency_pct,
            "actual_directivity_dbi": actual_directivity_dbi,
            "accepted": accepted,
            "solver_status": "completed" if accepted else "completed",
            "simulation_time_sec": round(float(rng.uniform(20.0, 55.0)), 3),
            "notes": "formula_bootstrap_rect_patch",
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows, columns=list(RECT_PATCH_SYNTH_COLUMNS))


def generate_amc_synth_dataset(*, rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, object]] = []

    for index in range(rows):
        substrate = _pick(rng, _AMC_SUBSTRATES)
        conductor = _pick(rng, _CONDUCTORS)
        substrate_height_mm = round(float(rng.uniform(0.8, 3.0)), 3)
        substrate["height_mm"] = substrate_height_mm
        target_frequency_ghz = round(float(rng.uniform(1.5, 6.0)), 4)
        target_bandwidth_mhz = round(float(rng.uniform(60.0, 450.0)), 3)
        target_minimum_gain_dbi = round(float(rng.uniform(4.0, 10.0)), 3)
        target_maximum_vswr = round(float(rng.uniform(1.4, 2.0)), 3)
        wavelength_mm = (C0 / (target_frequency_ghz * 1e9)) * 1000.0

        bandwidth_norm = _clamp((target_bandwidth_mhz - 60.0) / 390.0, 0.0, 1.0)
        gain_norm = _clamp((target_minimum_gain_dbi - 4.0) / 6.0, 0.0, 1.0)
        period_ratio = _clamp(0.18 + (0.07 * bandwidth_norm) + (0.03 * gain_norm) + float(rng.normal(0.0, 0.006)), 0.15, 0.30)
        patch_ratio = _clamp(0.84 - (0.12 * bandwidth_norm) + (0.05 * (1.0 - gain_norm)) + float(rng.normal(0.0, 0.008)), 0.60, 0.90)
        via_ratio = _clamp(0.025 + (0.012 * gain_norm) + float(rng.normal(0.0, 0.002)), 0.02, 0.05)
        air_gap_ratio = _clamp(0.025 + (0.025 * gain_norm) + (0.015 * bandwidth_norm) + float(rng.normal(0.0, 0.003)), 0.02, 0.08)

        amc_unit_cell_period_mm = round(wavelength_mm * period_ratio, 4)
        amc_patch_size_mm = round(amc_unit_cell_period_mm * patch_ratio, 4)
        amc_gap_mm = round(max(amc_unit_cell_period_mm - amc_patch_size_mm, 0.2), 4)
        amc_via_radius_mm = round(amc_unit_cell_period_mm * via_ratio, 4)
        amc_via_height_mm = substrate_height_mm
        amc_array_rows = int(round(_clamp(4.0 + (3.5 * gain_norm) + (2.0 * bandwidth_norm) + float(rng.uniform(-0.6, 0.6)), 3.0, 10.0)))
        amc_array_cols = int(round(_clamp(4.0 + (4.0 * gain_norm) + (1.5 * bandwidth_norm) + float(rng.uniform(-0.6, 0.6)), 3.0, 10.0)))
        amc_air_gap_mm = round(wavelength_mm * air_gap_ratio, 4)
        amc_ground_size_mm = round(max(amc_array_rows, amc_array_cols) * amc_unit_cell_period_mm, 4)

        gap_ratio = amc_gap_mm / max(amc_unit_cell_period_mm, 1e-6)
        air_ratio = amc_air_gap_mm / max(wavelength_mm, 1e-6)
        resonance_shift = float(rng.normal(0.0, 0.008)) + (0.15 * (0.76 - patch_ratio)) + (0.09 * (gap_ratio - 0.16)) - (0.06 * (air_ratio - 0.04))

        actual_reflection_phase_center_ghz = round(target_frequency_ghz * (1.0 + resonance_shift), 4)
        actual_reflection_phase_bandwidth_mhz = round(
            _clamp((target_frequency_ghz * 1000.0) * float(rng.uniform(0.05, 0.20)) * (1.0 + (0.5 * gap_ratio)), 25.0, 0.22 * target_frequency_ghz * 1000.0),
            3,
        )
        actual_gain_improvement_dbi = round(
            _clamp(2.0 + (0.16 * (amc_array_rows + amc_array_cols)) + (12.0 * (air_ratio - 0.02)) + (2.0 * (patch_ratio - 0.60)) + float(rng.uniform(-0.4, 0.5)), 2.0, 6.5),
            3,
        )
        actual_back_lobe_reduction_db = round(
            _clamp(10.0 + (0.9 * (amc_array_rows + amc_array_cols)) + (8.0 * patch_ratio) + float(rng.uniform(-2.0, 2.0)), 10.0, 25.0),
            3,
        )
        actual_center_frequency_ghz = actual_reflection_phase_center_ghz
        actual_bandwidth_mhz = actual_reflection_phase_bandwidth_mhz
        actual_return_loss_db = round(-_clamp(10.0 + (11.0 * (1.0 - min(abs(resonance_shift) * 5.5, 0.8))) + float(rng.uniform(-2.0, 2.0)), 7.0, 24.0), 3)
        actual_vswr = round(_clamp(1.08 + max(0.0, 2.0 - (abs(actual_return_loss_db) / 10.0)) + (abs(resonance_shift) * 3.5), 1.03, 3.5), 4)
        loss_tangent = float(substrate["loss_tangent"])
        actual_radiation_efficiency_pct = round(_clamp(82.0 - (450.0 * loss_tangent) + (0.9 * (amc_array_rows + amc_array_cols)) + float(rng.uniform(-3.0, 3.0)), 50.0, 98.0), 3)
        actual_total_efficiency_pct = round(_clamp(actual_radiation_efficiency_pct * float(rng.uniform(0.93, 0.99)), 42.0, actual_radiation_efficiency_pct), 3)
        actual_gain_dbi = round(_clamp(4.0 + actual_gain_improvement_dbi + float(rng.uniform(-0.5, 0.7)), 2.0, 13.5), 3)
        actual_directivity_dbi = round(_clamp(actual_gain_dbi + float(rng.uniform(1.0, 2.3)), actual_gain_dbi, 15.0), 3)
        accepted = _is_accepted(target_frequency_ghz, actual_center_frequency_ghz, actual_vswr, actual_return_loss_db)

        row: dict[str, object] = {
            **_common_inputs(
                family="amc_patch",
                patch_shape="rectangular",
                feed_type="surface_coupled",
                polarization="linear",
                substrate=substrate,
                conductor=conductor,
                target_frequency_ghz=target_frequency_ghz,
                target_bandwidth_mhz=target_bandwidth_mhz,
                target_minimum_gain_dbi=target_minimum_gain_dbi,
                target_maximum_vswr=target_maximum_vswr,
                index=index,
                seed=seed,
            ),
            "amc_unit_cell_period_mm": amc_unit_cell_period_mm,
            "amc_patch_size_mm": amc_patch_size_mm,
            "amc_gap_mm": amc_gap_mm,
            "amc_via_radius_mm": amc_via_radius_mm,
            "amc_via_height_mm": amc_via_height_mm,
            "amc_ground_size_mm": amc_ground_size_mm,
            "amc_array_rows": amc_array_rows,
            "amc_array_cols": amc_array_cols,
            "amc_air_gap_mm": amc_air_gap_mm,
            "actual_reflection_phase_center_ghz": actual_reflection_phase_center_ghz,
            "actual_reflection_phase_bandwidth_mhz": actual_reflection_phase_bandwidth_mhz,
            "actual_gain_improvement_dbi": actual_gain_improvement_dbi,
            "actual_back_lobe_reduction_db": actual_back_lobe_reduction_db,
            "actual_center_frequency_ghz": actual_center_frequency_ghz,
            "actual_bandwidth_mhz": actual_bandwidth_mhz,
            "actual_return_loss_db": actual_return_loss_db,
            "actual_vswr": actual_vswr,
            "actual_gain_dbi": actual_gain_dbi,
            "actual_radiation_efficiency_pct": actual_radiation_efficiency_pct,
            "actual_total_efficiency_pct": actual_total_efficiency_pct,
            "actual_directivity_dbi": actual_directivity_dbi,
            "accepted": accepted,
            "solver_status": "completed",
            "simulation_time_sec": round(float(rng.uniform(25.0, 70.0)), 3),
            "notes": "formula_bootstrap_amc_patch",
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows, columns=list(AMC_SYNTH_COLUMNS))


def generate_wban_synth_dataset(*, rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, object]] = []

    for index in range(rows):
        substrate = _pick(rng, _WBAN_SUBSTRATES)
        conductor = _pick(rng, _CONDUCTORS)
        substrate_height_mm = round(float(rng.uniform(0.8, 3.0)), 3)
        substrate["height_mm"] = substrate_height_mm
        target_frequency_ghz = round(float(rng.uniform(2.0, 6.0)), 4)
        design_frequency_ghz = target_frequency_ghz * float(rng.uniform(1.03, 1.10))
        target_bandwidth_mhz = round(float(rng.uniform(40.0, 280.0)), 3)
        target_minimum_gain_dbi = round(float(rng.uniform(1.0, 6.5)), 3)
        target_maximum_vswr = round(float(rng.uniform(1.5, 2.2)), 3)
        geometry = _microstrip_geometry(
            target_frequency_ghz=design_frequency_ghz,
            substrate_epsilon_r=float(substrate["epsilon_r"]),
            substrate_height_mm=substrate_height_mm,
            target_bandwidth_mhz=target_bandwidth_mhz,
            rng=rng,
            feed_offset_span=(0.25, 0.45),
            feed_width_span=(0.03, 0.08),
        )

        body_distance_mm = round(float(rng.uniform(2.0, 10.0)), 3)
        bending_radius_mm = round(float(rng.uniform(30.0, 100.0)), 3)
        bandwidth_norm = _clamp((target_bandwidth_mhz - 40.0) / 240.0, 0.0, 1.0)
        body_factor = (10.0 - body_distance_mm) / 8.0
        bend_factor = (100.0 - bending_radius_mm) / 70.0

        ground_slot_length_ratio = _clamp(0.22 + (0.20 * bandwidth_norm) + (0.06 * body_factor) + float(rng.normal(0.0, 0.01)), 0.20, 0.50)
        ground_slot_width_ratio = _clamp(0.03 + (0.04 * bandwidth_norm) + (0.02 * bend_factor) + float(rng.normal(0.0, 0.004)), 0.02, 0.10)
        notch_length_ratio = _clamp(0.06 + (0.06 * body_factor) + (0.05 * bandwidth_norm) + float(rng.normal(0.0, 0.01)), 0.05, 0.20)
        notch_width_ratio = _clamp(0.015 + (0.02 * bandwidth_norm) + (0.015 * bend_factor) + float(rng.normal(0.0, 0.003)), 0.01, 0.06)

        ground_slot_length_mm = round(ground_slot_length_ratio * geometry["patch_length_mm"], 4)
        ground_slot_width_mm = round(ground_slot_width_ratio * geometry["patch_width_mm"], 4)
        notch_length_mm = round(notch_length_ratio * geometry["patch_length_mm"], 4)
        notch_width_mm = round(notch_width_ratio * geometry["patch_width_mm"], 4)
        loading_shift = _clamp(0.02 + (0.04 * body_factor) + (0.02 * bend_factor) + float(rng.uniform(-0.004, 0.004)), 0.02, 0.08)
        actual_center_frequency_ghz = round(design_frequency_ghz * (1.0 - loading_shift), 4)
        actual_detuning_mhz = round(abs(actual_center_frequency_ghz - target_frequency_ghz) * 1000.0, 3)

        slot_factor = (ground_slot_length_mm / max(geometry["patch_length_mm"], 1e-6)) + (ground_slot_width_mm / max(geometry["patch_width_mm"], 1e-6))
        actual_bandwidth_mhz = round(
            _clamp(
                target_bandwidth_mhz
                * (0.82 + (0.22 * slot_factor) + (0.10 * (float(substrate["height_mm"]) / 2.0)) - (0.12 * body_factor) + float(rng.uniform(-0.04, 0.05))),
                10.0,
                target_bandwidth_mhz * 1.45,
            ),
            3,
        )
        actual_return_loss_db = round(-_clamp(10.0 + (10.0 * (1.0 - min(actual_detuning_mhz / max(target_frequency_ghz * 1000.0, 1e-6), 0.8))) + (3.0 * slot_factor) + float(rng.uniform(-2.0, 2.0)), 7.0, 28.0), 3)
        actual_vswr = round(_clamp(1.10 + max(0.0, 2.0 - (abs(actual_return_loss_db) / 10.0)) + (actual_detuning_mhz / 1000.0), 1.05, 3.8), 4)

        loss_tangent = float(substrate["loss_tangent"])
        free_space_eff = _clamp(92.0 - (520.0 * loss_tangent) + (3.0 * (float(substrate["height_mm"]) / 2.0)) + float(rng.uniform(-2.5, 2.5)), 50.0, 98.0)
        actual_efficiency_on_body_pct = round(_clamp((free_space_eff * float(rng.uniform(0.4, 0.8))) - (10.0 * body_factor) - (4.0 * bend_factor), 25.0, 90.0), 3)
        actual_radiation_efficiency_pct = actual_efficiency_on_body_pct
        actual_total_efficiency_pct = round(_clamp(actual_efficiency_on_body_pct * float(rng.uniform(0.94, 0.99)), 20.0, actual_efficiency_on_body_pct), 3)

        aperture_factor = (geometry["patch_length_mm"] * geometry["patch_width_mm"]) / 950.0
        actual_off_body_gain_dbi = round(_clamp(2.4 + (1.9 * aperture_factor) + float(rng.uniform(-0.6, 0.8)), 0.0, 9.0), 3)
        actual_on_body_gain_dbi = round(_clamp(actual_off_body_gain_dbi - (0.8 + (2.2 * body_factor) + (0.6 * bend_factor)) + float(rng.uniform(-0.4, 0.4)), -5.0, 7.0), 3)
        actual_gain_dbi = actual_on_body_gain_dbi
        actual_directivity_dbi = round(_clamp(actual_off_body_gain_dbi + float(rng.uniform(1.0, 2.2)), actual_off_body_gain_dbi, 12.0), 3)
        actual_sar_1g_wkg = round(_clamp(0.15 + (0.08 * target_frequency_ghz) + (4.2 / (body_distance_mm ** 1.35)) + float(rng.uniform(-0.08, 0.08)), 0.05, 3.0), 3)
        actual_sar_10g_wkg = round(_clamp(actual_sar_1g_wkg * float(rng.uniform(0.55, 0.80)), 0.03, actual_sar_1g_wkg), 3)
        accepted = _is_accepted(target_frequency_ghz, actual_center_frequency_ghz, actual_vswr, actual_return_loss_db)

        row: dict[str, object] = {
            **_common_inputs(
                family="wban_patch",
                patch_shape="rectangular",
                feed_type="edge",
                polarization="linear",
                substrate=substrate,
                conductor=conductor,
                target_frequency_ghz=target_frequency_ghz,
                target_bandwidth_mhz=target_bandwidth_mhz,
                target_minimum_gain_dbi=target_minimum_gain_dbi,
                target_maximum_vswr=target_maximum_vswr,
                index=index,
                seed=seed,
            ),
            **geometry,
            "body_distance_mm": body_distance_mm,
            "bending_radius_mm": bending_radius_mm,
            "ground_slot_length_mm": ground_slot_length_mm,
            "ground_slot_width_mm": ground_slot_width_mm,
            "notch_length_mm": notch_length_mm,
            "notch_width_mm": notch_width_mm,
            "actual_on_body_gain_dbi": actual_on_body_gain_dbi,
            "actual_off_body_gain_dbi": actual_off_body_gain_dbi,
            "actual_sar_1g_wkg": actual_sar_1g_wkg,
            "actual_sar_10g_wkg": actual_sar_10g_wkg,
            "actual_detuning_mhz": actual_detuning_mhz,
            "actual_efficiency_on_body_pct": actual_efficiency_on_body_pct,
            "actual_center_frequency_ghz": actual_center_frequency_ghz,
            "actual_bandwidth_mhz": actual_bandwidth_mhz,
            "actual_return_loss_db": actual_return_loss_db,
            "actual_vswr": actual_vswr,
            "actual_gain_dbi": actual_gain_dbi,
            "actual_radiation_efficiency_pct": actual_radiation_efficiency_pct,
            "actual_total_efficiency_pct": actual_total_efficiency_pct,
            "actual_directivity_dbi": actual_directivity_dbi,
            "accepted": accepted,
            "solver_status": "completed",
            "simulation_time_sec": round(float(rng.uniform(28.0, 75.0)), 3),
            "notes": "formula_bootstrap_wban_patch",
        }
        output_rows.append(row)

    return pd.DataFrame(output_rows, columns=list(WBAN_SYNTH_COLUMNS))


def write_family_synth_dataset(
    family: str,
    *,
    rows: int = 15000,
    seed: int = 42,
    csv_path: Path | None = None,
) -> FamilySynthArtifacts:
    normalized = family.strip().lower()
    generators: dict[str, Callable[..., pd.DataFrame]] = {
        "microstrip_patch": generate_rect_patch_synth_dataset,
        "amc_patch": generate_amc_synth_dataset,
        "wban_patch": generate_wban_synth_dataset,
    }
    if normalized not in generators:
        supported = ", ".join(sorted(generators))
        raise ValueError(f"Unsupported family '{family}'. Supported families: {supported}")

    dataset = generators[normalized](rows=rows, seed=seed)
    output_path = csv_path or _DEFAULT_OUTPUT_PATHS[normalized]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    accepted_rows = int(dataset["accepted"].astype(bool).sum()) if "accepted" in dataset else 0
    return FamilySynthArtifacts(family=normalized, csv_path=output_path, rows=len(dataset), accepted_rows=accepted_rows)


def write_all_family_synth_datasets(*, rows_per_family: int = 15000, seed: int = 42) -> list[FamilySynthArtifacts]:
    families = ("microstrip_patch", "amc_patch", "wban_patch")
    return [
        write_family_synth_dataset(family, rows=rows_per_family, seed=seed + offset)
        for offset, family in enumerate(families)
    ]


__all__ = [
    "AMC_SYNTH_COLUMNS",
    "RECT_PATCH_SYNTH_COLUMNS",
    "WBAN_SYNTH_COLUMNS",
    "FamilySynthArtifacts",
    "generate_amc_synth_dataset",
    "generate_rect_patch_synth_dataset",
    "generate_wban_synth_dataset",
    "write_all_family_synth_datasets",
    "write_family_synth_dataset",
]
