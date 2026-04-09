from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data import family_feedback as family_feedback


def _amc_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "amc-run-001",
        "timestamp_utc": "2026-04-09T12:00:00Z",
        "antenna_family": "amc_patch",
        "patch_shape": "rectangular",
        "feed_type": "aperture",
        "polarization": "linear",
        "substrate_name": "FR-4 (lossy)",
        "substrate_epsilon_r": 4.4,
        "substrate_loss_tangent": 0.02,
        "substrate_height_mm": 1.6,
        "conductor_name": "Copper (annealed)",
        "conductor_conductivity_s_per_m": 5.8e7,
        "target_frequency_ghz": 2.45,
        "target_bandwidth_mhz": 80.0,
        "target_minimum_gain_dbi": 5.0,
        "target_maximum_vswr": 2.0,
        "target_minimum_return_loss_db": -12.0,
        "amc_unit_cell_period_mm": 18.0,
        "amc_patch_size_mm": 14.5,
        "amc_gap_mm": 1.2,
        "amc_via_radius_mm": 0.5,
        "amc_via_height_mm": 1.6,
        "amc_ground_size_mm": 21.0,
        "amc_array_rows": 6,
        "amc_array_cols": 6,
        "amc_air_gap_mm": 3.5,
        "actual_reflection_phase_center_ghz": 2.44,
        "actual_reflection_phase_bandwidth_mhz": 120.0,
        "actual_gain_improvement_dbi": 3.1,
        "actual_back_lobe_reduction_db": 7.2,
        "actual_center_frequency_ghz": 2.43,
        "actual_bandwidth_mhz": 86.0,
        "actual_return_loss_db": -19.0,
        "actual_vswr": 1.4,
        "actual_gain_dbi": 6.2,
        "actual_radiation_efficiency_pct": 78.0,
        "actual_total_efficiency_pct": 74.0,
        "actual_directivity_dbi": 7.8,
        "accepted": True,
        "solver_status": "completed",
        "simulation_time_sec": 45.0,
        "notes": "good amc result",
    }
    row.update(overrides)
    return row


def _wban_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "wban-run-001",
        "timestamp_utc": "2026-04-09T12:00:00Z",
        "antenna_family": "wban_patch",
        "patch_shape": "rectangular",
        "feed_type": "edge",
        "polarization": "linear",
        "substrate_name": "Rogers RO3003",
        "substrate_epsilon_r": 3.0,
        "substrate_loss_tangent": 0.0013,
        "substrate_height_mm": 1.2,
        "conductor_name": "Copper (annealed)",
        "conductor_conductivity_s_per_m": 5.8e7,
        "target_frequency_ghz": 2.45,
        "target_bandwidth_mhz": 100.0,
        "target_minimum_gain_dbi": 4.0,
        "target_maximum_vswr": 2.0,
        "target_minimum_return_loss_db": -10.0,
        "patch_length_mm": 33.0,
        "patch_width_mm": 40.0,
        "patch_height_mm": 0.035,
        "substrate_length_mm": 48.0,
        "substrate_width_mm": 56.0,
        "feed_length_mm": 12.0,
        "feed_width_mm": 1.9,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": -6.5,
        "body_distance_mm": 4.0,
        "bending_radius_mm": 65.0,
        "ground_slot_length_mm": 10.0,
        "ground_slot_width_mm": 2.5,
        "notch_length_mm": 3.0,
        "notch_width_mm": 1.5,
        "actual_on_body_gain_dbi": 4.1,
        "actual_off_body_gain_dbi": 5.0,
        "actual_sar_1g_wkg": 0.8,
        "actual_sar_10g_wkg": 0.4,
        "actual_detuning_mhz": 18.0,
        "actual_efficiency_on_body_pct": 70.0,
        "actual_center_frequency_ghz": 2.46,
        "actual_bandwidth_mhz": 104.0,
        "actual_return_loss_db": -17.5,
        "actual_vswr": 1.5,
        "actual_gain_dbi": 4.7,
        "actual_radiation_efficiency_pct": 74.0,
        "actual_total_efficiency_pct": 69.0,
        "actual_directivity_dbi": 6.2,
        "accepted": True,
        "solver_status": "completed",
        "simulation_time_sec": 38.0,
        "notes": "good wban result",
    }
    row.update(overrides)
    return row


def test_validate_amc_patch_feedback_filters_invalid_rows() -> None:
    df = pd.DataFrame(
        [
            _amc_row(),
            _amc_row(run_id="amc-run-002", antenna_family="microstrip_patch"),
        ],
        columns=list(family_feedback.AMC_PATCH_FEEDBACK_COLUMNS),
    )

    result = family_feedback.validate_amc_patch_feedback(df)

    assert len(result.valid_df) == 1
    assert len(result.rejected_df) == 1
    assert result.valid_df.iloc[0]["run_id"] == "amc-run-001"


def test_validate_wban_patch_feedback_filters_invalid_rows() -> None:
    df = pd.DataFrame(
        [
            _wban_row(),
            _wban_row(run_id="wban-run-002", body_distance_mm=-1.0),
        ],
        columns=list(family_feedback.WBAN_PATCH_FEEDBACK_COLUMNS),
    )

    result = family_feedback.validate_wban_patch_feedback(df)

    assert len(result.valid_df) == 1
    assert len(result.rejected_df) == 1
    assert result.valid_df.iloc[0]["run_id"] == "wban-run-001"


def test_build_family_feedback_datasets_write_outputs(tmp_path: Path) -> None:
    amc_raw = tmp_path / "amc_raw.csv"
    wban_raw = tmp_path / "wban_raw.csv"
    pd.DataFrame([_amc_row()], columns=list(family_feedback.AMC_PATCH_FEEDBACK_COLUMNS)).to_csv(amc_raw, index=False)
    pd.DataFrame([_wban_row()], columns=list(family_feedback.WBAN_PATCH_FEEDBACK_COLUMNS)).to_csv(wban_raw, index=False)

    amc_artifacts = family_feedback.build_amc_patch_datasets(
        raw_feedback_path=amc_raw,
        validated_feedback_path=tmp_path / "amc_validated.csv",
        rejected_feedback_path=tmp_path / "amc_rejected.csv",
    )
    wban_artifacts = family_feedback.build_wban_patch_datasets(
        raw_feedback_path=wban_raw,
        validated_feedback_path=tmp_path / "wban_validated.csv",
        rejected_feedback_path=tmp_path / "wban_rejected.csv",
    )

    assert amc_artifacts.valid_rows == 1
    assert wban_artifacts.valid_rows == 1
    assert (tmp_path / "amc_validated.csv").exists()
    assert (tmp_path / "wban_validated.csv").exists()
