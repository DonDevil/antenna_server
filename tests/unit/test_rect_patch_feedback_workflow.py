from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.rect_patch_feedback import (
    RECT_PATCH_FEEDBACK_COLUMNS,
    RECT_PATCH_FORWARD_INPUT_COLUMNS,
    RECT_PATCH_FORWARD_OUTPUT_COLUMNS,
    RECT_PATCH_INVERSE_INPUT_COLUMNS,
    RECT_PATCH_INVERSE_OUTPUT_COLUMNS,
    build_rect_patch_datasets,
    derive_rect_patch_training_frames,
    validate_rect_patch_feedback,
)


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-001",
        "timestamp_utc": "2026-04-09T12:00:00Z",
        "antenna_family": "microstrip_patch",
        "patch_shape": "rectangular",
        "feed_type": "edge",
        "polarization": "linear",
        "substrate_name": "Rogers RT/duroid 5880",
        "conductor_name": "Copper (annealed)",
        "target_frequency_ghz": 2.45,
        "target_bandwidth_mhz": 80.0,
        "target_minimum_gain_dbi": 3.0,
        "target_maximum_vswr": 2.0,
        "target_minimum_return_loss_db": -10.0,
        "substrate_epsilon_r": 2.2,
        "substrate_height_mm": 1.6,
        "patch_length_mm": 38.0,
        "patch_width_mm": 47.0,
        "patch_height_mm": 0.035,
        "substrate_length_mm": 54.0,
        "substrate_width_mm": 63.0,
        "feed_length_mm": 14.0,
        "feed_width_mm": 2.1,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": -7.5,
        "actual_center_frequency_ghz": 2.41,
        "actual_bandwidth_mhz": 88.0,
        "actual_return_loss_db": -17.0,
        "actual_vswr": 1.7,
        "actual_gain_dbi": 4.8,
        "actual_radiation_efficiency_pct": 82.0,
        "actual_total_efficiency_pct": 79.0,
        "actual_directivity_dbi": 6.1,
        "actual_peak_theta_deg": 20.0,
        "actual_peak_phi_deg": 180.0,
        "actual_front_to_back_db": 12.5,
        "actual_axial_ratio_db": 30.0,
        "accepted": True,
        "solver_status": "completed",
        "simulation_time_sec": 32.0,
        "notes": "baseline",
        "farfield_artifact_path": "farfield/run-001.json",
        "s11_artifact_path": "s11/run-001.csv",
    }
    row.update(overrides)
    return row


def test_validate_rect_patch_feedback_filters_invalid_rows() -> None:
    df = pd.DataFrame(
        [
            _valid_row(),
            _valid_row(run_id="run-002", feed_type="coaxial"),
        ],
        columns=list(RECT_PATCH_FEEDBACK_COLUMNS),
    )

    result = validate_rect_patch_feedback(df)

    assert len(result.valid_df) == 1
    assert len(result.rejected_df) == 1
    assert result.valid_df.iloc[0]["run_id"] == "run-001"
    assert result.rejected_df.iloc[0]["run_id"] == "run-002"


def test_derive_rect_patch_training_frames_uses_expected_column_order() -> None:
    validated_df = pd.DataFrame([_valid_row()], columns=list(RECT_PATCH_FEEDBACK_COLUMNS))

    inverse_df, forward_df = derive_rect_patch_training_frames(validated_df)

    assert list(inverse_df.columns) == [*RECT_PATCH_INVERSE_INPUT_COLUMNS, *RECT_PATCH_INVERSE_OUTPUT_COLUMNS]
    assert list(forward_df.columns) == [*RECT_PATCH_FORWARD_INPUT_COLUMNS, *RECT_PATCH_FORWARD_OUTPUT_COLUMNS]


def test_build_rect_patch_datasets_writes_all_outputs(tmp_path: Path) -> None:
    raw_path = tmp_path / "rect_patch_feedback.csv"
    validated_path = tmp_path / "validated.csv"
    rejected_path = tmp_path / "rejected.csv"
    inverse_path = tmp_path / "inverse.csv"
    forward_path = tmp_path / "forward.csv"

    pd.DataFrame([_valid_row()], columns=list(RECT_PATCH_FEEDBACK_COLUMNS)).to_csv(raw_path, index=False)

    artifacts = build_rect_patch_datasets(
        raw_feedback_path=raw_path,
        validated_feedback_path=validated_path,
        rejected_feedback_path=rejected_path,
        inverse_train_path=inverse_path,
        forward_train_path=forward_path,
    )

    assert artifacts.valid_rows == 1
    assert artifacts.rejected_rows == 0
    assert validated_path.exists()
    assert rejected_path.exists()
    assert inverse_path.exists()
    assert forward_path.exists()