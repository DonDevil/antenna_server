from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import BOUNDS, RECT_PATCH_DATA_SETTINGS


RECT_PATCH_FEEDBACK_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "antenna_family",
    "patch_shape",
    "feed_type",
    "polarization",
    "substrate_name",
    "conductor_name",
    "target_frequency_ghz",
    "target_bandwidth_mhz",
    "target_minimum_gain_dbi",
    "target_maximum_vswr",
    "target_minimum_return_loss_db",
    "substrate_epsilon_r",
    "substrate_height_mm",
    "patch_length_mm",
    "patch_width_mm",
    "patch_height_mm",
    "substrate_length_mm",
    "substrate_width_mm",
    "feed_length_mm",
    "feed_width_mm",
    "feed_offset_x_mm",
    "feed_offset_y_mm",
    "actual_center_frequency_ghz",
    "actual_bandwidth_mhz",
    "actual_return_loss_db",
    "actual_vswr",
    "actual_gain_dbi",
    "actual_radiation_efficiency_pct",
    "actual_total_efficiency_pct",
    "actual_directivity_dbi",
    "actual_peak_theta_deg",
    "actual_peak_phi_deg",
    "actual_front_to_back_db",
    "actual_axial_ratio_db",
    "accepted",
    "solver_status",
    "simulation_time_sec",
    "notes",
    "farfield_artifact_path",
    "s11_artifact_path",
)

RECT_PATCH_INVERSE_INPUT_COLUMNS: tuple[str, ...] = (
    "target_frequency_ghz",
    "target_bandwidth_mhz",
    "substrate_epsilon_r",
    "substrate_height_mm",
)

RECT_PATCH_INVERSE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "patch_length_mm",
    "patch_width_mm",
    "feed_width_mm",
    "feed_offset_y_mm",
)

RECT_PATCH_FORWARD_OUTPUT_COLUMNS: tuple[str, ...] = (
    "actual_center_frequency_ghz",
    "actual_bandwidth_mhz",
    "actual_return_loss_db",
    "actual_vswr",
    "actual_gain_dbi",
    "actual_radiation_efficiency_pct",
    "actual_total_efficiency_pct",
    "actual_directivity_dbi",
    "actual_front_to_back_db",
    "actual_axial_ratio_db",
)

RECT_PATCH_FORWARD_INPUT_COLUMNS: tuple[str, ...] = (
    *RECT_PATCH_INVERSE_INPUT_COLUMNS,
    *RECT_PATCH_INVERSE_OUTPUT_COLUMNS,
)

_TEXT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "antenna_family",
    "patch_shape",
    "feed_type",
    "polarization",
    "substrate_name",
    "conductor_name",
    "solver_status",
    "notes",
    "farfield_artifact_path",
    "s11_artifact_path",
)

_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in RECT_PATCH_FEEDBACK_COLUMNS
    if column not in _TEXT_COLUMNS and column != "accepted"
)

_FIXED_TEXT_EXPECTATIONS: dict[str, str] = {
    "antenna_family": "microstrip_patch",
    "patch_shape": "rectangular",
    "feed_type": "edge",
}

_VALID_SOLVER_STATUSES = {"completed", "success"}

_RECT_PATCH_LIMITS: dict[str, tuple[float, float]] = {
    "target_frequency_ghz": (2.0, 7.0),
    "target_bandwidth_mhz": (30.0, 300.0),
    "substrate_epsilon_r": (2.2, 4.4),
    "substrate_height_mm": (0.8, 3.2),
    "patch_length_mm": (5.0, 80.0),
    "patch_width_mm": (5.0, 100.0),
    "patch_height_mm": BOUNDS.patch_height_mm,
    "substrate_length_mm": BOUNDS.substrate_length_mm,
    "substrate_width_mm": BOUNDS.substrate_width_mm,
    "feed_length_mm": BOUNDS.feed_length_mm,
    "feed_width_mm": (0.2, 8.0),
    "feed_offset_x_mm": BOUNDS.feed_offset_x_mm,
    "feed_offset_y_mm": (-50.0, 0.0),
    "actual_center_frequency_ghz": BOUNDS.frequency_ghz,
    "actual_bandwidth_mhz": BOUNDS.bandwidth_mhz,
    "actual_return_loss_db": (-120.0, 10.0),
    "actual_vswr": (1.0, 100.0),
    "actual_gain_dbi": (-100.0, 100.0),
    "actual_radiation_efficiency_pct": (0.0, 100.0),
    "actual_total_efficiency_pct": (0.0, 100.0),
    "actual_directivity_dbi": (-100.0, 100.0),
    "actual_peak_theta_deg": (0.0, 360.0),
    "actual_peak_phi_deg": (0.0, 360.0),
    "actual_front_to_back_db": (-100.0, 100.0),
    "actual_axial_ratio_db": (0.0, 100.0),
    "simulation_time_sec": (0.0, 86400.0),
    "target_minimum_gain_dbi": (-100.0, 100.0),
    "target_maximum_vswr": (1.0, 100.0),
    "target_minimum_return_loss_db": (-120.0, 10.0),
}


@dataclass
class RectPatchFeedbackValidationResult:
    valid_df: pd.DataFrame
    rejected_df: pd.DataFrame


@dataclass
class RectPatchDatasetArtifacts:
    validated_feedback_path: Path
    rejected_feedback_path: Path
    inverse_train_path: Path
    forward_train_path: Path
    valid_rows: int
    rejected_rows: int


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _to_bool(series: pd.Series) -> pd.Series:
    lowered = series.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y"})


def _in_bounds(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.between(low, high, inclusive="both")


def validate_rect_patch_feedback(df: pd.DataFrame) -> RectPatchFeedbackValidationResult:
    missing_columns = [column for column in RECT_PATCH_FEEDBACK_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Rectangular patch feedback dataset is missing required columns: {missing_columns}")

    working = df.loc[:, list(RECT_PATCH_FEEDBACK_COLUMNS)].copy()
    for column in _TEXT_COLUMNS:
        working[column] = _normalize_text(working[column])
    for column in _NUMERIC_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["accepted"] = _to_bool(working["accepted"])

    valid_mask = pd.Series(True, index=working.index)

    for column, expected_value in _FIXED_TEXT_EXPECTATIONS.items():
        valid_mask &= working[column].str.lower() == expected_value

    valid_mask &= working["polarization"].str.lower().isin({"", "linear"})
    valid_mask &= working["solver_status"].str.lower().isin(_VALID_SOLVER_STATUSES)
    valid_mask &= working["run_id"] != ""
    valid_mask &= working["timestamp_utc"] != ""
    valid_mask &= working["substrate_name"] != ""
    valid_mask &= working["conductor_name"] != ""

    for column, (low, high) in _RECT_PATCH_LIMITS.items():
        valid_mask &= _in_bounds(working[column], low, high)

    valid_df = working[valid_mask].copy()
    rejected_df = working[~valid_mask].copy()
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = "invalid_rect_patch_feedback_row"

    return RectPatchFeedbackValidationResult(valid_df=valid_df, rejected_df=rejected_df)


def derive_rect_patch_training_frames(validated_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inverse_columns = [*RECT_PATCH_INVERSE_INPUT_COLUMNS, *RECT_PATCH_INVERSE_OUTPUT_COLUMNS]
    forward_columns = [*RECT_PATCH_FORWARD_INPUT_COLUMNS, *RECT_PATCH_FORWARD_OUTPUT_COLUMNS]
    return validated_df.loc[:, inverse_columns].copy(), validated_df.loc[:, forward_columns].copy()


def build_rect_patch_datasets(
    *,
    raw_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.raw_feedback_path,
    validated_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.validated_feedback_path,
    rejected_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.rejected_feedback_path,
    inverse_train_path: Path = RECT_PATCH_DATA_SETTINGS.inverse_train_path,
    forward_train_path: Path = RECT_PATCH_DATA_SETTINGS.forward_train_path,
) -> RectPatchDatasetArtifacts:
    raw_df = pd.read_csv(raw_feedback_path)
    result = validate_rect_patch_feedback(raw_df)
    inverse_df, forward_df = derive_rect_patch_training_frames(result.valid_df)

    validated_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    inverse_train_path.parent.mkdir(parents=True, exist_ok=True)
    forward_train_path.parent.mkdir(parents=True, exist_ok=True)

    result.valid_df.to_csv(validated_feedback_path, index=False)
    result.rejected_df.to_csv(rejected_feedback_path, index=False)
    inverse_df.to_csv(inverse_train_path, index=False)
    forward_df.to_csv(forward_train_path, index=False)

    return RectPatchDatasetArtifacts(
        validated_feedback_path=validated_feedback_path,
        rejected_feedback_path=rejected_feedback_path,
        inverse_train_path=inverse_train_path,
        forward_train_path=forward_train_path,
        valid_rows=len(result.valid_df),
        rejected_rows=len(result.rejected_df),
    )