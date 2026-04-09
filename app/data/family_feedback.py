from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.family_dataset_generators import AMC_SYNTH_COLUMNS, WBAN_SYNTH_COLUMNS
from config import AMC_PATCH_DATA_SETTINGS, BOUNDS, WBAN_PATCH_DATA_SETTINGS

AMC_PATCH_FEEDBACK_COLUMNS: tuple[str, ...] = AMC_SYNTH_COLUMNS
WBAN_PATCH_FEEDBACK_COLUMNS: tuple[str, ...] = WBAN_SYNTH_COLUMNS

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
)

_COMMON_LIMITS: dict[str, tuple[float, float]] = {
    "substrate_epsilon_r": (1.0, 20.0),
    "substrate_loss_tangent": (0.0, 1.0),
    "substrate_height_mm": (0.1, 10.0),
    "conductor_conductivity_s_per_m": (1e5, 1e9),
    "target_frequency_ghz": BOUNDS.frequency_ghz,
    "target_bandwidth_mhz": BOUNDS.bandwidth_mhz,
    "target_minimum_gain_dbi": (-100.0, 100.0),
    "target_maximum_vswr": (1.0, 100.0),
    "target_minimum_return_loss_db": (-120.0, 10.0),
    "actual_center_frequency_ghz": BOUNDS.frequency_ghz,
    "actual_bandwidth_mhz": BOUNDS.bandwidth_mhz,
    "actual_return_loss_db": (-120.0, 10.0),
    "actual_vswr": (1.0, 100.0),
    "actual_gain_dbi": (-100.0, 100.0),
    "actual_radiation_efficiency_pct": (0.0, 100.0),
    "actual_total_efficiency_pct": (0.0, 100.0),
    "actual_directivity_dbi": (-100.0, 100.0),
    "simulation_time_sec": (0.0, 86400.0),
}

_AMC_LIMITS: dict[str, tuple[float, float]] = {
    **_COMMON_LIMITS,
    "amc_unit_cell_period_mm": (3.0, 120.0),
    "amc_patch_size_mm": (2.0, 120.0),
    "amc_gap_mm": (0.05, 20.0),
    "amc_via_radius_mm": (0.01, 10.0),
    "amc_via_height_mm": (0.1, 20.0),
    "amc_ground_size_mm": (3.0, 150.0),
    "amc_array_rows": (1.0, 100.0),
    "amc_array_cols": (1.0, 100.0),
    "amc_air_gap_mm": (0.0, 30.0),
    "actual_reflection_phase_center_ghz": BOUNDS.frequency_ghz,
    "actual_reflection_phase_bandwidth_mhz": (0.0, 4000.0),
    "actual_gain_improvement_dbi": (-20.0, 30.0),
    "actual_back_lobe_reduction_db": (-20.0, 60.0),
}

_WBAN_LIMITS: dict[str, tuple[float, float]] = {
    **_COMMON_LIMITS,
    "patch_length_mm": BOUNDS.patch_length_mm,
    "patch_width_mm": BOUNDS.patch_width_mm,
    "patch_height_mm": BOUNDS.patch_height_mm,
    "substrate_length_mm": BOUNDS.substrate_length_mm,
    "substrate_width_mm": BOUNDS.substrate_width_mm,
    "feed_length_mm": BOUNDS.feed_length_mm,
    "feed_width_mm": (0.2, 8.0),
    "feed_offset_x_mm": BOUNDS.feed_offset_x_mm,
    "feed_offset_y_mm": (-50.0, 0.0),
    "body_distance_mm": (0.0, 50.0),
    "bending_radius_mm": (10.0, 500.0),
    "ground_slot_length_mm": (0.0, 80.0),
    "ground_slot_width_mm": (0.0, 40.0),
    "notch_length_mm": (0.0, 40.0),
    "notch_width_mm": (0.0, 20.0),
    "actual_on_body_gain_dbi": (-100.0, 100.0),
    "actual_off_body_gain_dbi": (-100.0, 100.0),
    "actual_sar_1g_wkg": (0.0, 20.0),
    "actual_sar_10g_wkg": (0.0, 20.0),
    "actual_detuning_mhz": (0.0, 2000.0),
    "actual_efficiency_on_body_pct": (0.0, 100.0),
}

_VALID_SOLVER_STATUSES = {"completed", "success"}


@dataclass
class FamilyFeedbackValidationResult:
    valid_df: pd.DataFrame
    rejected_df: pd.DataFrame


@dataclass
class FamilyFeedbackDatasetArtifacts:
    validated_feedback_path: Path
    rejected_feedback_path: Path
    valid_rows: int
    rejected_rows: int


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _to_bool(series: pd.Series) -> pd.Series:
    lowered = series.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "yes", "y"})


def _in_bounds(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.between(low, high, inclusive="both")


def _validate_family_feedback(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    family: str,
    numeric_limits: dict[str, tuple[float, float]],
) -> FamilyFeedbackValidationResult:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{family} feedback dataset is missing required columns: {missing_columns}")

    working = df.loc[:, list(columns)].copy()
    for column in _TEXT_COLUMNS:
        if column in working.columns:
            working[column] = _normalize_text(working[column])
    for column in working.columns:
        if column not in _TEXT_COLUMNS and column != "accepted":
            working[column] = pd.to_numeric(working[column], errors="coerce")
    if "accepted" in working.columns:
        working["accepted"] = _to_bool(working["accepted"])

    valid_mask = pd.Series(True, index=working.index)
    valid_mask &= working["antenna_family"].str.lower() == family
    valid_mask &= working["run_id"] != ""
    valid_mask &= working["timestamp_utc"] != ""
    valid_mask &= working["substrate_name"] != ""
    valid_mask &= working["conductor_name"] != ""
    valid_mask &= working["solver_status"].str.lower().isin(_VALID_SOLVER_STATUSES)

    for column, (low, high) in numeric_limits.items():
        if column in working.columns:
            valid_mask &= _in_bounds(working[column], low, high)

    valid_df = working[valid_mask].copy()
    rejected_df = working[~valid_mask].copy()
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = f"invalid_{family}_feedback_row"
    return FamilyFeedbackValidationResult(valid_df=valid_df, rejected_df=rejected_df)


def validate_amc_patch_feedback(df: pd.DataFrame) -> FamilyFeedbackValidationResult:
    return _validate_family_feedback(
        df,
        columns=AMC_PATCH_FEEDBACK_COLUMNS,
        family="amc_patch",
        numeric_limits=_AMC_LIMITS,
    )


def validate_wban_patch_feedback(df: pd.DataFrame) -> FamilyFeedbackValidationResult:
    return _validate_family_feedback(
        df,
        columns=WBAN_PATCH_FEEDBACK_COLUMNS,
        family="wban_patch",
        numeric_limits=_WBAN_LIMITS,
    )


def _validate_single_row(row: dict[str, Any], *, family: str, columns: tuple[str, ...]) -> dict[str, Any]:
    frame = pd.DataFrame([{column: row.get(column, "") for column in columns}], columns=list(columns))
    result = validate_amc_patch_feedback(frame) if family == "amc_patch" else validate_wban_patch_feedback(frame)
    if result.valid_df.empty:
        rejection_reason = result.rejected_df.iloc[0].get("rejection_reason", "invalid_row") if not result.rejected_df.empty else "invalid_row"
        raise ValueError(f"Invalid {family} feedback row: {rejection_reason}")
    return {column: result.valid_df.iloc[0][column] for column in columns}


def ensure_family_feedback_header(csv_path: Path, columns: tuple[str, ...]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()


def _row_already_logged(run_id: str, csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for existing in reader:
            if str(existing.get("run_id", "")).strip() == str(run_id).strip():
                return True
    return False


def _append_family_row(
    row: dict[str, Any],
    *,
    family: str,
    columns: tuple[str, ...],
    csv_path: Path,
) -> dict[str, Any]:
    normalized = _validate_single_row(row, family=family, columns=columns)
    ensure_family_feedback_header(csv_path, columns)
    if _row_already_logged(str(normalized.get("run_id", "")), csv_path):
        return normalized
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writerow(normalized)
    return normalized


def append_amc_patch_feedback_row(
    row: dict[str, Any],
    csv_path: Path = AMC_PATCH_DATA_SETTINGS.raw_feedback_path,
) -> dict[str, Any]:
    return _append_family_row(row, family="amc_patch", columns=AMC_PATCH_FEEDBACK_COLUMNS, csv_path=csv_path)


def append_wban_patch_feedback_row(
    row: dict[str, Any],
    csv_path: Path = WBAN_PATCH_DATA_SETTINGS.raw_feedback_path,
) -> dict[str, Any]:
    return _append_family_row(row, family="wban_patch", columns=WBAN_PATCH_FEEDBACK_COLUMNS, csv_path=csv_path)


def _build_family_datasets(
    *,
    raw_feedback_path: Path,
    validated_feedback_path: Path,
    rejected_feedback_path: Path,
    validator: Any,
) -> FamilyFeedbackDatasetArtifacts:
    raw_df = pd.read_csv(raw_feedback_path)
    result = validator(raw_df)
    validated_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    result.valid_df.to_csv(validated_feedback_path, index=False)
    result.rejected_df.to_csv(rejected_feedback_path, index=False)
    return FamilyFeedbackDatasetArtifacts(
        validated_feedback_path=validated_feedback_path,
        rejected_feedback_path=rejected_feedback_path,
        valid_rows=len(result.valid_df),
        rejected_rows=len(result.rejected_df),
    )


def build_amc_patch_datasets(
    *,
    raw_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.raw_feedback_path,
    validated_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.validated_feedback_path,
    rejected_feedback_path: Path = AMC_PATCH_DATA_SETTINGS.rejected_feedback_path,
) -> FamilyFeedbackDatasetArtifacts:
    return _build_family_datasets(
        raw_feedback_path=raw_feedback_path,
        validated_feedback_path=validated_feedback_path,
        rejected_feedback_path=rejected_feedback_path,
        validator=validate_amc_patch_feedback,
    )


def build_wban_patch_datasets(
    *,
    raw_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.raw_feedback_path,
    validated_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.validated_feedback_path,
    rejected_feedback_path: Path = WBAN_PATCH_DATA_SETTINGS.rejected_feedback_path,
) -> FamilyFeedbackDatasetArtifacts:
    return _build_family_datasets(
        raw_feedback_path=raw_feedback_path,
        validated_feedback_path=validated_feedback_path,
        rejected_feedback_path=rejected_feedback_path,
        validator=validate_wban_patch_feedback,
    )


__all__ = [
    "AMC_PATCH_FEEDBACK_COLUMNS",
    "WBAN_PATCH_FEEDBACK_COLUMNS",
    "FamilyFeedbackDatasetArtifacts",
    "FamilyFeedbackValidationResult",
    "append_amc_patch_feedback_row",
    "append_wban_patch_feedback_row",
    "build_amc_patch_datasets",
    "build_wban_patch_datasets",
    "validate_amc_patch_feedback",
    "validate_wban_patch_feedback",
]
