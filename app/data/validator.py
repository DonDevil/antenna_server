from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.data.schema import REQUIRED_COLUMNS
from config import BOUNDS


@dataclass
class ValidationResult:
    valid_df: pd.DataFrame
    rejected_df: pd.DataFrame


def _is_in_range(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.between(low, high, inclusive="both")


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = pd.Series(True, index=df.index)
    valid_mask &= _is_in_range(df["frequency_ghz"], *BOUNDS.frequency_ghz)
    valid_mask &= _is_in_range(df["bandwidth_mhz"], *BOUNDS.bandwidth_mhz)
    valid_mask &= _is_in_range(df["patch_length_mm"], *BOUNDS.patch_length_mm)
    valid_mask &= _is_in_range(df["patch_width_mm"], *BOUNDS.patch_width_mm)
    valid_mask &= _is_in_range(df["patch_height_mm"], *BOUNDS.patch_height_mm)
    valid_mask &= _is_in_range(df["substrate_length_mm"], *BOUNDS.substrate_length_mm)
    valid_mask &= _is_in_range(df["substrate_width_mm"], *BOUNDS.substrate_width_mm)
    valid_mask &= _is_in_range(df["substrate_height_mm"], *BOUNDS.substrate_height_mm)
    valid_mask &= _is_in_range(df["feed_length_mm"], *BOUNDS.feed_length_mm)
    valid_mask &= _is_in_range(df["feed_width_mm"], *BOUNDS.feed_width_mm)
    valid_mask &= _is_in_range(df["feed_offset_x_mm"], *BOUNDS.feed_offset_x_mm)
    valid_mask &= _is_in_range(df["feed_offset_y_mm"], *BOUNDS.feed_offset_y_mm)

    valid_df = df[valid_mask].copy()
    rejected_df = df[~valid_mask].copy()
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = "out_of_supported_range_or_invalid_numeric"
    return ValidationResult(valid_df=valid_df, rejected_df=rejected_df)
