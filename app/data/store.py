from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.validator import ValidationResult, validate_dataset


def ingest_and_validate(raw_csv_path: Path, validated_out: Path, rejected_out: Path) -> ValidationResult:
    df = pd.read_csv(raw_csv_path)
    result = validate_dataset(df)
    validated_out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    result.valid_df.to_csv(validated_out, index=False)
    result.rejected_df.to_csv(rejected_out, index=False)
    return result
