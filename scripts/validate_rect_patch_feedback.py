from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from app.data.rect_patch_feedback import validate_rect_patch_feedback
from config import RECT_PATCH_DATA_SETTINGS


def main() -> None:
    raw_df = pd.read_csv(RECT_PATCH_DATA_SETTINGS.raw_feedback_path)
    result = validate_rect_patch_feedback(raw_df)
    RECT_PATCH_DATA_SETTINGS.validated_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    RECT_PATCH_DATA_SETTINGS.rejected_feedback_path.parent.mkdir(parents=True, exist_ok=True)
    result.valid_df.to_csv(RECT_PATCH_DATA_SETTINGS.validated_feedback_path, index=False)
    result.rejected_df.to_csv(RECT_PATCH_DATA_SETTINGS.rejected_feedback_path, index=False)
    print(f"Validated rows: {len(result.valid_df)}")
    print(f"Rejected rows: {len(result.rejected_df)}")
    print(f"Validated CSV: {RECT_PATCH_DATA_SETTINGS.validated_feedback_path}")
    print(f"Rejected CSV: {RECT_PATCH_DATA_SETTINGS.rejected_feedback_path}")


if __name__ == "__main__":
    main()