from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data.store import ingest_and_validate
from config import DATA_SETTINGS


def main() -> None:
    result = ingest_and_validate(
        raw_csv_path=DATA_SETTINGS.raw_dataset_path,
        validated_out=DATA_SETTINGS.validated_dataset_path,
        rejected_out=DATA_SETTINGS.rejected_dataset_path,
    )
    print(f"Validated rows: {len(result.valid_df)}")
    print(f"Rejected rows: {len(result.rejected_df)}")


if __name__ == "__main__":
    main()
