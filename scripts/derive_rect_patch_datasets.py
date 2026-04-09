from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data.rect_patch_feedback import build_rect_patch_datasets


def main() -> None:
    artifacts = build_rect_patch_datasets()
    print(f"Validated rows: {artifacts.valid_rows}")
    print(f"Rejected rows: {artifacts.rejected_rows}")
    print(f"Validated feedback: {artifacts.validated_feedback_path}")
    print(f"Inverse train CSV: {artifacts.inverse_train_path}")
    print(f"Forward train CSV: {artifacts.forward_train_path}")


if __name__ == "__main__":
    main()