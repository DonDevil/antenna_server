from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.rect_patch_trainer import train_rect_patch_inverse_ann
from app.data.rect_patch_feedback import build_rect_patch_datasets


def main() -> None:
    artifacts = build_rect_patch_datasets()
    print(f"Validated rows: {artifacts.valid_rows}")
    print(f"Rejected rows: {artifacts.rejected_rows}")
    print(f"Validated feedback: {artifacts.validated_feedback_path}")
    print(f"Inverse train CSV: {artifacts.inverse_train_path}")
    print(f"Forward train CSV: {artifacts.forward_train_path}")

    try:
        trained = train_rect_patch_inverse_ann(inverse_csv=artifacts.inverse_train_path)
    except ValueError as exc:
        print(f"Training skipped: {exc}")
        return

    print(f"Checkpoint: {trained.checkpoint_path}")
    print(f"Metadata: {trained.metadata_path}")
    print(f"Train rows: {trained.train_rows}")
    print(f"Validation rows: {trained.validation_rows}")
    print(f"Test rows: {trained.test_rows}")
    print(f"Best validation loss: {trained.best_validation_loss:.6f}")


if __name__ == "__main__":
    main()