from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.rect_patch_trainer import train_rect_patch_inverse_ann


def main() -> None:
    artifacts = train_rect_patch_inverse_ann()
    print(f"Checkpoint: {artifacts.checkpoint_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Validation rows: {artifacts.validation_rows}")
    print(f"Test rows: {artifacts.test_rows}")
    print(f"Best validation loss: {artifacts.best_validation_loss:.6f}")


if __name__ == "__main__":
    main()