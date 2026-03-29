from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.trainer import train_ann
from config import ANN_SETTINGS, DATA_SETTINGS


def main() -> None:
    artifacts = train_ann(
        validated_csv=DATA_SETTINGS.validated_dataset_path,
        checkpoint_path=ANN_SETTINGS.checkpoint_path,
        metadata_path=ANN_SETTINGS.metadata_path,
        epochs=250,
    )
    print(f"Checkpoint: {artifacts.checkpoint_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Rows: {artifacts.train_rows}")
    print(f"Final Loss: {artifacts.final_loss:.6f}")


if __name__ == "__main__":
    main()
