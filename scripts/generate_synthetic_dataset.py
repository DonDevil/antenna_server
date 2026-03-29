from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BOUNDS, DATA_SETTINGS


def synthesize_row(rng: np.random.Generator) -> dict:
    f = float(rng.uniform(*BOUNDS.frequency_ghz))
    bw = float(rng.uniform(20.0, 500.0))

    eps_eff = 4.2
    c = 3e8
    patch_length_mm = (c / (2 * (f * 1e9) * math.sqrt(eps_eff))) * 1e3
    patch_width_mm = patch_length_mm * float(rng.uniform(1.05, 1.3))
    substrate_height_mm = float(rng.uniform(*BOUNDS.substrate_height_mm))
    patch_height_mm = 0.035
    substrate_length_mm = patch_length_mm + 6 * substrate_height_mm
    substrate_width_mm = patch_width_mm + 6 * substrate_height_mm
    feed_length_mm = patch_length_mm * float(rng.uniform(0.3, 0.6))
    feed_width_mm = max(0.5, min(8.0, bw / 120.0))

    return {
        "frequency_ghz": f,
        "bandwidth_mhz": bw,
        "patch_length_mm": patch_length_mm,
        "patch_width_mm": patch_width_mm,
        "patch_height_mm": patch_height_mm,
        "substrate_length_mm": substrate_length_mm,
        "substrate_width_mm": substrate_width_mm,
        "substrate_height_mm": substrate_height_mm,
        "feed_length_mm": feed_length_mm,
        "feed_width_mm": feed_width_mm,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": -patch_length_mm / 4,
    }


def main() -> None:
    rng = np.random.default_rng(42)
    rows = [synthesize_row(rng) for _ in range(3000)]
    df = pd.DataFrame(rows)
    DATA_SETTINGS.raw_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_SETTINGS.raw_dataset_path, index=False)
    print(f"Generated dataset: {DATA_SETTINGS.raw_dataset_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
