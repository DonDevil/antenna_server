from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.features import build_ann_feature_map
from app.antenna.recipes import generate_recipe
from app.core.schemas import (
    ClientCapabilities,
    DesignConstraints,
    OptimizationPolicy,
    OptimizeRequest,
    RuntimePreferences,
    TargetSpec,
)
from config import BOUNDS, DATA_SETTINGS


def synthesize_row(rng: np.random.Generator) -> dict:
    f = float(rng.uniform(*BOUNDS.frequency_ghz))
    bw = float(rng.uniform(20.0, 500.0))
    family = str(rng.choice(["microstrip_patch", "amc_patch", "wban_patch"], p=[0.5, 0.3, 0.2]))
    shape = str(rng.choice(["rectangular", "circular"], p=[0.75, 0.25])) if family == "microstrip_patch" else "rectangular"
    substrate = {
        "microstrip_patch": "Rogers RT/duroid 5880",
        "amc_patch": "FR-4 (lossy)",
        "wban_patch": "Rogers RO3003",
    }[family]
    min_gain = float(rng.uniform(0.0, 6.0))
    max_vswr = float(rng.uniform(1.6, 2.5))

    request = OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request=f"Design a {shape} {family} antenna at {f:.3f} GHz",
        target_spec=TargetSpec(
            frequency_ghz=f,
            bandwidth_mhz=bw,
            antenna_family=family,
            patch_shape=shape,
        ),
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=[substrate],
        ),
        optimization_policy=OptimizationPolicy(
            acceptance={
                "center_tolerance_mhz": 20.0,
                "minimum_bandwidth_mhz": bw,
                "maximum_vswr": max_vswr,
                "minimum_gain_dbi": min_gain,
            }
        ),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )

    recipe = generate_recipe(request)
    dims = dict(recipe["dimensions"])
    for name in (
        "patch_length_mm",
        "patch_width_mm",
        "substrate_length_mm",
        "substrate_width_mm",
        "feed_length_mm",
        "feed_width_mm",
    ):
        dims[name] = float(dims[name]) * float(rng.uniform(0.97, 1.03))
    dims["feed_offset_y_mm"] = float(dims["feed_offset_y_mm"]) + float(rng.uniform(-0.3, 0.3))

    row = {
        "frequency_ghz": f,
        "bandwidth_mhz": bw,
        **build_ann_feature_map(request, recipe),
        **dims,
    }
    return row


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
