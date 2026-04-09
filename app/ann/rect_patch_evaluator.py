from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from app.ann.predictor import AnnPredictor
from app.antenna.recipes import generate_recipe
from app.core.schemas import AcceptanceSpec, ClientCapabilities, DesignConstraints, OptimizationPolicy, OptimizeRequest, RuntimePreferences, TargetSpec
from config import RECT_PATCH_ANN_SETTINGS, RECT_PATCH_DATA_SETTINGS


EVALUATION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "patch_length_mm",
    "patch_width_mm",
    "feed_width_mm",
    "feed_offset_y_mm",
)


def _request_from_row(row: pd.Series) -> OptimizeRequest:
    return OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request=(
            f"Design a rectangular microstrip patch antenna at {float(row['target_frequency_ghz']):.3f} GHz"
        ),
        target_spec=TargetSpec(
            frequency_ghz=float(row["target_frequency_ghz"]),
            bandwidth_mhz=float(row["target_bandwidth_mhz"]),
            antenna_family="microstrip_patch",
            patch_shape="rectangular",
            feed_type="edge",
            polarization="linear",
        ),
        design_constraints=DesignConstraints(
            allowed_materials=[str(row["conductor_name"])],
            allowed_substrates=[str(row["substrate_name"])],
        ),
        optimization_policy=OptimizationPolicy(
            acceptance=AcceptanceSpec(
                minimum_gain_dbi=float(row["target_minimum_gain_dbi"]),
                maximum_vswr=float(row["target_maximum_vswr"]),
                minimum_return_loss_db=float(row["target_minimum_return_loss_db"]),
                minimum_bandwidth_mhz=float(row["target_bandwidth_mhz"]),
            )
        ),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )


def evaluate_rect_patch_inverse_ann(
    *,
    validated_feedback_path: Path = RECT_PATCH_DATA_SETTINGS.validated_feedback_path,
    checkpoint_path: Path = RECT_PATCH_ANN_SETTINGS.checkpoint_path,
    metadata_path: Path = RECT_PATCH_ANN_SETTINGS.metadata_path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    feedback_df = pd.read_csv(validated_feedback_path)
    if feedback_df.empty:
        raise ValueError("No validated rectangular-patch feedback rows available for evaluation")

    if max_rows is not None and max_rows > 0 and len(feedback_df) > max_rows:
        feedback_df = feedback_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    predictor = AnnPredictor(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
    if not predictor.warm_up():
        raise ValueError(f"Rectangular-patch ANN artifacts are not ready: {predictor.last_error()}")

    recipe_errors: list[float] = []
    ann_errors: list[float] = []
    ann_better_count = 0

    for _, row in feedback_df.iterrows():
        request = _request_from_row(row)
        recipe_prediction = generate_recipe(request)["dimensions"]
        ann_prediction = predictor.predict(request)

        row_recipe_errors: list[float] = []
        row_ann_errors: list[float] = []
        for column in EVALUATION_OUTPUT_COLUMNS:
            actual = float(row[column])
            denom = max(abs(actual), 1e-6)
            row_recipe_errors.append(abs(float(recipe_prediction[column]) - actual) / denom)
            row_ann_errors.append(abs(float(getattr(ann_prediction.dimensions, column)) - actual) / denom)

        recipe_mean_error = mean(row_recipe_errors)
        ann_mean_error = mean(row_ann_errors)
        recipe_errors.append(recipe_mean_error)
        ann_errors.append(ann_mean_error)
        if ann_mean_error < recipe_mean_error:
            ann_better_count += 1

    return {
        "rows": len(feedback_df),
        "recipe_mean_mape": mean(recipe_errors) * 100.0,
        "ann_mean_mape": mean(ann_errors) * 100.0,
        "recipe_median_mape": median(recipe_errors) * 100.0,
        "ann_median_mape": median(ann_errors) * 100.0,
        "ann_better_pct": (ann_better_count / len(feedback_df)) * 100.0,
        "modeled_columns": list(EVALUATION_OUTPUT_COLUMNS),
        "model_version": json.loads(metadata_path.read_text(encoding="utf-8")).get("model_version", RECT_PATCH_ANN_SETTINGS.model_version),
    }