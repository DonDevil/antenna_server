from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.ann.features import build_ann_feature_map
from app.ann.model import InverseAnnRegressor
from app.antenna.recipes import generate_recipe
from app.core.schemas import (
    AnnPrediction,
    ClientCapabilities,
    DesignConstraints,
    DimensionPrediction,
    OptimizationPolicy,
    OptimizeRequest,
    RuntimePreferences,
    TargetSpec,
)
from config import ANN_SETTINGS


class AnnPredictor:
    def __init__(self, checkpoint_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or ANN_SETTINGS.checkpoint_path
        self.metadata_path = metadata_path or ANN_SETTINGS.metadata_path
        self._model = None
        self._meta: dict[str, Any] | None = None
        self._last_error: str | None = None

    def is_ready(self) -> bool:
        return self.checkpoint_path.exists() and self.metadata_path.exists()

    def is_loaded(self) -> bool:
        return self._model is not None and self._meta is not None

    def last_error(self) -> str | None:
        return self._last_error

    def warm_up(self) -> bool:
        if not self.is_ready():
            self._last_error = "ann_artifacts_missing"
            return False
        try:
            self._load()
        except Exception as exc:
            self._model = None
            self._meta = None
            self._last_error = str(exc)
            return False
        self._last_error = None
        return self.is_loaded()

    def _load(self) -> None:
        if self._model is not None:
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self._meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        model = InverseAnnRegressor(
            input_dim=int(checkpoint["input_dim"]),
            output_dim=int(checkpoint["output_dim"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._model = model
        self._last_error = None

    @staticmethod
    def _coerce_request(payload: OptimizeRequest | TargetSpec) -> OptimizeRequest:
        if isinstance(payload, OptimizeRequest):
            return payload
        return OptimizeRequest(
            schema_version="optimize_request.v1",
            user_request=f"Design a {payload.antenna_family} antenna at {payload.frequency_ghz} GHz",
            target_spec=payload,
            design_constraints=DesignConstraints(),
            optimization_policy=OptimizationPolicy(),
            runtime_preferences=RuntimePreferences(),
            client_capabilities=ClientCapabilities(),
        )

    @staticmethod
    def _baseline_result(request: OptimizeRequest, recipe: dict[str, Any], *, model_version: str, confidence: float, hint: str) -> AnnPrediction:
        dims = dict(recipe["dimensions"])
        return AnnPrediction(
            ann_model_version=model_version,
            confidence=confidence,
            dimensions=DimensionPrediction(**dims),
            recipe_name=str(recipe.get("recipe_name")),
            patch_shape=str(recipe.get("patch_shape")),
            optimizer_hint=hint,
        )

    @staticmethod
    def _combine_recipe_and_model(recipe: dict[str, Any], model_values: dict[str, float], mode: str) -> dict[str, float]:
        combined = {key: float(value) for key, value in dict(recipe["dimensions"]).items() if value is not None}
        for name in ANN_SETTINGS.output_columns:
            if name not in model_values:
                continue
            base = float(combined.get(name, 0.0))
            predicted = float(model_values[name])
            if mode.startswith("residual"):
                value = base + predicted
            else:
                value = (0.65 * base) + (0.35 * predicted)
            if name.endswith("_mm") and not name.startswith("feed_offset"):
                value = max(0.01, value)
            combined[name] = value

        if combined.get("patch_radius_mm") is None:
            combined["patch_radius_mm"] = max(0.01, float(combined.get("patch_width_mm", 1.0)) / 2.0)
        return combined

    def predict(self, payload: OptimizeRequest | TargetSpec) -> AnnPrediction:
        request = self._coerce_request(payload)
        recipe = generate_recipe(request)

        if not self.is_ready():
            return self._baseline_result(
                request,
                recipe,
                model_version="recipe_cold_start",
                confidence=0.55,
                hint="recipe_only",
            )

        try:
            self._load()
            assert self._meta is not None
            assert self._model is not None

            feature_map = build_ann_feature_map(request, recipe)
            input_columns = list(self._meta.get("input_columns", ANN_SETTINGS.input_columns))
            output_columns = list(self._meta.get("output_columns", ANN_SETTINGS.output_columns))

            x = np.array([[float(feature_map.get(name, 0.0)) for name in input_columns]], dtype=np.float32)
            x_mean = np.array(self._meta.get("x_mean", [0.0] * len(input_columns)), dtype=np.float32)
            x_std = np.array(self._meta.get("x_std", [1.0] * len(input_columns)), dtype=np.float32)
            y_mean = np.array(self._meta.get("y_mean", [0.0] * len(output_columns)), dtype=np.float32)
            y_std = np.array(self._meta.get("y_std", [1.0] * len(output_columns)), dtype=np.float32)
            x_std[x_std == 0] = 1.0
            y_std[y_std == 0] = 1.0
            x_scaled = (x - x_mean) / x_std

            with torch.no_grad():
                y_scaled = self._model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
            y = (y_scaled * y_std) + y_mean

            model_values = {name: float(y[0][idx]) for idx, name in enumerate(output_columns)}
            prediction_mode = str(self._meta.get("prediction_mode", "absolute_blend"))
            values = self._combine_recipe_and_model(recipe, model_values, prediction_mode)
            return AnnPrediction(
                ann_model_version=str(self._meta.get("model_version", ANN_SETTINGS.model_version)),
                confidence=0.82,
                dimensions=DimensionPrediction(**values),
                recipe_name=str(recipe.get("recipe_name")),
                patch_shape=str(recipe.get("patch_shape")),
                optimizer_hint="recipe_plus_ann_blend",
            )
        except Exception as exc:
            self._last_error = str(exc)
            return self._baseline_result(
                request,
                recipe,
                model_version="recipe_fallback_after_ann_error",
                confidence=0.48,
                hint="recipe_fallback",
            )
