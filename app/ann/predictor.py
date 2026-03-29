from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.ann.baseline import baseline_dimensions
from app.ann.model import InverseAnnRegressor
from app.core.schemas import AnnPrediction, DimensionPrediction, TargetSpec
from config import ANN_SETTINGS


class AnnPredictor:
    def __init__(self, checkpoint_path: Path | None = None, metadata_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or ANN_SETTINGS.checkpoint_path
        self.metadata_path = metadata_path or ANN_SETTINGS.metadata_path
        self._model = None
        self._meta = None

    def is_ready(self) -> bool:
        return self.checkpoint_path.exists() and self.metadata_path.exists()

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

    def predict(self, target: TargetSpec) -> AnnPrediction:
        if not self.is_ready():
            dims = baseline_dimensions(target.frequency_ghz, target.bandwidth_mhz)
            return AnnPrediction(
                ann_model_version="baseline_cold_start",
                confidence=0.35,
                dimensions=DimensionPrediction(**dims),
            )

        self._load()
        assert self._meta is not None
        assert self._model is not None

        x = np.array([[target.frequency_ghz, target.bandwidth_mhz]], dtype=np.float32)
        x_mean = np.array(self._meta["x_mean"], dtype=np.float32)
        x_std = np.array(self._meta["x_std"], dtype=np.float32)
        y_mean = np.array(self._meta["y_mean"], dtype=np.float32)
        y_std = np.array(self._meta["y_std"], dtype=np.float32)
        x_scaled = (x - x_mean) / x_std

        with torch.no_grad():
            y_scaled = self._model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
        y = (y_scaled * y_std) + y_mean

        values = {name: float(y[0][idx]) for idx, name in enumerate(ANN_SETTINGS.output_columns)}
        return AnnPrediction(
            ann_model_version=str(self._meta["model_version"]),
            confidence=0.8,
            dimensions=DimensionPrediction(**values),
        )
