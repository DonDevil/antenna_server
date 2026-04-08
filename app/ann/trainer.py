from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ann.model import InverseAnnRegressor
from config import ANN_SETTINGS, DATA_SETTINGS


@dataclass
class TrainingArtifacts:
    checkpoint_path: Path
    metadata_path: Path
    final_loss: float
    train_rows: int


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    defaults: dict[str, float] = {
        "substrate_epsilon_r": 4.4,
        "minimum_gain_dbi": 0.0,
        "maximum_vswr": 2.0,
        "priority_s11_minimize": 1.0,
        "priority_bandwidth_maximize": 1.0,
        "priority_gain_maximize": 1.0,
        "priority_efficiency_maximize": 1.0,
        "family_is_amc_patch": 0.0,
        "family_is_microstrip_patch": 1.0,
        "family_is_wban_patch": 0.0,
        "shape_is_rectangular": 1.0,
        "shape_is_circular": 0.0,
    }
    for column in ANN_SETTINGS.input_columns:
        if column in enriched.columns:
            continue
        if column == "substrate_height_mm" and "substrate_height_mm" in enriched.columns:
            continue
        enriched[column] = defaults.get(column, 0.0)
    return enriched


def train_ann(validated_csv: Path, checkpoint_path: Path, metadata_path: Path, epochs: int = 200) -> TrainingArtifacts:
    df = _ensure_feature_columns(pd.read_csv(validated_csv))
    if len(df) < DATA_SETTINGS.min_rows_for_training:
        raise ValueError(
            f"Need at least {DATA_SETTINGS.min_rows_for_training} validated rows for training, found {len(df)}"
        )

    x = df[list(ANN_SETTINGS.input_columns)].to_numpy(dtype=np.float32)
    y = df[list(ANN_SETTINGS.output_columns)].to_numpy(dtype=np.float32)

    x_scaled, x_mean, x_std = _standardize(x)
    y_scaled, y_mean, y_std = _standardize(y)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=64, shuffle=True)

    model = InverseAnnRegressor(input_dim=x.shape[1], output_dim=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "output_dim": int(y.shape[1]),
        },
        checkpoint_path,
    )

    metadata = {
        "model_version": ANN_SETTINGS.model_version,
        "feature_schema_version": "ann_recipe_features.v2",
        "prediction_mode": "absolute_blend",
        "input_columns": list(ANN_SETTINGS.input_columns),
        "output_columns": list(ANN_SETTINGS.output_columns),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "rows": int(len(df)),
        "epochs": int(epochs),
        "final_loss": final_loss,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        final_loss=final_loss,
        train_rows=int(len(df)),
    )
