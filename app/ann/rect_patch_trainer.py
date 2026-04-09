from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ann.model import InverseAnnRegressor
from app.data.rect_patch_feedback import RECT_PATCH_INVERSE_INPUT_COLUMNS, RECT_PATCH_INVERSE_OUTPUT_COLUMNS
from config import BOUNDS, RECT_PATCH_ANN_SETTINGS, RECT_PATCH_DATA_SETTINGS


@dataclass
class RectPatchTrainingArtifacts:
    checkpoint_path: Path
    metadata_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    best_validation_loss: float


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def _split_indices(size: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    train_end = int(size * 0.70)
    val_end = int(size * 0.85)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _safe_bounds() -> dict[str, list[float]]:
    return {
        "patch_length_mm": list(BOUNDS.patch_length_mm),
        "patch_width_mm": list(BOUNDS.patch_width_mm),
        "feed_width_mm": [0.2, 8.0],
        "feed_offset_y_mm": [-50.0, 0.0],
    }


def train_rect_patch_inverse_ann(
    *,
    inverse_csv: Path = RECT_PATCH_DATA_SETTINGS.inverse_train_path,
    checkpoint_path: Path = RECT_PATCH_ANN_SETTINGS.checkpoint_path,
    metadata_path: Path = RECT_PATCH_ANN_SETTINGS.metadata_path,
    epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 20,
) -> RectPatchTrainingArtifacts:
    df = pd.read_csv(inverse_csv)
    if len(df) < RECT_PATCH_DATA_SETTINGS.min_rows_for_training:
        raise ValueError(
            "Need at least "
            f"{RECT_PATCH_DATA_SETTINGS.min_rows_for_training} validated rectangular-patch rows for training, "
            f"found {len(df)}"
        )

    x = df.loc[:, list(RECT_PATCH_INVERSE_INPUT_COLUMNS)].to_numpy(dtype=np.float32)
    y = df.loc[:, list(RECT_PATCH_INVERSE_OUTPUT_COLUMNS)].to_numpy(dtype=np.float32)

    x_scaled, x_mean, x_std = _standardize(x)
    y_scaled, y_mean, y_std = _standardize(y)
    train_idx, val_idx, test_idx = _split_indices(len(df))
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Need enough rows to create non-empty train, validation, and test splits")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_scaled[train_idx], dtype=torch.float32),
            torch.tensor(y_scaled[train_idx], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.tensor(x_scaled[val_idx], dtype=torch.float32)
    val_y = torch.tensor(y_scaled[val_idx], dtype=torch.float32)
    test_x = torch.tensor(x_scaled[test_idx], dtype=torch.float32)

    model = InverseAnnRegressor(input_dim=x.shape[1], output_dim=y.shape[1])
    optimizer = cast(torch.optim.Optimizer, torch.optim.Adam(model.parameters(), lr=learning_rate))
    criterion = nn.HuberLoss(delta=1.0)

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = early_stopping_patience

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x), val_y).cpu().item())

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred_scaled = model(test_x).cpu().numpy()
    test_pred = (test_pred_scaled * y_std) + y_mean
    test_actual = y[test_idx]
    test_mae = np.mean(np.abs(test_pred - test_actual), axis=0)
    test_mape = np.mean(np.abs((test_pred - test_actual) / np.maximum(np.abs(test_actual), 1e-6)), axis=0) * 100.0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "output_dim": int(y.shape[1]),
        },
        checkpoint_path,
    )

    metadata: dict[str, Any] = {
        "model_version": RECT_PATCH_ANN_SETTINGS.model_version,
        "family": RECT_PATCH_ANN_SETTINGS.family,
        "patch_shape": RECT_PATCH_ANN_SETTINGS.patch_shape,
        "feed_type": RECT_PATCH_ANN_SETTINGS.feed_type,
        "feature_schema_version": "rect_patch_feedback.v1",
        "prediction_mode": "selected_output_override",
        "input_columns": list(RECT_PATCH_INVERSE_INPUT_COLUMNS),
        "output_columns": list(RECT_PATCH_INVERSE_OUTPUT_COLUMNS),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "best_validation_loss": best_validation_loss,
        "test_mae": {name: float(test_mae[idx]) for idx, name in enumerate(RECT_PATCH_INVERSE_OUTPUT_COLUMNS)},
        "test_mape": {name: float(test_mape[idx]) for idx, name in enumerate(RECT_PATCH_INVERSE_OUTPUT_COLUMNS)},
        "safe_output_bounds": _safe_bounds(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return RectPatchTrainingArtifacts(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        train_rows=len(train_idx),
        validation_rows=len(val_idx),
        test_rows=len(test_idx),
        best_validation_loss=best_validation_loss,
    )