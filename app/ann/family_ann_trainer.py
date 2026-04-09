from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ann.model import InverseAnnRegressor
from config import DATA_DIR, MODELS_DIR, RECT_PATCH_ANN_SETTINGS

FitStatus = Literal["balanced", "underfitting", "overfitting"]


@dataclass(frozen=True)
class FamilyAnnSpec:
    family: str
    model_version: str
    dataset_path: Path
    model_dir: Path
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    min_rows: int = 500
    safe_output_bounds: dict[str, tuple[float, float]] | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "inverse_ann.pt"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"


@dataclass(frozen=True)
class FamilyAnnTrainingArtifacts:
    family: str
    checkpoint_path: Path
    metadata_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    best_validation_loss: float
    train_mape: float
    validation_mape: float
    test_mape: float
    fit_status: FitStatus


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


def assess_fit_status(*, train_mape: float, validation_mape: float, test_mape: float) -> FitStatus:
    generalization_mape = max(validation_mape, test_mape)
    if train_mape <= 6.0 and generalization_mape > max(train_mape * 1.8, train_mape + 4.0):
        return "overfitting"
    if train_mape >= 8.0 and generalization_mape <= max(train_mape * 1.35, train_mape + 3.0):
        return "underfitting"
    return "balanced"


def _coerce_training_frame(spec: FamilyAnnSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.dataset_path)
    required_columns = [*spec.input_columns, *spec.output_columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset for family '{spec.family}' is missing required columns: {missing_columns}")

    working = df.loc[:, required_columns].copy()
    for column in required_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna().drop_duplicates().reset_index(drop=True)

    if len(working) < spec.min_rows:
        raise ValueError(
            f"Need at least {spec.min_rows} clean rows for family '{spec.family}', found {len(working)}"
        )
    return working


def _evaluate_predictions(predicted: np.ndarray, actual: np.ndarray, output_columns: tuple[str, ...]) -> tuple[float, dict[str, float], dict[str, float]]:
    mae = np.mean(np.abs(predicted - actual), axis=0)
    mape = np.mean(np.abs((predicted - actual) / np.maximum(np.abs(actual), 1e-6)), axis=0) * 100.0
    mae_by_column = {name: float(mae[idx]) for idx, name in enumerate(output_columns)}
    mape_by_column = {name: float(mape[idx]) for idx, name in enumerate(output_columns)}
    return float(mape.mean()), mae_by_column, mape_by_column


def _resolve_safe_bounds(spec: FamilyAnnSpec, frame: pd.DataFrame) -> dict[str, list[float]]:
    if spec.safe_output_bounds:
        return {name: [float(bounds[0]), float(bounds[1])] for name, bounds in spec.safe_output_bounds.items()}

    safe_bounds: dict[str, list[float]] = {}
    for column in spec.output_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        low = float(series.min())
        high = float(series.max())
        pad = max((high - low) * 0.03, 1e-3)
        safe_bounds[column] = [low - pad, high + pad]
    return safe_bounds


def train_family_ann_model(
    spec: FamilyAnnSpec,
    *,
    epochs: int = 180,
    batch_size: int = 128,
    learning_rate: float = 8e-4,
    early_stopping_patience: int = 18,
) -> FamilyAnnTrainingArtifacts:
    df = _coerce_training_frame(spec)
    x = df.loc[:, list(spec.input_columns)].to_numpy(dtype=np.float32)
    y = df.loc[:, list(spec.output_columns)].to_numpy(dtype=np.float32)

    x_scaled, x_mean, x_std = _standardize(x)
    y_scaled, y_mean, y_std = _standardize(y)
    train_idx, val_idx, test_idx = _split_indices(len(df))
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Need enough rows to create train/validation/test splits for family '{spec.family}'")

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
    train_x = torch.tensor(x_scaled[train_idx], dtype=torch.float32)
    test_x = torch.tensor(x_scaled[test_idx], dtype=torch.float32)

    hidden_dims = (128, 256, 128)
    model = InverseAnnRegressor(input_dim=x.shape[1], output_dim=y.shape[1], hidden_dims=hidden_dims)
    optimizer = cast(torch.optim.Optimizer, torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5))
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
        raise RuntimeError(f"Training did not produce a valid model state for family '{spec.family}'")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_pred_scaled = model(train_x).cpu().numpy()
        val_pred_scaled = model(val_x).cpu().numpy()
        test_pred_scaled = model(test_x).cpu().numpy()

    train_pred = (train_pred_scaled * y_std) + y_mean
    val_pred = (val_pred_scaled * y_std) + y_mean
    test_pred = (test_pred_scaled * y_std) + y_mean

    train_mape, train_mae_by_column, train_mape_by_column = _evaluate_predictions(train_pred, y[train_idx], spec.output_columns)
    validation_mape, validation_mae_by_column, validation_mape_by_column = _evaluate_predictions(val_pred, y[val_idx], spec.output_columns)
    test_mape, test_mae_by_column, test_mape_by_column = _evaluate_predictions(test_pred, y[test_idx], spec.output_columns)
    fit_status = assess_fit_status(train_mape=train_mape, validation_mape=validation_mape, test_mape=test_mape)

    spec.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "output_dim": int(y.shape[1]),
            "hidden_dims": list(hidden_dims),
        },
        spec.checkpoint_path,
    )

    metadata: dict[str, Any] = {
        "model_version": spec.model_version,
        "family": spec.family,
        "prediction_mode": "selected_output_override",
        "hidden_dims": list(hidden_dims),
        "input_columns": list(spec.input_columns),
        "output_columns": list(spec.output_columns),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "best_validation_loss": best_validation_loss,
        "train_mape": train_mape,
        "validation_mape": validation_mape,
        "test_mape": test_mape,
        "fit_status": fit_status,
        "train_mae": train_mae_by_column,
        "validation_mae": validation_mae_by_column,
        "test_mae": test_mae_by_column,
        "train_mape_by_column": train_mape_by_column,
        "validation_mape_by_column": validation_mape_by_column,
        "test_mape_by_column": test_mape_by_column,
        "safe_output_bounds": _resolve_safe_bounds(spec, df),
    }
    spec.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FamilyAnnTrainingArtifacts(
        family=spec.family,
        checkpoint_path=spec.checkpoint_path,
        metadata_path=spec.metadata_path,
        train_rows=len(train_idx),
        validation_rows=len(val_idx),
        test_rows=len(test_idx),
        best_validation_loss=best_validation_loss,
        train_mape=train_mape,
        validation_mape=validation_mape,
        test_mape=test_mape,
        fit_status=fit_status,
    )


def evaluate_family_ann_model(
    spec: FamilyAnnSpec,
    *,
    reference_df: pd.DataFrame,
    checkpoint_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path or spec.checkpoint_path, map_location="cpu")
    metadata = json.loads((metadata_path or spec.metadata_path).read_text(encoding="utf-8"))
    model = InverseAnnRegressor(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=(128, 256, 128),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    required_columns = [*spec.input_columns, *spec.output_columns]
    missing = [column for column in required_columns if column not in reference_df.columns]
    if missing:
        raise ValueError(f"Reference frame for family '{spec.family}' is missing required columns: {missing}")

    working = reference_df.loc[:, required_columns].copy()
    for column in required_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna().reset_index(drop=True)
    if working.empty:
        raise ValueError(f"Reference frame for family '{spec.family}' has no valid rows after cleanup")

    x = working.loc[:, list(spec.input_columns)].to_numpy(dtype=np.float32)
    y = working.loc[:, list(spec.output_columns)].to_numpy(dtype=np.float32)
    x_mean = np.array(metadata.get("x_mean", [0.0] * len(spec.input_columns)), dtype=np.float32)
    x_std = np.array(metadata.get("x_std", [1.0] * len(spec.input_columns)), dtype=np.float32)
    y_mean = np.array(metadata.get("y_mean", [0.0] * len(spec.output_columns)), dtype=np.float32)
    y_std = np.array(metadata.get("y_std", [1.0] * len(spec.output_columns)), dtype=np.float32)
    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    x_scaled = (x - x_mean) / x_std

    with torch.no_grad():
        predicted_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
    predicted = (predicted_scaled * y_std) + y_mean
    mean_mape, mae_by_column, mape_by_column = _evaluate_predictions(predicted, y, spec.output_columns)

    return {
        "family": spec.family,
        "rows": int(len(working)),
        "mean_mape": mean_mape,
        "mae_by_column": mae_by_column,
        "mape_by_column": mape_by_column,
    }


def default_family_specs() -> dict[str, FamilyAnnSpec]:
    return {
        "microstrip_patch": FamilyAnnSpec(
            family="microstrip_patch",
            model_version="rect_patch_formula_bootstrap_v2",
            dataset_path=DATA_DIR / "raw" / "rect_patch_formula_synth_v1.csv",
            model_dir=RECT_PATCH_ANN_SETTINGS.model_dir,
            input_columns=(
                "target_frequency_ghz",
                "target_bandwidth_mhz",
                "substrate_epsilon_r",
                "substrate_height_mm",
            ),
            output_columns=(
                "patch_length_mm",
                "patch_width_mm",
                "feed_width_mm",
                "feed_offset_y_mm",
            ),
            min_rows=500,
            safe_output_bounds={
                "patch_length_mm": (5.0, 120.0),
                "patch_width_mm": (5.0, 120.0),
                "feed_width_mm": (0.4, 8.0),
                "feed_offset_y_mm": (-50.0, 0.0),
            },
        ),
        "wban_patch": FamilyAnnSpec(
            family="wban_patch",
            model_version="wban_patch_formula_bootstrap_v1",
            dataset_path=DATA_DIR / "raw" / "wban_patch_formula_synth_v1.csv",
            model_dir=MODELS_DIR / "ann" / "wban_patch_v1",
            input_columns=(
                "target_frequency_ghz",
                "target_bandwidth_mhz",
                "substrate_epsilon_r",
                "substrate_height_mm",
                "body_distance_mm",
                "bending_radius_mm",
            ),
            output_columns=(
                "patch_length_mm",
                "patch_width_mm",
                "feed_width_mm",
                "feed_offset_y_mm",
                "ground_slot_length_mm",
                "ground_slot_width_mm",
            ),
            min_rows=2000,
        ),
    }


def train_all_family_anns(*, specs: dict[str, FamilyAnnSpec] | None = None) -> list[FamilyAnnTrainingArtifacts]:
    resolved_specs = specs or default_family_specs()
    return [train_family_ann_model(spec) for spec in resolved_specs.values()]


__all__ = [
    "FamilyAnnSpec",
    "FamilyAnnTrainingArtifacts",
    "assess_fit_status",
    "default_family_specs",
    "evaluate_family_ann_model",
    "train_all_family_anns",
    "train_family_ann_model",
]
