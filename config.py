from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONTEXT_DIR = ROOT_DIR / "context_files"
SCHEMAS_DIR = ROOT_DIR / "schemas"
SESSIONS_DIR = ROOT_DIR / "sessions"


@dataclass(frozen=True)
class ApiSettings:
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "AMC Antenna Optimization Server"
    version: str = "0.1.0"


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    model_name: str = "deepseek-r1:8b"
    timeout_sec: int = 45


@dataclass(frozen=True)
class AnnSettings:
    model_version: str = "v1"
    checkpoint_path: Path = MODELS_DIR / "ann" / "v1" / "inverse_ann.pt"
    metadata_path: Path = MODELS_DIR / "ann" / "v1" / "metadata.json"
    input_columns: tuple[str, ...] = ("frequency_ghz", "bandwidth_mhz")
    output_columns: tuple[str, ...] = (
        "patch_length_mm",
        "patch_width_mm",
        "patch_height_mm",
        "substrate_length_mm",
        "substrate_width_mm",
        "substrate_height_mm",
        "feed_length_mm",
        "feed_width_mm",
        "feed_offset_x_mm",
        "feed_offset_y_mm",
    )


@dataclass(frozen=True)
class DataSettings:
    raw_dataset_path: Path = DATA_DIR / "raw" / "dataset.csv"
    validated_dataset_path: Path = DATA_DIR / "validated" / "dataset_validated.csv"
    rejected_dataset_path: Path = DATA_DIR / "rejected" / "dataset_rejected.csv"
    min_rows_for_training: int = 200


@dataclass(frozen=True)
class Bounds:
    frequency_ghz: tuple[float, float] = (0.5, 10.0)
    bandwidth_mhz: tuple[float, float] = (5.0, 2000.0)
    patch_length_mm: tuple[float, float] = (5.0, 120.0)
    patch_width_mm: tuple[float, float] = (5.0, 120.0)
    patch_height_mm: tuple[float, float] = (0.01, 3.0)
    substrate_length_mm: tuple[float, float] = (10.0, 200.0)
    substrate_width_mm: tuple[float, float] = (10.0, 200.0)
    substrate_height_mm: tuple[float, float] = (0.1, 10.0)
    feed_length_mm: tuple[float, float] = (1.0, 100.0)
    feed_width_mm: tuple[float, float] = (0.1, 20.0)
    feed_offset_x_mm: tuple[float, float] = (-80.0, 80.0)
    feed_offset_y_mm: tuple[float, float] = (-80.0, 80.0)


API_SETTINGS = ApiSettings()
OLLAMA_SETTINGS = OllamaSettings()
ANN_SETTINGS = AnnSettings()
DATA_SETTINGS = DataSettings()
BOUNDS = Bounds()
