from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    fast_model_name: str = "deepseek-r1:1.5b"
    big_model_name: str = "qwen3.5:4b"
    model_name: str = "qwen3.5:4b"
    timeout_sec: int = 90


@dataclass(frozen=True)
class PlannerSettings:
    mode: Literal["fixed", "dynamic"] = "dynamic"
    dynamic_enabled: bool = True
    command_catalog_version: str = "v2"
    action_plan_version: str = "v1"
    llm_enabled_for_intent: bool = True
    llm_enabled_for_refinement: bool = True
    llm_refinement_confidence_threshold: float = 0.62
    llm_max_calls_per_session: int = 2
    llm_max_calls_per_iteration: int = 1
    session_context_max_history: int = 3


@dataclass(frozen=True)
class AnnSettings:
    model_version: str = "v1"
    checkpoint_path: Path = MODELS_DIR / "ann" / "v1" / "inverse_ann.pt"
    metadata_path: Path = MODELS_DIR / "ann" / "v1" / "metadata.json"
    input_columns: tuple[str, ...] = (
        "frequency_ghz",
        "bandwidth_mhz",
        "substrate_epsilon_r",
        "substrate_height_mm",
        "minimum_gain_dbi",
        "maximum_vswr",
        "priority_s11_minimize",
        "priority_bandwidth_maximize",
        "priority_gain_maximize",
        "priority_efficiency_maximize",
        "family_is_amc_patch",
        "family_is_microstrip_patch",
        "family_is_wban_patch",
        "shape_is_rectangular",
        "shape_is_circular",
    )
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
PLANNER_SETTINGS = PlannerSettings()
ANN_SETTINGS = AnnSettings()
DATA_SETTINGS = DataSettings()
BOUNDS = Bounds()
