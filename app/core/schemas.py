from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RangeSpec(BaseModel):
    min: float
    max: float

    @field_validator("max")
    @classmethod
    def validate_max(cls, value: float, info):
        if "min" in info.data and value < float(info.data["min"]):
            raise ValueError("range max must be >= min")
        return value


class TargetSpec(BaseModel):
    frequency_ghz: float = Field(gt=0)
    bandwidth_mhz: float = Field(gt=0)
    antenna_family: str = "amc_patch"


class DesignConstraints(BaseModel):
    patch_length_mm: RangeSpec | None = None
    patch_width_mm: RangeSpec | None = None
    patch_height_mm: RangeSpec | None = None
    substrate_length_mm: RangeSpec | None = None
    substrate_width_mm: RangeSpec | None = None
    substrate_height_mm: RangeSpec | None = None
    feed_length_mm: RangeSpec | None = None
    feed_width_mm: RangeSpec | None = None
    allowed_materials: list[str] = Field(default_factory=lambda: ["Copper (annealed)"])
    allowed_substrates: list[str] = Field(default_factory=lambda: ["FR-4 (lossy)"])


class AcceptanceSpec(BaseModel):
    center_tolerance_mhz: float = 20.0
    minimum_bandwidth_mhz: float = 10.0
    maximum_vswr: float = 2.0
    minimum_gain_dbi: float = 0.0


class OptimizationPolicy(BaseModel):
    mode: Literal["single_pass", "auto_iterate"] = "auto_iterate"
    max_iterations: int = Field(default=5, ge=1, le=20)
    stop_on_first_valid: bool = True
    acceptance: AcceptanceSpec = Field(default_factory=AcceptanceSpec)


class RuntimePreferences(BaseModel):
    require_explanations: bool = False
    persist_artifacts: bool = True
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_budget_sec: int = Field(default=300, ge=30, le=3600)


class ClientCapabilities(BaseModel):
    supports_farfield_export: bool = True
    supports_current_distribution_export: bool = False
    supports_parameter_sweep: bool = False
    max_simulation_timeout_sec: int = Field(default=600, ge=10, le=7200)
    export_formats: list[Literal["json", "csv", "txt"]] = Field(default_factory=lambda: ["json"])


class OptimizeRequest(BaseModel):
    schema_version: str = "optimize_request.v1"
    session_id: str | None = None
    user_request: str = Field(min_length=3, max_length=4000)
    target_spec: TargetSpec
    design_constraints: DesignConstraints
    optimization_policy: OptimizationPolicy = Field(default_factory=OptimizationPolicy)
    runtime_preferences: RuntimePreferences = Field(default_factory=RuntimePreferences)
    client_capabilities: ClientCapabilities


class DimensionPrediction(BaseModel):
    patch_length_mm: float
    patch_width_mm: float
    patch_height_mm: float
    substrate_length_mm: float
    substrate_width_mm: float
    substrate_height_mm: float
    feed_length_mm: float
    feed_width_mm: float
    feed_offset_x_mm: float
    feed_offset_y_mm: float


class AnnPrediction(BaseModel):
    ann_model_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    dimensions: DimensionPrediction


class OptimizeResponse(BaseModel):
    schema_version: str = "optimize_response.v1"
    status: Literal["accepted", "completed", "clarification_required", "error"]
    session_id: str
    trace_id: str
    current_stage: str
    ann_prediction: AnnPrediction | None = None
    command_package: dict | None = None
    clarification: dict | None = None
    error: dict | None = None
