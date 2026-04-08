from __future__ import annotations

from typing import Literal

from app.commands.planner import build_command_package
from app.core.schemas import (
    AnnPrediction,
    DesignConstraints,
    DimensionPrediction,
    OptimizeRequest,
    OptimizationPolicy,
    RuntimePreferences,
    ClientCapabilities,
    TargetSpec,
)
from app.antenna.recipes import generate_recipe


PatchShape = Literal["rectangular", "circular"]


def _request(*, frequency_ghz: float = 2.45, bandwidth_mhz: float = 80.0, patch_shape: PatchShape = "rectangular") -> OptimizeRequest:
    return OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request=f"Design a {patch_shape} microstrip patch antenna at {frequency_ghz} GHz",
        target_spec=TargetSpec(
            frequency_ghz=frequency_ghz,
            bandwidth_mhz=bandwidth_mhz,
            antenna_family="microstrip_patch",
            patch_shape=patch_shape,
        ),
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=["Rogers RT/duroid 5880"],
        ),
        optimization_policy=OptimizationPolicy(),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )


def _ann() -> AnnPrediction:
    return AnnPrediction(
        ann_model_version="test",
        confidence=0.8,
        dimensions=DimensionPrediction(
            patch_length_mm=30.0,
            patch_width_mm=38.0,
            patch_height_mm=0.035,
            substrate_length_mm=60.0,
            substrate_width_mm=70.0,
            substrate_height_mm=1.6,
            feed_length_mm=12.0,
            feed_width_mm=3.0,
            feed_offset_x_mm=0.0,
            feed_offset_y_mm=-5.0,
        ),
    )


def test_rectangular_recipe_dimensions_reduce_with_higher_frequency() -> None:
    low = generate_recipe(_request(frequency_ghz=2.0, patch_shape="rectangular"))
    high = generate_recipe(_request(frequency_ghz=5.0, patch_shape="rectangular"))

    assert low["recipe_name"] == "rectangular_microstrip_patch"
    assert high["recipe_name"] == "rectangular_microstrip_patch"
    assert low["dimensions"]["patch_length_mm"] > high["dimensions"]["patch_length_mm"]
    assert low["dimensions"]["patch_width_mm"] > high["dimensions"]["patch_width_mm"]


def test_circular_recipe_emits_patch_radius_and_metadata() -> None:
    recipe = generate_recipe(_request(patch_shape="circular"))

    assert recipe["patch_shape"] == "circular"
    assert recipe["recipe_name"] == "circular_microstrip_patch"
    assert recipe["dimensions"]["patch_radius_mm"] > 0.0


def test_command_package_carries_design_recipe_for_circular_patch() -> None:
    request = _request(patch_shape="circular")
    command_package = build_command_package(
        request,
        _ann(),
        session_id="session-recipe",
        trace_id="trace-recipe",
        iteration_index=0,
    )

    assert command_package["design_recipe"]["patch_shape"] == "circular"
    patch_command = next(item for item in command_package["commands"] if item["command"] == "define_cylinder")
    assert patch_command["params"]["component"] == "antenna"
    assert patch_command["params"]["outer_radius"] > 0.0
