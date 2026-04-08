from __future__ import annotations

from app.core.refinement import refine_prediction_with_strategy
from app.core.schemas import (
    AnnPrediction,
    ClientCapabilities,
    DesignConstraints,
    DimensionPrediction,
    OptimizeRequest,
    OptimizationPolicy,
    RuntimePreferences,
    TargetSpec,
)


def _request(*, patch_shape: str = "rectangular") -> OptimizeRequest:
    return OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request=f"Design a {patch_shape} microstrip patch antenna at 2.45 GHz",
        target_spec=TargetSpec(
            frequency_ghz=2.45,
            bandwidth_mhz=90.0,
            antenna_family="microstrip_patch",
            patch_shape=patch_shape,
        ),
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=["Rogers RT/duroid 5880"],
        ),
        optimization_policy=OptimizationPolicy(
            acceptance={
                "center_tolerance_mhz": 20.0,
                "minimum_bandwidth_mhz": 90.0,
                "maximum_vswr": 2.0,
                "minimum_gain_dbi": 5.0,
                "minimum_return_loss_db": -15.0,
            }
        ),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )


def _current_prediction(*, patch_shape: str = "rectangular") -> AnnPrediction:
    return AnnPrediction(
        ann_model_version="v1",
        confidence=0.85,
        patch_shape=patch_shape,
        dimensions=DimensionPrediction(
            patch_length_mm=30.0,
            patch_width_mm=38.0,
            patch_height_mm=0.035,
            patch_radius_mm=19.0,
            substrate_length_mm=60.0,
            substrate_width_mm=70.0,
            substrate_height_mm=1.6,
            feed_length_mm=12.0,
            feed_width_mm=3.0,
            feed_offset_x_mm=0.0,
            feed_offset_y_mm=-5.0,
        ),
    )


def test_refinement_formula_applies_stronger_matching_and_bandwidth_corrections() -> None:
    refined = refine_prediction_with_strategy(
        _request(patch_shape="rectangular"),
        _current_prediction(patch_shape="rectangular"),
        {
            "freq_error_mhz": 150.0,
            "bandwidth_gap_mhz": 35.0,
            "return_loss_gap_db": 7.0,
            "vswr_gap": 1.1,
            "gain_gap": 1.8,
        },
        next_iteration_index=1,
        strategy=None,
        action_name="generic_refinement",
    )

    assert refined.dimensions.patch_length_mm > 30.9
    assert refined.dimensions.feed_width_mm > 3.15
    assert refined.dimensions.substrate_height_mm > 1.66
    assert refined.dimensions.feed_offset_y_mm < -5.1


def test_circular_refinement_updates_patch_radius_with_frequency_error() -> None:
    refined = refine_prediction_with_strategy(
        _request(patch_shape="circular"),
        _current_prediction(patch_shape="circular"),
        {
            "freq_error_mhz": 120.0,
            "bandwidth_gap_mhz": 10.0,
            "return_loss_gap_db": 4.0,
            "vswr_gap": 0.4,
            "gain_gap": 0.5,
        },
        next_iteration_index=1,
        strategy=None,
        action_name="generic_refinement",
    )

    assert refined.dimensions.patch_radius_mm is not None
    assert refined.dimensions.patch_radius_mm > 19.3
