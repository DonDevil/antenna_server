from __future__ import annotations

from pathlib import Path
from typing import Literal

from app.ann.features import build_ann_feature_map
from app.ann.predictor import AnnPredictor
from app.antenna.recipes import generate_recipe
from app.core.schemas import ClientCapabilities, DesignConstraints, OptimizeRequest, OptimizationPolicy, RuntimePreferences, TargetSpec


PatchShape = Literal["rectangular", "circular"]


def _request(patch_shape: PatchShape = "rectangular") -> OptimizeRequest:
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
        optimization_policy=OptimizationPolicy(),
        runtime_preferences=RuntimePreferences(),
        client_capabilities=ClientCapabilities(),
    )


def test_ann_feature_map_includes_family_shape_material_and_objective_features() -> None:
    features = build_ann_feature_map(_request("circular"))

    assert features["frequency_ghz"] == 2.45
    assert features["family_is_microstrip_patch"] == 1.0
    assert features["shape_is_circular"] == 1.0
    assert features["shape_is_rectangular"] == 0.0
    assert features["substrate_epsilon_r"] == 2.2
    assert features["priority_gain_maximize"] == 1.0


def test_predictor_uses_recipe_baseline_when_model_artifacts_are_missing(tmp_path: Path) -> None:
    predictor = AnnPredictor(
        checkpoint_path=tmp_path / "missing.pt",
        metadata_path=tmp_path / "missing.json",
    )

    ann = predictor.predict(_request("circular"))

    assert ann.recipe_name == "circular_microstrip_patch"
    assert ann.patch_shape == "circular"
    assert ann.optimizer_hint == "recipe_only"
    assert ann.dimensions.patch_radius_mm is not None
    assert ann.dimensions.patch_radius_mm > 0.0


def test_predictor_selected_output_override_only_changes_modeled_dimensions() -> None:
    recipe = generate_recipe(_request("rectangular"))
    combined = AnnPredictor.merge_recipe_and_model_outputs(
        recipe,
        {
            "patch_length_mm": 25.0,
            "patch_width_mm": 34.0,
            "feed_width_mm": 1.8,
            "feed_offset_y_mm": -4.2,
        },
        "selected_output_override",
    )

    assert combined["patch_length_mm"] == 25.0
    assert combined["patch_width_mm"] == 34.0
    assert combined["feed_width_mm"] == 1.8
    assert combined["feed_offset_y_mm"] == -4.2
    assert combined["substrate_length_mm"] == recipe["dimensions"]["substrate_length_mm"]
    assert combined["feed_length_mm"] == recipe["dimensions"]["feed_length_mm"]
    assert combined["patch_radius_mm"] == 17.0
