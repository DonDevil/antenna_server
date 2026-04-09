from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

from app.ann.features import build_ann_feature_map
from app.ann.model import InverseAnnRegressor
from app.ann.predictor import AnnPredictor
from app.antenna.recipes import generate_recipe
from app.commands.planner import build_command_package
from app.core.schemas import ClientCapabilities, DesignConstraints, OptimizeRequest, OptimizationPolicy, RangeSpec, RuntimePreferences, TargetSpec


PatchShape = Literal["rectangular", "circular"]


def _request(
    patch_shape: PatchShape = "rectangular",
    *,
    antenna_family: str = "microstrip_patch",
    design_constraints: DesignConstraints | None = None,
) -> OptimizeRequest:
    return OptimizeRequest(
        schema_version="optimize_request.v1",
        user_request=f"Design a {patch_shape} {antenna_family} antenna at 2.45 GHz",
        target_spec=TargetSpec(
            frequency_ghz=2.45,
            bandwidth_mhz=90.0,
            antenna_family=antenna_family,
            patch_shape=patch_shape,
        ),
        design_constraints=design_constraints
        or DesignConstraints(
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


def test_feature_map_uses_wban_body_and_bending_constraints() -> None:
    request = _request(
        "rectangular",
        antenna_family="wban_patch",
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=["Rogers RT/duroid 5880"],
            body_distance_mm=RangeSpec(min=4.0, max=6.0),
            bending_radius_mm=RangeSpec(min=50.0, max=70.0),
        ),
    )

    features = build_ann_feature_map(request)

    assert features["body_distance_mm"] == 5.0
    assert features["bending_radius_mm"] == 60.0


def test_generate_recipe_exposes_amc_and_wban_family_parameters() -> None:
    amc_recipe = generate_recipe(_request("rectangular", antenna_family="amc_patch"))
    wban_recipe = generate_recipe(_request("rectangular", antenna_family="wban_patch"))

    assert amc_recipe["recipe_name"] == "amc_backed_rectangular_patch"
    assert amc_recipe["family_parameters"]["amc_unit_cell_period_mm"] > 0.0
    assert wban_recipe["recipe_name"] == "wban_detuned_rectangular_patch"
    assert wban_recipe["family_parameters"]["design_frequency_ghz"] > 2.45


def test_predictor_uses_recipe_only_for_amc_and_family_model_for_wban() -> None:
    predictor = AnnPredictor()

    amc_prediction = predictor.predict(_request("rectangular", antenna_family="amc_patch"))
    wban_prediction = predictor.predict(_request("rectangular", antenna_family="wban_patch"))

    assert amc_prediction.ann_model_version == "amc_client_local_implementation"
    assert amc_prediction.optimizer_hint == "client_implement_amc"
    assert amc_prediction.family_parameters == {}
    assert wban_prediction.ann_model_version == "wban_patch_formula_bootstrap_v1"
    assert wban_prediction.family_parameters["ground_slot_length_mm"] > 0.0


def test_command_package_emits_implement_amc_and_wban_family_parameters() -> None:
    predictor = AnnPredictor()

    amc_request = _request("rectangular", antenna_family="amc_patch")
    amc_prediction = predictor.predict(amc_request)
    amc_package = build_command_package(amc_request, amc_prediction, session_id="sess-amc", trace_id="trace-amc")
    amc_commands = amc_package["commands"]

    assert any(cmd["command"] == "implement_amc" for cmd in amc_commands)
    assert not any(str(cmd["params"].get("name", "")).startswith("amc_") for cmd in amc_commands if cmd["command"] == "define_parameter")
    assert not any(str((cmd.get("params") or {}).get("name", "")).startswith("amc_cell_") for cmd in amc_commands if cmd["command"] == "define_brick")

    wban_request = _request(
        "rectangular",
        antenna_family="wban_patch",
        design_constraints=DesignConstraints(
            allowed_materials=["Copper (annealed)"],
            allowed_substrates=["Rogers RT/duroid 5880"],
            body_distance_mm=RangeSpec(min=4.0, max=6.0),
            bending_radius_mm=RangeSpec(min=50.0, max=70.0),
        ),
    )
    wban_prediction = predictor.predict(wban_request)
    wban_package = build_command_package(wban_request, wban_prediction, session_id="sess-wban", trace_id="trace-wban")

    assert amc_package["design_recipe"]["selected_materials"]["conductor"] == "Copper (annealed)"
    assert amc_package["design_recipe"]["selected_materials"]["substrate"] == "Rogers RT/duroid 5880"

    assert wban_package["design_recipe"]["family_parameters"]["ground_slot_length_mm"] > 0.0
    assert wban_package["design_recipe"]["selected_materials"]["conductor"] == "Copper (annealed)"
    assert wban_package["design_recipe"]["selected_materials"]["substrate"] == "Rogers RT/duroid 5880"
    assert any(str(cmd["params"].get("name", "")) == "body_distance_mm" for cmd in wban_package["commands"] if cmd["command"] == "define_parameter")


def test_predictor_warm_up_loads_checkpoint_with_non_default_hidden_dims(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "inverse_ann.pt"
    metadata_path = tmp_path / "metadata.json"
    model = InverseAnnRegressor(input_dim=4, output_dim=4, hidden_dims=(128, 256, 128))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": 4,
            "output_dim": 4,
        },
        checkpoint_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "test_family_model",
                "input_columns": [
                    "target_frequency_ghz",
                    "target_bandwidth_mhz",
                    "substrate_epsilon_r",
                    "substrate_height_mm",
                ],
                "output_columns": [
                    "patch_length_mm",
                    "patch_width_mm",
                    "feed_width_mm",
                    "feed_offset_y_mm",
                ],
                "x_mean": [0.0, 0.0, 0.0, 0.0],
                "x_std": [1.0, 1.0, 1.0, 1.0],
                "y_mean": [0.0, 0.0, 0.0, 0.0],
                "y_std": [1.0, 1.0, 1.0, 1.0],
                "prediction_mode": "selected_output_override",
            }
        ),
        encoding="utf-8",
    )

    predictor = AnnPredictor(checkpoint_path=checkpoint_path, metadata_path=metadata_path)

    assert predictor.warm_up() is True
    assert predictor.is_loaded() is True
    assert predictor.last_error() is None
