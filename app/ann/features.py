from __future__ import annotations

from typing import Any

from app.antenna.materials import get_substrate_properties
from app.antenna.recipes import generate_recipe, resolve_patch_shape
from app.core.schemas import OptimizeRequest


_SUPPORTED_FAMILIES = ("amc_patch", "microstrip_patch", "wban_patch")


def build_ann_feature_map(request: OptimizeRequest, recipe: dict[str, Any] | None = None) -> dict[str, float]:
    recipe_payload = recipe or generate_recipe(request)
    substrate_name = request.design_constraints.allowed_substrates[0] if request.design_constraints.allowed_substrates else None
    substrate = get_substrate_properties(substrate_name)
    patch_shape = resolve_patch_shape(request)
    primary = request.optimization_targets.primary
    acceptance = request.optimization_policy.acceptance

    feature_map: dict[str, float] = {
        "frequency_ghz": float(request.target_spec.frequency_ghz),
        "bandwidth_mhz": float(request.target_spec.bandwidth_mhz),
        "substrate_epsilon_r": float(substrate["epsilon_r"]),
        "substrate_height_mm": float(recipe_payload.get("dimensions", {}).get("substrate_height_mm", 1.6)),
        "minimum_gain_dbi": float(acceptance.minimum_gain_dbi),
        "maximum_vswr": float(acceptance.maximum_vswr),
        "priority_s11_minimize": 1.0 if primary.s11 == "minimize" else 0.0,
        "priority_bandwidth_maximize": 1.0 if primary.bandwidth == "maximize" else 0.0,
        "priority_gain_maximize": 1.0 if primary.gain == "maximize" else 0.0,
        "priority_efficiency_maximize": 1.0 if primary.efficiency == "maximize" else 0.0,
        "shape_is_rectangular": 1.0 if patch_shape == "rectangular" else 0.0,
        "shape_is_circular": 1.0 if patch_shape == "circular" else 0.0,
    }

    family_name = str(request.target_spec.antenna_family).strip().lower()
    for family in _SUPPORTED_FAMILIES:
        feature_map[f"family_is_{family}"] = 1.0 if family_name == family else 0.0

    return feature_map
