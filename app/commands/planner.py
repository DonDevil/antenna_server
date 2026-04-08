from __future__ import annotations

from typing import Any, cast

from app.antenna.recipes import generate_recipe
from app.core.json_contracts import validate_contract
from app.core.schemas import AnnPrediction, OptimizeRequest
from app.planning.command_compiler import compile_action_plan
from config import PLANNER_SETTINGS


def build_fixed_action_plan(
    request: OptimizeRequest,
    ann: AnnPrediction,
    session_id: str,
    trace_id: str,
    iteration_index: int = 0,
) -> dict[str, Any]:
    req_any = cast(Any, request)
    ann_any = cast(Any, ann)

    target_frequency = float(req_any.target_spec.frequency_ghz)
    target_bandwidth = float(req_any.target_spec.bandwidth_mhz)
    allowed_material = str(req_any.design_constraints.allowed_materials[0])
    allowed_substrate = str(req_any.design_constraints.allowed_substrates[0])
    export_format = str(req_any.client_capabilities.export_formats[0])
    max_sim_timeout = int(req_any.client_capabilities.max_simulation_timeout_sec)
    supports_farfield = bool(req_any.client_capabilities.supports_farfield_export)

    recipe = generate_recipe(request)
    recipe_dims = cast(dict[str, Any], recipe["dimensions"])
    patch_radius = getattr(ann_any.dimensions, "patch_radius_mm", None)
    if patch_radius is None:
        patch_radius = recipe_dims.get("patch_radius_mm")

    dims = {
        "patch_length_mm": float(ann_any.dimensions.patch_length_mm),
        "patch_width_mm": float(ann_any.dimensions.patch_width_mm),
        "patch_height_mm": float(ann_any.dimensions.patch_height_mm),
        "patch_radius_mm": float(patch_radius if patch_radius is not None else (float(ann_any.dimensions.patch_width_mm) / 2.0)),
        "substrate_length_mm": float(ann_any.dimensions.substrate_length_mm),
        "substrate_width_mm": float(ann_any.dimensions.substrate_width_mm),
        "substrate_height_mm": float(ann_any.dimensions.substrate_height_mm),
        "feed_length_mm": float(ann_any.dimensions.feed_length_mm),
        "feed_width_mm": float(ann_any.dimensions.feed_width_mm),
        "feed_offset_x_mm": float(ann_any.dimensions.feed_offset_x_mm),
        "feed_offset_y_mm": float(ann_any.dimensions.feed_offset_y_mm),
    }
    patch_shape = str(recipe.get("patch_shape", getattr(req_any.target_spec, "patch_shape", "rectangular")))
    component_name = "antenna"
    substrate_x_half = dims["substrate_width_mm"] / 2.0
    substrate_y_half = dims["substrate_length_mm"] / 2.0
    patch_x_half = dims["patch_width_mm"] / 2.0
    patch_y_half = dims["patch_length_mm"] / 2.0
    feed_half_width = dims["feed_width_mm"] / 2.0
    feed_y_min = min(dims["feed_offset_y_mm"] - dims["feed_length_mm"], dims["feed_offset_y_mm"])
    feed_y_max = max(dims["feed_offset_y_mm"] - dims["feed_length_mm"], dims["feed_offset_y_mm"])
    patch_z_min = dims["substrate_height_mm"]
    patch_z_max = dims["substrate_height_mm"] + dims["patch_height_mm"]

    if patch_shape == "circular":
        patch_action = {
            "seq": 9,
            "action": "define_cylinder",
            "command": "define_cylinder",
            "params": {
                "name": "patch",
                "component": component_name,
                "material": allowed_material,
                "axis": "z",
                "center": [0.0, 0.0],
                "outer_radius": dims["patch_radius_mm"],
                "inner_radius": 0.0,
                "zrange": [patch_z_min, patch_z_max],
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry", "patch_shape:circular"],
            "expected_effects": ["patch_created"],
        }
    else:
        patch_action = {
            "seq": 9,
            "action": "define_brick",
            "command": "define_brick",
            "params": {
                "name": "patch",
                "component": component_name,
                "material": allowed_material,
                "xrange": [-patch_x_half, patch_x_half],
                "yrange": [-patch_y_half, patch_y_half],
                "zrange": [patch_z_min, patch_z_max],
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry", "patch_shape:rectangular"],
            "expected_effects": ["patch_created"],
        }

    actions: list[dict[str, Any]] = [
        {
            "seq": 1,
            "action": "create_project",
            "command": "create_project",
            "params": {"project_name": f"design_{session_id}"},
            "on_failure": "abort",
            "checksum_scope": "all",
            "rationale_tags": ["baseline_setup"],
            "expected_effects": ["project_initialized"],
        },
        {
            "seq": 2,
            "action": "set_units",
            "command": "set_units",
            "params": {"geometry": "mm", "frequency": "ghz"},
            "on_failure": "abort",
            "checksum_scope": "all",
            "rationale_tags": ["baseline_setup"],
            "expected_effects": ["units_normalized"],
        },
        {
            "seq": 3,
            "action": "set_frequency_range",
            "command": "set_frequency_range",
            "params": {
                "start_ghz": max(0.1, target_frequency - 0.5),
                "stop_ghz": target_frequency + 0.5,
            },
            "on_failure": "abort",
            "checksum_scope": "all",
            "rationale_tags": ["target_driven"],
            "expected_effects": ["solver_frequency_window_set"],
        },
        {
            "seq": 4,
            "action": "define_material",
            "command": "define_material",
            "params": {"name": allowed_material, "kind": "conductor", "conductivity_s_per_m": 5.8e7},
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["family_constraints"],
            "expected_effects": ["conductor_material_defined"],
        },
        {
            "seq": 5,
            "action": "define_material",
            "command": "define_material",
            "params": {"name": allowed_substrate, "kind": "substrate", "epsilon_r": 4.4, "loss_tangent": 0.02},
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["family_constraints"],
            "expected_effects": ["substrate_material_defined"],
        },
        {
            "seq": 6,
            "action": "create_component",
            "command": "create_component",
            "params": {"component": component_name},
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["baseline_setup", "component_scope"],
            "expected_effects": ["component_initialized"],
        },
        {
            "seq": 7,
            "action": "define_brick",
            "command": "define_brick",
            "params": {
                "name": "substrate",
                "component": component_name,
                "material": allowed_substrate,
                "xrange": [-substrate_x_half, substrate_x_half],
                "yrange": [-substrate_y_half, substrate_y_half],
                "zrange": [0.0, dims["substrate_height_mm"]],
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry", "substrate_volume"],
            "expected_effects": ["substrate_created"],
        },
        {
            "seq": 8,
            "action": "define_brick",
            "command": "define_brick",
            "params": {
                "name": "ground",
                "component": component_name,
                "material": allowed_material,
                "xrange": [-substrate_x_half, substrate_x_half],
                "yrange": [-substrate_y_half, substrate_y_half],
                "zrange": [-dims["patch_height_mm"], 0.0],
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry", "ground_reference"],
            "expected_effects": ["ground_created"],
        },
        patch_action,
        {
            "seq": 10,
            "action": "define_brick",
            "command": "define_brick",
            "params": {
                "name": "feed",
                "component": component_name,
                "material": allowed_material,
                "xrange": [dims["feed_offset_x_mm"] - feed_half_width, dims["feed_offset_x_mm"] + feed_half_width],
                "yrange": [feed_y_min, feed_y_max],
                "zrange": [patch_z_min, patch_z_max],
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry", "feed_matching"],
            "expected_effects": ["feedline_created"],
        },
        {
            "seq": 11,
            "action": "create_port",
            "command": "create_port",
            "params": {"port_id": 1, "port_type": "discrete", "impedance_ohm": 50.0, "reference_mm": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_excitation"],
            "expected_effects": ["port_defined"],
        },
        {
            "seq": 12,
            "action": "set_boundary",
            "command": "set_boundary",
            "params": {"boundary_type": "open_add_space", "padding_mm": 15.0},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["boundary_set"],
        },
        {
            "seq": 13,
            "action": "set_solver",
            "command": "set_solver",
            "params": {"solver_type": "time_domain", "mesh_cells_per_wavelength": 20},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["solver_ready"],
        },
        {
            "seq": 15,
            "action": "run_simulation",
            "command": "run_simulation",
            "params": {"timeout_sec": min(max_sim_timeout, 900)},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["simulation_started"],
        },
        {
            "seq": 16,
            "action": "export_s_parameters",
            "command": "export_s_parameters",
            "params": {"format": export_format, "destination_hint": "s11"},
            "on_failure": "continue",
            "checksum_scope": "exports",
            "rationale_tags": ["baseline_exports"],
            "expected_effects": ["s_parameters_available"],
        },
        {
            "seq": 17,
            "action": "extract_summary_metrics",
            "command": "extract_summary_metrics",
            "params": {"metrics": ["center_frequency_ghz", "bandwidth_mhz", "return_loss_db", "vswr", "gain_dbi"]},
            "on_failure": "continue",
            "checksum_scope": "exports",
            "rationale_tags": ["baseline_exports"],
            "expected_effects": ["summary_metrics_available"],
        },
    ]

    if supports_farfield:
        actions.insert(
            13,
            {
                "seq": 14,
                "action": "add_farfield_monitor",
                "command": "add_farfield_monitor",
                "params": {
                    "monitor_name": f"farfield_{target_frequency:.3f}ghz".replace(".", "p"),
                    "frequency_ghz": target_frequency,
                },
                "on_failure": "abort",
                "checksum_scope": "simulation",
                "rationale_tags": ["client_capability_export", "simulation_preconditions"],
                "expected_effects": ["farfield_monitor_configured"],
            },
        )
        actions.append(
            {
                "seq": 18,
                "action": "export_farfield",
                "command": "export_farfield",
                "params": {
                    "format": export_format,
                    "frequency_ghz": target_frequency,
                    "destination_hint": "farfield",
                },
                "on_failure": "continue",
                "checksum_scope": "exports",
                "rationale_tags": ["client_capability_export"],
                "expected_effects": ["farfield_available"],
            }
        )

    action_plan = {
        "schema_version": "action_plan.v1",
        "plan_version": PLANNER_SETTINGS.action_plan_version,
        "planner_mode": "fixed",
        "command_catalog_version": PLANNER_SETTINGS.command_catalog_version,
        "session_id": session_id,
        "trace_id": trace_id,
        "design_id": f"design_{session_id}",
        "iteration_index": int(iteration_index),
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": dims,
        "predicted_metrics": {
            "center_frequency_ghz": target_frequency,
            "bandwidth_mhz": target_bandwidth,
        },
        "design_recipe": {
            "family": str(req_any.target_spec.antenna_family),
            "patch_shape": patch_shape,
            "feed_type": str(getattr(req_any.target_spec, "feed_type", "auto")),
            "recipe_name": str(recipe.get("recipe_name", "unknown_recipe")),
            "substrate": recipe.get("substrate"),
            "notes": recipe.get("notes", []),
        },
        "actions": actions,
        "expected_exports": ["s_parameters", "summary_metrics"] + (["farfield"] if supports_farfield else []),
        "safety_checks": [
            "dimensions_within_constraints",
            "command_order_validated",
            "materials_whitelisted",
            "client_capability_compatible",
            "simulation_timeout_bounded",
        ],
    }
    validate_contract("action_plan", action_plan)
    return action_plan


def build_command_package(
    request: OptimizeRequest,
    ann: AnnPrediction,
    session_id: str,
    trace_id: str,
    iteration_index: int = 0,
) -> dict[str, Any]:
    if PLANNER_SETTINGS.mode != "fixed" and not PLANNER_SETTINGS.dynamic_enabled:
        raise ValueError("dynamic planner mode is disabled by configuration")

    action_plan = build_fixed_action_plan(
        request,
        ann,
        session_id=session_id,
        trace_id=trace_id,
        iteration_index=iteration_index,
    )
    return compile_action_plan(action_plan)
