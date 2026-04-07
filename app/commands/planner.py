from __future__ import annotations

from typing import Any, cast

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

    dims = {
        "patch_length_mm": float(ann_any.dimensions.patch_length_mm),
        "patch_width_mm": float(ann_any.dimensions.patch_width_mm),
        "patch_height_mm": float(ann_any.dimensions.patch_height_mm),
        "substrate_length_mm": float(ann_any.dimensions.substrate_length_mm),
        "substrate_width_mm": float(ann_any.dimensions.substrate_width_mm),
        "substrate_height_mm": float(ann_any.dimensions.substrate_height_mm),
        "feed_length_mm": float(ann_any.dimensions.feed_length_mm),
        "feed_width_mm": float(ann_any.dimensions.feed_width_mm),
        "feed_offset_x_mm": float(ann_any.dimensions.feed_offset_x_mm),
        "feed_offset_y_mm": float(ann_any.dimensions.feed_offset_y_mm),
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
            "action": "create_substrate",
            "command": "create_substrate",
            "params": {
                "name": "substrate",
                "material": allowed_substrate,
                "length_mm": dims["substrate_length_mm"],
                "width_mm": dims["substrate_width_mm"],
                "height_mm": dims["substrate_height_mm"],
                "origin_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry"],
            "expected_effects": ["substrate_created"],
        },
        {
            "seq": 7,
            "action": "create_ground_plane",
            "command": "create_ground_plane",
            "params": {
                "name": "ground",
                "material": allowed_material,
                "length_mm": dims["substrate_length_mm"],
                "width_mm": dims["substrate_width_mm"],
                "thickness_mm": dims["patch_height_mm"],
                "z_mm": 0.0,
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry"],
            "expected_effects": ["ground_created"],
        },
        {
            "seq": 8,
            "action": "create_patch",
            "command": "create_patch",
            "params": {
                "name": "patch",
                "material": allowed_material,
                "length_mm": dims["patch_length_mm"],
                "width_mm": dims["patch_width_mm"],
                "thickness_mm": dims["patch_height_mm"],
                "center_mm": {"x": 0.0, "y": 0.0, "z": dims["substrate_height_mm"]},
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry"],
            "expected_effects": ["patch_created"],
        },
        {
            "seq": 9,
            "action": "create_feedline",
            "command": "create_feedline",
            "params": {
                "name": "feed",
                "material": allowed_material,
                "length_mm": dims["feed_length_mm"],
                "width_mm": dims["feed_width_mm"],
                "thickness_mm": dims["patch_height_mm"],
                "start_mm": {"x": dims["feed_offset_x_mm"], "y": dims["feed_offset_y_mm"], "z": dims["substrate_height_mm"]},
                "end_mm": {
                    "x": dims["feed_offset_x_mm"],
                    "y": dims["feed_offset_y_mm"] - dims["feed_length_mm"],
                    "z": dims["substrate_height_mm"],
                },
            },
            "on_failure": "abort",
            "checksum_scope": "geometry",
            "rationale_tags": ["ann_baseline_geometry"],
            "expected_effects": ["feedline_created"],
        },
        {
            "seq": 10,
            "action": "create_port",
            "command": "create_port",
            "params": {"port_id": 1, "port_type": "discrete", "impedance_ohm": 50.0, "reference_mm": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_excitation"],
            "expected_effects": ["port_defined"],
        },
        {
            "seq": 11,
            "action": "set_boundary",
            "command": "set_boundary",
            "params": {"boundary_type": "open_add_space", "padding_mm": 15.0},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["boundary_set"],
        },
        {
            "seq": 12,
            "action": "set_solver",
            "command": "set_solver",
            "params": {"solver_type": "time_domain", "mesh_cells_per_wavelength": 20},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["solver_ready"],
        },
        {
            "seq": 14,
            "action": "run_simulation",
            "command": "run_simulation",
            "params": {"timeout_sec": min(max_sim_timeout, 900)},
            "on_failure": "abort",
            "checksum_scope": "simulation",
            "rationale_tags": ["baseline_simulation"],
            "expected_effects": ["simulation_started"],
        },
        {
            "seq": 15,
            "action": "export_s_parameters",
            "command": "export_s_parameters",
            "params": {"format": export_format, "destination_hint": "s11"},
            "on_failure": "continue",
            "checksum_scope": "exports",
            "rationale_tags": ["baseline_exports"],
            "expected_effects": ["s_parameters_available"],
        },
        {
            "seq": 16,
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
            12,
            {
                "seq": 13,
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
                "seq": 17,
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
