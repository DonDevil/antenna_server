from __future__ import annotations

from typing import Any, cast

from app.core.schemas import AnnPrediction, OptimizeRequest


def build_command_package(
    request: OptimizeRequest,
    ann: AnnPrediction,
    session_id: str,
    trace_id: str,
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

    commands: list[dict[str, Any]] = [
        {"seq": 1, "command": "create_project", "params": {"project_name": f"design_{session_id}"}, "on_failure": "abort", "checksum_scope": "all"},
        {"seq": 2, "command": "set_units", "params": {"geometry": "mm", "frequency": "ghz"}, "on_failure": "abort", "checksum_scope": "all"},
        {
            "seq": 3,
            "command": "set_frequency_range",
            "params": {
                "start_ghz": max(0.1, target_frequency - 0.5),
                "stop_ghz": target_frequency + 0.5,
            },
            "on_failure": "abort",
            "checksum_scope": "all",
        },
        {
            "seq": 4,
            "command": "define_material",
            "params": {"name": allowed_material, "kind": "conductor", "conductivity_s_per_m": 5.8e7},
            "on_failure": "abort",
            "checksum_scope": "geometry",
        },
        {
            "seq": 5,
            "command": "define_material",
            "params": {"name": allowed_substrate, "kind": "substrate", "epsilon_r": 4.4, "loss_tangent": 0.02},
            "on_failure": "abort",
            "checksum_scope": "geometry",
        },
        {
            "seq": 6,
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
        },
        {
            "seq": 7,
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
        },
        {
            "seq": 8,
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
        },
        {
            "seq": 9,
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
        },
        {
            "seq": 10,
            "command": "create_port",
            "params": {"port_id": 1, "port_type": "discrete", "impedance_ohm": 50.0, "reference_mm": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "on_failure": "abort",
            "checksum_scope": "simulation",
        },
        {"seq": 11, "command": "set_boundary", "params": {"boundary_type": "open_add_space", "padding_mm": 15.0}, "on_failure": "abort", "checksum_scope": "simulation"},
        {"seq": 12, "command": "set_solver", "params": {"solver_type": "time_domain", "mesh_cells_per_wavelength": 20}, "on_failure": "abort", "checksum_scope": "simulation"},
        {"seq": 13, "command": "run_simulation", "params": {"timeout_sec": min(max_sim_timeout, 900)}, "on_failure": "abort", "checksum_scope": "simulation"},
        {"seq": 14, "command": "export_s_parameters", "params": {"format": export_format, "destination_hint": "s11"}, "on_failure": "continue", "checksum_scope": "exports"},
        {"seq": 15, "command": "extract_summary_metrics", "params": {"metrics": ["center_frequency_ghz", "bandwidth_mhz", "return_loss_db", "vswr", "gain_dbi"]}, "on_failure": "continue", "checksum_scope": "exports"},
    ]

    if supports_farfield:
        commands.append(
            {
                "seq": 16,
                "command": "export_farfield",
                "params": {
                    "format": export_format,
                    "frequency_ghz": target_frequency,
                    "destination_hint": "farfield",
                },
                "on_failure": "continue",
                "checksum_scope": "exports",
            }
        )

    return {
        "schema_version": "cst_command_package.v1",
        "command_catalog_version": "v1",
        "session_id": session_id,
        "trace_id": trace_id,
        "design_id": f"design_{session_id}",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": dims,
        "predicted_metrics": {
            "center_frequency_ghz": target_frequency,
            "bandwidth_mhz": target_bandwidth,
        },
        "commands": commands,
        "expected_exports": ["s_parameters", "summary_metrics"]
        + (["farfield"] if supports_farfield else []),
        "safety_checks": [
            "dimensions_within_constraints",
            "command_order_validated",
            "materials_whitelisted",
            "client_capability_compatible",
            "simulation_timeout_bounded",
        ],
    }
