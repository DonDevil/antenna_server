from __future__ import annotations

from typing import Any, cast

from app.antenna.recipes import generate_recipe
from app.core.json_contracts import validate_contract
from app.core.schemas import AnnPrediction, OptimizeRequest
from app.planning.command_compiler import compile_action_plan
from config import PLANNER_SETTINGS


_PARAMETER_NAME_MAP: tuple[tuple[str, str], ...] = (
    ("px", "patch_width_mm"),
    ("py", "patch_length_mm"),
    ("t_cu", "patch_height_mm"),
    ("pr", "patch_radius_mm"),
    ("sx", "substrate_width_mm"),
    ("sy", "substrate_length_mm"),
    ("h_sub", "substrate_height_mm"),
    ("feed_len", "feed_length_mm"),
    ("feed_w", "feed_width_mm"),
    ("feed_x", "feed_offset_x_mm"),
    ("feed_y", "feed_offset_y_mm"),
)


def _build_dimensions(request: OptimizeRequest, ann: AnnPrediction) -> tuple[dict[str, float], dict[str, Any], str]:
    req_any = cast(Any, request)
    ann_any = cast(Any, ann)

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
    return dims, recipe, patch_shape


def _build_parameter_values(dims: dict[str, float]) -> dict[str, float]:
    return {param_name: float(dims[dim_name]) for param_name, dim_name in _PARAMETER_NAME_MAP}


def _build_family_parameter_values(
    *,
    family: str,
    recipe: dict[str, Any],
    ann: AnnPrediction,
    dims: dict[str, float],
) -> dict[str, float]:
    if family == "amc_patch":
        return {}

    merged: dict[str, Any] = {}
    if isinstance(recipe.get("family_parameters"), dict):
        merged.update(cast(dict[str, Any], recipe["family_parameters"]))
    ann_family_parameters = getattr(ann, "family_parameters", {}) or {}
    if isinstance(ann_family_parameters, dict):
        merged.update(dict(ann_family_parameters))

    family_parameters: dict[str, float] = {}
    for key, value in merged.items():
        if isinstance(value, bool):
            family_parameters[str(key)] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            family_parameters[str(key)] = float(value)

    if family == "amc_patch":
        period = float(family_parameters.get("amc_unit_cell_period_mm", max(6.0, dims["patch_width_mm"] * 0.48)))
        patch_size = float(family_parameters.get("amc_patch_size_mm", max(4.0, period * 0.82)))
        air_gap = float(family_parameters.get("amc_air_gap_mm", max(0.5, period * 0.12)))
        rows = max(1, int(round(family_parameters.get("amc_array_rows", 5.0))))
        cols = max(1, int(round(family_parameters.get("amc_array_cols", 5.0))))
        family_parameters.update(
            {
                "amc_period": period,
                "amc_cell": patch_size,
                "amc_gap": float(family_parameters.get("amc_gap_mm", max(0.2, period - patch_size))),
                "amc_air_gap": air_gap,
                "amc_sub_h": float(dims["substrate_height_mm"]),
                "amc_nx": float(cols),
                "amc_ny": float(rows),
                "amc_size_x": float(cols * period),
                "amc_size_y": float(rows * period),
            }
        )
    return family_parameters


def _amc_cell_center_expr(index: int, count: int, period_symbol: str = "amc_period") -> str:
    offset = index - ((count - 1) / 2.0)
    if abs(offset) < 1e-9:
        return "0"
    return f"({offset:g}*{period_symbol})"


def build_fixed_action_plan(
    request: OptimizeRequest,
    ann: AnnPrediction,
    session_id: str,
    trace_id: str,
    iteration_index: int = 0,
    previous_ann: AnnPrediction | None = None,
) -> dict[str, Any]:
    req_any = cast(Any, request)

    target_frequency = float(req_any.target_spec.frequency_ghz)
    target_bandwidth = float(req_any.target_spec.bandwidth_mhz)
    allowed_material = str(req_any.design_constraints.allowed_materials[0])
    allowed_substrate = str(req_any.design_constraints.allowed_substrates[0])
    export_format = str(req_any.client_capabilities.export_formats[0])
    max_sim_timeout = int(req_any.client_capabilities.max_simulation_timeout_sec)
    supports_farfield = bool(req_any.client_capabilities.supports_farfield_export)

    dims, recipe, patch_shape = _build_dimensions(request, ann)
    family_name = str(req_any.target_spec.antenna_family).strip().lower()
    parameter_values = _build_parameter_values(dims)
    family_parameter_values = _build_family_parameter_values(
        family=family_name,
        recipe=recipe,
        ann=ann,
        dims=dims,
    )
    previous_parameter_values: dict[str, float] | None = None
    previous_family_parameter_values: dict[str, float] | None = None
    if previous_ann is not None:
        previous_dims, previous_recipe, _ = _build_dimensions(request, previous_ann)
        previous_parameter_values = _build_parameter_values(previous_dims)
        previous_family_parameter_values = _build_family_parameter_values(
            family=family_name,
            recipe=previous_recipe,
            ann=previous_ann,
            dims=previous_dims,
        )

    component_name = "antenna"
    actions: list[dict[str, Any]] = []
    seq = 1

    def add_action(
        action: str,
        command: str,
        params: dict[str, Any],
        *,
        on_failure: str = "abort",
        checksum_scope: str = "command",
        rationale_tags: list[str] | None = None,
        expected_effects: list[str] | None = None,
    ) -> None:
        nonlocal seq
        actions.append(
            {
                "seq": seq,
                "action": action,
                "command": command,
                "params": params,
                "on_failure": on_failure,
                "checksum_scope": checksum_scope,
                "rationale_tags": rationale_tags or [],
                "expected_effects": expected_effects or [],
            }
        )
        seq += 1

    if int(iteration_index) == 0:
        add_action(
            "create_project",
            "create_project",
            {"project_name": f"design_{session_id}"},
            checksum_scope="all",
            rationale_tags=["baseline_setup"],
            expected_effects=["project_initialized"],
        )
        add_action(
            "set_units",
            "set_units",
            {"geometry": "mm", "frequency": "ghz"},
            checksum_scope="all",
            rationale_tags=["baseline_setup"],
            expected_effects=["units_normalized"],
        )
        add_action(
            "set_frequency_range",
            "set_frequency_range",
            {
                "start_ghz": max(0.1, target_frequency - 0.5),
                "stop_ghz": target_frequency + 0.5,
            },
            checksum_scope="all",
            rationale_tags=["target_driven"],
            expected_effects=["solver_frequency_window_set"],
        )
        add_action(
            "define_material",
            "define_material",
            {"name": allowed_material, "kind": "conductor", "conductivity_s_per_m": 5.8e7},
            checksum_scope="geometry",
            rationale_tags=["family_constraints"],
            expected_effects=["conductor_material_defined"],
        )
        add_action(
            "define_material",
            "define_material",
            {"name": allowed_substrate, "kind": "substrate", "epsilon_r": 4.4, "loss_tangent": 0.02},
            checksum_scope="geometry",
            rationale_tags=["family_constraints"],
            expected_effects=["substrate_material_defined"],
        )
        add_action(
            "create_component",
            "create_component",
            {"component": component_name},
            checksum_scope="geometry",
            rationale_tags=["baseline_setup", "component_scope"],
            expected_effects=["component_initialized"],
        )

        for param_name, param_value in parameter_values.items():
            add_action(
                "define_parameter",
                "define_parameter",
                {"name": param_name, "value": param_value},
                checksum_scope="geometry",
                rationale_tags=["parametric_geometry", f"parameter:{param_name}"],
                expected_effects=[f"parameter_{param_name}_defined"],
            )

        for param_name, param_value in family_parameter_values.items():
            add_action(
                "define_parameter",
                "define_parameter",
                {"name": param_name, "value": param_value},
                checksum_scope="geometry",
                rationale_tags=["family_geometry", f"family_parameter:{param_name}"],
                expected_effects=[f"family_parameter_{param_name}_defined"],
            )

        add_action(
            "define_brick",
            "define_brick",
            {
                "name": "substrate",
                "component": component_name,
                "material": allowed_substrate,
                "xrange": ["-sx/2", "sx/2"],
                "yrange": ["-sy/2", "sy/2"],
                "zrange": [0.0, "h_sub"],
            },
            checksum_scope="geometry",
            rationale_tags=["parametric_geometry", "substrate_volume"],
            expected_effects=["substrate_created"],
        )
        add_action(
            "define_brick",
            "define_brick",
            {
                "name": "ground",
                "component": component_name,
                "material": allowed_material,
                "xrange": ["-sx/2", "sx/2"],
                "yrange": ["-sy/2", "sy/2"],
                "zrange": ["-t_cu", 0.0],
            },
            checksum_scope="geometry",
            rationale_tags=["parametric_geometry", "ground_reference"],
            expected_effects=["ground_created"],
        )

        if patch_shape == "circular":
            add_action(
                "define_cylinder",
                "define_cylinder",
                {
                    "name": "patch",
                    "component": component_name,
                    "material": allowed_material,
                    "axis": "z",
                    "center": [0.0, 0.0],
                    "outer_radius": "pr",
                    "inner_radius": 0.0,
                    "zrange": ["h_sub", "h_sub+t_cu"],
                },
                checksum_scope="geometry",
                rationale_tags=["parametric_geometry", "patch_shape:circular"],
                expected_effects=["patch_created"],
            )
        else:
            add_action(
                "define_brick",
                "define_brick",
                {
                    "name": "patch",
                    "component": component_name,
                    "material": allowed_material,
                    "xrange": ["-px/2", "px/2"],
                    "yrange": ["-py/2", "py/2"],
                    "zrange": ["h_sub", "h_sub+t_cu"],
                },
                checksum_scope="geometry",
                rationale_tags=["parametric_geometry", "patch_shape:rectangular"],
                expected_effects=["patch_created"],
            )

        add_action(
            "define_brick",
            "define_brick",
            {
                "name": "feed",
                "component": component_name,
                "material": allowed_material,
                "xrange": ["feed_x-(feed_w/2)", "feed_x+(feed_w/2)"],
                "yrange": ["feed_y-feed_len", "feed_y"],
                "zrange": ["h_sub", "h_sub+t_cu"],
            },
            checksum_scope="geometry",
            rationale_tags=["parametric_geometry", "feed_matching"],
            expected_effects=["feedline_created"],
        )

        if family_name == "amc_patch":
            add_action(
                "implement_amc",
                "implement_amc",
                {
                    "component": "amc",
                    "strategy": "client_heuristic",
                    "relative_to": "patch",
                },
                checksum_scope="geometry",
                rationale_tags=["family_geometry", "family:amc_patch", "client_local_amc"],
                expected_effects=["amc_geometry_created_by_client"],
            )

        add_action(
            "rebuild_model",
            "rebuild_model",
            {},
            checksum_scope="simulation",
            rationale_tags=["parametric_control"],
            expected_effects=["model_rebuilt"],
        )
        add_action(
            "create_port",
            "create_port",
            {"port_id": 1, "port_type": "discrete", "impedance_ohm": 50.0, "reference_mm": {"x": 0.0, "y": 0.0, "z": 0.0}},
            checksum_scope="simulation",
            rationale_tags=["baseline_excitation"],
            expected_effects=["port_defined"],
        )
        add_action(
            "set_boundary",
            "set_boundary",
            {"boundary_type": "open_add_space", "padding_mm": 15.0},
            checksum_scope="simulation",
            rationale_tags=["baseline_simulation"],
            expected_effects=["boundary_set"],
        )
        add_action(
            "set_solver",
            "set_solver",
            {"solver_type": "time_domain", "mesh_cells_per_wavelength": 20},
            checksum_scope="simulation",
            rationale_tags=["baseline_simulation"],
            expected_effects=["solver_ready"],
        )
        if supports_farfield:
            add_action(
                "add_farfield_monitor",
                "add_farfield_monitor",
                {
                    "monitor_name": f"farfield_{target_frequency:.3f}ghz".replace(".", "p"),
                    "frequency_ghz": target_frequency,
                },
                checksum_scope="simulation",
                rationale_tags=["client_capability_export", "simulation_preconditions"],
                expected_effects=["farfield_monitor_configured"],
            )
    else:
        changed_parameters: list[tuple[str, float]] = []
        if previous_parameter_values is None:
            changed_parameters = list(parameter_values.items())
        else:
            for param_name, param_value in parameter_values.items():
                previous_value = previous_parameter_values.get(param_name)
                if previous_value is None or abs(param_value - previous_value) > 1e-9:
                    changed_parameters.append((param_name, param_value))

        changed_family_parameters: list[tuple[str, float]] = []
        if previous_family_parameter_values is None:
            changed_family_parameters = list(family_parameter_values.items())
        else:
            for param_name, param_value in family_parameter_values.items():
                previous_value = previous_family_parameter_values.get(param_name)
                if previous_value is None or abs(param_value - previous_value) > 1e-9:
                    changed_family_parameters.append((param_name, param_value))

        for param_name, param_value in changed_parameters:
            add_action(
                "update_parameter",
                "update_parameter",
                {"name": param_name, "value": param_value},
                checksum_scope="geometry",
                rationale_tags=["parametric_delta", f"delta:{param_name}"],
                expected_effects=[f"parameter_{param_name}_updated"],
            )

        for param_name, param_value in changed_family_parameters:
            add_action(
                "update_parameter",
                "update_parameter",
                {"name": param_name, "value": param_value},
                checksum_scope="geometry",
                rationale_tags=["family_delta", f"family_delta:{param_name}"],
                expected_effects=[f"family_parameter_{param_name}_updated"],
            )

        add_action(
            "rebuild_model",
            "rebuild_model",
            {},
            checksum_scope="simulation",
            rationale_tags=["parametric_delta", "control_rebuild"],
            expected_effects=["model_rebuilt"],
        )

    add_action(
        "run_simulation",
        "run_simulation",
        {"timeout_sec": min(max_sim_timeout, 900)},
        checksum_scope="simulation",
        rationale_tags=["baseline_simulation"],
        expected_effects=["simulation_started"],
    )
    add_action(
        "export_s_parameters",
        "export_s_parameters",
        {"format": export_format, "destination_hint": "s11"},
        on_failure="continue",
        checksum_scope="exports",
        rationale_tags=["baseline_exports"],
        expected_effects=["s_parameters_available"],
    )
    add_action(
        "extract_summary_metrics",
        "extract_summary_metrics",
        {"metrics": ["center_frequency_ghz", "bandwidth_mhz", "return_loss_db", "vswr", "gain_dbi"]},
        on_failure="continue",
        checksum_scope="exports",
        rationale_tags=["baseline_exports"],
        expected_effects=["summary_metrics_available"],
    )
    if supports_farfield:
        add_action(
            "export_farfield",
            "export_farfield",
            {
                "format": export_format,
                "frequency_ghz": target_frequency,
                "destination_hint": "farfield",
            },
            on_failure="continue",
            checksum_scope="exports",
            rationale_tags=["client_capability_export"],
            expected_effects=["farfield_available"],
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
            "center_frequency_ghz": float(family_parameter_values.get("design_frequency_ghz", target_frequency)),
            "bandwidth_mhz": target_bandwidth,
        },
        "design_recipe": {
            "family": str(req_any.target_spec.antenna_family),
            "patch_shape": patch_shape,
            "feed_type": str(getattr(req_any.target_spec, "feed_type", "auto")),
            "recipe_name": str(recipe.get("recipe_name", "unknown_recipe")),
            "substrate": recipe.get("substrate"),
            "notes": recipe.get("notes", []),
            "selected_materials": {
                "conductor": allowed_material,
                "substrate": allowed_substrate,
            },
            "family_parameters": (
                {
                    "implementation_command": "implement_amc",
                    "implementation_strategy": "client_heuristic",
                }
                if family_name == "amc_patch"
                else {
                    **(recipe.get("family_parameters", {}) if isinstance(recipe.get("family_parameters"), dict) else {}),
                    **dict(getattr(ann, "family_parameters", {}) or {}),
                }
            ),
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
    previous_ann: AnnPrediction | None = None,
) -> dict[str, Any]:
    if PLANNER_SETTINGS.mode != "fixed" and not PLANNER_SETTINGS.dynamic_enabled:
        raise ValueError("dynamic planner mode is disabled by configuration")

    action_plan = build_fixed_action_plan(
        request,
        ann,
        session_id=session_id,
        trace_id=trace_id,
        iteration_index=iteration_index,
        previous_ann=previous_ann,
    )
    return compile_action_plan(action_plan)
