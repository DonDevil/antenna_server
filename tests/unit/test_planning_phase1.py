from __future__ import annotations

from app.commands.planner import build_command_package, build_fixed_action_plan
from app.core.json_contracts import validate_contract
from app.core.schemas import AnnPrediction, DimensionPrediction, OptimizeRequest
from app.planning.action_catalog import get_action_catalog_payload
from app.planning.command_compiler import compile_action_plan
from app.planning.v2_command_contract import validate_command_package_v2


def _request() -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "schema_version": "optimize_request.v1",
            "user_request": "Design a microstrip patch antenna around 2.4 GHz with 80 MHz bandwidth.",
            "target_spec": {
                "frequency_ghz": 2.4,
                "bandwidth_mhz": 80.0,
                "antenna_family": "microstrip_patch",
            },
            "design_constraints": {
                "allowed_materials": ["Copper (annealed)"],
                "allowed_substrates": ["Rogers RT/duroid 5880"],
            },
            "optimization_policy": {
                "mode": "auto_iterate",
                "max_iterations": 3,
                "stop_on_first_valid": True,
                "acceptance": {
                    "center_tolerance_mhz": 20.0,
                    "minimum_bandwidth_mhz": 20.0,
                    "maximum_vswr": 2.0,
                    "minimum_gain_dbi": 0.0,
                },
                "fallback_behavior": "best_effort",
            },
            "runtime_preferences": {
                "require_explanations": False,
                "persist_artifacts": True,
                "llm_temperature": 0.0,
                "timeout_budget_sec": 300,
            },
            "client_capabilities": {
                "supports_farfield_export": True,
                "supports_current_distribution_export": False,
                "supports_parameter_sweep": False,
                "max_simulation_timeout_sec": 600,
                "export_formats": ["json"],
            },
        }
    )


def _ann() -> AnnPrediction:
    return AnnPrediction(
        ann_model_version="v1",
        confidence=0.91,
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


def test_action_catalog_validates_against_contract() -> None:
    payload = get_action_catalog_payload()
    validate_contract("action_catalog", payload)
    assert payload["catalog_version"] == "v2"
    assert len(payload["actions"]) >= 15


def test_fixed_action_plan_compiles_to_valid_command_package() -> None:
    action_plan = build_fixed_action_plan(
        _request(),
        _ann(),
        session_id="session-123",
        trace_id="trace-123",
        iteration_index=0,
    )
    validate_contract("action_plan", action_plan)

    command_package = compile_action_plan(action_plan)
    validate_contract("command_package", command_package)
    assert command_package["design_id"] == "design_session-123"
    assert command_package["commands"][0]["command"] == "create_project"
    assert command_package["commands"][-1]["command"] == "export_farfield"


def test_farfield_monitor_command_is_inserted_before_simulation() -> None:
    command_package = build_command_package(
        _request(),
        _ann(),
        session_id="session-monitor",
        trace_id="trace-monitor",
        iteration_index=0,
    )

    commands = [item["command"] for item in command_package["commands"]]
    assert "add_farfield_monitor" in commands
    assert commands.index("add_farfield_monitor") < commands.index("run_simulation")


def test_initial_build_emits_parametric_geometry_and_rebuild() -> None:
    command_package = build_command_package(
        _request(),
        _ann(),
        session_id="session-parametric",
        trace_id="trace-parametric",
        iteration_index=0,
    )

    commands = command_package["commands"]
    command_names = [item["command"] for item in commands]

    assert "define_parameter" in command_names
    assert "rebuild_model" in command_names
    assert command_names.index("rebuild_model") < command_names.index("run_simulation")

    brick_commands = [item for item in commands if item["command"] == "define_brick"]
    assert brick_commands
    assert brick_commands[0]["params"]["component"] == "antenna"
    assert any(isinstance(value, str) and "sx" in value for value in brick_commands[0]["params"]["xrange"])


def test_build_command_package_uses_delta_compiler_path_for_refinement_iterations() -> None:
    command_package = build_command_package(
        _request(),
        _ann(),
        session_id="session-456",
        trace_id="trace-456",
        iteration_index=2,
    )

    commands = [item["command"] for item in command_package["commands"]]

    assert command_package["schema_version"] == "cst_command_package.v2"
    assert command_package["command_catalog_version"] == "v2"
    assert command_package["iteration_index"] == 2
    assert commands[0] == "update_parameter"
    assert "rebuild_model" in commands
    assert "run_simulation" in commands
    assert "define_brick" not in commands
    assert "define_cylinder" not in commands


def test_v2_preflight_accepts_valid_alias_rich_package() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {
                "seq": 1,
                "command": "new_component",
                "params": {"name": "component1"},
            },
            {
                "seq": 2,
                "command": "define_extrude",
                "params": {
                    "name": "solid1",
                    "component": "component1",
                    "points": [[0, 0], [10, 0], [10, 5]],
                },
            },
            {
                "seq": 3,
                "command": "pick_end_point",
                "params": {
                    "component": "component1",
                    "solid": "solid1",
                    "endpoint_id": 1,
                },
            },
            {
                "seq": 4,
                "command": "create_port",
                "params": {
                    "port_id": 1,
                    "impedance_ohm": 50.0,
                    "reference_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
        ],
        "expected_exports": ["summary_metrics"],
        "safety_checks": ["command_order_validated"],
    }

    validate_command_package_v2(package)


def test_v2_preflight_rejects_unknown_command() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {"seq": 1, "command": "not_a_real_v2_command", "params": {}},
        ],
        "expected_exports": [],
        "safety_checks": [],
    }

    try:
        validate_command_package_v2(package)
        assert False, "expected unknown command to be rejected"
    except ValueError as exc:
        assert "not declared in strict V2 contract" in str(exc)


def test_v2_preflight_rejects_missing_required_params() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {
                "seq": 1,
                "command": "define_brick",
                "params": {"name": "solid1", "xrange": [0, 1], "yrange": [0, 1]},
            },
        ],
        "expected_exports": [],
        "safety_checks": [],
    }

    try:
        validate_command_package_v2(package)
        assert False, "expected missing params to be rejected"
    except ValueError as exc:
        assert "missing required params" in str(exc)
        assert "component" in str(exc)
        assert "zrange" in str(exc)


def test_v2_preflight_accepts_parameter_update_commands() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 1,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {"seq": 1, "command": "update_parameter", "params": {"name": "px", "value": 41.5}},
            {"seq": 2, "command": "set_parameter", "params": {"name": "py", "value": "py+0.2"}},
            {"seq": 3, "command": "rebuild_model", "params": {}},
            {"seq": 4, "command": "run_simulation", "params": {"timeout_sec": 120}},
        ],
        "expected_exports": ["s_parameters"],
        "safety_checks": ["command_order_validated"],
    }

    validate_command_package_v2(package)


def test_v2_preflight_enforces_any_of_groups() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {
                "seq": 1,
                "command": "create_port",
                "params": {"port_id": 1, "impedance_ohm": 50.0},
            },
        ],
        "expected_exports": [],
        "safety_checks": [],
    }

    try:
        validate_command_package_v2(package)
        assert False, "expected any_of rule to be enforced"
    except ValueError as exc:
        assert "requires one of parameter groups" in str(exc)


def test_v2_preflight_enforces_non_empty_points_lists() -> None:
    package = {
        "schema_version": "cst_command_package.v2",
        "command_catalog_version": "v2",
        "session_id": "session-v2",
        "trace_id": "trace-v2",
        "design_id": "design-v2",
        "iteration_index": 0,
        "units": {"geometry": "mm", "frequency": "ghz"},
        "predicted_dimensions": {},
        "commands": [
            {
                "seq": 1,
                "command": "define_rotate",
                "params": {"name": "rot1", "component": "component1", "points": []},
            },
        ],
        "expected_exports": [],
        "safety_checks": [],
    }

    try:
        validate_command_package_v2(package)
        assert False, "expected non-empty points list rule to be enforced"
    except ValueError as exc:
        assert "non-empty list param 'points'" in str(exc)