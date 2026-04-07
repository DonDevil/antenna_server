from __future__ import annotations

from app.commands.planner import build_command_package, build_fixed_action_plan
from app.core.json_contracts import validate_contract
from app.core.schemas import AnnPrediction, DimensionPrediction, OptimizeRequest
from app.planning.action_catalog import get_action_catalog_payload
from app.planning.command_compiler import compile_action_plan


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
    assert payload["catalog_version"] == "v1"
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


def test_build_command_package_uses_phase1_compiler_path() -> None:
    command_package = build_command_package(
        _request(),
        _ann(),
        session_id="session-456",
        trace_id="trace-456",
        iteration_index=2,
    )

    assert command_package["schema_version"] == "cst_command_package.v1"
    assert command_package["command_catalog_version"] == "v1"
    assert command_package["iteration_index"] == 2
    assert command_package["commands"][2]["command"] == "set_frequency_range"