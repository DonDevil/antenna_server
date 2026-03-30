from __future__ import annotations

from typing import Any

from app.core.json_contracts import validate_contract
from app.planning.action_catalog import get_action_specs


def compile_action_plan(action_plan: dict[str, Any]) -> dict[str, Any]:
    validate_contract("action_plan", action_plan)

    action_specs = get_action_specs()
    commands: list[dict[str, Any]] = []
    previous_seq = 0
    for item in action_plan["actions"]:
        seq = int(item["seq"])
        if seq <= previous_seq:
            raise ValueError("action plan sequence must be strictly increasing")
        previous_seq = seq

        action_name = str(item["action"])
        command_name = str(item["command"])
        spec = action_specs.get(action_name)
        if spec is None:
            raise ValueError(f"Unknown action in plan: {action_name}")
        if command_name != str(spec["command"]):
            raise ValueError(f"Action {action_name} must compile to command {spec['command']}")

        commands.append(
            {
                "seq": seq,
                "command": command_name,
                "params": item["params"],
                "on_failure": item["on_failure"],
                "checksum_scope": item["checksum_scope"],
            }
        )

    command_package = {
        "schema_version": "cst_command_package.v1",
        "command_catalog_version": action_plan["command_catalog_version"],
        "session_id": action_plan["session_id"],
        "trace_id": action_plan["trace_id"],
        "design_id": action_plan["design_id"],
        "iteration_index": int(action_plan["iteration_index"]),
        "units": action_plan["units"],
        "predicted_dimensions": action_plan["predicted_dimensions"],
        "predicted_metrics": action_plan["predicted_metrics"],
        "commands": commands,
        "expected_exports": action_plan["expected_exports"],
        "safety_checks": action_plan["safety_checks"],
    }
    validate_contract("command_package", command_package)
    return command_package