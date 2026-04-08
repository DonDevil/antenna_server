from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, cast


TOP_LEVEL_REQUIRED_FIELDS = (
    "schema_version",
    "command_catalog_version",
    "session_id",
    "trace_id",
    "design_id",
    "iteration_index",
    "units",
    "predicted_dimensions",
    "commands",
    "expected_exports",
    "safety_checks",
)


class V2CommandContractValidator:
    """Command-level preflight validator for cst_command_package.v2 payloads."""

    DEFAULT_CONTRACT_PATH = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "commands"
        / "cst_command_package.v2.command_contract.json"
    )

    def __init__(self, contract_path: Path | None = None) -> None:
        self.contract_path = contract_path or self.DEFAULT_CONTRACT_PATH
        self._commands = self._load_contract()

    def _load_contract(self) -> Dict[str, Dict[str, Any]]:
        if not self.contract_path.exists():
            raise ValueError(f"V2 command contract not found: {self.contract_path}")

        data: Dict[str, Any] = json.loads(self.contract_path.read_text(encoding="utf-8"))
        commands_raw = data.get("commands")
        if not isinstance(commands_raw, dict) or not commands_raw:
            raise ValueError("V2 command contract must define a non-empty 'commands' map")

        commands: Dict[str, Dict[str, Any]] = {}
        command_specs = cast(Dict[object, object], commands_raw)
        for name, spec in command_specs.items():
            if isinstance(spec, dict):
                commands[str(name)] = cast(Dict[str, Any], spec)
        if not commands:
            raise ValueError("V2 command contract must include at least one command spec")
        return commands

    def validate_package(self, package: Any) -> None:
        if not isinstance(package, Mapping):
            raise ValueError("Command package must be an object")

        package_map = cast(Mapping[str, Any], package)

        missing_fields = [field for field in TOP_LEVEL_REQUIRED_FIELDS if field not in package_map]
        if missing_fields:
            raise ValueError(
                f"Command package missing required fields: {', '.join(missing_fields)}"
            )

        schema_version = package_map.get("schema_version")
        if schema_version != "cst_command_package.v2":
            raise ValueError(
                f"Command package schema_version must be 'cst_command_package.v2', got {schema_version!r}"
            )

        command_catalog_version = package_map.get("command_catalog_version")
        if not isinstance(command_catalog_version, str) or not command_catalog_version.strip():
            raise ValueError("Command package command_catalog_version must be a non-empty string")

        for field_name in ("session_id", "trace_id", "design_id"):
            value = package_map.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Command package {field_name} must be a non-empty string")

        iteration_index = package_map.get("iteration_index")
        if not isinstance(iteration_index, int) or iteration_index < 0:
            raise ValueError("Command package iteration_index must be an integer >= 0")

        units = package_map.get("units")
        if not isinstance(units, Mapping):
            raise ValueError("Command package units must be an object")

        predicted_dimensions = package_map.get("predicted_dimensions")
        if not isinstance(predicted_dimensions, Mapping):
            raise ValueError("Command package predicted_dimensions must be an object")

        expected_exports = package_map.get("expected_exports")
        if not isinstance(expected_exports, list):
            raise ValueError("Command package expected_exports must be a list")

        safety_checks = package_map.get("safety_checks")
        if not isinstance(safety_checks, list):
            raise ValueError("Command package safety_checks must be a list")

        commands_raw = package_map.get("commands")
        if not isinstance(commands_raw, list) or not commands_raw:
            raise ValueError("Command package commands must be a non-empty list")

        command_list = cast(List[object], commands_raw)
        previous_seq = 0
        for index, command_raw in enumerate(command_list, start=1):
            if not isinstance(command_raw, dict):
                raise ValueError(f"Command entry {index} must be an object")

            command = cast(Dict[str, Any], command_raw)
            seq_value = command.get("seq")
            if not isinstance(seq_value, int):
                raise ValueError(f"Command entry {index} missing integer seq")
            if seq_value <= previous_seq:
                raise ValueError("Command sequence must be strictly increasing")
            previous_seq = seq_value

            command_name_value = command.get("command")
            if not isinstance(command_name_value, str) or not command_name_value.strip():
                raise ValueError(f"Command {seq_value} missing command name")

            params_value = command.get("params")
            if not isinstance(params_value, dict):
                raise ValueError(f"Command {seq_value}:{command_name_value} has invalid params payload")
            params_dict = cast(Dict[str, Any], params_value)

            command.setdefault("on_failure", "abort")
            command.setdefault("checksum_scope", "command")

            on_failure = command.get("on_failure")
            if on_failure not in {"abort", "retry_once", "continue"}:
                raise ValueError(
                    f"Command {seq_value}:{command_name_value} has invalid on_failure value: {on_failure!r}"
                )

            checksum_scope = command.get("checksum_scope")
            if not isinstance(checksum_scope, str) or not checksum_scope.strip():
                raise ValueError(f"Command {seq_value}:{command_name_value} has invalid checksum_scope")

            self.validate_command(
                command_name=command_name_value,
                params=params_dict,
                seq=seq_value,
            )
    def validate_commands(self, commands: Iterable[Any]) -> None:
        previous_seq = 0
        for cmd in commands:
            seq = int(getattr(cmd, "seq", -1))
            if seq <= previous_seq:
                raise ValueError("Command sequence must be strictly increasing")
            previous_seq = seq
            self.validate_command(
                command_name=str(getattr(cmd, "command", "")),
                params=getattr(cmd, "params", {}),
                seq=seq,
            )

    def validate_command(self, command_name: str, params: Mapping[str, Any], seq: int) -> None:
        spec = self._commands.get(command_name)
        if spec is None:
            raise ValueError(f"Command {seq}:{command_name} is not declared in strict V2 contract")

        required = self._list_of_str(spec.get("required_params"))
        missing = [key for key in required if key not in params or params.get(key) is None]
        if missing:
            raise ValueError(
                f"Command {seq}:{command_name} missing required params: {', '.join(missing)}"
            )

        any_of_groups = spec.get("any_of", [])
        if any_of_groups:
            group_ok = False
            for group in any_of_groups:
                group_keys = self._list_of_str(group)
                if group_keys and all(key in params and params.get(key) is not None for key in group_keys):
                    group_ok = True
                    break
            if not group_ok:
                group_text = " or ".join("[" + ", ".join(self._list_of_str(group)) + "]" for group in any_of_groups)
                raise ValueError(
                    f"Command {seq}:{command_name} requires one of parameter groups: {group_text}"
                )

        non_empty_lists = self._list_of_str(spec.get("non_empty_lists"))
        for key in non_empty_lists:
            value = params.get(key)
            if not isinstance(value, list):
                raise ValueError(f"Command {seq}:{command_name} requires non-empty list param '{key}'")
            value_list = cast(List[object], value)
            if len(value_list) == 0:
                raise ValueError(f"Command {seq}:{command_name} requires non-empty list param '{key}'")

    @staticmethod
    def _list_of_str(value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Contract list field must be a list")

        items = cast(List[object], value)
        result: List[str] = []
        for item in items:
            result.append(str(item))
        return result

_DEFAULT_VALIDATOR = V2CommandContractValidator()


def validate_command_package_v2(package: Mapping[str, Any]) -> None:
    _DEFAULT_VALIDATOR.validate_package(package)
