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

GEOMETRY_COMMANDS_REQUIRING_COMPONENT = {
    "create_substrate",
    "create_ground_plane",
    "create_patch",
    "create_feedline",
    "define_brick",
    "define_sphere",
    "define_cone",
    "define_torus",
    "define_cylinder",
    "define_ecylinder",
    "define_extrude",
    "define_rotate",
    "define_loft",
    "brick",
    "sphere",
    "cone",
    "torus",
    "cylinder",
    "ecylinder",
    "extrude",
    "rotate",
    "loft",
}
PARAMETER_MUTATION_COMMANDS = {"define_parameter", "update_parameter", "set_parameter"}
SIMULATION_EXPORT_COMMANDS = {
    "run_simulation",
    "export_s_parameters",
    "extract_summary_metrics",
    "export_farfield",
}


class V2CommandValidationError(ValueError):
    """Structured validation error for strict V2 command packages."""

    def __init__(
        self,
        message: str,
        *,
        command_index: int | None = None,
        command_name: str | None = None,
        invalid_fields: Iterable[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.command_index = command_index
        self.command_name = command_name
        self.invalid_fields = list(invalid_fields or [])

    def as_detail(self) -> dict[str, Any]:
        return {
            "command_index": self.command_index,
            "command_name": self.command_name,
            "invalid_fields": self.invalid_fields,
        }


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
            raise V2CommandValidationError("Command package must be an object")

        package_map = cast(Mapping[str, Any], package)

        missing_fields = [field for field in TOP_LEVEL_REQUIRED_FIELDS if field not in package_map]
        if missing_fields:
            raise V2CommandValidationError(
                f"Command package missing required fields: {', '.join(missing_fields)}",
                invalid_fields=missing_fields,
            )

        schema_version = package_map.get("schema_version")
        if schema_version != "cst_command_package.v2":
            raise V2CommandValidationError(
                f"Command package schema_version must be 'cst_command_package.v2', got {schema_version!r}",
                invalid_fields=["schema_version"],
            )

        command_catalog_version = package_map.get("command_catalog_version")
        if not isinstance(command_catalog_version, str) or not command_catalog_version.strip():
            raise V2CommandValidationError(
                "Command package command_catalog_version must be a non-empty string",
                invalid_fields=["command_catalog_version"],
            )

        for field_name in ("session_id", "trace_id", "design_id"):
            value = package_map.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise V2CommandValidationError(
                    f"Command package {field_name} must be a non-empty string",
                    invalid_fields=[field_name],
                )

        iteration_index = package_map.get("iteration_index")
        if not isinstance(iteration_index, int) or iteration_index < 0:
            raise V2CommandValidationError(
                "Command package iteration_index must be an integer >= 0",
                invalid_fields=["iteration_index"],
            )

        units = package_map.get("units")
        if not isinstance(units, Mapping):
            raise V2CommandValidationError("Command package units must be an object", invalid_fields=["units"])

        predicted_dimensions = package_map.get("predicted_dimensions")
        if not isinstance(predicted_dimensions, Mapping):
            raise V2CommandValidationError(
                "Command package predicted_dimensions must be an object",
                invalid_fields=["predicted_dimensions"],
            )

        expected_exports = package_map.get("expected_exports")
        if not isinstance(expected_exports, list):
            raise V2CommandValidationError(
                "Command package expected_exports must be a list",
                invalid_fields=["expected_exports"],
            )

        safety_checks = package_map.get("safety_checks")
        if not isinstance(safety_checks, list):
            raise V2CommandValidationError(
                "Command package safety_checks must be a list",
                invalid_fields=["safety_checks"],
            )

        commands_raw = package_map.get("commands")
        if not isinstance(commands_raw, list) or not commands_raw:
            raise V2CommandValidationError(
                "Command package commands must be a non-empty list",
                invalid_fields=["commands"],
            )

        command_list = cast(List[object], commands_raw)
        previous_seq = 0
        last_parameter_mutation_seq: int | None = None
        rebuild_sequences: list[int] = []
        first_simulation_export_seq: int | None = None

        for index, command_raw in enumerate(command_list, start=1):
            if not isinstance(command_raw, dict):
                raise V2CommandValidationError(f"Command entry {index} must be an object", command_index=index)

            command = cast(Dict[str, Any], command_raw)
            seq_value = command.get("seq")
            if not isinstance(seq_value, int):
                raise V2CommandValidationError(
                    f"Command entry {index} missing integer seq",
                    command_index=index,
                    invalid_fields=["seq"],
                )
            if seq_value <= previous_seq:
                raise V2CommandValidationError(
                    "Command sequence must be strictly increasing",
                    command_index=seq_value,
                    invalid_fields=["seq"],
                )
            previous_seq = seq_value

            command_name_value = command.get("command")
            if not isinstance(command_name_value, str) or not command_name_value.strip():
                raise V2CommandValidationError(
                    f"Command {seq_value} missing command name",
                    command_index=seq_value,
                    invalid_fields=["command"],
                )

            params_value = command.get("params")
            if not isinstance(params_value, dict):
                raise V2CommandValidationError(
                    f"Command {seq_value}:{command_name_value} has invalid params payload",
                    command_index=seq_value,
                    command_name=command_name_value,
                    invalid_fields=["params"],
                )
            params_dict = cast(Dict[str, Any], params_value)

            command.setdefault("on_failure", "abort")
            command.setdefault("checksum_scope", "command")

            on_failure = command.get("on_failure")
            if on_failure not in {"abort", "retry_once", "continue"}:
                raise V2CommandValidationError(
                    f"Command {seq_value}:{command_name_value} has invalid on_failure value: {on_failure!r}",
                    command_index=seq_value,
                    command_name=command_name_value,
                    invalid_fields=["on_failure"],
                )

            checksum_scope = command.get("checksum_scope")
            if not isinstance(checksum_scope, str) or not checksum_scope.strip():
                raise V2CommandValidationError(
                    f"Command {seq_value}:{command_name_value} has invalid checksum_scope",
                    command_index=seq_value,
                    command_name=command_name_value,
                    invalid_fields=["checksum_scope"],
                )

            self.validate_command(
                command_name=command_name_value,
                params=params_dict,
                seq=seq_value,
            )

            if command_name_value in PARAMETER_MUTATION_COMMANDS:
                last_parameter_mutation_seq = seq_value
            elif command_name_value == "rebuild_model":
                rebuild_sequences.append(seq_value)
            elif command_name_value in SIMULATION_EXPORT_COMMANDS and first_simulation_export_seq is None:
                first_simulation_export_seq = seq_value

        if last_parameter_mutation_seq is not None:
            rebuild_after_mutation = [seq for seq in rebuild_sequences if seq > last_parameter_mutation_seq]
            if not rebuild_after_mutation:
                raise V2CommandValidationError(
                    "Command package must include rebuild_model after parameter mutations",
                    invalid_fields=["rebuild_model"],
                )

            earliest_rebuild = min(rebuild_after_mutation)
            if first_simulation_export_seq is not None and earliest_rebuild > first_simulation_export_seq:
                raise V2CommandValidationError(
                    "rebuild_model must appear before simulation or export commands after parameter mutations",
                    command_index=earliest_rebuild,
                    command_name="rebuild_model",
                    invalid_fields=["rebuild_model"],
                )

    def validate_commands(self, commands: Iterable[Any]) -> None:
        previous_seq = 0
        for cmd in commands:
            seq = int(getattr(cmd, "seq", -1))
            if seq <= previous_seq:
                raise V2CommandValidationError(
                    "Command sequence must be strictly increasing",
                    command_index=seq,
                    invalid_fields=["seq"],
                )
            previous_seq = seq
            self.validate_command(
                command_name=str(getattr(cmd, "command", "")),
                params=getattr(cmd, "params", {}),
                seq=seq,
            )

    def validate_command(self, command_name: str, params: Mapping[str, Any], seq: int) -> None:
        spec = self._commands.get(command_name)
        if spec is None:
            raise V2CommandValidationError(
                f"Command {seq}:{command_name} is not declared in strict V2 contract",
                command_index=seq,
                command_name=command_name,
                invalid_fields=["command"],
            )

        required = self._list_of_str(spec.get("required_params"))
        missing = [key for key in required if key not in params or params.get(key) is None]

        if command_name in GEOMETRY_COMMANDS_REQUIRING_COMPONENT:
            component = params.get("component")
            if not isinstance(component, str) or not component.strip():
                if "component" not in missing:
                    missing.append("component")

        if command_name in PARAMETER_MUTATION_COMMANDS:
            name_value = params.get("name")
            if not isinstance(name_value, str) or not name_value.strip():
                if "name" not in missing:
                    missing.append("name")
            value = params.get("value")
            if value is None or (isinstance(value, str) and not value.strip()):
                if "value" not in missing:
                    missing.append("value")

        if missing:
            raise V2CommandValidationError(
                f"Command {seq}:{command_name} missing required params: {', '.join(missing)}",
                command_index=seq,
                command_name=command_name,
                invalid_fields=missing,
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
                raise V2CommandValidationError(
                    f"Command {seq}:{command_name} requires one of parameter groups: {group_text}",
                    command_index=seq,
                    command_name=command_name,
                )

        non_empty_lists = self._list_of_str(spec.get("non_empty_lists"))
        for key in non_empty_lists:
            value = params.get(key)
            if not isinstance(value, list):
                raise V2CommandValidationError(
                    f"Command {seq}:{command_name} requires non-empty list param '{key}'",
                    command_index=seq,
                    command_name=command_name,
                    invalid_fields=[key],
                )
            value_list = cast(List[object], value)
            if len(value_list) == 0:
                raise V2CommandValidationError(
                    f"Command {seq}:{command_name} requires non-empty list param '{key}'",
                    command_index=seq,
                    command_name=command_name,
                    invalid_fields=[key],
                )

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
