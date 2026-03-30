from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from config import SCHEMAS_DIR


SCHEMA_FILES: dict[str, Path] = {
    "optimize_request": SCHEMAS_DIR / "http" / "optimize_request.v1.json",
    "optimize_response": SCHEMAS_DIR / "http" / "optimize_response.v1.json",
    "command_package": SCHEMAS_DIR / "commands" / "cst_command_package.v1.json",
    "action_catalog": SCHEMAS_DIR / "planning" / "action_catalog.v1.json",
    "action_plan": SCHEMAS_DIR / "planning" / "action_plan.v1.json",
    "client_feedback": SCHEMAS_DIR / "http" / "client_feedback.v1.json",
    "session_event": SCHEMAS_DIR / "ws" / "session_event.v1.json",
}


class ContractValidationError(ValueError):
    """Raised when payload does not satisfy a JSON schema contract."""


@lru_cache(maxsize=None)
def _get_validator(schema_key: str) -> Draft202012Validator:
    if schema_key not in SCHEMA_FILES:
        raise KeyError(f"Unknown schema key: {schema_key}")
    schema_path = SCHEMA_FILES[schema_key]
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema_data)


def validate_contract(schema_key: str, payload: dict[str, Any]) -> None:
    validator = _get_validator(schema_key)
    errors = sorted(validator.iter_errors(payload), key=lambda e: str(list(e.path)))
    if not errors:
        return

    first_five = errors[:5]
    formatted = []
    for err in first_five:
        path = ".".join(str(p) for p in err.path) or "<root>"
        formatted.append(f"{path}: {err.message}")

    raise ContractValidationError("; ".join(formatted))
