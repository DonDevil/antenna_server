from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.json_contracts import validate_contract
from config import PLANNER_SETTINGS


ACTION_CATALOG: dict[str, Any] = {
    "schema_version": "action_catalog.v1",
    "catalog_version": PLANNER_SETTINGS.command_catalog_version,
    "actions": [
        {
            "action": "create_project",
            "command": "create_project",
            "description": "Create a CST project container for the current design session.",
            "phase": "setup",
            "prerequisites": [],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "project_name", "type": "string", "required": True, "min_length": 1, "max_length": 120}
            ],
        },
        {
            "action": "set_units",
            "command": "set_units",
            "description": "Set geometry and frequency units before any geometry or solver calls.",
            "phase": "setup",
            "prerequisites": ["create_project"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "geometry", "type": "string", "required": True, "enum": ["mm"]},
                {"name": "frequency", "type": "string", "required": True, "enum": ["ghz"]}
            ],
        },
        {
            "action": "set_frequency_range",
            "command": "set_frequency_range",
            "description": "Configure the simulation frequency sweep around the target region.",
            "phase": "setup",
            "prerequisites": ["set_units"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "start_ghz", "type": "number", "required": True, "minimum": 0.0, "unit": "ghz"},
                {"name": "stop_ghz", "type": "number", "required": True, "minimum": 0.0, "unit": "ghz"}
            ],
        },
        {
            "action": "define_material",
            "command": "define_material",
            "description": "Register a conductor or substrate material in the CST project.",
            "phase": "materials",
            "prerequisites": ["set_units"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "name", "type": "string", "required": True, "min_length": 1},
                {"name": "kind", "type": "string", "required": True, "enum": ["conductor", "substrate"]},
                {"name": "epsilon_r", "type": "number", "required": False, "minimum": 0.0},
                {"name": "loss_tangent", "type": "number", "required": False, "minimum": 0.0, "maximum": 10.0},
                {"name": "conductivity_s_per_m", "type": "number", "required": False, "minimum": 0.0}
            ],
        },
        {
            "action": "create_substrate",
            "command": "create_substrate",
            "description": "Create the substrate solid with ANN-predicted dimensions.",
            "phase": "geometry",
            "prerequisites": ["define_material"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "name", "type": "string", "required": True},
                {"name": "material", "type": "string", "required": True},
                {"name": "length_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "width_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "height_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "origin_mm", "type": "object", "required": True}
            ],
        },
        {
            "action": "create_ground_plane",
            "command": "create_ground_plane",
            "description": "Create the ground plane beneath the substrate.",
            "phase": "geometry",
            "prerequisites": ["create_substrate"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "name", "type": "string", "required": True},
                {"name": "material", "type": "string", "required": True},
                {"name": "length_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "width_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "thickness_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "z_mm", "type": "number", "required": True, "unit": "mm"}
            ],
        },
        {
            "action": "create_patch",
            "command": "create_patch",
            "description": "Create the radiating patch geometry.",
            "phase": "geometry",
            "prerequisites": ["create_substrate"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "name", "type": "string", "required": True},
                {"name": "material", "type": "string", "required": True},
                {"name": "length_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "width_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "thickness_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "center_mm", "type": "object", "required": True}
            ],
        },
        {
            "action": "create_feedline",
            "command": "create_feedline",
            "description": "Create the feedline geometry and placement.",
            "phase": "geometry",
            "prerequisites": ["create_patch"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "name", "type": "string", "required": True},
                {"name": "material", "type": "string", "required": True},
                {"name": "length_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "width_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "thickness_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"},
                {"name": "start_mm", "type": "object", "required": True},
                {"name": "end_mm", "type": "object", "required": True}
            ],
        },
        {
            "action": "create_port",
            "command": "create_port",
            "description": "Attach the port definition used for simulation excitation.",
            "phase": "simulation",
            "prerequisites": ["create_feedline"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "port_id", "type": "integer", "required": True, "minimum": 1},
                {"name": "port_type", "type": "string", "required": True, "enum": ["discrete"]},
                {"name": "impedance_ohm", "type": "number", "required": True, "minimum": 0.0, "unit": "ohm"},
                {"name": "reference_mm", "type": "object", "required": True}
            ],
        },
        {
            "action": "set_boundary",
            "command": "set_boundary",
            "description": "Configure the simulation boundary padding.",
            "phase": "simulation",
            "prerequisites": ["create_port"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "boundary_type", "type": "string", "required": True, "enum": ["open_add_space"]},
                {"name": "padding_mm", "type": "number", "required": True, "minimum": 0.0, "unit": "mm"}
            ],
        },
        {
            "action": "set_solver",
            "command": "set_solver",
            "description": "Configure the CST solver settings.",
            "phase": "simulation",
            "prerequisites": ["set_boundary"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "solver_type", "type": "string", "required": True, "enum": ["time_domain"]},
                {"name": "mesh_cells_per_wavelength", "type": "integer", "required": True, "minimum": 1}
            ],
        },
        {
            "action": "run_simulation",
            "command": "run_simulation",
            "description": "Run the CST simulation with bounded timeout.",
            "phase": "simulation",
            "prerequisites": ["set_solver"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "timeout_sec", "type": "integer", "required": True, "minimum": 1, "unit": "sec"}
            ],
        },
        {
            "action": "export_s_parameters",
            "command": "export_s_parameters",
            "description": "Export S-parameter data after simulation.",
            "phase": "exports",
            "prerequisites": ["run_simulation"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "format", "type": "string", "required": True, "enum": ["json", "csv", "txt"]},
                {"name": "destination_hint", "type": "string", "required": True, "min_length": 1}
            ],
        },
        {
            "action": "extract_summary_metrics",
            "command": "extract_summary_metrics",
            "description": "Extract scalar summary metrics from the simulation results.",
            "phase": "exports",
            "prerequisites": ["run_simulation"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "metrics", "type": "array", "required": True, "min_items": 1}
            ],
        },
        {
            "action": "export_farfield",
            "command": "export_farfield",
            "description": "Export farfield results when supported by the client.",
            "phase": "exports",
            "prerequisites": ["run_simulation"],
            "incompatible_with": [],
            "parameter_definitions": [
                {"name": "format", "type": "string", "required": True, "enum": ["json", "csv", "txt"]},
                {"name": "frequency_ghz", "type": "number", "required": True, "minimum": 0.0, "unit": "ghz"},
                {"name": "destination_hint", "type": "string", "required": True, "min_length": 1}
            ],
        },
    ],
}


@lru_cache(maxsize=1)
def get_action_catalog_payload() -> dict[str, Any]:
    validate_contract("action_catalog", ACTION_CATALOG)
    return ACTION_CATALOG


@lru_cache(maxsize=1)
def get_action_specs() -> dict[str, dict[str, Any]]:
    payload = get_action_catalog_payload()
    actions = payload.get("actions", [])
    return {str(action["action"]): action for action in actions}