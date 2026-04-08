from __future__ import annotations

from app.commands.planner import build_command_package
from app.core.schemas import AnnPrediction, DimensionPrediction, OptimizeRequest


def _request(*, patch_shape: str) -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "schema_version": "optimize_request.v1",
            "user_request": f"Design a {patch_shape} microstrip patch antenna around 2.45 GHz with 80 MHz bandwidth.",
            "target_spec": {
                "frequency_ghz": 2.45,
                "bandwidth_mhz": 80.0,
                "antenna_family": "microstrip_patch",
                "patch_shape": patch_shape,
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
                "priority": "normal",
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
            patch_radius_mm=19.0,
            substrate_length_mm=60.0,
            substrate_width_mm=70.0,
            substrate_height_mm=1.6,
            feed_length_mm=12.0,
            feed_width_mm=3.0,
            feed_offset_x_mm=0.0,
            feed_offset_y_mm=-5.0,
        ),
    )


def test_rectangular_package_uses_explicit_brick_geometry_commands() -> None:
    package = build_command_package(_request(patch_shape="rectangular"), _ann(), session_id="sess-r", trace_id="trace-r")
    commands = package["commands"]

    assert any(item["command"] == "create_component" for item in commands)
    assert any(item["command"] == "define_brick" and item["params"].get("name") == "patch" for item in commands)


def test_circular_package_uses_cylinder_patch_geometry_command() -> None:
    package = build_command_package(_request(patch_shape="circular"), _ann(), session_id="sess-c", trace_id="trace-c")
    commands = package["commands"]

    assert any(item["command"] == "create_component" for item in commands)
    assert any(item["command"] == "define_cylinder" and item["params"].get("name") == "patch" for item in commands)
