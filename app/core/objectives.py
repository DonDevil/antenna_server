from __future__ import annotations

from typing import Any

from app.core.schemas import OptimizeRequest


def build_initial_objective_state(request: OptimizeRequest) -> dict[str, Any]:
    primary = request.optimization_targets.primary
    secondary = request.optimization_targets.secondary
    return {
        "overall_status": "pending",
        "focus_area": "initial_geometry",
        "primary": {
            "s11": {"goal": primary.s11, "status": "pending"},
            "bandwidth": {"goal": primary.bandwidth, "status": "pending"},
            "gain": {"goal": primary.gain, "status": "pending"},
            "efficiency": {"goal": primary.efficiency, "status": "pending"},
        },
        "secondary": {
            "zin": {"target": secondary.zin_target, "status": "pending" if secondary.zin_target else "disabled"},
            "axial_ratio": {"target": secondary.axial_ratio, "status": "pending" if secondary.axial_ratio else "disabled"},
            "sll": {"goal": secondary.sll, "status": "pending" if secondary.sll != "ignore" else "disabled"},
            "front_to_back": {"goal": secondary.front_to_back, "status": "pending" if secondary.front_to_back != "ignore" else "disabled"},
        },
    }


def _status(condition: bool) -> str:
    return "met" if condition else "not_met"


def evaluate_objective_state(
    request: OptimizeRequest,
    feedback: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    acceptance = request.optimization_policy.acceptance
    target_bw = max(float(request.target_spec.bandwidth_mhz), float(acceptance.minimum_bandwidth_mhz))
    target_return_loss = float(acceptance.minimum_return_loss_db)

    actual_return_loss = feedback.get("actual_return_loss_db")
    actual_vswr = float(feedback.get("actual_vswr", 99.0))
    actual_bw = float(feedback.get("actual_bandwidth_mhz", 0.0))
    actual_gain = float(feedback.get("actual_gain_dbi", -99.0))
    actual_efficiency = feedback.get("actual_efficiency")
    actual_zin = feedback.get("actual_input_impedance") or feedback.get("actual_zin")
    actual_axial_ratio = feedback.get("actual_axial_ratio_db")
    actual_sll = feedback.get("actual_sll_db")
    actual_front_to_back = feedback.get("actual_front_to_back_db")

    if actual_return_loss is None:
        s11_status = "unknown"
    else:
        s11_status = _status(float(actual_return_loss) <= target_return_loss and actual_vswr <= float(acceptance.maximum_vswr))

    bandwidth_status = _status(actual_bw >= target_bw)
    gain_status = _status(actual_gain >= float(acceptance.minimum_gain_dbi))

    if actual_efficiency is None:
        efficiency_status = "unknown"
    else:
        efficiency_status = _status(float(actual_efficiency) >= 0.5)

    focus_area = "balanced_fine_tuning"
    if s11_status == "not_met":
        focus_area = "impedance_matching"
    elif abs(float(evaluation.get("freq_error_mhz", 0.0))) > float(acceptance.center_tolerance_mhz):
        focus_area = "frequency_tuning"
    elif bandwidth_status == "not_met":
        focus_area = "bandwidth_expansion"
    elif gain_status == "not_met":
        focus_area = "gain_improvement"

    return {
        "overall_status": "accepted" if bool(evaluation.get("accepted")) else "needs_refinement",
        "focus_area": focus_area,
        "primary": {
            "s11": {"goal": request.optimization_targets.primary.s11, "status": s11_status, "measured_return_loss_db": actual_return_loss},
            "bandwidth": {"goal": request.optimization_targets.primary.bandwidth, "status": bandwidth_status, "measured_bandwidth_mhz": actual_bw},
            "gain": {"goal": request.optimization_targets.primary.gain, "status": gain_status, "measured_gain_dbi": actual_gain},
            "efficiency": {"goal": request.optimization_targets.primary.efficiency, "status": efficiency_status, "measured_efficiency": actual_efficiency},
        },
        "secondary": {
            "zin": {"target": request.optimization_targets.secondary.zin_target, "status": "unknown" if actual_zin is None else "observed", "measured": actual_zin},
            "axial_ratio": {"target": request.optimization_targets.secondary.axial_ratio, "status": "unknown" if actual_axial_ratio is None else "observed", "measured_db": actual_axial_ratio},
            "sll": {"goal": request.optimization_targets.secondary.sll, "status": "unknown" if actual_sll is None else "observed", "measured_db": actual_sll},
            "front_to_back": {"goal": request.optimization_targets.secondary.front_to_back, "status": "unknown" if actual_front_to_back is None else "observed", "measured_db": actual_front_to_back},
        },
    }
