from __future__ import annotations

from typing import Any

from app.core.objectives import evaluate_objective_state
from app.core.schemas import OptimizeRequest


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def derive_feedback_features(
    request: OptimizeRequest,
    feedback: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    target_f = float(request.target_spec.frequency_ghz)
    target_bw = float(request.target_spec.bandwidth_mhz)

    actual_f = float(feedback["actual_center_frequency_ghz"])
    actual_bw = float(feedback["actual_bandwidth_mhz"])
    actual_return_loss = float(feedback.get("actual_return_loss_db", -10.0))
    actual_vswr = float(feedback.get("actual_vswr", 99.0))
    actual_gain = float(feedback.get("actual_gain_dbi", -99.0))

    freq_error_mhz = float(evaluation["freq_error_mhz"])
    bandwidth_gap_mhz = float(evaluation["bandwidth_gap_mhz"])
    return_loss_gap_db = float(evaluation.get("return_loss_gap_db", 0.0))
    vswr_gap = float(evaluation["vswr_gap"])
    gain_gap = float(evaluation["gain_gap"])

    abs_freq_error_mhz = abs(freq_error_mhz)
    freq_error_ratio = abs_freq_error_mhz / max(target_f * 1000.0, 1.0)
    bandwidth_shortfall_ratio = max(0.0, bandwidth_gap_mhz) / max(target_bw, 1.0)

    return_loss_deficit_db = max(0.0, return_loss_gap_db)
    vswr_excess = max(0.0, vswr_gap)
    gain_deficit = max(0.0, gain_gap)

    severity = _clip(
        (0.30 * _clip(freq_error_ratio * 10.0, 0.0, 1.0))
        + (0.20 * _clip(bandwidth_shortfall_ratio * 2.0, 0.0, 1.0))
        + (0.25 * _clip(vswr_excess / 2.0, 0.0, 1.0))
        + (0.10 * _clip(return_loss_deficit_db / 10.0, 0.0, 1.0))
        + (0.15 * _clip(gain_deficit / 5.0, 0.0, 1.0)),
        0.0,
        1.0,
    )

    if abs_freq_error_mhz <= 5.0:
        frequency_state = "on_target"
    elif freq_error_mhz > 0.0:
        frequency_state = "too_high"
    else:
        frequency_state = "too_low"

    matching_state = "matched"
    if vswr_gap > 0.3 or return_loss_deficit_db > 3.0:
        matching_state = "poor"
    elif vswr_gap > 0.0 or return_loss_deficit_db > 0.0:
        matching_state = "off_target"

    objective_state = evaluate_objective_state(request, feedback, evaluation)

    return {
        "accepted": bool(evaluation["accepted"]),
        "target_frequency_ghz": target_f,
        "target_bandwidth_mhz": target_bw,
        "actual_frequency_ghz": actual_f,
        "actual_bandwidth_mhz": actual_bw,
        "actual_return_loss_db": actual_return_loss,
        "actual_vswr": actual_vswr,
        "actual_gain_dbi": actual_gain,
        "freq_error_mhz": freq_error_mhz,
        "abs_freq_error_mhz": abs_freq_error_mhz,
        "bandwidth_gap_mhz": bandwidth_gap_mhz,
        "return_loss_gap_db": return_loss_gap_db,
        "vswr_gap": vswr_gap,
        "gain_gap": gain_gap,
        "freq_error_ratio": freq_error_ratio,
        "bandwidth_shortfall_ratio": bandwidth_shortfall_ratio,
        "return_loss_deficit_db": return_loss_deficit_db,
        "vswr_excess": vswr_excess,
        "gain_deficit": gain_deficit,
        "severity": severity,
        "frequency_state": frequency_state,
        "bandwidth_state": "sufficient" if bandwidth_gap_mhz <= 0.0 else "narrow",
        "matching_state": matching_state,
        "gain_state": "sufficient" if gain_gap <= 0.0 else "low",
        "objective_state": objective_state,
        "needs_frequency_shift": abs_freq_error_mhz > 5.0,
        "needs_bandwidth_boost": bandwidth_gap_mhz > 0.0,
        "needs_matching_fix": vswr_gap > 0.0 or return_loss_deficit_db > 0.0,
        "needs_gain_boost": gain_gap > 0.0,
        "frequency_direction": "increase" if freq_error_mhz < 0 else "decrease",
    }