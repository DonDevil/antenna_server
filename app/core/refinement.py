from __future__ import annotations

from typing import Any

from app.core.schemas import AnnPrediction, DimensionPrediction, OptimizeRequest
from app.planning.geometry_guardrails import apply_geometry_guardrails
from config import BOUNDS


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _constraint_range(request: OptimizeRequest, field_name: str, default_range: tuple[float, float]) -> tuple[float, float]:
    maybe = getattr(request.design_constraints, field_name, None)
    if maybe is None:
        return default_range
    return float(maybe.min), float(maybe.max)


def evaluate_acceptance(request: OptimizeRequest, feedback: dict[str, Any]) -> dict[str, Any]:
    target_f = float(request.target_spec.frequency_ghz)
    target_bw = float(request.target_spec.bandwidth_mhz)
    acceptance = request.optimization_policy.acceptance

    actual_f = float(feedback["actual_center_frequency_ghz"])
    actual_bw = float(feedback["actual_bandwidth_mhz"])
    actual_vswr = float(feedback["actual_vswr"])
    actual_gain = float(feedback["actual_gain_dbi"])

    freq_error_mhz = (actual_f - target_f) * 1000.0
    bandwidth_gap_mhz = target_bw - actual_bw
    vswr_gap = actual_vswr - float(acceptance.maximum_vswr)
    gain_gap = float(acceptance.minimum_gain_dbi) - actual_gain

    accepted = (
        abs(freq_error_mhz) <= float(acceptance.center_tolerance_mhz)
        and actual_bw >= max(target_bw, float(acceptance.minimum_bandwidth_mhz))
        and actual_vswr <= float(acceptance.maximum_vswr)
        and actual_gain >= float(acceptance.minimum_gain_dbi)
    )

    return {
        "accepted": bool(accepted),
        "freq_error_mhz": freq_error_mhz,
        "bandwidth_gap_mhz": bandwidth_gap_mhz,
        "vswr_gap": vswr_gap,
        "gain_gap": gain_gap,
        "target": {
            "frequency_ghz": target_f,
            "bandwidth_mhz": target_bw,
        },
        "actual": {
            "frequency_ghz": actual_f,
            "bandwidth_mhz": actual_bw,
            "vswr": actual_vswr,
            "gain_dbi": actual_gain,
        },
    }


def refine_prediction(request: OptimizeRequest, current: AnnPrediction, evaluation: dict[str, Any], next_iteration_index: int) -> AnnPrediction:
    return refine_prediction_with_strategy(
        request,
        current,
        evaluation,
        next_iteration_index=next_iteration_index,
        strategy=None,
        action_name="generic_refinement",
    )


def refine_prediction_with_strategy(
    request: OptimizeRequest,
    current: AnnPrediction,
    evaluation: dict[str, Any],
    next_iteration_index: int,
    strategy: dict[str, Any] | None,
    action_name: str = "generic_refinement",
) -> AnnPrediction:
    d = current.dimensions.model_dump()

    freq_error_mhz = float(evaluation["freq_error_mhz"])
    bandwidth_gap_mhz = float(evaluation["bandwidth_gap_mhz"])
    vswr_gap = float(evaluation["vswr_gap"])
    gain_gap = float(evaluation["gain_gap"])

    if isinstance(strategy, dict) and strategy:
        scale = strategy.get("scale", {})
        offset = strategy.get("offset", {})
        if isinstance(scale, dict):
            for key, factor in scale.items():
                if key in d:
                    d[key] = float(d[key]) * float(factor)
        if isinstance(offset, dict):
            for key, delta in offset.items():
                if key in d:
                    d[key] = float(d[key]) + float(delta)
    else:
        # If frequency is too high, increase patch length. If low, decrease it.
        if freq_error_mhz > 0:
            d["patch_length_mm"] *= 1.02
        elif freq_error_mhz < 0:
            d["patch_length_mm"] *= 0.98

        # If bandwidth is below target, increase substrate height and feed width.
        if bandwidth_gap_mhz > 0:
            d["substrate_height_mm"] *= 1.05
            d["feed_width_mm"] *= 1.05

        # If VSWR is above threshold, nudge feed geometry.
        if vswr_gap > 0:
            d["feed_width_mm"] *= 1.03
            d["feed_offset_y_mm"] *= 0.95

        # If gain is below threshold, increase patch width and substrate size.
        if gain_gap > 0:
            d["patch_width_mm"] *= 1.02
            d["substrate_length_mm"] *= 1.02
            d["substrate_width_mm"] *= 1.02

    # Hard geometry guardrail: clamp each dimension to the max allowed
    # absolute delta for the selected action before executing.
    original_d = current.dimensions.model_dump()
    d, _clamped = apply_geometry_guardrails(original_d, d, action_name)

    # Clamp with request constraints when available, otherwise global defaults.
    d["patch_length_mm"] = _clamp(d["patch_length_mm"], *_constraint_range(request, "patch_length_mm", BOUNDS.patch_length_mm))
    d["patch_width_mm"] = _clamp(d["patch_width_mm"], *_constraint_range(request, "patch_width_mm", BOUNDS.patch_width_mm))
    d["patch_height_mm"] = _clamp(d["patch_height_mm"], *BOUNDS.patch_height_mm)
    d["substrate_length_mm"] = _clamp(d["substrate_length_mm"], *_constraint_range(request, "substrate_length_mm", BOUNDS.substrate_length_mm))
    d["substrate_width_mm"] = _clamp(d["substrate_width_mm"], *_constraint_range(request, "substrate_width_mm", BOUNDS.substrate_width_mm))
    d["substrate_height_mm"] = _clamp(d["substrate_height_mm"], *_constraint_range(request, "substrate_height_mm", BOUNDS.substrate_height_mm))
    d["feed_length_mm"] = _clamp(d["feed_length_mm"], *_constraint_range(request, "feed_length_mm", BOUNDS.feed_length_mm))
    d["feed_width_mm"] = _clamp(d["feed_width_mm"], *_constraint_range(request, "feed_width_mm", BOUNDS.feed_width_mm))
    d["feed_offset_x_mm"] = _clamp(d["feed_offset_x_mm"], *BOUNDS.feed_offset_x_mm)
    d["feed_offset_y_mm"] = _clamp(d["feed_offset_y_mm"], *BOUNDS.feed_offset_y_mm)

    next_conf = max(0.2, float(current.confidence) - (0.03 * max(1, next_iteration_index)))
    return AnnPrediction(
        ann_model_version=f"{current.ann_model_version}-refined",
        confidence=next_conf,
        dimensions=DimensionPrediction(**d),
    )
