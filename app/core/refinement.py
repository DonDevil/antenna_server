from __future__ import annotations

from typing import Any

from app.core.schemas import AnnPrediction, DimensionPrediction, OptimizeRequest
from app.planning.geometry_guardrails import apply_geometry_guardrails
from config import BOUNDS


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _shift_offset_away_from_center(value: float, delta_mm: float) -> float:
    if value < 0.0:
        return value - abs(delta_mm)
    if value > 0.0:
        return value + abs(delta_mm)
    return -abs(delta_mm)


def _normalize_patch_shape_dimensions(dimensions: dict[str, Any], patch_shape: str) -> None:
    shape = patch_shape.strip().lower()
    if shape == "circular":
        radius = dimensions.get("patch_radius_mm")
        if radius is None:
            radius = max(float(dimensions.get("patch_length_mm", 1.0)), float(dimensions.get("patch_width_mm", 1.0))) / 2.0
        radius_f = max(0.01, float(radius))
        dimensions["patch_radius_mm"] = radius_f
        dimensions["patch_length_mm"] = 2.0 * radius_f
        dimensions["patch_width_mm"] = 2.0 * radius_f
    else:
        width = max(0.01, float(dimensions.get("patch_width_mm", 1.0)))
        dimensions["patch_radius_mm"] = width / 2.0


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
    actual_return_loss = float(feedback.get("actual_return_loss_db", -10.0))
    actual_vswr = float(feedback["actual_vswr"])
    actual_gain = float(feedback["actual_gain_dbi"])

    freq_error_mhz = (actual_f - target_f) * 1000.0
    bandwidth_gap_mhz = target_bw - actual_bw
    return_loss_gap_db = actual_return_loss - float(acceptance.minimum_return_loss_db)
    vswr_gap = actual_vswr - float(acceptance.maximum_vswr)
    gain_gap = float(acceptance.minimum_gain_dbi) - actual_gain

    accepted = (
        abs(freq_error_mhz) <= float(acceptance.center_tolerance_mhz)
        and actual_bw >= max(target_bw, float(acceptance.minimum_bandwidth_mhz))
        and actual_return_loss <= float(acceptance.minimum_return_loss_db)
        and actual_vswr <= float(acceptance.maximum_vswr)
        and actual_gain >= float(acceptance.minimum_gain_dbi)
    )

    return {
        "accepted": bool(accepted),
        "freq_error_mhz": freq_error_mhz,
        "bandwidth_gap_mhz": bandwidth_gap_mhz,
        "return_loss_gap_db": return_loss_gap_db,
        "vswr_gap": vswr_gap,
        "gain_gap": gain_gap,
        "target": {
            "frequency_ghz": target_f,
            "bandwidth_mhz": target_bw,
        },
        "actual": {
            "frequency_ghz": actual_f,
            "bandwidth_mhz": actual_bw,
            "return_loss_db": actual_return_loss,
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

    target_frequency_mhz = float(request.target_spec.frequency_ghz) * 1000.0
    target_bandwidth_mhz = max(float(request.target_spec.bandwidth_mhz), 1.0)
    patch_shape = str(current.patch_shape or request.target_spec.patch_shape or "rectangular")

    freq_error_mhz = float(evaluation["freq_error_mhz"])
    bandwidth_gap_mhz = float(evaluation["bandwidth_gap_mhz"])
    return_loss_gap_db = float(evaluation.get("return_loss_gap_db", 0.0))
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
        freq_ratio = _clamp(freq_error_mhz / max(target_frequency_mhz, 1.0), -0.12, 0.12)
        resonance_scale = 1.0 + (0.60 * freq_ratio)
        feed_length_scale = 1.0 + (0.18 * freq_ratio)

        if patch_shape.strip().lower() == "circular":
            current_radius = float(d.get("patch_radius_mm") or (float(d.get("patch_width_mm", 1.0)) / 2.0))
            d["patch_radius_mm"] = max(0.01, current_radius * resonance_scale)
        else:
            d["patch_length_mm"] *= resonance_scale
        d["feed_length_mm"] *= max(0.90, feed_length_scale)

        if bandwidth_gap_mhz > 0:
            bw_ratio = _clamp(bandwidth_gap_mhz / target_bandwidth_mhz, 0.0, 1.0)
            d["substrate_height_mm"] *= 1.0 + 0.05 + (0.12 * bw_ratio)
            d["feed_width_mm"] *= 1.0 + 0.03 + (0.08 * bw_ratio)
            if patch_shape.strip().lower() == "circular":
                d["patch_radius_mm"] = float(d.get("patch_radius_mm") or 1.0) * (1.0 + 0.01 + (0.04 * bw_ratio))
            else:
                d["patch_width_mm"] *= 1.0 + 0.02 + (0.05 * bw_ratio)

        matching_severity = _clamp(max(vswr_gap / 2.0, return_loss_gap_db / 10.0), 0.0, 1.0)
        if matching_severity > 0.0:
            d["feed_width_mm"] *= 1.0 + 0.04 + (0.07 * matching_severity)
            d["feed_length_mm"] *= 1.0 + (0.02 * matching_severity)
            d["feed_offset_y_mm"] = _shift_offset_away_from_center(float(d["feed_offset_y_mm"]), 0.20 + (0.55 * matching_severity))
            if patch_shape.strip().lower() == "circular":
                d["feed_offset_x_mm"] = _clamp(float(d["feed_offset_x_mm"]) * 0.95, *BOUNDS.feed_offset_x_mm)

        if gain_gap > 0:
            gain_ratio = _clamp(gain_gap / 5.0, 0.0, 1.0)
            d["substrate_length_mm"] *= 1.0 + 0.02 + (0.05 * gain_ratio)
            d["substrate_width_mm"] *= 1.0 + 0.02 + (0.05 * gain_ratio)
            if patch_shape.strip().lower() == "circular":
                d["patch_radius_mm"] = float(d.get("patch_radius_mm") or 1.0) * (1.0 + 0.01 + (0.03 * gain_ratio))
            else:
                d["patch_width_mm"] *= 1.0 + 0.01 + (0.04 * gain_ratio)

    _normalize_patch_shape_dimensions(d, patch_shape)

    # Hard geometry guardrail: clamp each dimension to the max allowed
    # absolute delta for the selected action before executing.
    original_d = current.dimensions.model_dump()
    d, _clamped = apply_geometry_guardrails(original_d, d, action_name)

    # Clamp with request constraints when available, otherwise global defaults.
    d["patch_length_mm"] = _clamp(d["patch_length_mm"], *_constraint_range(request, "patch_length_mm", BOUNDS.patch_length_mm))
    d["patch_width_mm"] = _clamp(d["patch_width_mm"], *_constraint_range(request, "patch_width_mm", BOUNDS.patch_width_mm))
    if d.get("patch_radius_mm") is not None:
        d["patch_radius_mm"] = _clamp(float(d["patch_radius_mm"]), 0.01, BOUNDS.patch_width_mm[1] / 2.0)
    d["patch_height_mm"] = _clamp(d["patch_height_mm"], *BOUNDS.patch_height_mm)
    d["substrate_length_mm"] = _clamp(d["substrate_length_mm"], *_constraint_range(request, "substrate_length_mm", BOUNDS.substrate_length_mm))
    d["substrate_width_mm"] = _clamp(d["substrate_width_mm"], *_constraint_range(request, "substrate_width_mm", BOUNDS.substrate_width_mm))
    d["substrate_height_mm"] = _clamp(d["substrate_height_mm"], *_constraint_range(request, "substrate_height_mm", BOUNDS.substrate_height_mm))
    d["feed_length_mm"] = _clamp(d["feed_length_mm"], *_constraint_range(request, "feed_length_mm", BOUNDS.feed_length_mm))
    d["feed_width_mm"] = _clamp(d["feed_width_mm"], *_constraint_range(request, "feed_width_mm", BOUNDS.feed_width_mm))
    d["feed_offset_x_mm"] = _clamp(d["feed_offset_x_mm"], *BOUNDS.feed_offset_x_mm)
    d["feed_offset_y_mm"] = _clamp(d["feed_offset_y_mm"], *BOUNDS.feed_offset_y_mm)

    _normalize_patch_shape_dimensions(d, patch_shape)

    next_conf = max(0.2, float(current.confidence) - (0.03 * max(1, next_iteration_index)))
    return AnnPrediction(
        ann_model_version=f"{current.ann_model_version}-refined",
        confidence=next_conf,
        dimensions=DimensionPrediction(**d),
        recipe_name=current.recipe_name,
        patch_shape=patch_shape,
        optimizer_hint=f"refined:{action_name}",
    )
