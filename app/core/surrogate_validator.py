from __future__ import annotations

from typing import Any

from app.core.schemas import AnnPrediction, OptimizeRequest
from config import BOUNDS


def _clamp_score(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def validate_with_surrogate(request: OptimizeRequest, ann: AnnPrediction) -> dict[str, Any]:
    """Estimate expected RF behavior from predicted geometry and derive confidence.

    This is a lightweight forward surrogate heuristic used as a sanity gate before
    command package generation. It does not replace CST simulation feedback.
    """
    target_freq = float(request.target_spec.frequency_ghz)
    target_bw = float(request.target_spec.bandwidth_mhz)

    dims = ann.dimensions
    patch_length = max(1e-6, float(dims.patch_length_mm))
    substrate_height = max(1e-6, float(dims.substrate_height_mm))

    # Approximate center frequency for a patch-like radiator from effective length.
    estimated_center_freq_ghz = 74.0 / patch_length
    # Approximate bandwidth trend from substrate thickness to patch length ratio.
    estimated_bandwidth_mhz = 600.0 * (substrate_height / patch_length)

    freq_abs_error = abs(estimated_center_freq_ghz - target_freq)
    bw_abs_error = abs(estimated_bandwidth_mhz - target_bw)

    freq_tolerance = max(target_freq * 0.35, 0.2)
    bw_tolerance = max(target_bw * 0.8, 20.0)

    freq_score = _clamp_score(1.0 - (freq_abs_error / freq_tolerance))
    bw_score = _clamp_score(1.0 - (bw_abs_error / bw_tolerance))

    # Domain-support score: confidence decays near limits of known training range.
    freq_min, freq_max = BOUNDS.frequency_ghz
    bw_min, bw_max = BOUNDS.bandwidth_mhz
    freq_margin = min(target_freq - freq_min, freq_max - target_freq) / ((freq_max - freq_min) / 2.0)
    bw_margin = min(target_bw - bw_min, bw_max - target_bw) / ((bw_max - bw_min) / 2.0)
    domain_score = _clamp_score(min(freq_margin, bw_margin))

    final_confidence = _clamp_score(
        (float(ann.confidence) * 0.40)
        + (freq_score * 0.35)
        + (bw_score * 0.15)
        + (domain_score * 0.10)
    )

    decision_reason = "surrogate_confidence_sufficient"
    if final_confidence < 0.45:
        decision_reason = "surrogate_confidence_below_threshold"

    return {
        "surrogate_model_version": "heuristic_forward.v1",
        "confidence": final_confidence,
        "threshold": 0.45,
        "accepted": bool(final_confidence >= 0.45),
        "decision_reason": decision_reason,
        "estimated_metrics": {
            "center_frequency_ghz": estimated_center_freq_ghz,
            "bandwidth_mhz": estimated_bandwidth_mhz,
        },
        "target_metrics": {
            "center_frequency_ghz": target_freq,
            "bandwidth_mhz": target_bw,
        },
        "residual": {
            "center_frequency_abs_error_ghz": freq_abs_error,
            "bandwidth_abs_error_mhz": bw_abs_error,
        },
        "component_scores": {
            "ann_score": float(ann.confidence),
            "freq_score": freq_score,
            "bandwidth_score": bw_score,
            "domain_support_score": domain_score,
        },
    }
