"""Unit tests: geometry guardrails and Ollama health-check fast fallback."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.planning.geometry_guardrails import apply_geometry_guardrails


# ── guardrail tests ──────────────────────────────────────────────────────────

def _base_dims() -> dict[str, float]:
    return {
        "patch_length_mm": 30.0,
        "patch_width_mm": 28.0,
        "patch_height_mm": 0.5,
        "substrate_length_mm": 40.0,
        "substrate_width_mm": 38.0,
        "substrate_height_mm": 1.6,
        "feed_length_mm": 10.0,
        "feed_width_mm": 3.0,
        "feed_offset_x_mm": 0.0,
        "feed_offset_y_mm": 2.0,
    }


def test_guardrail_clamps_oversized_patch_length_delta() -> None:
    """increase_patch_length allows max 6 mm change on patch_length_mm."""
    before = _base_dims()
    after = dict(before)
    after["patch_length_mm"] = before["patch_length_mm"] + 20.0  # 20 mm — way over limit

    result, clamped = apply_geometry_guardrails(before, after, "increase_patch_length")

    assert "patch_length_mm" in clamped
    assert result["patch_length_mm"] == pytest.approx(before["patch_length_mm"] + 6.0)


def test_guardrail_clamps_undersized_patch_length_delta() -> None:
    """Negative oversized delta (decrease) is also clamped symmetrically."""
    before = _base_dims()
    after = dict(before)
    after["patch_length_mm"] = before["patch_length_mm"] - 15.0

    result, clamped = apply_geometry_guardrails(before, after, "decrease_patch_length")

    assert "patch_length_mm" in clamped
    assert result["patch_length_mm"] == pytest.approx(before["patch_length_mm"] - 6.0)


def test_guardrail_passes_small_delta_unchanged() -> None:
    """A delta within the limit must not be touched."""
    before = _base_dims()
    after = dict(before)
    after["patch_length_mm"] = before["patch_length_mm"] + 2.0  # well within 6 mm limit

    result, clamped = apply_geometry_guardrails(before, after, "increase_patch_length")

    assert "patch_length_mm" not in clamped
    assert result["patch_length_mm"] == pytest.approx(after["patch_length_mm"])


def test_guardrail_action_specific_substrate_height_limit() -> None:
    """increase_substrate_height allows max 0.5 mm on substrate_height_mm."""
    before = _base_dims()
    after = dict(before)
    after["substrate_height_mm"] = before["substrate_height_mm"] + 1.2  # over 0.5 limit

    result, clamped = apply_geometry_guardrails(before, after, "increase_substrate_height")

    assert "substrate_height_mm" in clamped
    assert result["substrate_height_mm"] == pytest.approx(before["substrate_height_mm"] + 0.5)


def test_guardrail_non_primary_dim_uses_default_limit() -> None:
    """For increase_patch_length, feed_width_mm falls back to _default=1.0 mm."""
    before = _base_dims()
    after = dict(before)
    after["feed_width_mm"] = before["feed_width_mm"] + 5.0  # over _default=1.0

    result, clamped = apply_geometry_guardrails(before, after, "increase_patch_length")

    assert "feed_width_mm" in clamped
    assert result["feed_width_mm"] == pytest.approx(before["feed_width_mm"] + 1.0)


def test_guardrail_unknown_action_uses_global_default() -> None:
    """An unrecognised action name falls back to global default (3.0 mm)."""
    before = _base_dims()
    after = dict(before)
    after["patch_length_mm"] = before["patch_length_mm"] + 10.0  # over global default 3.0

    result, clamped = apply_geometry_guardrails(before, after, "completely_unknown_action")

    assert "patch_length_mm" in clamped
    assert result["patch_length_mm"] == pytest.approx(before["patch_length_mm"] + 3.0)


def test_guardrail_no_clamping_returns_empty_list() -> None:
    """When all deltas are within budget, clamped_fields is empty."""
    before = _base_dims()
    after = dict(before)  # no changes at all

    _, clamped = apply_geometry_guardrails(before, after, "generic_refinement")

    assert clamped == []


# ── Ollama health-check fallback tests ───────────────────────────────────────

def test_ollama_health_returns_false_on_connection_error() -> None:
    """check_ollama_health must return False when Ollama is unreachable."""
    import httpx
    from app.llm.ollama_client import check_ollama_health

    with patch("app.llm.ollama_client.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.side_effect = httpx.ConnectError("refused")
        result = check_ollama_health(timeout_sec=1)

    assert result is False


def test_ollama_health_returns_true_on_200() -> None:
    """check_ollama_health must return True when the probe gets HTTP 200."""
    from unittest.mock import MagicMock

    from app.llm.ollama_client import check_ollama_health

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp

    with patch("app.llm.ollama_client.httpx.Client", return_value=mock_client):
        result = check_ollama_health(timeout_sec=1)

    assert result is True


def test_dynamic_planner_falls_back_when_ollama_unavailable() -> None:
    """plan_refinement_strategy must not call LLM and must set reason=ollama_unavailable
    when check_ollama_health returns False."""
    from app.planning.dynamic_planner import plan_refinement_strategy

    features = {
        "freq_error_mhz": 120.0,
        "bandwidth_gap_mhz": 0.0,
        "vswr_excess": 0.0,
        "gain_deficit": 0.0,
        "severity": 0.9,
        "needs_frequency_shift": True,
        "needs_bandwidth_boost": False,
        "needs_matching_improvement": False,
        "needs_gain_boost": False,
    }
    session: dict = {
        "policy_runtime": {
            "llm_enabled_for_refinement": True,
            "llm_refinement_confidence_threshold": 1.01,  # never reachable → budget gate stays open
            "max_llm_calls_per_session": 10,
            "max_llm_calls_per_iteration": 10,
            "llm_calls_total": 0,
            "llm_calls_by_iteration": {},
        },
        "history": [],
    }

    with patch("app.planning.dynamic_planner.check_ollama_health", return_value=False):
        result = plan_refinement_strategy(session=session, features=features, iteration_index=1)

    assert result["llm_used"] is False
    assert result["llm_reason"] == "ollama_unavailable_fast_fallback"
    assert result["decision_source"] == "deterministic_rule"
