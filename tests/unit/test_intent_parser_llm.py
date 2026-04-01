"""Unit tests for LLM-powered intent parsing."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.llm.intent_parser import summarize_user_intent


def test_intent_parser_regex_fallback_extracts_ghz() -> None:
    """Regex fallback extracts GHz and MHz even when LLM is disabled."""
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = False
        result = summarize_user_intent("I need a 2.4 GHz antenna with 50 MHz bandwidth")
    assert result["parsed_frequency_ghz"] == pytest.approx(2.4)
    assert result["parsed_bandwidth_mhz"] == pytest.approx(50.0)
    assert result["missing_fields"] == []
    assert result["llm_intent"] is None


def test_intent_parser_llm_override_when_enabled() -> None:
    """When LLM is enabled and returns a different parse, it overrides regex."""
    llm_response = {
        "frequency_ghz": 5.8,
        "bandwidth_mhz": 200.0,
    }
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = True
        with patch("app.llm.intent_parser.generate_json", return_value=llm_response):
            result = summarize_user_intent(
                "Design me something working at five point eight gigahertz with lots of bandwidth"
            )

    assert result["parsed_frequency_ghz"] == pytest.approx(5.8)
    assert result["parsed_bandwidth_mhz"] == pytest.approx(200.0)
    assert result["llm_intent"]["source"] == "llm"
    assert result["llm_intent"]["llm_response"] == llm_response


def test_intent_parser_llm_fallback_to_regex_on_failure() -> None:
    """If LLM returns invalid response, parser falls back to regex."""
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = True
        with patch("app.llm.intent_parser.generate_json", return_value=None):
            result = summarize_user_intent("Design a 2.45 GHz antenna with 80 MHz BW")

    # Should get regex fallback values
    assert result["parsed_frequency_ghz"] == pytest.approx(2.45)
    assert result["parsed_bandwidth_mhz"] == pytest.approx(80.0)
    # But llm_intent should be None because LLM failed
    assert result["llm_intent"] is None


def test_intent_parser_llm_partial_extraction() -> None:
    """LLM can provide one spec even if regex got it; both are merged."""
    llm_response = {
        "frequency_ghz": 2.4,
        "bandwidth_mhz": None,  # LLM didn't understand bandwidth
    }
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = True
        with patch("app.llm.intent_parser.generate_json", return_value=llm_response):
            result = summarize_user_intent(
                "2.4 GHz (don't understand bandwidth specs in text)"
            )

    # LLM provided freq, use that
    assert result["parsed_frequency_ghz"] == pytest.approx(2.4)
    # Bandwidth from regex (or None if not in text)
    assert result["parsed_bandwidth_mhz"] is None
    assert result["llm_intent"]["source"] == "llm"


def test_intent_parser_missing_fields_reported() -> None:
    """missing_fields lists specs the parser couldn't extract."""
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = False
        result = summarize_user_intent("Just design something, I don't care about specs")
    assert result["parsed_frequency_ghz"] is None
    assert result["parsed_bandwidth_mhz"] is None
    assert set(result["missing_fields"]) == {"frequency_ghz", "bandwidth_mhz"}


def test_intent_parser_with_llm_disabled_ignores_llm_config() -> None:
    """Even if LLM setting exists, disabled mode uses regex only."""
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = False
        with patch("app.llm.intent_parser.generate_json") as mock_gen:
            result = summarize_user_intent("Design 10 GHz 150 MHz antenna")
            # LLM should never be called
            mock_gen.assert_not_called()

    assert result["parsed_frequency_ghz"] == pytest.approx(10.0)
    assert result["parsed_bandwidth_mhz"] == pytest.approx(150.0)
    assert result["llm_intent"] is None


def test_intent_parser_detects_rectangular_patch_as_microstrip() -> None:
    """Text with 'rectangular patch' should map to microstrip_patch, not amc_patch."""
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = False
        result = summarize_user_intent("Design a rectangular patch antenna at 2.45 GHz with 80 MHz bandwidth")

    assert result["parsed_antenna_family"] == "microstrip_patch"


def test_intent_parser_llm_can_set_family() -> None:
    """When enabled, LLM family value should be respected if valid."""
    llm_response = {
        "frequency_ghz": 2.45,
        "bandwidth_mhz": 80.0,
        "antenna_family": "microstrip_patch",
    }
    with patch("app.llm.intent_parser.PLANNER_SETTINGS") as mock_settings:
        mock_settings.llm_enabled_for_intent = True
        with patch("app.llm.intent_parser.generate_json", return_value=llm_response):
            result = summarize_user_intent("build me a rectangular patch")

    assert result["parsed_antenna_family"] == "microstrip_patch"
