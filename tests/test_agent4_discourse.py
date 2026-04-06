"""
Tests — A4 Discourse Analyser (test_agent4_discourse.py)
LLM calls are mocked — no network or GPU required.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.a4_discourse import a4_discourse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(informativeness=0.7, argumentation=0.6, constructiveness=0.8,
                        hq_indices=None, rationale="Test"):
    content = json.dumps({
        "informativeness": informativeness,
        "argumentation": argumentation,
        "constructiveness": constructiveness,
        "high_quality_indices": hq_indices or [],
        "rationale": rationale,
    })
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


_CLEANED = [
    {"text": "Comment 0", "cleaned_text": "comment 0"},
    {"text": "Comment 1", "cleaned_text": "comment 1"},
    {"text": "Comment 2", "cleaned_text": "comment 2"},
]

_STATE = {"cleaned_comments": _CLEANED}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestA4DiscourseHappyPath:
    @patch("agents.a4_discourse.get_llm")
    def test_returns_discourse_key(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response()
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        assert "discourse" in result

    @patch("agents.a4_discourse.get_llm")
    def test_three_dimensions_present(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(0.7, 0.6, 0.8)
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        d = result["discourse"]
        assert "informativeness" in d
        assert "argumentation" in d
        assert "constructiveness" in d

    @patch("agents.a4_discourse.get_llm")
    def test_discourse_score_is_average(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(0.6, 0.9, 0.6)
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        d = result["discourse"]
        expected_avg = (60.0 + 90.0 + 60.0) / 3
        assert d["discourse_score"] == pytest.approx(expected_avg, rel=1e-3)

    @patch("agents.a4_discourse.get_llm")
    def test_dimensions_scaled_to_100(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(0.75, 0.50, 0.85)
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        d = result["discourse"]
        assert d["informativeness"] == pytest.approx(75.0)
        assert d["argumentation"] == pytest.approx(50.0)
        assert d["constructiveness"] == pytest.approx(85.0)

    @patch("agents.a4_discourse.get_llm")
    def test_high_quality_indices_passed_through(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(hq_indices=[0, 2])
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        assert result["discourse"]["high_quality_indices"] == [0, 2]

    @patch("agents.a4_discourse.get_llm")
    def test_empty_high_quality_indices_allowed(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(hq_indices=[])
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        assert result["discourse"]["high_quality_indices"] == []


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestA4DiscourseFallback:
    @patch("agents.a4_discourse.get_llm", return_value=None)
    def test_llm_unavailable_returns_default(self, _):
        result = a4_discourse(_STATE)
        d = result["discourse"]
        assert d["discourse_score"] == 50.0
        assert d["high_quality_indices"] == []

    @patch("agents.a4_discourse.get_llm")
    def test_parse_error_returns_default(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="{{invalid")
        mock_get_llm.return_value = mock_llm

        result = a4_discourse(_STATE)
        assert "discourse" in result
        assert result["discourse"]["discourse_score"] == 50.0
