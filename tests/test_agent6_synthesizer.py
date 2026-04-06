"""
Tests — A6 Synthesizer (test_agent6_synthesizer.py)
LLM (summary generation) is mocked; scoring is pure math.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.a6_synthesizer import a6_synthesizer, _quality_label, W_SENTIMENT, W_DISCOURSE, W_NOISE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_STATE = {
    "sentiment": {"sentiment_label": "positive", "sentiment_score": 72.0, "rationale": "Good"},
    "discourse": {"discourse_score": 65.0, "rationale": "Decent"},
    "noise": {"noise_score": 80.0, "rationale": "Clean"},
}

# Score_Global = 0.35*72 + 0.40*65 + 0.25*80 = 25.2 + 26.0 + 20.0 = 71.2
_EXPECTED_SCORE = round(0.35 * 72.0 + 0.40 * 65.0 + 0.25 * 80.0, 2)


# ---------------------------------------------------------------------------
# Score computation (no LLM)
# ---------------------------------------------------------------------------

class TestA6ScoringMath:
    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_score_global_computed_correctly(self, _):
        result = a6_synthesizer(_FULL_STATE)
        assert result["score_global"] == pytest.approx(_EXPECTED_SCORE)

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_score_global_in_state(self, _):
        result = a6_synthesizer(_FULL_STATE)
        assert "score_global" in result

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_synthesis_key_present(self, _):
        result = a6_synthesizer(_FULL_STATE)
        assert "synthesis" in result

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_quality_label_in_synthesis(self, _):
        result = a6_synthesizer(_FULL_STATE)
        assert "quality_label" in result["synthesis"]

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_correct_quality_label(self, _):
        result = a6_synthesizer(_FULL_STATE)
        # 71.2 → "Bon" (50-74)
        assert result["synthesis"]["quality_label"] == "Bon"

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_all_zeros_gives_faible(self, _):
        state = {
            "sentiment": {"sentiment_score": 0.0},
            "discourse": {"discourse_score": 0.0},
            "noise": {"noise_score": 0.0},
        }
        result = a6_synthesizer(state)
        assert result["synthesis"]["quality_label"] == "Faible"

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_all_100_gives_excellent(self, _):
        state = {
            "sentiment": {"sentiment_score": 100.0},
            "discourse": {"discourse_score": 100.0},
            "noise": {"noise_score": 100.0},
        }
        result = a6_synthesizer(state)
        assert result["synthesis"]["quality_label"] == "Excellent"
        assert result["score_global"] == pytest.approx(100.0)

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_missing_scores_use_defaults(self, _):
        """Agents not yet run → default values used."""
        result = a6_synthesizer({})
        assert "score_global" in result
        assert result["score_global"] > 0  # defaults: s=50, d=50, n=70


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

class TestA6SummaryGeneration:
    @patch("agents.a6_synthesizer.get_llm")
    def test_summary_populated_when_llm_available(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="This is a quality summary.")
        mock_get_llm.return_value = mock_llm

        result = a6_synthesizer(_FULL_STATE)
        assert result["synthesis"]["summary"] == "This is a quality summary."

    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    def test_summary_empty_when_llm_unavailable(self, _):
        result = a6_synthesizer(_FULL_STATE)
        assert result["synthesis"]["summary"] == ""

    @patch("agents.a6_synthesizer.get_llm")
    def test_summary_error_does_not_raise(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")
        mock_get_llm.return_value = mock_llm

        result = a6_synthesizer(_FULL_STATE)
        # Score still computed correctly despite LLM error
        assert "score_global" in result
        assert result["synthesis"]["summary"] == ""
