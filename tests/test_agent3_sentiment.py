"""
Tests — A3 Sentiment Analyser (test_agent3_sentiment.py)
LLM calls are mocked — no network or GPU required.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.a3_sentiment import a3_sentiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(label: str, score: float, rationale: str = "Test rationale") -> MagicMock:
    """Build a mock LLM response with a valid JSON body."""
    import json
    content = json.dumps({
        "sentiment_label": label,
        "sentiment_score": score,
        "rationale": rationale,
    })
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


_CLEANED = [
    {"text": "Great video!", "cleaned_text": "great video!"},
    {"text": "Very informative.", "cleaned_text": "very informative."},
]

_STATE = {"cleaned_comments": _CLEANED}


# ---------------------------------------------------------------------------
# Happy path (mocked LLM)
# ---------------------------------------------------------------------------

class TestA3SentimentHappyPath:
    @patch("agents.a3_sentiment.get_llm")
    def test_returns_sentiment_key(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("positive", 0.8)
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "sentiment" in result

    @patch("agents.a3_sentiment.get_llm")
    def test_positive_sentiment_score(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("positive", 0.80)
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "positive"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(80.0)

    @patch("agents.a3_sentiment.get_llm")
    def test_neutral_sentiment(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("neutral", 0.5)
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "neutral"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(50.0)

    @patch("agents.a3_sentiment.get_llm")
    def test_negative_sentiment(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("negative", 0.2)
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "negative"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(20.0)

    @patch("agents.a3_sentiment.get_llm")
    def test_rationale_present(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("positive", 0.9, "Very positive corpus")
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "rationale" in result["sentiment"]
        assert result["sentiment"]["rationale"] == "Very positive corpus"

    @patch("agents.a3_sentiment.get_llm")
    def test_score_scaled_to_100(self, mock_get_llm):
        """sentiment_score [0,1] must be multiplied by 100."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response("positive", 0.73)
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_score"] == pytest.approx(73.0)


# ---------------------------------------------------------------------------
# Fallback when LLM unavailable
# ---------------------------------------------------------------------------

class TestA3SentimentFallback:
    @patch("agents.a3_sentiment.get_llm", return_value=None)
    def test_llm_unavailable_returns_neutral(self, _):
        result = a3_sentiment(_STATE)
        assert "sentiment" in result
        assert result["sentiment"]["sentiment_label"] == "neutral"
        assert result["sentiment"]["sentiment_score"] == 50.0

    @patch("agents.a3_sentiment.get_llm")
    def test_llm_parse_error_fallback(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not valid json {{")
        mock_get_llm.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "sentiment" in result
        assert result["sentiment"]["sentiment_label"] == "neutral"

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    def test_empty_corpus_fallback(self, _):
        result = a3_sentiment({"cleaned_comments": []})
        assert "sentiment" in result
