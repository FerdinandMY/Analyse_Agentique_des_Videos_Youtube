"""
Tests — A3 Sentiment Analyser (test_agent3_sentiment.py)
LLM calls are mocked — no network or GPU required.

v3.0 : mock sur agents.a3_sentiment.get_llm ET utils.llm_caller.get_llm
       (safe_llm_call importe get_llm indépendamment)
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.a3_sentiment import a3_sentiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(label: str, score: float, rationale: str = "Test rationale",
                        reasoning: str = "Thought 1: ...", confidence: float = 0.9) -> MagicMock:
    """Build a mock LLM response with the v3.0 ReAct JSON schema."""
    content = json.dumps({
        "sentiment_label":  label,
        "sentiment_score":  score,
        "confidence":       confidence,
        "rationale":        rationale,
        "reasoning":        reasoning,
        "sarcasm_detected": False,
    })
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


_CLEANED = [
    {"text": "Great video!", "cleaned_text": "great video!"},
    {"text": "Very informative.", "cleaned_text": "very informative."},
]

_STATE = {"cleaned_comments": _CLEANED}

# Patch combiné : agent + llm_caller (les deux appellent get_llm)
_PATCHES = [
    patch("agents.a3_sentiment.get_llm"),
    patch("models.llm_loader.get_llm"),
]


def _apply_patches(label, score, rationale="Test rationale"):
    """Retourne (mock_agent_llm, mock_caller_llm) avec la même réponse."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_llm_response(label, score, rationale)
    return mock_llm


# ---------------------------------------------------------------------------
# Happy path (mocked LLM)
# ---------------------------------------------------------------------------

class TestA3SentimentHappyPath:

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_returns_sentiment_key(self, mock_agent, mock_caller):
        mock_llm = _apply_patches("positive", 0.8)
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "sentiment" in result

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_positive_sentiment_score(self, mock_agent, mock_caller):
        mock_llm = _apply_patches("positive", 0.80)
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "positive"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(80.0)

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_neutral_sentiment(self, mock_agent, mock_caller):
        mock_llm = _apply_patches("neutral", 0.5)
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "neutral"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(50.0)

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_negative_sentiment(self, mock_agent, mock_caller):
        mock_llm = _apply_patches("negative", 0.2)
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_label"] == "negative"
        assert result["sentiment"]["sentiment_score"] == pytest.approx(20.0)

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_rationale_present(self, mock_agent, mock_caller):
        mock_llm = _apply_patches("positive", 0.9, "Very positive corpus")
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "rationale" in result["sentiment"]
        assert result["sentiment"]["rationale"] == "Very positive corpus"

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_score_scaled_to_100(self, mock_agent, mock_caller):
        """sentiment_score [0,1] must be multiplied by 100."""
        mock_llm = _apply_patches("positive", 0.73)
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert result["sentiment"]["sentiment_score"] == pytest.approx(73.0)


# ---------------------------------------------------------------------------
# Fallback when LLM unavailable
# ---------------------------------------------------------------------------

class TestA3SentimentFallback:

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    def test_llm_unavailable_returns_valid_sentiment(self, _):
        """v3.0 : le fallback utilise VADER — label peut être positive/neutral/negative."""
        result = a3_sentiment(_STATE)
        assert "sentiment" in result
        assert result["sentiment"]["sentiment_label"] in {"positive", "neutral", "negative"}
        assert 0.0 <= result["sentiment"]["sentiment_score"] <= 100.0

    @patch("models.llm_loader.get_llm")
    @patch("agents.a3_sentiment.get_llm")
    def test_llm_parse_error_fallback(self, mock_agent, mock_caller):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not valid json {{")
        mock_agent.return_value  = mock_llm
        mock_caller.return_value = mock_llm

        result = a3_sentiment(_STATE)
        assert "sentiment" in result
        assert result["sentiment"]["sentiment_label"] in {"positive", "neutral", "negative"}

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    def test_empty_corpus_fallback(self, _):
        result = a3_sentiment({"cleaned_comments": []})
        assert "sentiment" in result
