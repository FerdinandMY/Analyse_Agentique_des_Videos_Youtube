"""
Tests — A5 Noise Detector (test_agent5_noise.py)
LLM calls are mocked — no network or GPU required.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.a5_noise import a5_noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(spam=0.1, offtopic=0.05, reaction=0.2, toxic=0.02, bot=0.01,
                        noise_ratio=0.3, rationale="Test noise"):
    content = json.dumps({
        "spam_ratio": spam,
        "offtopic_ratio": offtopic,
        "reaction_ratio": reaction,
        "toxic_ratio": toxic,
        "bot_ratio": bot,
        "noise_ratio": noise_ratio,
        "rationale": rationale,
    })
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


_CLEANED = [
    {"text": "Great!", "cleaned_text": "great!"},
    {"text": "Buy my product!!", "cleaned_text": "buy my product!!"},
    {"text": "Informative video.", "cleaned_text": "informative video."},
]

_STATE = {"cleaned_comments": _CLEANED}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestA5NoiseHappyPath:
    @patch("agents.a5_noise.get_llm")
    def test_returns_noise_key(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response()
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        assert "noise" in result

    @patch("agents.a5_noise.get_llm")
    def test_five_categories_present(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response()
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        n = result["noise"]
        for cat in ("spam_ratio", "offtopic_ratio", "reaction_ratio", "toxic_ratio", "bot_ratio"):
            assert cat in n

    @patch("agents.a5_noise.get_llm")
    def test_noise_score_formula(self, mock_get_llm):
        """Score_Bruit = (1 - noise_ratio) * 100"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(noise_ratio=0.30)
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        expected = round((1.0 - 0.30) * 100, 2)
        assert result["noise"]["noise_score"] == pytest.approx(expected)

    @patch("agents.a5_noise.get_llm")
    def test_zero_noise_score_100(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(noise_ratio=0.0)
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        assert result["noise"]["noise_score"] == pytest.approx(100.0)

    @patch("agents.a5_noise.get_llm")
    def test_full_noise_score_0(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(noise_ratio=1.0)
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        assert result["noise"]["noise_score"] == pytest.approx(0.0)

    @patch("agents.a5_noise.get_llm")
    def test_ratios_scaled_to_percent(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _make_llm_response(spam=0.15, offtopic=0.05)
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        assert result["noise"]["spam_ratio"] == pytest.approx(15.0)
        assert result["noise"]["offtopic_ratio"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestA5NoiseFallback:
    @patch("agents.a5_noise.get_llm", return_value=None)
    def test_llm_unavailable_returns_default(self, _):
        result = a5_noise(_STATE)
        assert "noise" in result
        assert result["noise"]["noise_score"] == pytest.approx(70.0)

    @patch("agents.a5_noise.get_llm")
    def test_parse_error_returns_default(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="not json")
        mock_get_llm.return_value = mock_llm

        result = a5_noise(_STATE)
        assert "noise" in result
        assert "noise_score" in result["noise"]
