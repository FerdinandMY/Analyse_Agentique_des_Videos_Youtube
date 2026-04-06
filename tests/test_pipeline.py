"""
Tests — Full Pipeline (test_pipeline.py)
Integration tests for the LangGraph pipeline. LLM calls are mocked.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from graph import build_graph, run_pipeline


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_llm_for_all_agents():
    """Returns a mock LLM that provides valid JSON for all agents."""
    mock_llm = MagicMock()

    def _invoke(messages, **kwargs):
        # Return contextual response based on message content
        prompt = str(messages[-1].content if hasattr(messages[-1], "content") else messages)

        if "sentiment" in prompt.lower() or "sentiment_label" in prompt:
            content = json.dumps({
                "sentiment_label": "positive",
                "sentiment_score": 0.75,
                "rationale": "Mostly positive",
            })
        elif "informativeness" in prompt or "argumentation" in prompt:
            content = json.dumps({
                "informativeness": 0.7,
                "argumentation": 0.6,
                "constructiveness": 0.8,
                "high_quality_indices": [0],
                "rationale": "Good discourse",
            })
        elif "noise_ratio" in prompt or "spam_ratio" in prompt:
            content = json.dumps({
                "spam_ratio": 0.1,
                "offtopic_ratio": 0.05,
                "reaction_ratio": 0.1,
                "toxic_ratio": 0.0,
                "bot_ratio": 0.0,
                "noise_ratio": 0.2,
                "rationale": "Low noise",
            })
        elif "pertinence_score" in prompt or "topic" in prompt.lower():
            content = json.dumps({
                "pertinence_score": 0.80,
                "verdict": "Highly relevant to the topic",
            })
        else:
            content = "Summary: Good quality video."

        return MagicMock(content=content)

    mock_llm.invoke.side_effect = _invoke
    return mock_llm


_RAW_COMMENTS = [
    {"text": "This is a great tutorial on machine learning!"},
    {"text": "Very informative, learned a lot about gradient descent."},
    {"text": "Excellent explanation of neural networks."},
    {"text": "Best course I have ever taken."},
    {"text": "Could you explain backpropagation in more detail?"},
]


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_graph_compiles(self):
        app = build_graph()
        assert app is not None

    def test_graph_has_correct_nodes(self):
        # build_graph should not raise
        app = build_graph()
        assert hasattr(app, "invoke")


# ---------------------------------------------------------------------------
# run_pipeline — with mocked LLM
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("agents.a3_sentiment.get_llm")
    @patch("agents.a4_discourse.get_llm")
    @patch("agents.a5_noise.get_llm")
    @patch("agents.a6_synthesizer.get_llm")
    @patch("agents.a7_topic_matcher.get_llm")
    def test_pipeline_returns_report(self, m7, m6, m5, m4, m3):
        mock_llm = _mock_llm_for_all_agents()
        m3.return_value = m4.return_value = m5.return_value = m6.return_value = m7.return_value = mock_llm

        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="test123", topic="machine learning")
        assert result is not None

    @patch("agents.a3_sentiment.get_llm")
    @patch("agents.a4_discourse.get_llm")
    @patch("agents.a5_noise.get_llm")
    @patch("agents.a6_synthesizer.get_llm")
    @patch("agents.a7_topic_matcher.get_llm")
    def test_pipeline_score_global_range(self, m7, m6, m5, m4, m3):
        mock_llm = _mock_llm_for_all_agents()
        m3.return_value = m4.return_value = m5.return_value = m6.return_value = m7.return_value = mock_llm

        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="test123")
        score = result.get("score_global")
        assert score is not None
        assert 0.0 <= score <= 100.0

    @patch("agents.a3_sentiment.get_llm")
    @patch("agents.a4_discourse.get_llm")
    @patch("agents.a5_noise.get_llm")
    @patch("agents.a6_synthesizer.get_llm")
    @patch("agents.a7_topic_matcher.get_llm")
    def test_pipeline_score_final_range(self, m7, m6, m5, m4, m3):
        mock_llm = _mock_llm_for_all_agents()
        m3.return_value = m4.return_value = m5.return_value = m6.return_value = m7.return_value = mock_llm

        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="test123", topic="ML")
        score = result.get("score_final")
        assert score is not None
        assert 0.0 <= score <= 100.0

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    @patch("agents.a4_discourse.get_llm", return_value=None)
    @patch("agents.a5_noise.get_llm", return_value=None)
    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    @patch("agents.a7_topic_matcher.get_llm", return_value=None)
    def test_pipeline_runs_without_llm(self, *_):
        """Pipeline must complete even when all LLMs are unavailable."""
        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="test_fallback")
        assert result is not None
        assert result.get("score_global") is not None

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    @patch("agents.a4_discourse.get_llm", return_value=None)
    @patch("agents.a5_noise.get_llm", return_value=None)
    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    @patch("agents.a7_topic_matcher.get_llm", return_value=None)
    def test_pipeline_no_topic_score_final_equals_global(self, *_):
        """Without topic, score_final must equal score_global."""
        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="no_topic")
        assert result.get("score_final") == result.get("score_global")

    @patch("agents.a3_sentiment.get_llm", return_value=None)
    @patch("agents.a4_discourse.get_llm", return_value=None)
    @patch("agents.a5_noise.get_llm", return_value=None)
    @patch("agents.a6_synthesizer.get_llm", return_value=None)
    @patch("agents.a7_topic_matcher.get_llm", return_value=None)
    def test_pipeline_quality_label_valid(self, *_):
        result = run_pipeline(raw_comments=_RAW_COMMENTS, video_id="label_test")
        label = result.get("quality_label")
        assert label in ("Faible", "Moyen", "Bon", "Excellent")

    def test_pipeline_empty_raw_comments(self):
        """Pipeline must not crash on empty input."""
        result = run_pipeline(raw_comments=[], video_id="empty_test")
        assert result is not None

    def test_pipeline_missing_csv_returns_error(self):
        result = run_pipeline(csv_path="/nonexistent/file.csv", video_id="err_test")
        errors = result.get("errors") or []
        assert len(errors) > 0
