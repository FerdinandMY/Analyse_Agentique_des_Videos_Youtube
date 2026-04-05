"""
A7 — Topic Matcher  [NOUVEAU — PRD v2.0]
==========================================
Evaluates how relevant the video is to the user's thematic query.

    Score_Pertinence [0-100] — LLM comparison of topic vs high-quality comments

    Score_Final = 0.60 × Score_Global + 0.40 × Score_Pertinence

High-quality comments are those flagged by A4 (discourse_score >= 70).
If none flagged, the full cleaned_comments corpus is used.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.prompt_loader import load_prompt

logger = get_logger("a7_topic_matcher")

# Scoring weights (PRD §4.3)
W_GLOBAL = 0.60
W_PERTINENCE = 0.40

_SYSTEM = "You are a video relevance expert. Return strictly valid JSON following the schema."

_DEFAULT_PROMPT = """\
A user is looking for content about the following topic:
Topic: "{{topic}}"

Below are the most insightful comments from a YouTube video (selected for discourse quality):

{{high_quality_comments}}

Evaluate how relevant this video is to the user's topic:
- pertinence_score: float [0.0, 1.0]  (1.0 = perfectly aligned with the topic)
- verdict: a personalised 1–2 sentence explanation of why this video is or is not relevant to the topic

{{format_instructions}}"""


class _TopicOutput(BaseModel):
    pertinence_score: float = Field(ge=0.0, le=1.0)
    verdict: str = Field(default="")


def _get_high_quality_comments(state: PipelineState, max_comments: int = 20) -> str:
    cleaned = state.get("cleaned_comments") or []
    discourse = state.get("discourse") or {}
    hq_indices = discourse.get("high_quality_indices") or []

    if hq_indices:
        candidates = [cleaned[i] for i in hq_indices if i < len(cleaned)]
    else:
        candidates = cleaned

    pieces = []
    for c in candidates[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments available)"


def a7_topic_matcher(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A7 Topic Matcher."""
    topic = (state.get("topic") or "").strip()
    score_global = float(state.get("score_global") or 0.0)

    if not topic:
        # No topic provided: score_final = score_global, pertinence neutral
        logger.info("a7_topic_matcher: no topic provided, score_final = score_global.")
        score_final = score_global
        result = {
            "score_pertinence": 50.0,
            "score_final": round(score_final, 2),
            "topic_verdict": "Aucune thématique fournie — score final basé sur la qualité globale uniquement.",
        }
        save_checkpoint("a7_topic_matcher", result)
        return result

    llm = get_llm()
    if llm is None:
        logger.warning("a7_topic_matcher: LLM unavailable, using score_global as score_final.")
        score_final = round(W_GLOBAL * score_global + W_PERTINENCE * 50.0, 2)
        result = {
            "score_pertinence": 50.0,
            "score_final": score_final,
            "topic_verdict": "LLM indisponible — pertinence thématique non évaluée.",
        }
        save_checkpoint("a7_topic_matcher", result)
        return result

    parser = PydanticOutputParser(pydantic_object=_TopicOutput)
    template = load_prompt("prompts/topic_matcher_v1.txt", _DEFAULT_PROMPT)
    user_msg = (
        template
        .replace("{{topic}}", topic)
        .replace("{{high_quality_comments}}", _get_high_quality_comments(state))
        .replace("{{format_instructions}}", parser.get_format_instructions())
    )

    try:
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
        parsed = parser.parse(resp.content)
        score_pertinence = round(parsed.pertinence_score * 100, 2)
        score_final = round(W_GLOBAL * score_global + W_PERTINENCE * score_pertinence, 2)
        result = {
            "score_pertinence": score_pertinence,
            "score_final": score_final,
            "topic_verdict": parsed.verdict,
        }
    except Exception as exc:
        logger.error("a7_topic_matcher: LLM/parse error — %s", exc)
        score_pertinence = 50.0
        score_final = round(W_GLOBAL * score_global + W_PERTINENCE * score_pertinence, 2)
        result = {
            "score_pertinence": score_pertinence,
            "score_final": score_final,
            "topic_verdict": f"Erreur d'évaluation: {exc}",
        }

    logger.info(
        "a7_topic_matcher: pertinence=%.1f score_final=%.1f",
        result["score_pertinence"],
        result["score_final"],
    )
    save_checkpoint("a7_topic_matcher", result)
    return result
