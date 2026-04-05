"""
A3 — Sentiment Analyser
========================
Classifies the overall sentiment of the comment corpus and produces
a Score_Sentiment in [0-100].

Sentiment interpretation:
  positive → high score (indicates viewers reacted positively)
  neutral  → mid score
  negative → low score
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

logger = get_logger("a3_sentiment")

_SYSTEM = "You are an expert analyst. Return strictly valid JSON following the schema."

_DEFAULT_PROMPT = """\
Analyse the overall sentiment expressed by the following YouTube comments.
Comments represent viewer reactions to a video.

Comments:
{{context}}

Produce:
- sentiment_label: one of "positive", "neutral", "negative"
- sentiment_score: float [0.0, 1.0] (1.0 = overwhelmingly positive)
- rationale: one sentence explaining your decision

{{format_instructions}}"""


class _SentimentOutput(BaseModel):
    sentiment_label: str = Field(description="positive | neutral | negative")
    sentiment_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for c in (state.get("cleaned_comments") or [])[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments)"


def a3_sentiment(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A3 Sentiment."""
    llm = get_llm()
    if llm is None:
        logger.warning("a3_sentiment: LLM unavailable, returning default score.")
        return {"sentiment": {"sentiment_label": "neutral", "sentiment_score": 50.0, "rationale": "LLM unavailable"}}

    parser = PydanticOutputParser(pydantic_object=_SentimentOutput)
    template = load_prompt("prompts/sentiment_v1.txt", _DEFAULT_PROMPT)
    user_msg = (
        template
        .replace("{{context}}", _build_context(state))
        .replace("{{format_instructions}}", parser.get_format_instructions())
    )

    try:
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
        parsed = parser.parse(resp.content)
        score_100 = round(parsed.sentiment_score * 100, 2)
        result = {
            "sentiment_label": parsed.sentiment_label,
            "sentiment_score": score_100,
            "rationale": parsed.rationale,
        }
    except Exception as exc:
        logger.error("a3_sentiment: LLM/parse error — %s", exc)
        result = {"sentiment_label": "neutral", "sentiment_score": 50.0, "rationale": str(exc)}

    save_checkpoint("a3_sentiment", result)
    return {"sentiment": result}
