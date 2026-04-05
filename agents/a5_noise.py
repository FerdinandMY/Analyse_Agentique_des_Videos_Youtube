"""
A5 — Noise Detector
====================
Detects and classifies noise across 5 categories:
  N1 — Spam / promotional content
  N2 — Off-topic comments (unrelated to video)
  N3 — Empty reactions ("lol", "first", emojis only)
  N4 — Toxic / hateful content
  N5 — Bot-generated patterns

Score_Bruit [0-100]:
  High noise → low score (bad quality signal)
  noise_ratio is the fraction of noisy comments [0-1]
  Score_Bruit = (1 - noise_ratio) * 100
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

logger = get_logger("a5_noise")

_SYSTEM = "You are an expert content moderator. Return strictly valid JSON following the schema."

_DEFAULT_PROMPT = """\
Analyse the following YouTube comments and classify the noise level.

Comments:
{{context}}

Estimate the proportion of comments belonging to each noise category (0.0–1.0 fraction of total):
- spam_ratio:      promotional, repetitive, or spammy comments
- offtopic_ratio:  comments unrelated to the video content
- reaction_ratio:  empty reactions (single words, emojis only, "first!", "lol")
- toxic_ratio:     hateful, offensive, or toxic language
- bot_ratio:       suspected bot-generated patterns

Also provide:
- noise_ratio:  overall fraction of noisy comments [0.0, 1.0] (can be your weighted estimate)
- rationale:    one sentence summary

{{format_instructions}}"""


class _NoiseOutput(BaseModel):
    spam_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    offtopic_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    reaction_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    toxic_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    bot_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    noise_ratio: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for c in (state.get("cleaned_comments") or [])[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments)"


def a5_noise(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A5 Noise Detector."""
    llm = get_llm()
    if llm is None:
        logger.warning("a5_noise: LLM unavailable, returning default score.")
        return {"noise": {"noise_ratio": 0.3, "noise_score": 70.0, "rationale": "LLM unavailable"}}

    parser = PydanticOutputParser(pydantic_object=_NoiseOutput)
    template = load_prompt("prompts/noise_detection_v1.txt", _DEFAULT_PROMPT)
    user_msg = (
        template
        .replace("{{context}}", _build_context(state))
        .replace("{{format_instructions}}", parser.get_format_instructions())
    )

    try:
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
        parsed = parser.parse(resp.content)
        noise_score = round((1.0 - parsed.noise_ratio) * 100, 2)
        result = {
            "spam_ratio": round(parsed.spam_ratio * 100, 2),
            "offtopic_ratio": round(parsed.offtopic_ratio * 100, 2),
            "reaction_ratio": round(parsed.reaction_ratio * 100, 2),
            "toxic_ratio": round(parsed.toxic_ratio * 100, 2),
            "bot_ratio": round(parsed.bot_ratio * 100, 2),
            "noise_ratio": round(parsed.noise_ratio * 100, 2),
            "noise_score": noise_score,
            "rationale": parsed.rationale,
        }
    except Exception as exc:
        logger.error("a5_noise: LLM/parse error — %s", exc)
        result = {"noise_ratio": 30.0, "noise_score": 70.0, "rationale": str(exc)}

    save_checkpoint("a5_noise", result)
    return {"noise": result}
