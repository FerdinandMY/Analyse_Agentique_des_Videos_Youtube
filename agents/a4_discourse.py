"""
A4 — Discourse Analyser
========================
Evaluates the depth and quality of discourse across three dimensions:
  D1 — Informativeness  (does the comment add knowledge / facts?)
  D2 — Argumentation    (are claims supported by reasoning?)
  D3 — Constructiveness (is the tone constructive rather than reactive?)

Score_Discours [0-100] = mean of the three dimension scores.

High-quality comments (dimension score >= 0.7) are flagged for A7.
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

logger = get_logger("a4_discourse")

_SYSTEM = "You are an expert discourse analyst. Return strictly valid JSON following the schema."

_DEFAULT_PROMPT = """\
Evaluate the quality of discourse in the following YouTube comments across three dimensions.

Comments:
{{context}}

Score each dimension from 0.0 (very poor) to 1.0 (excellent):
- informativeness: Does the comment corpus add knowledge, facts, or context about the video topic?
- argumentation:   Are claims supported by reasoning or evidence?
- constructiveness: Is the overall tone constructive rather than purely reactive or emotional?

Also list comment indices (0-based) that have an average dimension score >= 0.7 as high_quality_indices.

{{format_instructions}}"""


class _DiscourseOutput(BaseModel):
    informativeness: float = Field(ge=0.0, le=1.0)
    argumentation: float = Field(ge=0.0, le=1.0)
    constructiveness: float = Field(ge=0.0, le=1.0)
    high_quality_indices: list[int] = Field(default_factory=list)
    rationale: str = Field(default="")


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for i, c in enumerate((state.get("cleaned_comments") or [])[:max_comments]):
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"[{i}] {t}")
    return "\n".join(pieces) or "(no comments)"


def a4_discourse(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A4 Discourse."""
    llm = get_llm()
    if llm is None:
        logger.warning("a4_discourse: LLM unavailable, returning default score.")
        return {"discourse": {"informativeness": 50.0, "argumentation": 50.0, "constructiveness": 50.0, "discourse_score": 50.0, "high_quality_indices": [], "rationale": "LLM unavailable"}}

    parser = PydanticOutputParser(pydantic_object=_DiscourseOutput)
    template = load_prompt("prompts/discourse_v1.txt", _DEFAULT_PROMPT)
    user_msg = (
        template
        .replace("{{context}}", _build_context(state))
        .replace("{{format_instructions}}", parser.get_format_instructions())
    )

    try:
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)])
        parsed = parser.parse(resp.content)
        avg = (parsed.informativeness + parsed.argumentation + parsed.constructiveness) / 3
        result = {
            "informativeness": round(parsed.informativeness * 100, 2),
            "argumentation": round(parsed.argumentation * 100, 2),
            "constructiveness": round(parsed.constructiveness * 100, 2),
            "discourse_score": round(avg * 100, 2),
            "high_quality_indices": parsed.high_quality_indices,
            "rationale": parsed.rationale,
        }
    except Exception as exc:
        logger.error("a4_discourse: LLM/parse error — %s", exc)
        result = {"informativeness": 50.0, "argumentation": 50.0, "constructiveness": 50.0, "discourse_score": 50.0, "high_quality_indices": [], "rationale": str(exc)}

    save_checkpoint("a4_discourse", result)
    return {"discourse": result}
