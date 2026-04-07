"""
A6 — Synthesizer
=================
Aggregates scores from A3, A4, A5 into Score_Global [0-100]:

    Score_Global = 0.35 × Score_Sentiment
                 + 0.40 × Score_Discours
                 + 0.25 × Score_Bruit

Produces a natural-language summary via LLM and assigns a quality level label.
"""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.prompt_loader import load_prompt

logger = get_logger("a6_synthesizer")

# Scoring weights (PRD §4.3)
W_SENTIMENT = 0.35
W_DISCOURSE = 0.40
W_NOISE = 0.25

_SYSTEM_EN = "You are a video quality analyst. Write concise, factual summaries in English."
_SYSTEM_FR = "Tu es un analyste de qualité vidéo. Rédige des résumés concis et factuels en français."

_DEFAULT_SUMMARY_PROMPT = """\
{{lang_instruction}}

Based on the analysis of YouTube comments, write a 2–3 sentence summary explaining
the overall quality of the video as reflected by its comment section.

Scores (0–100):
- Sentiment score:  {{sentiment_score}}
- Discourse score:  {{discourse_score}}
- Noise score:      {{noise_score}}
- Global score:     {{score_global}}

Sentiment rationale:  {{sentiment_rationale}}
Discourse rationale:  {{discourse_rationale}}
Noise rationale:      {{noise_rationale}}

Summary:"""

_LANG_INSTRUCTION = {
    "fr": "Réponds UNIQUEMENT en français.",
    "en": "Reply ONLY in English.",
}

QualityLabel = Literal["Faible", "Moyen", "Bon", "Excellent"]


def _quality_label(score: float) -> QualityLabel:
    if score < 25:
        return "Faible"
    if score < 50:
        return "Moyen"
    if score < 75:
        return "Bon"
    return "Excellent"


def a6_synthesizer(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A6 Synthesizer."""
    sentiment = state.get("sentiment") or {}
    discourse = state.get("discourse") or {}
    noise = state.get("noise") or {}
    lang = (state.get("lang") or "fr").lower()

    s = float(sentiment.get("sentiment_score", 50.0))
    d = float(discourse.get("discourse_score", 50.0))
    n = float(noise.get("noise_score", 70.0))

    score_global = round(W_SENTIMENT * s + W_DISCOURSE * d + W_NOISE * n, 2)
    label = _quality_label(score_global)

    # Generate LLM summary
    summary = ""
    llm = get_llm()
    if llm is not None:
        system_msg = _SYSTEM_FR if lang == "fr" else _SYSTEM_EN
        lang_instruction = _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["fr"])
        template = load_prompt("prompts/synthesis_v1.txt", _DEFAULT_SUMMARY_PROMPT)
        user_msg = (
            template
            .replace("{{lang_instruction}}", lang_instruction)
            .replace("{{sentiment_score}}", str(s))
            .replace("{{discourse_score}}", str(d))
            .replace("{{noise_score}}", str(n))
            .replace("{{score_global}}", str(score_global))
            .replace("{{sentiment_rationale}}", sentiment.get("rationale", ""))
            .replace("{{discourse_rationale}}", discourse.get("rationale", ""))
            .replace("{{noise_rationale}}", noise.get("rationale", ""))
        )
        try:
            resp = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
            summary = resp.content.strip()
        except Exception as exc:
            logger.error("a6_synthesizer: summary LLM error — %s", exc)

    synthesis = {
        "score_global": score_global,
        "quality_label": label,
        "sentiment_score": s,
        "discourse_score": d,
        "noise_score": n,
        "summary": summary,
    }

    logger.info("a6_synthesizer: Score_Global=%.1f (%s)", score_global, label)
    save_checkpoint("a6_synthesizer", synthesis)

    return {"score_global": score_global, "synthesis": synthesis}
