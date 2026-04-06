"""
A3 — Sentiment Analyser  [PRD v3.0 — ReAct (CoT) + Tools]
===========================================================
Technique : ReAct (Thought → Action → Observation → Result)
Température : 0.1
Tools appelés AVANT le LLM :
  1. vader_sentiment         → score de grounding
  2. detect_sarcasm_markers  → correction sarcasme/ironie

Pipeline anti-hallucination :
  safe_llm_call → SentimentValidator → check_coherence → retry x3
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from tools.sentiment_tools import vader_sentiment, detect_sarcasm_markers
from utils.checkpoint import save_checkpoint
from utils.llm_caller import safe_llm_call
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.validators import SentimentValidator

logger = get_logger("a3_sentiment")

_SYSTEM = (
    "You are an expert sentiment analyst. "
    "Use step-by-step reasoning (ReAct). "
    "Return ONLY strictly valid JSON. Temperature: 0.1."
)

_FALLBACK = {
    "sentiment_label": "neutral",
    "sentiment_score": 50.0,
    "confidence":      0.5,
    "rationale":       "LLM unavailable — fallback heuristique",
    "reasoning":       "",
    "sarcasm_detected": False,
}

_TOT_AMBIGUITY_ZONE = (0.35, 0.65)  # non utilisé en A3, réservé A4


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for c in (state.get("cleaned_comments") or [])[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments)"


def a3_sentiment(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A3 Sentiment (ReAct + Tools)."""

    context = _build_context(state)
    all_text = " ".join(
        (c.get("cleaned_text") or c.get("text") or "")
        for c in (state.get("cleaned_comments") or [])[:30]
        if isinstance(c, dict)
    )

    # ── Étape 1 : Tools (grounding déterministe) ──────────────────────────────
    vader_result   = vader_sentiment.invoke({"text": all_text})
    sarcasm_result = detect_sarcasm_markers.invoke({"text": all_text})

    logger.info(
        "a3 tools — vader=%s sarcasm=%s",
        vader_result.get("label"), sarcasm_result.get("has_sarcasm"),
    )

    # ── Étape 2 : Fallback rapide si LLM indisponible ────────────────────────
    if get_llm() is None:
        logger.warning("a3_sentiment: LLM indisponible — fallback heuristique")
        label = vader_result.get("label", "neutral")
        score = {
            "positive": 70.0,
            "neutral":  50.0,
            "negative": 30.0,
        }.get(label, 50.0)
        result = {**_FALLBACK, "sentiment_label": label, "sentiment_score": score}
        save_checkpoint("a3_sentiment", result)
        return {"sentiment": result}

    # ── Étape 3 : Prompt ReAct avec résultats tools injectés ─────────────────
    default_prompt = (
        "Analyse the overall sentiment expressed by the following YouTube comments.\n"
        "Comments represent the collective viewer reaction to a video.\n\n"
        "Tool measurements (ground your reasoning on these facts):\n"
        "- VADER sentiment: {{vader_result}}\n"
        "- Sarcasm detection: {{sarcasm_result}}\n\n"
        "Comments:\n{{context}}\n\n"
        "Use ReAct: Thought 1 (VADER baseline) → Thought 2 (sarcasm correction) "
        "→ Thought 3 (final tone) → Result.\n\n"
        'Return JSON: {"reasoning":"...","sentiment_label":"positive|neutral|negative",'
        '"sentiment_score":<0.0-1.0>,"confidence":<0.0-1.0>,'
        '"sarcasm_detected":<bool>,"rationale":"..."}\n\nJSON only.'
    )
    template = load_prompt("prompts/sentiment_v1.txt", default_prompt)
    user_msg = (
        template
        .replace("{{context}}", context)
        .replace("{{vader_result}}", str(vader_result))
        .replace("{{sarcasm_result}}", str(sarcasm_result))
    )

    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]

    # ── Étape 4 : Appel LLM sécurisé (retry x3 + validation) ─────────────────
    result, meta = safe_llm_call(
        messages=messages,
        validator=SentimentValidator(),
        fallback=_FALLBACK,
        agent_name="A3",
    )

    # Normalise score [0-1] → [0-100]
    score = result.get("sentiment_score", 50.0)
    if score <= 1.0:
        score = round(score * 100, 2)
    result["sentiment_score"] = score

    # Enrichit avec les résultats tools
    result["vader_signal"]   = vader_result.get("label")
    result["sarcasm_signal"] = sarcasm_result.get("has_sarcasm")

    # Propage hallucination_flags dans le state
    h_flags = meta.get("hallucination_flags", [])

    logger.info(
        "a3_sentiment: label=%s score=%.1f retries=%d fallback=%s flags=%s",
        result.get("sentiment_label"), result.get("sentiment_score"),
        meta.get("retries", 0), meta.get("fallback_used"), h_flags,
    )

    save_checkpoint("a3_sentiment", result)
    return {
        "sentiment":          result,
        "hallucination_flags": h_flags,
    }
