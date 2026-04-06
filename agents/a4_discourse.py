"""
A4 — Discourse Analyser  [PRD v3.0 — CoT + ToT conditionnel + Tools]
======================================================================
Technique :
  - CoT standard si score hors zone [0.35–0.65]
  - ToT déclenché si score CoT dans la zone d'ambiguïté [0.35–0.65]
Température : 0.1 (CoT) / 0.2 (ToT)
Tools appelés AVANT le LLM :
  1. compute_text_stats            → statistiques textuelles
  2. detect_argumentative_markers  → marqueurs argumentatifs

Pipeline anti-hallucination :
  safe_llm_call → DiscourseValidator → check_coherence → retry x3
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from tools.discourse_tools import compute_text_stats, detect_argumentative_markers
from utils.checkpoint import save_checkpoint
from utils.llm_caller import safe_llm_call
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.validators import DiscourseValidator

logger = get_logger("a4_discourse")

_SYSTEM_COT = (
    "You are an expert discourse analyst. "
    "Think step by step before scoring. "
    "Return ONLY strictly valid JSON. Temperature: 0.1."
)
_SYSTEM_TOT = (
    "You are an expert discourse analyst. "
    "Explore three independent reasoning branches before concluding. "
    "Return ONLY strictly valid JSON. Temperature: 0.2."
)

# Zone d'ambiguïté → déclenchement ToT (PRD §3.2)
_TOT_LOW  = 0.35
_TOT_HIGH = 0.65

_FALLBACK = {
    "informativeness":      50.0,
    "argumentation":        50.0,
    "constructiveness":     50.0,
    "discourse_score":      50.0,
    "high_quality_indices": [],
    "rationale":            "LLM unavailable — fallback heuristique",
    "reasoning":            "",
    "tot_used":             False,
}


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for i, c in enumerate((state.get("cleaned_comments") or [])[:max_comments]):
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"[{i}] {t}")
    return "\n".join(pieces) or "(no comments)"


def _heuristic_score(text_stats: dict, arg_markers: dict) -> dict:
    """Fallback heuristique basé sur les tools."""
    word_count  = text_stats.get("word_count", 0)
    unique_r    = text_stats.get("unique_word_ratio", 0.5)
    arg_score   = arg_markers.get("argumentation_score", 0.0)

    info  = round(min(1.0, word_count / 100) * 0.6 + unique_r * 0.4, 2)
    arg   = round(arg_score, 2)
    const = round(0.5 + unique_r * 0.3, 2)
    avg   = round((info + arg + const) / 3, 2)

    return {
        **_FALLBACK,
        "informativeness":  round(info * 100, 2),
        "argumentation":    round(arg * 100, 2),
        "constructiveness": round(const * 100, 2),
        "discourse_score":  round(avg * 100, 2),
    }


def _run_cot(context: str, text_stats: dict, arg_markers: dict) -> tuple[dict, dict]:
    """Lance le CoT standard et retourne (result, meta)."""
    default_prompt = (
        "Evaluate the quality of discourse in the following YouTube comments.\n\n"
        "Tool measurements:\n"
        "- Text stats: {{text_stats}}\n"
        "- Argumentative markers: {{arg_markers}}\n\n"
        "Comments:\n{{context}}\n\n"
        "Think step by step: "
        "Thought 1 (informativeness) → Thought 2 (argumentation) → "
        "Thought 3 (constructiveness) → Thought 4 (high-quality indices) → Result.\n\n"
        'Return JSON: {"reasoning":"...","informativeness":<0-1>,"argumentation":<0-1>,'
        '"constructiveness":<0-1>,"discourse_score":<0-1>,'
        '"high_quality_indices":[...],"rationale":"..."}\nJSON only.'
    )
    template = load_prompt("prompts/discourse_cot_v1.txt", default_prompt)
    user_msg = (
        template
        .replace("{{context}}", context)
        .replace("{{text_stats}}", json.dumps(text_stats, ensure_ascii=False))
        .replace("{{arg_markers}}", json.dumps(arg_markers, ensure_ascii=False))
    )
    messages = [SystemMessage(content=_SYSTEM_COT), HumanMessage(content=user_msg)]
    return safe_llm_call(
        messages=messages,
        validator=DiscourseValidator(),
        fallback=_FALLBACK,
        agent_name="A4-CoT",
    )


def _run_tot(context: str, text_stats: dict, arg_markers: dict) -> tuple[dict, dict]:
    """Lance le ToT (3 branches) pour les scores ambigus."""
    default_prompt = (
        "The discourse score is ambiguous. Explore 3 branches:\n"
        "Tool measurements:\n"
        "- Text stats: {{text_stats}}\n- Arg markers: {{arg_markers}}\n\n"
        "Comments:\n{{context}}\n\n"
        "Branch 1 (Content Depth) → Branch 2 (Discussion Quality) → "
        "Branch 3 (Epistemic Value) → Synthesis.\n\n"
        'Return JSON: {"tot_used":true,"reasoning":"...","tot_branches":{...},'
        '"informativeness":<0-1>,"argumentation":<0-1>,"constructiveness":<0-1>,'
        '"discourse_score":<0-1>,"high_quality_indices":[...],"rationale":"..."}\nJSON only.'
    )
    template = load_prompt("prompts/discourse_tot_v1.txt", default_prompt)
    user_msg = (
        template
        .replace("{{context}}", context)
        .replace("{{text_stats}}", json.dumps(text_stats, ensure_ascii=False))
        .replace("{{arg_markers}}", json.dumps(arg_markers, ensure_ascii=False))
    )
    messages = [SystemMessage(content=_SYSTEM_TOT), HumanMessage(content=user_msg)]
    return safe_llm_call(
        messages=messages,
        validator=DiscourseValidator(),
        fallback={**_FALLBACK, "tot_used": True},
        agent_name="A4-ToT",
    )


def a4_discourse(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A4 Discourse (CoT + ToT conditionnel + Tools)."""

    context  = _build_context(state)
    all_text = " ".join(
        (c.get("cleaned_text") or c.get("text") or "")
        for c in (state.get("cleaned_comments") or [])[:30]
        if isinstance(c, dict)
    )

    # ── Étape 1 : Tools (grounding déterministe) ──────────────────────────────
    text_stats  = compute_text_stats.invoke({"text": all_text})
    arg_markers = detect_argumentative_markers.invoke({"text": all_text})

    logger.info(
        "a4 tools — words=%d arg_score=%.2f",
        text_stats.get("word_count", 0),
        arg_markers.get("argumentation_score", 0),
    )

    # ── Étape 2 : Fallback si LLM indisponible ────────────────────────────────
    if get_llm() is None:
        logger.warning("a4_discourse: LLM indisponible — fallback heuristique")
        result = _heuristic_score(text_stats, arg_markers)
        save_checkpoint("a4_discourse", result)
        return {"discourse": result}

    # ── Étape 3 : CoT standard ────────────────────────────────────────────────
    result, meta = _run_cot(context, text_stats, arg_markers)
    cot_score    = result.get("discourse_score", 50.0)
    # Normalise [0-1] → [0-100] si nécessaire
    if cot_score <= 1.0:
        cot_score = cot_score * 100
    cot_score_norm = cot_score / 100  # [0-1] pour test de zone

    h_flags = meta.get("hallucination_flags", [])

    # ── Étape 4 : ToT conditionnel si score ambigu ────────────────────────────
    tot_triggered = _TOT_LOW <= cot_score_norm <= _TOT_HIGH and not meta.get("fallback_used")

    if tot_triggered:
        logger.info(
            "a4_discourse: CoT score=%.2f dans zone [%.2f–%.2f] → ToT déclenché",
            cot_score_norm, _TOT_LOW, _TOT_HIGH,
        )
        result_tot, meta_tot = _run_tot(context, text_stats, arg_markers)
        # Le ToT remplace le CoT si il n'est pas en fallback
        if not meta_tot.get("fallback_used"):
            result = result_tot
            h_flags.extend(meta_tot.get("hallucination_flags", []))
            h_flags.append("tot_triggered")
            meta = meta_tot
        else:
            h_flags.append("tot_fallback_to_cot")

    # ── Normalisation finale [0-1] → [0-100] ─────────────────────────────────
    for field in ("informativeness", "argumentation", "constructiveness", "discourse_score"):
        v = result.get(field, 50.0)
        if v is not None and v <= 1.0:
            result[field] = round(v * 100, 2)

    # Enrichit avec les résultats tools
    result["text_stats"]   = text_stats
    result["arg_markers"]  = arg_markers
    result["tot_used"]     = result.get("tot_used", tot_triggered)

    logger.info(
        "a4_discourse: score=%.1f tot=%s retries=%d fallback=%s",
        result.get("discourse_score"), result.get("tot_used"),
        meta.get("retries", 0), meta.get("fallback_used"),
    )

    save_checkpoint("a4_discourse", result)
    return {
        "discourse":           result,
        "hallucination_flags": h_flags,
    }
