"""
A5 — Noise Detector  [PRD v3.0 — SVM first + CoT léger + Tools]
=================================================================
Technique :
  - svm_spam_detector   → premier filtre déterministe (tous les commentaires)
  - count_repeated_chars → détection patterns bots
  - CoT léger LLM        → uniquement pour les cas ambigus (SVM confidence < 0.75)
Température : 0.1
Pipeline anti-hallucination :
  safe_llm_call → NoiseValidator → check_coherence → retry x3
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from tools.noise_tools import svm_spam_detector, count_repeated_chars
from utils.checkpoint import save_checkpoint
from utils.llm_caller import safe_llm_call
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.validators import NoiseValidator

logger = get_logger("a5_noise")

_SYSTEM = (
    "You are an expert content moderator. "
    "Think briefly step by step, then produce final noise ratios. "
    "Return ONLY strictly valid JSON. Temperature: 0.1."
)

_FALLBACK = {
    "spam_ratio":     0.0,
    "offtopic_ratio": 0.0,
    "reaction_ratio": 0.0,
    "toxic_ratio":    0.0,
    "bot_ratio":      0.0,
    "noise_ratio":    30.0,
    "noise_score":    70.0,
    "rationale":      "LLM unavailable — fallback heuristique",
    "reasoning":      "",
    "svm_used":       True,
}

# Seuil de confiance SVM en-dessous duquel le CoT est déclenché (PRD §3.2)
_SVM_CONFIDENCE_THRESHOLD = 0.75


def _build_context(state: PipelineState, max_comments: int = 30) -> str:
    pieces = []
    for c in (state.get("cleaned_comments") or [])[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments)"


def _run_svm_filter(comments: list[dict]) -> dict[str, Any]:
    """
    Applique le SVM sur tous les commentaires.
    Retourne un résumé agrégé + flag cot_needed.
    """
    spam_count    = 0
    bot_count     = 0
    low_conf_count = 0
    total         = max(len(comments), 1)

    for c in comments[:30]:
        text = c.get("cleaned_text") or c.get("text") or ""
        if not text:
            continue

        # SVM spam
        svm_res = svm_spam_detector.invoke({"text": text})
        if svm_res.get("is_spam"):
            spam_count += 1
        if svm_res.get("confidence", 1.0) < _SVM_CONFIDENCE_THRESHOLD:
            low_conf_count += 1

        # Répétitions / bots
        rep_res = count_repeated_chars.invoke({"text": text})
        if rep_res.get("is_bot_suspect"):
            bot_count += 1

    spam_ratio   = round(spam_count / total, 4)
    bot_ratio    = round(bot_count / total, 4)
    avg_conf_low = low_conf_count / total

    return {
        "svm_spam_ratio": spam_ratio,
        "svm_bot_ratio":  bot_ratio,
        "low_conf_ratio": avg_conf_low,
        "cot_needed":     avg_conf_low >= (1 - _SVM_CONFIDENCE_THRESHOLD),
    }


def a5_noise(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A5 Noise Detector (SVM first + CoT léger + Tools)."""

    comments = (state.get("cleaned_comments") or [])[:30]
    context  = _build_context(state)

    # ── Étape 1 : SVM filter (tous les commentaires) ──────────────────────────
    svm_summary = _run_svm_filter(comments)
    logger.info(
        "a5 SVM — spam=%.1f%% bot=%.1f%% cot_needed=%s",
        svm_summary["svm_spam_ratio"] * 100,
        svm_summary["svm_bot_ratio"] * 100,
        svm_summary["cot_needed"],
    )

    # ── Étape 2 : Fallback si LLM indisponible ou CoT non nécessaire ─────────
    if get_llm() is None or not svm_summary["cot_needed"]:
        if get_llm() is None:
            logger.warning("a5_noise: LLM indisponible — fallback heuristique SVM")
        else:
            logger.info("a5_noise: SVM confiant — pas de CoT LLM")

        spam_r  = svm_summary["svm_spam_ratio"] * 100
        bot_r   = svm_summary["svm_bot_ratio"] * 100
        noise_r = round(min(100.0, spam_r + bot_r * 0.5 + 10.0), 2)  # estimation

        result = {
            **_FALLBACK,
            "spam_ratio":  round(spam_r, 2),
            "bot_ratio":   round(bot_r, 2),
            "noise_ratio": noise_r,
            "noise_score": round(100.0 - noise_r, 2),
            "svm_used":    True,
            "rationale":   f"SVM only — spam={spam_r:.1f}% bot={bot_r:.1f}%",
        }
        save_checkpoint("a5_noise", result)
        return {"noise": result}

    # ── Étape 3 : CoT léger LLM (cas ambigus) ────────────────────────────────
    default_prompt = (
        "Analyse the following YouTube comments and classify the noise level.\n\n"
        "SVM pre-filter results:\n"
        "- Spam ratio (SVM): {{svm_spam_ratio}}\n"
        "- Bot suspect ratio: {{svm_bot_ratio}}\n\n"
        "Comments:\n{{context}}\n\n"
        "Brief reasoning (CoT):\n"
        "Thought 1: Confirm or adjust the SVM spam estimate based on the comment content.\n"
        "Thought 2: Estimate off-topic, reaction-only, and toxic proportions.\n"
        "Thought 3: Compute overall noise_ratio.\n\n"
        "Return JSON: {\"reasoning\":\"...\",\"spam_ratio\":<0-1>,\"offtopic_ratio\":<0-1>,"
        "\"reaction_ratio\":<0-1>,\"toxic_ratio\":<0-1>,\"bot_ratio\":<0-1>,"
        "\"noise_ratio\":<0-1>,\"rationale\":\"...\",\"svm_used\":true}\nJSON only."
    )
    template = load_prompt("prompts/noise_detection_v1.txt", default_prompt)
    user_msg = (
        template
        .replace("{{context}}", context)
        .replace("{{svm_spam_ratio}}", f"{svm_summary['svm_spam_ratio']:.2f}")
        .replace("{{svm_bot_ratio}}", f"{svm_summary['svm_bot_ratio']:.2f}")
    )

    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]

    result, meta = safe_llm_call(
        messages=messages,
        validator=NoiseValidator(),
        fallback=_FALLBACK,
        agent_name="A5",
    )

    # Normalise les ratios [0-1] → [0-100]
    for field in ("spam_ratio", "offtopic_ratio", "reaction_ratio", "toxic_ratio",
                  "bot_ratio", "noise_ratio"):
        v = result.get(field, 0.0)
        if v is not None and v <= 1.0:
            result[field] = round(v * 100, 2)

    noise_ratio_pct = result.get("noise_ratio", 30.0)
    result["noise_score"] = round(100.0 - noise_ratio_pct, 2)
    result["svm_used"]    = True

    h_flags = meta.get("hallucination_flags", [])

    logger.info(
        "a5_noise: noise=%.1f score=%.1f retries=%d fallback=%s",
        result.get("noise_ratio"), result.get("noise_score"),
        meta.get("retries", 0), meta.get("fallback_used"),
    )

    save_checkpoint("a5_noise", result)
    return {
        "noise":               result,
        "hallucination_flags": h_flags,
    }
