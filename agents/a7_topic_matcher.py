"""
A7 — Topic Matcher  [PRD v3.0 — ToT systématique + Self-Consistency + Tools]
=============================================================================
Technique :
  - ToT systématique (3 branches : contenu, niveau, pédagogie) — toujours actif
  - Self-Consistency : 3 runs indépendants + vote majoritaire label + moyenne score
  - Si consensus < 2/3 → flag 'low_consensus' dans le rapport (PRD FR-60)
Température : 0.3
Tools appelés AVANT le LLM :
  1. compute_semantic_similarity → similarité cosinus thématique/commentaires
  2. extract_key_topics          → thèmes dominants du corpus

Pipeline anti-hallucination :
  safe_llm_call → TopicValidator → check_coherence → retry x3 (par run SC)
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from models.llm_loader import get_llm
from pipeline_state import PipelineState
from tools.topic_tools import compute_semantic_similarity, extract_key_topics
from utils.checkpoint import save_checkpoint
from utils.llm_caller import safe_llm_call
from utils.logger import get_logger
from utils.prompt_loader import load_prompt
from utils.validators import TopicValidator

logger = get_logger("a7_topic_matcher")

W_GLOBAL     = 0.60
W_PERTINENCE = 0.40

# Self-Consistency : nombre de runs indépendants (PRD FR-57)
_SC_RUNS = 3
# Seuil de consensus (PRD FR-60)
_SC_CONSENSUS_THRESHOLD = 2

_SYSTEM = (
    "You are a video relevance expert. "
    "Explore three reasoning branches before concluding. "
    "Return ONLY strictly valid JSON. Temperature: 0.3."
)

_FALLBACK_TOPIC = {
    "pertinence_score": 50.0,
    "verdict":          "Évaluation indisponible — pertinence thématique non évaluée.",
    "tot_branches":     {},
    "sc_runs":          [],
    "sc_consensus":     True,
    "low_consensus":    False,
}


def _get_high_quality_comments(state: PipelineState, max_comments: int = 20) -> str:
    cleaned  = state.get("cleaned_comments") or []
    discourse = state.get("discourse") or {}
    hq_indices = discourse.get("high_quality_indices") or []

    candidates = (
        [cleaned[i] for i in hq_indices if i < len(cleaned)]
        if hq_indices else cleaned
    )

    pieces = []
    for c in candidates[:max_comments]:
        t = c.get("cleaned_text") or c.get("text") if isinstance(c, dict) else str(c)
        if t:
            pieces.append(f"- {t}")
    return "\n".join(pieces) or "(no comments available)"


def _build_prompt(
    topic: str,
    hq_comments: str,
    similarity_score: float,
    key_topics: list[str],
) -> str:
    default_prompt = (
        'A user is researching "{{topic}}". Evaluate video relevance.\n\n'
        "Tool measurements:\n"
        "- Semantic similarity: {{similarity_score}}\n"
        "- Key topics in comments: {{key_topics}}\n\n"
        "High-quality comments:\n{{high_quality_comments}}\n\n"
        "Explore 3 branches: Content alignment | Audience level | Pedagogical value.\n"
        "Then synthesise with Self-Consistency in mind.\n\n"
        'Return JSON: {"reasoning":"...","tot_branches":{...},'
        '"pertinence_score":<0-1>,"verdict":"..."}\nJSON only.'
    )
    template = load_prompt("prompts/topic_matcher_v1.txt", default_prompt)
    return (
        template
        .replace("{{topic}}", topic)
        .replace("{{high_quality_comments}}", hq_comments)
        .replace("{{similarity_score}}", f"{similarity_score:.4f}")
        .replace("{{key_topics}}", ", ".join(key_topics[:10]) if key_topics else "N/A")
    )


def _single_run(messages: list) -> tuple[dict, dict]:
    """Exécute un seul run ToT via safe_llm_call."""
    return safe_llm_call(
        messages=messages,
        validator=TopicValidator(),
        fallback=_FALLBACK_TOPIC,
        agent_name="A7",
    )


def _self_consistency(
    messages: list,
) -> tuple[dict, list[dict], bool]:
    """
    Exécute _SC_RUNS runs indépendants, agrège par vote majoritaire (label)
    et moyenne (score). Retourne (result, runs, consensus_reached).
    """
    runs       = []
    scores     = []
    verdicts   = []
    tot_branches_list = []

    for i in range(_SC_RUNS):
        run_result, run_meta = _single_run(messages)
        score   = run_result.get("pertinence_score", 50.0)
        verdict = run_result.get("verdict", "")

        # Label de pertinence dérivé du score
        if score >= 65:
            label = "pertinent"
        elif score >= 35:
            label = "moyennement_pertinent"
        else:
            label = "non_pertinent"

        runs.append({
            "run":             i + 1,
            "pertinence_score": score,
            "label":           label,
            "verdict":         verdict,
            "fallback_used":   run_meta.get("fallback_used", False),
        })
        scores.append(score)
        verdicts.append(label)
        if run_result.get("tot_branches"):
            tot_branches_list.append(run_result["tot_branches"])

        logger.info("a7 SC run %d/%d — score=%.1f label=%s", i + 1, _SC_RUNS, score, label)

    # Vote majoritaire (PRD FR-59)
    label_counts  = Counter(verdicts)
    majority_label, majority_count = label_counts.most_common(1)[0]
    consensus = majority_count >= _SC_CONSENSUS_THRESHOLD  # >= 2/3

    # Score final = moyenne arithmétique (PRD FR-58)
    final_score = round(statistics.mean(scores), 2)

    # Verdict du run majoritaire
    best_run = next(
        (r for r in runs if r["label"] == majority_label and not r["fallback_used"]),
        runs[0],
    )
    final_verdict = best_run["verdict"]

    # Branches ToT : prendre celles du run majoritaire
    final_branches = tot_branches_list[0] if tot_branches_list else {}

    result = {
        "pertinence_score": final_score,
        "verdict":          final_verdict,
        "tot_branches":     final_branches,
        "sc_runs":          runs,
        "sc_consensus":     consensus,
        "low_consensus":    not consensus,
    }
    return result, runs, consensus


def a7_topic_matcher(state: PipelineState) -> dict[str, Any]:
    """LangGraph node — A7 Topic Matcher (ToT systématique + Self-Consistency + Tools)."""

    topic        = (state.get("topic") or "").strip()
    score_global = float(state.get("score_global") or 0.0)

    # ── Pas de thématique → score_final = score_global ────────────────────────
    if not topic:
        logger.info("a7_topic_matcher: pas de thématique — score_final = score_global")
        result = {
            "score_pertinence": 50.0,
            "score_final":      round(score_global, 2),
            "topic_verdict":    "Aucune thématique fournie — score basé sur la qualité globale.",
        }
        save_checkpoint("a7_topic_matcher", result)
        return result

    # ── Étape 1 : Tools (grounding déterministe) ──────────────────────────────
    hq_comments  = _get_high_quality_comments(state)
    all_comments = " ".join(
        (c.get("cleaned_text") or c.get("text") or "")
        for c in (state.get("cleaned_comments") or [])[:20]
        if isinstance(c, dict)
    )

    sim_result  = compute_semantic_similarity.invoke({
        "topic": topic, "comments_text": all_comments
    })
    topics_result = extract_key_topics.invoke({"comments_text": all_comments, "top_n": 10})

    sim_score  = float(sim_result.get("similarity_score", 0.5))
    key_topics = topics_result.get("topics", [])

    logger.info(
        "a7 tools — similarity=%.3f topics=%s",
        sim_score, key_topics[:5],
    )

    # ── Étape 2 : Fallback si LLM indisponible ────────────────────────────────
    if get_llm() is None:
        logger.warning("a7_topic_matcher: LLM indisponible — fallback similarité")
        score_pertinence = round(sim_score * 100, 2)
        score_final      = round(W_GLOBAL * score_global + W_PERTINENCE * score_pertinence, 2)
        result = {
            "score_pertinence": score_pertinence,
            "score_final":      score_final,
            "topic_verdict":    f"LLM indisponible — pertinence basée sur similarité sémantique ({sim_score:.2f}).",
        }
        save_checkpoint("a7_topic_matcher", result)
        return result

    # ── Étape 3 : Construction du prompt ToT ──────────────────────────────────
    user_msg  = _build_prompt(topic, hq_comments, sim_score, key_topics)
    messages  = [SystemMessage(content=_SYSTEM), HumanMessage(content=user_msg)]

    # ── Étape 4 : Self-Consistency (3 runs) ───────────────────────────────────
    sc_result, sc_runs, consensus = _self_consistency(messages)

    score_pertinence = sc_result.get("pertinence_score", 50.0)
    score_final      = round(W_GLOBAL * score_global + W_PERTINENCE * score_pertinence, 2)

    h_flags = []
    if not consensus:
        h_flags.append("low_consensus_a7")
        logger.warning(
            "a7_topic_matcher: consensus faible — %d/%d runs d'accord",
            max(Counter(r["label"] for r in sc_runs).values()), _SC_RUNS,
        )

    result = {
        "score_pertinence": score_pertinence,
        "score_final":      score_final,
        "topic_verdict":    sc_result.get("verdict", ""),
        # Traces traçabilité
        "tot_branches":     sc_result.get("tot_branches", {}),
        "sc_runs":          sc_runs,
        "sc_consensus":     consensus,
        "low_consensus":    not consensus,
        "similarity_score": sim_score,
        "key_topics":       key_topics,
    }

    logger.info(
        "a7_topic_matcher: pertinence=%.1f final=%.1f consensus=%s",
        score_pertinence, score_final, consensus,
    )

    save_checkpoint("a7_topic_matcher", result)
    return {
        **result,
        "hallucination_flags": h_flags,
    }
