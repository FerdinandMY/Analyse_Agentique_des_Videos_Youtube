from __future__ import annotations

import operator
from typing import Any, Annotated, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph pipeline.

    A1 → A2 → [A3 ‖ A4 ‖ A5] → A6 → A7 → report

    v3.0 : ajout hallucination_flags, retries, traces CoT/ToT/SC
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    csv_path: str                        # Path to CSV already collected
    video_id: str                        # YouTube video identifier
    topic: str                           # User's thematic query (A7)
    lang: str                            # Langue de sortie LLM : "fr" | "en"

    # ── A1: Loader output ─────────────────────────────────────────────────────
    raw_comments: list[dict[str, Any]]   # [{text, video_id?, author_likes?, reply_count?}, ...]

    # ── A2: Preprocessor output ───────────────────────────────────────────────
    cleaned_comments: list[dict[str, Any]]  # [{text, cleaned_text, language, ...}, ...]

    # ── A3/A4/A5: Parallel analysis outputs ───────────────────────────────────
    sentiment: Optional[dict[str, Any]]
    # {sentiment_label, sentiment_score [0-100], confidence, reasoning (CoT),
    #  sarcasm_detected, vader_signal, rationale}

    discourse: Optional[dict[str, Any]]
    # {informativeness, argumentation, constructiveness, discourse_score [0-100],
    #  high_quality_indices, reasoning (CoT), tot_branches (ToT), tot_used, rationale}

    noise: Optional[dict[str, Any]]
    # {spam_ratio, offtopic_ratio, reaction_ratio, toxic_ratio, bot_ratio,
    #  noise_ratio, noise_score [0-100], svm_used, reasoning (CoT), rationale}

    # ── A6: Synthesizer output ─────────────────────────────────────────────────
    score_global: float                  # [0-100] = 0.35*S + 0.40*D + 0.25*N
    synthesis: Optional[dict[str, Any]]  # {score_global, quality_level, summary}

    # ── A7: Topic Matcher output ───────────────────────────────────────────────
    score_pertinence: float              # [0-100] topic relevance score
    score_final: float                   # [0-100] = 0.60*score_global + 0.40*score_pertinence
    topic_verdict: Optional[str]         # Personalized explanation of topic relevance
    tot_branches: Optional[dict[str, Any]]   # Traces ToT A7 (3 branches + scores)
    sc_runs: Optional[list[dict[str, Any]]]  # Self-Consistency runs (3 runs + votes)
    sc_consensus: Optional[bool]             # True si majorité 2/3 atteinte
    low_consensus: Optional[bool]            # True si consensus < 2/3 (flag rapport)

    # ── Final assembled report ─────────────────────────────────────────────────
    report: Optional[dict[str, Any]]

    # ── Anti-hallucination tracking (PRD v3.0 §3.5) ───────────────────────────
    hallucination_flags: Annotated[list[str], operator.add]
    # Flags cumulatifs : incohérences inter-champs, retries épuisés,
    # low_consensus A7, fallback_used, etc.

    retries: Optional[dict[str, int]]
    # {agent_name: n_retries} — nombre de retries par agent

    fallback_used: Optional[bool]
    # True si au moins un agent a utilisé le fallback heuristique

    # ── A0: Collector output (PRD v1.1) ───────────────────────────────────────
    source: Optional[str]
    # "api_v3" | "csv_fallback" | "cache" | "pre_loaded" (FR-81)

    quota_used: Optional[int]
    # Unités quota YouTube Data API v3 consommées par A0 (FR-79)

    collected_at: Optional[str]
    # Timestamp ISO8601 de la collecte A0

    transcript: Optional[list]
    # [{text: str, start: float, duration: float}, ...] — sous-titres vidéo (FR-84)

    transcript_available: Optional[bool]
    # False si youtube-transcript-api lève une exception (FR-85)

    qa_context: Optional[dict]
    # {transcript, transcript_available, top_comments, video_title} — stocké en cache (FR-86)

    # ── Error accumulator (LangGraph reducer) ─────────────────────────────────
    errors: Annotated[list[str], operator.add]
