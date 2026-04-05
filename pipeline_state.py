from __future__ import annotations

import operator
from typing import Any, Annotated, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph pipeline.

    A1 → A2 → [A3 ‖ A4 ‖ A5] → A6 → A7 → report
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    csv_path: str                        # Path to CSV already collected
    video_id: str                        # YouTube video identifier
    topic: str                           # User's thematic query (A7)

    # ── A1: Loader output ─────────────────────────────────────────────────────
    raw_comments: list[dict[str, Any]]   # [{text, video_id?, author_likes?, reply_count?}, ...]

    # ── A2: Preprocessor output ───────────────────────────────────────────────
    cleaned_comments: list[dict[str, Any]]  # [{text, cleaned_text, language, ...}, ...]

    # ── A3/A4/A5: Parallel analysis outputs ───────────────────────────────────
    sentiment: Optional[dict[str, Any]]  # {sentiment_label, sentiment_score [0-100], rationale}
    discourse: Optional[dict[str, Any]]  # {discourse_dimensions, discourse_score [0-100], rationale}
    noise: Optional[dict[str, Any]]      # {noise_categories, noise_score [0-100], rationale}

    # ── A6: Synthesizer output ─────────────────────────────────────────────────
    score_global: float                  # [0-100] = 0.35*S + 0.40*D + 0.25*N
    synthesis: Optional[dict[str, Any]]  # {score_global, quality_level, summary}

    # ── A7: Topic Matcher output ───────────────────────────────────────────────
    score_pertinence: float              # [0-100] topic relevance score
    score_final: float                   # [0-100] = 0.60*score_global + 0.40*score_pertinence
    topic_verdict: Optional[str]         # Personalized explanation of topic relevance

    # ── Final assembled report ─────────────────────────────────────────────────
    report: Optional[dict[str, Any]]

    # ── Error accumulator (LangGraph reducer) ─────────────────────────────────
    errors: Annotated[list[str], operator.add]
