"""
api/schemas.py — Pydantic Request / Response Models
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class CommentItem(BaseModel):
    text: str
    author_likes: Optional[int] = None
    reply_count: Optional[int] = None


class AnalyzeRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video identifier")
    topic: str = Field(default="", description="User thematic query for personalised scoring")
    comments: list[CommentItem] = Field(..., min_length=1, description="Pre-collected comments")


# ── Response ──────────────────────────────────────────────────────────────────

class ScoreDetails(BaseModel):
    sentiment_score: float
    discourse_score: float
    noise_score: float


class AnalyzeResponse(BaseModel):
    video_id: str
    topic: str
    score_global: float = Field(description="Global quality score [0-100]")
    score_pertinence: float = Field(description="Topic relevance score [0-100]")
    score_final: float = Field(description="Weighted final score [0-100]")
    quality_label: str = Field(description="Faible | Moyen | Bon | Excellent")
    summary: Optional[str] = None
    topic_verdict: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    comment_count: int
    errors: list[str] = Field(default_factory=list)
    # ── v3.0 anti-hallucination fields ────────────────────────────────────────
    hallucination_flags: list[str] = Field(
        default_factory=list,
        description="Incohérences inter-champs et flags fallback détectés par le pipeline",
    )
    fallback_used: bool = Field(
        default=False,
        description="True si au moins un agent a utilisé le fallback heuristique",
    )
    sc_consensus: Optional[bool] = Field(
        default=None,
        description="Self-Consistency A7 : consensus >= 2/3 atteint",
    )
    low_consensus: Optional[bool] = Field(
        default=None,
        description="True si le consensus A7 est inférieur à 2/3 (résultat moins fiable)",
    )


class ReportNotFound(BaseModel):
    detail: str = "Report not found. Run POST /analyze first."
