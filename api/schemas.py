"""
api/schemas.py — Pydantic Request / Response Models
=====================================================
v1.1 : ajout AskRequest, AskResponse, source, quota_used (PRD v1.1 §6.2)
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Analyze Request / Response ────────────────────────────────────────────────

class CommentItem(BaseModel):
    text: str
    author_likes: Optional[int] = None
    reply_count: Optional[int] = None


class AnalyzeRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video identifier or URL")
    topic: str = Field(default="", description="User thematic query for personalised scoring")
    lang: str = Field(
        default="fr",
        description="Langue des textes générés par le LLM : 'fr' (français) ou 'en' (anglais)",
        pattern="^(fr|en)$",
    )
    comments: Optional[list[CommentItem]] = Field(
        default=None,
        description=(
            "Commentaires pré-collectés (optionnel, usage avancé uniquement). "
            "Laisser VIDE — A0 collecte automatiquement depuis YouTube. "
            "Si fournis, doivent contenir au moins 5 éléments sinon ignorés."
        ),
    )
    force_refresh: bool = Field(
        default=False,
        description="Si True, ignore le cache et relance le pipeline complet",
    )


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
    # ── Langue de sortie ──────────────────────────────────────────────────────
    lang: Optional[str] = Field(
        default="fr",
        description="Langue des textes générés : 'fr' ou 'en'",
    )
    # ── v1.1 A0 Collector fields ──────────────────────────────────────────────
    source: Optional[str] = Field(
        default=None,
        description="Origine des données : 'api_v3' | 'csv_fallback' | 'cache' | 'pre_loaded'",
    )
    quota_used: Optional[int] = Field(
        default=None,
        description="Unités quota YouTube Data API v3 consommées par A0",
    )


class ReportNotFound(BaseModel):
    detail: str = "Report not found. Run POST /analyze first."


# ── Q&A Request / Response (PRD v1.1 §6.2) ───────────────────────────────────

class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' ou 'assistant'")
    content: str


class AskRequest(BaseModel):
    video_id: str = Field(..., description="Identifiant YouTube de la vidéo déjà analysée")
    question: str = Field(..., max_length=500, description="Question en langage naturel (max 500 chars)")
    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Historique de conversation (max 5 tours conservés)",
    )


class QASource(BaseModel):
    type: str = Field(..., description="'transcript' ou 'comment'")
    text: str
    start: Optional[float] = Field(default=None, description="Timestamp début (transcription uniquement)")
    comment_id: Optional[str] = Field(default=None, description="ID du commentaire (type=comment uniquement)")


class AskResponse(BaseModel):
    answer: str = Field(description="Réponse grounded du LLM")
    sources: list[QASource] = Field(
        default_factory=list,
        description="Passages de transcription ou commentaires utilisés pour répondre (FR-89)",
    )
    confidence: float = Field(
        default=0.0,
        description="Score de confiance de la réponse [0-1]",
    )
    transcript_used: bool = Field(
        default=False,
        description="True si la transcription a contribué à la réponse",
    )
    fallback_used: bool = Field(
        default=False,
        description="True si le LLM est indisponible et une réponse de secours est retournée",
    )
