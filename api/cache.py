"""
api/cache.py — In-Memory Report Cache
=======================================
Caches pipeline reports keyed by (video_id, topic) to avoid re-running
the full pipeline for the same request.

v1.1 : ajout de exists(), set_qa_context(), get_qa_context()
       pour supporter le module Q&A sans re-appel API (PRD v1.1 §4.3).

v1.2 : statut d'enrichissement background (pending/done).
       Quand un rapport "rapide" (300 commentaires) est servi, le pipeline
       complet tourne en arrière-plan sur le corpus entier. Le cache stocke
       le statut et remplace le rapport dès que l'enrichissement est terminé.

Redis peut remplacer ce module en production (v2) — l'interface reste identique.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

EnrichStatus = Literal["none", "pending", "done"]


class ReportCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._qa_store: dict[str, Any] = {}
        self._enrich_status: dict[str, EnrichStatus] = {}  # key → statut enrichissement

    @staticmethod
    def _key(video_id: str, topic: str) -> str:
        """Clé de cache incluant le topic normalisé (FR-82)."""
        return f"{video_id}::{topic.strip().lower()}"

    # ── Rapport principal ──────────────────────────────────────────────────────

    def get(self, video_id: str, topic: str) -> Optional[dict[str, Any]]:
        return self._store.get(self._key(video_id, topic))

    def set(self, video_id: str, topic: str, report: dict[str, Any]) -> None:
        self._store[self._key(video_id, topic)] = report

    def exists(self, video_id: str, topic: str) -> bool:
        """True si un rapport est déjà en cache pour cette (video_id, topic) (FR-83)."""
        return self._key(video_id, topic) in self._store

    def get_latest(self, video_id: str) -> Optional[dict[str, Any]]:
        """Retourne le rapport le plus récent pour un video_id (tous topics confondus)."""
        matches = {k: v for k, v in self._store.items() if k.startswith(f"{video_id}::")}
        if not matches:
            return None
        return list(matches.values())[-1]

    # ── Contexte Q&A ──────────────────────────────────────────────────────────

    def set_qa_context(self, video_id: str, context: dict[str, Any]) -> None:
        """
        Stocke le contexte Q&A pour un video_id (PRD v1.1 §4.4 — FR-86).

        Le contexte est indexé uniquement par video_id (pas par topic)
        car la transcription et les top_comments sont indépendants du topic.

        Args:
            video_id: Identifiant YouTube de la vidéo.
            context:  {
                transcript: [{text, start, duration}],
                transcript_available: bool,
                top_comments: [{comment_id, text, score_a4}],
                video_title: str,
            }
        """
        self._qa_store[video_id] = context

    def get_qa_context(self, video_id: str) -> Optional[dict[str, Any]]:
        """
        Retourne le contexte Q&A pour un video_id, ou None si absent (FR-87).
        """
        return self._qa_store.get(video_id)

    def has_qa_context(self, video_id: str) -> bool:
        """True si un contexte Q&A est disponible pour ce video_id."""
        return video_id in self._qa_store

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def clear_video(self, video_id: str) -> int:
        """Supprime toutes les entrées de cache pour un video_id donné (tous topics)."""
        keys_to_del = [k for k in self._store if k.startswith(f"{video_id}::")]
        for k in keys_to_del:
            del self._store[k]
        self._qa_store.pop(video_id, None)
        return len(keys_to_del)

    # ── Enrichissement background (v1.2) ──────────────────────────────────────

    def get_enrich_status(self, video_id: str, topic: str) -> EnrichStatus:
        """Retourne le statut d'enrichissement : 'none' | 'pending' | 'done'."""
        return self._enrich_status.get(self._key(video_id, topic), "none")

    def set_enrich_status(self, video_id: str, topic: str, status: EnrichStatus) -> None:
        self._enrich_status[self._key(video_id, topic)] = status

    def set_enriched(self, video_id: str, topic: str, report: dict[str, Any]) -> None:
        """
        Remplace le rapport rapide par le rapport enrichi (corpus complet).
        Marque le statut 'done' et met à jour le flag dans le rapport.
        """
        report = {**report, "enriched": True}
        self._store[self._key(video_id, topic)] = report
        self._enrich_status[self._key(video_id, topic)] = "done"

    def clear(self) -> None:
        self._store.clear()
        self._qa_store.clear()
        self._enrich_status.clear()


# Module-level singleton partagé par toute l'application FastAPI
cache = ReportCache()
