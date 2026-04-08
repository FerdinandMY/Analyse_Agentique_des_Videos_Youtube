"""
api/background.py — Worker d'enrichissement background
========================================================
Principe (deux phases) :

  Phase 1 — Réponse rapide (dans la requête HTTP)
    A2 cap = 300 commentaires → scores calculés → réponse servie en ~8s
    Le rapport est marqué enriched=False.

  Phase 2 — Enrichissement (thread daemon, hors requête)
    Pipeline complet relancé avec TOUS les commentaires nettoyés
    (pas de cap A2_MAX_COMMENTS). Le rapport enrichi remplace l'entrée
    rapide dans le cache. Le flag enriched=True permet au client de
    détecter la mise à jour disponible (polling GET /report/{video_id}).

Déclenchement :
    from api.background import enrich_in_background
    enrich_in_background(video_id, topic, lang, raw_comments,
                         source, transcript, transcript_available,
                         video_title, video_description)

Thread safety :
    Chaque enrichissement tourne dans un daemon thread indépendant.
    Le cache est un singleton module-level (api.cache.cache) — les
    dict Python sont thread-safe pour les opérations get/set unitaires.
"""
from __future__ import annotations

import threading
from typing import Any

from utils.logger import get_logger

logger = get_logger("background_enricher")


def _run_enrichment(
    video_id:            str,
    topic:               str,
    lang:                str,
    raw_comments:        list[dict],
    source:              str | None,
    transcript:          list | None,
    transcript_available: bool | None,
    video_title:         str,
    video_description:   str,
) -> None:
    """
    Exécute le pipeline complet sans cap sur le nombre de commentaires.
    Appelé dans un thread daemon — ne bloque jamais la réponse HTTP.
    """
    from api.cache import cache
    from api.qa import extract_top_comments

    key_log = f"video_id={video_id} topic={topic!r}"
    logger.info("enricher: debut enrichissement %s (%d commentaires)", key_log, len(raw_comments))

    try:
        # ── Désactive temporairement le cap A2 pour cet appel ─────────────────
        # On injecte directement tous les commentaires nettoyés sans passer
        # par run_pipeline() pour éviter de re-télécharger depuis A0.
        import os
        original_cap = os.environ.get("A2_MAX_COMMENTS")
        os.environ["A2_MAX_COMMENTS"] = str(len(raw_comments))  # pas de cap

        try:
            from graph import run_pipeline
            report = run_pipeline(
                video_id=video_id,
                topic=topic,
                lang=lang,
                raw_comments=raw_comments,
                thread_id=f"{video_id}-enrich",
                source=source,
                transcript=transcript,
                transcript_available=transcript_available,
                video_title=video_title,
                video_description=video_description,
            )
        finally:
            # Restaure la variable d'environnement dans tous les cas
            if original_cap is None:
                os.environ.pop("A2_MAX_COMMENTS", None)
            else:
                os.environ["A2_MAX_COMMENTS"] = original_cap

        # ── Mise à jour du cache ───────────────────────────────────────────────
        report["comment_count_full"] = len(raw_comments)

        # Mise à jour du contexte Q&A avec TOUS les top_comments
        discourse_result = (report.get("details") or {}).get("discourse") or {}
        cleaned_comments = report.get("cleaned_comments") or raw_comments
        top_comments = extract_top_comments(
            cleaned_comments=cleaned_comments,
            discourse_result=discourse_result,
            threshold=0.7,
        )
        cache.set_qa_context(video_id, {
            "transcript":           transcript or [],
            "transcript_available": transcript_available or False,
            "top_comments":         top_comments,
            "video_title":          video_title,
            "video_description":    video_description,
        })

        cache.set_enriched(video_id, topic, report)
        logger.info(
            "enricher: enrichissement termine %s — %d commentaires, enriched=True",
            key_log, len(raw_comments),
        )

    except Exception as exc:
        # Enrichissement non bloquant — on log l'erreur et on continue
        logger.error("enricher: echec enrichissement %s — %s", key_log, exc)
        from api.cache import cache
        cache.set_enrich_status(video_id, topic, "done")  # évite une boucle infinie


def enrich_in_background(
    video_id:             str,
    topic:                str,
    lang:                 str,
    raw_comments:         list[dict],
    source:               str | None       = None,
    transcript:           list | None      = None,
    transcript_available: bool | None      = None,
    video_title:          str              = "",
    video_description:    str              = "",
) -> None:
    """
    Lance l'enrichissement dans un thread daemon.

    Vérifie d'abord si l'enrichissement est déjà en cours ou terminé
    pour éviter les doublons sur des requêtes rapprochées.

    Le corpus complet est déjà en mémoire (raw_comments) — pas de
    re-téléchargement API.
    """
    from api.cache import cache

    status = cache.get_enrich_status(video_id, topic)
    if status in ("pending", "done"):
        logger.info(
            "enricher: enrichissement deja %s pour video_id=%s topic=%r — ignore",
            status, video_id, topic,
        )
        return

    # Vérifier si le corpus complet apporterait un gain réel
    # (inutile de relancer si on avait déjà tous les commentaires)
    import os
    cap = int(os.environ.get("A2_MAX_COMMENTS", "300"))
    if len(raw_comments) <= cap:
        logger.info(
            "enricher: corpus (%d) <= cap (%d) — enrichissement non necessaire",
            len(raw_comments), cap,
        )
        cache.set_enrich_status(video_id, topic, "done")
        return

    cache.set_enrich_status(video_id, topic, "pending")
    logger.info(
        "enricher: lancement thread daemon — video_id=%s %d commentaires (cap=%d)",
        video_id, len(raw_comments), cap,
    )

    t = threading.Thread(
        target=_run_enrichment,
        args=(
            video_id, topic, lang, raw_comments,
            source, transcript, transcript_available,
            video_title, video_description,
        ),
        daemon=True,
        name=f"enrich-{video_id[:8]}",
    )
    t.start()
