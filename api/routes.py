"""
api/routes.py — API Endpoints
================================
POST /analyze  — Run the pipeline for a video URL/id + topic
                 (A0 collecte les commentaires si non fournis en pré-collecté)
GET  /report/{video_id} — Return cached report
POST /ask      — Module Q&A : questions en langage naturel sur une vidéo analysée (FR-87)

v1.1 : intégration A0 Collector + endpoint /ask (PRD v1.1 §2 et §6)
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from api.cache import cache
from api.qa import answer_question, extract_top_comments
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AskRequest,
    AskResponse,
)
from graph import run_pipeline
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("api.routes")


# ── POST /analyze ─────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, summary="Run quality analysis pipeline")
def analyze(request: AnalyzeRequest) -> Any:
    """
    Lance le pipeline 7-agents pour une vidéo YouTube.

    - **video_id**: URL YouTube ou identifiant direct (11 caractères)
    - **topic**: Thématique utilisateur (optionnel — améliore A7)
    - **comments**: Commentaires pré-collectés (optionnel — A0 collecte si absent)

    Logique de décision (PRD v1.1 §2) :
    1. Cache HIT  → retour immédiat (< 3s)
    2. Cache MISS + comments fournis → pipeline A1→A7 direct
    3. Cache MISS + comments absents → A0 collecte via YouTube API, puis A1→A7
    """
    # ── 1. Résolution du video_id ──────────────────────────────────────────────
    url_or_id = request.video_id  # peut être une URL complète ou un ID direct
    video_id = url_or_id

    # Tenter d'extraire le video_id propre depuis une URL (sans importer A0 si non nécessaire)
    try:
        from agents.a0_collector import extract_video_id
        video_id = extract_video_id(url_or_id)
    except ValueError:
        # Pas une URL YouTube reconnue — on continue avec la valeur brute
        pass

    # ── 2. Vérification cache (FR-78) ─────────────────────────────────────────
    if not request.force_refresh:
        cached = cache.get(video_id, request.topic)
        if cached:
            logger.info("Cache hit pour video_id=%s topic=%s", video_id, request.topic)
            return cached
    else:
        logger.info("force_refresh=True — cache ignoré pour video_id=%s topic=%s", video_id, request.topic)

    # ── 3. Collecte A0 si commentaires absents (PRD v1.1 §2) ──────────────────
    source:               str | None  = None
    quota_used:           int | None  = None
    collected_at:         str | None  = None
    transcript:           list | None = None
    transcript_available: bool | None = None
    video_title:          str | None  = None
    video_description:    str | None  = None
    raw_comments: list[dict] | None   = None

    _MIN_COMMENTS_THRESHOLD = 5  # En dessous, on ignore les comments fournis et on appelle A0
    use_preloaded = request.comments and len(request.comments) >= _MIN_COMMENTS_THRESHOLD

    if use_preloaded:
        # Commentaires pré-fournis en nombre suffisant → bypass A0
        raw_comments = [c.model_dump() for c in request.comments]
        source = "pre_loaded"
        logger.info(
            "analyze: commentaires pré-fournis video_id=%s count=%d",
            video_id, len(raw_comments),
        )
    else:
        # Pas de commentaires, ou trop peu (<5) → invoquer A0
        if request.comments:
            logger.warning(
                "analyze: %d commentaire(s) fournis < seuil %d — A0 invoqué à la place",
                len(request.comments), _MIN_COMMENTS_THRESHOLD,
            )
        else:
            logger.info("analyze: invocation A0 Collector pour video_id=%s", video_id)
        try:
            from agents.a0_collector import a0_collector

            a0_state = a0_collector({
                "url_or_id": url_or_id,
                "video_id":  video_id,
                "topic":     request.topic,
            })

            a0_errors = a0_state.get("errors") or []
            if a0_errors:
                # Erreurs bloquantes : vidéo introuvable, commentaires désactivés, etc.
                blocking = [e for e in a0_errors if any(
                    code in e for code in [
                        "VIDEO_NOT_FOUND", "COMMENTS_DISABLED",
                        "API_KEY_INVALID", "INSUFFICIENT_COMMENTS",
                    ]
                )]
                if blocking:
                    raise HTTPException(status_code=422, detail={"errors": blocking})

            raw_comments         = a0_state.get("raw_comments") or []
            source               = a0_state.get("source", "api_v3")
            quota_used           = a0_state.get("quota_used")
            collected_at         = a0_state.get("collected_at")
            transcript           = a0_state.get("transcript")
            transcript_available = a0_state.get("transcript_available")
            video_title          = a0_state.get("video_title", "")
            video_description    = a0_state.get("video_description", "")

            # Récupérer le video_id normalisé retourné par A0
            if a0_state.get("video_id"):
                video_id = a0_state["video_id"]

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("analyze: A0 Collector error — %s", exc)
            raise HTTPException(status_code=500, detail=f"A0 Collector error: {exc}") from exc

    thread_id = f"{video_id}-{uuid.uuid4().hex[:8]}"
    logger.info(
        "analyze: video_id=%s topic=%s comments=%d source=%s thread=%s",
        video_id, request.topic,
        len(raw_comments) if raw_comments else 0,
        source, thread_id,
    )

    # ── 4. Exécution du pipeline A1→A7 ───────────────────────────────────────
    try:
        report = run_pipeline(
            video_id=video_id,
            topic=request.topic,
            lang=request.lang,
            raw_comments=raw_comments,
            thread_id=thread_id,
            source=source,
            quota_used=quota_used,
            collected_at=collected_at,
            transcript=transcript,
            transcript_available=transcript_available,
            video_title=video_title,
            video_description=video_description,
        )
    except Exception as exc:
        logger.error("analyze: pipeline error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # ── 5. Assemblage de la réponse ───────────────────────────────────────────
    response = AnalyzeResponse(
        video_id=report.get("video_id") or video_id,
        topic=report.get("topic") or request.topic,
        lang=report.get("lang") or request.lang,
        score_global=report.get("score_global") or 0.0,
        score_pertinence=report.get("score_pertinence") or 0.0,
        score_final=report.get("score_final") or 0.0,
        quality_label=report.get("quality_label") or "Moyen",
        summary=report.get("summary"),
        topic_verdict=report.get("topic_verdict"),
        details=report.get("details"),
        comment_count=report.get("comment_count") or (len(raw_comments) if raw_comments else 0),
        errors=report.get("errors") or [],
        hallucination_flags=report.get("hallucination_flags") or [],
        fallback_used=bool(report.get("fallback_used", False)),
        sc_consensus=report.get("sc_consensus"),
        low_consensus=report.get("low_consensus"),
        source=source or report.get("source"),
        quota_used=quota_used or report.get("quota_used"),
    )

    # ── 6. Écriture cache rapport ─────────────────────────────────────────────
    cache.set(video_id, request.topic, response.model_dump())

    # ── 7. Écriture cache Q&A context (FR-86) ────────────────────────────────
    # Extraire les top_comments A4 (score >= 0.7) pour le module Q&A
    discourse_result = (report.get("details") or {}).get("discourse") or {}
    cleaned_comments = report.get("cleaned_comments") or []

    # Fallback : si cleaned_comments absent du report, utiliser raw_comments
    if not cleaned_comments and raw_comments:
        cleaned_comments = raw_comments

    top_comments = extract_top_comments(
        cleaned_comments=cleaned_comments,
        discourse_result=discourse_result,
        threshold=0.7,
    )

    qa_ctx: dict[str, Any] = {
        "transcript":           transcript or [],
        "transcript_available": transcript_available if transcript_available is not None else False,
        "top_comments":         top_comments,
        "video_title":          report.get("video_title") or video_title or "",
        "video_description":    report.get("video_description") or video_description or "",
    }
    cache.set_qa_context(video_id, qa_ctx)
    logger.info(
        "analyze: qa_context écrit — transcript=%s top_comments=%d",
        transcript_available, len(top_comments),
    )

    return response


# ── GET /report/{video_id} ────────────────────────────────────────────────────

@router.get(
    "/report/{video_id}",
    response_model=AnalyzeResponse,
    summary="Retrieve cached report",
    responses={404: {"description": "Report not found"}},
)
def get_report(video_id: str) -> Any:
    """
    Retourne le rapport d'analyse le plus récent en cache pour une vidéo.

    Retourne **404** si aucun rapport n'existe — lancer `POST /analyze` d'abord.
    """
    report = cache.get_latest(video_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun rapport en cache pour video_id='{video_id}'. Lancer POST /analyze d'abord.",
        )
    return report


# ── POST /ask ─────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Q&A sur une vidéo analysée (grounded sur transcription + commentaires)",
    responses={
        404: {"description": "Aucun rapport en cache — analyser la vidéo d'abord"},
        422: {"description": "Question trop longue (> 500 caractères)"},
    },
)
def ask(request: AskRequest) -> Any:
    """
    Répond à une question en langage naturel sur une vidéo YouTube déjà analysée.

    - **video_id**: Identifiant YouTube de la vidéo (doit être en cache)
    - **question**: Question en langage naturel (max 500 caractères)
    - **history**: Historique de conversation (max 5 tours conservés)

    Le module Q&A s'appuie exclusivement sur :
    - La transcription vidéo (si disponible)
    - Les commentaires de haute qualité sélectionnés par A4 (score >= 0.7)

    Principe anti-hallucination FR-88 : aucune réponse sans ancrage explicite sur le contexte.
    """
    # ── 1. Validation de la question ──────────────────────────────────────────
    if not request.question.strip():
        raise HTTPException(status_code=422, detail={"error_code": "EMPTY_QUESTION"})

    # ── 2. Vérification du rapport en cache (FR-87) ───────────────────────────
    # Essayer de résoudre le video_id si c'est une URL
    video_id = request.video_id
    try:
        from agents.a0_collector import extract_video_id
        video_id = extract_video_id(request.video_id)
    except ValueError:
        pass

    qa_context = cache.get_qa_context(video_id)
    if qa_context is None:
        # Vérifier aussi si le rapport principal existe
        report = cache.get_latest(video_id)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "NO_REPORT_IN_CACHE",
                    "message": f"Analysez d'abord cette vidéo via POST /analyze (video_id='{video_id}')",
                },
            )
        # Rapport présent mais qa_context absent (pipeline lancé sans A0)
        qa_context = {
            "transcript":           [],
            "transcript_available": False,
            "top_comments":         [],
            "video_title":          "",
        }
        logger.warning("ask: qa_context absent pour video_id=%s — mode dégradé", video_id)

    # ── 3. Appel module Q&A ───────────────────────────────────────────────────
    history = [turn.model_dump() for turn in (request.history or [])]

    logger.info(
        "ask: video_id=%s question=%r history_turns=%d",
        video_id, request.question[:80], len(history),
    )

    result = answer_question(
        question=request.question,
        qa_context=qa_context,
        history=history,
    )

    # ── 4. Construction de la réponse ─────────────────────────────────────────
    return AskResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        transcript_used=result.get("transcript_used", False),
        fallback_used=result.get("fallback_used", False),
    )


# ── DELETE /cache ─────────────────────────────────────────────────────────────

@router.delete(
    "/cache",
    summary="Vider le cache (tous les rapports et contextes Q&A)",
    tags=["admin"],
)
def clear_cache() -> dict[str, str]:
    """
    Vide entièrement le cache en mémoire.
    Utile pour forcer une re-collecte sans redémarrer le serveur.
    """
    cache.clear()
    logger.info("Cache vidé via DELETE /cache")
    return {"status": "ok", "message": "Cache vidé."}


@router.get(
    "/debug/qa_context/{video_id}",
    summary="[DEBUG] Inspecter le qa_context en cache pour une vidéo",
    tags=["admin"],
)
def debug_qa_context(video_id: str) -> Any:
    ctx = cache.get_qa_context(video_id)
    if ctx is None:
        return {"qa_context": None, "message": "Absent du cache"}
    return {
        "transcript_available": ctx.get("transcript_available"),
        "transcript_segments":  len(ctx.get("transcript") or []),
        "top_comments_count":   len(ctx.get("top_comments") or []),
        "top_comments_preview": [c.get("text", "")[:80] for c in (ctx.get("top_comments") or [])[:5]],
        "video_title":          ctx.get("video_title"),
    }


@router.delete(
    "/cache/{video_id}",
    summary="Vider le cache pour une vidéo spécifique",
    tags=["admin"],
)
def clear_cache_for_video(video_id: str) -> dict[str, str]:
    """
    Supprime toutes les entrées de cache pour un video_id donné (tous topics).
    """
    removed = cache.clear_video(video_id)
    logger.info("Cache vidé pour video_id=%s (%d entrées supprimées)", video_id, removed)
    return {"status": "ok", "video_id": video_id, "entries_removed": str(removed)}
