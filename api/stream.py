"""
api/stream.py — Endpoint SSE POST /analyze/stream
===================================================
Retourne les résultats du pipeline progressivement via Server-Sent Events (SSE).
L'utilisateur voit les scores apparaître au fur et à mesure :

  t=0s   → event: started       (collecte lancée)
  t+2s   → event: collected     (A0 terminé — nb commentaires, transcript)
  t+3s   → event: preprocessed  (A2 terminé — nb commentaires retenus)
  t+6s   → event: scores        (A3+A4+A5 terminés — scores partiels)
  t+7s   → event: global        (A6 terminé — score global + label)
  t+8s   → event: final         (A7 terminé — score final + verdict complet)
  t+8s   → event: done          (fin du stream)

Format SSE :
  data: {"event": "scores", "sentiment": 72.5, "discourse": 68.0, "noise": 85.0}

Utilisation côté client (JS) :
    const es = new EventSource('/analyze/stream?video_id=...&topic=...');
    es.onmessage = (e) => updateUI(JSON.parse(e.data));
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Generator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api.cache import cache
from utils.logger import get_logger

router_stream = APIRouter()
logger = get_logger("api.stream")


# ── Helpers SSE ───────────────────────────────────────────────────────────────

def _sse(event: str, data: dict[str, Any]) -> str:
    """Formate un événement SSE : 'data: {...}\\n\\n'."""
    payload = json.dumps({"event": event, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _error_sse(message: str, code: str = "PIPELINE_ERROR") -> str:
    return _sse("error", {"error_code": code, "message": message})


# ── Générateur principal ──────────────────────────────────────────────────────

def _stream_pipeline(
    url_or_id: str,
    topic: str,
    lang: str,
    force_refresh: bool,
) -> Generator[str, None, None]:
    """
    Générateur SSE qui pilote le pipeline agent par agent et émet un événement
    dès que chaque étape est terminée.
    """
    t0 = time.time()

    # ── Résolution video_id ───────────────────────────────────────────────────
    video_id = url_or_id
    try:
        from agents.a0_collector import extract_video_id
        video_id = extract_video_id(url_or_id)
    except ValueError as exc:
        yield _error_sse(str(exc), "INVALID_VIDEO_ID")
        return

    yield _sse("started", {"video_id": video_id, "topic": topic})

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if not force_refresh:
        cached = cache.get(video_id, topic)
        if cached:
            logger.info("stream: cache hit video_id=%s", video_id)
            yield _sse("cache_hit", {"video_id": video_id})
            yield _sse("final", cached)
            yield _sse("done", {"elapsed_s": round(time.time() - t0, 2)})
            return

    # ── A0 : Collecte ─────────────────────────────────────────────────────────
    source = transcript = transcript_available = None
    video_title = video_description = ""
    raw_comments: list[dict] = []

    try:
        from agents.a0_collector import a0_collector
        a0_state = a0_collector({"url_or_id": url_or_id, "video_id": video_id, "topic": topic})

        blocking = [e for e in (a0_state.get("errors") or []) if any(
            c in e for c in ["VIDEO_NOT_FOUND", "COMMENTS_DISABLED", "INSUFFICIENT_COMMENTS"]
        )]
        if blocking:
            yield _error_sse(blocking[0], "COLLECTION_ERROR")
            return

        raw_comments         = a0_state.get("raw_comments") or []
        source               = a0_state.get("source", "api_v3")
        transcript           = a0_state.get("transcript")
        transcript_available = a0_state.get("transcript_available", False)
        video_title          = a0_state.get("video_title", "")
        video_description    = a0_state.get("video_description", "")

        if a0_state.get("video_id"):
            video_id = a0_state["video_id"]

    except Exception as exc:
        logger.error("stream: A0 error — %s", exc)
        yield _error_sse(f"Collection error: {exc}", "A0_ERROR")
        return

    yield _sse("collected", {
        "comment_count":      len(raw_comments),
        "source":             source,
        "transcript_available": transcript_available,
        "video_title":        video_title,
        "elapsed_s":          round(time.time() - t0, 2),
    })

    # ── A1 + A2 : Load + Preprocess ───────────────────────────────────────────
    try:
        from pipeline_state import PipelineState
        from agents.a1_loader import a1_loader
        from agents.a2_preprocessor import a2_preprocessor

        state: PipelineState = {
            "video_id": video_id,
            "topic":    topic,
            "lang":     lang,
            "errors":   [],
            "raw_comments": raw_comments,
            "source":               source,
            "transcript":           transcript or [],
            "transcript_available": transcript_available or False,
            "video_title":          video_title,
            "video_description":    video_description,
        }

        a1_out = a1_loader(state)
        state.update(a1_out)

        a2_out = a2_preprocessor(state)
        state.update(a2_out)

        cleaned = state.get("cleaned_comments") or []
    except Exception as exc:
        logger.error("stream: A1/A2 error — %s", exc)
        yield _error_sse(f"Preprocessing error: {exc}", "A2_ERROR")
        return

    yield _sse("preprocessed", {
        "comments_retained": len(cleaned),
        "elapsed_s":         round(time.time() - t0, 2),
    })

    # ── A3 + A4 + A5 : Analyse parallèle ──────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agents.a3_sentiment import a3_sentiment
    from agents.a4_discourse import a4_discourse
    from agents.a5_noise     import a5_noise

    sentiment = discourse = noise = None

    def _run_agent(fn, s):
        return fn.__name__, fn(s)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(_run_agent, a3_sentiment, state): "a3",
            pool.submit(_run_agent, a4_discourse,  state): "a4",
            pool.submit(_run_agent, a5_noise,      state): "a5",
        }
        for fut in as_completed(futs):
            try:
                name, result = fut.result()
                state.update(result)
                if name == "a3_sentiment":
                    sentiment = state.get("sentiment") or {}
                elif name == "a4_discourse":
                    discourse = state.get("discourse") or {}
                elif name == "a5_noise":
                    noise = state.get("noise") or {}
            except Exception as exc:
                logger.warning("stream: agent error — %s", exc)

    yield _sse("scores", {
        "sentiment_score": (sentiment or {}).get("sentiment_score"),
        "sentiment_label": (sentiment or {}).get("sentiment_label"),
        "discourse_score": (discourse or {}).get("discourse_score"),
        "noise_score":     (noise or {}).get("noise_score"),
        "elapsed_s":       round(time.time() - t0, 2),
    })

    # ── A6 : Synthèse ─────────────────────────────────────────────────────────
    from agents.a6_synthesizer import a6_synthesizer
    try:
        a6_out = a6_synthesizer(state)
        state.update(a6_out)
    except Exception as exc:
        logger.error("stream: A6 error — %s", exc)

    synthesis    = state.get("synthesis") or {}
    score_global = state.get("score_global", 0.0)

    yield _sse("global", {
        "score_global":  score_global,
        "quality_label": synthesis.get("quality_label"),
        "summary":       synthesis.get("summary"),
        "elapsed_s":     round(time.time() - t0, 2),
    })

    # ── A7 : Topic Matcher ────────────────────────────────────────────────────
    from agents.a7_topic_matcher import a7_topic_matcher
    try:
        a7_out = a7_topic_matcher(state)
        state.update(a7_out)
    except Exception as exc:
        logger.error("stream: A7 error — %s", exc)

    # ── Assemblage rapport final ───────────────────────────────────────────────
    from graph import _assemble_report
    final_state = _assemble_report(state)
    state.update(final_state)
    report = state.get("report") or state

    # Ecriture cache
    from api.qa import extract_top_comments
    discourse_result = (report.get("details") or {}).get("discourse") or {}
    top_comments = extract_top_comments(
        cleaned_comments=cleaned,
        discourse_result=discourse_result,
        threshold=0.7,
    )
    cache.set(video_id, topic, report)
    cache.set_qa_context(video_id, {
        "transcript":           transcript or [],
        "transcript_available": transcript_available or False,
        "top_comments":         top_comments,
        "video_title":          video_title,
        "video_description":    video_description,
    })

    n_raw = len(raw_comments)
    import os
    cap = int(os.environ.get("A2_MAX_COMMENTS", "300"))
    enrich_status = "pending" if n_raw > cap else "none"

    yield _sse("final", {
        **{k: v for k, v in report.items() if k != "cleaned_comments"},
        "enriched":      False,
        "enrich_status": enrich_status,
        "elapsed_s":     round(time.time() - t0, 2),
    })

    # ── Enrichissement background ──────────────────────────────────────────────
    from api.background import enrich_in_background
    enrich_in_background(
        video_id=video_id,
        topic=topic,
        lang=lang,
        raw_comments=raw_comments,
        source=source,
        transcript=transcript,
        transcript_available=transcript_available,
        video_title=video_title,
        video_description=video_description,
    )

    yield _sse("done", {
        "elapsed_s":     round(time.time() - t0, 2),
        "enrich_status": enrich_status,
    })


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router_stream.post(
    "/analyze/stream",
    summary="Analyse progressive via Server-Sent Events (SSE)",
    response_class=StreamingResponse,
    tags=["stream"],
)
def analyze_stream(
    video_id:      str  = Query(...,   description="URL YouTube ou video_id"),
    topic:         str  = Query("",   description="Thématique utilisateur (optionnel)"),
    lang:          str  = Query("fr", description="Langue de sortie : 'fr' ou 'en'"),
    force_refresh: bool = Query(False, description="Ignore le cache et relance le pipeline"),
) -> StreamingResponse:
    """
    Lance le pipeline en streaming SSE.
    Chaque agent émet un événement dès qu'il termine :

    - `started`      : pipeline lancé
    - `cache_hit`    : résultat servi depuis le cache (pas de pipeline)
    - `collected`    : A0 terminé — nb commentaires, transcript disponible
    - `preprocessed` : A2 terminé — nb commentaires retenus pour l'analyse
    - `scores`       : A3/A4/A5 terminés — scores sentiment, discours, bruit
    - `global`       : A6 terminé — Score_Global + label qualité + résumé
    - `final`        : A7 terminé — rapport complet avec Score_Final
    - `done`         : fin du stream avec temps total
    - `error`        : erreur bloquante
    """
    thread_id = uuid.uuid4().hex[:8]
    logger.info("stream: video_id=%s topic=%r thread=%s", video_id, topic, thread_id)

    return StreamingResponse(
        _stream_pipeline(video_id, topic, lang, force_refresh),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # désactive le buffering nginx
        },
    )
