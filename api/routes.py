"""
api/routes.py — API Endpoints
================================
POST /analyze  — Run the pipeline for a video_id + topic + comments
GET  /report/{video_id} — Return cached report
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from api.cache import cache
from api.schemas import AnalyzeRequest, AnalyzeResponse
from graph import run_pipeline
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("api.routes")


@router.post("/analyze", response_model=AnalyzeResponse, summary="Run quality analysis pipeline")
def analyze(request: AnalyzeRequest) -> Any:
    """
    Launch the 7-agent pipeline for a video.

    - **video_id**: YouTube video identifier
    - **topic**: User thematic query (optional, improves personalisation via A7)
    - **comments**: List of pre-collected comment objects `{text, author_likes?, reply_count?}`
    """
    # Check cache first
    cached = cache.get(request.video_id, request.topic)
    if cached:
        logger.info("Cache hit for video_id=%s topic=%s", request.video_id, request.topic)
        return cached

    raw_comments = [c.model_dump() for c in request.comments]
    thread_id = f"{request.video_id}-{uuid.uuid4().hex[:8]}"

    logger.info(
        "analyze: video_id=%s topic=%s comments=%d thread=%s",
        request.video_id,
        request.topic,
        len(raw_comments),
        thread_id,
    )

    try:
        report = run_pipeline(
            video_id=request.video_id,
            topic=request.topic,
            raw_comments=raw_comments,
            thread_id=thread_id,
        )
    except Exception as exc:
        logger.error("analyze: pipeline error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # Normalise to response schema
    response = AnalyzeResponse(
        video_id=report.get("video_id") or request.video_id,
        topic=report.get("topic") or request.topic,
        score_global=report.get("score_global") or 0.0,
        score_pertinence=report.get("score_pertinence") or 0.0,
        score_final=report.get("score_final") or 0.0,
        quality_label=report.get("quality_label") or "Moyen",
        summary=report.get("summary"),
        topic_verdict=report.get("topic_verdict"),
        details=report.get("details"),
        comment_count=report.get("comment_count") or len(raw_comments),
        errors=report.get("errors") or [],
        hallucination_flags=report.get("hallucination_flags") or [],
        fallback_used=bool(report.get("fallback_used", False)),
        sc_consensus=report.get("sc_consensus"),
        low_consensus=report.get("low_consensus"),
    )

    cache.set(request.video_id, request.topic, response.model_dump())
    return response


@router.get(
    "/report/{video_id}",
    response_model=AnalyzeResponse,
    summary="Retrieve cached report",
    responses={404: {"description": "Report not found"}},
)
def get_report(video_id: str) -> Any:
    """
    Return the most recently cached analysis report for a video.

    Returns **404** if no report exists — run `POST /analyze` first.
    """
    report = cache.get_latest(video_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached report found for video_id='{video_id}'. Run POST /analyze first.",
        )
    return report
