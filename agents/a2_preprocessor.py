"""
A2 — Preprocessor
==================
Cleans and normalises raw comments:
  - text normalisation (lowercase, collapse whitespace)
  - length filtering (3–2000 chars)
  - exact-duplicate removal
  - language detection (best-effort via langdetect)
"""
from __future__ import annotations

from typing import Any

from pipeline_state import PipelineState
from utils.checkpoint import save_checkpoint
from utils.language_detector import detect_language
from utils.logger import get_logger
from utils.text_cleaner import normalize_text

logger = get_logger("a2_preprocessor")

MIN_CHARS = 3
MAX_CHARS = 2000


def _extract_text(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("comment") or item.get("content")
    return None


def a2_preprocessor(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node — A2 Preprocessor.
    """
    raw = state.get("raw_comments") or []
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    for item in raw:
        text = _extract_text(item)
        if text is None:
            continue

        norm = normalize_text(text)
        if norm is None:
            continue

        if len(norm) < MIN_CHARS:
            continue
        if len(norm) > MAX_CHARS:
            norm = norm[:MAX_CHARS]

        if norm in seen:
            continue
        seen.add(norm)

        record: dict[str, Any] = {"text": text, "cleaned_text": norm}

        # Preserve extra metadata (author_likes, reply_count, video_id, …)
        if isinstance(item, dict):
            for k, v in item.items():
                if k not in record:
                    record[k] = v

        record["language"] = detect_language(text)
        cleaned.append(record)

    logger.info("a2_preprocessor: %d → %d comments after preprocessing", len(raw), len(cleaned))
    save_checkpoint("a2_preprocessor", {"input": len(raw), "output": len(cleaned)})

    return {"cleaned_comments": cleaned}
