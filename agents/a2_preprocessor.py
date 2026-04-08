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

# Nombre maximum de commentaires transmis aux agents analytiques (A3/A4/A5).
# Au-delà, un sampling stratifié est appliqué : les commentaires les plus
# likés sont prioritaires, le reste est échantillonné uniformément.
# Réduire si la latence est prioritaire sur la couverture (min conseillé : 100).
MAX_COMMENTS_FOR_ANALYSIS = int(
    __import__("os").environ.get("A2_MAX_COMMENTS", "300")
)


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

    # ── Sampling stratifié si trop de commentaires ────────────────────────────
    # On garde MAX_COMMENTS_FOR_ANALYSIS commentaires au maximum pour les agents
    # analytiques. Stratégie : top-N par likes + échantillon uniforme du reste.
    sampled = cleaned
    if len(cleaned) > MAX_COMMENTS_FOR_ANALYSIS:
        # Trier par author_likes décroissant pour prioriser les commentaires
        # les plus engageants (signal qualité fort)
        sorted_by_likes = sorted(
            cleaned,
            key=lambda c: int(c.get("author_likes") or 0),
            reverse=True,
        )
        top_n     = MAX_COMMENTS_FOR_ANALYSIS // 2
        top_liked = sorted_by_likes[:top_n]

        # Echantillonnage uniforme du reste (diversité)
        rest       = sorted_by_likes[top_n:]
        rest_quota = MAX_COMMENTS_FOR_ANALYSIS - top_n
        step       = max(1, len(rest) // rest_quota)
        rest_sample = rest[::step][:rest_quota]

        sampled = top_liked + rest_sample
        logger.info(
            "a2_preprocessor: sampling %d → %d (top_likes=%d + uniform=%d)",
            len(cleaned), len(sampled), len(top_liked), len(rest_sample),
        )

    logger.info(
        "a2_preprocessor: %d → %d cleaned → %d sampled for analysis",
        len(raw), len(cleaned), len(sampled),
    )
    save_checkpoint("a2_preprocessor", {"input": len(raw), "output": len(cleaned), "sampled": len(sampled)})

    return {"cleaned_comments": sampled}
