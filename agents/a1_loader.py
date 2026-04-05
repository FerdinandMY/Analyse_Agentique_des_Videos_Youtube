"""
A1 — Loader / Validator
========================
Loads and validates a CSV file of pre-collected YouTube comments.

Expected CSV columns (at minimum):
    text          — comment body
    video_id      — YouTube video identifier (optional but recommended)
    author_likes  — likes on the comment (optional)
    reply_count   — number of replies (optional)
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from pipeline_state import PipelineState
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger

logger = get_logger("a1_loader")

REQUIRED_COLUMNS = {"text"}
OPTIONAL_COLUMNS = {"video_id", "author_likes", "reply_count"}


def a1_loader(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node — A1 Loader / Validator.

    Reads state["csv_path"] and returns {"raw_comments": [...]}.
    Falls back to state["raw_comments"] if already populated (e.g. tests).
    """
    # Allow callers to bypass CSV loading by pre-populating raw_comments.
    if state.get("raw_comments"):
        logger.info("a1_loader: raw_comments already provided, skipping CSV load.")
        return {}

    csv_path = state.get("csv_path", "")
    if not csv_path:
        return {"errors": ["a1_loader: csv_path is missing from state."]}

    if not os.path.isfile(csv_path):
        return {"errors": [f"a1_loader: file not found — {csv_path}"]}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return {"errors": [f"a1_loader: failed to read CSV — {exc}"]}

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return {"errors": [f"a1_loader: missing required columns: {missing}"]}

    # Drop rows with empty text
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Build raw_comments records
    raw_comments: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {"text": row["text"]}
        for col in OPTIONAL_COLUMNS:
            if col in df.columns:
                record[col] = row[col]
        raw_comments.append(record)

    logger.info("a1_loader: loaded %d comments from %s", len(raw_comments), csv_path)
    save_checkpoint("a1_loader", {"count": len(raw_comments)})

    return {"raw_comments": raw_comments}
