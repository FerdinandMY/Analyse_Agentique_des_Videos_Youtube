"""
api/cache.py — In-Memory Report Cache
=======================================
Caches pipeline reports keyed by (video_id, topic) to avoid re-running
the full pipeline for the same request.

Redis can replace this in production (v2) — the interface stays identical.
"""
from __future__ import annotations

from typing import Any, Optional


class ReportCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    @staticmethod
    def _key(video_id: str, topic: str) -> str:
        return f"{video_id}::{topic.strip().lower()}"

    def get(self, video_id: str, topic: str) -> Optional[dict[str, Any]]:
        return self._store.get(self._key(video_id, topic))

    def set(self, video_id: str, topic: str, report: dict[str, Any]) -> None:
        self._store[self._key(video_id, topic)] = report

    def get_latest(self, video_id: str) -> Optional[dict[str, Any]]:
        """Return the most recently cached report for a video_id (any topic)."""
        matches = {k: v for k, v in self._store.items() if k.startswith(f"{video_id}::")}
        if not matches:
            return None
        # Return last inserted
        return list(matches.values())[-1]

    def clear(self) -> None:
        self._store.clear()


# Module-level singleton shared across the FastAPI app lifetime
cache = ReportCache()
