"""
utils/language_detector.py — Language Detection (best-effort)
"""
from __future__ import annotations

from typing import Optional


def detect_language(text: Optional[str], default: str = "en") -> str:
    """
    Detect language using langdetect if available, else return default.
    """
    if not text:
        return default
    try:
        from langdetect import detect  # type: ignore
        return detect(text)
    except Exception:
        return default
