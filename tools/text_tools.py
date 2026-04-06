"""
tools/text_tools.py
====================
Tools LangChain pour A2 — PRD v3.0 §3.4

Tools :
  - detect_language : détecte la langue via langdetect
  - clean_text      : normalise les emojis, URLs, HTML
"""
from __future__ import annotations

import re
import html
from typing import Any

from langchain_core.tools import tool


# ── detect_language ───────────────────────────────────────────────────────────

@tool
def detect_language(text: str) -> dict[str, Any]:
    """
    Détecte la langue d'un texte via langdetect.
    Retourne {"language": "fr", "confidence": 0.99, "supported": true}.
    Fallback "unknown" si la détection échoue (texte trop court, etc.).
    """
    if not text or len(text.strip()) < 5:
        return {"language": "unknown", "confidence": 0.0, "supported": False}

    try:
        from langdetect import detect, detect_langs  # type: ignore
        langs = detect_langs(text)
        top   = langs[0]
        lang  = str(top.lang)
        conf  = round(float(top.prob), 4)
        supported = lang in {"fr", "en", "es", "de", "it", "pt", "nl"}
        return {"language": lang, "confidence": conf, "supported": supported}
    except Exception:
        try:
            from langdetect import detect  # type: ignore
            lang = detect(text)
            return {"language": lang, "confidence": 0.5, "supported": lang in {"fr", "en"}}
        except Exception:
            return {"language": "unknown", "confidence": 0.0, "supported": False}


# ── clean_text ────────────────────────────────────────────────────────────────

_URL_RE       = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE  = re.compile(r"<[^>]+>")
_EMOJI_RE     = re.compile(
    r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F]+",
    re.UNICODE,
)
_WHITESPACE_RE = re.compile(r"\s{2,}")


@tool
def clean_text(text: str) -> dict[str, Any]:
    """
    Nettoie un texte de commentaire YouTube :
    - Décode les entités HTML (&#39; → ', &amp; → &)
    - Supprime les balises HTML (<br>, <a>, etc.)
    - Remplace les URLs par [URL]
    - Normalise les emojis en [EMOJI]
    - Collapse les espaces multiples
    Retourne {"cleaned": str, "url_count": int, "emoji_count": int, "original_length": int}.
    """
    if not text:
        return {"cleaned": "", "url_count": 0, "emoji_count": 0, "original_length": 0}

    original_length = len(text)

    # Décode HTML
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)

    # Compte et remplace URLs
    urls    = _URL_RE.findall(cleaned)
    cleaned = _URL_RE.sub("[URL]", cleaned)

    # Compte et remplace emojis
    emojis  = _EMOJI_RE.findall(cleaned)
    cleaned = _EMOJI_RE.sub("[EMOJI]", cleaned)

    # Normalise espaces
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    return {
        "cleaned":         cleaned,
        "url_count":       len(urls),
        "emoji_count":     len(emojis),
        "original_length": original_length,
    }
