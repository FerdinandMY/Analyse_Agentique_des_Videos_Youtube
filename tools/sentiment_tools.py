"""
tools/sentiment_tools.py
=========================
Tools LangChain pour A3 — PRD v3.0 §3.4

Tools :
  - vader_sentiment        : score sentiment VADER (fallback LLM)
  - detect_sarcasm_markers : détecte les marqueurs linguistiques de sarcasme
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool


# ── vader_sentiment ───────────────────────────────────────────────────────────

@tool
def vader_sentiment(text: str) -> dict[str, Any]:
    """
    Calcule le score de sentiment VADER pour un texte.
    Retourne {"label": "positive|neutral|negative", "compound": float,
              "pos": float, "neu": float, "neg": float, "method": "vader"|"heuristic"}.
    VADER est utilisé comme signal de grounding avant le raisonnement LLM.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        label = (
            "positive" if compound >= 0.05
            else "negative" if compound <= -0.05
            else "neutral"
        )
        return {
            "label":    label,
            "compound": round(compound, 4),
            "pos":      round(scores["pos"], 4),
            "neu":      round(scores["neu"], 4),
            "neg":      round(scores["neg"], 4),
            "method":   "vader",
        }
    except ImportError:
        # Fallback heuristique si vaderSentiment non installé
        return _heuristic_sentiment(text)


def _heuristic_sentiment(text: str) -> dict[str, Any]:
    """Fallback quand VADER n'est pas disponible."""
    text_lower = text.lower()
    pos_words = ["bon", "super", "excellent", "merci", "bravo", "top", "génial",
                 "good", "great", "excellent", "thank", "amazing", "love", "best"]
    neg_words = ["nul", "mauvais", "horrible", "décevant", "problème",
                 "bad", "terrible", "awful", "hate", "worst", "boring", "useless"]
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    if pos_count > neg_count:
        label, compound = "positive", 0.3
    elif neg_count > pos_count:
        label, compound = "negative", -0.3
    else:
        label, compound = "neutral", 0.0
    return {"label": label, "compound": compound, "pos": 0.0, "neu": 0.0, "neg": 0.0, "method": "heuristic"}


# ── detect_sarcasm_markers ────────────────────────────────────────────────────

# Marqueurs de sarcasme et ironie (FR + EN)
_SARCASM_PATTERNS = [
    # Guillemets ironiques autour d'un mot positif
    r'["\'](?:super|génial|excellent|formidable|parfait|great|amazing|wonderful)["\']',
    # Ponctuation excessive
    r"!!{2,}",
    r"\.{3,}",
    # Formules ironiques françaises
    r"\b(?:bien sûr|évidemment|vraiment|tellement|comme d'habitude|encore une fois)\b",
    # Formules ironiques anglaises
    r"\b(?:yeah right|sure thing|totally|oh great|wow thanks|brilliant)\b",
    # Majuscules excessives (>50% du texte)
]

_SARCASM_RE = re.compile("|".join(_SARCASM_PATTERNS), re.IGNORECASE | re.UNICODE)

# Mots qui contredisent le ton apparent
_CONTRADICTION_PAIRS = [
    (r"\b(?:super|génial|bravo)\b", r"\b(?:déçu|nul|mauvais|horrible)\b"),
    (r"\b(?:great|amazing|excellent)\b", r"\b(?:disappointed|boring|terrible)\b"),
]


@tool
def detect_sarcasm_markers(text: str) -> dict[str, Any]:
    """
    Détecte les marqueurs linguistiques de sarcasme et d'ironie dans un commentaire.
    Retourne {"has_sarcasm": bool, "confidence": float, "markers": list[str],
              "contradiction_detected": bool}.
    Utilisé par A3 comme signal avant la classification sentiment finale.
    """
    if not text:
        return {"has_sarcasm": False, "confidence": 0.0, "markers": [], "contradiction_detected": False}

    markers = []
    score   = 0.0

    # Test des patterns sarcasme
    for match in _SARCASM_RE.finditer(text):
        markers.append(match.group(0))
        score += 0.25

    # Test majuscules excessives (>40% de lettres en maj)
    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.4 and len(letters) > 5:
            markers.append(f"[CAPS_RATIO:{caps_ratio:.0%}]")
            score += 0.20

    # Test contradictions lexicales
    contradiction = False
    for pos_pat, neg_pat in _CONTRADICTION_PAIRS:
        if re.search(pos_pat, text, re.I) and re.search(neg_pat, text, re.I):
            contradiction = True
            markers.append("[LEXICAL_CONTRADICTION]")
            score += 0.30
            break

    # Texte court avec marqueur fort → sarcasme probable
    if len(text.split()) < 10 and score > 0:
        score += 0.10

    confidence = round(min(score, 1.0), 4)
    has_sarcasm = confidence >= 0.25

    return {
        "has_sarcasm":            has_sarcasm,
        "confidence":             confidence,
        "markers":                markers[:10],   # max 10 pour éviter les outputs trop longs
        "contradiction_detected": contradiction,
    }
