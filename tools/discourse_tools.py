"""
tools/discourse_tools.py
=========================
Tools LangChain pour A4 — PRD v3.0 §3.4

Tools :
  - compute_text_stats             : longueur, densité, ponctuation
  - detect_argumentative_markers   : marqueurs de causalité, contraste, exemples
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool


# ── compute_text_stats ────────────────────────────────────────────────────────

@tool
def compute_text_stats(text: str) -> dict[str, Any]:
    """
    Calcule des statistiques textuelles d'un commentaire.
    Retourne {"word_count": int, "char_count": int, "sentence_count": int,
              "avg_word_length": float, "question_count": int,
              "unique_word_ratio": float, "caps_ratio": float}.
    Ces stats ancrent le raisonnement de A4 sur des faits mesurés.
    """
    if not text or not text.strip():
        return {
            "word_count": 0, "char_count": 0, "sentence_count": 0,
            "avg_word_length": 0.0, "question_count": 0,
            "unique_word_ratio": 0.0, "caps_ratio": 0.0,
        }

    words = text.split()
    word_count  = len(words)
    char_count  = len(text)

    # Phrases (terminées par . ! ?)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = max(1, sum(1 for s in sentences if s.strip()))

    # Longueur moyenne des mots
    avg_word_length = round(
        sum(len(w.strip(".,!?;:\"'")) for w in words) / max(word_count, 1), 2
    )

    # Questions
    question_count = text.count("?")

    # Ratio de mots uniques (diversité lexicale)
    unique_words = set(w.lower().strip(".,!?;:\"'") for w in words)
    unique_word_ratio = round(len(unique_words) / max(word_count, 1), 4)

    # Ratio de majuscules
    letters = [c for c in text if c.isalpha()]
    caps_ratio = round(
        sum(1 for c in letters if c.isupper()) / max(len(letters), 1), 4
    )

    return {
        "word_count":        word_count,
        "char_count":        char_count,
        "sentence_count":    sentence_count,
        "avg_word_length":   avg_word_length,
        "question_count":    question_count,
        "unique_word_ratio": unique_word_ratio,
        "caps_ratio":        caps_ratio,
    }


# ── detect_argumentative_markers ──────────────────────────────────────────────

# Marqueurs argumentatifs (FR + EN)
_ARG_CATEGORIES = {
    "causality": [
        r"\b(?:parce que|car|puisque|étant donné|vu que|comme)\b",
        r"\b(?:because|since|given that|as|therefore|thus|hence)\b",
    ],
    "contrast": [
        r"\b(?:mais|cependant|néanmoins|toutefois|en revanche|pourtant|par contre)\b",
        r"\b(?:but|however|nevertheless|yet|although|despite|on the other hand)\b",
    ],
    "example": [
        r"\b(?:par exemple|notamment|comme|tel que|entre autres|à savoir)\b",
        r"\b(?:for example|for instance|such as|like|namely|e\.g\.)\b",
    ],
    "conclusion": [
        r"\b(?:donc|ainsi|en conclusion|en résumé|finalement|bref|en somme)\b",
        r"\b(?:so|thus|therefore|in conclusion|to summarize|finally|overall)\b",
    ],
    "nuance": [
        r"\b(?:certes|bien que|même si|quoique|malgré|hormis)\b",
        r"\b(?:although|even though|admittedly|granted|despite)\b",
    ],
}


@tool
def detect_argumentative_markers(text: str) -> dict[str, Any]:
    """
    Détecte les marqueurs argumentatifs dans un commentaire YouTube.
    Retourne {"total_markers": int, "categories": dict, "argumentation_score": float,
              "is_argumentative": bool}.
    Utilisé par A4 pour objectiver la profondeur du discours avant le scoring LLM.
    """
    if not text:
        return {
            "total_markers": 0, "categories": {},
            "argumentation_score": 0.0, "is_argumentative": False,
        }

    found: dict[str, list[str]] = {}
    total = 0

    for category, patterns in _ARG_CATEGORIES.items():
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE | re.UNICODE):
                matches.append(m.group(0))
        if matches:
            found[category] = matches
            total += len(matches)

    # Score d'argumentation [0-1] : basé sur le nombre de catégories distinctes
    n_categories = len(found)
    argumentation_score = round(min(n_categories / len(_ARG_CATEGORIES), 1.0), 4)

    return {
        "total_markers":       total,
        "categories":          {k: v[:3] for k, v in found.items()},  # max 3 exemples par catégorie
        "argumentation_score": argumentation_score,
        "is_argumentative":    argumentation_score >= 0.4,
    }
