"""
tools/noise_tools.py
=====================
Tools LangChain pour A5 — PRD v3.0 §3.4

Tools :
  - svm_spam_detector  : classifieur SVM TF-IDF pré-entraîné (heuristique si absent)
  - count_repeated_chars : détecte répétitions de caractères et caps ratio
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool


# ── Patterns de spam déterministes ────────────────────────────────────────────

_SPAM_PATTERNS = [
    r"https?://\S+",                          # URLs
    r"(?:subscribe|abonne|follow|like)\s+(?:me|us|my|mon|ma)",  # appel à l'abonnement
    r"\b(?:giveaway|concours|win|gagner)\b",  # concours
    r"(?:check out|visitez|visite)\s+(?:my|mon|ma|notre)",      # auto-promo
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}",            # email
    r"\+\d[\d\s\-]{8,}",                      # numéro de téléphone
]
_SPAM_RE = re.compile("|".join(_SPAM_PATTERNS), re.IGNORECASE | re.UNICODE)

# Features TF-IDF simplifiées pour le fallback SVM
_SPAM_KEYWORDS = {
    "high": ["subscribe", "abonnez", "like", "follow", "check out", "giveaway",
             "promo", "discount", "free", "gratuit", "gagnez", "telegram"],
    "medium": ["merci", "thanks", "lol", "haha", "premier", "first", "wow"],
}


def _heuristic_spam_score(text: str) -> float:
    """Score de spam heuristique [0-1] quand le SVM est absent."""
    text_lower = text.lower()
    score = 0.0

    # Patterns déterministes
    if _SPAM_RE.search(text):
        score += 0.4

    # Mots-clés
    for word in _SPAM_KEYWORDS["high"]:
        if word in text_lower:
            score += 0.15

    # Texte très court (< 3 mots) → réaction vide
    if len(text.split()) < 3:
        score += 0.1

    return round(min(score, 1.0), 4)


# ── svm_spam_detector ─────────────────────────────────────────────────────────

@tool
def svm_spam_detector(text: str) -> dict[str, Any]:
    """
    Classifie un commentaire comme spam/non-spam via un SVM TF-IDF.
    Si le modèle SVM n'est pas disponible, utilise une heuristique déterministe.
    Retourne {"is_spam": bool, "spam_score": float, "confidence": float, "method": str}.
    Premier filtre de A5 — traite tous les commentaires avant le CoT.
    """
    if not text or not text.strip():
        return {"is_spam": False, "spam_score": 0.0, "confidence": 1.0, "method": "empty_text"}

    try:
        import pickle
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "models" / "svm_spam_classifier.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            proba = model.predict_proba([text])[0]
            spam_score = round(float(proba[1]), 4)
            return {
                "is_spam":    spam_score >= 0.5,
                "spam_score": spam_score,
                "confidence": round(max(proba), 4),
                "method":     "svm",
            }
    except Exception:
        pass

    # Fallback heuristique
    spam_score = _heuristic_spam_score(text)
    return {
        "is_spam":    spam_score >= 0.5,
        "spam_score": spam_score,
        "confidence": 0.6,  # confiance modérée pour l'heuristique
        "method":     "heuristic",
    }


# ── count_repeated_chars ──────────────────────────────────────────────────────

_REPEATED_CHAR_RE   = re.compile(r"(.)\1{3,}")      # 4+ répétitions du même caractère
_REPEATED_WORD_RE   = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)  # mot répété 3x+
_ONLY_PUNCTUATION_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)


@tool
def count_repeated_chars(text: str) -> dict[str, Any]:
    """
    Analyse les patterns de répétition dans un commentaire.
    Retourne {"repeated_char_count": int, "repeated_words": list,
              "caps_ratio": float, "is_bot_suspect": bool, "noise_signals": list}.
    Utilisé par A5 pour détecter les patterns bots et le contenu non-informatif.
    """
    if not text:
        return {
            "repeated_char_count": 0, "repeated_words": [],
            "caps_ratio": 0.0, "is_bot_suspect": False, "noise_signals": [],
        }

    noise_signals = []

    # Répétitions de caractères (hahahaha, !!!!, aaaa)
    repeated_chars = _REPEATED_CHAR_RE.findall(text)
    if repeated_chars:
        noise_signals.append(f"repeated_chars: {repeated_chars[:3]}")

    # Répétitions de mots (spam spam spam)
    repeated_words_matches = _REPEATED_WORD_RE.findall(text)
    repeated_words = list(set(repeated_words_matches))
    if repeated_words:
        noise_signals.append(f"repeated_words: {repeated_words[:3]}")

    # Caps ratio
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (
        round(sum(1 for c in letters if c.isupper()) / len(letters), 4)
        if letters else 0.0
    )
    if caps_ratio > 0.6 and len(letters) > 5:
        noise_signals.append(f"high_caps: {caps_ratio:.0%}")

    # Ponctuation seule
    if _ONLY_PUNCTUATION_RE.match(text.strip()):
        noise_signals.append("only_punctuation")

    # Texte identique répété (copy-paste bot)
    words = text.split()
    if len(words) >= 4:
        half = len(words) // 2
        if words[:half] == words[half : half * 2]:
            noise_signals.append("copy_paste_pattern")

    is_bot_suspect = len(noise_signals) >= 2 or caps_ratio > 0.7

    return {
        "repeated_char_count": len(repeated_chars),
        "repeated_words":      repeated_words[:5],
        "caps_ratio":          caps_ratio,
        "is_bot_suspect":      is_bot_suspect,
        "noise_signals":       noise_signals,
    }
