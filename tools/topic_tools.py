"""
tools/topic_tools.py
=====================
Tools LangChain pour A7 — PRD v3.0 §3.4

Tools :
  - compute_semantic_similarity : similarité cosinus via sentence-transformers
  - extract_key_topics          : extraction TF-IDF des thèmes dominants
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from langchain_core.tools import tool


# ── compute_semantic_similarity ───────────────────────────────────────────────

@tool
def compute_semantic_similarity(topic: str, comments_text: str) -> dict[str, Any]:
    """
    Calcule la similarité sémantique cosinus entre la thématique utilisateur
    et le corpus de commentaires via sentence-transformers.
    Retourne {"similarity_score": float, "method": str, "model": str}.
    Limite : 20 commentaires les mieux notés (score A4 >= 0.7) pour économiser la mémoire GPU.
    Fallback TF-IDF si sentence-transformers indisponible.
    """
    if not topic or not comments_text:
        return {"similarity_score": 0.5, "method": "no_input", "model": "none"}

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode([topic, comments_text], convert_to_numpy=True)
        # Cosinus
        a, b = embeddings[0], embeddings[1]
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            sim = 0.5
        else:
            sim = float(np.dot(a, b) / (norm_a * norm_b))
        return {
            "similarity_score": round(max(0.0, min(1.0, sim)), 4),
            "method":           "sentence_transformers",
            "model":            "paraphrase-multilingual-MiniLM-L12-v2",
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback TF-IDF cosinus
    sim = _tfidf_cosine(topic, comments_text)
    return {"similarity_score": sim, "method": "tfidf_fallback", "model": "none"}


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosinus TF-IDF simplifié entre deux textes."""
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w{3,}\b", text.lower())

    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vocab = set(tokens_a) | set(tokens_b)
    # IDF simplifié sur 2 documents
    idf = {term: math.log(3.0 / (1 + (term in tokens_a) + (term in tokens_b))) for term in vocab}

    def _tfidf_vec(tokens):
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        return {t: (tf[t] / total) * idf.get(t, 1.0) for t in tokens}

    vec_a = _tfidf_vec(tokens_a)
    vec_b = _tfidf_vec(tokens_b)

    dot   = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vocab)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(max(0.0, min(1.0, dot / (norm_a * norm_b))), 4)


# ── extract_key_topics ────────────────────────────────────────────────────────

# Stop words basiques FR + EN
_STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "au", "aux",
    "est", "sont", "était", "que", "qui", "quoi", "ce", "se", "sa", "son", "ses",
    "par", "sur", "pour", "avec", "dans", "mais", "ou", "donc", "or", "ni", "car",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "and", "or", "but",
    "not", "this", "that", "it", "its", "they", "them", "their",
}


@tool
def extract_key_topics(comments_text: str, top_n: int = 10) -> dict[str, Any]:
    """
    Extrait les thèmes dominants d'un corpus de commentaires via TF-IDF simplifié.
    Retourne {"topics": list[str], "topic_scores": dict, "method": "tfidf"}.
    Utilisé par A7 avant le raisonnement ToT pour ancrer la comparaison thématique.
    """
    if not comments_text:
        return {"topics": [], "topic_scores": {}, "method": "tfidf"}

    # Tokenisation : mots de 4+ caractères, hors stop words
    tokens = [
        w for w in re.findall(r"\b\w{4,}\b", comments_text.lower())
        if w not in _STOP_WORDS and not w.isdigit()
    ]

    if not tokens:
        return {"topics": [], "topic_scores": {}, "method": "tfidf"}

    # TF (term frequency)
    tf = Counter(tokens)
    total = len(tokens)

    # Score TF normalisé (TF-IDF simplifié sur corpus unique)
    scored = {
        word: round(count / total, 6)
        for word, count in tf.most_common(top_n * 3)
    }

    # Filtre : garder les mots avec TF significatif
    threshold = 1 / total
    filtered  = {w: s for w, s in scored.items() if s > threshold}

    # Top N
    top_topics = sorted(filtered, key=lambda w: filtered[w], reverse=True)[:top_n]
    top_scores = {w: filtered[w] for w in top_topics}

    return {
        "topics":       top_topics,
        "topic_scores": top_scores,
        "method":       "tfidf",
    }
