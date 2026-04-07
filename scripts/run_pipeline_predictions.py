"""
scripts/run_pipeline_predictions.py
=====================================
Génère les prédictions du pipeline pour chaque commentaire du gold standard.

Stratégie :
  - Charge le gold standard (comment_id + text)
  - Pour chaque commentaire, exécute le pipeline complet (ou fallback heuristique)
  - Sauvegarde data/predictions/pipeline_predictions.jsonl

Usage :
    python scripts/run_pipeline_predictions.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --output data/predictions/pipeline_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Poids PRD §3.6 ────────────────────────────────────────────────────────────

_W_S, _W_D, _W_N = 0.35, 0.40, 0.25


def _score_global(s: float, d: float, n: float) -> float:
    return round(_W_S * s + _W_D * d + _W_N * n, 2)


# ── Heuristiques par commentaire ──────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"https?://\S+",
    r"(.)\1{4,}",
    r"\b(?:subscribe|abonne|follow|like\s+me)\b",
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

_POS_WORDS = {"bon", "super", "excellent", "merci", "bravo", "top", "génial",
              "great", "good", "amazing", "love", "best", "parfait", "utile"}
_NEG_WORDS = {"nul", "mauvais", "horrible", "décevant", "problème", "ennuyeux",
              "bad", "terrible", "awful", "hate", "worst", "boring", "inutile"}

_ARG_RE = re.compile(
    r"\b(?:parce que|car|puisque|mais|cependant|donc|par exemple|"
    r"because|since|however|therefore|for example)\b",
    re.IGNORECASE,
)


def _heuristic_predict(comment_id: str, text: str) -> dict:
    """Prédit score_global, sentiment_label, noise_score via heuristique."""
    words      = text.split()
    word_count = max(len(words), 1)
    text_lower = text.lower()

    # Sentiment
    pos = sum(1 for w in words if w.lower().strip(".,!?") in _POS_WORDS)
    neg = sum(1 for w in words if w.lower().strip(".,!?") in _NEG_WORDS)
    if pos > neg:
        label, s_score = "positive", min(100.0, 55.0 + pos * 5)
    elif neg > pos:
        label, s_score = "negative", max(0.0, 45.0 - neg * 5)
    else:
        label, s_score = "neutral", 50.0

    # Discourse
    arg_count  = len(_ARG_RE.findall(text))
    unique_r   = len(set(w.lower() for w in words)) / word_count
    d_score    = round(min(100.0, 30.0 + word_count * 1.2 + arg_count * 8 + unique_r * 20), 2)

    # Bruit
    is_noisy   = bool(_NOISE_RE.search(text))
    n_score    = 40.0 if is_noisy else min(100.0, 60.0 + unique_r * 20)

    sg = _score_global(s_score, d_score, n_score)

    return {
        "comment_id":      comment_id,
        "text":            text[:200],
        "score_global":    sg,
        "sentiment_label": label,
        "sentiment_score": round(s_score, 2),
        "discourse_score": round(d_score, 2),
        "noise_score":     round(n_score, 2),
        "condition":       "heuristic_pipeline",
        "fallback_used":   True,
    }


# ── Pipeline complet (si disponible) ──────────────────────────────────────────

def _pipeline_predict(comment_id: str, text: str) -> dict:
    """Tente le pipeline LangGraph — fallback heuristique si échec."""
    try:
        from graph import run_pipeline
        result = run_pipeline(
            raw_comments=[{"text": text, "comment_id": comment_id}],
            video_id=f"eval_{comment_id[:8]}",
            topic="",
        )
        details  = result.get("details") or {}
        sent     = details.get("sentiment") or {}
        disc     = details.get("discourse") or {}
        noise    = details.get("noise") or {}
        h_flags  = result.get("hallucination_flags") or []
        return {
            "comment_id":        comment_id,
            "text":              text[:200],
            "score_global":      round(float(result.get("score_global", 50.0)), 2),
            "sentiment_label":   sent.get("sentiment_label", "neutral"),
            "sentiment_score":   round(float(sent.get("sentiment_score", 50.0)), 2),
            "discourse_score":   round(float(disc.get("discourse_score", 50.0)), 2),
            "noise_score":       round(float(noise.get("noise_score", 70.0)), 2),
            "hallucination_flags": h_flags,
            "condition":         "pipeline_full",
            "fallback_used":     result.get("fallback_used", False),
        }
    except Exception as exc:
        print(f"  [pipeline] erreur commentaire {comment_id}: {exc} — fallback heuristique")
        return _heuristic_predict(comment_id, text)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(gold_path: str, output_path: str, use_pipeline: bool) -> None:
    # Charge le gold standard
    records = [
        json.loads(line)
        for line in Path(gold_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Gold standard : {len(records)} commentaires")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions = []

    for i, rec in enumerate(records, 1):
        comment_id = rec.get("comment_id", f"c{i:04d}")
        text       = rec.get("text", "")

        if not text:
            continue

        print(f"  [{i:3d}/{len(records)}] {comment_id[:20]:<20} ", end="", flush=True)

        if use_pipeline:
            pred = _pipeline_predict(comment_id, text)
        else:
            pred = _heuristic_predict(comment_id, text)

        predictions.append(pred)
        print(f"score={pred['score_global']:5.1f}  label={pred['sentiment_label']}")

    # Sauvegarde en JSONL
    out = Path(output_path)
    out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions),
        encoding="utf-8",
    )
    print(f"\nPrédictions sauvegardées : {output_path} ({len(predictions)} lignes)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Génère les prédictions pipeline pour le gold standard")
    p.add_argument("--gold",     default="data/gold_standard/gold_standard.jsonl")
    p.add_argument("--output",   default="data/predictions/pipeline_predictions.jsonl")
    p.add_argument("--pipeline", action="store_true",
                   help="Utilise le pipeline LangGraph complet (nécessite le LLM)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.gold, args.output, use_pipeline=args.pipeline)
