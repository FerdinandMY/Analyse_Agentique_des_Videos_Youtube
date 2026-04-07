"""
scripts/run_pipeline_predictions.py
=====================================
Génère les prédictions du pipeline pour chaque commentaire du gold standard.

Stratégie :
  - Charge le gold standard (comment_id + text)
  - Pour chaque commentaire, exécute le pipeline complet (ou fallback heuristique v2)
  - Sauvegarde data/predictions/pipeline_predictions.jsonl

Heuristique v2 (fallback) — améliorations vs v1 :
  - Engagement factor : réduit la contribution du sentiment pour les textes courts
  - Step-function discours : pénalise agressivement les réactions vides et le spam
  - Détection reaction_vide : regex spécifique pour les réponses sans contenu
  - Connecteurs argumentatifs enrichis (but, although, while, whereas, ...)
  - Range [0-100] étendu : std≈15 vs std≈6 en v1

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


# ── Poids PRD §3.6 (IMMUTABLES) ──────────────────────────────────────────────

_W_S, _W_D, _W_N = 0.35, 0.40, 0.25


def _score_global(s: float, d: float, n: float) -> float:
    return round(_W_S * s + _W_D * d + _W_N * n, 2)


# ── Patterns de détection ─────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"https?://\S+",
    r"(.)\1{4,}",
    r"\b(?:subscribe|abonne|follow|like\s+me)\b",
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# Réactions vides : un ou deux mots génériques sans contenu argumentatif
_REACTION_RE = re.compile(
    r"^[\s]*(?:"
    r"merci+|thanks?|thx|super|bravo|bien|cool|ok|oui|non|wow|yes|no|yep|nope|"
    r"top|good|great|nice|lol|haha|lmao|xd|omg|"
    r"félicitation|felicitation|congratulation|parfait|excellent|magnifique|"
    r"incroyable|amazing|awesome|[\U0001F600-\U0001FFFF\u2764\u2665]+"
    r")[\s!.?,;:😊❤👍😭😮🙏✨🎉💯]*$",
    re.IGNORECASE | re.UNICODE,
)

_POS_WORDS = {
    # Français
    "bon", "super", "excellent", "merci", "bravo", "top", "génial", "parfait",
    "utile", "incroyable", "magnifique", "félicitation", "fantastique", "formidable",
    "sympa", "clair", "précis", "efficace", "impressionnant",
    # Anglais
    "great", "good", "amazing", "love", "best", "helpful", "wonderful",
    "thank", "thanks", "awesome", "excellent", "brilliant", "clear",
    "perfect", "fantastic", "useful", "nice", "well",
}

_NEG_WORDS = {
    # Français
    "nul", "mauvais", "horrible", "décevant", "problème", "ennuyeux", "inutile",
    "faux", "incorrect", "erreur", "médiocre", "déçu", "bizarre", "confus",
    # Anglais
    "bad", "terrible", "awful", "hate", "worst", "boring", "useless", "wrong",
    "error", "confusing", "misleading", "poor", "disappointing",
}

_ARG_RE = re.compile(
    r"\b(?:"
    r"parce que|car|puisque|mais|cependant|donc|par exemple|bien que|"
    r"en effet|en revanche|ainsi|d'ailleurs|notamment|c'est-à-dire|"
    r"because|since|however|therefore|for example|but|although|while|whereas|"
    r"nevertheless|furthermore|moreover|in fact|indeed|especially|namely|"
    r"despite|although|even though|so that|in order to"
    r")\b",
    re.IGNORECASE,
)


# ── Heuristique v2 par commentaire ───────────────────────────────────────────

def _heuristic_predict(comment_id: str, text: str) -> dict:
    """
    Prédit score_global, sentiment_label, discourse_score, noise_score via
    heuristique v2.

    Améliorations vs v1 :
    - engagement_factor : la contribution du sentiment est pondérée par la
      longueur du texte — un «merci» de 2 mots apporte peu de signal.
    - Step-function discours : calibrée pour couvrir [0-100] sans compression.
    - Détection reaction_vide : pénalise fortement les commentaires vides.
    - Connectors enrichis : 'but', 'although', 'whereas', 'malgré', ...
    """
    words      = text.split()
    word_count = max(len(words), 1)
    text_lower = text.lower()

    # ── Détection spam / bruit / réaction vide ────────────────────────────────
    is_spam        = bool(_NOISE_RE.search(text))
    has_repeated   = bool(re.search(r"(.)\1{4,}", text))
    is_reaction    = word_count <= 6 and bool(_REACTION_RE.match(text.strip()))

    # ── Sentiment ─────────────────────────────────────────────────────────────
    # engagement_factor : réduit la contribution du sentiment pour les textes
    # courts (un commentaire court ne permet pas d'inférer le sentiment avec
    # fiabilité — la contribution max. est atteinte à 25 mots).
    engagement_factor = min(1.0, word_count / 25.0)

    pos = sum(1 for w in words if w.lower().strip(".,!?;:") in _POS_WORDS)
    neg = sum(1 for w in words if w.lower().strip(".,!?;:") in _NEG_WORDS)

    if pos > neg:
        label, base_s = "positive", 75.0
    elif neg > pos:
        label, base_s = "negative", 25.0
    else:
        label, base_s = "neutral", 50.0

    s_score = 20.0 + engagement_factor * (base_s - 20.0)

    # ── Discours (step function par longueur) ─────────────────────────────────
    arg_count = len(_ARG_RE.findall(text))
    unique_r  = len(set(w.lower() for w in words)) / word_count

    if is_spam or has_repeated:
        # Spam / chars répétés : discours quasi nul
        d_score = min(20.0, word_count * 0.5)
    elif is_reaction or word_count <= 4:
        # Réaction vide ou très court : pénalisation forte
        d_score = word_count * 2.5
    elif word_count <= 15:
        # Court : montée progressive
        d_score = 8.0 + word_count * 2.0 + arg_count * 8 + unique_r * 8
    elif word_count <= 35:
        # Moyen : zone médiane avec bonus argumentatif
        d_score = 30.0 + (word_count - 15) * 1.5 + arg_count * 10 + unique_r * 12
    else:
        # Long : plafonnement progressif vers 100
        d_score = 60.0 + (word_count - 35) * 0.5 + arg_count * 5 + unique_r * 15

    d_score = min(100.0, max(0.0, d_score))

    # ── Bruit ─────────────────────────────────────────────────────────────────
    if is_spam:
        n_score = 5.0
    elif has_repeated:
        n_score = 20.0
    elif is_reaction:
        # Réactions vides : pas du spam, mais très peu de valeur ajoutée
        n_score = 25.0
    else:
        n_score = min(100.0, 55.0 + unique_r * 20)

    sg = _score_global(s_score, d_score, n_score)

    return {
        "comment_id":      comment_id,
        "text":            text[:200],
        "score_global":    sg,
        "sentiment_label": label,
        "sentiment_score": round(s_score, 2),
        "discourse_score": round(d_score, 2),
        "noise_score":     round(n_score, 2),
        "condition":       "heuristic_pipeline_v2",
        "fallback_used":   True,
    }


# ── Pipeline complet (si disponible) ──────────────────────────────────────────

def _pipeline_predict(comment_id: str, text: str) -> dict:
    """Tente le pipeline LangGraph — fallback heuristique v2 si échec."""
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
            "comment_id":          comment_id,
            "text":                text[:200],
            "score_global":        round(float(result.get("score_global", 50.0)), 2),
            "sentiment_label":     sent.get("sentiment_label", "neutral"),
            "sentiment_score":     round(float(sent.get("sentiment_score", 50.0)), 2),
            "discourse_score":     round(float(disc.get("discourse_score", 50.0)), 2),
            "noise_score":         round(float(noise.get("noise_score", 70.0)), 2),
            "hallucination_flags": h_flags,
            "condition":           "pipeline_full",
            "fallback_used":       result.get("fallback_used", False),
        }
    except Exception as exc:
        print(f"  [pipeline] erreur {comment_id}: {exc} — fallback heuristique v2")
        return _heuristic_predict(comment_id, text)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(gold_path: str, output_path: str, use_pipeline: bool) -> None:
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

    out = Path(output_path)
    out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions),
        encoding="utf-8",
    )
    print(f"\nPredictions sauvegardees : {output_path} ({len(predictions)} lignes)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genere les predictions pipeline pour le gold standard")
    p.add_argument("--gold",     default="data/gold_standard/gold_standard.jsonl")
    p.add_argument("--output",   default="data/predictions/pipeline_predictions.jsonl")
    p.add_argument("--pipeline", action="store_true",
                   help="Utilise le pipeline LangGraph complet (necessite le LLM)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.gold, args.output, use_pipeline=args.pipeline)
