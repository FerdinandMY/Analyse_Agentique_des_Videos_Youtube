"""
evaluation/ablation_study.py
==============================
Etude d'ablation — PRD v3.0 AC-09

Compare 4 conditions de raisonnement pour tester H4 :
  B0 — Prompt direct (longueur seule)          : aucun raisonnement structure
  B1 — + Sentiment (CoT)                        : ajoute detection mots-cles
  B2 — + Discours + Bruit (CoT + ToT)           : ajoute argumentation + diversite lexicale
  B3 — CoT + ToT + Tools (pipeline v3.0)        : predictions completes (pipeline_predictions.jsonl)

H4 : le CoT/ToT/Tools reduit le taux d'erreur de classification d'au moins 15%
     par rapport au prompt direct (B0 -> B3).
     Metrique : gain F1-macro sentiment (3 classes) x 100 pts.

Methodologie v2 (par commentaire) :
  Chaque condition est une heuristique deterministe appliquee independamment
  a chaque commentaire — meme granularite que H2 (1 score/commentaire).
  B0-B2 simulent des prompts LLM de complexite croissante via des heuristiques
  progresses. B3 charge les predictions depuis pipeline_predictions.jsonl.

Usage :
    python evaluation/ablation_study.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --preds data/predictions/pipeline_predictions.jsonl \\
        --output evaluation/results/ablation_study.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compute_metrics import pearson_r, mae, f1_macro, load_gold


# ── Poids PRD §3.6 (IMMUTABLES) ──────────────────────────────────────────────

_W_S, _W_D, _W_N = 0.35, 0.40, 0.25


def _score_global(s: float, d: float, n: float) -> float:
    return round(_W_S * s + _W_D * d + _W_N * n, 2)


# ── Vocabulaires partagés ─────────────────────────────────────────────────────

_POS_WORDS = {
    "bon", "super", "excellent", "merci", "bravo", "top", "génial", "parfait",
    "utile", "incroyable", "magnifique", "fantastique", "formidable", "sympa",
    "great", "good", "amazing", "love", "best", "helpful", "wonderful",
    "thank", "thanks", "awesome", "brilliant", "clear", "perfect", "nice", "well",
}

_NEG_WORDS = {
    "nul", "mauvais", "horrible", "décevant", "problème", "ennuyeux", "inutile",
    "faux", "incorrect", "erreur", "déçu", "confus",
    "bad", "terrible", "awful", "hate", "worst", "boring", "useless", "wrong",
    "confusing", "misleading", "poor", "disappointing",
}

_ARG_RE = re.compile(
    r"\b(?:"
    r"parce que|car|puisque|mais|cependant|donc|par exemple|bien que|"
    r"en effet|en revanche|ainsi|notamment|"
    r"because|since|however|therefore|for example|but|although|while|whereas|"
    r"nevertheless|furthermore|moreover|in fact|indeed|especially"
    r")\b",
    re.IGNORECASE,
)

_NOISE_RE = re.compile(
    r"https?://\S+|(.)\1{4,}|\b(?:subscribe|abonne|follow|like\s+me)\b",
    re.IGNORECASE,
)

_REACTION_RE = re.compile(
    r"^[\s]*(?:"
    r"merci+|thanks?|thx|super|bravo|bien|cool|ok|oui|non|wow|yes|no|top|"
    r"good|great|nice|lol|haha|félicitation|felicitation|parfait|excellent|"
    r"magnifique|incroyable|amazing|awesome|[\U0001F600-\U0001FFFF\u2764\u2665]+"
    r")[\s!.?,;:😊❤👍😭😮🙏✨🎉💯]*$",
    re.IGNORECASE | re.UNICODE,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONS D'ABLATION (heuristiques per-comment)
# ══════════════════════════════════════════════════════════════════════════════

def _b0_length_only(text: str) -> tuple[float, str]:
    """
    B0 — Prompt direct (sans raisonnement structure).
    Simule un LLM qui n'exploite que la longueur brute du texte.
    Sentiment fixe = 'neutral'. Bruit ignore.
    """
    wc     = max(len(text.split()), 1)
    d      = min(100.0, wc * 2.5)
    sg     = _score_global(50.0, d, 70.0)
    return sg, "neutral"


def _b1_plus_sentiment(text: str) -> tuple[float, str]:
    """
    B1 — + Sentiment (CoT).
    Ajoute la detection de mots-cles positifs/negatifs.
    Discours = longueur seule (pas de detection d'argumentation).
    """
    words = text.split()
    wc    = max(len(words), 1)

    pos = sum(1 for w in words if w.lower().strip(".,!?;:") in _POS_WORDS)
    neg = sum(1 for w in words if w.lower().strip(".,!?;:") in _NEG_WORDS)
    if pos > neg:
        label, s = "positive", min(100.0, 55.0 + pos * 8)
    elif neg > pos:
        label, s = "negative", max(0.0, 45.0 - neg * 8)
    else:
        label, s = "neutral", 50.0

    d  = min(100.0, wc * 2.5)
    sg = _score_global(s, d, 70.0)
    return sg, label


def _b2_plus_discourse_noise(text: str) -> tuple[float, str]:
    """
    B2 — + Discours + Bruit (CoT + ToT).
    Ajoute la detection des connecteurs argumentatifs (argumentation),
    la diversite lexicale (richesse), et la detection du bruit (URL, spam).
    Simule un prompt structurant 3 branches : sentiment / discours / bruit.
    """
    words = text.split()
    wc    = max(len(words), 1)

    pos = sum(1 for w in words if w.lower().strip(".,!?;:") in _POS_WORDS)
    neg = sum(1 for w in words if w.lower().strip(".,!?;:") in _NEG_WORDS)
    if pos > neg:
        label, s = "positive", min(100.0, 55.0 + pos * 8)
    elif neg > pos:
        label, s = "negative", max(0.0, 45.0 - neg * 8)
    else:
        label, s = "neutral", 50.0

    ac  = len(_ARG_RE.findall(text))
    ur  = len(set(w.lower() for w in words)) / wc
    d   = min(100.0, 30.0 + wc * 1.5 + ac * 10 + ur * 15)

    is_noisy = bool(_NOISE_RE.search(text))
    n = 30.0 if is_noisy else min(100.0, 60.0 + ur * 20)

    sg = _score_global(s, d, n)
    return sg, label


# ── B3 : pipeline complet (chargé depuis pipeline_predictions.jsonl) ──────────

def load_b3_predictions(path: str) -> dict[str, tuple[float, str]]:
    """
    Charge les predictions B3 (pipeline 7 agents) depuis le fichier JSONL.
    Retourne {comment_id: (score_global, sentiment_label)}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier de predictions introuvable : {path}\n"
            "Lancez d'abord : python scripts/run_pipeline_predictions.py"
        )
    records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        r["comment_id"]: (float(r["score_global"]), r.get("sentiment_label", "neutral"))
        for r in records
    }


# ══════════════════════════════════════════════════════════════════════════════
# ETUDE D'ABLATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

_CONDITIONS_DEF: list[tuple[str, str, Any]] = [
    ("B0", "Prompt direct (longueur seule)",       _b0_length_only),
    ("B1", "CoT : + sentiment mots-cles",          _b1_plus_sentiment),
    ("B2", "CoT+ToT : + argumentation + bruit",    _b2_plus_discourse_noise),
    # B3 est traite separement (source = fichier externe)
]


def run_ablation(
    gold:       pd.DataFrame,
    preds_path: str,
    verbose:    bool = True,
) -> dict:
    """
    Exécute les 4 conditions d'ablation per-commentaire et calcule :
      - Pearson r / MAE par condition
      - F1-macro sentiment (3 classes) par condition
      - Gain F1 vs B0 (H4 : gain >= 15 pts)
    """
    gold_scores    = gold["gold_score"].tolist()
    gold_sentiments = gold["sentiment_label"].tolist() if "sentiment_label" in gold.columns else []
    comment_ids    = gold["comment_id"].tolist()
    texts          = gold["text"].tolist()
    n              = len(texts)

    # ── Conditions B0–B2 (heuristiques per-comment) ───────────────────────────
    condition_results: dict[str, dict] = {}

    for cid, label, fn in _CONDITIONS_DEF:
        scores, labels = [], []
        for text in texts:
            sg, lbl = fn(text)
            scores.append(sg)
            labels.append(lbl)

        r_val   = pearson_r(gold_scores, scores)
        mae_val = mae(gold_scores, scores)
        f1_val  = f1_macro(gold_sentiments, labels) if gold_sentiments else None

        condition_results[cid] = {
            "label":         label,
            "pearson_r":     r_val,
            "mae":           mae_val,
            "f1_sentiment":  f1_val,
            "fallback_count": 0,
            "fallback_rate":  0.0,
            "description": (
                "Heuristique par commentaire, aucun LLM requis. "
                f"Simule un prompt de complexite croissante ({cid})."
            ),
        }

    # ── Condition B3 (pipeline complet depuis fichier) ────────────────────────
    b3_map = load_b3_predictions(preds_path)

    b3_gold_scores, b3_pred_scores, b3_pred_labels, b3_gold_labels = [], [], [], []
    missing = 0
    for cid_c, text, gs, gl in zip(comment_ids, texts, gold_scores, gold_sentiments or [None]*n):
        if cid_c not in b3_map:
            missing += 1
            continue
        sg, lbl = b3_map[cid_c]
        b3_gold_scores.append(gs)
        b3_pred_scores.append(sg)
        b3_pred_labels.append(lbl)
        if gl is not None:
            b3_gold_labels.append(gl)

    if missing:
        print(f"[avert] {missing} commentaire(s) absents des predictions B3 (ignores).")

    b3_r   = pearson_r(b3_gold_scores, b3_pred_scores)
    b3_mae = mae(b3_gold_scores, b3_pred_scores)
    b3_f1  = f1_macro(b3_gold_labels, b3_pred_labels) if b3_gold_labels else None

    condition_results["B3"] = {
        "label":         "CoT+ToT+Tools (pipeline 7 agents)",
        "pearson_r":     b3_r,
        "mae":           b3_mae,
        "f1_sentiment":  b3_f1,
        "fallback_count": 0,
        "fallback_rate":  0.0,
        "description": (
            "Pipeline LangGraph complet : A3 Sentiment (VADER + mots-cles), "
            "A4 Discours (connecteurs arg. + diversite), A5 Bruit (SVM + patterns), "
            "A6 Synthese. Predictions chargees depuis pipeline_predictions.jsonl."
        ),
    }

    # ── H4 : gain F1 B0 -> B3 ────────────────────────────────────────────────
    f1_b0 = condition_results["B0"]["f1_sentiment"]
    f1_b3 = condition_results["B3"]["f1_sentiment"]
    r_b0  = condition_results["B0"]["pearson_r"]
    r_b3  = condition_results["B3"]["pearson_r"]

    delta_pearson_b0_b3 = round(r_b3 - r_b0, 4)
    delta_f1_b0_b3 = (
        round((f1_b3 - f1_b0) * 100, 2)
        if (f1_b0 is not None and f1_b3 is not None)
        else None
    )
    h4_satisfied = delta_f1_b0_b3 is not None and delta_f1_b0_b3 >= 15.0

    # Gains intermediaires
    gains: dict[str, dict] = {}
    pairs = [("B0->B1", "B0", "B1"), ("B1->B2", "B1", "B2"), ("B2->B3", "B2", "B3")]
    for pair_name, ca, cb in pairs:
        f1a = condition_results[ca]["f1_sentiment"]
        f1b = condition_results[cb]["f1_sentiment"]
        gains[pair_name] = {
            "delta_pearson_r": round(
                condition_results[cb]["pearson_r"] - condition_results[ca]["pearson_r"], 4
            ),
            "delta_f1_pts": (
                round((f1b - f1a) * 100, 2)
                if (f1a is not None and f1b is not None)
                else None
            ),
        }

    interp = ""
    if delta_f1_b0_b3 is not None:
        ok = "H4 validee (gain >= 15 pts)" if h4_satisfied else "H4 non validee (gain < 15 pts)"
        interp = f"Gain F1 B0->B3 = {delta_f1_b0_b3:+.1f} pts  ({ok})"
    else:
        interp = "Gain F1 non calculable (gold sans sentiment_label)"

    report = {
        "n_comments":     n,
        "methodology":    "per_comment_ablation_v2",
        "conditions":     condition_results,
        "gains_by_step":  gains,
        "h4": {
            "delta_pearson_r_b0_b3":    delta_pearson_b0_b3,
            "delta_f1_pts_b0_b3":       delta_f1_b0_b3,
            "h4_satisfied":             h4_satisfied,
            "h4_threshold_f1_gain_pts": 15.0,
            "interpretation":           interp,
        },
    }

    if verbose:
        _print_report(report)

    return report


def _print_report(report: dict) -> None:
    print()
    print("=" * 76)
    print("ETUDE D'ABLATION -- CoT / ToT / Tools (H4 - v2 per-comment)")
    print("=" * 76)
    print(f"{'Condition':<38} {'Pearson r':>10} {'MAE':>7} {'F1-sent':>9}")
    print("-" * 76)
    for cid, res in report["conditions"].items():
        f1 = f"{res['f1_sentiment']:.4f}" if res["f1_sentiment"] is not None else "    N/A"
        print(
            f"  {cid} -- {res['label']:<33} {res['pearson_r']:>10.4f}"
            f" {res['mae']:>7.2f} {f1:>9}"
        )
    print("-" * 76)
    for step, vals in report["gains_by_step"].items():
        f1g = f"{vals['delta_f1_pts']:+.1f} pts F1" if vals["delta_f1_pts"] is not None else "N/A"
        print(f"  Gain {step:<12}  dr={vals['delta_pearson_r']:+.4f}  {f1g}")
    print("-" * 76)
    print(f"\n  {report['h4']['interpretation']}")
    print("=" * 76)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation study -- PRD v3.0 AC-09 (B0 / B1 CoT / B2 CoT+ToT / B3 full)"
    )
    p.add_argument("--gold",    required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument(
        "--preds",
        default="data/predictions/pipeline_predictions.jsonl",
        help="Predictions pipeline B3 par commentaire (.jsonl)",
    )
    p.add_argument("--output",  default="evaluation/results/ablation_study.json")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    print(f"Gold standard : {len(gold)} commentaires")

    report = run_ablation(gold, preds_path=args.preds, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegarde : {args.output}")


if __name__ == "__main__":
    main()
