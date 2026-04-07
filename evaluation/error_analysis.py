"""
evaluation/error_analysis.py
==============================
Analyse des erreurs systématiques du pipeline multi-agents.

Identifie :
  - Faux positifs / faux négatifs du détecteur de bruit (A5)
  - Erreurs de classification sentiment sur sarcasme / ironie (A3)
  - Commentaires où Score_Global dévie fortement du gold (outliers)
  - Patterns linguistiques corrélés aux erreurs

Usage :
    python evaluation/error_analysis.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --preds data/predictions/pipeline_predictions.jsonl \\
        --output evaluation/results/error_analysis.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compute_metrics import load_gold, load_predictions, mae


# ── Détection des outliers de score ──────────────────────────────────────────

def score_outliers(merged: pd.DataFrame, threshold: float = 20.0) -> list[dict]:
    """
    Retourne les commentaires où |gold_score - score_global| > threshold.
    """
    if "gold_score" not in merged.columns or "score_global" not in merged.columns:
        return []

    merged = merged.copy()
    merged["abs_error"] = (merged["gold_score"] - merged["score_global"]).abs()
    outliers = merged[merged["abs_error"] > threshold].sort_values("abs_error", ascending=False)

    def _get_text(row):
        for col in ("text_gold", "text", "text_pred"):
            if col in row.index and str(row[col]) not in ("", "nan"):
                return str(row[col])[:200]
        return ""

    return [
        {
            "comment_id":   row.get("comment_id", i),
            "text":         _get_text(row),
            "gold_score":   round(float(row["gold_score"]), 2),
            "pred_score":   round(float(row["score_global"]), 2),
            "abs_error":    round(float(row["abs_error"]), 2),
            "direction":    "over" if row["score_global"] > row["gold_score"] else "under",
        }
        for i, (_, row) in enumerate(outliers.iterrows())
    ]


# ── Erreurs de sentiment ──────────────────────────────────────────────────────

def sentiment_errors(merged: pd.DataFrame) -> dict:
    """
    Analyse les erreurs de classification sentiment.
    Détecte les commentaires sarcatiques mal classifiés (positif prédit → négatif gold).
    """
    gold_col = "sentiment_label_gold" if "sentiment_label_gold" in merged.columns else "sentiment_label"
    pred_col = "sentiment_label_pred" if "sentiment_label_pred" in merged.columns else "sentiment_pred"

    if gold_col not in merged.columns or pred_col not in merged.columns:
        return {"available": False}

    errors = merged[merged[gold_col] != merged[pred_col]].copy()
    n_total  = len(merged)
    n_errors = len(errors)

    # Sarcasme heuristique : "!" + ponctuation + longueur courte + mots ironiques
    _sarcasm_re = re.compile(
        r"\b(bien sûr|évidemment|vraiment|tellement|super|génial|incroyable)\b"
        r"|\.\.\.|!!+",
        re.IGNORECASE,
    )

    def _is_sarcasm_candidate(text: str) -> bool:
        return bool(_sarcasm_re.search(str(text))) and len(str(text).split()) < 20

    error_records = []
    for _, row in errors.iterrows():
        text_col_e = "text_gold" if "text_gold" in row.index else "text"
        text = str(row.get(text_col_e, "") or "")
        error_records.append({
            "comment_id":      row.get("comment_id", ""),
            "text":            text[:200],
            "gold_sentiment":  row[gold_col],
            "pred_sentiment":  row[pred_col],
            "sarcasm_suspect": _is_sarcasm_candidate(text),
        })

    sarcasm_count = sum(1 for r in error_records if r["sarcasm_suspect"])

    # Confusion matrix textuelle
    labels = sorted(merged[gold_col].unique())
    confusion: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in labels} for l in labels}
    for _, row in merged.iterrows():
        t = row[gold_col]
        p = row[pred_col]
        if t in confusion and p in confusion.get(t, {}):
            confusion[t][p] += 1

    return {
        "available":      True,
        "n_total":        n_total,
        "n_errors":       n_errors,
        "error_rate":     round(n_errors / n_total, 4) if n_total > 0 else 0.0,
        "sarcasm_suspect_errors": sarcasm_count,
        "confusion_matrix": confusion,
        "top_errors":     error_records[:20],
    }


# ── Faux positifs/négatifs bruit (A5) ────────────────────────────────────────

def noise_errors(merged: pd.DataFrame) -> dict:
    """
    Compare noise_flag gold (si disponible) avec noise_score prédit.
    Calcule FP / FN pour le détecteur de bruit.
    """
    if "noise_flag" not in merged.columns or "noise_score" not in merged.columns:
        return {"available": False, "reason": "colonnes noise_flag / noise_score absentes"}

    # Binarise le score prédit : bruit si noise_score < 50
    merged = merged.copy()
    merged["pred_noisy"] = merged["noise_score"] < 50.0
    gold_noisy = merged["noise_flag"].astype(bool)
    pred_noisy = merged["pred_noisy"].astype(bool)

    tp = int((gold_noisy & pred_noisy).sum())
    fp = int((~gold_noisy & pred_noisy).sum())
    fn = int((gold_noisy & ~pred_noisy).sum())
    tn = int((~gold_noisy & ~pred_noisy).sum())

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    # Exemples de FP et FN — "text" peut être renommé après merge
    text_col = "text_gold" if "text_gold" in merged.columns else "text"
    fp_examples = merged[(~gold_noisy) & pred_noisy][text_col].head(5).tolist()
    fn_examples = merged[gold_noisy & (~pred_noisy)][text_col].head(5).tolist()

    return {
        "available": True,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "false_positive_examples": [str(t)[:150] for t in fp_examples],
        "false_negative_examples": [str(t)[:150] for t in fn_examples],
    }


# ── Patterns textuels corrélés aux erreurs ───────────────────────────────────

def linguistic_patterns(outliers: list[dict]) -> dict:
    """
    Analyse les patterns linguistiques des commentaires outliers.
    """
    if not outliers:
        return {}

    over_preds  = [o for o in outliers if o["direction"] == "over"]
    under_preds = [o for o in outliers if o["direction"] == "under"]

    def _avg_len(items):
        if not items:
            return 0.0
        return round(sum(len(o["text"].split()) for o in items) / len(items), 1)

    def _url_ratio(items):
        if not items:
            return 0.0
        return round(sum(1 for o in items if re.search(r"https?://", o["text"])) / len(items), 3)

    def _emoji_ratio(items):
        if not items:
            return 0.0
        emoji_re = re.compile(r"[\U0001F300-\U0001FFFF]", re.UNICODE)
        return round(sum(1 for o in items if emoji_re.search(o["text"])) / len(items), 3)

    return {
        "over_predicted": {
            "count":       len(over_preds),
            "avg_text_len": _avg_len(over_preds),
            "url_ratio":   _url_ratio(over_preds),
            "emoji_ratio": _emoji_ratio(over_preds),
        },
        "under_predicted": {
            "count":       len(under_preds),
            "avg_text_len": _avg_len(under_preds),
            "url_ratio":   _url_ratio(under_preds),
            "emoji_ratio": _emoji_ratio(under_preds),
        },
    }


# ── Rapport principal ─────────────────────────────────────────────────────────

def run_error_analysis(
    gold:       pd.DataFrame,
    preds:      pd.DataFrame,
    threshold:  float = 20.0,
    verbose:    bool  = True,
) -> dict:
    """
    Fusionne gold et preds, puis exécute toutes les analyses d'erreurs.
    """
    merged = gold.merge(preds, on="comment_id", suffixes=("_gold", "_pred"))
    n = len(merged)

    if n == 0:
        raise ValueError("Aucun commentaire en commun entre gold et preds (vérifiez comment_id).")

    print(f"Commentaires évalués : {n}")

    outliers     = score_outliers(merged, threshold)
    sent_errors  = sentiment_errors(merged)
    noise_err    = noise_errors(merged)
    ling_patterns = linguistic_patterns(outliers)

    # MAE global
    if "gold_score" in merged.columns and "score_global" in merged.columns:
        global_mae = mae(merged["gold_score"].tolist(), merged["score_global"].tolist())
    else:
        global_mae = None

    report = {
        "n_evaluated":    n,
        "mae_global":     global_mae,
        "outlier_threshold": threshold,
        "n_outliers":     len(outliers),
        "outlier_rate":   round(len(outliers) / n, 4) if n > 0 else 0.0,
        "score_outliers": outliers[:30],
        "sentiment_errors": sent_errors,
        "noise_detection":  noise_err,
        "linguistic_patterns": ling_patterns,
    }

    if verbose:
        print("\n" + "=" * 56)
        print("ANALYSE DES ERREURS DU PIPELINE")
        print("=" * 56)
        if global_mae is not None:
            print(f"MAE global          : {global_mae:.2f}")
        print(f"Outliers (|err|>{threshold}) : {len(outliers)}/{n} ({report['outlier_rate']:.1%})")

        if sent_errors.get("available"):
            er = sent_errors["error_rate"]
            sc = sent_errors["sarcasm_suspect_errors"]
            print(f"Erreurs sentiment   : {sent_errors['n_errors']}/{n} ({er:.1%})"
                  f"  dont sarcasme suspect : {sc}")

        if noise_err.get("available"):
            print(f"Bruit — Précision : {noise_err['precision']:.3f}  "
                  f"Rappel : {noise_err['recall']:.3f}  F1 : {noise_err['f1']:.3f}")

        if outliers:
            print(f"\nTop 5 outliers :")
            for o in outliers[:5]:
                print(f"  [{o['direction']:5s}] Δ={o['abs_error']:5.1f}  "
                      f"gold={o['gold_score']:5.1f}  pred={o['pred_score']:5.1f}  "
                      f"\"{o['text'][:60]}\"")
        print("=" * 56)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse des erreurs du pipeline multi-agents")
    p.add_argument("--gold",      required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument("--preds",     required=True, help="Prédictions pipeline (.jsonl/.csv)")
    p.add_argument("--threshold", type=float, default=20.0,
                   help="Seuil d'écart |gold-pred| pour qualifier un outlier (défaut 20)")
    p.add_argument("--output",    default="evaluation/results/error_analysis.json")
    p.add_argument("--verbose",   action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold  = load_gold(args.gold)
    preds = load_predictions(args.preds)

    print(f"Gold standard : {len(gold)} commentaires")
    print(f"Prédictions   : {len(preds)} commentaires")

    report = run_error_analysis(gold, preds, threshold=args.threshold, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
