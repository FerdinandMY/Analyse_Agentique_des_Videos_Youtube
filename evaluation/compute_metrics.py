"""
evaluation/compute_metrics.py
==============================
Calcule les métriques d'évaluation du pipeline YouTube Quality Analyzer
par rapport au gold standard annoté (Phase 2).

Métriques calculées :
  - Pearson r entre Score_Global et quality_score humain (H1 : r ≥ 0.60)
  - MAE / RMSE entre Score_Global et gold standard
  - Accuracy / F1-macro sur sentiment_label (3 classes)
  - Cohen's kappa entre prédictions pipeline et gold standard

Usage :
    python evaluation/compute_metrics.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --preds data/predictions/pipeline_predictions.jsonl \\
        --output evaluation/results/metrics_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Métriques numériques ──────────────────────────────────────────────────────

def pearson_r(x: list[float], y: list[float]) -> float:
    """Coefficient de corrélation de Pearson."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(
        sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)
    )
    return round(num / den, 4) if den > 0 else 0.0


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Absolute Error."""
    return round(sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true), 4)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root Mean Squared Error."""
    return round(math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)), 4)


def spearman_r(x: list[float], y: list[float]) -> float:
    """Coefficient de corrélation de Spearman (basé sur les rangs)."""
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[sorted_idx[j + 1]] == lst[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[sorted_idx[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = _ranks(x), _ranks(y)
    return pearson_r(rx, ry)


def fallback_rate(pred_conditions: list[str], fallback_marker: str = "fallback") -> float:
    """Taux d'appels LLM ayant nécessité le fallback heuristique."""
    if not pred_conditions:
        return 0.0
    n_fallback = sum(1 for c in pred_conditions if fallback_marker in c)
    return round(n_fallback / len(pred_conditions), 4)


# ── Métriques de classification ───────────────────────────────────────────────

def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {l: {l2: 0 for l2 in labels} for l in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4)


def f1_macro(y_true: list[str], y_pred: list[str]) -> float:
    """F1-score macro-averaged sur toutes les classes."""
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa entre deux listes de labels."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        matrix[cat_idx[a]][cat_idx[b]] += 1
    po = matrix.diagonal().sum() / n
    pe = float(np.dot(matrix.sum(axis=1) / n, matrix.sum(axis=0) / n))
    return round((po - pe) / (1 - pe), 4) if pe < 1.0 else 1.0


# ── Chargement des données ────────────────────────────────────────────────────

def load_gold(path: str) -> pd.DataFrame:
    """
    Charge le gold standard.
    Formats supportés : .jsonl, .json, .csv
    Retourne 1 ligne par commentaire (passage 1 uniquement si 2 annotateurs).
    """
    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(path)
    elif p.suffix in (".jsonl", ".json"):
        records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Format non supporté : {p.suffix}")

    # Garder seulement le premier annotateur si plusieurs passages
    if "annotator" in df.columns:
        first = df["annotator"].unique()[0]
        df = df[df["annotator"] == first].copy()

    # quality_score [1-5] → gold_score [0-100] pour comparaison avec Score_Global
    if "quality_score" in df.columns:
        df["gold_score"] = (df["quality_score"] - 1) / 4 * 100

    return df.reset_index(drop=True)


def load_predictions(path: str) -> pd.DataFrame:
    """
    Charge les prédictions du pipeline.
    Format attendu : .jsonl avec champs comment_id, score_global, sentiment_label.
    """
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(path)
    records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


# ── Rapport principal ─────────────────────────────────────────────────────────

def compute_all_metrics(gold: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """
    Fusionne gold et preds sur comment_id et calcule toutes les métriques.
    """
    merged = gold.merge(preds, on="comment_id", suffixes=("_gold", "_pred"))
    n = len(merged)

    if n == 0:
        raise ValueError("Aucun commentaire en commun entre gold et preds (vérifiez comment_id).")

    print(f"Commentaires évalués : {n}/{len(gold)} du gold standard")

    report: dict = {"n_evaluated": n}

    # ── H1 : Corrélation Pearson + Spearman (Score_Global vs gold_score) ───────
    if "gold_score" in merged.columns and "score_global" in merged.columns:
        gs = merged["gold_score"].tolist()
        sg = merged["score_global"].tolist()
        r   = pearson_r(gs, sg)
        rho = spearman_r(gs, sg)
        report["pearson_r"]         = r
        report["spearman_r"]        = rho
        report["h1_satisfied"]      = r >= 0.60
        report["h1_spearman_ok"]    = rho >= 0.55
        report["mae_score_global"]  = mae(gs, sg)
        report["rmse_score_global"] = rmse(gs, sg)
        print(f"\n── H1 : Corrélation Score_Global vs Gold ──")
        print(f"  Pearson r  = {r:.4f}  {'H1 validee (r >= 0.60)' if r >= 0.60 else 'H1 non validee (r < 0.60)'}")
        print(f"  Spearman r = {rho:.4f}  {'ok (>= 0.55)' if rho >= 0.55 else 'ko (< 0.55)'}")
        print(f"  MAE        = {report['mae_score_global']:.2f}")
        print(f"  RMSE       = {report['rmse_score_global']:.2f}")

    # ── Taux de fallback ──────────────────────────────────────────────────────
    if "condition" in merged.columns:
        rate = fallback_rate(merged["condition"].tolist())
        report["fallback_rate"]    = rate
        report["fallback_rate_ok"] = rate < 0.05
        print(f"\n── Taux de fallback ──")
        print(f"  {rate:.1%}  {'ok (< 5%)' if rate < 0.05 else 'KO (>= 5%)'}")

    # ── Sentiment classification ──────────────────────────────────────────────
    gold_sent_col = "sentiment_label_gold" if "sentiment_label_gold" in merged.columns else "sentiment_label"
    pred_sent_col = "sentiment_label_pred" if "sentiment_label_pred" in merged.columns else "sentiment_pred"

    if gold_sent_col in merged.columns and pred_sent_col in merged.columns:
        y_true = merged[gold_sent_col].tolist()
        y_pred = merged[pred_sent_col].tolist()
        acc  = accuracy(y_true, y_pred)
        f1   = f1_macro(y_true, y_pred)
        kap  = cohen_kappa(y_true, y_pred)
        report["sentiment_accuracy"] = acc
        report["sentiment_f1_macro"] = f1
        report["sentiment_kappa"]    = kap
        print(f"\n── Sentiment (3 classes) ──")
        print(f"  Accuracy  = {acc:.4f}")
        print(f"  F1-macro  = {f1:.4f}")
        print(f"  Kappa     = {kap:.4f}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calcule les métriques d'évaluation du pipeline")
    p.add_argument("--gold",   required=True, help="Chemin du gold standard (.jsonl/.csv)")
    p.add_argument("--preds",  required=True, help="Chemin des prédictions pipeline (.jsonl/.csv)")
    p.add_argument("--output", default="evaluation/results/metrics_report.json",
                   help="Chemin du rapport JSON de sortie")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gold  = load_gold(args.gold)
    preds = load_predictions(args.preds)

    print(f"Gold standard : {len(gold)} commentaires")
    print(f"Prédictions   : {len(preds)} commentaires")

    report = compute_all_metrics(gold, preds)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
