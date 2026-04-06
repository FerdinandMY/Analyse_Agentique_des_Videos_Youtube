"""
evaluation/ablation_study.py
==============================
Étude d'ablation : mesure l'impact de chaque agent analytique (A3, A4, A5)
sur la corrélation avec le gold standard.

Teste H3 : ΔPearson(sans A4) > ΔPearson(sans A3) > ΔPearson(sans A5)

Stratégie : pour chaque agent désactivé, le pipeline tourne avec un résultat
neutre/constant pour cet agent. On mesure ensuite ΔPearson(complet – sans Ax)
pour quantifier la contribution de chaque agent.

Usage :
    python evaluation/ablation_study.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --output evaluation/results/ablation_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compute_metrics import pearson_r, mae, load_gold


# ── Scores neutres quand un agent est désactivé ───────────────────────────────

_NEUTRAL_SENTIMENT = {
    "sentiment_label": "neutral",
    "sentiment_score": 50.0,
    "confidence":      0.5,
}

_NEUTRAL_DISCOURSE = {
    "discourse_score": 50.0,
    "avg_length":      10.0,
    "unique_ratio":    0.5,
    "question_ratio":  0.1,
}

_NEUTRAL_NOISE = {
    "noise_score": 70.0,
    "noise_ratio": 0.30,
}

# Poids PRD §4.3
_W_S = 0.35
_W_D = 0.40
_W_N = 0.25


def _score_global(s: float, d: float, n: float) -> float:
    return round(_W_S * s + _W_D * d + _W_N * n, 2)


# ── Pipeline complet ──────────────────────────────────────────────────────────

def _run_full_pipeline(comments: list[str]) -> float:
    try:
        from graph import run_pipeline
        result = run_pipeline(raw_comments=[{"text": t} for t in comments],
                              video_id="ablation_full", topic="")
        return float(result.get("score_global", 50.0))
    except Exception as exc:
        print(f"  [full] erreur : {exc} — fallback")
        return _heuristic_full(comments)


def _heuristic_full(comments: list[str]) -> float:
    import re
    patterns = [r"https?://\S+", r"(.)\1{4,}", r"subscribe|abonne"]
    noisy = sum(1 for c in comments if any(re.search(p, c, re.I) for p in patterns))
    noise_ratio = noisy / max(len(comments), 1)
    avg_len = sum(len(c.split()) for c in comments) / max(len(comments), 1)
    return _score_global(60.0, min(100.0, avg_len * 2), round((1 - noise_ratio) * 100, 2))


# ── Pipeline avec un agent neutralisé ────────────────────────────────────────

def _run_without_agent(comments: list[str], disabled: str) -> float:
    try:
        from graph import run_pipeline

        patch_targets = {
            "A3": ("agents.a3_sentiment.run_sentiment_agent",
                   lambda state: {**state, "sentiment": _NEUTRAL_SENTIMENT}),
            "A4": ("agents.a4_discourse.run_discourse_agent",
                   lambda state: {**state, "discourse": _NEUTRAL_DISCOURSE}),
            "A5": ("agents.a5_noise.run_noise_agent",
                   lambda state: {**state, "noise": _NEUTRAL_NOISE}),
        }
        module_path, neutral_fn = patch_targets[disabled]
        with patch(module_path, side_effect=neutral_fn):
            result = run_pipeline(raw_comments=[{"text": t} for t in comments],
                                  video_id=f"ablation_no_{disabled}", topic="")
        return float(result.get("score_global", 50.0))
    except Exception as exc:
        print(f"  [no {disabled}] erreur : {exc} — fallback")
        return _heuristic_without(comments, disabled)


def _heuristic_without(comments: list[str], disabled: str) -> float:
    import re
    patterns = [r"https?://\S+", r"(.)\1{4,}", r"subscribe|abonne"]
    noisy = sum(1 for c in comments if any(re.search(p, c, re.I) for p in patterns))
    noise_ratio = noisy / max(len(comments), 1)
    avg_len = sum(len(c.split()) for c in comments) / max(len(comments), 1)

    s = _NEUTRAL_SENTIMENT["sentiment_score"] if disabled == "A3" else 60.0
    d = _NEUTRAL_DISCOURSE["discourse_score"]  if disabled == "A4" else min(100.0, avg_len * 2)
    n = _NEUTRAL_NOISE["noise_score"]          if disabled == "A5" else round((1 - noise_ratio) * 100, 2)
    return _score_global(s, d, n)


# ── Étude d'ablation ──────────────────────────────────────────────────────────

def run_ablation(gold: pd.DataFrame, verbose: bool = True) -> dict:
    comments    = gold["text"].tolist()
    gold_scores = gold["gold_score"].tolist()
    n           = len(comments)
    batch_size  = 20

    def _batch_scores(fn) -> list[float]:
        out = []
        for i in range(0, n, batch_size):
            batch = comments[i : i + batch_size]
            score = fn(batch)
            out.extend([score] * len(batch))
        return out

    print("Pipeline complet (référence)...")
    full_scores = _batch_scores(_run_full_pipeline)
    r_full   = pearson_r(gold_scores, full_scores)
    mae_full = mae(gold_scores, full_scores)

    results = {}
    for agent in ("A3", "A4", "A5"):
        print(f"Pipeline sans {agent}...")
        no_scores = _batch_scores(lambda batch, a=agent: _run_without_agent(batch, a))
        r_no   = pearson_r(gold_scores, no_scores)
        mae_no = mae(gold_scores, no_scores)
        delta  = round(r_full - r_no, 4)
        results[agent] = {
            "pearson_r_without": r_no,
            "mae_without":       mae_no,
            "delta_pearson_r":   delta,
        }

    d_a3 = results["A3"]["delta_pearson_r"]
    d_a4 = results["A4"]["delta_pearson_r"]
    d_a5 = results["A5"]["delta_pearson_r"]
    h3_satisfied = d_a4 > d_a3 > d_a5

    ranking = sorted(
        [("A3", d_a3), ("A4", d_a4), ("A5", d_a5)],
        key=lambda x: x[1], reverse=True,
    )

    report = {
        "n_comments": n,
        "full_pipeline": {"pearson_r": r_full, "mae": mae_full},
        "ablation":      results,
        "ranking_by_contribution": [{"agent": a, "delta_pearson_r": d} for a, d in ranking],
        "h3_satisfied":     h3_satisfied,
        "h3_interpretation": (
            f"ΔPearson A4={d_a4:+.4f}, A3={d_a3:+.4f}, A5={d_a5:+.4f} "
            f"({'✓ H3 validée (A4>A3>A5)' if h3_satisfied else '✗ H3 non validée'})"
        ),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("ÉTUDE D'ABLATION — Contribution par agent (H3)")
        print("=" * 60)
        print(f"{'Pipeline':<30} {'Pearson r':>10} {'MAE':>8}")
        print("-" * 60)
        print(f"{'Complet (A1–A7)':<30} {r_full:>10.4f} {mae_full:>8.2f}")
        for agent, res in results.items():
            delta = res["delta_pearson_r"]
            print(f"  {'Sans ' + agent:<28} {res['pearson_r_without']:>10.4f}"
                  f" {res['mae_without']:>8.2f}  (Δ={delta:+.4f})")
        print("-" * 60)
        print("Classement des contributions :")
        for rank, (agent, delta) in enumerate(ranking, 1):
            print(f"  {rank}. {agent} : Δr = {delta:+.4f}")
        print(f"\n{'✓' if h3_satisfied else '✗'} H3 : A4 > A3 > A5 "
              f"— {'validée' if h3_satisfied else 'non validée'}")
        print("=" * 60)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study — contribution de chaque agent (H3)")
    p.add_argument("--gold",    required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument("--output",  default="evaluation/results/ablation_study.json")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    print(f"Gold standard : {len(gold)} commentaires")

    report = run_ablation(gold, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
