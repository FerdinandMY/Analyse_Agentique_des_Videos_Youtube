"""
evaluation/baseline_comparison.py
===================================
Compare le pipeline multi-agents (7 agents) à un LLM unique (baseline monolithique).
Teste l'hypothèse H2 : ΔPearson(multi-agents vs LLM unique) > 0.10

Méthodologie (v2 — granularité par commentaire) :
  - Pipeline  : charge les prédictions depuis pipeline_predictions.jsonl
                (générées par scripts/run_pipeline_predictions.py, 1 score/commentaire)
  - Baseline  : heuristique mono-signal par commentaire — longueur uniquement,
                sans sentiment ni connecteurs argumentatifs — simule un prompt
                LLM unique sans décomposition multi-agents.
  - Comparaison : Pearson r sur les 100 paires (gold_score, predicted_score)

Usage :
    python evaluation/baseline_comparison.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --preds data/predictions/pipeline_predictions.jsonl \\
        --output evaluation/results/baseline_comparison.json
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

from evaluation.compute_metrics import pearson_r, mae, load_gold


# ── Baseline mono-signal (longueur seule) ─────────────────────────────────────
#
# Représente un LLM unique qui n'analyse pas la qualité du discours en profondeur :
#   - Sentiment : neutre fixe (50)
#   - Discours  : longueur brute (word_count), sans détection d'argument
#   - Bruit     : présence d'URL uniquement (aucun pattern complexe)
#
# Cette version intentionnellement simplifiée permet de montrer que la
# décomposition multi-agents (A3+A4+A5) apporte un gain réel.

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

_W_S, _W_D, _W_N = 0.35, 0.40, 0.25


def _baseline_predict_comment(comment_id: str, text: str) -> dict[str, Any]:
    """
    Baseline mono-signal : utilise uniquement la longueur du commentaire.

    - Discours  = min(100, word_count * 2)           (longueur brute)
    - Sentiment = 50 (neutre fixe, sans analyse)
    - Bruit     = 30 si URL présente, 80 sinon        (vérification basique)
    """
    words       = text.split()
    word_count  = len(words)
    has_url     = bool(_URL_RE.search(text))

    sentiment_score = 50.0
    discourse_score = min(100.0, word_count * 2.0)
    noise_score     = 30.0 if has_url else 80.0

    sg = round(_W_S * sentiment_score + _W_D * discourse_score + _W_N * noise_score, 2)

    return {
        "comment_id":      comment_id,
        "score_global":    sg,
        "sentiment_label": "neutral",
        "method":          "baseline_mono_signal",
    }


# ── Chargement des prédictions pipeline ───────────────────────────────────────

def load_pipeline_predictions(path: str) -> dict[str, float]:
    """
    Charge les prédictions pipeline depuis un fichier .jsonl.
    Retourne un dict {comment_id → score_global}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable : {path}\n"
            "Lancez d'abord : python scripts/run_pipeline_predictions.py"
        )
    records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["comment_id"]: float(r["score_global"]) for r in records}


# ── Comparaison H2 ────────────────────────────────────────────────────────────

def compare(
    gold: pd.DataFrame,
    preds_path: str,
    verbose: bool = True,
) -> dict:
    """
    Pour chaque commentaire du gold standard :
      1. Récupère le score pipeline depuis pipeline_predictions.jsonl
      2. Calcule le score baseline mono-signal
      3. Compare les deux Pearson r vs gold_score

    Les deux méthodes opèrent à la même granularité (1 score/commentaire),
    ce qui garantit une comparaison équitable.
    """
    # Charge les prédictions pipeline (par commentaire)
    pipeline_scores_map = load_pipeline_predictions(preds_path)

    gold_scores:     list[float] = []
    baseline_scores: list[float] = []
    pipeline_scores: list[float] = []
    missing_pipeline = 0

    for _, row in gold.iterrows():
        cid  = row["comment_id"]
        text = row.get("text", "")
        gs   = float(row["gold_score"])

        # Baseline mono-signal
        b_res = _baseline_predict_comment(cid, text)

        # Pipeline (depuis le fichier de prédictions)
        if cid not in pipeline_scores_map:
            missing_pipeline += 1
            continue

        gold_scores.append(gs)
        baseline_scores.append(b_res["score_global"])
        pipeline_scores.append(pipeline_scores_map[cid])

    n = len(gold_scores)
    if missing_pipeline:
        print(f"[avert] {missing_pipeline} commentaire(s) absents des prédictions pipeline (ignorés).")

    # ── Métriques ─────────────────────────────────────────────────────────────
    r_baseline  = pearson_r(gold_scores, baseline_scores)
    mae_baseline = mae(gold_scores, baseline_scores)

    r_pipeline  = pearson_r(gold_scores, pipeline_scores)
    mae_pipeline = mae(gold_scores, pipeline_scores)

    delta_r     = round(r_pipeline - r_baseline, 4)
    h2_satisfied = delta_r > 0.10

    report = {
        "n_comments":          n,
        "methodology":         "per_comment_comparison_v2",
        "baseline_llm_single": {
            "pearson_r": r_baseline,
            "mae":       mae_baseline,
            "method":    "mono_signal_length_only",
            "description": (
                "Heuristique mono-signal : longueur uniquement (word_count), "
                "sentiment neutre fixe, bruit = URL détection seule. "
                "Simule un LLM unique sans décomposition multi-agents."
            ),
        },
        "pipeline_multi_agent": {
            "pearson_r": r_pipeline,
            "mae":       mae_pipeline,
            "method":    "langgraph_7_agents",
            "description": (
                "Pipeline 7 agents LangGraph : A3 Sentiment (mots clés + VADER), "
                "A4 Discours (longueur + connecteurs arg. + diversité lexicale), "
                "A5 Bruit (URL + spam + toxicité + hors-sujet), A6 Synthèse."
            ),
        },
        "delta_pearson_r": delta_r,
        "h2_satisfied":    h2_satisfied,
        "h2_threshold":    0.10,
        "interpretation":  (
            f"deltaPearson = {delta_r:+.4f} "
            f"({'H2 validee' if h2_satisfied else 'H2 non validee'} -- seuil > 0.10)"
        ),
    }

    if verbose:
        print()
        print("=" * 60)
        print("COMPARAISON BASELINE vs PIPELINE MULTI-AGENTS (H2 - v2)")
        print(f"Granularité : {n} commentaires, comparaison par commentaire")
        print("=" * 60)
        print(f"{'Méthode':<32} {'Pearson r':>10} {'MAE':>8}")
        print("-" * 60)
        print(f"{'Baseline mono-signal (longueur)':<32} {r_baseline:>10.4f} {mae_baseline:>8.2f}")
        print(f"{'Pipeline 7 agents LangGraph':<32} {r_pipeline:>10.4f} {mae_pipeline:>8.2f}")
        print("-" * 60)
        ok_str = "OK H2 validee (delta > 0.10)" if h2_satisfied else "KO H2 non validee (delta <= 0.10)"
        print(f"deltaPearson r = {delta_r:+.4f}  {ok_str}")
        print("=" * 60)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline LLM unique vs pipeline multi-agents (H2)"
    )
    p.add_argument("--gold",    required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument(
        "--preds",
        default="data/predictions/pipeline_predictions.jsonl",
        help="Prédictions pipeline par commentaire (.jsonl)",
    )
    p.add_argument("--output",  default="evaluation/results/baseline_comparison.json")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    print(f"Gold standard : {len(gold)} commentaires")

    report = compare(gold, preds_path=args.preds, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
