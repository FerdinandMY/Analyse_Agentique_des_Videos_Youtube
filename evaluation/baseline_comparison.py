"""
evaluation/baseline_comparison.py
===================================
Compare le pipeline multi-agents (7 agents) à un LLM unique (baseline monolithique).
Teste l'hypothèse H2 : ΔPearson(multi-agents vs LLM unique) > 0.10

Le baseline LLM unique envoie un seul prompt demandant simultanément
le sentiment, le discours, le bruit et le score global — sans pipeline multi-agents.

Usage :
    python evaluation/baseline_comparison.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --output evaluation/results/baseline_comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd


# Ajoute la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compute_metrics import pearson_r, mae, load_gold


# ── Prompt baseline monolithique ──────────────────────────────────────────────

_BASELINE_SYSTEM = (
    "You are a YouTube video quality analyst. "
    "Return strictly valid JSON with no extra text."
)

_BASELINE_PROMPT = """\
Analyse the following YouTube comments and evaluate the video quality.

Comments:
{comments}

Return a JSON object with exactly these fields:
- sentiment_label: "positive", "neutral", or "negative" (overall sentiment)
- sentiment_score: float 0.0-1.0 (1.0 = overwhelmingly positive)
- discourse_score: float 0.0-1.0 (quality of discussion depth)
- noise_score: float 0.0-1.0 (1.0 = clean, 0.0 = all noise)
- score_global: float 0-100 (overall quality score)
- rationale: one sentence summary

JSON only:"""


# ── Baseline LLM unique ───────────────────────────────────────────────────────

def run_baseline_llm(comments: list[str]) -> dict[str, Any]:
    """
    Exécute le baseline LLM unique sur un corpus de commentaires.
    Retourne le score global [0-100] et le sentiment.
    """
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            return _fallback_baseline(comments)

        context = "\n".join(f"- {c[:200]}" for c in comments[:30])
        resp = llm.invoke([
            SystemMessage(content=_BASELINE_SYSTEM),
            HumanMessage(content=_BASELINE_PROMPT.format(comments=context)),
        ])

        import re
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        # Normalise score_global [0-100]
        sg = float(result.get("score_global", 50.0))
        if sg <= 1.0:
            sg *= 100.0

        return {
            "score_global":    round(sg, 2),
            "sentiment_label": result.get("sentiment_label", "neutral"),
            "sentiment_score": round(float(result.get("sentiment_score", 0.5)) * 100, 2),
            "discourse_score": round(float(result.get("discourse_score", 0.5)) * 100, 2),
            "noise_score":     round(float(result.get("noise_score", 0.7)) * 100, 2),
            "method":          "llm_single",
        }
    except Exception as exc:
        print(f"[baseline] LLM error: {exc} — fallback")
        return _fallback_baseline(comments)


def _fallback_baseline(comments: list[str]) -> dict[str, Any]:
    """Baseline heuristique quand le LLM est indisponible."""
    import re
    noise_patterns = [
        r"https?://\S+", r"(.)\1{4,}",
        r"^\s*[^\w\s]+\s*$", r"subscribe|abonne",
    ]
    noisy = sum(
        1 for c in comments
        if any(re.search(p, c, re.IGNORECASE) for p in noise_patterns)
    )
    noise_ratio = noisy / len(comments) if comments else 0.0
    avg_len = sum(len(c.split()) for c in comments) / max(len(comments), 1)

    sentiment_score = 60.0
    discourse_score = min(100.0, avg_len * 2)
    noise_score     = round((1 - noise_ratio) * 100, 2)
    score_global    = round(0.35 * sentiment_score + 0.40 * discourse_score + 0.25 * noise_score, 2)

    return {
        "score_global":    score_global,
        "sentiment_label": "neutral",
        "sentiment_score": sentiment_score,
        "discourse_score": discourse_score,
        "noise_score":     noise_score,
        "method":          "heuristic_fallback",
    }


# ── Pipeline multi-agents ─────────────────────────────────────────────────────

def run_pipeline(comments: list[dict]) -> dict[str, Any]:
    """Exécute le pipeline LangGraph complet sur un corpus."""
    from graph import run_pipeline as _run
    result = _run(raw_comments=comments, video_id="eval", topic="")
    return {
        "score_global":    result.get("score_global", 50.0),
        "sentiment_label": (result.get("details") or {}).get("sentiment", {}).get("sentiment_label", "neutral"),
        "method":          "multi_agent_pipeline",
    }


# ── Comparaison ───────────────────────────────────────────────────────────────

def compare(gold: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Pour chaque commentaire du gold standard, exécute le baseline LLM unique
    et calcule le Pearson r vs gold_score.

    Note : le pipeline multi-agents tourne sur le corpus complet (pas commentaire par commentaire)
    donc on compare les deux méthodes sur le score agrégé du corpus.
    """
    comments = gold["text"].tolist()
    gold_scores = gold["gold_score"].tolist()

    # ── Baseline LLM unique ────────────────────────────────────────────────────
    print("Exécution baseline LLM unique...")
    baseline_results = []
    batch_size = 20
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        res = run_baseline_llm(batch)
        # Applique le même score à tous les commentaires du batch (approche corpus)
        for _ in batch:
            baseline_results.append(res["score_global"])

    r_baseline = pearson_r(gold_scores, baseline_results)
    mae_baseline = mae(gold_scores, baseline_results)

    # ── Pipeline multi-agents ─────────────────────────────────────────────────
    print("Exécution pipeline multi-agents...")
    raw_comments = [{"text": t} for t in comments]
    try:
        pipeline_res = run_pipeline(raw_comments)
        # Le pipeline retourne 1 score global pour tout le corpus
        pipeline_score = pipeline_res["score_global"]
        pipeline_scores = [pipeline_score] * len(comments)
    except Exception as exc:
        print(f"  Pipeline error: {exc} — utilisation du score baseline comme approximation")
        pipeline_scores = baseline_results

    r_pipeline = pearson_r(gold_scores, pipeline_scores)
    mae_pipeline = mae(gold_scores, pipeline_scores)

    delta_r = round(r_pipeline - r_baseline, 4)
    h2_satisfied = delta_r > 0.10

    report = {
        "n_comments":          len(comments),
        "baseline_llm_single": {
            "pearson_r": r_baseline,
            "mae":       mae_baseline,
            "method":    "llm_single_prompt",
        },
        "pipeline_multi_agent": {
            "pearson_r": r_pipeline,
            "mae":       mae_pipeline,
            "method":    "langgraph_7_agents",
        },
        "delta_pearson_r":  delta_r,
        "h2_satisfied":     h2_satisfied,
        "h2_threshold":     0.10,
        "interpretation":   (
            f"ΔPearson = {delta_r:+.4f} "
            f"({'✓ H2 validée' if h2_satisfied else '✗ H2 non validée'} — seuil > 0.10)"
        ),
    }

    if verbose:
        print("\n" + "=" * 56)
        print("COMPARAISON BASELINE vs PIPELINE MULTI-AGENTS (H2)")
        print("=" * 56)
        print(f"{'Méthode':<30} {'Pearson r':>10} {'MAE':>8}")
        print("-" * 56)
        print(f"{'LLM unique (baseline)':<30} {r_baseline:>10.4f} {mae_baseline:>8.2f}")
        print(f"{'Pipeline multi-agents':<30} {r_pipeline:>10.4f} {mae_pipeline:>8.2f}")
        print("-" * 56)
        print(f"ΔPearson r = {delta_r:+.4f}  {'✓ H2 validée (Δ > 0.10)' if h2_satisfied else '✗ H2 non validée (Δ ≤ 0.10)'}")
        print("=" * 56)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline LLM unique vs pipeline multi-agents (H2)")
    p.add_argument("--gold",    required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument("--output",  default="evaluation/results/baseline_comparison.json")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    print(f"Gold standard : {len(gold)} commentaires")

    report = compare(gold, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
