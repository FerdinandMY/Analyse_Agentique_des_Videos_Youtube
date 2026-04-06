"""
evaluation/ablation_study.py
==============================
Étude d'ablation — PRD v3.0 AC-09

Compare 4 conditions de raisonnement pour tester H4 :
  B0 — Prompt direct (sans CoT)         : baseline sans raisonnement structuré
  B1 — CoT seul                          : Thought/Action/Result, sans tools
  B2 — CoT + ToT                         : + ToT conditionnel A4 / systématique A7
  B3 — CoT + ToT + Tools (pipeline v3.0) : pipeline complet, référence

H4 : le CoT/ToT réduit le taux d'erreur de classification d'au moins 15%
     par rapport au prompt direct (B0 → B3).

Métriques calculées par condition :
  - Pearson r (Score_Global vs gold_score)
  - MAE
  - F1-macro sentiment (si gold contient sentiment_label)
  - Gain F1 par rapport à B0

Usage :
    python evaluation/ablation_study.py \\
        --gold  data/gold_standard/gold_standard.jsonl \\
        --output evaluation/results/ablation_study.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compute_metrics import pearson_r, mae, f1_macro, load_gold


# ── Poids PRD §3.6 ────────────────────────────────────────────────────────────

_W_S, _W_D, _W_N = 0.35, 0.40, 0.25


def _score_global(s: float, d: float, n: float) -> float:
    return round(_W_S * s + _W_D * d + _W_N * n, 2)


# ── Heuristiques de base (partagées par tous les fallbacks) ───────────────────

def _heuristic_components(comments: list[str]) -> tuple[float, float, float]:
    """Retourne (sentiment, discourse, noise) calculés heuristiquement."""
    noise_patterns = [r"https?://\S+", r"(.)\1{4,}", r"subscribe|abonne"]
    noisy = sum(1 for c in comments if any(re.search(p, c, re.I) for p in noise_patterns))
    noise_ratio = noisy / max(len(comments), 1)
    avg_len = sum(len(c.split()) for c in comments) / max(len(comments), 1)
    return 60.0, min(100.0, avg_len * 2), round((1 - noise_ratio) * 100, 2)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONS D'ABLATION
# ══════════════════════════════════════════════════════════════════════════════

# ── B0 — Prompt direct (sans CoT) ────────────────────────────────────────────

_B0_SYSTEM = "You are a YouTube comment analyst. Return strictly valid JSON."

_B0_PROMPT = """\
Analyse these YouTube comments and rate the video quality.

Comments:
{comments}

Return JSON with:
- sentiment_label: "positive", "neutral", or "negative"
- sentiment_score: float 0-100
- discourse_score: float 0-100
- noise_score: float 0-100
- score_global: float 0-100

JSON only:"""


def run_b0_direct(comments: list[str]) -> dict:
    """B0 : prompt direct sans raisonnement structuré."""
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            return _b0_heuristic(comments)

        context = "\n".join(f"- {c[:200]}" for c in comments[:30])
        resp = llm.invoke([
            SystemMessage(content=_B0_SYSTEM),
            HumanMessage(content=_B0_PROMPT.format(comments=context)),
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        sg = float(result.get("score_global", 50.0))
        if sg <= 1.0:
            sg *= 100.0
        return {
            "score_global":    round(sg, 2),
            "sentiment_label": result.get("sentiment_label", "neutral"),
            "sentiment_score": round(float(result.get("sentiment_score", 50.0)), 2),
            "discourse_score": round(float(result.get("discourse_score", 50.0)), 2),
            "noise_score":     round(float(result.get("noise_score", 70.0)), 2),
            "condition":       "B0_direct",
        }
    except Exception as exc:
        print(f"  [B0] erreur LLM : {exc} — fallback heuristique")
        return _b0_heuristic(comments)


def _b0_heuristic(comments: list[str]) -> dict:
    s, d, n = _heuristic_components(comments)
    return {
        "score_global":    _score_global(s, d, n),
        "sentiment_label": "neutral",
        "sentiment_score": s,
        "discourse_score": d,
        "noise_score":     n,
        "condition":       "B0_heuristic_fallback",
    }


# ── B1 — CoT seul (sans tools) ───────────────────────────────────────────────

_B1_PROMPT = """\
Analyse these YouTube comments step by step.

Comments:
{comments}

Think step by step:
Thought 1: What is the overall emotional tone? Look for positive/negative/neutral words.
Thought 2: Is there sarcasm or irony? Check for contradictory signals.
Thought 3: How deep and informative is the discussion? Check argument length and variety.
Thought 4: How much noise (spam, ads, repetitions) is present?
Result: Synthesise into scores.

Return JSON only:
{{
  "reasoning": "your step-by-step thoughts",
  "sentiment_label": "positive|neutral|negative",
  "sentiment_score": <0-100>,
  "discourse_score": <0-100>,
  "noise_score": <0-100>,
  "score_global": <0-100>
}}"""


def run_b1_cot(comments: list[str]) -> dict:
    """B1 : CoT sans tools."""
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            return _b0_heuristic(comments)  # même fallback que B0

        context = "\n".join(f"- {c[:200]}" for c in comments[:30])
        resp = llm.invoke([
            SystemMessage(content="You are a YouTube quality analyst. Use step-by-step reasoning. Return JSON only."),
            HumanMessage(content=_B1_PROMPT.format(comments=context)),
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        sg = float(result.get("score_global", 50.0))
        if sg <= 1.0:
            sg *= 100.0
        return {
            "score_global":    round(sg, 2),
            "sentiment_label": result.get("sentiment_label", "neutral"),
            "sentiment_score": round(float(result.get("sentiment_score", 50.0)), 2),
            "discourse_score": round(float(result.get("discourse_score", 50.0)), 2),
            "noise_score":     round(float(result.get("noise_score", 70.0)), 2),
            "reasoning":       result.get("reasoning", ""),
            "condition":       "B1_cot",
        }
    except Exception as exc:
        print(f"  [B1] erreur LLM : {exc} — fallback heuristique")
        return _b0_heuristic(comments)


# ── B2 — CoT + ToT (sans tools) ──────────────────────────────────────────────

_B2_PROMPT = """\
Analyse these YouTube comments using multiple reasoning angles.

Comments:
{comments}

Explore 3 branches in parallel:

Branch 1 — Sentiment & Emotion:
  - What is the dominant emotion? (positive/neutral/negative)
  - Is there sarcasm or irony? Score: <0-100>

Branch 2 — Discourse Quality:
  - How deep and informative is the discussion?
  - Are there arguments, examples, or just reactions? Score: <0-100>

Branch 3 — Noise & Spam:
  - What proportion is spam, ads, or repetitions?
  - Score purity (100 = fully clean): <0-100>

Evaluate each branch, then synthesise.

Return JSON only:
{{
  "branch_1": {{"sentiment_label": "...", "sentiment_score": <0-100>, "rationale": "..."}},
  "branch_2": {{"discourse_score": <0-100>, "rationale": "..."}},
  "branch_3": {{"noise_score": <0-100>, "rationale": "..."}},
  "synthesis": {{
    "sentiment_label": "...",
    "sentiment_score": <0-100>,
    "discourse_score": <0-100>,
    "noise_score": <0-100>,
    "score_global": <0-100>
  }}
}}"""


def run_b2_cot_tot(comments: list[str]) -> dict:
    """B2 : CoT + ToT sans tools."""
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            return _b0_heuristic(comments)

        context = "\n".join(f"- {c[:200]}" for c in comments[:30])
        resp = llm.invoke([
            SystemMessage(content="You are a YouTube quality analyst. Explore multiple reasoning angles. Return JSON only."),
            HumanMessage(content=_B2_PROMPT.format(comments=context)),
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        synth = result.get("synthesis", {})
        sg = float(synth.get("score_global", 50.0))
        if sg <= 1.0:
            sg *= 100.0

        return {
            "score_global":    round(sg, 2),
            "sentiment_label": synth.get("sentiment_label", "neutral"),
            "sentiment_score": round(float(synth.get("sentiment_score", 50.0)), 2),
            "discourse_score": round(float(synth.get("discourse_score", 50.0)), 2),
            "noise_score":     round(float(synth.get("noise_score", 70.0)), 2),
            "tot_branches":    {k: result[k] for k in ("branch_1", "branch_2", "branch_3") if k in result},
            "condition":       "B2_cot_tot",
        }
    except Exception as exc:
        print(f"  [B2] erreur LLM : {exc} — fallback heuristique")
        return _b0_heuristic(comments)


# ── B3 — CoT + ToT + Tools (pipeline complet v3.0) ───────────────────────────

def run_b3_full_pipeline(comments: list[str]) -> dict:
    """B3 : pipeline complet LangGraph v3.0."""
    try:
        from graph import run_pipeline
        result = run_pipeline(
            raw_comments=[{"text": t} for t in comments],
            video_id="ablation_b3",
            topic="",
        )
        sg = float(result.get("score_global", 50.0))
        details = result.get("details") or {}
        sent = details.get("sentiment") or {}
        return {
            "score_global":    round(sg, 2),
            "sentiment_label": sent.get("sentiment_label", "neutral"),
            "sentiment_score": round(float(sent.get("sentiment_score", 50.0)), 2),
            "discourse_score": round(float((details.get("discourse") or {}).get("discourse_score", 50.0)), 2),
            "noise_score":     round(float((details.get("noise") or {}).get("noise_score", 70.0)), 2),
            "fallback_used":   result.get("fallback_used", False),
            "hallucination_flags": result.get("hallucination_flags", []),
            "condition":       "B3_cot_tot_tools",
        }
    except Exception as exc:
        print(f"  [B3] erreur pipeline : {exc} — fallback heuristique")
        s, d, n = _heuristic_components(comments)
        return {
            "score_global":    _score_global(s, d, n),
            "sentiment_label": "neutral",
            "sentiment_score": s,
            "discourse_score": d,
            "noise_score":     n,
            "condition":       "B3_heuristic_fallback",
        }


# ══════════════════════════════════════════════════════════════════════════════
# ÉTUDE D'ABLATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

_CONDITIONS = [
    ("B0", "Prompt direct (sans CoT)",     run_b0_direct),
    ("B1", "CoT seul",                     run_b1_cot),
    ("B2", "CoT + ToT",                    run_b2_cot_tot),
    ("B3", "CoT + ToT + Tools (pipeline)", run_b3_full_pipeline),
]


def run_ablation(gold: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Exécute les 4 conditions d'ablation et calcule :
    - Pearson r / MAE par condition
    - F1-macro sentiment si disponible dans le gold
    - Gain F1 vs B0 (H4 : gain >= 15 pts)
    """
    comments     = gold["text"].tolist()
    gold_scores  = gold["gold_score"].tolist()
    gold_sentiments = gold["sentiment_label"].tolist() if "sentiment_label" in gold.columns else None
    n            = len(comments)
    batch_size   = 20

    def _batch_run(fn) -> list[dict]:
        """Exécute fn par batch, duplique le résultat pour chaque commentaire du batch."""
        results = []
        for i in range(0, n, batch_size):
            batch = comments[i : i + batch_size]
            res   = fn(batch)
            results.extend([res] * len(batch))
        return results

    condition_results = {}

    for cid, label, fn in _CONDITIONS:
        print(f"Condition {cid} — {label}...")
        batch_results = _batch_run(fn)

        scores = [r["score_global"] for r in batch_results]
        labels = [r.get("sentiment_label", "neutral") for r in batch_results]

        r_val   = pearson_r(gold_scores, scores)
        mae_val = mae(gold_scores, scores)

        f1_val  = None
        if gold_sentiments:
            f1_val = f1_macro(gold_sentiments, labels)

        fallback_count = sum(
            1 for r in batch_results
            if "fallback" in r.get("condition", "")
        )

        condition_results[cid] = {
            "label":        label,
            "pearson_r":    r_val,
            "mae":          mae_val,
            "f1_sentiment": f1_val,
            "fallback_count": fallback_count,
            "fallback_rate": round(fallback_count / n, 4) if n > 0 else 0.0,
        }

    # ── H4 : gain CoT/ToT vs prompt direct ───────────────────────────────────
    r_b0  = condition_results["B0"]["pearson_r"]
    r_b3  = condition_results["B3"]["pearson_r"]
    f1_b0 = condition_results["B0"]["f1_sentiment"]
    f1_b3 = condition_results["B3"]["f1_sentiment"]

    # Gain Pearson (global quality)
    delta_pearson_b0_b3 = round(r_b3 - r_b0, 4)

    # Gain F1 sentiment (classification error reduction)
    delta_f1_b0_b3 = round((f1_b3 - f1_b0) * 100, 2) if (f1_b0 is not None and f1_b3 is not None) else None
    h4_satisfied   = (delta_f1_b0_b3 is not None and delta_f1_b0_b3 >= 15.0)

    # Gain intermédiaires
    gains = {}
    for pair, (ca, cb) in [("B0->B1", ("B0","B1")), ("B1->B2", ("B1","B2")), ("B2->B3", ("B2","B3"))]:
        f1a = condition_results[ca]["f1_sentiment"]
        f1b = condition_results[cb]["f1_sentiment"]
        gains[pair] = {
            "delta_pearson_r": round(condition_results[cb]["pearson_r"] - condition_results[ca]["pearson_r"], 4),
            "delta_f1_pts":    round((f1b - f1a) * 100, 2) if (f1a is not None and f1b is not None) else None,
        }

    report = {
        "n_comments": n,
        "conditions": condition_results,
        "gains_by_step": gains,
        "h4": {
            "delta_pearson_r_b0_b3":  delta_pearson_b0_b3,
            "delta_f1_pts_b0_b3":     delta_f1_b0_b3,
            "h4_satisfied":           h4_satisfied,
            "h4_threshold_f1_gain_pts": 15.0,
            "interpretation": (
                f"Gain F1 B0→B3 = {delta_f1_b0_b3:+.1f} pts"
                if delta_f1_b0_b3 is not None
                else "Gain F1 non calculable (gold sans sentiment_label)"
            ) + (
                f"  ({'H4 validee (gain >= 15 pts)' if h4_satisfied else 'H4 non validee (gain < 15 pts)'})"
                if delta_f1_b0_b3 is not None else ""
            ),
        },
    }

    if verbose:
        _print_report(report)

    return report


def _print_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print("ETUDE D'ABLATION — CoT / ToT / Tools (H4)")
    print("=" * 72)
    print(f"{'Condition':<35} {'Pearson r':>10} {'MAE':>7} {'F1-sent':>9} {'Fallback':>9}")
    print("-" * 72)
    for cid, res in report["conditions"].items():
        f1  = f"{res['f1_sentiment']:.4f}" if res["f1_sentiment"] is not None else "    N/A"
        print(f"  {cid} — {res['label']:<30} {res['pearson_r']:>10.4f}"
              f" {res['mae']:>7.2f} {f1:>9} {res['fallback_rate']:>8.1%}")
    print("-" * 72)
    g = report["gains_by_step"]
    for step, vals in g.items():
        f1g = f"{vals['delta_f1_pts']:+.1f} pts F1" if vals["delta_f1_pts"] is not None else "N/A"
        print(f"  Gain {step:<12} Δr={vals['delta_pearson_r']:+.4f}  {f1g}")
    print("-" * 72)
    h4 = report["h4"]
    print(f"\n{h4['interpretation']}")
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation study — PRD v3.0 AC-09 (B0 direct / B1 CoT / B2 CoT+ToT / B3 full)"
    )
    p.add_argument("--gold",    required=True, help="Gold standard (.jsonl/.csv)")
    p.add_argument("--output",  default="evaluation/results/ablation_study.json")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--conditions", nargs="+", choices=["B0", "B1", "B2", "B3"],
                   default=["B0", "B1", "B2", "B3"],
                   help="Conditions à exécuter (défaut : toutes)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    print(f"Gold standard : {len(gold)} commentaires")

    # Filtre les conditions si --conditions est partiel
    global _CONDITIONS
    if args.conditions != ["B0", "B1", "B2", "B3"]:
        _CONDITIONS = [(cid, lbl, fn) for cid, lbl, fn in _CONDITIONS if cid in args.conditions]

    report = run_ablation(gold, verbose=args.verbose)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRapport sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
