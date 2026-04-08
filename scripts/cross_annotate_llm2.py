"""
scripts/cross_annotate_llm2.py — Annotation croisée LLM1 x LLM2
=================================================================
Implémente l'annotation de niveau 2 : deux LLMs architecturalement
différents annotent le même gold standard et on calcule le kappa
inter-modèles — mesure bien plus rigoureuse que deux passes du même LLM.

LLM1 : Backend configuré dans .env (OpenAI-compatible, ex. gpt-4o-mini)
LLM2 : Groq (Llama-3.1-70B)   — gratuit, clé sur console.groq.com
     OU Google Gemini          — gratuit, clé sur aistudio.google.com

Pré-requis :
    pip install langchain-groq          # pour Groq
    pip install langchain-google-genai  # pour Gemini

Variables d'environnement (.env) :
    # LLM1 (déjà en place)
    OPENAI_API_KEY=...
    LLM_MODEL=gpt-4o-mini

    # LLM2 — choisir UN des deux :
    GROQ_API_KEY=...               # option A (recommandé)
    GOOGLE_API_KEY=...             # option B

Usage :
    python scripts/cross_annotate_llm2.py \\
        --gold data/gold_standard/gold_standard.jsonl \\
        --output evaluation/results/cross_annotation_kappa.json

    # Forcer LLM2 = Gemini (si GROQ_API_KEY absent)
    python scripts/cross_annotate_llm2.py --llm2 gemini

    # Tester sur 10 commentaires seulement
    python scripts/cross_annotate_llm2.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Charger .env si présent
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("cross_annotate")

# ── Prompt d'annotation (identique à annotate_gold_standard.py) ──────────────

_SYSTEM_PROMPT = (
    "You are an expert annotator for YouTube comment quality analysis. "
    "Return strictly valid JSON with no extra text."
)

_ANNOTATION_PROMPT = """\
Annotate the following YouTube comment for a quality analysis gold standard.

Comment:
\"\"\"{comment}\"\"\"

Return a JSON object with exactly these fields:
- sentiment_label: "positive", "neutral", or "negative"
- quality_score: integer 1-5
    (1=pure noise/spam, 2=low value, 3=neutral/ok, 4=informative, 5=very insightful)
- noise_category: one of "spam", "offtopic", "reaction_vide", "toxique", "bot", "ok"
- confidence: float 0.0-1.0

JSON only:"""


# ── Construction des clients LLM ──────────────────────────────────────────────

def _build_llm1() -> Any:
    """LLM1 : backend principal du projet (OpenAI-compatible)."""
    from models.llm_loader import get_llm
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "LLM1 indisponible. Vérifiez OPENAI_API_KEY dans votre .env"
        )
    return llm


def _build_llm2_groq(model: str = "llama-3.1-70b-versatile") -> Any:
    """
    LLM2 : Groq — Llama 3.1 70B (gratuit).
    Clé API gratuite sur https://console.groq.com/
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise RuntimeError(
            "Package manquant. Installez : pip install langchain-groq"
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY absent du .env\n"
            "  1. Créez un compte gratuit sur https://console.groq.com/\n"
            "  2. Générez une clé API\n"
            "  3. Ajoutez GROQ_API_KEY=gsk_... dans votre fichier .env"
        )

    return ChatGroq(api_key=api_key, model=model, temperature=0.1)


def _build_llm2_gemini(model: str = "gemini-1.5-flash") -> Any:
    """
    LLM2 : Google Gemini 1.5 Flash (gratuit, 1M tokens/mois).
    Clé API gratuite sur https://aistudio.google.com/
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise RuntimeError(
            "Package manquant. Installez : pip install langchain-google-genai"
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY absent du .env\n"
            "  1. Allez sur https://aistudio.google.com/\n"
            "  2. Cliquez 'Get API Key'\n"
            "  3. Ajoutez GOOGLE_API_KEY=AIza... dans votre fichier .env"
        )

    return ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=0.1)


# ── Appel d'annotation sur un commentaire ─────────────────────────────────────

def _annotate_one(llm: Any, text: str, llm_name: str, retries: int = 3) -> dict:
    """
    Appelle le LLM pour annoter un commentaire.
    Retourne {sentiment_label, quality_score, noise_category, confidence}.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_ANNOTATION_PROMPT.format(comment=text[:500])),
    ]

    for attempt in range(retries):
        try:
            resp = llm.invoke(messages)
            raw  = resp.content if hasattr(resp, "content") else str(resp)

            # Nettoyer le markdown éventuel
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return {
                "sentiment_label": str(data.get("sentiment_label", "neutral")),
                "quality_score":   int(data.get("quality_score", 3)),
                "noise_category":  str(data.get("noise_category", "ok")),
                "confidence":      float(data.get("confidence", 0.5)),
            }

        except (json.JSONDecodeError, ValueError):
            logger.warning("%s: JSON malformé (tentative %d/%d)", llm_name, attempt+1, retries)
            if attempt < retries - 1:
                time.sleep(1)

        except Exception as exc:
            logger.warning("%s: erreur — %s (tentative %d/%d)", llm_name, exc, attempt+1, retries)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # backoff exponentiel

    # Fallback neutre
    return {"sentiment_label": "neutral", "quality_score": 3,
            "noise_category": "ok", "confidence": 0.0}


# ── Métriques de concordance ──────────────────────────────────────────────────

def _cohen_kappa(labels1: list[str], labels2: list[str]) -> float:
    """Cohen's kappa entre deux listes de labels catégoriels."""
    classes = sorted(set(labels1) | set(labels2))
    n       = len(labels1)
    if n == 0:
        return 0.0

    # Accord observé
    p_o = sum(a == b for a, b in zip(labels1, labels2)) / n

    # Accord attendu par hasard
    p_e = 0.0
    for cls in classes:
        p1 = labels1.count(cls) / n
        p2 = labels2.count(cls) / n
        p_e += p1 * p2

    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def _mae(scores1: list[float], scores2: list[float]) -> float:
    """MAE entre deux listes de scores numériques."""
    if not scores1:
        return 0.0
    return round(sum(abs(a - b) for a, b in zip(scores1, scores2)) / len(scores1), 4)


def _pearson(x: list[float], y: list[float]) -> float:
    """Coefficient de Pearson."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    den = math.sqrt(
        sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y)
    )
    return round(num/den, 4) if den > 0 else 0.0


# ── Fonction principale ───────────────────────────────────────────────────────

def cross_annotate(
    gold_path:   str,
    output_path: str,
    llm2_type:   str = "auto",
    limit:       Optional[int] = None,
) -> dict:
    """
    Charge le gold standard, re-annote avec LLM2, calcule le kappa inter-modèles.

    Args:
        gold_path  : Chemin vers gold_standard.jsonl
        output_path: Chemin de sortie JSON des résultats
        llm2_type  : "groq" | "gemini" | "auto" (essaie Groq d'abord)
        limit      : Nombre max de commentaires à traiter (None = tous)
    """
    # ── Charger le gold standard ──────────────────────────────────────────────
    gold_path_p = Path(gold_path)
    if not gold_path_p.exists():
        raise FileNotFoundError(f"Gold standard introuvable : {gold_path}")

    with open(gold_path_p, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit:
        records = records[:limit]

    logger.info("cross_annotate: %d commentaires charges", len(records))

    # ── Construire les deux LLMs ──────────────────────────────────────────────
    print("\n[1/4] Initialisation LLM1 (backend principal)...")
    llm1 = _build_llm1()
    llm1_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    print(f"      LLM1 OK : {llm1_name}")

    print("[1/4] Initialisation LLM2 (annotateur independant)...")
    llm2 = None
    llm2_name = ""

    if llm2_type in ("auto", "groq"):
        try:
            llm2 = _build_llm2_groq()
            llm2_name = "llama-3.1-70b-versatile (Groq)"
            print(f"      LLM2 OK : {llm2_name}")
        except RuntimeError as e:
            if llm2_type == "groq":
                raise
            print(f"      Groq indisponible ({e.__class__.__name__}) — essai Gemini...")

    if llm2 is None and llm2_type in ("auto", "gemini"):
        llm2 = _build_llm2_gemini()
        llm2_name = "gemini-1.5-flash (Google)"
        print(f"      LLM2 OK : {llm2_name}")

    if llm2 is None:
        raise RuntimeError("Aucun LLM2 disponible. Configurez GROQ_API_KEY ou GOOGLE_API_KEY.")

    # ── Annotation par les deux LLMs ──────────────────────────────────────────
    print(f"\n[2/4] Annotation des {len(records)} commentaires avec LLM1 ({llm1_name})...")
    annotations_llm1 = []
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        ann  = _annotate_one(llm1, text, llm1_name)
        annotations_llm1.append(ann)
        if (i+1) % 10 == 0:
            print(f"      {i+1}/{len(records)} done")
        time.sleep(0.3)  # respect rate limit

    print(f"\n[3/4] Annotation des {len(records)} commentaires avec LLM2 ({llm2_name})...")
    annotations_llm2 = []
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        ann  = _annotate_one(llm2, text, llm2_name)
        annotations_llm2.append(ann)
        if (i+1) % 10 == 0:
            print(f"      {i+1}/{len(records)} done")
        time.sleep(0.3)

    # ── Calcul des métriques ──────────────────────────────────────────────────
    print("\n[4/4] Calcul du kappa et des metriques...")

    sent1 = [a["sentiment_label"] for a in annotations_llm1]
    sent2 = [a["sentiment_label"] for a in annotations_llm2]

    noise1 = [a["noise_category"] for a in annotations_llm1]
    noise2 = [a["noise_category"] for a in annotations_llm2]

    qual1  = [float(a["quality_score"]) for a in annotations_llm1]
    qual2  = [float(a["quality_score"]) for a in annotations_llm2]

    kappa_sentiment = _cohen_kappa(sent1, sent2)
    kappa_noise     = _cohen_kappa(noise1, noise2)
    mae_quality     = _mae(qual1, qual2)
    pearson_quality = _pearson(qual1, qual2)

    # Accord exact sur sentiment
    sentiment_agreement = sum(a == b for a, b in zip(sent1, sent2)) / len(sent1)

    # ── Assemblage du rapport ─────────────────────────────────────────────────
    results = {
        "n_comments":   len(records),
        "llm1":         llm1_name,
        "llm2":         llm2_name,
        "methodology":  "cross_llm_annotation_level2",

        "kappa_sentiment": kappa_sentiment,
        "kappa_noise":     kappa_noise,
        "mae_quality_score": mae_quality,
        "pearson_quality":   pearson_quality,
        "sentiment_exact_agreement": round(sentiment_agreement, 4),

        "kappa_interpretation": (
            "excellent (>0.80)"  if kappa_sentiment > 0.80 else
            "bon (0.60-0.80)"    if kappa_sentiment > 0.60 else
            "modere (0.40-0.60)" if kappa_sentiment > 0.40 else
            "faible (<0.40)"
        ),
        "ac08_satisfied": kappa_sentiment >= 0.70,
        "ac08_threshold": 0.70,

        # Détail par commentaire
        "per_comment": [
            {
                "comment_id":       rec.get("comment_id", f"idx_{i}"),
                "text_preview":     rec.get("text", "")[:80],
                "llm1_sentiment":   annotations_llm1[i]["sentiment_label"],
                "llm2_sentiment":   annotations_llm2[i]["sentiment_label"],
                "agree_sentiment":  annotations_llm1[i]["sentiment_label"] == annotations_llm2[i]["sentiment_label"],
                "llm1_quality":     annotations_llm1[i]["quality_score"],
                "llm2_quality":     annotations_llm2[i]["quality_score"],
                "quality_delta":    abs(annotations_llm1[i]["quality_score"] - annotations_llm2[i]["quality_score"]),
            }
            for i, rec in enumerate(records)
        ],
    }

    # ── Affichage résumé ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  RESULTATS ANNOTATION CROISEE LLM1 x LLM2")
    print("="*55)
    print(f"  LLM1 : {llm1_name}")
    print(f"  LLM2 : {llm2_name}")
    print(f"  N    : {len(records)} commentaires")
    print("-"*55)
    print(f"  Kappa sentiment  : {kappa_sentiment:.4f}  [{results['kappa_interpretation']}]")
    print(f"  Kappa bruit      : {kappa_noise:.4f}")
    print(f"  MAE quality_score: {mae_quality:.4f}")
    print(f"  Pearson quality  : {pearson_quality:.4f}")
    print(f"  Accord sentiment : {sentiment_agreement*100:.1f}%")
    print("-"*55)
    print(f"  AC-08 (kappa >= 0.70) : {'OK' if results['ac08_satisfied'] else 'NON ATTEINT'}")
    print("="*55)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Rapport sauvegarde : {output_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation croisée LLM1 x LLM2")
    parser.add_argument("--gold",   default="data/gold_standard/gold_standard.jsonl")
    parser.add_argument("--output", default="evaluation/results/cross_annotation_kappa.json")
    parser.add_argument("--llm2",   choices=["auto", "groq", "gemini"], default="auto",
                        help="LLM2 à utiliser (défaut: auto — essaie Groq puis Gemini)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Limiter à N commentaires (test rapide)")
    args = parser.parse_args()

    cross_annotate(
        gold_path=args.gold,
        output_path=args.output,
        llm2_type=args.llm2,
        limit=args.limit,
    )
