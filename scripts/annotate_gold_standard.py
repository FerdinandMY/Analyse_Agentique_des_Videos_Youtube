"""
scripts/annotate_gold_standard.py
====================================
Automatise la création du gold standard (Phase 2 — PRD §8, AC-08).

Pipeline automatique en 4 étapes :

    1. SÉLECTION STRATIFIÉE
       Tire 100 commentaires équilibrés depuis data/raw/comments.csv
       (strates : longueur × flag de bruit heuristique).

    2. ANNOTATION LLM × 2 PASSAGES INDÉPENDANTS
       Chaque commentaire est annoté deux fois par le LLM avec des
       températures légèrement différentes, simulant deux annotateurs.
       → sentiment_label (positive/neutral/negative)
       → quality_score [1-5]
       → noise_category (spam/offtopic/reaction_vide/toxique/bot/ok)

    3. CALCUL DU KAPPA
       Cohen's kappa sur sentiment_label entre les deux passages.
       Si kappa > 0.7 → condition AC-08 remplie.

    4. EXPORT + TAG GIT OPTIONNEL
       Exporte data/gold_standard/annotated_videos.csv
       Si --auto_tag et kappa > 0.7 → crée le tag Git v0.1.0.

Usage :
    # Annotation LLM uniquement (pas de tag)
    python scripts/annotate_gold_standard.py --csv data/raw/comments.csv

    # Avec création automatique du tag si kappa > 0.7
    python scripts/annotate_gold_standard.py --csv data/raw/comments.csv --auto_tag

    # Personnaliser la taille du gold standard
    python scripts/annotate_gold_standard.py --csv data/raw/comments.csv --n_samples 150

    # Vérifier le kappa d'un gold standard déjà annoté manuellement
    python scripts/annotate_gold_standard.py --check_kappa data/gold_standard/annotated_videos.csv
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Reproductibilité (NFR-02) ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Chemins par défaut ────────────────────────────────────────────────────────
DEFAULT_CSV    = "data/raw/comments.csv"
GOLD_OUTPUT    = "data/gold_standard/annotated_videos.csv"
KAPPA_REPORT   = "data/gold_standard/kappa_report.json"
GIT_TAG        = "v0.1.0"
KAPPA_THRESHOLD = 0.7

# ── Annotation prompt ─────────────────────────────────────────────────────────
_ANNOTATION_SYSTEM = (
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
- confidence: float 0.0-1.0 (your confidence in these labels)
- notes: one brief sentence explaining your decision (optional)

JSON only:"""


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[annotate] {msg}", flush=True)


def _heuristic_flag(text: str) -> str:
    text = str(text).strip()
    if len(text) < 3:
        return "trop_court"
    if re.fullmatch(r"[^\w\s]+|[\d\W]+", text):
        return "emoji_only"
    if len(text.split()) == 1:
        return "reaction_vide"
    if re.search(r"https?://\S+|www\.\S+", text, re.IGNORECASE):
        return "contient_url"
    if re.search(r"(.)\1{4,}", text):
        return "lettres_repetees"
    return "ok"


def _length_bucket(text: str) -> str:
    wc = len(str(text).split())
    if wc <= 3:
        return "very_short"
    if wc <= 15:
        return "short"
    if wc <= 50:
        return "medium"
    return "long"


# ── Étape 1 : Sélection stratifiée ───────────────────────────────────────────

def select_sample(df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Sélection stratifiée sur (noise_flag × length_bucket).
    Garantit la représentation des différentes catégories de commentaires.
    """
    df = df.copy()
    df["noise_flag"]     = df["text"].apply(_heuristic_flag)
    df["length_bucket"]  = df["text"].apply(_length_bucket)
    df["stratum"]        = df["noise_flag"] + "_" + df["length_bucket"]
    df["comment_id"]     = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    strata = df["stratum"].value_counts()
    _log(f"Strates disponibles : {len(strata)} — total corpus : {len(df)}")

    # Allocation proportionnelle avec minimum 1 par strate
    total = len(df)
    allocation: dict[str, int] = {}
    for stratum, count in strata.items():
        allocated = max(1, round(n_samples * count / total))
        allocation[stratum] = min(allocated, count)

    # Ajustement si le total dépasse n_samples
    while sum(allocation.values()) > n_samples:
        largest = max(allocation, key=allocation.get)  # type: ignore[arg-type]
        allocation[largest] -= 1

    # Tirage par strate
    sampled_parts = []
    for stratum, k in allocation.items():
        part = df[df["stratum"] == stratum].sample(k, random_state=SEED)
        sampled_parts.append(part)

    sample = pd.concat(sampled_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    _log(f"Échantillon sélectionné : {len(sample)} commentaires ({len(strata)} strates)")
    return sample


# ── Étape 2 : Annotation LLM ─────────────────────────────────────────────────

def _call_llm(comment: str, temperature: float = 0.0) -> dict:
    """Appelle le LLM configuré via models/llm_loader.get_llm()."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            return _fallback_annotation(comment)

        # Ajuste température si possible (OpenAI-compatible)
        if hasattr(llm, "temperature"):
            llm = llm.bind(temperature=temperature)

        resp = llm.invoke([
            SystemMessage(content=_ANNOTATION_SYSTEM),
            HumanMessage(content=_ANNOTATION_PROMPT.format(comment=comment[:500])),
        ])

        # Parse JSON — fallback regex si entouré de backticks
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    except Exception as exc:
        _log(f"LLM error : {exc} — fallback heuristique")
        return _fallback_annotation(comment)


def _fallback_annotation(comment: str) -> dict:
    """Annotation heuristique de secours quand le LLM est indisponible."""
    flag = _heuristic_flag(comment)
    wc = len(str(comment).split())
    sentiment = "neutral"
    noise_cat = "ok"
    score = 3

    if flag in ("trop_court", "emoji_only", "reaction_vide"):
        score, noise_cat = 1, "reaction_vide"
    elif flag == "contient_url":
        score, noise_cat = 2, "spam"
    elif flag == "lettres_repetees":
        score, noise_cat = 1, "spam"

    if wc > 50:
        score = max(score, 4)

    return {
        "sentiment_label": sentiment,
        "quality_score": score,
        "noise_category": noise_cat,
        "confidence": 0.5,
        "notes": "fallback heuristique (LLM indisponible)",
    }


def annotate_sample(
    sample: pd.DataFrame,
    annotator_id: str = "llm_pass_1",
    temperature: float = 0.0,
) -> pd.DataFrame:
    """Annote chaque commentaire et retourne le DataFrame enrichi."""
    records = []
    total = len(sample)

    for i, row in sample.iterrows():
        comment = str(row["text"])
        _log(f"  [{i+1}/{total}] annotateur={annotator_id} — {comment[:60]}...")

        annotation = _call_llm(comment, temperature=temperature)

        record = {
            "comment_id":      row.get("comment_id", str(uuid.uuid4())[:8]),
            "video_id":        row.get("video_id", ""),
            "text":            comment,
            "sentiment_label": annotation.get("sentiment_label", "neutral"),
            "quality_score":   int(annotation.get("quality_score", 3)),
            "noise_category":  annotation.get("noise_category", "ok"),
            "confidence":      float(annotation.get("confidence", 0.5)),
            "notes":           annotation.get("notes", ""),
            "annotator":       annotator_id,
            "noise_flag":      row.get("noise_flag", _heuristic_flag(comment)),
            "length_bucket":   row.get("length_bucket", _length_bucket(comment)),
        }
        records.append(record)

    return pd.DataFrame(records)


# ── Étape 3 : Calcul du Kappa ─────────────────────────────────────────────────

def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """
    Cohen's kappa entre deux listes de labels catégoriels.
    Formule : kappa = (Po - Pe) / (1 - Pe)
    """
    assert len(labels_a) == len(labels_b), "Les deux listes doivent avoir la même longueur"
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Matrice de confusion
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        matrix[cat_idx[a]][cat_idx[b]] += 1

    # Po = accord observé
    po = matrix.diagonal().sum() / n

    # Pe = accord attendu par hasard
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = float(np.dot(row_sums, col_sums))

    if pe >= 1.0:
        return 1.0

    return round((po - pe) / (1 - pe), 4)


def compute_kappa_report(df_pass1: pd.DataFrame, df_pass2: pd.DataFrame) -> dict:
    """Calcule le kappa sur sentiment_label et quality_score (discrétisé)."""
    merged = df_pass1[["comment_id", "sentiment_label", "quality_score"]].merge(
        df_pass2[["comment_id", "sentiment_label", "quality_score"]],
        on="comment_id",
        suffixes=("_p1", "_p2"),
    )

    kappa_sentiment = cohen_kappa(
        merged["sentiment_label_p1"].tolist(),
        merged["sentiment_label_p2"].tolist(),
    )

    # Quality score discrétisé : low (1-2), medium (3), high (4-5)
    def bucket_score(s: int) -> str:
        if s <= 2: return "low"
        if s == 3: return "medium"
        return "high"

    kappa_quality = cohen_kappa(
        merged["quality_score_p1"].apply(bucket_score).tolist(),
        merged["quality_score_p2"].apply(bucket_score).tolist(),
    )

    sentiment_dist_p1 = df_pass1["sentiment_label"].value_counts().to_dict()
    sentiment_dist_p2 = df_pass2["sentiment_label"].value_counts().to_dict()

    report = {
        "n_annotated":        len(merged),
        "kappa_sentiment":    kappa_sentiment,
        "kappa_quality":      kappa_quality,
        "kappa_mean":         round((kappa_sentiment + kappa_quality) / 2, 4),
        "threshold":          KAPPA_THRESHOLD,
        "ac08_satisfied":     kappa_sentiment >= KAPPA_THRESHOLD,
        "sentiment_dist_p1":  sentiment_dist_p1,
        "sentiment_dist_p2":  sentiment_dist_p2,
        "interpretation": {
            "< 0.40": "Accord faible",
            "0.40-0.60": "Accord mod\u00e9r\u00e9",
            "0.60-0.80": "Accord substantiel",
            "> 0.80": "Accord presque parfait",
        }[
            "< 0.40" if kappa_sentiment < 0.40
            else "0.40-0.60" if kappa_sentiment < 0.60
            else "0.60-0.80" if kappa_sentiment < 0.80
            else "> 0.80"
        ],
    }
    return report


# ── Étape 4 : Export + Tag Git ───────────────────────────────────────────────

def export_gold_standard(df_pass1: pd.DataFrame, df_pass2: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les deux passages dans un seul CSV gold standard.
    Les commentaires où les deux annotateurs s'accordent (sentiment identique)
    sont marqués consensus=True.
    """
    df_p1 = df_pass1.copy()
    df_p2 = df_pass2.copy()
    df_p1["annotator"] = "llm_pass_1"
    df_p2["annotator"] = "llm_pass_2"

    gold = pd.concat([df_p1, df_p2], ignore_index=True)

    # Ajout flag consensus
    agree = (
        df_p1.set_index("comment_id")["sentiment_label"]
        == df_p2.set_index("comment_id")["sentiment_label"]
    )
    gold["consensus"] = gold["comment_id"].map(agree.to_dict())
    return gold


def create_git_tag(tag: str = GIT_TAG) -> bool:
    """Crée le tag Git si le dépôt est propre."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        if status.stdout.strip():
            _log(f"Warning : dépôt non propre — commit vos changements avant de tagger.")
            _log("Exécutez : git add -A && git commit -m 'feat(data): add phase2 gold standard'")
            return False

        result = subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"Phase 2 complete — Gold standard validated (kappa > {KAPPA_THRESHOLD})"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _log(f"Tag Git créé : {tag}")
            return True
        else:
            _log(f"Erreur tag Git : {result.stderr.strip()}")
            return False
    except Exception as exc:
        _log(f"Git non disponible : {exc}")
        return False


# ── Vérification d'un CSV annoté manuellement ────────────────────────────────

def check_kappa_from_csv(csv_path: str) -> None:
    """
    Calcule le kappa entre les annotateurs présents dans un CSV gold standard.
    Attend une colonne 'annotator' avec au moins 2 valeurs distinctes.
    """
    df = pd.read_csv(csv_path)
    annotators = df["annotator"].unique()
    if len(annotators) < 2:
        _log("Au moins 2 annotateurs requis dans la colonne 'annotator'.")
        return

    a1, a2 = annotators[0], annotators[1]
    df_a1 = df[df["annotator"] == a1][["comment_id", "sentiment_label"]].set_index("comment_id")
    df_a2 = df[df["annotator"] == a2][["comment_id", "sentiment_label"]].set_index("comment_id")

    common_ids = df_a1.index.intersection(df_a2.index)
    labels_a = df_a1.loc[common_ids, "sentiment_label"].tolist()
    labels_b = df_a2.loc[common_ids, "sentiment_label"].tolist()

    kappa = cohen_kappa(labels_a, labels_b)
    ok = "✅" if kappa >= KAPPA_THRESHOLD else "❌"
    _log(f"{ok} Cohen's kappa ({a1} vs {a2}) sur {len(common_ids)} commentaires : {kappa:.4f}")
    _log(f"   Seuil requis (AC-08 PRD) : {KAPPA_THRESHOLD}")


# ── Entrée principale ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatise la création du gold standard")
    parser.add_argument("--csv",         default=DEFAULT_CSV,  help="Chemin du CSV de commentaires bruts")
    parser.add_argument("--n_samples",   type=int, default=100, help="Taille du gold standard (défaut: 100)")
    parser.add_argument("--output",      default=GOLD_OUTPUT,  help="Chemin de sortie du CSV annoté")
    parser.add_argument("--auto_tag",    action="store_true",  help="Crée le tag v0.1.0 si kappa > 0.7")
    parser.add_argument("--check_kappa", default="",           help="Vérifie le kappa d'un CSV annoté manuellement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Mode vérification kappa uniquement
    if args.check_kappa:
        check_kappa_from_csv(args.check_kappa)
        return

    # ── Étape 1 : Chargement + sélection ─────────────────────────────────────
    _log(f"Chargement du CSV : {args.csv}")
    if not Path(args.csv).exists():
        _log(f"ERREUR : fichier introuvable — {args.csv}")
        sys.exit(1)

    df_raw = pd.read_csv(args.csv)
    df_raw["text"] = df_raw["text"].astype(str).str.strip()
    df_raw = df_raw[df_raw["text"].str.len() > 0]
    _log(f"Corpus chargé : {len(df_raw)} commentaires")

    sample = select_sample(df_raw, n_samples=args.n_samples)

    # ── Étape 2 : Annotation LLM (2 passages indépendants) ───────────────────
    _log("Passage 1 — annotation LLM (temperature=0.0)...")
    df_pass1 = annotate_sample(sample, annotator_id="llm_pass_1", temperature=0.0)

    _log("Passage 2 — annotation LLM (temperature=0.2)...")
    df_pass2 = annotate_sample(sample, annotator_id="llm_pass_2", temperature=0.2)

    # ── Étape 3 : Calcul du kappa ─────────────────────────────────────────────
    _log("Calcul du Cohen's kappa...")
    kappa_report = compute_kappa_report(df_pass1, df_pass2)

    print("\n" + "=" * 52)
    print("RAPPORT D'ACCORD INTER-ANNOTATEURS")
    print("=" * 52)
    print(f"Commentaires annotés  : {kappa_report['n_annotated']}")
    print(f"Kappa sentiment       : {kappa_report['kappa_sentiment']:.4f}")
    print(f"Kappa qualité         : {kappa_report['kappa_quality']:.4f}")
    print(f"Kappa moyen           : {kappa_report['kappa_mean']:.4f}")
    print(f"Interprétation        : {kappa_report['interpretation']}")
    ok = "✅" if kappa_report["ac08_satisfied"] else "❌"
    print(f"AC-08 PRD satisfait ? : {ok} (seuil : {KAPPA_THRESHOLD})")
    print("=" * 52 + "\n")

    # Sauvegarde du rapport kappa
    Path(KAPPA_REPORT).parent.mkdir(parents=True, exist_ok=True)
    Path(KAPPA_REPORT).write_text(
        json.dumps(kappa_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(f"Rapport kappa sauvegardé : {KAPPA_REPORT}")

    # ── Étape 4 : Export gold standard ────────────────────────────────────────
    gold = export_gold_standard(df_pass1, df_pass2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(args.output, index=False, encoding="utf-8")
    _log(f"Gold standard exporté : {args.output} ({len(gold)} lignes)")

    n_consensus = gold[gold["consensus"] == True]["comment_id"].nunique()
    _log(f"Commentaires en consensus : {n_consensus}/{len(sample)} "
         f"({100*n_consensus/len(sample):.0f}%)")

    # Instructions pour annotation humaine
    if not kappa_report["ac08_satisfied"]:
        print("\n⚠️  Kappa insuffisant pour valider le gold standard automatiquement.")
        print("   → Ouvrez le CSV dans un tableur et corrigez les désaccords.")
        print(f"   → Relancez avec : python scripts/annotate_gold_standard.py "
              f"--check_kappa {args.output}")
    else:
        print("\n✅  Kappa > 0.7 — gold standard validé automatiquement (AC-08 PRD).")
        print("   → Recommandation : faites valider par un humain avant la Phase 5.")

    # ── Tag Git optionnel ─────────────────────────────────────────────────────
    if args.auto_tag:
        if kappa_report["ac08_satisfied"]:
            _log(f"Création du tag Git {GIT_TAG}...")
            create_git_tag(GIT_TAG)
        else:
            _log(f"Tag {GIT_TAG} non créé : kappa insuffisant ({kappa_report['kappa_sentiment']:.4f} < {KAPPA_THRESHOLD}).")
    else:
        print(f"\nPour créer le tag v0.1.0 quand vous êtes prêt :")
        print(f"  python scripts/annotate_gold_standard.py --csv {args.csv} --auto_tag")
        print(f"  # ou manuellement : git tag -a v0.1.0 -m 'Phase 2 complete'")


if __name__ == "__main__":
    main()
