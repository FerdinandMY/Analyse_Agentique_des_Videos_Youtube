# Protocole d'évaluation — Phase 5 (PRD v3.0)

## Vue d'ensemble

Ce document décrit le protocole complet d'évaluation du pipeline YouTube Quality Analyzer v3.0,
couvrant les cinq hypothèses (H1–H5) et les procédures associées.

---

## 1. Gold Standard

### Composition
- **Taille** : 100 commentaires (échantillon stratifié)
- **Stratification** : croisement `noise_flag × length_bucket`
  - `length_bucket` : court (< 10 mots), moyen (10–30 mots), long (> 30 mots)
  - `noise_flag` : 0 (propre) / 1 (bruyant)
- **Source** : `data/raw/comments_raw.csv` (42 860 commentaires, 591 vidéos)

### Annotation
- **Annotateurs** : 2 annotateurs humains indépendants
- **Dimensions annotées** :
  - `quality_score` : note [1–5] (1 = très mauvais, 5 = excellent)
  - `sentiment_label` : `positive` / `neutral` / `negative`
  - `noise_flag` : 0 / 1
- **Accord inter-annotateurs** : Cohen's κ ≥ 0.70 requis (AC-08 PRD)
- **Procédure** : voir `docs/phase2_data/annotation_guide.md`

### Conversion
`gold_score = (quality_score − 1) / 4 × 100` → plage [0–100]

---

## 2. Hypothèses et critères de validation

| ID  | Hypothèse | Métrique | Seuil minimal | Objectif |
|-----|-----------|----------|--------------|---------|
| H1  | Score_Global corrélé au jugement humain | Pearson r ≥ 0.60 / Spearman ≥ 0.55 | 0.60 | 0.75 |
| H2  | Multi-agents > LLM unique | ΔPearson > 0.10 | 0.10 | — |
| H3  | Discours = dimension la plus discriminante | ΔPearson A4 > A3 > A5 | classement | — |
| H4  | CoT/ToT réduit l'erreur de classification | Gain F1 (B0→B3) ≥ +15 pts | +15 pts F1 | +15 pts |
| H5  | Score pertinence A7 améliore satisfaction | Wilcoxon p < 0.05 (avec vs sans A7) | p < 0.05 | — |

---

## 3. Conditions d'ablation (AC-09)

L'ablation v3.0 compare 4 conditions de raisonnement pour valider H4 :

| ID | Condition | Raisonnement | Tools |
|----|-----------|-------------|-------|
| B0 | Prompt direct | Aucun (réponse immédiate) | Non |
| B1 | CoT seul | Thought/Action/Result | Non |
| B2 | CoT + ToT | + exploration multi-branches | Non |
| B3 | CoT + ToT + Tools | Pipeline complet v3.0 | Oui |

**Métrique principale** : gain F1-macro sentiment B0 → B3 ≥ +15 pts (H4).

---

## 4. Scripts d'évaluation

| Script | Rôle | Sorties |
|--------|------|---------|
| `evaluation/compute_metrics.py` | H1 — Pearson + Spearman + F1 + Kappa + fallback rate | `metrics_report.json` |
| `evaluation/baseline_comparison.py` | H2 — multi-agents vs LLM unique | `baseline_comparison.json` |
| `evaluation/ablation_study.py` | H4 — B0/B1/B2/B3 (CoT/ToT/Tools) | `ablation_study.json` |
| `evaluation/error_analysis.py` | Outliers · sarcasme · hallucination_flags · fallback | `error_analysis.json` |

---

## 5. Procédure d'exécution

### Prérequis
```bash
# Générer le gold standard
python scripts/annotate_gold_standard.py \
    --csv data/raw/comments_raw.csv \
    --n 100 \
    --output data/gold_standard/gold_standard.jsonl

# Générer les prédictions (pipeline B3 complet)
python scripts/run_pipeline_predictions.py \
    --gold data/gold_standard/gold_standard.jsonl \
    --output data/predictions/pipeline_predictions.jsonl
```

### Métriques de base (H1)
```bash
python evaluation/compute_metrics.py \
    --gold  data/gold_standard/gold_standard.jsonl \
    --preds data/predictions/pipeline_predictions.jsonl \
    --output evaluation/results/metrics_report.json
```

### Comparaison baseline (H2)
```bash
python evaluation/baseline_comparison.py \
    --gold   data/gold_standard/gold_standard.jsonl \
    --output evaluation/results/baseline_comparison.json
```

### Étude d'ablation CoT/ToT/Tools (H4)
```bash
# Toutes les conditions
python evaluation/ablation_study.py \
    --gold   data/gold_standard/gold_standard.jsonl \
    --output evaluation/results/ablation_study.json

# Conditions spécifiques seulement
python evaluation/ablation_study.py \
    --gold data/gold_standard/gold_standard.jsonl \
    --conditions B0 B3
```

### Analyse d'erreurs
```bash
python evaluation/error_analysis.py \
    --gold      data/gold_standard/gold_standard.jsonl \
    --preds     data/predictions/pipeline_predictions.jsonl \
    --threshold 20 \
    --output    evaluation/results/error_analysis.json
```

---

## 6. Métriques calculées

### Numériques (Score_Global vs gold_score)
- **Pearson r** — corrélation linéaire (H1, seuil ≥ 0.60)
- **Spearman r** — corrélation de rang (H1, seuil ≥ 0.55)
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

### Classification sentiment (3 classes)
- **Accuracy**
- **F1-macro** (seuil cible > 0.85)
- **Cohen's κ** (seuil ≥ 0.70)

### Classification bruit (5 catégories)
- **F1-macro bruit** (seuil cible > 0.80)

### Robustesse LLM
- **Taux de fallback** — % d'appels nécessitant le fallback heuristique (cible < 5%)
- **hallucination_flags** — incohérences inter-champs détectées par le pipeline

### Ablation (H4)
- **Gain F1 B0→B3** — amélioration du CoT/ToT/Tools vs prompt direct (seuil ≥ +15 pts)
- **Gain F1 B1→B2** — apport spécifique du ToT
- **Gain Pearson B0→B3**

---

## 7. Gestion des fallbacks

Quand le LLM est indisponible :
- Toutes les conditions → fallback heuristique (composants textuel + longueur + URL ratio)
- Les résultats en mode fallback sont marqués `"condition": "..._heuristic_fallback"`
- Le taux de fallback est calculé et reporté dans `metrics_report.json`

NFR-09 : taux de fallback < 5% sur le dataset d'évaluation.

---

## 8. Résultats et interprétation

Les rapports JSON sont agrégés dans `notebooks/phase5_evaluation.ipynb`.
La table des résultats finale est dans `docs/phase5_evaluation/results_table.md`.
