# Protocole d'évaluation — Phase 5

## Vue d'ensemble

Ce document décrit le protocole complet d'évaluation du pipeline YouTube Quality Analyzer,
couvrant les quatre hypothèses testées (H1–H4) et les procédures associées.

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

| ID  | Hypothèse | Métrique | Seuil |
|-----|-----------|----------|-------|
| H1  | Pearson r(Score_Global, gold_score) ≥ 0.60 | Pearson r | ≥ 0.60 |
| H2  | ΔPearson(multi-agents − LLM unique) > 0.10 | ΔPearson r | > 0.10 |
| H3  | Contribution A4 > A3 > A5 (ablation) | ΔPearson r par agent | classement |
| H4  | Wilcoxon p < 0.05 (Score_Final avec vs sans topic) | p-value | < 0.05 |

---

## 3. Scripts d'évaluation

| Script | Rôle | Sorties |
|--------|------|---------|
| `evaluation/compute_metrics.py` | H1 — métriques de base | `metrics_report.json` |
| `evaluation/baseline_comparison.py` | H2 — comparaison baseline | `baseline_comparison.json` |
| `evaluation/ablation_study.py` | H3 — étude d'ablation | `ablation_study.json` |
| `evaluation/error_analysis.py` | Analyse des erreurs systématiques | `error_analysis.json` |

---

## 4. Procédure d'exécution

### Prérequis
```bash
# Générer le gold standard (requiert annotations humaines ou LLM)
python scripts/annotate_gold_standard.py \
    --csv data/raw/comments_raw.csv \
    --n 100 \
    --output data/gold_standard/gold_standard.jsonl

# Générer les prédictions du pipeline
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

### Étude d'ablation (H3)
```bash
python evaluation/ablation_study.py \
    --gold   data/gold_standard/gold_standard.jsonl \
    --output evaluation/results/ablation_study.json
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

## 5. Métriques calculées

### Numériques (Score_Global vs gold_score)
- **Pearson r** — corrélation linéaire (H1)
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

### Classification (sentiment_label)
- **Accuracy** — taux de classification correct
- **F1-macro** — F1 moyenné sur les 3 classes
- **Cohen's κ** — accord inter-systèmes

### Bruit (noise_flag)
- **Précision / Rappel / F1** pour le détecteur binaire

---

## 6. Gestion des fallbacks

Quand le LLM est indisponible :
- `baseline_comparison.py` → `_fallback_baseline()` (heuristique textuelle)
- `ablation_study.py` → `_heuristic_without()` (valeurs neutres par agent)

Les résultats en mode fallback sont marqués `"method": "heuristic_fallback"`.

---

## 7. Résultats et interprétation

Les rapports JSON sont agrégés dans `notebooks/phase5_evaluation.ipynb`.
La table des résultats finale est dans `docs/phase5_evaluation/results_table.md`.
