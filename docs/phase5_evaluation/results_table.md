# Tableau des résultats — Phase 5

> Ce fichier est mis à jour après chaque run d'évaluation.
> Valeurs cibles PRD v2.0 — à compléter avec les résultats réels.

---

## H1 — Corrélation Pearson (Score_Global vs gold_score)

| Métrique | Valeur cible | Résultat observé | Statut |
|----------|-------------|-----------------|--------|
| Pearson r | ≥ 0.60 | — | En attente |
| MAE | — | — | — |
| RMSE | — | — | — |
| n évalués | 100 | — | — |

**Interprétation** : *à compléter après run `compute_metrics.py`*

---

## H2 — Comparaison baseline (LLM unique vs pipeline multi-agents)

| Méthode | Pearson r | MAE |
|---------|-----------|-----|
| LLM unique (baseline) | — | — |
| Pipeline multi-agents | — | — |
| **ΔPearson r** | **—** | — |

**Seuil H2** : ΔPearson > 0.10

**Statut** : En attente — *à compléter après run `baseline_comparison.py`*

---

## H3 — Étude d'ablation (contribution par agent)

| Agent désactivé | Pearson r sans Ax | ΔPearson (contribution) | Rang |
|----------------|------------------|------------------------|------|
| Pipeline complet | — | référence | — |
| Sans A3 (sentiment) | — | — | — |
| Sans A4 (discourse) | — | — | — |
| Sans A5 (noise) | — | — | — |

**H3 attendue** : contribution A4 > A3 > A5 (pondérations PRD : D=0.40, S=0.35, N=0.25)

**Statut** : En attente — *à compléter après run `ablation_study.py`*

---

## Métriques sentiment

| Métrique | Valeur observée |
|----------|----------------|
| Accuracy | — |
| F1-macro | — |
| Cohen's κ | — |

---

## Métriques bruit (A5)

| Métrique | Valeur observée |
|----------|----------------|
| Précision | — |
| Rappel | — |
| F1 | — |

---

## Analyse des erreurs

| Indicateur | Valeur |
|------------|--------|
| Outliers (|err| > 20) | — |
| Taux d'outliers | — |
| Erreurs sentiment | — |
| Erreurs sarcasme suspect | — |

---

## Configuration d'évaluation

| Paramètre | Valeur |
|-----------|--------|
| Gold standard | `data/gold_standard/gold_standard.jsonl` |
| Taille du gold | 100 commentaires |
| Stratification | noise_flag × length_bucket |
| Seuil outlier | 20 points |
| Date d'évaluation | — |
| Modèle LLM | claude-sonnet-4-6 |
