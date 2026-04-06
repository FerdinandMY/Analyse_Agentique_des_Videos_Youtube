# Tableau des résultats — Phase 5 (PRD v3.0)

> Ce fichier est mis à jour après chaque run d'évaluation.
> Valeurs cibles PRD v3.0 §7 — à compléter avec les résultats réels.

---

## H1 — Corrélation (Score_Global vs gold_score)

| Métrique | Seuil minimal | Objectif | Résultat observé | Statut |
|----------|--------------|---------|-----------------|--------|
| Pearson r | ≥ 0.60 | > 0.75 | — | En attente |
| Spearman r | ≥ 0.55 | > 0.70 | — | En attente |
| MAE | — | — | — | — |
| RMSE | — | — | — | — |
| n évalués | 100 | — | — | — |

---

## H2 — Pipeline multi-agents vs LLM unique (baseline)

| Méthode | Pearson r | MAE |
|---------|-----------|-----|
| LLM unique (baseline B3 single prompt) | — | — |
| Pipeline multi-agents (7 agents) | — | — |
| **ΔPearson r** | **seuil > 0.10** | — |

**Statut** : En attente — *à compléter après run `baseline_comparison.py`*

---

## H3 — Contribution des agents analytiques (A4 > A3 > A5)

| Agent désactivé | Pearson r sans Ax | ΔPearson (contribution) | Rang |
|----------------|------------------|------------------------|------|
| Pipeline complet (B3) | — | référence | — |
| Sans A3 (sentiment, w=0.35) | — | — | — |
| Sans A4 (discourse, w=0.40) | — | — | — |
| Sans A5 (noise, w=0.25) | — | — | — |

**Attendu** : A4 > A3 > A5 (pondérations PRD §3.6)

**Statut** : Non testé dans v3.0 — H3 remplacée par l'ablation CoT/ToT (H4) en AC-09

---

## H4 — Ablation CoT / ToT / Tools (AC-09)

| Condition | Pearson r | MAE | F1-sentiment | Fallback rate |
|-----------|-----------|-----|-------------|--------------|
| B0 — Prompt direct | — | — | — | — |
| B1 — CoT seul | — | — | — | — |
| B2 — CoT + ToT | — | — | — | — |
| B3 — CoT + ToT + Tools | — | — | — | — |

### Gains par étape

| Transition | ΔPearson r | ΔF1 (pts) |
|-----------|-----------|----------|
| B0 → B1 (ajout CoT) | — | — |
| B1 → B2 (ajout ToT) | — | — |
| B2 → B3 (ajout Tools) | — | — |
| **B0 → B3 (gain total)** | **—** | **seuil ≥ +15 pts** |

**H4 satisfaite si** : Gain F1 B0→B3 ≥ +15 pts

**Statut** : En attente — *à compléter après run `ablation_study.py`*

---

## Métriques sentiment (classification 3 classes)

| Métrique | Seuil minimal | Objectif | Résultat |
|----------|--------------|---------|---------|
| Accuracy | — | — | — |
| F1-macro | > 0.75 | > 0.85 | — |
| Cohen's κ | ≥ 0.70 | — | — |

---

## Métriques bruit (5 catégories — A5)

| Métrique | Seuil minimal | Objectif | Résultat |
|----------|--------------|---------|---------|
| F1-macro bruit | > 0.70 | > 0.80 | — |
| Précision | — | — | — |
| Rappel | — | — | — |

---

## Robustesse LLM (NFR-09)

| Indicateur | Seuil | Résultat |
|------------|-------|---------|
| Taux de fallback global | < 5% | — |
| Taux de fallback A3 | < 5% | — |
| Taux de fallback A4 | < 5% | — |
| Taux de fallback A5 | < 5% | — |
| Taux de fallback A7 | < 5% | — |
| hallucination_flags détectés | — | — |

---

## Analyse des erreurs

| Indicateur | Valeur |
|------------|--------|
| Outliers score (|err| > 20) | — |
| Taux d'outliers | — |
| Erreurs sentiment | — |
| Sarcasme suspect | — |
| FP bruit (propre classé bruit) | — |
| FN bruit (bruit non détecté) | — |

---

## Configuration d'évaluation

| Paramètre | Valeur |
|-----------|--------|
| Gold standard | `data/gold_standard/gold_standard.jsonl` |
| Taille du gold | 100 commentaires |
| Stratification | noise_flag × length_bucket |
| Seuil outlier | 20 points |
| Modèle LLM | claude-sonnet-4-6 |
| Températures | A3/A4/A5 = 0.1 · A7 = 0.3 |
| Self-Consistency A7 | 3 runs + vote majoritaire |
| Date d'évaluation | — |
