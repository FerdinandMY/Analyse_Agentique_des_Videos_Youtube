# Hypothèses Scientifiques — Phase 1
## YouTube Quality Analyzer
**Version** : 1.0 | **Date** : Avril 2025

---

## Contexte

Les 4 hypothèses suivantes découlent de la revue des 5 articles (MetaGPT, AutoGen, Musleh et al., Airlangga, Thapa et al.) et des 7 gaps identifiés dans `gap_analysis.md`. Elles structurent le protocole d'évaluation de la Phase 5 et répondent aux 4 questions de recherche du PRD (RQ1→RQ4).

---

## H1 — Prédictibilité de la qualité à partir des seuls commentaires
> **RQ1 — Les commentaires YouTube contiennent suffisamment d'information sémantique pour prédire la qualité d'une vidéo avec une corrélation de Pearson ≥ 0.60 avec les annotations humaines.**

### Fondement littéraire
- **Musleh et al. (A3)** : kappa inter-annotateurs = 0.818 sur 4 212 commentaires — preuve que les commentaires portent un signal de qualité cohérent et reproductible.
- **Airlangga (A4)** : F1 = 95.32 % pour la détection de spam — confirme que les patterns discriminants sont extractibles automatiquement.
- **Thapa et al. (A5)** : les LLM surpassent SVM/Naïve Bayes pour la compréhension sémantique fine, notamment les nuances de discours et le sarcasme.

### Opérationnalisation
| Élément | Détail |
|---|---|
| Variable prédite | Score_Global [0-100] produit par A6 |
| Variable critère | Score humain annoté dans le gold standard (Likert 1-5 → [0-100]) |
| Métrique | Coefficient de corrélation de Pearson r |
| Seuil de validation | r ≥ 0.60, p < 0.05 |
| Ensemble de test | 30 % du gold standard (split stratifié, seed=42) |
| Baseline | Score brut YouTube (likes / vues) et LLM unique (H2) |

### Résultat attendu
r ≥ 0.60 confirme H1 et valide la prémisse centrale du projet : les commentaires sont un proxy fiable de la qualité vidéo, compensant la suppression du compteur de dislikes (novembre 2021).

---

## H2 — Supériorité de l'architecture multi-agents sur un LLM unique
> **RQ2 — Une architecture à 7 agents LLM spécialisés (LangGraph) produit un Score_Global plus corrélé à la qualité humaine qu'un LLM unique analysant tous les aspects simultanément.**

### Fondement littéraire
- **MetaGPT (A1)** : la spécialisation des rôles et les SOPs réduisent les hallucinations en cascade — le taux de réussite augmente significativement par rapport à un LLM unique (benchmark SoftwareDev).
- **AutoGen (A2)** : la collaboration multi-agents dépasse GPT-4 seul sur MATH (69.48 % vs performances inférieures) et réduit le code de 430 à 100 lignes (OptiGuide).
- **Thapa et al. (A5)** : le système HAD (agents LLM hétérogènes) surpasse un modèle unique pour l'analyse de sentiments financiers ; le framework COLA avec rôles distincts améliore la détection de positions complexes.

### Opérationnalisation
| Condition | Description |
|---|---|
| **A — Notre système** | Pipeline complet A1→A7 (7 agents spécialisés) |
| **B — Baseline LLM unique** | Un seul appel LLM : prompt demandant sentiment + discours + bruit + score global simultanément |
| Métriques | Pearson r vs gold standard, F1-score sur 4 classes (Faible/Moyen/Bon/Excellent) |
| Protocole | Ablation study — retrait successif de chaque agent (AC-09 PRD) |

### Résultat attendu
Pearson r(A) > Pearson r(B) avec Δr > 0.10, confirmant que la décomposition en agents spécialisés réduit les hallucinations et améliore la précision.

---

## H3 — Primauté de la dimension Discours dans la prédiction de qualité
> **RQ3 — La profondeur du discours (w2 = 0.40 dans Score_Global) est la dimension la plus discriminante pour distinguer les vidéos de haute qualité des contenus superficiels.**

### Fondement littéraire
- **Thapa et al. (A5)** : segmente l'analyse CSS en 3 niveaux — *énoncé* (sentiment), *discours* (interactions), *document* (thèmes). Le niveau discours capture des informations structurellement différentes du sentiment.
- **Musleh et al. (A3)** : les commentaires longs (> 100 mots) tendent à être négatifs **mais informatifs** — la longueur et la profondeur argumentative sont des signaux de qualité indépendants du sentiment.
- **AutoGen (A2)** : l'agent "Critique" qui vérifie les analyses des autres agents est l'analogie directe de A4 — la vérification de la constructivité argumentative est la valeur ajoutée clé dans les systèmes multi-agents.

### Opérationnalisation
| Test | Description |
|---|---|
| Ablation A4 retiré | Pipeline sans agent Discours : Score_Global = 0.35×S + 0.25×N (renormalisé) |
| Ablation A3 retiré | Pipeline sans agent Sentiment |
| Ablation A5 retiré | Pipeline sans agent Bruit |
| Métrique | ΔPearson r par agent retiré vs pipeline complet |

**Prédiction** : ΔPearson(sans A4) > ΔPearson(sans A3) > ΔPearson(sans A5)

### Résultat attendu
La dégradation la plus forte lors du retrait de A4 (Discours) confirme H3 et justifie scientifiquement le poids w2 = 0.40 devant le comité académique.

---

## H4 — Amélioration de l'utilité perçue par le Topic Matcher (A7)
> **RQ4 — Le Score_Final personnalisé (A7 Topic Matcher) augmente significativement la satisfaction utilisateur par rapport au Score_Global seul.**

### Fondement littéraire
- **AutoGen (A2)** : le "Human-in-the-loop" pour guider l'analyse quand le contexte est ambigu → A7 est l'équivalent automatisé : il adapte le score au contexte de recherche de l'utilisateur.
- **Thapa et al. (A5)** : les systèmes RAG (Retrieval-Augmented Generation) permettent aux agents de répondre à des questions spécifiques sur un corpus — même logique que la comparaison thématique de A7.
- **MetaGPT (A1)** : tâche SoftwareDev ID-8 — extraction d'informations de réseaux sociaux pour un objectif spécifique — analogue direct à A7 évaluant la pertinence d'un contenu pour une intention utilisateur.

### Opérationnalisation
| Condition | Description |
|---|---|
| **A — Sans A7** | Score_Final = Score_Global (pas de personnalisation thématique) |
| **B — Avec A7** | Score_Final = 0.60 × Score_Global + 0.40 × Score_Pertinence |
| Métrique | Score de satisfaction (Likert 1-5) + pertinence perçue (Likert 1-5) |
| Protocole | 10 vidéos × 3 thématiques différentes, ≥ 5 évaluateurs humains |
| Test statistique | Test de Wilcoxon (non-paramétrique, petits échantillons) |

### Résultat attendu
Satisfaction moyenne(B) > Satisfaction moyenne(A), p < 0.05 — confirme que la personnalisation thématique est un apport réel à l'utilisateur au-delà du score générique.

---

## Tableau récapitulatif

| Hypothèse | Question | Métrique clé | Seuil | Phase |
|---|---|---|---|---|
| **H1** — Prédictibilité | RQ1 | Pearson r (Score_Global vs gold standard) | r ≥ 0.60, p < 0.05 | P5 |
| **H2** — Multi-agents | RQ2 | ΔPearson (multi-agents vs LLM unique) | Δr > 0.10 | P5 — Ablation |
| **H3** — Primauté Discours | RQ3 | ΔPearson par agent retiré | ΔA4 > ΔA3 > ΔA5 | P5 — Ablation |
| **H4** — Topic Matcher | RQ4 | Satisfaction Likert (Wilcoxon) | p < 0.05 | P5 — Tests utilisateurs |

---

## Note méthodologique

- Toutes les expériences utilisent **seed = 42** (NFR-02 — reproductibilité).
- H1 et H2 sont testées sur le **gold standard** (Phase 2, AC-08 : kappa > 0.7, ≥ 2 annotateurs).
- H3 et H4 requièrent l'**analyse d'ablation** (Phase 5, AC-09).
- Le split train/test est **70 % / 30 %** avec stratification sur la variable `quality_label`.
- Si H1 n'est pas satisfaite (r < 0.60), optimiser les poids par régression linéaire sur 70 % du gold standard (Plan de mitigation — Risque R-04 du PRD).
