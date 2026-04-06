# ADR-001 — Architecture Multi-Agents LangGraph
## YouTube Quality Analyzer — Phase 3
**Statut** : Acceptée | **Date** : Avril 2025 | **Décideurs** : Équipe YouTube Quality Analyzer

---

## Contexte

Le projet YouTube Quality Analyzer doit produire un score de qualité composite à partir de commentaires YouTube pré-collectés. La suppression du compteur de dislikes (novembre 2021) a créé un vide dans les signaux de qualité vidéo disponibles publiquement.

Trois approches architecturales ont été considérées pour analyser ces commentaires et produire un score [0-100] :

1. **LLM unique (monolithique)** — Un seul prompt demandant simultanément le sentiment, le discours, le bruit et le score global.
2. **Pipeline séquentiel classique** — Enchaînement linéaire d'agents sans parallélisme.
3. **Graphe multi-agents parallèle (LangGraph)** — Architecture à 7 agents spécialisés avec exécution parallèle des analyses.

---

## Décision

**Choix retenu : Architecture à 7 agents LangGraph avec fan-out parallèle.**

```
A1 (Loader) → A2 (Préprocesseur) → ┌─ A3 (Sentiment) ─┐
                                    ├─ A4 (Discours)  ──┼→ A6 (Synthétiseur) → A7 (Topic Matcher) → END
                                    └─ A5 (Bruit)    ──┘
```

---

## Justification

### 1. Spécialisation des agents (MetaGPT, A1)

MetaGPT (Hong et al.) démontre que l'encodage de SOPs (Standard Operating Procedures) dans des agents spécialisés réduit les hallucinations en cascade. Chaque agent de notre pipeline ne traite qu'une dimension analytique :

- **A3** : Sentiment uniquement (pas de discours, pas de bruit)
- **A4** : Profondeur du discours uniquement
- **A5** : Détection de bruit uniquement

Cette séparation permet à chaque agent d'utiliser un prompt optimisé pour sa tâche, sans interférence des autres dimensions.

### 2. Supériorité empirique du multi-agents (AutoGen, A2)

AutoGen (Wu et al.) montre que la collaboration multi-agents dépasse GPT-4 seul sur le benchmark MATH (69.48 % vs performances inférieures) et réduit la complexité du code (430 → 100 lignes dans OptiGuide). Notre Hypothèse H2 teste formellement cette supériorité : ΔPearson(multi-agents vs LLM unique) > 0.10.

### 3. Parallélisme naturel des dimensions d'analyse

Les 3 dimensions (sentiment, discours, bruit) sont **indépendantes** : le score de sentiment d'un commentaire ne dépend pas de son score de discours. LangGraph permet leur exécution parallèle (fan-out), réduisant la latence d'un facteur ~3 par rapport à un pipeline séquentiel.

### 4. Gestion d'état partagé et checkpointing

LangGraph fournit un `TypedDict` partagé (`PipelineState`) permettant à chaque agent de lire les sorties des agents précédents sans couplage direct. Le checkpointing intégré (`MemorySaver` / `SqliteSaver`) garantit la reprise en cas d'interruption (contrainte Kaggle : sessions limitées à 9h, NFR-03).

### 5. Personnalisation thématique (A7)

L'agent A7 (Topic Matcher) est rendu possible par l'architecture multi-agents : il peut consommer les `high_quality_indices` produits par A4 (commentaires à score ≥ 0.7) pour construire un score de pertinence thématique personnalisé. Ce couplage sélectif n'est pas possible avec un LLM unique.

---

## Alternatives rejetées

### LLM unique monolithique

**Raisons du rejet :**
- Hallucinations croisées : la détection de bruit interfère avec l'analyse de sentiment dans un prompt unique
- Impossibilité de tester l'hypothèse H2 (ablation study par agent impossible)
- Prompt trop long → dégradation de la cohérence sur de grands corpus (> 50 commentaires)
- Pas de parallélisme possible

**Avantage** : Latence minimale, un seul appel LLM. Utilisé comme **baseline** pour H2.

### Pipeline séquentiel simple (LangChain LCEL)

**Raisons du rejet :**
- Latence triée par 3 (A3, A4, A5 en série) vs parallèle
- Pas de gestion native de l'état partagé entre agents
- Pas de checkpointing intégré pour la reprise
- Moins extensible pour ajouter A7 ou de futurs agents

**Avantage** : Plus simple à déboguer. Conservé comme pattern de fallback pour les environnements sans GPU.

---

## Conséquences

### Positives

- **Maintenabilité** : Chaque agent est un module Python indépendant (`agents/a{N}_{nom}.py`)
- **Testabilité** : Tests unitaires isolés par agent (PR-29 : couverture 80%)
- **Extensibilité** : Ajout d'un agent A8 sans modifier les agents existants
- **Reproductibilité** : Seed 42 + checkpoints JSON pour la reprise exacte (NFR-02, NFR-03)
- **Performance** : Fan-out parallèle A3/A4/A5 réduit la latence de ~66%

### Négatives / Contraintes

- **Complexité** : 7 agents vs 1 LLM — plus de fichiers, plus de prompts à maintenir
- **Coût GPU** : Un modèle par agent idéalement (PR-13), mais limité à 1 modèle en mémoire à la fois sur Kaggle T4 (PR-11) — les agents doivent partager le même LLM ou délester le modèle entre agents
- **Latence accrue** : Malgré le parallélisme, l'overhead LangGraph est non nul (~0.5-1s par agent)
- **Dépendance LangGraph** : Version ≥ 0.2.0 requise — migrations potentielles si l'API change

---

## Paramètres de configuration

| Paramètre | Valeur | Source |
|---|---|---|
| Framework d'orchestration | LangGraph ≥ 0.2.0 | PRD §3.2 |
| Checkpointer dev | MemorySaver | NFR-03 |
| Checkpointer prod | SqliteSaver | NFR-03 |
| Modèle par défaut | OpenAI-compatible (LLM_BACKEND=openai) | models/llm_loader.py |
| Modèle Kaggle | HuggingFace float16 (LLM_BACKEND=huggingface) | NFR-04 |
| VRAM max | 14 Go (T4 16 Go − 2 Go système) | NFR-04 |
| Seed | 42 | NFR-02 |
| Poids Score_Global | w1=0.35, w2=0.40, w3=0.25 | PRD §4.3 |
| Poids Score_Final | w_global=0.60, w_pertinence=0.40 | PRD §4.3 |

---

## Statut de validation

| Critère | Statut |
|---|---|
| Graphe LangGraph implémenté (`graph.py`) | ✅ Fait |
| 7 agents implémentés (`agents/a{1-7}_*.py`) | ✅ Fait |
| Fan-out parallèle A3/A4/A5 | ✅ Fait |
| Checkpointing MemorySaver | ✅ Fait |
| Tests unitaires agents (PR-29 : 80%) | ❌ En attente |
| Ablation study (H2, H3) | ❌ Phase 5 |
| Benchmark vs LLM unique (H2) | ❌ Phase 5 |

---

## Références

- Hong, S. et al. (2023). *MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework*. arXiv:2308.00352
- Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- PRD YouTube Quality Analyzer v2.0, §3.2 Architecture
