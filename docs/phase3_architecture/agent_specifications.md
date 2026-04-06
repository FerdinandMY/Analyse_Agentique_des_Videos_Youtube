# Spécifications des Agents — Pipeline YouTube Quality Analyzer
## Phase 3 — Architecture
**Version** : 1.0 | **Date** : Avril 2025

---

## Vue d'ensemble du pipeline

```
Entrée utilisateur
    ↓
[csv_path, video_id, topic] → PipelineState
    ↓
A1 — Loader / Validator
    ↓
A2 — Préprocesseur
    ↓         ↓         ↓
A3 Sentiment  A4 Discours  A5 Bruit   ← exécution parallèle (fan-out LangGraph)
    ↓         ↓         ↓
    └─────────┴─────────┘
              ↓
        A6 — Synthétiseur
              ↓
        A7 — Topic Matcher
              ↓
        assemble_report
              ↓
           END → AnalyzeResponse
```

---

## PipelineState (TypedDict partagé)

```python
class PipelineState(TypedDict, total=False):
    # Entrées
    csv_path: str
    video_id: str
    topic: str
    # Intermédiaires
    raw_comments: list[dict[str, Any]]
    cleaned_comments: list[dict[str, Any]]
    sentiment: Optional[dict]
    discourse: Optional[dict]
    noise: Optional[dict]
    synthesis: Optional[dict]
    # Sorties
    score_global: float
    score_pertinence: float
    score_final: float
    topic_verdict: Optional[str]
    report: Optional[dict]
    # Accumulation d'erreurs
    errors: Annotated[list[str], operator.add]
```

---

## A1 — Loader / Validator

**Fichier** : [agents/a1_loader.py](../../agents/a1_loader.py)

### Rôle
Charger et valider le fichier CSV de commentaires pré-collectés. Point d'entrée unique du pipeline.

### Entrées (depuis PipelineState)

| Clé | Type | Obligatoire | Description |
|---|---|---|---|
| `csv_path` | string | Oui (sauf si `raw_comments` déjà peuplé) | Chemin vers le CSV |
| `raw_comments` | list | Non | Pré-peuplé pour les tests — bypasse le chargement CSV |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `raw_comments` | `list[dict]` | Liste de records `{text, video_id?, author_likes?, reply_count?}` |
| `errors` | `list[str]` | Erreurs cumulées (colonne manquante, fichier introuvable) |

### Contraintes

- Colonne **`text`** obligatoire — erreur si absente
- Colonnes optionnelles : `video_id`, `author_likes`, `reply_count`
- Lignes avec `text` vide ou NaN sont ignorées
- Si `raw_comments` déjà présent dans l'état → bypass (idempotent)

### Comportement d'erreur

Toute erreur est ajoutée à `errors` et la clé `raw_comments` n'est pas modifiée (reste `[]` ou l'ancienne valeur).

---

## A2 — Préprocesseur

**Fichier** : [agents/a2_preprocessor.py](../../agents/a2_preprocessor.py)

### Rôle
Nettoyer, normaliser et filtrer les commentaires bruts pour produire un corpus propre et dédupliqué.

### Entrées (depuis PipelineState)

| Clé | Type | Obligatoire |
|---|---|---|
| `raw_comments` | `list[dict]` | Oui |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `cleaned_comments` | `list[dict]` | Records avec champ `cleaned_text` ajouté |
| `errors` | `list[str]` | Erreurs éventuelles |

### Transformation `normalize_text()`

```
1. str.lower()
2. Collapse espaces multiples → espace simple
3. str.strip()
```

### Critères de filtrage

| Critère | Paramètre | Valeur |
|---|---|---|
| Longueur minimale | `MIN_CHARS` | 3 |
| Longueur maximale | `MAX_CHARS` | 2000 |
| Déduplication | Exact match sur `cleaned_text` | Doublons supprimés |
| Détection de langue | `langdetect` | Langue détectée → champ `lang` ajouté |

### Structure de sortie par commentaire

```python
{
    "text": "Texte brut original",
    "cleaned_text": "texte normalisé",
    "lang": "fr",          # détecté par langdetect
    "video_id": "abc123",  # copié si présent
    "author_likes": 5,     # copié si présent
    "reply_count": 2,      # copié si présent
}
```

---

## A3 — Sentiment Analyser

**Fichier** : [agents/a3_sentiment.py](../../agents/a3_sentiment.py)  
**Modèle recommandé** : Phi-3-mini-4k-instruct (ou OpenAI-compatible par défaut)  
**Prompt** : [prompts/sentiment_v1.txt](../../prompts/sentiment_v1.txt)

### Rôle
Classifier le sentiment global du corpus de commentaires et produire un `Score_Sentiment` [0-100].

### Entrées (depuis PipelineState)

| Clé | Type | Description |
|---|---|---|
| `cleaned_comments` | `list[dict]` | Corpus nettoyé (max 30 premiers commentaires utilisés) |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `sentiment` | dict | Dictionnaire de résultats (voir ci-dessous) |

### Structure de sortie `sentiment`

```python
{
    "sentiment_label": "positive",  # positive | neutral | negative
    "sentiment_score": 72.5,        # [0-100] (=parsed.sentiment_score × 100)
    "rationale": "..."              # explication en une phrase
}
```

### Schéma Pydantic (`_SentimentOutput`)

```python
class _SentimentOutput(BaseModel):
    sentiment_label: str    # positive | neutral | negative
    sentiment_score: float  # [0.0, 1.0] — multiplié par 100 → Score_Sentiment
    rationale: str
```

### Comportement de fallback

Si LLM indisponible ou erreur de parsing → `{sentiment_label: "neutral", sentiment_score: 50.0}`.

### Interprétation du Score_Sentiment

| Valeur | Interprétation |
|---|---|
| 0–33 | Sentiment très négatif |
| 34–66 | Sentiment neutre ou mixte |
| 67–100 | Sentiment positif |

---

## A4 — Discourse Analyser

**Fichier** : [agents/a4_discourse.py](../../agents/a4_discourse.py)  
**Modèle recommandé** : Qwen1.5-7B-Chat  
**Prompt** : [prompts/discourse_v1.txt](../../prompts/discourse_v1.txt)

### Rôle
Évaluer la profondeur intellectuelle du corpus sur 3 dimensions et identifier les commentaires de haute valeur pour A7.

### Entrées (depuis PipelineState)

| Clé | Type | Description |
|---|---|---|
| `cleaned_comments` | `list[dict]` | Corpus nettoyé (max 30 premiers commentaires, indexés `[0]...[N]`) |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `discourse` | dict | Dictionnaire de résultats (voir ci-dessous) |

### Structure de sortie `discourse`

```python
{
    "informativeness": 65.0,       # [0-100]
    "argumentation": 48.0,         # [0-100]
    "constructiveness": 72.0,      # [0-100]
    "discourse_score": 61.67,      # moyenne des 3 dimensions [0-100]
    "high_quality_indices": [2, 7, 15],  # indices 0-based des commentaires ≥ 0.7
    "rationale": "..."
}
```

### Schéma Pydantic (`_DiscourseOutput`)

```python
class _DiscourseOutput(BaseModel):
    informativeness: float          # [0.0, 1.0]
    argumentation: float            # [0.0, 1.0]
    constructiveness: float         # [0.0, 1.0]
    high_quality_indices: list[int] # indices 0-based, score moyen ≥ 0.7
    rationale: str
```

### Dimensions d'évaluation

| Dimension | Question posée | Poids dans discourse_score |
|---|---|---|
| **Informativité (D1)** | Le corpus ajoute-t-il des faits, connaissances ou contexte? | 1/3 |
| **Argumentation (D2)** | Les affirmations sont-elles soutenues par du raisonnement? | 1/3 |
| **Constructivité (D3)** | Le ton est-il constructif plutôt que purement réactif? | 1/3 |

### High-quality indices

Commentaires avec score moyen (D1+D2+D3)/3 ≥ 0.70. Passés à A7 pour le matching thématique. Si aucun commentaire ne dépasse le seuil, A7 utilise l'intégralité du corpus.

---

## A5 — Noise Detector

**Fichier** : [agents/a5_noise.py](../../agents/a5_noise.py)  
**Modèle recommandé** : Phi-3-mini-4k-instruct + heuristique SVM (référence Airlangga)  
**Prompt** : [prompts/noise_detection_v1.txt](../../prompts/noise_detection_v1.txt)

### Rôle
Estimer la proportion de bruit dans le corpus selon 5 catégories et produire un `Score_Bruit` [0-100].

### Entrées (depuis PipelineState)

| Clé | Type | Description |
|---|---|---|
| `cleaned_comments` | `list[dict]` | Corpus nettoyé (max 30 premiers commentaires) |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `noise` | dict | Dictionnaire de résultats (voir ci-dessous) |

### Structure de sortie `noise`

```python
{
    "spam_ratio": 12.0,       # % de commentaires spam [0-100]
    "offtopic_ratio": 8.0,    # % hors-sujet
    "reaction_ratio": 25.0,   # % réactions vides
    "toxic_ratio": 3.0,       # % toxiques
    "bot_ratio": 2.0,         # % bots suspectés
    "noise_ratio": 35.0,      # % total bruité [0-100]
    "noise_score": 65.0,      # = (1 - noise_ratio/100) × 100
    "rationale": "..."
}
```

### Formule Score_Bruit

```
Score_Bruit = (1 − noise_ratio) × 100
```

| noise_ratio | Score_Bruit | Interprétation |
|---|---|---|
| 0.0 (0%) | 100 | Corpus entièrement propre |
| 0.3 (30%) | 70 | Bruit modéré |
| 0.5 (50%) | 50 | Bruit élevé |
| 1.0 (100%) | 0 | Corpus entièrement bruité |

### Catégories de bruit

| Code | Catégorie | Signaux |
|---|---|---|
| N1 | `spam` | URL externe, call-to-action, contenu promotionnel |
| N2 | `offtopic` | Contenu sans rapport avec la vidéo |
| N3 | `reaction_vide` | < 5 mots, émojis seuls, onomatopées |
| N4 | `toxique` | Insultes, contenu haineux, menaces |
| N5 | `bot` | Pattern répétitif identique, timestamps réguliers |

---

## A6 — Synthétiseur

**Fichier** : [agents/a6_synthesizer.py](../../agents/a6_synthesizer.py)  
**Modèle recommandé** : Qwen1.5-7B-Chat  
**Prompt** : [prompts/synthesis_v1.txt](../../prompts/synthesis_v1.txt)

### Rôle
Agréger les scores de A3, A4, A5 selon la formule pondérée du PRD et générer un résumé en langage naturel.

### Entrées (depuis PipelineState)

| Clé | Type | Description |
|---|---|---|
| `sentiment` | dict | Sortie de A3 |
| `discourse` | dict | Sortie de A4 |
| `noise` | dict | Sortie de A5 |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `score_global` | float | Score_Global [0-100] |
| `synthesis` | dict | Détail complet + résumé LLM |

### Formule Score_Global (immuable — PRD §4.3)

```
Score_Global = 0.35 × Score_Sentiment
             + 0.40 × Score_Discours
             + 0.25 × Score_Bruit
```

| Poids | Dimension | Constante |
|---|---|---|
| `W_SENTIMENT = 0.35` | A3 → `sentiment["sentiment_score"]` | Immuable |
| `W_DISCOURSE = 0.40` | A4 → `discourse["discourse_score"]` | Immuable |
| `W_NOISE = 0.25` | A5 → `noise["noise_score"]` | Immuable |

### Quality Label

| Score_Global | Label |
|---|---|
| < 25 | `Faible` |
| 25–49 | `Moyen` |
| 50–74 | `Bon` |
| ≥ 75 | `Excellent` |

### Structure de sortie `synthesis`

```python
{
    "score_global": 62.5,
    "quality_label": "Bon",
    "sentiment_score": 72.5,
    "discourse_score": 61.67,
    "noise_score": 65.0,
    "summary": "Les commentaires reflètent un engagement positif..."
}
```

---

## A7 — Topic Matcher (v2.0)

**Fichier** : [agents/a7_topic_matcher.py](../../agents/a7_topic_matcher.py)  
**Modèle recommandé** : Qwen1.5-7B-Chat  
**Prompt** : [prompts/topic_matcher_v1.txt](../../prompts/topic_matcher_v1.txt)

### Rôle
Évaluer la pertinence thématique de la vidéo par rapport à l'intention de l'utilisateur et produire un `Score_Final` personnalisé.

### Entrées (depuis PipelineState)

| Clé | Type | Description |
|---|---|---|
| `topic` | string | Thématique fournie par l'utilisateur (ex: "machine learning") |
| `score_global` | float | Score_Global produit par A6 |
| `cleaned_comments` | `list[dict]` | Corpus nettoyé |
| `discourse` | dict | Contient `high_quality_indices` de A4 |

### Sorties (vers PipelineState)

| Clé | Type | Description |
|---|---|---|
| `score_pertinence` | float | Score_Pertinence [0-100] |
| `score_final` | float | Score_Final personnalisé [0-100] |
| `topic_verdict` | string | Explication de la pertinence en langage naturel |

### Formule Score_Final (immuable — PRD §4.3)

```
Score_Final = 0.60 × Score_Global + 0.40 × Score_Pertinence
```

### Sélection des commentaires pour le matching

```python
# Priorité aux commentaires de haute qualité identifiés par A4
if high_quality_indices:
    candidates = [cleaned_comments[i] for i in high_quality_indices]
else:
    candidates = cleaned_comments  # fallback sur tout le corpus

# Limite à 20 commentaires maximum
```

### Comportement selon le contexte

| Cas | Score_Pertinence | Score_Final |
|---|---|---|
| `topic` non fourni (vide) | 50.0 (neutre) | = Score_Global |
| LLM indisponible | 50.0 (neutre) | 0.60×Global + 0.40×50 |
| Erreur de parsing | 50.0 (neutre) | 0.60×Global + 0.40×50 |
| Cas normal | Valeur LLM × 100 | 0.60×Global + 0.40×Pertinence |

### Schéma Pydantic (`_TopicOutput`)

```python
class _TopicOutput(BaseModel):
    pertinence_score: float  # [0.0, 1.0] → multiplié par 100
    verdict: str             # Explication personnalisée
```

---

## Nœud assemble_report

**Fichier** : [graph.py](../../graph.py) — fonction `assemble_report()`

### Rôle
Consolider toutes les sorties du pipeline en un rapport structuré.

### Sortie

```python
{
    "report": {
        "video_id": str,
        "topic": str,
        "score_global": float,
        "score_pertinence": float,
        "score_final": float,
        "quality_label": str,
        "topic_verdict": str,
        "summary": str,
        "sentiment": dict,
        "discourse": dict,
        "noise": dict,
        "comment_count": int,
        "errors": list[str]
    }
}
```

---

## Tableau récapitulatif des agents

| Agent | Entrée principale | Sortie principale | Score produit | Modèle recommandé |
|---|---|---|---|---|
| **A1** Loader | csv_path | raw_comments | — | — |
| **A2** Préprocesseur | raw_comments | cleaned_comments | — | langdetect |
| **A3** Sentiment | cleaned_comments | sentiment | Score_Sentiment [0-100] | Phi-3-mini-4k |
| **A4** Discours | cleaned_comments | discourse | Score_Discours [0-100] | Qwen1.5-7B |
| **A5** Bruit | cleaned_comments | noise | Score_Bruit [0-100] | Phi-3-mini-4k |
| **A6** Synthétiseur | sentiment+discourse+noise | synthesis, score_global | Score_Global [0-100] | Qwen1.5-7B |
| **A7** Topic Matcher | score_global+discourse+topic | score_pertinence, score_final, topic_verdict | Score_Final [0-100] | Qwen1.5-7B |

---

## Contraintes transverses

| Contrainte | Valeur | Source |
|---|---|---|
| Seed reproductibilité | 42 | NFR-02 |
| VRAM max (Kaggle T4) | 14 Go | NFR-04 |
| Format modèles HF | float16 | NFR-04 |
| 1 modèle en mémoire à la fois | Obligatoire sur Kaggle | PR-11 |
| Max commentaires par agent | 30 (A3, A5), 30 indexés (A4), 20 (A7) | Contrainte VRAM |
| Checkpointing | `checkpoints/{agent}_{pipeline_id}.json` | NFR-03 |
| Poids Score_Global | w1=0.35, w2=0.40, w3=0.25 | PRD §4.3 — immuables |
| Poids Score_Final | w_global=0.60, w_pertinence=0.40 | PRD §4.3 — immuables |
