# Data Card — Corpus YouTube Comments
## YouTube Quality Analyzer — Phase 2
**Version** : 1.0 | **Date** : Avril 2025

---

## 1. Résumé

| Attribut | Valeur |
|---|---|
| **Nom du dataset** | YouTube Comment Quality Corpus (YCQC) |
| **Version** | 0.1.0 (gold standard), TBD (corpus complet) |
| **Taille cible** | 100 commentaires annotés (gold standard) + corpus brut variable |
| **Format** | CSV (corpus brut) + JSON (gold standard annoté) |
| **Langue(s)** | Principalement français et anglais (détection automatique via `langdetect`) |
| **Source** | YouTube API v3 ou export manuel — commentaires pré-collectés |
| **Collecte** | Agent A1 (Loader/Validator) + script de collecte externe |
| **Annotation** | Semi-automatique : LLM (2 passes) + validation humaine |
| **Licence** | Usage académique interne uniquement — voir §7 |

---

## 2. Description du corpus

### 2.1 Structure du CSV brut (`data/raw/comments.csv`)

| Colonne | Type | Obligatoire | Description |
|---|---|---|---|
| `text` | string | **Oui** | Texte brut du commentaire |
| `video_id` | string | Non | Identifiant YouTube de la vidéo (format: 11 chars alphanumériques) |
| `author_likes` | integer | Non | Nombre de likes reçus par l'auteur du commentaire |
| `reply_count` | integer | Non | Nombre de réponses au commentaire |

### 2.2 Structure du gold standard (`data/gold_standard/gold_standard.jsonl`)

Voir `data/gold_standard/annotation_schema.json` pour le schéma complet.

Champs principaux :
- `comment_id` — identifiant unique (format: `yt_{video_id}_{index:03d}`)
- `text` — texte brut
- `sentiment_label` — `positive` / `neutral` / `negative`
- `quality_score` — entier [1-5]
- `noise_category` — `spam` / `offtopic` / `reaction_vide` / `toxique` / `bot` / `ok`
- `annotator` — `human_1`, `human_2`, `llm_pass_1`, `llm_pass_2`
- `confidence` — flottant [0.0-1.0]

---

## 3. Statistiques du corpus

> **Note** : Les statistiques ci-dessous seront mises à jour après fourniture du CSV brut (`data/raw/comments.csv`). Les valeurs sont des estimations cibles.

### 3.1 Taille et distribution cibles

| Métrique | Valeur cible |
|---|---|
| Commentaires bruts | ≥ 500 |
| Commentaires après filtrage A2 | ≥ 300 (longueur 3-2000 chars, dédup) |
| Gold standard | 100 (sélection stratifiée) |
| Vidéos représentées | ≥ 5 |
| Langues détectées | ≥ 2 (fr, en principalement) |

### 3.2 Distribution cible du gold standard (stratification)

| Strate | Critère | Proportion cible |
|---|---|---|
| Bruit court | noise_flag=True, length_bucket=court | 15% |
| Bruit long | noise_flag=True, length_bucket=long | 10% |
| Normal court | noise_flag=False, length_bucket=court | 25% |
| Normal long | noise_flag=False, length_bucket=long | 50% |

> Allocation proportionnelle avec minimum 5 exemples par strate.

### 3.3 Distributions cibles des labels

| Label | Proportion cible |
|---|---|
| sentiment: positive | 40-60% |
| sentiment: neutral | 20-35% |
| sentiment: negative | 10-25% |
| quality_score ≥ 4 | 30-40% |
| noise_category = ok | 60-75% |

---

## 4. Processus de collecte

### 4.1 Source des données

Les commentaires sont collectés via **YouTube API v3** (quota: 10 000 unités/jour gratuit) ou exportés manuellement depuis YouTube Studio. L'extension Chrome (hors périmètre de ce dépôt) peut également fournir des CSV pré-collectés.

### 4.2 Pipeline de collecte

```
YouTube API v3
    ↓
Export CSV (colonnes: text, video_id, author_likes, reply_count)
    ↓ [fourni manuellement par l'utilisateur]
data/raw/comments.csv
    ↓
Agent A1 (Loader/Validator) — validation colonnes requises
    ↓
Agent A2 (Préprocesseur) — nettoyage, filtrage, déduplication
    ↓
notebooks/phase2_eda.ipynb — EDA + heuristique bruit
    ↓
scripts/annotate_gold_standard.py — échantillonnage stratifié + annotation LLM
    ↓
data/gold_standard/gold_standard.jsonl
```

### 4.3 Critères d'exclusion (Agent A2)

- Longueur < 3 caractères
- Longueur > 2 000 caractères
- Doublon exact (texte identique)
- Langue non détectable (`langdetect` confidence < 0.5)

---

## 5. Qualité et biais

### 5.1 Biais potentiels

| Biais | Description | Mitigation |
|---|---|---|
| **Biais de sélection** | Les commentaires collectés dépendent des vidéos choisies — pas représentatif de YouTube global | Diversifier les vidéos et les thématiques |
| **Biais de langue** | Prédominance probable du français/anglais | Documenter la distribution de langues dans l'EDA |
| **Biais de confirmation** | L'annotation LLM peut reproduire les biais du modèle de langage | 2 passes LLM indépendantes + validation humaine |
| **Biais de longueur** | Les commentaires courts sont souvent plus bruités — surestimation possible du bruit | Stratification par longueur (length_bucket) |
| **Biais temporel** | Les commentaires récents et anciens peuvent avoir des distributions différentes | Documenter les dates de collecte si disponibles |
| **Suppression des dislikes** | Absence du signal dislike depuis nov. 2021 — motivation centrale du projet | Le corpus lui-même constitue le signal alternatif |

### 5.2 Limites connues

- Pas d'accès aux métadonnées de la vidéo (titre, description, vues) par défaut
- `author_likes` et `reply_count` optionnels — absents dans certains exports
- Pas de lien direct vers les threads de discussion (contexte de réponse perdu)
- Les commentaires épinglés ou mis en avant par YouTube peuvent créer un biais de visibilité

### 5.3 Accord inter-annotateurs

| Round | κ mesuré | Statut |
|---|---|---|
| Initial (LLM pass 1 vs 2) | TBD | En attente du CSV |
| Validation humaine | TBD | En attente du CSV |

Seuil requis : κ > 0.70 (AC-08 PRD). Si non atteint, révision du guide d'annotation et 2e round.

---

## 6. Utilisation et accès

### 6.1 Utilisation prévue

- Évaluation du pipeline de qualité YouTube (Phase 5)
- Validation des hypothèses H1, H2, H3, H4
- Ablation study (retrait successif des agents A3, A4, A5)
- Tests unitaires et d'intégration des agents

### 6.2 Utilisation non prévue

- Production en temps réel sur des corpus > 10 000 commentaires (contrainte VRAM Kaggle T4)
- Classification de commentaires dans d'autres langues non représentées
- Détection de fausses informations ou fact-checking
- Profilage d'auteurs de commentaires

### 6.3 Accès

```
data/
├── raw/
│   └── comments.csv          ← CSV brut fourni par l'utilisateur
└── gold_standard/
    ├── annotation_schema.json ← Schéma JSON de référence
    └── gold_standard.jsonl    ← Annotations (généré par scripts/annotate_gold_standard.py)
```

---

## 7. Considérations éthiques et légales

### 7.1 Données YouTube

Les commentaires YouTube sont publics mais soumis aux [Conditions d'utilisation YouTube](https://www.youtube.com/t/terms) et à la politique de l'API YouTube Data v3. Usage autorisé pour la **recherche académique non commerciale**.

### 7.2 Vie privée

- Les `comment_id` et `video_id` sont des identifiants publics YouTube — pas de donnée personnelle directe
- Les pseudonymes d'auteurs ne sont pas collectés dans ce pipeline
- Le gold standard ne stocke pas d'informations d'identification personnelle (PII)

### 7.3 Contenu sensible

Le corpus peut contenir des commentaires toxiques ou offensants. Ces commentaires sont inclus dans le gold standard avec `noise_category = toxique` pour permettre l'évaluation de la détection de bruit (Agent A5). L'exposition à ces contenus lors de l'annotation est inévitable mais minimisée par l'annotation LLM automatique.

---

## 8. Maintenance et versionnage

| Événement | Action |
|---|---|
| Gold standard validé (κ > 0.70) | Tag `v0.1.0` sur le dépôt git |
| Ajout de nouvelles vidéos au corpus | Incrémenter la version mineure (v0.2.0) |
| Révision du schéma d'annotation | Incrémenter la version majeure (v1.0.0) |
| Découverte d'erreurs d'annotation | Patch version (v0.1.1) + re-calcul du kappa |

**Script de mise à jour** : `scripts/annotate_gold_standard.py --check_kappa`
