# Analyse des erreurs — Phase 5

## Objectif

Identifier les patterns d'erreur systématiques du pipeline pour orienter les
améliorations futures et comprendre les limites du système actuel.

---

## 1. Catégories d'erreurs

### 1.1 Outliers de score (|gold − pred| > 20)

**Définition** : commentaires pour lesquels le Score_Global prédit dévie de plus
de 20 points par rapport au gold_score humain.

**Types observés** :

| Type | Description | Cause probable |
|------|-------------|----------------|
| Sur-estimation | pred > gold + 20 | Commentaires courts mal classifiés comme positifs |
| Sous-estimation | pred < gold − 20 | Commentaires longs et riches mal évalués par A4 |

**Patterns linguistiques corrélés** :
- **Sur-estimation** : commentaires très courts (< 5 mots), présence d'emojis positifs
- **Sous-estimation** : URLs longues, répétitions de caractères

### 1.2 Erreurs de classification sentiment (A3)

**Sarcasme et ironie** : principal point de défaillance.

Indicateurs heuristiques de sarcasme :
- Formules élogieuses + ponctuation excessive (`!!`, `...`)
- Mots ironiques : "bien sûr", "évidemment", "tellement", "génial"
- Longueur courte (< 20 mots) avec marqueurs d'ironie

**Distribution des erreurs par classe** (cible) :
- `positive → negative` : erreurs sur sarcasme
- `negative → neutral` : sous-détection sur commentaires courts négatifs
- `neutral → positive` : sur-détection sur commentaires polis mais vides

**Matrice de confusion** : *voir `evaluation/results/error_analysis.json`*

### 1.3 Faux positifs / faux négatifs du détecteur de bruit (A5)

**Faux positifs** (prédit bruit, mais propre) :
- Commentaires en langue étrangère (espagnol, arabe) détectés comme bruit
- Commentaires avec URLs légitimes (sources académiques)
- Texte en majuscules POUR EMPHASE

**Faux négatifs** (prédit propre, mais bruit) :
- Spam sophistiqué sans URLs (phrases génériques et positives)
- Promotions déguisées en commentaires authentiques
- Bots imitant le langage naturel

---

## 2. Analyse par segment

### Vidéos longues vs courtes

Les vidéos avec peu de commentaires (< 10) tendent à avoir des scores moins
stables car le pipeline manque de signal statistique.

### Commentaires en langue non-française

Le pipeline est optimisé pour le français. Les commentaires en d'autres langues
peuvent être :
- Mal classifiés par A3 (sentiment)
- Surévalués par A4 (discourse) si la syntaxe est différente
- Faussement marqués comme bruit par A5

---

## 3. Recommandations d'amélioration

### Court terme

1. **A3 — Sarcasme** : ajouter un prompt spécifique demandant de détecter l'ironie
   avant de classifier le sentiment
2. **A5 — Langue** : détecter la langue avant d'appliquer les patterns de bruit
3. **A4 — Corpus réduit** : ajouter un avertissement quand n < 10 commentaires

### Moyen terme

1. **Fine-tuning** : collecter des exemples d'erreurs annotés pour fine-tuner A3
2. **Seuils adaptatifs** : ajuster les seuils de bruit par langue
3. **Ensemble** : combiner plusieurs modèles de sentiment pour réduire les erreurs

---

## 4. Limites connues du système

| Limite | Impact | Workaround |
|--------|--------|------------|
| LLM indisponible → fallback heuristique | Pearson r réduit | Vérifier la disponibilité avant run |
| Score corpus-level (1 score pour N commentaires) | Pearson r artificiellement limité | Évaluer par vidéo plutôt que par commentaire |
| Gold standard heuristique si LLM absent | Biais dans H1 | Annoter manuellement au moins 30 exemples |
| Sarcasme non détectable sans contexte | Erreurs A3 systématiques sur ironie | Post-processing avec règles |

---

## 5. Fichiers de référence

| Fichier | Contenu |
|---------|---------|
| `evaluation/error_analysis.py` | Script d'analyse des erreurs |
| `evaluation/results/error_analysis.json` | Résultats détaillés (après run) |
| `notebooks/phase5_evaluation.ipynb` | Visualisations des erreurs |
