# Guide d'Annotation — Gold Standard
## YouTube Quality Analyzer — Phase 2
**Version** : 1.0 | **Date** : Avril 2025 | **Auteur** : Équipe YouTube Quality Analyzer

---

## 1. Objectif

Ce guide définit les consignes données aux annotateurs humains (et LLM simulant l'annotation) pour labelliser le **gold standard** de 100 commentaires YouTube. Le gold standard sert de référence pour l'évaluation des hypothèses H1 et H2 (Phase 5).

**Critère de qualité** : accord inter-annotateurs κ (Cohen's Kappa) > 0.70 (AC-08 PRD).

---

## 2. Tâche d'annotation

Pour chaque commentaire, l'annotateur renseigne **3 champs obligatoires** :

| Champ | Type | Description |
|---|---|---|
| `sentiment_label` | enum | Sentiment global du commentaire |
| `quality_score` | int [1-5] | Contribution à la qualité de l'évaluation de la vidéo |
| `noise_category` | enum | Catégorie de bruit principale |

Et **2 champs facultatifs** :

| Champ | Type | Description |
|---|---|---|
| `confidence` | float [0-1] | Niveau de certitude de l'annotateur |
| `notes` | string | Observations libres (ambiguïtés, cas limites) |

---

## 3. Sentiment Label

### Définitions

| Label | Définition | Exemples typiques |
|---|---|---|
| `positive` | Le commentaire exprime une appréciation favorable envers la vidéo, le créateur ou le sujet | "Excellent video!", "J'ai appris énormément, merci!" |
| `neutral` | Le commentaire est factuel, interrogatif ou ambigu sans valeur affective dominante | "À quelle minute parle-t-il de X?", "OK" |
| `negative` | Le commentaire exprime une critique, une déception ou une opposition | "Cette vidéo est nulle", "Complètement faux" |

### Règles de décision

1. **Sarcasme** : Classifier selon le sentiment *réel* exprimé, non le sens littéral.
   - "Bien sûr, c'est le meilleur conseil que j'aie jamais reçu 🙄" → **negative**
2. **Mix positif/négatif** : Choisir le sentiment **dominant** (≥ 60% du contenu).
   - "Super vidéo mais l'audio est horrible" → **positive** (si l'éloge domine) ou **negative** (si la critique domine)
3. **Émojis seuls** : Interpréter selon le contexte culturel standard.
   - "❤️🔥" → **positive** | "😴💤" → **negative** | "🤔" → **neutral**
4. **Langue autre que le français** : Annoter normalement si la langue est identifiable.
5. **Cas incertain** : Utiliser `neutral` et noter dans `notes`.

---

## 4. Quality Score

### Échelle Likert [1-5]

| Score | Label | Définition | Exemple |
|---|---|---|---|
| **1** | Bruit pur | Aucun apport informatif : spam, caractères aléatoires, réaction vide, lien seul | "PREMIER!!!", "aaaa", "Check this out: bit.ly/xxx" |
| **2** | Bruit léger | Très peu d'information, réaction superficielle sans substance | "Lol", "😂😂😂", "+1" |
| **3** | Neutre | Information minimale, présence sans apport significatif, question basique | "C'est bien", "Merci pour la vidéo", "À quelle heure?" |
| **4** | Informatif | Apport réel : question précise, nuance, référence externe, expérience personnelle pertinente | "J'ai essayé cette méthode et ça m'a pris 2h, voici pourquoi…" |
| **5** | Très informatif | Contribution de haute valeur : analyse, contre-argument, ressources, expertise vérifiable | "En fait, l'étude citée a été réfutée par [ref]. Voici pourquoi..." |

### Règles de décision

1. **Longueur ≠ qualité** : Un commentaire long mais répétitif reste à score 2-3.
2. **Lien URL** : Diminue le score d'1 point si le lien n'est pas accompagné d'explication.
3. **Question pertinente** : Une question précise et contextuelle vaut 4.
4. **Expertise** : Mentionner une source vérifiable ou un diplôme/expérience → score 4-5.
5. **Hors-sujet** : Score 1-2 même si le commentaire est bien écrit.

---

## 5. Noise Category

### Catégories

| Catégorie | Description | Signaux | Exemples |
|---|---|---|---|
| `spam` | Contenu promotionnel, lien non sollicité, phishing | URL externe, call-to-action, répétition | "Abonne-toi à ma chaîne!", "Gagne 500€/jour ici: …" |
| `offtopic` | Hors-sujet par rapport à la vidéo | Référence à un autre sujet non lié | "Qui regarde ça en 2025?", discussion politique sur vidéo culinaire |
| `reaction_vide` | Réaction émotionnelle sans substance | < 5 mots, émojis seuls, onomatopées | "Lol", "🔥🔥🔥", "INCROYABLE!!!" |
| `toxique` | Contenu agressif, haineux, insultant | Insultes, discriminations, menaces | "@pseudo t'es nul", contenu haineux |
| `bot` | Généré automatiquement | Pattern répétitif exact, timestamp régulier | Même texte sur plusieurs vidéos |
| `ok` | Pas de bruit significatif | Commentaire normal | Tout commentaire informatif/neutre sans les signaux ci-dessus |

### Règles de décision

1. **Priorité** : Si plusieurs catégories s'appliquent → choisir la plus grave : `toxique` > `spam` > `bot` > `offtopic` > `reaction_vide` > `ok`.
2. **Doute spam/offtopic** : Vérifier si une URL est présente → `spam`.
3. **Commentaire "ok" avec sentiment négatif** : Possible (critique sincère = `ok`).
4. **Bot** : Nécessite une preuve de répétition — ne pas classifier `bot` sur un seul commentaire sans contexte.

---

## 6. Procédure d'annotation

### Étapes

```
1. Lire le commentaire intégralement (avec le contexte video_id si disponible)
2. Déterminer noise_category en premier (élimine les bruits évidents)
3. Déterminer sentiment_label
4. Attribuer quality_score en cohérence avec noise_category et sentiment
5. Renseigner confidence [0.0–1.0] :
   - 1.0 = totalement certain
   - 0.7 = légères ambiguïtés
   - 0.5 = cas limite (noter dans notes)
   - < 0.5 = consulter un second annotateur
6. Renseigner notes si confidence < 0.7 ou cas particulier
```

### Cohérence obligatoire

| Si noise_category = | Alors quality_score ≤ |
|---|---|
| `spam` ou `bot` | 1 |
| `reaction_vide` | 2 |
| `offtopic` | 2 |
| `toxique` | 2 |
| `ok` | 3 à 5 |

**Incohérence interdite** : `noise_category = ok` et `quality_score = 1` → impossible.

---

## 7. Accord inter-annotateurs

### Protocole

- **Minimum** : 2 annotateurs indépendants par commentaire (human_1, human_2 ou llm_pass_1, llm_pass_2)
- **Calcul** : Cohen's Kappa sur `sentiment_label` (3 classes) et catégorisation de `quality_score` (Faible=1-2, Moyen=3, Bon=4-5)
- **Seuil de validation** : κ > 0.70 (AC-08 PRD)

### En cas de désaccord

| Type de désaccord | Action |
|---|---|
| κ 0.60–0.70 | Révision du guide + 2e round d'annotation sur les cas litigieux |
| κ < 0.60 | Réunion de calibrage obligatoire + réécriture des critères ambigus |
| Désaccord ponctuel (< 5 commentaires) | Annotation par consensus (discussion et accord mutuel) |

---

## 8. Exemples annotés

### Exemple 1 — Score 5, Positive, OK

```
Texte : "L'approche présentée ici est analogue à ce que décrit Goodfellow dans Deep Learning 
(2016, ch.9). Cependant, le batch normalization était absent dans leur implémentation, 
ce qui explique probablement l'instabilité d'entraînement décrite à 12:34."

sentiment_label: positive
quality_score: 5
noise_category: ok
confidence: 0.95
notes: Référence académique vérifiable, critique constructive et précise.
```

### Exemple 2 — Score 1, Neutral, Spam

```
Texte : "🔥 Gagnez jusqu'à 500€/jour depuis chez vous! Sans expérience! 
Cliquez ici: bit.ly/argent-facile 🔥"

sentiment_label: neutral
quality_score: 1
noise_category: spam
confidence: 1.0
```

### Exemple 3 — Score 3, Positive, OK

```
Texte : "Merci pour cette vidéo, c'est très bien expliqué!"

sentiment_label: positive
quality_score: 3
noise_category: ok
confidence: 0.85
notes: Appréciation générique sans substance informative.
```

### Exemple 4 — Score 2, Negative, Reaction Vide

```
Texte : "Bof 😐"

sentiment_label: negative
quality_score: 2
noise_category: reaction_vide
confidence: 0.9
```

### Exemple 5 — Score 4, Negative, OK

```
Texte : "La conclusion à 18:45 contredit ce qui a été dit à 5:20. 
Si X implique Y, alors Z ne peut pas être vrai simultanément — 
il faudrait revoir la démonstration."

sentiment_label: negative
quality_score: 4
noise_category: ok
confidence: 0.9
notes: Critique argumentée et précise. Négatif mais constructif.
```

---

## 9. Format de sortie

Chaque annotation est exportée en JSON selon le schéma `data/gold_standard/annotation_schema.json` :

```json
{
  "comment_id": "yt_abc123_001",
  "video_id": "abc123",
  "text": "...",
  "sentiment_label": "positive",
  "quality_score": 4,
  "noise_category": "ok",
  "annotator": "human_1",
  "confidence": 0.9,
  "notes": ""
}
```

---

## 10. Checklist avant soumission

- [ ] Tous les champs obligatoires renseignés (comment_id, text, sentiment_label, quality_score, annotator)
- [ ] Cohérence noise_category / quality_score vérifiée
- [ ] confidence ≥ 0.7 ou notes renseignées si < 0.7
- [ ] Aucun doublon de comment_id
- [ ] Fichier valide contre `annotation_schema.json` (`python scripts/annotate_gold_standard.py --check_kappa`)
