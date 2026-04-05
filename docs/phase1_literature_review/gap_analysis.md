# Gap Analysis — Revue de littérature
## YouTube Quality Analyzer — Phase 1
**Version** : 1.0 | **Date** : Avril 2025

---

## 1. Synthèse des articles analysés

| # | Titre (abrégé) | Auteurs | Contribution principale | Pertinence projet |
|---|---|---|---|---|
| A1 | MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework | Hong et al. | Intégration de SOPs humaines dans des systèmes multi-agents LLM | Architecture multi-agents, communication structurée via PipelineState |
| A2 | AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | Wu et al. | Framework d'agents conversables avec programmation de conversation | Orchestration d'agents, Human-in-the-loop |
| A3 | Arabic Sentiment Analysis of YouTube Comments | Musleh et al. | Pipeline NLP complet pour l'analyse de sentiment sur YouTube | A3 Sentiment, A5 Bruit, protocole d'annotation gold standard |
| A4 | Spam Detection on YouTube Comments Using Advanced ML Models | Airlangga | Étude comparative de 10 modèles ML pour la détection de spam YouTube | A5 Bruit (catégorie spam), LinearSVC comme baseline de comparaison |
| A5 | LLM in Computational Social Science | Thapa et al. | État des lieux des LLM en CSS, taxonomie des applications | Justification scientifique du choix LLM, limitations éthiques |

---

## 2. Ce que la littérature a déjà résolu

### 2.1 Analyse de sentiment sur les commentaires YouTube
- **A3 (Musleh et al.)** atteint **94,62 % d'exactitude** avec Naïve Bayes sur 4 212 commentaires arabes (Fleiss' Kappa = 0.818 inter-annotateurs).
- **A4 (Airlangga)** atteint **95,33 % d'exactitude** avec LinearSVC pour la détection de spam (F1 = 95.32 %, validation croisée stratifiée à 10 plis).
- Ces travaux prouvent la **faisabilité technique** de l'analyse automatique de commentaires YouTube avec des niveaux de performance exploitables en production.
- La combinaison texte + métadonnées (auteur, nom de la vidéo) améliore significativement la détection (A4).

### 2.2 Architectures multi-agents LLM
- **A1 (MetaGPT)** démontre que la spécialisation des rôles et les Procédures Opérationnelles Standardisées (SOPs) réduisent les hallucinations en cascade et améliorent la cohérence logique entre agents.
- **A2 (AutoGen)** fournit un paradigme de "conversation programming" unifiant agents LLM, outils de code Python et intervention humaine sous une interface unifiée.
- **A5 (Thapa et al.)** recense des systèmes multi-agents (COLA, HAD) surpassant systématiquement les LLM uniques pour des tâches de CSS complexes.

### 2.3 Détection de bruit et de contenu toxique
- **A4** établit que **LinearSVC** est optimal pour le spam YouTube : meilleur compromis précision/efficacité computationnelle parmi 10 modèles testés.
- **A3** documente un pipeline de nettoyage validé : suppression URL/emojis, normalisation, tokenisation, racinisation (ISRIStemmer).
- **A5** recense des jeux de données annotés pour la haine, la toxicité et la "Hope Speech" sur YouTube, confirmant la richesse sémantique des commentaires.

---

## 3. Les lacunes identifiées (Gaps)

### GAP-01 — Absence de score de qualité global multi-dimensionnel
**Constat** : Les travaux existants analysent une seule dimension à la fois : sentiment (A3) **OU** spam (A4) **OU** toxicité (A5). Aucun ne propose un **score composite** intégrant simultanément sentiment, profondeur du discours et niveau de bruit dans un pipeline unifié.

**Notre réponse** : Agent A6 Synthétiseur avec formule pondérée scientifiquement justifiée :
```
Score_Global = 0.35 × Score_Sentiment + 0.40 × Score_Discours + 0.25 × Score_Bruit
```

---

### GAP-02 — Absence de personnalisation par thématique utilisateur
**Constat** : Tous les systèmes analysés produisent un score générique, indépendant de l'intention de recherche de l'utilisateur. Aucun ne permet d'adapter la pertinence d'une vidéo à un topic spécifique (ex : "machine learning" vs "cuisine").

**Notre réponse** : Agent A7 Topic Matcher — score de pertinence personnalisé :
```
Score_Final = 0.60 × Score_Global + 0.40 × Score_Pertinence_Topic
```

---

### GAP-03 — Évaluation de la profondeur du discours non traitée
**Constat** : Les travaux existants détectent des catégories binaires (positif/négatif, spam/non-spam) mais n'évaluent pas la **qualité argumentative** ou l'**informativité** des commentaires. A5 mentionne le discours comme niveau d'analyse pertinent mais ne propose aucune implémentation opérationnelle.

**Notre réponse** : Agent A4 Discours — évaluation sur **3 dimensions** (informativité, argumentation, constructivité), avec identification automatique des commentaires à haute valeur pour A7 (score ≥ 0.7).

---

### GAP-04 — Aucun framework multi-agents LLM appliqué à l'évaluation de qualité vidéo
**Constat** : MetaGPT (A1) et AutoGen (A2) prouvent l'efficacité des systèmes multi-agents LLM pour des tâches logicielles et mathématiques. Aucun travail existant n'applique cette architecture à la **prédiction de qualité de contenu vidéo** à partir de commentaires utilisateurs.

**Notre réponse** : Pipeline LangGraph à **7 agents spécialisés** — première application de cette classe d'architecture à la prédiction de qualité YouTube selon la littérature recensée.

---

### GAP-05 — Signal de qualité négatif absent depuis 2021
**Constat** : A3 (Musleh et al.) soulève explicitement la suppression du compteur de dislikes (novembre 2021) comme motivation principale. Cependant, la réponse proposée (analyse de sentiment binaire) ne produit pas de signal exploitable par l'utilisateur final **avant** visionnage.

**Notre réponse** : Score de qualité [0-100] avec label qualitatif (Faible/Moyen/Bon/Excellent) — remplace fonctionnellement le signal de dislike supprimé et est exposé directement via API.

---

### GAP-06 — Manque d'intégration dans l'expérience utilisateur de navigation
**Constat** : Tous les systèmes existants sont des outils de recherche académique offline. Aucun n'expose son analyse via une **API standardisée** consommable directement dans l'interface de navigation YouTube.

**Notre réponse** : API FastAPI (POST /analyze, GET /report/{video_id}) avec CORS configuré, exposant le pipeline pour intégration dans n'importe quel client (extension Chrome, webapp, notebook).

---

### GAP-07 — Gold standard multilingue et multi-dimensionnel inexistant
**Constat** : A3 utilise un gold standard de 4 212 commentaires **arabes uniquement**, sur une seule dimension (sentiment binaire). Aucun travail ne propose un gold standard multilingue annoté simultanément sur sentiment, qualité du discours et qualité globale perçue.

**Notre réponse** : Gold standard annoté par ≥ 2 annotateurs humains, accord kappa > 0.7, sur des commentaires multilingues (EN/FR/AR), avec annotation sur **3 dimensions** (Critère AC-08 du PRD).

---

## 4. Tableau de positionnement comparatif

| Capacité | A3 (Musleh) | A4 (Airlangga) | A5 (Thapa) | **Notre projet** |
|---|:---:|:---:|:---:|:---:|
| Analyse de sentiment | ✅ Binaire | ❌ | ⚠️ Évoqué | ✅ [0-100] + label |
| Évaluation du discours | ❌ | ❌ | ⚠️ Évoqué | ✅ 3 dimensions |
| Détection de bruit/spam | ❌ | ✅ Binaire | ⚠️ Évoqué | ✅ 5 catégories |
| Score composite pondéré | ❌ | ❌ | ❌ | ✅ Formule scientifique |
| Personnalisation thématique | ❌ | ❌ | ❌ | ✅ A7 Topic Matcher |
| Architecture multi-agents LLM | ❌ | ❌ | ⚠️ Review | ✅ LangGraph 7 agents |
| API exposée | ❌ | ❌ | ❌ | ✅ FastAPI REST |
| Support multilingue | ❌ Arabe seul | ❌ Anglais | ✅ Review | ✅ langdetect |
| Évaluation avant visionnage | ⚠️ Partiel | ❌ | ❌ | ✅ Score + verdict |

---

## 5. Conclusion

La littérature existante valide la **faisabilité** de l'analyse automatique de commentaires YouTube avec des performances élevées (> 94 %). Elle confirme également la supériorité des architectures multi-agents LLM sur les approches monolithiques.

Cependant, elle ne répond pas au besoin central de l'utilisateur final : **prédire la qualité d'une vidéo de façon personnalisée et intégrée** dans son expérience de navigation, en compensant la disparition du signal de dislike.

Ce projet comble **7 lacunes identifiées** (GAP-01 à GAP-07) en combinant :
- Une architecture multi-agents LLM avec LangGraph (inspirée de MetaGPT/AutoGen)
- Une analyse tri-dimensionnelle originale (sentiment + discours + bruit)
- Un score composite personnalisé par thématique (A7 Topic Matcher)
- Une exposition via API REST consommable

**Ces gaps constituent la contribution originale du projet et justifient scientifiquement sa valeur devant le comité académique.**
