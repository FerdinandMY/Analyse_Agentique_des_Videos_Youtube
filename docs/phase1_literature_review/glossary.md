# Glossaire — YouTube Quality Analyzer
**Version** : 1.0 | **Date** : Avril 2025

---

## A — Agents & Architecture

**Agent (A1 → A7)**
Nœud fonctionnel du pipeline LangGraph. Chaque agent reçoit le `PipelineState` partagé, exécute une tâche spécialisée et retourne uniquement les clés d'état qu'il modifie. Les agents ne communiquent jamais directement entre eux.

**A1 — Loader / Validator**
Premier agent du pipeline. Charge et valide le fichier CSV de commentaires pré-collectés. Vérifie la présence de la colonne `text` et des colonnes optionnelles (`video_id`, `author_likes`, `reply_count`).

**A2 — Préprocesseur**
Nettoie et normalise les commentaires bruts : mise en minuscules, collapsage des espaces, filtrage par longueur (3–2000 chars), déduplication exacte, détection de langue (`langdetect`).

**A3 — Sentiment Analyser**
Classifie le sentiment global du corpus de commentaires (positive / neutral / negative) et produit un `Score_Sentiment` [0-100]. Utilise Phi-3-mini-4k-instruct via le LLM loader.

**A4 — Discourse Analyser**
Évalue la profondeur du discours sur 3 dimensions (informativité, argumentation, constructivité) et produit un `Score_Discours` [0-100]. Identifie les commentaires à haute valeur (score ≥ 0.7) utilisés par A7. Utilise Qwen1.5-7B-Chat.

**A5 — Noise Detector**
Détecte et classifie le bruit en 5 catégories (spam, offtopic, réaction vide, toxique, bot) et produit un `Score_Bruit` [0-100] = (1 - noise_ratio) × 100. Combine SVM heuristique et LLM.

**A6 — Synthétiseur**
Agrège les scores de A3, A4, A5 selon la formule pondérée et génère un résumé LLM en langage naturel. Produit le `Score_Global` et le `quality_label`.

**A7 — Topic Matcher** *(nouveau v2.0)*
Compare la thématique utilisateur avec les commentaires de haute qualité sélectionnés par A4. Produit le `Score_Pertinence` [0-100] et le `Score_Final` personnalisé. Utilise Qwen1.5-7B-Chat.

---

## C — Concepts NLP & ML

**Chain-of-Thought (CoT)**
Technique de prompting guidant le LLM à raisonner étape par étape avant de produire une réponse. Améliore les performances sur les tâches nécessitant de l'inférence (sarcasme, nuances de discours). Référencé dans Thapa et al. (A5).

**Cohen's Kappa (κ)**
Mesure de l'accord inter-annotateurs corrigée par le hasard. Formule : κ = (Po − Pe) / (1 − Pe). Seuils d'interprétation : < 0.40 faible, 0.40–0.60 modéré, 0.60–0.80 substantiel, > 0.80 presque parfait. Seuil requis par AC-08 PRD : κ > 0.70.

**Corrélation de Pearson (r)**
Mesure linéaire de la corrélation entre deux variables continues. Utilisée pour évaluer la corrélation entre le Score_Global prédit et le score humain du gold standard. Seuil de validation H1 : r ≥ 0.60, p < 0.05.

**F1-Score**
Moyenne harmonique de la précision et du rappel. F1 = 2 × (Précision × Rappel) / (Précision + Rappel). Utilisé pour évaluer la classification qualité (Faible / Moyen / Bon / Excellent) en Phase 5.

**Fleiss' Kappa**
Extension du Cohen's Kappa pour > 2 annotateurs. Utilisé par Musleh et al. (A3) : κ = 0.818 sur 4 212 commentaires arabes.

**Gold Standard**
Ensemble de données annoté manuellement (ou semi-automatiquement par LLM) servant de référence pour l'évaluation. Annoté par ≥ 2 annotateurs avec accord kappa > 0.7 (AC-08 PRD). Taille cible : 100 commentaires équilibrés.

**LangGraph**
Framework d'orchestration de graphes d'agents LLM développé par LangChain. Permet de définir des pipelines sous forme de graphes orientés avec parallélisme, gestion d'état partagé (`TypedDict`) et checkpointing. Version utilisée : ≥ 0.2.0.

**LangChain**
Bibliothèque Python facilitant l'intégration de LLM dans des applications. Fournit les briques `ChatOpenAI`, `PydanticOutputParser`, `SystemMessage`, `HumanMessage`. Utilisée pour les appels LLM dans chaque agent.

**LLM (Large Language Model)**
Modèle de langage entraîné sur de grandes quantités de texte, capable de comprendre et générer du langage naturel. Modèles utilisés dans ce projet : Phi-3-mini-4k-instruct (A3, A5) et Qwen1.5-7B-Chat (A4, A6, A7).

**LinearSVC**
Support Vector Classifier linéaire. Meilleur modèle pour la détection de spam YouTube selon Airlangga (A4) : accuracy 95.33 %, F1 95.32 %. Utilisé comme baseline de comparaison pour A5.

**MCC (Matthews Correlation Coefficient)**
Métrique d'évaluation robuste aux classes déséquilibrées. MCC ∈ [-1, 1], 1 = prédiction parfaite. Utilisé par Musleh et al. (A3) : MCC = 91.46 % avec Naïve Bayes.

**N-gram**
Séquence de N mots consécutifs. Utilisé comme feature pour les modèles ML classiques. Musleh et al. (A3) utilisent des bigrammes et trigrammes pour améliorer la classification de sentiment.

**PipelineState**
`TypedDict` LangGraph partagé entre tous les agents. Contient les données d'entrée (`csv_path`, `topic`), les sorties intermédiaires (`raw_comments`, `cleaned_comments`, `sentiment`, `discourse`, `noise`, `synthesis`) et les scores finaux (`score_global`, `score_pertinence`, `score_final`).

**Pydantic**
Bibliothèque Python de validation de données par types. Utilisée pour les schémas de sortie LLM (`SentimentOutput`, `DiscourseOutput`…) via `PydanticOutputParser`, et pour les modèles API FastAPI (`AnalyzeRequest`, `AnalyzeResponse`).

**RAG (Retrieval-Augmented Generation)**
Architecture combinant un moteur de recherche (retrieval) et un LLM (generation). Permet aux agents d'interroger un corpus de données avant de générer une réponse. Référencé dans AutoGen (A2) et Thapa et al. (A5) comme piste pour A7 Topic Matcher.

**ReAct**
Paradigme de prompting combinant Reasoning (raisonnement) et Acting (actions). Guide le LLM à alterner entre réflexion et appel d'outils. Mentionné dans les spécifications d'agents analytiques (A3, A5).

**Seed**
Valeur initiale du générateur de nombres aléatoires garantissant la reproductibilité des expériences. **Valeur fixée à 42** dans tout le projet (NFR-02).

**SMOTE (Synthetic Minority Over-sampling Technique)**
Technique d'augmentation de données pour équilibrer les classes. Recommandée par Airlangga (A4) quand les commentaires légitimes sont beaucoup plus nombreux que le spam.

**SOP (Standard Operating Procedure)**
Procédure opérationnelle standardisée. Dans MetaGPT (A1), les SOPs humaines sont encodées dans le système multi-agents pour structurer les flux de travail et réduire les hallucinations.

**TF-IDF (Term Frequency-Inverse Document Frequency)**
Mesure de l'importance d'un mot dans un document par rapport à un corpus. Améliore la plupart des classificateurs ML sauf Naïve Bayes (Musleh et al., A3). Utilisé par Airlangga (A4) pour transformer les commentaires en vecteurs numériques.

**TypedDict**
Type Python (module `typing`) permettant de définir des dictionnaires avec des types de valeurs spécifiés. Utilisé pour définir `PipelineState` dans LangGraph.

---

## D — Données & Évaluation

**Ablation Study (Étude d'ablation)**
Protocole d'évaluation consistant à retirer successivement chaque composant du système pour mesurer sa contribution individuelle. Requise par AC-09 du PRD. Teste H2 (multi-agents vs LLM unique) et H3 (primauté du discours).

**Annotation Guide**
Document spécifiant les consignes données aux annotateurs humains pour labelliser le gold standard. Décrit les critères de classification pour `sentiment_label`, `quality_score` [1-5] et `noise_category`.

**Data Card**
Fiche descriptive d'un dataset documentant son origine, ses statistiques, ses biais potentiels et ses conditions d'utilisation. Livrable Phase 2 du projet.

**EDA (Exploratory Data Analysis)**
Analyse exploratoire des données. Phase préliminaire examinant la distribution, la qualité et les caractéristiques du corpus avant de lancer le pipeline. Implémentée dans `notebooks/phase2_eda.ipynb`.

**Gold Standard** → voir section C

**Kappa** → voir Cohen's Kappa (section C)

**Quality Label**
Label qualitatif assigné par A6 sur la base du Score_Global : **Faible** (< 25), **Moyen** (25–50), **Bon** (50–75), **Excellent** (≥ 75).

---

## S — Scores

**Score_Sentiment**
Sortie de A3. Mesure la positivité globale du corpus de commentaires [0-100]. 0 = très négatif, 50 = neutre, 100 = très positif.

**Score_Discours**
Sortie de A4. Moyenne des 3 dimensions (informativité, argumentation, constructivité) [0-100]. Reflète la profondeur intellectuelle des commentaires.

**Score_Bruit**
Sortie de A5. Calculé comme (1 − noise_ratio) × 100. 0 = corpus entièrement bruité, 100 = corpus entièrement propre.

**Score_Global**
Sortie de A6. Score composite pondéré [0-100] :
```
Score_Global = 0.35 × Score_Sentiment
             + 0.40 × Score_Discours
             + 0.25 × Score_Bruit
```
Les poids (w1=0.35, w2=0.40, w3=0.25) sont **immuables** (définis dans le PRD §4.3).

**Score_Pertinence**
Sortie de A7. Mesure l'alignement entre la thématique utilisateur et le contenu de la vidéo inféré des commentaires [0-100].

**Score_Final**
Score personnalisé combinant qualité globale et pertinence thématique [0-100] :
```
Score_Final = 0.60 × Score_Global + 0.40 × Score_Pertinence
```

---

## T — Termes Techniques Projet

**Checkpoint**
Sauvegarde JSON de la sortie d'un agent après son exécution. Permet la reprise du pipeline en cas d'interruption de session Kaggle (NFR-03). Nommage : `checkpoint_{pipeline_id}_{agent}.json`. Implémenté dans `utils/checkpoint.py`.

**CORS (Cross-Origin Resource Sharing)**
Mécanisme HTTP permettant à une page web d'une origine (l'extension Chrome) d'accéder à des ressources d'une autre origine (l'API FastAPI). Configuré dans `api/main.py` (PR-16).

**Dislike supprimé**
YouTube a supprimé le compteur public de dislikes en novembre 2021. Cette suppression motive la création d'un indicateur alternatif de qualité basé sur les commentaires. Problématique centrale du projet (Musleh et al., A3).

**FastAPI**
Framework Python moderne pour la construction d'APIs REST. Utilise Pydantic pour la validation automatique des données et génère automatiquement la documentation OpenAPI (Swagger). Point d'entrée : `uvicorn api.main:app`.

**float16 (FP16)**
Format de nombre flottant sur 16 bits. Utilisé pour charger les modèles HuggingFace en réduisant la consommation VRAM de moitié (≈8 Go vs 16 Go en float32). Requis par NFR-04 (contrainte GPU T4 Kaggle ≤ 14 Go VRAM).

**Kaggle Free Tier**
Environnement d'exécution gratuit proposé par Kaggle : GPU NVIDIA T4 (16 Go VRAM), 30h/semaine. Contrainte principale du projet (NFR-04).

**Manifest V3**
Version actuelle du format de manifeste pour les extensions Chrome. Impose des restrictions de sécurité plus strictes (Service Worker au lieu de Background Page, CSP renforcée). Utilisé pour l'extension Chrome du projet (hors périmètre du dépôt backend).

**MemorySaver**
Checkpointer LangGraph en mémoire (RAM). Utilisé en développement. Remplacé par `SqliteSaver` pour la persistance entre sessions.

**Pipeline ID**
Identifiant unique généré pour chaque exécution du pipeline (timestamp Unix par défaut). Utilisé pour nommer les checkpoints et tracer les analyses (NFR-07).

**Prompt versioning**
Pratique consistant à versionner les prompts LLM dans des fichiers texte séparés (`prompts/{agent}_v{N}.txt`). Permet de comparer les performances de différentes versions de prompts sans modifier le code. Requis par PR-23.

**Topic (Thématique)**
Chaîne de caractères fournie par l'utilisateur décrivant son intention de recherche (ex : "machine learning", "cuisine végane"). Entrée principale de A7 Topic Matcher. Transmise via l'API (`AnalyzeRequest.topic`) ou le CLI (`--topic`).

**VRAM (Video RAM)**
Mémoire dédiée au GPU. Contrainte critique sur Kaggle T4 : 16 Go disponibles, 14 Go utilisables (NFR-04). Un seul modèle LLM chargé à la fois (PR-11), en float16 (PR-12).
