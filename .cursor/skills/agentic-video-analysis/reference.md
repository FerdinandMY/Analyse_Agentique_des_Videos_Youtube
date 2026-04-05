skills.yml — AGENTIC BASED VIDEO ANALYSIS

```yml
# ═══════════════════════════════════════════════════════════════════
# skills.yml — AGENTIC BASED VIDEO ANALYSIS
# Fichier décrivant les compétences techniques attendues et
# les capacités de l'IA sur ce projet
# ═══════════════════════════════════════════════════════════════════

project:
  name: "Agentic Based Video Analysis"
  description: >
    Système multi-agents LLM pour prédire la qualité d'une vidéo
    YouTube avant visionnage, à partir de l'analyse des commentaires.
  language: "Python 3.10+"
  environment: "Kaggle Free Tier"
  context: "Recherche académique — présentation devant comité scientifique"


# ───────────────────────────────────────────────────────────────────
# SKILL 1 — ARCHITECTURE MULTI-AGENTS
# ───────────────────────────────────────────────────────────────────
skills:

  - id: "SK-01"
    name: "Conception d'architecture multi-agents LLM"
    category: "Architecture"
    level: "Expert"
    description: >
      Concevoir, documenter et implémenter un pipeline de six agents
      LLM spécialisés avec des responsabilités uniques, une communication
      structurée via messages JSON, et une zone d'exécution parallèle.
    competencies:
      - "Définir les interfaces d'entrée/sortie de chaque agent"
      - "Implémenter le pattern BaseAgent avec héritage obligatoire"
      - "Concevoir le protocole de communication via la classe Message"
      - "Orchestrer l'exécution parallèle avec ThreadPoolExecutor"
      - "Rédiger un Architecture Decision Record (ADR) complet"
    references:
      - "MetaGPT — Hong et al., ICLR 2024"
      - "AutoGen — Wu et al., 2023"
      - "LLM-based Multi-Agent Systems — Guo et al., IJCAI 2024"
    files:
      - "src/core/base_agent.py"
      - "src/core/message.py"
      - "src/core/pipeline.py"
      - "docs/phase3_architecture/ADR_001_multi_agent_architecture.md"


  - id: "SK-02"
    name: "Collecte de données via YouTube Data API v3"
    category: "Data Engineering"
    level: "Intermédiaire"
    description: >
      Extraire des commentaires YouTube de manière fiable via l'API
      officielle, gérer la pagination, les quotas et les erreurs réseau.
    competencies:
      - "Configurer et authentifier l'accès à YouTube Data API v3"
      - "Implémenter la pagination avec list_next()"
      - "Gérer les erreurs de quota (HttpError 403) et les retries"
      - "Structurer la sortie en JSON avec métadonnées vidéo complètes"
      - "Stocker les clés API dans Kaggle Secrets (jamais dans le code)"
    tools:
      - "google-api-python-client"
      - "kaggle_secrets.UserSecretsClient"
    files:
      - "src/agents/agent1_collector.py"
      - "src/utils/youtube_api.py"
      - "notebooks/phase4_agent1_collector.ipynb"


  - id: "SK-03"
    name: "Prétraitement et normalisation de texte NLP"
    category: "NLP"
    level: "Intermédiaire"
    description: >
      Nettoyer, normaliser et structurer des commentaires de réseaux
      sociaux pour les rendre exploitables par des modèles LLM.
    competencies:
      - "Détecter la langue avec langdetect et filtrer par langue cible"
      - "Nettoyer le texte : URLs, emojis, caractères spéciaux"
      - "Dédupliquer les commentaires avec hashing"
      - "Tokeniser avec NLTK ou spaCy"
      - "Ajouter des identifiants UUID traçables à chaque commentaire"
    tools:
      - "langdetect"
      - "nltk"
      - "spacy (en_core_web_sm)"
      - "re (expressions régulières)"
    files:
      - "src/agents/agent2_preprocessor.py"
      - "src/utils/text_cleaner.py"
      - "src/utils/language_detector.py"


  - id: "SK-04"
    name: "Analyse de sentiment avec LLM et pattern ReAct"
    category: "NLP / LLM"
    level: "Expert"
    description: >
      Analyser la polarité émotionnelle de commentaires YouTube en
      utilisant des LLMs légers avec le pattern Thought/Action/Result,
      et agréger les résultats en un score global pondéré.
    competencies:
      - "Implémenter le pattern ReAct (Thought → Action → Result)"
      - "Concevoir des prompts few-shot avec 3 à 5 exemples annotés"
      - "Parser et valider les sorties JSON structurées du LLM"
      - "Calculer le score de sentiment agrégé sur un corpus"
      - "Gérer les cas ambigus (ironie, sarcasme, multilinguisme)"
    formula: "Score_Sentiment = (pos×1 + neu×0.5 + neg×0) / total × 100"
    tools:
      - "microsoft/Phi-3-mini-4k-instruct"
      - "Qwen/Qwen1.5-7B-Chat"
      - "google-generativeai (gemini-1.5-flash)"
    references:
      - "Giri et al., 2024 — YouTube Comments Sentiment Analysis"
      - "ReAct — Yao et al., ICLR 2023"
    files:
      - "src/agents/agent3_sentiment.py"
      - "src/prompts/sentiment_v1.txt"


  - id: "SK-05"
    name: "Analyse du discours et profondeur argumentative"
    category: "NLP / LLM"
    level: "Expert"
    description: >
      Évaluer la profondeur intellectuelle des commentaires sur trois
      dimensions orthogonales : pertinence topique, profondeur
      argumentative et valeur informative.
    competencies:
      - "Définir et opérationnaliser trois dimensions d'analyse"
      - "Distinguer réactions superficielles et échanges substantiels"
      - "Calculer un score composite multi-dimensionnel"
      - "Prompts avec définitions explicites de chaque dimension"
      - "Valider l'orthogonalité des dimensions pour l'ablation"
    dimensions:
      topical_relevance:
        description: "Le commentaire traite-t-il du contenu de la vidéo ?"
        score_range: "0.0 – 1.0"
      argumentative_depth:
        description: "Présence de raisonnements, sources, questions constructives"
        score_range: "0.0 – 1.0"
      informative_value:
        description: "Apport d'informations, corrections ou compléments utiles"
        score_range: "0.0 – 1.0"
    formula: "Score_Discourse = mean(topicality + depth + value) × 100"
    references:
      - "He et al., 2023 — ABSA with BERT and Knowledge Graph"
      - "Systematic Review ABSA — Springer AI Review 2024"
    files:
      - "src/agents/agent4_discourse.py"
      - "src/prompts/discourse_v1.txt"


  - id: "SK-06"
    name: "Détection de bruit et contenu non informatif"
    category: "NLP / Classification"
    level: "Expert"
    description: >
      Identifier et quantifier les commentaires qui ne contribuent pas
      à l'évaluation de la qualité vidéo, en combinant un classifieur
      léger (SVM) et un LLM pour la classification sémantique fine.
    competencies:
      - "Entraîner un classifieur SVM sur le dataset SenTube"
      - "Classifier en 5 catégories : informatif, spam, hors-sujet, réaction, toxique"
      - "Combiner SVM (spam rapide) et LLM (sémantique fine)"
      - "Calculer le ratio de bruit et le score pénalité"
      - "Gérer les cas limites (commentaires très courts, multilangues)"
    categories:
      - "informative  : contribue à l'évaluation"
      - "spam         : promotion, liens commerciaux, répétitions"
      - "off_topic    : ne parle pas de la vidéo"
      - "pure_reaction: trop court et vide (lol, wow, +1)"
      - "toxic        : insultes, harcèlement"
    formula: "Score_Noise = 100 − (non_informative / total × 100)"
    tools:
      - "scikit-learn (SVM, TF-IDF)"
      - "SenTube dataset pour l'entraînement du SVM"
    references:
      - "Spam Detection on YouTube — Brilliance AI, 2024"
      - "LLMs in Computational Social Science — Springer 2025"
    files:
      - "src/agents/agent5_noise_detector.py"
      - "src/prompts/noise_detection_v1.txt"


  - id: "SK-07"
    name: "Agrégation pondérée et génération de résumé LLM"
    category: "LLM / Scoring"
    level: "Expert"
    description: >
      Agréger les scores partiels des trois agents analytiques en un
      score global unique via une formule pondérée justifiée, et
      générer un résumé explicatif en langage naturel.
    competencies:
      - "Appliquer la formule pondérée (0.35/0.40/0.25)"
      - "Catégoriser le score en niveau de qualité (4 niveaux)"
      - "Générer un résumé LLM contextuel et structuré"
      - "Identifier et formuler les points forts et faibles"
      - "Produire le JSON final complet pour l'utilisateur"
    formula: >
      Score_Global = (0.35 × S_sentiment) +
                     (0.40 × S_discourse)  +
                     (0.25 × S_noise)
    output_fields:
      - "quality_score    : int [0–100]"
      - "quality_level    : str [Faible|Moyen|Bon|Excellent]"
      - "partial_scores   : dict {sentiment, discourse, noise}"
      - "summary          : str (résumé LLM en langage naturel)"
      - "strengths        : list[str]"
      - "weaknesses       : list[str]"
      - "recommendation   : str"
    references:
      - "Heterogeneous LLM Agents — Du et al., ACM TMIS 2024"
    files:
      - "src/agents/agent6_synthesizer.py"
      - "src/scoring/scorer.py"
      - "src/prompts/synthesis_v1.txt"


  - id: "SK-08"
    name: "Ingénierie des prompts (Prompt Engineering)"
    category: "LLM"
    level: "Expert"
    description: >
      Concevoir, versionner, itérer et documenter des prompts LLM
      optimisés pour des tâches analytiques spécialisées sur des
      commentaires de réseaux sociaux.
    competencies:
      - "Structurer un prompt système + few-shot + instruction"
      - "Enforcer le format ReAct (Thought/Action/Result)"
      - "Enforcer les sorties JSON structurées"
      - "Itérer les prompts de manière scientifique et documentée"
      - "Mesurer l'impact de chaque modification de prompt"
      - "Versionner les prompts comme du code (v1, v2, v3...)"
    versioning_convention: "src/prompts/{agent}_v{N}.txt"
    journal: "docs/phase4_implementation/prompt_journal.md"
    references:
      - "Pre-train, Prompt, and Predict — Liu et al., ACM 2023"
      - "ReAct — Yao et al., ICLR 2023"


  - id: "SK-09"
    name: "Évaluation scientifique et analyse d'ablation"
    category: "Évaluation"
    level: "Expert"
    description: >
      Évaluer le système de manière rigoureuse avec des métriques
      adaptées, comparer à des baselines solides, et conduire une
      analyse d'ablation pour quantifier la contribution de chaque agent.
    competencies:
      - "Calculer Accuracy, Précision, Rappel, F1-score"
      - "Calculer les corrélations de Pearson et Spearman"
      - "Implémenter et comparer 3 baselines (YouTube brut, Naive Bayes, BERT seul)"
      - "Conduire une analyse d'ablation : retirer chaque agent et mesurer l'impact"
      - "Calculer les intervalles de confiance pour la validité statistique"
      - "Réaliser une analyse d'erreurs (error analysis) par catégorie"
    baselines:
      - "B1: Score YouTube brut (likes/dislikes ratio)"
      - "B2: Classifieur Naive Bayes sur les commentaires"
      - "B3: BERT seul sans architecture multi-agents"
    files:
      - "evaluation/compute_metrics.py"
      - "evaluation/ablation_study.py"
      - "evaluation/baseline_comparison.py"
      - "evaluation/error_analysis.py"
      - "notebooks/phase5_evaluation.ipynb"


  - id: "SK-10"
    name: "Développement sur Kaggle Free Tier"
    category: "Infrastructure"
    level: "Intermédiaire"
    description: >
      Développer et faire fonctionner un pipeline LLM complet dans
      les contraintes du Kaggle Free Tier (GPU T4, 30h/semaine,
      quota API, sessions interruptibles).
    competencies:
      - "Gérer le quota GPU (30h/semaine) de manière stratégique"
      - "Configurer Kaggle Secrets pour les clés API"
      - "Optimiser la mémoire GPU (float16, empty_cache, batching)"
      - "Implémenter le pattern de checkpoint pour les sessions interruptibles"
      - "Publier et documenter des notebooks reproductibles"
    memory_rules:
      - "Toujours utiliser torch.float16 (jamais float32)"
      - "Appeler torch.cuda.empty_cache() entre les agents"
      - "Traiter en batches de 10 commentaires maximum"
      - "Ne jamais charger deux modèles LLM simultanément"
    gpu_budget:
      phase4_unit_tests:  "8h"
      phase4_full_pipeline: "6h"
      phase5_evaluation:  "10h"
      phase5_baselines:   "4h"
      margin:             "2h"


  - id: "SK-11"
    name: "Rédaction scientifique et documentation académique"
    category: "Documentation"
    level: "Expert"
    description: >
      Produire la documentation scientifique du projet selon les
      standards académiques : rapport IMRaD, ADR, diagrammes,
      justifications bibliographiques.
    competencies:
      - "Rédiger un rapport scientifique au format IMRaD"
      - "Formuler une contribution originale claire et défendable"
      - "Rédiger un Architecture Decision Record (ADR) complet"
      - "Citer les sources académiques dans le code et la documentation"
      - "Produire des visualisations scientifiques (LaTeX, TikZ)"
      - "Anticiper et préparer les réponses aux questions du jury"
    documents:
      - "docs/phase3_architecture/ADR_001_multi_agent_architecture.md"
      - "docs/phase4_implementation/prompt_journal.md"
      - "docs/phase5_evaluation/results_table.md"
      - "docs/phase6_report/report_draft.md"
    references_used:
      - "MetaGPT — Hong et al., ICLR 2024"
      - "AutoGen — Wu et al., 2023"
      - "ReAct — Yao et al., ICLR 2023"
      - "Giri et al., 2024 — YouTube Sentiment"
      - "He et al., 2023 — ABSA BERT Knowledge Graph"
      - "LLM-MA Survey — Guo et al., IJCAI 2024"
      - "Heterogeneous LLM Agents — Du et al., ACM 2024"


# ───────────────────────────────────────────────────────────────────
# RÉSUMÉ DES NIVEAUX DE COMPÉTENCE
# ───────────────────────────────────────────────────────────────────

skill_summary:
  expert:
    - "SK-01 : Architecture multi-agents"
    - "SK-04 : Analyse de sentiment + ReAct"
    - "SK-05 : Analyse du discours"
    - "SK-06 : Détection de bruit"
    - "SK-07 : Agrégation + résumé LLM"
    - "SK-08 : Prompt engineering"
    - "SK-09 : Évaluation scientifique"
    - "SK-11 : Rédaction académique"
  intermediate:
    - "SK-02 : YouTube Data API v3"
    - "SK-03 : Prétraitement NLP"
    - "SK-10 : Kaggle Free Tier"

technology_stack:
  llm_models:
    - "microsoft/Phi-3-mini-4k-instruct"
    - "Qwen/Qwen1.5-7B-Chat"
    - "google/gemini-1.5-flash"
  nlp_libraries:
    - "transformers (HuggingFace)"
    - "langdetect"
    - "nltk"
    - "spacy"
    - "scikit-learn"
  data_libraries:
    - "pandas"
    - "numpy"
  apis:
    - "YouTube Data API v3"
    - "Google AI Studio (Gemini)"
  infrastructure:
    - "Kaggle Free Tier (GPU T4)"
    - "concurrent.futures (parallel execution)"
  documentation:
    - "LaTeX + TikZ (diagrams)"
    - "Markdown (ADR, journals)"
    - "JSON Schema (message contracts)"

# ═══════════════════════════════════════════════════════════════════
# END OF skills.yml
# ═══════════════════════════════════════════════════════════════════
```

