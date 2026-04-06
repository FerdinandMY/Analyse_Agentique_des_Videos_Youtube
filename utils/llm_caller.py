"""
utils/llm_caller.py
====================
Couche anti-hallucination — PRD v3.0 §3.5

Pipeline de sécurité à 5 niveaux pour tout appel LLM :
  1. Prompt structuré (JSON forcé + grounding) — responsabilité de l'appelant
  2. Température basse                          — via get_llm(temperature=...)
  3. json.loads → regex fallback → heuristique
  4. Validation Pydantic                        — via validators.py
  5. Retry x3 avec prompt renforcé

Usage :
    from utils.llm_caller import safe_llm_call

    result, meta = safe_llm_call(
        messages=[SystemMessage(...), HumanMessage(...)],
        validator=MySentimentValidator(),
        fallback={"sentiment_label": "neutral", "sentiment_score": 50.0},
        agent_name="A3",
    )
    # meta["method"] : "llm_json" | "llm_regex" | "fallback_used"
    # meta["retries"] : int
    # meta["hallucination_flags"] : list[str]
"""
from __future__ import annotations

import json
import re
from typing import Any

from utils.logger import get_logger

logger = get_logger("llm_caller")

# Nombre maximum de tentatives avant fallback
_MAX_RETRIES = 3

# Pattern pour extraire un bloc JSON d'une réponse LLM
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*([\s\S]+?)\s*```"   # bloc ```json...```
    r"|(\{[\s\S]*\})"                    # ou objet JSON brut
    r"|(\[[\s\S]*\])",                    # ou tableau JSON brut
    re.MULTILINE,
)


# ── Extraction JSON ───────────────────────────────────────────────────────────

def extract_json(raw: str) -> Any:
    """
    Tente d'extraire un objet JSON depuis une réponse LLM.
    Stratégie :
      1. json.loads direct
      2. Extraction de bloc ```json...``` ou objet {...}
      3. Lève ValueError si tout échoue
    """
    text = raw.strip()

    # Tentative 1 — parsing direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tentative 2 — regex pour extraire le bloc JSON
    for match in _JSON_BLOCK_RE.finditer(text):
        candidate = match.group(1) or match.group(2) or match.group(3)
        if candidate:
            try:
                return json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue

    # Tentative 3 — nettoyage agressif (supprimer tout avant { et après })
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Aucun JSON valide trouvé dans la réponse : {text[:200]!r}")


# ── Prompt de retry renforcé ──────────────────────────────────────────────────

def _build_retry_prompt(original_content: str, error: str, attempt: int) -> str:
    return (
        f"[RETRY {attempt}/{_MAX_RETRIES}] Your previous response could not be parsed as JSON.\n"
        f"Error: {error}\n\n"
        f"Original request:\n{original_content}\n\n"
        "IMPORTANT: Return ONLY a valid JSON object. No explanation, no markdown, no extra text."
    )


# ── Appel LLM sécurisé ────────────────────────────────────────────────────────

def safe_llm_call(
    messages: list,
    validator,
    fallback: dict[str, Any],
    agent_name: str = "unknown",
    temperature: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Exécute un appel LLM avec retry x3, parsing JSON, validation Pydantic.

    Paramètres
    ----------
    messages     : liste de SystemMessage / HumanMessage
    validator    : instance d'un validateur (doit implémenter .validate(data) → dict)
    fallback     : dict retourné si toutes les tentatives échouent
    agent_name   : nom de l'agent pour les logs
    temperature  : si fourni, remplace la température du LLM (non utilisé ici,
                   à configurer dans get_llm avant l'appel)

    Retourne
    --------
    (result_dict, meta)
    meta = {
        "method"            : "llm_json" | "llm_regex" | "fallback_used",
        "retries"           : int,
        "hallucination_flags": list[str],
        "fallback_used"     : bool,
    }
    """
    from models.llm_loader import get_llm
    from langchain_core.messages import HumanMessage

    meta: dict[str, Any] = {
        "method":             "llm_json",
        "retries":            0,
        "hallucination_flags": [],
        "fallback_used":      False,
    }

    llm = get_llm()
    if llm is None:
        logger.warning("[%s] LLM indisponible — fallback immédiat", agent_name)
        meta["method"]        = "fallback_used"
        meta["fallback_used"] = True
        meta["hallucination_flags"].append("llm_unavailable")
        return {**fallback, **meta}, meta

    last_error = ""
    current_messages = list(messages)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = llm.invoke(current_messages)
            raw  = resp.content.strip()

            # Niveau 3 — extraction JSON
            try:
                data = extract_json(raw)
                meta["method"] = "llm_json" if attempt == 1 else "llm_regex"
            except ValueError as parse_err:
                raise parse_err

            # Niveau 4 — validation Pydantic
            validated = validator.validate(data)

            # Niveau 5 — vérification cohérence inter-champs
            flags = validator.check_coherence(validated)
            meta["hallucination_flags"].extend(flags)

            meta["retries"] = attempt - 1
            return validated, meta

        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "[%s] Tentative %d/%d échouée : %s",
                agent_name, attempt, _MAX_RETRIES, last_error,
            )
            meta["retries"] = attempt

            if attempt < _MAX_RETRIES:
                # Injecter l'erreur dans le prochain prompt (retry renforcé)
                original_content = current_messages[-1].content if current_messages else ""
                retry_content = _build_retry_prompt(original_content, last_error, attempt)
                current_messages = current_messages[:-1] + [HumanMessage(content=retry_content)]

    # Toutes les tentatives ont échoué
    logger.error(
        "[%s] Échec après %d tentatives — fallback heuristique. Dernière erreur : %s",
        agent_name, _MAX_RETRIES, last_error,
    )
    meta["method"]        = "fallback_used"
    meta["fallback_used"] = True
    meta["hallucination_flags"].append(f"all_retries_failed: {last_error[:100]}")
    return {**fallback, **meta}, meta
