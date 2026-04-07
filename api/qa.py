"""
api/qa.py — Module Q&A vidéo (PRD v1.1 §6)
============================================
Répond aux questions en langage naturel sur une vidéo YouTube
en s'appuyant exclusivement sur :
  - La transcription vidéo stockée dans le cache (segments horodatés)
  - Les commentaires de haute qualité (score A4 >= 0.7) extraits du cache

Principe anti-hallucination FR-88 :
  "Réponds UNIQUEMENT à partir du contexte fourni."
  Toute réponse doit citer ses sources (FR-89).
  Le LLM DOIT décliner explicitement si la réponse dépasse le contexte (FR-91).
"""
from __future__ import annotations

import json
import re
from typing import Any

from utils.logger import get_logger

logger = get_logger("qa_module")

# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_TRANSCRIPT_TOKENS = 4_000   # Limite tokens transcription (PRD §6.4)
_MAX_DESCRIPTION_CHARS = 1_000   # Limite description vidéo injectée dans le prompt
_MAX_TOP_COMMENTS      = 50      # Limite commentaires top A4 (PRD §4.4)
_MAX_HISTORY_TURNS     = 5       # Limite historique conversation (FR-90)
_QA_TEMPERATURE        = 0.2     # Température LLM Q&A (PRD §6.4)

_SYSTEM_PROMPT = """\
Tu es un assistant pédagogique spécialisé dans l'analyse de vidéos YouTube.
Tu réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
Si la réponse n'est pas dans le contexte, dis-le explicitement.
Ne fabrique aucune information extérieure au contexte.

Tes réponses DOIVENT :
1. Être ancrées sur le contexte (transcription ou commentaires)
2. Citer les sources utilisées dans le champ "sources" de ta réponse JSON
3. Décliner poliment si la question dépasse le contexte disponible

Retourne UNIQUEMENT un JSON valide avec cette structure :
{
  "answer": "<ta réponse en langage naturel>",
  "sources": [
    {"type": "transcript", "start": <float>, "text": "<extrait>"},
    {"type": "comment", "comment_id": "<id>", "text": "<extrait>"}
  ],
  "confidence": <float 0.0-1.0>
}"""

# Réponses de refus explicites (FR-91)
_REFUSAL_OUT_OF_SCOPE = (
    "Cette question ne concerne pas le contenu de la vidéo analysée. "
    "Je ne peux répondre qu'à des questions sur le contenu de cette vidéo spécifique."
)
_REFUSAL_NOT_FOUND = (
    "Je n'ai pas trouvé cette information dans le contenu disponible de la vidéo. "
    "La transcription et les commentaires analysés ne contiennent pas de réponse à cette question."
)
_REFUSAL_EXTERNAL_KNOWLEDGE = (
    "Pour répondre à cette question, des connaissances extérieures à la vidéo sont nécessaires. "
    "Je ne peux pas les fournir de façon fiable depuis le contexte disponible."
)


# ── Construction du contexte ──────────────────────────────────────────────────

def _build_description_context(description: str, max_chars: int = _MAX_DESCRIPTION_CHARS) -> str:
    """
    Construit le bloc description vidéo pour le prompt.
    Tronque proprement à max_chars en coupant sur un mot entier.
    """
    if not description or not description.strip():
        return ""
    text = description.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + " [...]"


def _build_transcript_context(transcript: list[dict], max_tokens: int = _MAX_TRANSCRIPT_TOKENS) -> str:
    """
    Construit le bloc transcription horodaté pour le prompt.
    Tronque en conservant le début et la fin si nécessaire.
    """
    if not transcript:
        return ""

    lines = []
    for seg in transcript:
        start = seg.get("start", 0.0)
        text  = seg.get("text", "").strip()
        if text:
            minutes = int(start // 60)
            seconds = int(start % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")

    full = "\n".join(lines)

    # Estimation grossière : 1 token ≈ 4 caractères
    char_limit = max_tokens * 4
    if len(full) <= char_limit:
        return full

    # Tronquer : conserver début + fin
    half = char_limit // 2
    head = full[:half]
    tail = full[-half:]
    return head + "\n[...contenu tronqué...]\n" + tail


def _build_comments_context(top_comments: list[dict], max_comments: int = _MAX_TOP_COMMENTS) -> str:
    """Construit le bloc commentaires haute qualité pour le prompt."""
    if not top_comments:
        return ""
    lines = []
    for i, c in enumerate(top_comments[:max_comments], 1):
        text = c.get("text", "").strip()[:300]
        cid  = c.get("comment_id", f"c{i}")
        lines.append(f"[{cid}] {text}")
    return "\n".join(lines)


def _build_history_context(history: list[dict], max_turns: int = _MAX_HISTORY_TURNS) -> str:
    """Formate l'historique de conversation (FR-90)."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]  # 2 messages par tour
    lines = []
    for turn in recent:
        role    = turn.get("role", "user")
        content = turn.get("content", "").strip()
        prefix  = "Utilisateur" if role == "user" else "Assistant"
        lines.append(f"{prefix} : {content}")
    return "\n".join(lines)


def _build_prompt(
    question: str,
    transcript_ctx: str,
    comments_ctx: str,
    history_ctx: str,
    video_title: str,
    transcript_available: bool,
    description_ctx: str = "",
) -> str:
    """Assemble le prompt complet avec toutes les sections de contexte."""
    parts = []

    if video_title:
        parts.append(f"VIDEO : {video_title}\n")

    # Description — toujours utile même sans transcription
    if description_ctx:
        parts.append(f"CONTEXTE — DESCRIPTION DE LA VIDEO :\n{description_ctx}\n")

    if transcript_available and transcript_ctx:
        parts.append(f"CONTEXTE — TRANSCRIPTION :\n{transcript_ctx}\n")
    else:
        parts.append("CONTEXTE — TRANSCRIPTION : Non disponible pour cette video.\n")

    if comments_ctx:
        parts.append(
            f"CONTEXTE — COMMENTAIRES (selection qualite A4 >= 0.7) :\n{comments_ctx}\n"
        )
    else:
        parts.append("CONTEXTE — COMMENTAIRES : Aucun commentaire de haute qualite disponible.\n")

    if history_ctx:
        parts.append(f"HISTORIQUE DE CONVERSATION :\n{history_ctx}\n")

    parts.append(f"QUESTION ACTUELLE :\n{question}")

    return "\n".join(parts)


# ── Parsing de la réponse LLM ─────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> dict[str, Any]:
    """
    Extrait le JSON de la réponse LLM.
    Fallback sur une réponse textuelle si le JSON est malformé.
    """
    # Tentative directe
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return {
            "answer":     str(data.get("answer", "")),
            "sources":    data.get("sources", []),
            "confidence": float(data.get("confidence", 0.5)),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # Extraction regex partielle
    answer_m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    answer   = answer_m.group(1) if answer_m else text[:500]

    return {
        "answer":     answer,
        "sources":    [],
        "confidence": 0.4,
    }


# ── Réponse de fallback ───────────────────────────────────────────────────────

def _fallback_response(reason: str = "llm_unavailable") -> dict[str, Any]:
    """Réponse utilisée si le LLM est indisponible."""
    return {
        "answer": (
            "Le service de questions-réponses est temporairement indisponible. "
            "Veuillez réessayer dans quelques instants."
        ),
        "sources":         [],
        "confidence":      0.0,
        "transcript_used": False,
        "fallback_used":   True,
    }


# ── Fonction principale ───────────────────────────────────────────────────────

def answer_question(
    question:    str,
    qa_context:  dict[str, Any],
    history:     list[dict] | None = None,
) -> dict[str, Any]:
    """
    Génère une réponse grounded à la question posée (PRD v1.1 §6.3).

    Args:
        question   : Question en langage naturel de l'utilisateur.
        qa_context : Contexte stocké en cache :
                     {transcript, transcript_available, top_comments, video_title}
        history    : Historique de conversation [{role, content}].

    Returns:
        {answer, sources, confidence, transcript_used, fallback_used}
    """
    if not question.strip():
        return {
            "answer":         "Veuillez poser une question.",
            "sources":        [],
            "confidence":     0.0,
            "transcript_used": False,
            "fallback_used":  False,
        }

    transcript           = qa_context.get("transcript") or []
    transcript_available = qa_context.get("transcript_available", False)
    top_comments         = qa_context.get("top_comments") or []
    video_title          = qa_context.get("video_title", "")
    video_description    = qa_context.get("video_description", "")
    history              = history or []

    # Construire les blocs de contexte
    transcript_ctx  = _build_transcript_context(transcript)
    description_ctx = _build_description_context(video_description)
    comments_ctx    = _build_comments_context(top_comments)
    history_ctx     = _build_history_context(history)

    user_prompt = _build_prompt(
        question=question,
        transcript_ctx=transcript_ctx,
        comments_ctx=comments_ctx,
        history_ctx=history_ctx,
        video_title=video_title,
        transcript_available=transcript_available,
        description_ctx=description_ctx,
    )

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            logger.warning("qa_module: LLM indisponible — fallback réponse")
            return _fallback_response("llm_unavailable")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        resp = llm.invoke(messages)
        raw  = resp.content if hasattr(resp, "content") else str(resp)

        parsed = _parse_llm_response(raw)

        # Normalise les sources
        sources = []
        for src in parsed.get("sources", []):
            if isinstance(src, dict):
                sources.append(src)

        transcript_used = any(s.get("type") == "transcript" for s in sources)

        logger.info(
            "qa_module: réponse générée | confidence=%.2f | sources=%d | transcript=%s",
            parsed["confidence"], len(sources), transcript_used,
        )

        return {
            "answer":          parsed["answer"],
            "sources":         sources,
            "confidence":      parsed["confidence"],
            "transcript_used": transcript_used,
            "fallback_used":   False,
        }

    except Exception as exc:
        logger.error("qa_module: erreur LLM — %s", exc)
        return _fallback_response(str(exc))


# ── Extraction des top_comments depuis le state A4 ────────────────────────────

def extract_top_comments(
    cleaned_comments: list[dict],
    discourse_result: dict[str, Any],
    threshold: float = 0.7,
    max_comments: int = _MAX_TOP_COMMENTS,
) -> list[dict]:
    """
    Extrait les commentaires de haute qualité pour le contexte Q&A.

    Stratégie en deux passes (PRD v1.1 §4.4) :
    1. Passe principale  : indices high_quality_indices retournés par A4
    2. Fallback          : si A4 n'en a sélectionné aucun, on prend les
                           max_comments commentaires avec le plus de likes,
                           en filtrant les textes trop courts (< 10 chars).

    Args:
        cleaned_comments : Liste de commentaires nettoyés par A2.
        discourse_result : Résultat de A4 contenant high_quality_indices.
        threshold        : Seuil score A4 (défaut 0.7).
        max_comments     : Nombre maximum de commentaires à conserver.

    Returns:
        Liste de {comment_id, text, score_a4}.
    """
    hq_indices = discourse_result.get("high_quality_indices") or []

    # ── Passe 1 : indices A4 ──────────────────────────────────────────────────
    if hq_indices:
        results = []
        for idx in hq_indices[:max_comments]:
            if isinstance(idx, int) and 0 <= idx < len(cleaned_comments):
                c    = cleaned_comments[idx]
                text = c.get("cleaned_text") or c.get("text") or ""
                if text.strip():
                    results.append({
                        "comment_id": c.get("comment_id", f"idx_{idx}"),
                        "text":       text[:500],
                        "score_a4":   threshold,
                    })
        if results:
            return results

    # ── Passe 2 : fallback sur les commentaires les plus likés ────────────────
    candidates = [
        c for c in cleaned_comments
        if isinstance(c, dict)
        and len((c.get("cleaned_text") or c.get("text") or "").strip()) >= 10
    ]
    candidates.sort(
        key=lambda c: int(c.get("author_likes") or 0),
        reverse=True,
    )
    results = []
    for c in candidates[:max_comments]:
        text = c.get("cleaned_text") or c.get("text") or ""
        results.append({
            "comment_id": c.get("comment_id", ""),
            "text":       text[:500],
            "score_a4":   0.0,   # sélectionné par popularité, pas par A4
        })

    return results
