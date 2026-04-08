"""
api/quiz.py — Module QCM (Quiz a Choix Multiples)
==================================================
Genere des questions pedagogiques ancrees sur le contenu d'une video YouTube.

Hierarchie des sources (par qualite decroissante) :
  1. Transcription horodatee   — qualite maximale, questions precises et verifiables
  2. Description video         — contexte general, questions thematiques
  3. Commentaires haute qualite A4 — perspectives externes, questions de synthese

Anti-hallucination : chaque question cite son source_segment verbatim.
Mode degrade     : sans transcript, n_questions <= 3, mode="degraded" signale.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from utils.logger import get_logger

logger = get_logger("quiz_module")

# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_TRANSCRIPT_CHARS  = 12_000  # ~3 000 tokens — couvre ~15 min de video
_MAX_DESCRIPTION_CHARS = 800
_MAX_COMMENTS_FOR_QUIZ = 15      # top_comments injectes dans le prompt
_DEFAULT_N_QUESTIONS   = 5
_DEGRADED_MAX_Q        = 3       # max sans transcript

_SYSTEM_PROMPT = """\
Tu es un expert en pedagogie specialise dans la creation de quiz educatifs.
Tu generes des questions a choix multiples (QCM) UNIQUEMENT a partir du contenu fourni.
Ne fabrique aucune information absente du contexte.

REGLES STRICTES :
1. Chaque question doit etre ancree sur un passage identifiable du contenu.
2. Exactement 4 options par question (indices 0 a 3).
3. Une seule bonne reponse, choisie parmi les 4 options.
4. Les 3 distracteurs doivent etre plausibles mais incorrects selon le contexte fourni.
5. Le champ "explanation" doit justifier pourquoi la bonne reponse est correcte
   ET pourquoi chaque distractor est incorrect.
6. Le champ "source" doit contenir un extrait verbatim du contenu utilise.
7. Varie la difficulte : environ 2 faciles, 2 moyennes, 1 difficile pour 5 questions.

Retourne UNIQUEMENT un objet JSON valide, sans texte avant ou apres."""


# ── Construction du prompt ────────────────────────────────────────────────────

def _truncate_transcript(transcript: list[dict], max_chars: int) -> str:
    """Construit le bloc transcription horodate, tronque si necessaire."""
    lines = []
    for seg in transcript:
        start = seg.get("start", 0.0)
        text  = seg.get("text", "").strip()
        if not text:
            continue
        m, s = int(start // 60), int(start % 60)
        lines.append(f"[{m:02d}:{s:02d}] {text}")

    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full

    # Conserver debut (intro) + fin (conclusion) — zones les plus riches
    half = max_chars // 2
    return full[:half] + "\n[...contenu tronque...]\n" + full[-half:]


def _build_quiz_prompt(
    video_title:          str,
    transcript_ctx:       str,
    description_ctx:      str,
    comments_ctx:         str,
    n_questions:          int,
    transcript_available: bool,
) -> str:
    """Assemble le prompt utilisateur avec toutes les sections de contexte."""
    parts = []

    if video_title:
        parts.append(f"VIDEO : {video_title}\n")

    if description_ctx:
        parts.append(f"DESCRIPTION :\n{description_ctx}\n")

    if transcript_available and transcript_ctx:
        parts.append(f"TRANSCRIPTION (horodatee) :\n{transcript_ctx}\n")
    else:
        parts.append("TRANSCRIPTION : Non disponible pour cette video.\n")

    if comments_ctx:
        parts.append(f"COMMENTAIRES SELECTIONNES :\n{comments_ctx}\n")

    # Schema d'exemple pour guider la generation
    example = {
        "questions": [
            {
                "question": "Selon l'auteur, quelle est la principale raison de... ?",
                "options": [
                    "Premiere option plausible",
                    "Deuxieme option correcte",
                    "Troisieme option plausible",
                    "Quatrieme option plausible",
                ],
                "correct": 1,
                "explanation": (
                    "L'option 1 est correcte car [citation du contenu]. "
                    "L'option 0 est incorrecte car... "
                    "L'option 2 est incorrecte car... "
                    "L'option 3 est incorrecte car..."
                ),
                "source": {
                    "start": 142.5,
                    "text": "extrait verbatim du contenu source utilise",
                    "source_type": "transcript",
                },
                "difficulty": "medium",
            }
        ]
    }

    parts.append(
        f"CONSIGNE : Genere exactement {n_questions} question(s) QCM pedagogiques "
        f"sur le contenu ci-dessus.\n"
        f"Format JSON attendu (respecte exactement cette structure) :\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )

    return "\n".join(parts)


# ── Parsing et validation ─────────────────────────────────────────────────────

def _normalize_question(q: dict) -> dict:
    """
    Normalise une question brute du LLM :
    - Convertit correct de 1-4 vers 0-3 si necessaire
    - Ajoute les champs manquants avec valeurs par defaut
    - Nettoie les prefixes A/B/C/D dans les options
    """
    q = dict(q)  # copie

    # Normalisation de l'index correct (LLM renvoie parfois 1-4 au lieu de 0-3)
    correct = q.get("correct", 0)
    if isinstance(correct, int) and correct >= 4:
        q["correct"] = correct - 1

    # Nettoyage des prefixes "A. ", "B) ", etc. dans les options
    options = q.get("options", [])
    cleaned_options = []
    for opt in options:
        if isinstance(opt, str):
            opt = re.sub(r"^[A-Da-d][.):\s]+", "", opt).strip()
        cleaned_options.append(opt)
    q["options"] = cleaned_options

    # Champs optionnels avec valeurs par defaut
    q.setdefault("difficulty", "medium")
    q.setdefault("explanation", "")
    if "source" not in q:
        q["source"] = None

    return q


def _validate_question(q: dict) -> bool:
    """Verifie la structure minimale d'une question."""
    if not isinstance(q.get("question"), str) or not q["question"].strip():
        return False
    options = q.get("options", [])
    if not isinstance(options, list) or len(options) != 4:
        return False
    if not all(isinstance(o, str) and o.strip() for o in options):
        return False
    correct = q.get("correct")
    if not isinstance(correct, int) or not (0 <= correct <= 3):
        return False
    if not isinstance(q.get("explanation"), str):
        return False
    return True


def _parse_quiz_response(raw: str, n_requested: int) -> list[dict]:
    """
    Extrait les questions depuis la reponse JSON du LLM.
    Fallback progressif si la reponse est malformee.
    """
    text = raw.strip()
    # Retirer les blocs code markdown eventuels
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Tentative directe
    try:
        data = json.loads(text)
        questions = data.get("questions", [])
        if isinstance(questions, list):
            return [_normalize_question(q) for q in questions[:n_requested]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Extraction du tableau "questions" par regex
    match = re.search(r'"questions"\s*:\s*(\[.*?\])\s*[,}]', text, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group(1))
            return [_normalize_question(q) for q in questions[:n_requested]]
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("quiz_module: impossible de parser la reponse LLM — raw=%r", raw[:200])
    return []


# ── Reponse vide (erreur / fallback) ─────────────────────────────────────────

def _empty_quiz(
    video_title:      str,
    mode:             str,
    transcript_used:  bool,
    error:            str,
) -> dict[str, Any]:
    return {
        "video_title":     video_title,
        "questions":       [],
        "n_questions":     0,
        "mode":            mode,
        "transcript_used": transcript_used,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "error":           error,
    }


# ── Fonction principale ───────────────────────────────────────────────────────

def generate_quiz(
    qa_context:  dict[str, Any],
    n_questions: int = _DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """
    Genere un QCM ancre sur le contenu de la video.

    Args:
        qa_context   : Contexte cache {transcript, transcript_available,
                        top_comments, video_title, video_description}.
        n_questions  : Nombre de questions souhaitees (defaut : 5).

    Returns:
        {video_title, questions, n_questions, mode, transcript_used,
         generated_at, error}
    """
    transcript           = qa_context.get("transcript") or []
    transcript_available = qa_context.get("transcript_available", False)
    top_comments         = qa_context.get("top_comments") or []
    video_title          = qa_context.get("video_title", "")
    video_description    = qa_context.get("video_description", "")

    # Mode degrade sans transcript
    if not transcript_available:
        n_questions = min(n_questions, _DEGRADED_MAX_Q)
        mode = "degraded"
        logger.info(
            "quiz_module: mode degrade (pas de transcript) — %d questions max",
            n_questions,
        )
    else:
        mode = "full"

    # Construction des blocs de contexte
    transcript_ctx  = _truncate_transcript(transcript, _MAX_TRANSCRIPT_CHARS)
    description_ctx = (video_description or "").strip()[:_MAX_DESCRIPTION_CHARS]
    comments_ctx    = "\n".join(
        f"- {c.get('text', '')[:200]}"
        for c in top_comments[:_MAX_COMMENTS_FOR_QUIZ]
        if c.get("text", "").strip()
    )

    user_prompt = _build_quiz_prompt(
        video_title=video_title,
        transcript_ctx=transcript_ctx,
        description_ctx=description_ctx,
        comments_ctx=comments_ctx,
        n_questions=n_questions,
        transcript_available=transcript_available,
    )

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    try:
        from models.llm_loader import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm()
        if llm is None:
            logger.warning("quiz_module: LLM indisponible")
            return _empty_quiz(video_title, mode, transcript_available, "llm_unavailable")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        resp = llm.invoke(messages)
        raw  = resp.content if hasattr(resp, "content") else str(resp)

        raw_questions   = _parse_quiz_response(raw, n_questions)
        valid_questions = [q for q in raw_questions if _validate_question(q)]

        if not valid_questions:
            logger.warning("quiz_module: aucune question valide generee")
            return _empty_quiz(video_title, mode, transcript_available, "no_valid_questions")

        logger.info(
            "quiz_module: %d/%d questions valides | mode=%s | transcript=%s",
            len(valid_questions), n_questions, mode, transcript_available,
        )

        return {
            "video_title":     video_title,
            "questions":       valid_questions,
            "n_questions":     len(valid_questions),
            "mode":            mode,
            "transcript_used": transcript_available,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "error":           None,
        }

    except Exception as exc:
        logger.error("quiz_module: erreur generation — %s", exc)
        return _empty_quiz(video_title, mode, transcript_available, str(exc))
