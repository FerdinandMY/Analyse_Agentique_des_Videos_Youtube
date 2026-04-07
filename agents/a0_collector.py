"""
A0 — Collector  [PRD v1.1 — Collecte Hybride]
==============================================
Responsabilités :
  1. Extraire le video_id depuis une URL YouTube (4 formats — FR-74)
  2. Émettre videos.list + commentThreads.list en parallèle (FR-75)
  3. Paginer jusqu'à 500 commentaires via nextPageToken (FR-76)
  4. Récupérer la transcription via youtube-transcript-api (FR-84)
  5. Produire raw_comments conforme au schéma attendu par A1 (FR-77)
  6. Fallback CSV pré-collecté si quota dépassé (FR-80)
  7. Écrire source / quota_used / transcript_available dans le state (FR-81/85)

Coût quota YouTube Data API v3 :
  - videos.list        : 1 unité
  - commentThreads.list: 2 unités × N pages (max 5 pages = 10 unités)
  - Transcription      : 0 unité (sous-titres publics)
  Total max : ~11 unités par analyse.

Clé API chargée depuis :
  1. Variable d'environnement  YOUTUBE_API_KEY
  2. Kaggle Secrets             YOUTUBE_API_KEY (si disponible)
"""
from __future__ import annotations

import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger("a0_collector")

# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_COMMENTS   = 500
_MAX_RESULTS    = 100     # par page YouTube API
_MAX_PAGES      = 5
_API_TIMEOUT    = 15      # secondes
_API_RETRIES    = 2

_URL_RE = re.compile(
    r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_TRANSCRIPT_LANGS = ["fr", "en", "fr-CA", "fr-BE", "fr-CH"]

_FALLBACK_CSV = "data/raw/comments_raw.csv"

# ── Chargement de la clé API ──────────────────────────────────────────────────

def _load_api_key() -> str:
    """
    Ordre de priorité :
    1. Variable d'environnement YOUTUBE_API_KEY (ou .env via python-dotenv)
    2. Kaggle Secrets (si disponible)
    """
    # Charger .env si présent (développement local)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(override=False)     # ne remplace pas les variables déjà définies
    except ImportError:
        pass

    key = os.environ.get("YOUTUBE_API_KEY", "")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        key = UserSecretsClient().get_secret("YOUTUBE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return ""


# ── Extraction du video_id ────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    """
    Extrait le video_id depuis une URL YouTube ou un ID direct (FR-74).

    Formats supportés :
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://www.youtube.com/shorts/dQw4w9WgXcQ
      - dQw4w9WgXcQ  (ID direct)

    Returns:
        video_id de 11 caractères.

    Raises:
        ValueError: Si aucun video_id valide n'est trouvé.
    """
    s = url_or_id.strip()
    # ID direct
    if _ID_RE.match(s):
        return s
    # URL
    m = _URL_RE.search(s)
    if m:
        return m.group(1)
    raise ValueError(f"Impossible d'extraire un video_id depuis : {url_or_id!r}")


# ── Appels YouTube Data API v3 ────────────────────────────────────────────────

def _build_service(api_key: str):
    """Construit le client googleapiclient."""
    from googleapiclient.discovery import build  # type: ignore
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _fetch_video_metadata(service, video_id: str) -> dict[str, Any]:
    """
    GET videos.list — coût : 1 unité quota (FR-75).
    Retourne les métadonnées de la vidéo.
    """
    resp = (
        service.videos()
        .list(id=video_id, part="snippet,statistics,contentDetails")
        .execute()
    )
    items = resp.get("items", [])
    if not items:
        raise ValueError(f"VIDEO_NOT_FOUND: {video_id}")

    item    = items[0]
    snippet = item.get("snippet", {})
    stats   = item.get("statistics", {})

    return {
        "video_id":        video_id,
        "title":           snippet.get("title", ""),
        "description":     snippet.get("description", "")[:2000],  # tronqué à 2000 chars
        "channel_name":    snippet.get("channelTitle", ""),
        "channel_id":      snippet.get("channelId", ""),
        "published_at":    snippet.get("publishedAt", ""),
        "view_count":      int(stats.get("viewCount", 0) or 0),
        "like_count":      int(stats.get("likeCount", 0) or 0),
        "comment_count":   int(stats.get("commentCount", 0) or 0),
    }


def _fetch_comments(service, video_id: str) -> tuple[list[dict], int]:
    """
    GET commentThreads.list avec pagination (FR-76).
    Coût : 2 unités × N pages.

    Returns:
        (comments_rows, quota_used)
    """
    comments: list[dict] = []
    next_page_token: Optional[str] = None
    pages = 0
    quota_used = 0

    while pages < _MAX_PAGES and len(comments) < _MAX_COMMENTS:
        kwargs: dict[str, Any] = {
            "videoId":    video_id,
            "part":       "snippet,replies",
            "maxResults": _MAX_RESULTS,
            "order":      "relevance",
        }
        if next_page_token:
            kwargs["pageToken"] = next_page_token

        try:
            resp = service.commentThreads().list(**kwargs).execute()
        except Exception as exc:
            err_str = str(exc)
            if "commentsDisabled" in err_str:
                raise ValueError("COMMENTS_DISABLED")
            if "quotaExceeded" in err_str or "403" in err_str:
                raise PermissionError("QUOTA_EXCEEDED")
            raise

        quota_used += 2
        pages += 1

        for thread in resp.get("items", []):
            if len(comments) >= _MAX_COMMENTS:
                break
            top = thread.get("snippet", {}).get("topLevelComment", {})
            top_snip = top.get("snippet", {})
            comments.append({
                "video_id":          video_id,
                "comment_id":        top.get("id", ""),
                "text":              top_snip.get("textOriginal", ""),
                "author":            top_snip.get("authorDisplayName", ""),
                "author_likes":      int(top_snip.get("likeCount", 0) or 0),
                "reply_count":       int(thread.get("snippet", {}).get("totalReplyCount", 0) or 0),
                "published_at":      top_snip.get("publishedAt", ""),
                "is_reply":          False,
                "parent_id":         "",
                "language_detected": "",
            })

            # Aplatir les replies
            for reply in thread.get("replies", {}).get("comments", []):
                if len(comments) >= _MAX_COMMENTS:
                    break
                r_snip = reply.get("snippet", {})
                comments.append({
                    "video_id":          video_id,
                    "comment_id":        reply.get("id", ""),
                    "text":              r_snip.get("textOriginal", ""),
                    "author":            r_snip.get("authorDisplayName", ""),
                    "author_likes":      int(r_snip.get("likeCount", 0) or 0),
                    "reply_count":       0,
                    "published_at":      r_snip.get("publishedAt", ""),
                    "is_reply":          True,
                    "parent_id":         top.get("id", ""),
                    "language_detected": "",
                })

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return comments, quota_used


# ── Transcription ─────────────────────────────────────────────────────────────

def _fetch_transcript(video_id: str) -> tuple[list[dict], bool]:
    """
    Récupère la transcription via youtube-transcript-api (FR-84).
    Ne consomme aucune unité de quota.

    Stratégie en 3 passes (couvre ~95% des vidéos) :
      1. Sous-titres manuels dans _TRANSCRIPT_LANGS (créateur les a activés)
      2. Sous-titres auto-générés par YouTube dans _TRANSCRIPT_LANGS
      3. N'importe quelle langue disponible, traduite en français via l'API

    Returns:
        (transcript_segments, available)
        transcript_segments : [{text, start, duration}]
        available           : False si aucune source disponible
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

        api = YouTubeTranscriptApi()

        # Lister toutes les transcriptions disponibles pour cette vidéo
        try:
            transcript_list = api.list(video_id)
        except Exception as exc:
            logger.warning("a0_collector: list transcripts indisponible %s — %s", video_id, exc)
            return [], False

        fetched = None

        # Passe 1 — sous-titres manuels (meilleure qualité)
        try:
            fetched = transcript_list.find_manually_created_transcript(_TRANSCRIPT_LANGS)
            logger.info("a0_collector: transcription manuelle trouvee pour %s", video_id)
        except Exception:
            pass

        # Passe 2 — auto-captions YouTube (génères automatiquement)
        if fetched is None:
            try:
                fetched = transcript_list.find_generated_transcript(_TRANSCRIPT_LANGS)
                logger.info("a0_collector: auto-caption trouvee pour %s", video_id)
            except Exception:
                pass

        # Passe 3 — n'importe quelle langue disponible, traduite en français
        if fetched is None:
            try:
                available = list(transcript_list)
                if available:
                    # Prendre le premier disponible et traduire en français
                    candidate = available[0]
                    if candidate.is_translatable:
                        fetched = candidate.translate("fr")
                        logger.info(
                            "a0_collector: transcription %s traduite en fr pour %s",
                            candidate.language_code, video_id,
                        )
                    else:
                        fetched = candidate
                        logger.info(
                            "a0_collector: transcription %s (sans traduction) pour %s",
                            candidate.language_code, video_id,
                        )
            except Exception:
                pass

        if fetched is None:
            logger.warning("a0_collector: aucune transcription disponible pour %s", video_id)
            return [], False

        # Récupérer les segments horodatés
        data = fetched.fetch()
        segments = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in data
        ]
        return segments, True

    except Exception as exc:
        logger.warning("a0_collector: transcription indisponible pour %s — %s", video_id, exc)
        return [], False


# ── Fallback CSV pré-collecté ─────────────────────────────────────────────────

def _load_from_csv_fallback(video_id: str, csv_path: str = _FALLBACK_CSV) -> list[dict]:
    """
    Charge les commentaires depuis le CSV pré-collecté si le video_id est présent (FR-80).
    Retourne une liste vide si le video_id est absent du CSV.
    """
    p = Path(csv_path)
    if not p.exists():
        return []
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(p, dtype=str)
        # Harmonise les noms de colonnes
        df = df.rename(columns={
            "texte_commentaire": "text",
            "commentaire_id":    "comment_id",
            "nb_likes_commentaire": "author_likes",
            "nb_reponses":          "reply_count",
            "publie_le":            "published_at",
        })
        if "video_id" not in df.columns or "text" not in df.columns:
            return []
        subset = df[df["video_id"] == video_id].copy()
        if subset.empty:
            return []
        rows = []
        for _, row in subset.iterrows():
            rows.append({
                "video_id":          video_id,
                "comment_id":        str(row.get("comment_id", "")),
                "text":              str(row.get("text", "")),
                "author":            str(row.get("author", "")),
                "author_likes":      int(row.get("author_likes", 0) or 0),
                "reply_count":       int(row.get("reply_count", 0) or 0),
                "published_at":      str(row.get("published_at", "")),
                "is_reply":          False,
                "parent_id":         "",
                "language_detected": "",
            })
        logger.info("a0_collector: CSV fallback — %d commentaires pour %s", len(rows), video_id)
        return rows
    except Exception as exc:
        logger.error("a0_collector: erreur CSV fallback — %s", exc)
        return []


# ── Agent principal ───────────────────────────────────────────────────────────

def a0_collector(state: dict) -> dict[str, Any]:
    """
    LangGraph node — A0 Collector (PRD v1.1).

    Lit depuis le state :
        url_or_id : str  — URL YouTube ou video_id direct
        topic     : str  — thématique utilisateur (transmis tel quel à A1)

    Écrit dans le state :
        raw_comments        : list[dict]  — commentaires conformes schéma A1
        video_id            : str
        source              : str         — 'api_v3' | 'csv_fallback'
        quota_used          : int
        collected_at        : str         — ISO8601
        transcript          : list[dict]  — [{text, start, duration}]
        transcript_available: bool
        errors              : list[str]
    """
    url_or_id = state.get("url_or_id") or state.get("video_id", "")
    errors: list[str] = []

    # ── 1. Extraction du video_id ─────────────────────────────────────────────
    try:
        video_id = extract_video_id(url_or_id)
    except ValueError as exc:
        logger.error("a0_collector: %s", exc)
        return {"errors": [str(exc)]}

    logger.info("a0_collector: video_id=%s", video_id)

    # ── 2. Chargement clé API ─────────────────────────────────────────────────
    api_key = _load_api_key()

    collected_at = datetime.now(timezone.utc).isoformat()
    quota_used   = 0
    source       = "api_v3"
    comments: list[dict] = []
    metadata: dict[str, Any] = {}

    # ── 3. Appels API en parallèle (videos.list + commentThreads.list) ────────
    if api_key:
        try:
            service = _build_service(api_key)

            # Lancement des deux appels en parallèle (FR-75)
            # On récupère les futures DANS le bloc with pour garder les timeouts actifs
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_meta     = pool.submit(_fetch_video_metadata, service, video_id)
                fut_comments = pool.submit(_fetch_comments,       service, video_id)

                # Métadonnées
                try:
                    metadata = fut_meta.result(timeout=_API_TIMEOUT)
                    quota_used += 1
                except ValueError as exc:
                    err = str(exc)
                    logger.error("a0_collector: %s", err)
                    pool.shutdown(wait=False, cancel_futures=True)
                    return {"errors": [err], "video_id": video_id}
                except Exception as exc:
                    logger.warning("a0_collector: metadata error — %s", exc)
                    errors.append(f"metadata_error: {exc}")

                # Commentaires
                try:
                    comments, q = fut_comments.result(timeout=_API_TIMEOUT * _MAX_PAGES)
                    quota_used += q
                except ValueError as exc:
                    err = str(exc)
                    if err == "COMMENTS_DISABLED":
                        logger.error("a0_collector: commentaires désactivés pour %s", video_id)
                        pool.shutdown(wait=False, cancel_futures=True)
                        return {"errors": [err], "video_id": video_id}
                    raise
                except PermissionError:
                    logger.warning("a0_collector: quota dépassé — tentative fallback CSV")
                    source   = "csv_fallback"
                    comments = _load_from_csv_fallback(video_id)
                except Exception as exc:
                    logger.warning("a0_collector: comments error — %s — fallback CSV", exc)
                    source   = "csv_fallback"
                    comments = _load_from_csv_fallback(video_id)
                    errors.append(f"comments_api_error: {exc}")

        except Exception as exc:
            logger.warning("a0_collector: API indisponible — %s — fallback CSV", exc)
            source   = "csv_fallback"
            comments = _load_from_csv_fallback(video_id)
            errors.append(f"api_error: {exc}")
    else:
        logger.warning("a0_collector: YOUTUBE_API_KEY absente — fallback CSV")
        source   = "csv_fallback"
        comments = _load_from_csv_fallback(video_id)
        errors.append("YOUTUBE_API_KEY_MISSING — fallback CSV")

    # ── 4. Vérification seuil minimum de commentaires (§5.3) ─────────────────
    _MIN_COMMENTS = int(os.environ.get("A0_MIN_COMMENTS", "5"))
    if len(comments) < _MIN_COMMENTS:
        err = f"INSUFFICIENT_COMMENTS: found={len(comments)} minimum={_MIN_COMMENTS}"
        logger.error("a0_collector: %s", err)
        return {"errors": [err], "video_id": video_id, "source": source}

    # ── 5. Récupération de la transcription (FR-84) ───────────────────────────
    transcript, transcript_available = _fetch_transcript(video_id)

    logger.info(
        "a0_collector: %d commentaires collectés | source=%s | quota=%d | transcript=%s",
        len(comments), source, quota_used, transcript_available,
    )

    return {
        "video_id":             video_id,
        "raw_comments":         comments,
        "source":               source,
        "quota_used":           quota_used,
        "collected_at":         collected_at,
        "transcript":           transcript,
        "transcript_available": transcript_available,
        "video_title":          metadata.get("title", ""),
        "video_description":    metadata.get("description", ""),
        "errors":               errors,
    }
