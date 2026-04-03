from __future__ import annotations

import operator
from typing import Any, Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph

from core.llm_factory import build_chat_llm
from core.llm_schemas import DiscourseResult, NoiseResult, SentimentResult, SynthesisResult
from core.scoring.quality_levels import QualityLevel
from core.scoring.scorer import Scorer
from core.config import Config
from utils.youtube_api import YouTubeAPI
from utils.language_detector import detect_language
from utils.text_cleaner import normalize_text


class PipelineState(TypedDict, total=False):
    video_ids: list[str]
    comments: list[Any]

    raw_comments: list[Any]
    cleaned_comments: list[Any]

    sentiment: Optional[dict[str, Any]]
    discourse: Optional[dict[str, Any]]
    noise: Optional[dict[str, Any]]

    synthesis: Optional[dict[str, Any]]
    final_output: Optional[dict[str, Any]]

    errors: Annotated[list[str], operator.add]


def _load_prompt(path: str, fallback: str) -> str:
    """
    Load a prompt template from a text file.

    If the file is missing or empty, fall back to the provided default string.

    The prompt can optionally contain the placeholders:
    - {{context}}             -> will be replaced by comments text
    - {{format_instructions}} -> will be replaced by parser instructions
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
    except FileNotFoundError:
        # Safe fallback: we use the inline default.
        pass
    return fallback


def _as_comment_text(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Common key names
        return item.get("text") or item.get("comment") or item.get("content")
    return None


def _extract_youtube_comment_text(item: Any) -> Optional[str]:
    """
    Extract a human-readable comment text from a YouTube API item.

    The expected structure is based on commentThreads -> snippet -> topLevelComment -> snippet.
    """
    if not isinstance(item, dict):
        return None

    snippet = item.get("snippet") or {}
    top_level = snippet.get("topLevelComment") or {}
    top_snippet = top_level.get("snippet") or {}

    # Prefer display text.
    return top_snippet.get("textDisplay") or top_snippet.get("textOriginal")


def _collect_comments_from_youtube(
    *, config: Config, video_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Collect YouTube comments for a batch of video ids.

    Returns:
      - raw_comments: list of dicts shaped with a `text` key
      - errors: list of error strings (best-effort collection)
    """
    api = YouTubeAPI(api_key=config.youtube_api_key or "")

    raw_comments: list[dict[str, Any]] = []
    errors: list[str] = []

    if not config.youtube_api_key:
        return [], ["Missing YOUTUBE_API_KEY for collector_node."]

    for video_id in video_ids:
        try:
            items = api.fetch_video_comments(
                video_id,
                max_results=config.youtube_max_results,
                max_pages=config.youtube_max_pages,
            )
        except Exception as e:
            errors.append(f"YouTube fetch failed for video_id={video_id}: {e}")
            continue

        for item in items:
            text = _extract_youtube_comment_text(item)
            if not text:
                continue

            snippet = item.get("snippet") or {}
            top_level = snippet.get("topLevelComment") or {}
            comment_meta = top_level.get("snippet") or {}
            comment_id = top_level.get("id")
            author = comment_meta.get("authorDisplayName")
            published_at = comment_meta.get("publishedAt")

            raw_comments.append(
                {
                    "text": text,
                    "video_id": video_id,
                    "comment_id": comment_id,
                    "author": author,
                    "published_at": published_at,
                }
            )

    return raw_comments, errors


def collector_node(state: PipelineState, config: Config) -> dict[str, Any]:
    """
    Collector node:
    - If `state["comments"]` is provided (by scripts or upstream), use it as best-effort `raw_comments`.
    - Else, if `state["video_ids"]` is provided, fetch YouTube comments and normalize them to dicts with a `text` key.

    This node is designed to be best-effort:
    - it records errors in `state["errors"]`
    - it continues processing other video_ids
    """

    # 1) Fallback: `comments` passed in already (useful for tests / Kaggle preloaded datasets).
    provided_comments = state.get("comments", []) or []
    if provided_comments:
        raw_comments: list[dict[str, Any]] = []
        for item in provided_comments:
            text = _as_comment_text(item)
            if text:
                raw_comments.append({"text": text})
        return {"raw_comments": raw_comments}

    # 2) Default: fetch from YouTube for each `video_id`.
    video_ids = state.get("video_ids", []) or []
    if not video_ids:
        return {
            "raw_comments": [],
            **_append_error(state, "collector_node: missing both `comments` and `video_ids`."),
        }

    raw_comments, errors = _collect_comments_from_youtube(config=config, video_ids=video_ids)
    updates: dict[str, Any] = {"raw_comments": raw_comments}
    if errors:
        # Return error delta as a list; LangGraph will merge using the reducer.
        updates["errors"] = errors
    return updates


def preprocessor_node(state: PipelineState, config: Config) -> dict[str, Any]:
    """
    Preprocess node (domain logic):
    - normalize text (via `normalize_text`)
    - filter too short/empty comments
    - deduplicate by exact normalized text
    - optionally attach a `language` field (best-effort, default fallback)

    This node remains deterministic and “state-first”:
    it only returns updates for `cleaned_comments` (and does not mutate the state directly).
    """
    # Small constants to keep prompts bounded and reduce LLM noise.
    MIN_COMMENT_CHARS = 3
    MAX_COMMENT_CHARS = 2000

    # Keep track of normalized texts to remove exact duplicates.
    seen_normalized: set[str] = set()

    cleaned: list[Any] = []
    for item in state.get("raw_comments", []) or []:
        text = _as_comment_text(item)
        if text is None:
            # Skip items that do not have any usable text.
            continue

        # Apply normalization early so filtering/dedup are consistent.
        norm = normalize_text(text)
        if norm is None:
            continue

        # Filter: empty/too short comments and overly long comments.
        # (We cap length to limit context expansion in later nodes.)
        if len(norm) < MIN_COMMENT_CHARS:
            continue
        if len(norm) > MAX_COMMENT_CHARS:
            norm = norm[:MAX_COMMENT_CHARS]

        # Deduplicate by exact normalized text.
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)

        # Preserve metadata when raw comment is a dict (YouTube collector returns dicts).
        if isinstance(item, dict):
            next_item = dict(item)
            next_item["cleaned_text"] = norm
            # Ensure `text` exists even if source used another key (best-effort).
            next_item.setdefault("text", text)
        else:
            next_item = {"text": text, "cleaned_text": norm}

        # Best-effort language detection:
        # - if `langdetect` is not installed, `detect_language` returns `config.default_language`.
        next_item["language"] = detect_language(text, default=config.default_language)

        cleaned.append(next_item)

    return {"cleaned_comments": cleaned}


def _build_context_text(state: PipelineState) -> str:
    pieces: list[str] = []
    for c in state.get("cleaned_comments", []) or []:
        if isinstance(c, dict):
            t = c.get("cleaned_text") or c.get("text")
        else:
            t = str(c)
        if t:
            pieces.append(t)
    return "\n".join(pieces[:30])  # cap to keep prompts bounded


def _append_error(state: PipelineState, msg: str) -> dict[str, Any]:
    # Only return the delta; LangGraph will merge using the reducer.
    return {"errors": [msg]}


def sentiment_node(state: PipelineState, config: Config) -> dict[str, Any]:
    llm = build_chat_llm(config)
    if llm is None:
        return {
            "sentiment": None,
            **_append_error(state, "LLM not configured (missing OPENAI_API_KEY/OPENAI_BASE_URL)."),
        }

    parser = PydanticOutputParser(pydantic_object=SentimentResult)
    format_instructions = parser.get_format_instructions()
    context_text = _build_context_text(state)

    system = "You are an analyst. Return strictly valid JSON following the schema."

    default_user_prompt = (
        "Analyze the overall sentiment expressed by the following comments.\n\n"
        "Comments:\n{{context}}\n\n"
        "Return sentiment_label and sentiment_score.\n"
        "{{format_instructions}}"
    )
    template = _load_prompt("src/prompts/sentiment_v1.txt", default_user_prompt)
    user = (
        template.replace("{{context}}", context_text)
        .replace("{{format_instructions}}", format_instructions)
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        parsed = parser.parse(resp.content)
        return {"sentiment": parsed.model_dump()}
    except Exception as e:
        return {
            "sentiment": None,
            **_append_error(state, f"Sentiment parsing/LLM failed: {e}"),
        }


def discourse_node(state: PipelineState, config: Config) -> dict[str, Any]:
    llm = build_chat_llm(config)
    if llm is None:
        return {
            "discourse": None,
            **_append_error(state, "LLM not configured (missing OPENAI_API_KEY/OPENAI_BASE_URL)."),
        }

    parser = PydanticOutputParser(pydantic_object=DiscourseResult)
    format_instructions = parser.get_format_instructions()
    context_text = _build_context_text(state)

    system = "You are an analyst. Return strictly valid JSON following the schema."

    default_user_prompt = (
        "Classify the dominant discourse intent in the following comments.\n\n"
        "Comments:\n{{context}}\n\n"
        "Return discourse_label and discourse_score.\n"
        "{{format_instructions}}"
    )
    template = _load_prompt("src/prompts/discourse_v1.txt", default_user_prompt)
    user = (
        template.replace("{{context}}", context_text)
        .replace("{{format_instructions}}", format_instructions)
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        parsed = parser.parse(resp.content)
        return {"discourse": parsed.model_dump()}
    except Exception as e:
        return {
            "discourse": None,
            **_append_error(state, f"Discourse parsing/LLM failed: {e}"),
        }


def noise_node(state: PipelineState, config: Config) -> dict[str, Any]:
    llm = build_chat_llm(config)
    if llm is None:
        return {
            "noise": None,
            **_append_error(state, "LLM not configured (missing OPENAI_API_KEY/OPENAI_BASE_URL)."),
        }

    parser = PydanticOutputParser(pydantic_object=NoiseResult)
    format_instructions = parser.get_format_instructions()
    context_text = _build_context_text(state)

    system = "You are an analyst. Return strictly valid JSON following the schema."

    default_user_prompt = (
        "Estimate how noisy/unclear the following comments are.\n\n"
        "Comments:\n{{context}}\n\n"
        "Return noise_level (0-1) and noise_label (low/medium/high).\n"
        "{{format_instructions}}"
    )
    template = _load_prompt("src/prompts/noise_detection_v1.txt", default_user_prompt)
    user = (
        template.replace("{{context}}", context_text)
        .replace("{{format_instructions}}", format_instructions)
    )

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        parsed = parser.parse(resp.content)
        return {"noise": parsed.model_dump()}
    except Exception as e:
        return {
            "noise": None,
            **_append_error(state, f"Noise parsing/LLM failed: {e}"),
        }


def synthesizer_node(state: PipelineState, config: Config) -> dict[str, Any]:
    sentiment_data = state.get("sentiment") or {}
    discourse_data = state.get("discourse") or {}
    noise_data = state.get("noise") or {}

    sentiment_score = float(sentiment_data.get("sentiment_score", 0.0) or 0.0)
    discourse_score = float(discourse_data.get("discourse_score", 0.0) or 0.0)
    noise_level = float(noise_data.get("noise_level", 0.0) or 0.0)

    # Treat "noise quality" as the inverse of noise_level.
    noise_quality = max(0.0, min(1.0, 1.0 - noise_level))

    scorer = Scorer()
    overall_score = scorer.compute(
        {"sentiment": sentiment_score, "discourse": discourse_score, "noise": noise_quality}
    )
    overall_score = max(0.0, min(1.0, overall_score))

    if overall_score < 0.25:
        quality_level = QualityLevel.low.value
    elif overall_score < 0.5:
        quality_level = QualityLevel.medium.value
    elif overall_score < 0.75:
        quality_level = QualityLevel.good.value
    else:
        quality_level = QualityLevel.excellent.value

    sentiment_res = SentimentResult(
        sentiment_label=sentiment_data.get("sentiment_label", "neutral"),
        sentiment_score=sentiment_score,
        rationale=sentiment_data.get("rationale"),
    )
    discourse_res = DiscourseResult(
        discourse_label=discourse_data.get("discourse_label", "other"),
        discourse_score=discourse_score,
        rationale=discourse_data.get("rationale"),
    )
    noise_res = NoiseResult(
        noise_level=noise_level,
        noise_label=noise_data.get("noise_label", "low"),
        rationale=noise_data.get("rationale"),
    )

    synthesis = SynthesisResult(
        sentiment=sentiment_res,
        discourse=discourse_res,
        noise=noise_res,
        overall_score=overall_score,
        quality_level=quality_level,
        summary=None,
    )

    payload = synthesis.model_dump()
    return {"synthesis": payload, "final_output": payload}


def build_langgraph_app(*, config: Optional[Config] = None, checkpointer: Any) -> Any:
    """
    Build and compile a LangGraph app.

    Note: `checkpointer` is typically MemorySaver (dev) or SqliteSaver (prod).
    """

    resolved_config = config or Config.from_env()

    builder: StateGraph[PipelineState] = StateGraph(PipelineState)

    # Collector / preprocessor are synchronous and deterministic.
    builder.add_node("collector", lambda state: collector_node(state, resolved_config))
    builder.add_node("preprocessor", lambda state: preprocessor_node(state, resolved_config))

    # LLM nodes: we wrap to inject Config.
    builder.add_node(
        "sentiment",
        lambda state: sentiment_node(state, resolved_config),
    )
    builder.add_node(
        "discourse",
        lambda state: discourse_node(state, resolved_config),
    )
    builder.add_node(
        "noise",
        lambda state: noise_node(state, resolved_config),
    )

    builder.add_node(
        "synthesizer",
        lambda state: synthesizer_node(state, resolved_config),
    )

    builder.add_edge(START, "collector")
    builder.add_edge("collector", "preprocessor")

    # Parallel branches
    builder.add_edge("preprocessor", "sentiment")
    builder.add_edge("preprocessor", "discourse")
    builder.add_edge("preprocessor", "noise")

    builder.add_edge("sentiment", "synthesizer")
    builder.add_edge("discourse", "synthesizer")
    builder.add_edge("noise", "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)

