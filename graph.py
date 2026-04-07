"""
graph.py — LangGraph Pipeline Definition
==========================================
Wires the 7 agents into a LangGraph StateGraph:

    A1 (Loader) → A2 (Preprocessor) → [ A3 ‖ A4 ‖ A5 ] → A6 (Synthesizer) → A7 (Topic Matcher)

A3, A4, A5 run in parallel (fan-out from A2, fan-in to A6).

v1.1 : _assemble_report propage hallucination_flags, fallback_used, sc_consensus,
        low_consensus, source, quota_used, collected_at (bug fix + champs A0).
"""
from __future__ import annotations

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.a1_loader import a1_loader
from agents.a2_preprocessor import a2_preprocessor
from agents.a3_sentiment import a3_sentiment
from agents.a4_discourse import a4_discourse
from agents.a5_noise import a5_noise
from agents.a6_synthesizer import a6_synthesizer
from agents.a7_topic_matcher import a7_topic_matcher
from pipeline_state import PipelineState


def _assemble_report(state: PipelineState) -> dict[str, Any]:
    """Final node — assembles the complete report dict from state."""
    synthesis = state.get("synthesis") or {}
    report = {
        "video_id":           state.get("video_id"),
        "topic":              state.get("topic"),
        "lang":               state.get("lang", "fr"),
        "score_global":       state.get("score_global"),
        "score_pertinence":   state.get("score_pertinence"),
        "score_final":        state.get("score_final"),
        "quality_label":      synthesis.get("quality_label"),
        "topic_verdict":      state.get("topic_verdict"),
        "summary":            synthesis.get("summary"),
        "details": {
            "sentiment": state.get("sentiment"),
            "discourse": state.get("discourse"),
            "noise":     state.get("noise"),
        },
        "comment_count":       len(state.get("cleaned_comments") or []),
        "cleaned_comments":    state.get("cleaned_comments") or [],
        "errors":              state.get("errors") or [],
        # ── v3.0 anti-hallucination (bug fix : ces champs étaient absents) ──────
        "hallucination_flags": state.get("hallucination_flags") or [],
        "fallback_used":       bool(state.get("fallback_used", False)),
        "sc_consensus":        state.get("sc_consensus"),
        "low_consensus":       state.get("low_consensus"),
        # ── v1.1 A0 Collector ────────────────────────────────────────────────────
        "source":              state.get("source"),
        "quota_used":          state.get("quota_used"),
        "collected_at":        state.get("collected_at"),
        "video_title":         state.get("video_title", ""),
        "video_description":   state.get("video_description", ""),
    }
    return {"report": report}


def build_graph(checkpointer: Any = None) -> Any:
    """
    Build and compile the LangGraph pipeline (A1 → A7).

    A0 n'est pas un nœud du graphe LangGraph — il est invoqué directement
    par api/routes.py avant d'appeler run_pipeline(), afin de garder
    A0 découplé du graphe et testable indépendamment (PRD v1.1 §2).

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver by default).

    Returns:
        Compiled LangGraph app.
    """
    builder: StateGraph[PipelineState] = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("a1_loader",         a1_loader)
    builder.add_node("a2_preprocessor",   a2_preprocessor)
    builder.add_node("a3_sentiment",      a3_sentiment)
    builder.add_node("a4_discourse",      a4_discourse)
    builder.add_node("a5_noise",          a5_noise)
    builder.add_node("a6_synthesizer",    a6_synthesizer)
    builder.add_node("a7_topic_matcher",  a7_topic_matcher)
    builder.add_node("assemble_report",   _assemble_report)

    # Sequential backbone
    builder.add_edge(START,          "a1_loader")
    builder.add_edge("a1_loader",    "a2_preprocessor")

    # Parallel fan-out: A2 → {A3, A4, A5}
    builder.add_edge("a2_preprocessor", "a3_sentiment")
    builder.add_edge("a2_preprocessor", "a4_discourse")
    builder.add_edge("a2_preprocessor", "a5_noise")

    # Fan-in: {A3, A4, A5} → A6
    builder.add_edge("a3_sentiment", "a6_synthesizer")
    builder.add_edge("a4_discourse", "a6_synthesizer")
    builder.add_edge("a5_noise",     "a6_synthesizer")

    # Sequential tail: A6 → A7 → assemble → END
    builder.add_edge("a6_synthesizer",   "a7_topic_matcher")
    builder.add_edge("a7_topic_matcher", "assemble_report")
    builder.add_edge("assemble_report",  END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=resolved_checkpointer)


def run_pipeline(
    *,
    csv_path:     str = "",
    video_id:     str = "",
    topic:        str = "",
    lang:         str = "fr",
    raw_comments: Optional[list] = None,
    thread_id:    str = "default",
    checkpointer: Any = None,
    # v1.1 — champs A0 transmis directement dans le state initial
    source:              Optional[str] = None,
    quota_used:          Optional[int] = None,
    collected_at:        Optional[str] = None,
    transcript:          Optional[list] = None,
    transcript_available: Optional[bool] = None,
    video_title:         Optional[str] = None,
    video_description:   Optional[str] = None,
) -> dict[str, Any]:
    """
    Convenience function: build graph, run pipeline, return report.

    Args:
        csv_path:     Path to pre-collected CSV (A1 input).
        video_id:     YouTube video ID (metadata, forwarded to report).
        topic:        User thematic query (A7 input).
        raw_comments: Pre-loaded comments list (bypasses A1 CSV loading).
        thread_id:    LangGraph thread identifier for checkpointing.
        checkpointer: Optional custom checkpointer.
        source:       Origine des données — 'api_v3' | 'csv_fallback' | 'pre_loaded'.
        quota_used:   Unités quota YouTube consommées par A0.
        collected_at: Timestamp ISO8601 de la collecte A0.
        transcript:   Segments de transcription [{text, start, duration}].
        transcript_available: False si sous-titres indisponibles.

    Returns:
        The final report dict (or partial state on error).
    """
    app = build_graph(checkpointer)

    initial_state: PipelineState = {
        "csv_path": csv_path,
        "video_id": video_id,
        "topic":    topic,
        "lang":     lang,
        "errors":   [],
    }

    if raw_comments is not None:
        initial_state["raw_comments"] = raw_comments  # type: ignore[assignment]

    # Champs A0 optionnels
    if source              is not None: initial_state["source"]               = source               # type: ignore[assignment]
    if quota_used          is not None: initial_state["quota_used"]           = quota_used           # type: ignore[assignment]
    if collected_at        is not None: initial_state["collected_at"]         = collected_at         # type: ignore[assignment]
    if transcript          is not None: initial_state["transcript"]           = transcript           # type: ignore[assignment]
    if transcript_available is not None: initial_state["transcript_available"] = transcript_available # type: ignore[assignment]
    if video_title         is not None: initial_state["video_title"]          = video_title          # type: ignore[assignment]
    if video_description   is not None: initial_state["video_description"]    = video_description    # type: ignore[assignment]

    config      = {"configurable": {"thread_id": thread_id}}
    final_state = app.invoke(initial_state, config=config)
    return final_state.get("report") or final_state
