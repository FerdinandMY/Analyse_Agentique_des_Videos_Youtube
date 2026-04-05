"""
graph.py — LangGraph Pipeline Definition
==========================================
Wires the 7 agents into a LangGraph StateGraph:

    A1 (Loader) → A2 (Preprocessor) → [ A3 ‖ A4 ‖ A5 ] → A6 (Synthesizer) → A7 (Topic Matcher)

A3, A4, A5 run in parallel (fan-out from A2, fan-in to A6).
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
        "video_id": state.get("video_id"),
        "topic": state.get("topic"),
        "score_global": state.get("score_global"),
        "score_pertinence": state.get("score_pertinence"),
        "score_final": state.get("score_final"),
        "quality_label": synthesis.get("quality_label"),
        "topic_verdict": state.get("topic_verdict"),
        "summary": synthesis.get("summary"),
        "details": {
            "sentiment": state.get("sentiment"),
            "discourse": state.get("discourse"),
            "noise": state.get("noise"),
        },
        "comment_count": len(state.get("cleaned_comments") or []),
        "errors": state.get("errors") or [],
    }
    return {"report": report}


def build_graph(checkpointer: Any = None) -> Any:
    """
    Build and compile the LangGraph pipeline.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver by default).
                      Pass SqliteSaver for persistent checkpointing.

    Returns:
        Compiled LangGraph app.
    """
    builder: StateGraph[PipelineState] = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("a1_loader", a1_loader)
    builder.add_node("a2_preprocessor", a2_preprocessor)
    builder.add_node("a3_sentiment", a3_sentiment)
    builder.add_node("a4_discourse", a4_discourse)
    builder.add_node("a5_noise", a5_noise)
    builder.add_node("a6_synthesizer", a6_synthesizer)
    builder.add_node("a7_topic_matcher", a7_topic_matcher)
    builder.add_node("assemble_report", _assemble_report)

    # Sequential backbone
    builder.add_edge(START, "a1_loader")
    builder.add_edge("a1_loader", "a2_preprocessor")

    # Parallel fan-out: A2 → {A3, A4, A5}
    builder.add_edge("a2_preprocessor", "a3_sentiment")
    builder.add_edge("a2_preprocessor", "a4_discourse")
    builder.add_edge("a2_preprocessor", "a5_noise")

    # Fan-in: {A3, A4, A5} → A6
    builder.add_edge("a3_sentiment", "a6_synthesizer")
    builder.add_edge("a4_discourse", "a6_synthesizer")
    builder.add_edge("a5_noise", "a6_synthesizer")

    # Sequential tail: A6 → A7 → assemble → END
    builder.add_edge("a6_synthesizer", "a7_topic_matcher")
    builder.add_edge("a7_topic_matcher", "assemble_report")
    builder.add_edge("assemble_report", END)

    resolved_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=resolved_checkpointer)


def run_pipeline(
    *,
    csv_path: str = "",
    video_id: str = "",
    topic: str = "",
    raw_comments: Optional[list] = None,
    thread_id: str = "default",
    checkpointer: Any = None,
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

    Returns:
        The final report dict (or partial state on error).
    """
    app = build_graph(checkpointer)

    initial_state: PipelineState = {
        "csv_path": csv_path,
        "video_id": video_id,
        "topic": topic,
        "errors": [],
    }
    if raw_comments is not None:
        initial_state["raw_comments"] = raw_comments  # type: ignore[assignment]

    config = {"configurable": {"thread_id": thread_id}}
    final_state = app.invoke(initial_state, config=config)
    return final_state.get("report") or final_state
