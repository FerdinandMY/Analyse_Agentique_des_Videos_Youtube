from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import Config
from core.langgraph_pipeline import PipelineState, build_langgraph_app


@dataclass(frozen=True)
class PipelineResult:
    status: str
    data: Dict[str, Any]
    errors: Optional[List[str]] = None


class Pipeline:
    """
    Orchestrates the multi-agent pipeline end-to-end.

    Implementation uses LangGraph with checkpointing (dev memory / prod sqlite).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        checkpoint_mode: str = "memory",
        sqlite_path: str = "checkpoints.db",
    ) -> None:
        self._config = config or Config.from_env()
        self._checkpoint_mode = checkpoint_mode
        self._sqlite_path = sqlite_path

    def run(self, inputs: Dict[str, Any]) -> PipelineResult:
        """
        Run the full pipeline.

        Expected inputs (placeholder):
        - video_ids: list[str]
        - other optional metadata
        """
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.sqlite import SqliteSaver

        thread_id = str(inputs.get("thread_id", "default"))
        state_input: PipelineState = {
            "video_ids": inputs.get("video_ids", []) or [],
            "comments": inputs.get("comments", []) or [],
            "errors": [],
        }
        config_run = {"configurable": {"thread_id": thread_id}}

        if self._checkpoint_mode == "sqlite":
            with SqliteSaver.from_conn_string(self._sqlite_path) as checkpointer:
                app = build_langgraph_app(config=self._config, checkpointer=checkpointer)
                result_state = app.invoke(state_input, config=config_run)
        else:
            checkpointer = MemorySaver()
            app = build_langgraph_app(config=self._config, checkpointer=checkpointer)
            result_state = app.invoke(state_input, config=config_run)

        errors = result_state.get("errors")
        return PipelineResult(status="ok", data=dict(result_state), errors=errors)