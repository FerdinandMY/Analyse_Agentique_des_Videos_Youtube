from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.config import Config


@dataclass(frozen=True)
class PipelineResult:
    status: str
    data: Dict[str, Any]
    errors: Optional[List[str]] = None


class Pipeline:
    """
    Orchestrates the multi-agent pipeline end-to-end.

    Current implementation is a scaffold: it does not execute LLM calls yet.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config.from_env()

    def run(self, inputs: Dict[str, Any]) -> PipelineResult:
        """
        Run the full pipeline.

        Expected inputs (placeholder):
        - video_ids: list[str]
        - other optional metadata
        """
        # This method is intentionally minimal for scaffolding purposes.
        return PipelineResult(
            status="not_implemented",
            data={"inputs": inputs, "note": "Pipeline scaffold created. Implement agents next."},
            errors=None,
        )