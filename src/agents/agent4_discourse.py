from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent4Discourse(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "discourse"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: extract discourse cues later.
        return {"input": payload, "discourse": None}