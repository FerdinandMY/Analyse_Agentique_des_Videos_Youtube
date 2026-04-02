from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent1Collector(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "collector"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: collect raw data (e.g., video comments) for later phases.
        return {"input": payload, "collected": False}