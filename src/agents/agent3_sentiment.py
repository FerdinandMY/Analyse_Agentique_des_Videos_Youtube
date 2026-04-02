from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent3Sentiment(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "sentiment"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: compute sentiment later with an LLM or model.
        return {"input": payload, "sentiment": None}