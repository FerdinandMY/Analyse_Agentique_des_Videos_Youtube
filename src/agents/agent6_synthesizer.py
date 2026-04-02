from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent6Synthesizer(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "synthesizer"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: synthesize final output later with LLM or rules.
        return {"input": payload, "final_output": None}