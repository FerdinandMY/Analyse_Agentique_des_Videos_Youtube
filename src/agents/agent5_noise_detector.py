from __future__ import annotations
from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent5NoiseDetector(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "noise_detector"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: detect noise later (e.g., audio quality / text noise).
        return {"input": payload, "noise_level": None}
from typing import Any, Dict

from core.base_agent import BaseAgent


class Agent5NoiseDetector(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "noise_detector"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: detect noise later (e.g., audio quality / text noise).
        return {"input": payload, "noise_level": None}