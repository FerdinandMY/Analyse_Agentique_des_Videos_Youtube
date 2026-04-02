from __future__ import annotations

from typing import Any, Dict

from core.base_agent import BaseAgent

from utils.text_cleaner import normalize_text


class Agent2Preprocessor(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "preprocessor"

    def run(self, payload: Any) -> Dict[str, Any]:
        # Scaffolding: normalize text fields in the payload.
        text = payload if isinstance(payload, str) else payload.get("text") if isinstance(payload, dict) else None
        cleaned = normalize_text(text) if isinstance(text, str) else None
        if isinstance(payload, dict):
            payload = dict(payload)
            if cleaned is not None:
                payload["cleaned_text"] = cleaned
        return {"input": payload, "preprocessed": True}