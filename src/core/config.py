from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """
    Global configuration for the pipeline.

    Notes:
    - Keep secrets in environment variables (no hardcoding).
    - Add fields progressively as agents get implemented.
    """

    openai_api_key: Optional[str] = None
    youtube_api_key: Optional[str] = None
    llm_provider: str = "openai"
    default_language: str = "en"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        )