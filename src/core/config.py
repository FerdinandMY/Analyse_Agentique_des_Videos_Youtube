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
    openai_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    youtube_api_key: Optional[str] = None
    youtube_max_results: int = 50
    youtube_max_pages: int = 1
    llm_provider: str = "openai"
    default_language: str = "en"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            youtube_api_key=os.getenv("YOUTUBE_API_KEY"),
            youtube_max_results=int(os.getenv("YOUTUBE_MAX_RESULTS", "50")),
            youtube_max_pages=int(os.getenv("YOUTUBE_MAX_PAGES", "1")),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        )