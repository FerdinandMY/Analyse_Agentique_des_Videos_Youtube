from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI

from core.config import Config


def build_chat_llm(config: Config) -> Optional[ChatOpenAI]:
    """
    Build an OpenAI-compatible Chat model.

    Works with Qwen when it is served via an OpenAI-compatible endpoint.
    """

    api_key = config.openai_api_key
    base_url = config.openai_base_url
    model = config.llm_model

    if not api_key or not base_url:
        return None

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

