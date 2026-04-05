"""
models/llm_loader.py — LLM Loader
====================================
Manages LLM access for the pipeline.

Two backends are supported:

1. OpenAI-compatible API (default / dev)
   Set env vars: OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
   Works with Qwen1.5-7B-Chat and Phi-3-mini served via vLLM / Ollama / LM Studio.

2. HuggingFace local (Kaggle GPU / offline)
   Set env vars: LLM_BACKEND=huggingface, HF_MODEL_ID
   Loads in float16 to stay within the 16 GB VRAM budget (NFR-04).
   Only one model is loaded at a time (unloads previous before loading next).

The public interface is a single function:
    llm = get_llm()   # returns an object with .invoke([messages]) -> response

Usage inside agent nodes:
    from models.llm_loader import get_llm
    llm = get_llm()
    if llm is None:
        ...  # fallback
"""
from __future__ import annotations

import os
from typing import Any, Optional

_cached_llm: Any = None
_cached_backend: str = ""


def _build_openai_llm() -> Any:
    """Build an OpenAI-compatible ChatLLM via langchain-openai."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain-openai is not installed.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key or not base_url:
        return None

    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0)


class _HuggingFaceLLM:
    """
    Thin wrapper around a HuggingFace text-generation pipeline.
    Exposes .invoke([messages]) to match the LangChain interface.
    Loads the model in float16 to respect Kaggle T4 VRAM constraints.
    """

    def __init__(self, model_id: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for HuggingFace backend.") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
        )

    def invoke(self, messages: list[Any]) -> Any:
        # Convert LangChain message objects to a single prompt string
        prompt_parts = []
        for m in messages:
            role = getattr(m, "type", "human")
            content = getattr(m, "content", str(m))
            prompt_parts.append(f"[{role.upper()}] {content}")
        prompt = "\n".join(prompt_parts)

        outputs = self._pipe(prompt)
        generated = outputs[0]["generated_text"][len(prompt):]

        # Return an object with a `.content` attribute to match LangChain convention
        class _Resp:
            content = generated.strip()

        return _Resp()


def _build_huggingface_llm() -> Optional[_HuggingFaceLLM]:
    model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen1.5-7B-Chat")
    try:
        return _HuggingFaceLLM(model_id)
    except Exception as exc:
        import logging
        logging.getLogger("llm_loader").error("HuggingFace load failed: %s", exc)
        return None


def get_llm(force_reload: bool = False) -> Optional[Any]:
    """
    Return the LLM instance (cached after first call).

    The backend is selected via LLM_BACKEND env var:
      - "openai"       (default) — OpenAI-compatible API
      - "huggingface"            — Local HuggingFace model (float16, GPU)

    Args:
        force_reload: Discard cached model and reload (useful when switching models).

    Returns:
        LLM instance with .invoke([messages]) interface, or None if unconfigured.
    """
    global _cached_llm, _cached_backend

    backend = os.getenv("LLM_BACKEND", "openai").lower()

    if _cached_llm is not None and not force_reload and backend == _cached_backend:
        return _cached_llm

    # Unload previous model to free VRAM (max 1 model at a time — NFR-04)
    if _cached_llm is not None and hasattr(_cached_llm, "_pipe"):
        try:
            import torch
            del _cached_llm
            torch.cuda.empty_cache()
        except Exception:
            pass
        _cached_llm = None

    if backend == "huggingface":
        _cached_llm = _build_huggingface_llm()
    else:
        _cached_llm = _build_openai_llm()

    _cached_backend = backend
    return _cached_llm
