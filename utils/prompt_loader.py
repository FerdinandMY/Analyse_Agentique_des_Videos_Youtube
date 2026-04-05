"""
utils/prompt_loader.py — Versioned Prompt Loader
=================================================
Loads a prompt template from a file. Falls back to a provided default
if the file is missing or empty.

Prompt files support two placeholders replaced by callers:
  {{context}}             — comment corpus text
  {{format_instructions}} — Pydantic output parser instructions
"""
from __future__ import annotations


def load_prompt(path: str, fallback: str) -> str:
    """
    Load a prompt template from a text file.

    Args:
        path:     Relative or absolute path to the prompt file.
        fallback: Inline default to use if file is missing / empty.

    Returns:
        Prompt string (file content or fallback).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
    except FileNotFoundError:
        pass
    return fallback
