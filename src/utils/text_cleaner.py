from __future__ import annotations

import re
from typing import Optional


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Basic text normalization for later NLP/LLM steps.

    - strip leading/trailing whitespace
    - collapse repeated whitespace
    - normalize to lower-case
    """
    if text is None:
        return None
    cleaned = _whitespace_re.sub(" ", text).strip()
    return cleaned.lower()