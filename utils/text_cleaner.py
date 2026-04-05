"""
utils/text_cleaner.py — Text Normalisation
"""
from __future__ import annotations

import re
from typing import Optional

_whitespace_re = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Basic text normalisation:
    - strip leading/trailing whitespace
    - collapse repeated whitespace
    - lowercase
    """
    if text is None:
        return None
    return _whitespace_re.sub(" ", text).strip().lower()
