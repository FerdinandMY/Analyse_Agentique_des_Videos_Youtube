"""
utils/text_cleaner.py — Text Normalisation
"""
from __future__ import annotations

import html
import re
from typing import Optional

_whitespace_re = re.compile(r"\s+")
_html_tag_re = re.compile(r"<[^>]+>")


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Text normalisation pipeline:
    1. Decode HTML entities  (&#39; → ', &amp; → &, <br> → space)
    2. Strip HTML tags       (<br>, <a href=...>, etc.)
    3. Collapse whitespace
    4. Strip edges
    5. Lowercase
    """
    if text is None:
        return None
    # 1. Decode HTML entities (YouTube encodes apostrophes, quotes, etc.)
    text = html.unescape(text)
    # 2. Replace HTML tags with a space
    text = _html_tag_re.sub(" ", text)
    # 3-5. Collapse whitespace + strip + lowercase
    return _whitespace_re.sub(" ", text).strip().lower()
