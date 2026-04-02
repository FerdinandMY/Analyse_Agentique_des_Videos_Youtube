from __future__ import annotations

from enum import Enum


class QualityLevel(str, Enum):
    low = "low"
    medium = "medium"
    good = "good"
    excellent = "excellent"