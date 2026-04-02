from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ScoringWeights:
    """
    Weighted aggregation formula:
    score = w1 * sentiment + w2 * discourse + w3 * noise
    """

    w1: float = 0.4
    w2: float = 0.4
    w3: float = 0.2


class Scorer:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self._weights = weights or ScoringWeights()

    def compute(self, features: Dict[str, float]) -> float:
        sentiment = float(features.get("sentiment", 0.0))
        discourse = float(features.get("discourse", 0.0))
        noise = float(features.get("noise", 0.0))
        return (
            self._weights.w1 * sentiment
            + self._weights.w2 * discourse
            + self._weights.w3 * noise
        )