from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    sentiment_label: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class DiscourseResult(BaseModel):
    discourse_label: Literal["support", "question", "complaint", "other"]
    discourse_score: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None


class NoiseResult(BaseModel):
    noise_level: float = Field(ge=0.0, le=1.0)
    noise_label: Literal["low", "medium", "high"]
    rationale: Optional[str] = None


class SynthesisResult(BaseModel):
    sentiment: SentimentResult
    discourse: DiscourseResult
    noise: NoiseResult
    overall_score: float = Field(ge=0.0, le=1.0)
    quality_level: Literal["low", "medium", "good", "excellent"]
    summary: Optional[str] = None
