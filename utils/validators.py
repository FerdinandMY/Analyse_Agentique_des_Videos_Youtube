"""
utils/validators.py
====================
Validateurs Pydantic par agent — PRD v3.0 §3.5 Niveau 4

Chaque validateur :
  - hérite de BaseValidator
  - implémente .validate(data) → dict normalisé
  - implémente .check_coherence(data) → list[str] (hallucination_flags)

Usage :
    from utils.validators import SentimentValidator, DiscourseValidator
    from utils.validators import NoiseValidator, TopicValidator

    validator = SentimentValidator()
    clean = validator.validate(raw_dict)
    flags = validator.check_coherence(clean)
"""
from __future__ import annotations

from typing import Any


# ── Classe de base ────────────────────────────────────────────────────────────

class BaseValidator:
    """Interface commune pour tous les validateurs."""

    def validate(self, data: Any) -> dict[str, Any]:
        raise NotImplementedError

    def check_coherence(self, data: dict[str, Any]) -> list[str]:
        return []

    # ── Helpers partagés ──────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
        try:
            v = float(value)
            return round(max(lo, min(hi, v)), 4)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_score(value: Any, default: float = 50.0) -> float:
        """Normalise un score vers [0-100] quel que soit le format d'entrée."""
        try:
            v = float(value)
            if v <= 1.0:        # format [0-1] → [0-100]
                v = v * 100
            return round(max(0.0, min(100.0, v)), 2)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _str(value: Any, default: str = "") -> str:
        return str(value).strip() if value is not None else default


# ── A3 — Sentiment ────────────────────────────────────────────────────────────

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}


class SentimentValidator(BaseValidator):
    """Valide la sortie de l'agent A3 Sentiment (ReAct / CoT)."""

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"SentimentValidator: attendu dict, reçu {type(data).__name__}")

        label = self._str(data.get("sentiment_label"), "neutral").lower()
        if label not in _VALID_SENTIMENTS:
            label = "neutral"

        score = self._normalize_score(data.get("sentiment_score"), 50.0)

        return {
            "sentiment_label":  label,
            "sentiment_score":  score,
            "confidence":       self._clamp(data.get("confidence"), 0.0, 1.0, 0.5),
            "rationale":        self._str(data.get("rationale")),
            "reasoning":        self._str(data.get("reasoning")),   # CoT trace
            "sarcasm_detected": bool(data.get("sarcasm_detected", False)),
        }

    def check_coherence(self, data: dict[str, Any]) -> list[str]:
        flags = []
        label = data["sentiment_label"]
        score = data["sentiment_score"]

        # Incohérence : label positif mais score < 30
        if label == "positive" and score < 30.0:
            flags.append("sentiment_label_score_mismatch: positive but score < 30")
        # Incohérence : label négatif mais score > 70
        if label == "negative" and score > 70.0:
            flags.append("sentiment_label_score_mismatch: negative but score > 70")
        # Confiance trop basse avec label fort
        if label != "neutral" and data["confidence"] < 0.3:
            flags.append("low_confidence_non_neutral_sentiment")
        return flags


# ── A4 — Discourse ────────────────────────────────────────────────────────────

class DiscourseValidator(BaseValidator):
    """Valide la sortie de l'agent A4 Discourse (CoT + ToT conditionnel)."""

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"DiscourseValidator: attendu dict, reçu {type(data).__name__}")

        informativeness  = self._normalize_score(data.get("informativeness"), 50.0)
        argumentation    = self._normalize_score(data.get("argumentation"), 50.0)
        constructiveness = self._normalize_score(data.get("constructiveness"), 50.0)
        discourse_score  = self._normalize_score(
            data.get("discourse_score"),
            round((informativeness + argumentation + constructiveness) / 3, 2),
        )

        hq = data.get("high_quality_indices") or []
        if not isinstance(hq, list):
            hq = []
        hq = [int(i) for i in hq if isinstance(i, (int, float, str)) and str(i).isdigit()]

        # Traces ToT (optionnel)
        tot_branches = data.get("tot_branches") or {}

        return {
            "informativeness":      informativeness,
            "argumentation":        argumentation,
            "constructiveness":     constructiveness,
            "discourse_score":      discourse_score,
            "high_quality_indices": hq,
            "rationale":            self._str(data.get("rationale")),
            "reasoning":            self._str(data.get("reasoning")),
            "tot_branches":         tot_branches,
            "tot_used":             bool(data.get("tot_used", False)),
        }

    def check_coherence(self, data: dict[str, Any]) -> list[str]:
        flags = []
        dims = [data["informativeness"], data["argumentation"], data["constructiveness"]]
        avg  = sum(dims) / 3

        # Le discourse_score devrait être proche de la moyenne des dimensions
        if abs(data["discourse_score"] - avg) > 20:
            flags.append(
                f"discourse_score_dim_mismatch: score={data['discourse_score']:.1f} vs dim_avg={avg:.1f}"
            )
        # High quality indices ne peuvent pas dépasser 30 (max commentaires analysés)
        invalid_hq = [i for i in data["high_quality_indices"] if i >= 30]
        if invalid_hq:
            flags.append(f"invalid_high_quality_indices: {invalid_hq}")
        return flags


# ── A5 — Noise ────────────────────────────────────────────────────────────────

class NoiseValidator(BaseValidator):
    """Valide la sortie de l'agent A5 Noise (SVM + CoT léger)."""

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"NoiseValidator: attendu dict, reçu {type(data).__name__}")

        spam_ratio     = self._clamp(data.get("spam_ratio"), 0.0, 100.0, 0.0)
        offtopic_ratio = self._clamp(data.get("offtopic_ratio"), 0.0, 100.0, 0.0)
        reaction_ratio = self._clamp(data.get("reaction_ratio"), 0.0, 100.0, 0.0)
        toxic_ratio    = self._clamp(data.get("toxic_ratio"), 0.0, 100.0, 0.0)
        bot_ratio      = self._clamp(data.get("bot_ratio"), 0.0, 100.0, 0.0)

        # Normaliser si en format [0-1]
        def _norm_ratio(v: float) -> float:
            return round(v * 100, 2) if v <= 1.0 else round(v, 2)

        spam_ratio     = _norm_ratio(spam_ratio)
        offtopic_ratio = _norm_ratio(offtopic_ratio)
        reaction_ratio = _norm_ratio(reaction_ratio)
        toxic_ratio    = _norm_ratio(toxic_ratio)
        bot_ratio      = _norm_ratio(bot_ratio)

        noise_ratio = self._normalize_score(data.get("noise_ratio"), 30.0)
        noise_score = round(100.0 - noise_ratio, 2)

        return {
            "spam_ratio":     spam_ratio,
            "offtopic_ratio": offtopic_ratio,
            "reaction_ratio": reaction_ratio,
            "toxic_ratio":    toxic_ratio,
            "bot_ratio":      bot_ratio,
            "noise_ratio":    noise_ratio,
            "noise_score":    noise_score,
            "rationale":      self._str(data.get("rationale")),
            "reasoning":      self._str(data.get("reasoning")),
            "svm_used":       bool(data.get("svm_used", False)),
        }

    def check_coherence(self, data: dict[str, Any]) -> list[str]:
        flags = []
        sub_total = (
            data["spam_ratio"] + data["offtopic_ratio"]
            + data["reaction_ratio"] + data["toxic_ratio"] + data["bot_ratio"]
        ) / 100  # ramène en [0-1]

        # La somme des ratios individuels ne devrait pas dépasser 1.0
        if sub_total > 1.5:
            flags.append(f"noise_ratios_sum_too_high: {sub_total:.2f} > 1.5")

        # noise_score devrait être cohérent avec noise_ratio
        expected_score = round(100.0 - data["noise_ratio"], 2)
        if abs(data["noise_score"] - expected_score) > 5:
            flags.append(
                f"noise_score_ratio_mismatch: score={data['noise_score']} vs expected={expected_score}"
            )
        return flags


# ── A7 — Topic Matcher ────────────────────────────────────────────────────────

class TopicValidator(BaseValidator):
    """Valide la sortie de l'agent A7 Topic Matcher (ToT + Self-Consistency)."""

    def validate(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(f"TopicValidator: attendu dict, reçu {type(data).__name__}")

        pertinence_score = self._normalize_score(data.get("pertinence_score"), 50.0)

        # Branches ToT (optionnel)
        branches = data.get("tot_branches") or {}

        # Self-Consistency runs (optionnel)
        sc_runs = data.get("sc_runs") or []
        if not isinstance(sc_runs, list):
            sc_runs = []

        return {
            "pertinence_score":  pertinence_score,
            "verdict":           self._str(data.get("verdict")),
            "tot_branches":      branches,
            "sc_runs":           sc_runs,
            "sc_consensus":      bool(data.get("sc_consensus", True)),
            "low_consensus":     bool(data.get("low_consensus", False)),
        }

    def check_coherence(self, data: dict[str, Any]) -> list[str]:
        flags = []
        # Si low_consensus mais sc_consensus=True → incohérence
        if data["low_consensus"] and data["sc_consensus"]:
            flags.append("topic_consensus_flag_contradiction")
        # Verdict vide avec score extrême
        if not data["verdict"] and (data["pertinence_score"] > 80 or data["pertinence_score"] < 20):
            flags.append("topic_extreme_score_no_verdict")
        return flags
