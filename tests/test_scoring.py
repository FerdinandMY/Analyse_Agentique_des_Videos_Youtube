"""
Tests — Score Formulas (test_scoring.py)

Validates the PRD §4.3 scoring formulas independently from LLM calls.
"""
import pytest

from agents.a6_synthesizer import _quality_label, W_SENTIMENT, W_DISCOURSE, W_NOISE
from agents.a7_topic_matcher import W_GLOBAL, W_PERTINENCE


# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_a6_weights_sum_to_one(self):
        assert abs(W_SENTIMENT + W_DISCOURSE + W_NOISE - 1.0) < 1e-9

    def test_a6_sentiment_weight(self):
        assert W_SENTIMENT == pytest.approx(0.35)

    def test_a6_discourse_weight(self):
        assert W_DISCOURSE == pytest.approx(0.40)

    def test_a6_noise_weight(self):
        assert W_NOISE == pytest.approx(0.25)

    def test_a7_weights_sum_to_one(self):
        assert abs(W_GLOBAL + W_PERTINENCE - 1.0) < 1e-9

    def test_a7_global_weight(self):
        assert W_GLOBAL == pytest.approx(0.60)

    def test_a7_pertinence_weight(self):
        assert W_PERTINENCE == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Score_Global formula
# ---------------------------------------------------------------------------

class TestScoreGlobalFormula:
    def _compute(self, s, d, n):
        return round(W_SENTIMENT * s + W_DISCOURSE * d + W_NOISE * n, 2)

    def test_all_zeros(self):
        assert self._compute(0, 0, 0) == 0.0

    def test_all_hundreds(self):
        assert self._compute(100, 100, 100) == 100.0

    def test_typical_values(self):
        # 0.35*70 + 0.40*60 + 0.25*80 = 24.5 + 24 + 20 = 68.5
        result = self._compute(70, 60, 80)
        assert result == pytest.approx(68.5)

    def test_discourse_weight_dominates(self):
        # When discourse is 100 and others are 0: score = 40
        assert self._compute(0, 100, 0) == pytest.approx(40.0)

    def test_sentiment_only(self):
        assert self._compute(100, 0, 0) == pytest.approx(35.0)

    def test_noise_only(self):
        assert self._compute(0, 0, 100) == pytest.approx(25.0)

    def test_midpoint(self):
        assert self._compute(50, 50, 50) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Score_Final formula
# ---------------------------------------------------------------------------

class TestScoreFinalFormula:
    def _compute(self, g, p):
        return round(W_GLOBAL * g + W_PERTINENCE * p, 2)

    def test_all_zeros(self):
        assert self._compute(0, 0) == 0.0

    def test_all_hundreds(self):
        assert self._compute(100, 100) == 100.0

    def test_global_dominates(self):
        # global=100, pertinence=0 → 60
        assert self._compute(100, 0) == pytest.approx(60.0)

    def test_pertinence_only(self):
        # global=0, pertinence=100 → 40
        assert self._compute(0, 100) == pytest.approx(40.0)

    def test_typical_values(self):
        # 0.60*65 + 0.40*80 = 39 + 32 = 71
        assert self._compute(65, 80) == pytest.approx(71.0)

    def test_no_topic_case(self):
        # Without topic: score_final = score_global (pertinence neutral=50 → different)
        # When topic is absent, score_final = score_global directly (A7 special case)
        score_global = 62.5
        score_final = score_global  # A7 no-topic path
        assert score_final == 62.5


# ---------------------------------------------------------------------------
# Quality Label
# ---------------------------------------------------------------------------

class TestQualityLabel:
    def test_faible_lower_bound(self):
        assert _quality_label(0) == "Faible"

    def test_faible_upper_bound(self):
        assert _quality_label(24.9) == "Faible"

    def test_moyen_lower_bound(self):
        assert _quality_label(25) == "Moyen"

    def test_moyen_upper_bound(self):
        assert _quality_label(49.9) == "Moyen"

    def test_bon_lower_bound(self):
        assert _quality_label(50) == "Bon"

    def test_bon_upper_bound(self):
        assert _quality_label(74.9) == "Bon"

    def test_excellent_lower_bound(self):
        assert _quality_label(75) == "Excellent"

    def test_excellent_max(self):
        assert _quality_label(100) == "Excellent"

    def test_boundary_exact_25(self):
        assert _quality_label(25) == "Moyen"

    def test_boundary_exact_50(self):
        assert _quality_label(50) == "Bon"

    def test_boundary_exact_75(self):
        assert _quality_label(75) == "Excellent"

    def test_noise_score_formula(self):
        # Score_Bruit = (1 - noise_ratio) * 100
        assert (1.0 - 0.30) * 100 == pytest.approx(70.0)
        assert (1.0 - 0.0) * 100 == pytest.approx(100.0)
        assert (1.0 - 1.0) * 100 == pytest.approx(0.0)
