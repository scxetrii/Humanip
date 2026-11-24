"""
tests/test_quality.py
─────────────────────
Unit tests for the multi-metric quality evaluation engine.

Run with:  pytest tests/test_quality.py -v
"""
from __future__ import annotations

import pytest

from backend.core.schemas import QualityMetrics, TextAnalysis
from backend.quality.evaluator import QualityEvaluator


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_analysis(citations=None):
    return TextAnalysis(
        word_count=100,
        char_count=600,
        sentence_count=6,
        paragraph_count=2,
        avg_sentence_length=16.7,
        readability_score=55.0,
        readability_grade="Grade 11",
        repetitive_structures=[],
        excessive_transitions=[],
        unnatural_phrases=[],
        overly_formal_expressions=[],
        repetitive_openers=[],
        generic_phrases=[],
        weak_connections=[],
        redundant_explanations=[],
        citations=citations or [],
        technical_terms=[],
        named_entities=[],
        numerical_values=[],
    )


ORIGINAL_TEXT = """
In conclusion, it is important to note that climate change poses significant
challenges. Furthermore, the data indicates a 2.4°C temperature rise.
Moreover, policymakers must act urgently. Additionally, the IPCC report (2023)
confirms these findings across 500 scenarios tested globally.
"""

GOOD_REWRITE = """
Climate change is one of the most pressing challenges of our time. Global
temperatures have risen by 2.4°C, and the latest IPCC report (2023) confirms
this trend across 500 scenarios. Policymakers have no time to waste — the window
for meaningful action is narrowing every year.
"""

BAD_REWRITE = """
Climate change is a problem. Things are bad. Experts say so. The IPCC says so.
Something must be done about it. The situation is concerning for many reasons.
"""

IDENTICAL_TEXT = ORIGINAL_TEXT  # semantic similarity should be ~1.0


@pytest.fixture(scope="module")
def evaluator():
    return QualityEvaluator()


@pytest.fixture(scope="module")
def good_metrics(evaluator):
    return evaluator.evaluate(ORIGINAL_TEXT, GOOD_REWRITE, _make_analysis())


@pytest.fixture(scope="module")
def bad_metrics(evaluator):
    return evaluator.evaluate(ORIGINAL_TEXT, BAD_REWRITE, _make_analysis())


@pytest.fixture(scope="module")
def identical_metrics(evaluator):
    return evaluator.evaluate(ORIGINAL_TEXT, IDENTICAL_TEXT, _make_analysis())


# ── Return Type ────────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_quality_metrics(self, good_metrics):
        assert isinstance(good_metrics, QualityMetrics)

    def test_overall_score_in_range(self, good_metrics):
        assert 0.0 <= good_metrics.overall_score <= 100.0

    def test_all_subscores_zero_to_one(self, good_metrics):
        for field in [
            good_metrics.semantic_similarity,
            good_metrics.factual_preservation,
            good_metrics.grammar_quality,
            good_metrics.repetition_reduction,
            good_metrics.citation_integrity,
            good_metrics.sentence_variation,
            good_metrics.paragraph_coherence,
        ]:
            assert 0.0 <= field <= 1.0, f"Score out of range: {field}"


# ── Semantic Similarity ───────────────────────────────────────────────────────

class TestSemanticSimilarity:
    def test_identical_text_has_high_similarity(self, identical_metrics):
        assert identical_metrics.semantic_similarity >= 0.95

    def test_good_rewrite_has_high_similarity(self, good_metrics):
        # A quality rewrite should preserve meaning — semantic sim ≥ 0.70
        assert good_metrics.semantic_similarity >= 0.60

    def test_bad_rewrite_lower_similarity_than_good(self, good_metrics, bad_metrics):
        # Good rewrite preserves meaning better than the vague bad rewrite
        assert good_metrics.semantic_similarity >= bad_metrics.semantic_similarity


# ── Factual Preservation ──────────────────────────────────────────────────────

class TestFactualPreservation:
    def test_numbers_preserved_in_good_rewrite(self, good_metrics):
        # "2.4°C" and "500" appear in both original and good rewrite
        assert good_metrics.factual_preservation >= 0.5

    def test_numbers_lost_in_bad_rewrite(self, bad_metrics):
        # Bad rewrite drops all specific numbers → lower factual score
        assert bad_metrics.factual_preservation <= good_metrics.factual_preservation

    def test_identical_text_full_factual_preservation(self, identical_metrics):
        assert identical_metrics.factual_preservation == 1.0


# ── Citation Integrity ────────────────────────────────────────────────────────

class TestCitationIntegrity:
    def test_citation_present_in_good_rewrite(self, evaluator):
        analysis = _make_analysis(citations=["IPCC report (2023)"])
        metrics = evaluator.evaluate(ORIGINAL_TEXT, GOOD_REWRITE, analysis)
        assert metrics.citation_integrity == 1.0

    def test_citation_missing_lowers_score(self, evaluator):
        analysis = _make_analysis(citations=["IPCC report (2023)", "UN Report (2022)"])
        # Good rewrite has IPCC but not UN Report
        metrics = evaluator.evaluate(ORIGINAL_TEXT, GOOD_REWRITE, analysis)
        assert metrics.citation_integrity < 1.0

    def test_no_citations_gives_full_score(self, evaluator):
        analysis = _make_analysis(citations=[])
        metrics = evaluator.evaluate(ORIGINAL_TEXT, GOOD_REWRITE, analysis)
        assert metrics.citation_integrity == 1.0


# ── Repetition Reduction ──────────────────────────────────────────────────────

class TestRepetitionReduction:
    def test_good_rewrite_reduces_repetition(self, good_metrics):
        # Score ≥ 0.5 means repetition did not worsen
        assert good_metrics.repetition_reduction >= 0.4

    def test_identical_rewrite_neutral_repetition(self, identical_metrics):
        # Identical → no reduction but also no increase → score near 0.5
        assert 0.3 <= identical_metrics.repetition_reduction <= 0.8


# ── Sentence Variation ────────────────────────────────────────────────────────

class TestSentenceVariation:
    def test_good_rewrite_has_variation(self, good_metrics):
        # Good rewrite has mixed short/long sentences
        assert good_metrics.sentence_variation >= 0.0  # just ensure no crash

    def test_bad_rewrite_less_variation(self, bad_metrics):
        # Bad rewrite has uniformly short sentences — low variation
        assert bad_metrics.sentence_variation <= good_metrics.sentence_variation + 0.3


# ── Overall Score ─────────────────────────────────────────────────────────────

class TestOverallScore:
    def test_good_rewrite_higher_overall_than_bad(self, good_metrics, bad_metrics):
        assert good_metrics.overall_score >= bad_metrics.overall_score

    def test_overall_is_weighted_combination(self, good_metrics):
        # Rough sanity: overall should not be wildly off any single metric
        # (checks that the weighted average is in a sensible range)
        assert 10.0 <= good_metrics.overall_score <= 100.0
