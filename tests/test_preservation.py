"""
tests/test_preservation.py
───────────────────────────
Unit tests for the PreservationGuardian placeholder system.

Run with:  pytest tests/test_preservation.py -v
"""
from __future__ import annotations

import pytest

from backend.core.schemas import TextAnalysis
from backend.preservation.guardian import PreservationGuardian


# ── Minimal TextAnalysis stub ─────────────────────────────────────────────────

def _make_analysis(citations=None, technical_terms=None, numerical_values=None):
    """Return a minimal TextAnalysis with just the fields Guardian reads."""
    return TextAnalysis(
        word_count=50,
        char_count=300,
        sentence_count=3,
        paragraph_count=1,
        avg_sentence_length=16.7,
        readability_score=60.0,
        readability_grade="Grade 10",
        repetitive_structures=[],
        excessive_transitions=[],
        unnatural_phrases=[],
        overly_formal_expressions=[],
        repetitive_openers=[],
        generic_phrases=[],
        weak_connections=[],
        redundant_explanations=[],
        citations=citations or [],
        technical_terms=technical_terms or [],
        named_entities=[],
        numerical_values=numerical_values or [],
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def guardian():
    return PreservationGuardian()


CITATION_TEXT = (
    "According to Smith et al. (2023), the model achieved 94.5% accuracy. "
    "Earlier work (Johnson & Lee, 2021) reported 87.3% on the same benchmark."
)

TECH_TERM_TEXT = (
    "The BERT model uses Transformer architecture with multi-head self-attention. "
    "Fine-tuning on GPT-4 required 1,200 GPU hours."
)


# ── Protect / Restore Round-Trip ──────────────────────────────────────────────

class TestRoundTrip:
    def test_citation_survives_round_trip(self, guardian):
        citations = ["Smith et al. (2023)", "Johnson & Lee, 2021"]
        analysis = _make_analysis(citations=citations)
        protected, ph_map = guardian.protect(CITATION_TEXT, analysis)
        restored = guardian.restore(protected, ph_map)
        for c in citations:
            assert c in restored, f"Citation '{c}' missing after restore"

    def test_number_survives_round_trip(self, guardian):
        analysis = _make_analysis(numerical_values=["94.5", "87.3"])
        protected, ph_map = guardian.protect(CITATION_TEXT, analysis)
        restored = guardian.restore(protected, ph_map)
        assert "94.5" in restored
        assert "87.3" in restored

    def test_technical_term_survives_round_trip(self, guardian):
        terms = ["BERT", "GPT-4", "Transformer"]
        analysis = _make_analysis(technical_terms=terms)
        protected, ph_map = guardian.protect(TECH_TERM_TEXT, analysis)
        restored = guardian.restore(protected, ph_map)
        for t in terms:
            assert t in restored, f"Term '{t}' missing after restore"

    def test_placeholder_format_in_protected_text(self, guardian):
        citations = ["Smith et al. (2023)"]
        analysis = _make_analysis(citations=citations)
        protected, ph_map = guardian.protect(CITATION_TEXT, analysis)
        # Protected text should contain ⟦...⟧ tokens
        assert "⟦" in protected and "⟧" in protected

    def test_no_placeholders_remain_after_restore(self, guardian):
        analysis = _make_analysis(citations=["Smith et al. (2023)"])
        protected, ph_map = guardian.protect(CITATION_TEXT, analysis)
        restored = guardian.restore(protected, ph_map)
        assert "⟦" not in restored
        assert "⟧" not in restored

    def test_text_without_preserved_elements_unchanged(self, guardian):
        plain_text = "This is a simple sentence with no special content."
        analysis = _make_analysis()
        protected, ph_map = guardian.protect(plain_text, analysis)
        restored = guardian.restore(protected, ph_map)
        # Numbers in plain text are still round-tripped but unchanged
        assert "simple sentence" in restored


# ── Factual Integrity Verification ───────────────────────────────────────────

class TestFactualVerification:
    def test_no_issues_when_numbers_match(self, guardian):
        original  = "The accuracy was 94.5% and loss was 0.023."
        rewritten = "With 94.5% accuracy and a loss of 0.023, the model performed well."
        analysis  = _make_analysis()
        issues = guardian.verify_factual_integrity(original, rewritten, analysis)
        assert len(issues) == 0

    def test_detects_missing_number(self, guardian):
        original  = "The accuracy was 94.5% and F1 was 0.87."
        rewritten = "The model performed well overall."  # numbers removed
        analysis  = _make_analysis()
        issues = guardian.verify_factual_integrity(original, rewritten, analysis)
        assert len(issues) > 0

    def test_detects_changed_number(self, guardian):
        original  = "1,200 patients participated in the study."
        rewritten = "1,500 patients participated in the study."  # changed!
        analysis  = _make_analysis()
        issues = guardian.verify_factual_integrity(original, rewritten, analysis)
        assert "1,200" in issues or len(issues) > 0


# ── Toggle Behaviour ──────────────────────────────────────────────────────────

class TestToggles:
    def test_protect_citations_false_does_not_replace_citations(self, guardian):
        analysis = _make_analysis(citations=["Smith et al. (2023)"])
        protected, _ = guardian.protect(
            CITATION_TEXT, analysis, protect_citations=False
        )
        assert "Smith et al. (2023)" in protected  # still in text, not replaced

    def test_protect_terms_false_does_not_replace_terms(self, guardian):
        terms = ["BERT", "GPT-4"]
        analysis = _make_analysis(technical_terms=terms)
        protected, _ = guardian.protect(
            TECH_TERM_TEXT, analysis, protect_terms=False
        )
        for t in terms:
            assert t in protected  # still literal, not replaced


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_text_no_crash(self, guardian):
        analysis = _make_analysis()
        protected, ph_map = guardian.protect("", analysis)
        restored = guardian.restore(protected, ph_map)
        assert restored == ""

    def test_citation_not_in_text_no_crash(self, guardian):
        """Citation listed in analysis but not present in text — no KeyError."""
        analysis = _make_analysis(citations=["NonExistent (2099)"])
        protected, ph_map = guardian.protect(CITATION_TEXT, analysis)
        restored = guardian.restore(protected, ph_map)
        assert len(restored) > 0

    def test_very_long_text(self, guardian):
        long_text = ("The value is 42.0 and the rate is 7.5%. " * 500)
        analysis = _make_analysis()
        protected, ph_map = guardian.protect(long_text, analysis)
        restored = guardian.restore(protected, ph_map)
        assert "42.0" in restored
        assert "7.5%" in restored
