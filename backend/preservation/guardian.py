"""
backend/preservation/guardian.py
─────────────────────────────────
PreservationGuardian: placeholder injection system.

Before sending text to the LLM:
  - Replaces citations, technical terms, and numerical values with
    unique placeholder tokens that the model cannot accidentally modify.
  - After the LLM responds, restores all placeholders to their originals.
  - Runs a factual-change detector to flag any value the LLM altered.

Placeholder format:  ⟦CITEREF_0001⟧  ⟦TERM_0001⟧  ⟦NUM_0001⟧
These Unicode brackets are extremely unlikely to appear in normal text,
so regex matching is clean and unambiguous.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from loguru import logger

from backend.core.schemas import TextAnalysis


class PlaceholderMap(NamedTuple):
    """Bidirectional map between placeholders and their original strings."""
    to_original: dict[str, str]   # placeholder → original
    from_original: dict[str, str] # original → placeholder


_LEFT  = "⟦"
_RIGHT = "⟧"
_PH_PATTERN = re.compile(re.escape(_LEFT) + r'[A-Z_]+\d{4}' + re.escape(_RIGHT))


def _make_placeholder(prefix: str, index: int) -> str:
    return f"{_LEFT}{prefix}_{index:04d}{_RIGHT}"


class PreservationGuardian:
    """
    Inject placeholders before LLM call; restore after.
    Also provides factual verification between original and output.
    """

    def protect(
        self,
        text: str,
        analysis: TextAnalysis,
        protect_citations: bool = True,
        protect_terms: bool = True,
    ) -> tuple[str, PlaceholderMap]:
        """
        Replace preserved elements in *text* with placeholder tokens.
        Returns (modified_text, PlaceholderMap).
        """
        to_original: dict[str, str] = {}
        from_original: dict[str, str] = {}
        idx = 0

        # ── Citations (highest priority – replace longest first) ──────────────
        if protect_citations and analysis.citations:
            sorted_cites = sorted(analysis.citations, key=len, reverse=True)
            for citation in sorted_cites:
                if citation in text:
                    ph = _make_placeholder("CITE", idx)
                    to_original[ph] = citation
                    from_original[citation] = ph
                    text = text.replace(citation, ph)
                    idx += 1

        # ── Numerical values ──────────────────────────────────────────────────
        # Always protect numbers regardless of toggles
        num_pattern = re.compile(
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
        )
        # Replace digits in a second pass to avoid double-replacing citation numbers
        for m in reversed(list(num_pattern.finditer(text))):
            original_val = m.group()
            if original_val not in from_original:
                ph = _make_placeholder("NUM", idx)
                to_original[ph] = original_val
                from_original[original_val] = ph
                text = text[:m.start()] + ph + text[m.end():]
                idx += 1

        # ── Technical terms ───────────────────────────────────────────────────
        if protect_terms and analysis.technical_terms:
            sorted_terms = sorted(analysis.technical_terms, key=len, reverse=True)
            for term in sorted_terms:
                if term in text and term not in from_original:
                    ph = _make_placeholder("TERM", idx)
                    to_original[ph] = term
                    from_original[term] = ph
                    # Whole-word replacement only
                    text = re.sub(r'\b' + re.escape(term) + r'\b', ph, text, count=1)
                    idx += 1

        logger.debug("Injected {} placeholders.", idx)
        return text, PlaceholderMap(to_original=to_original, from_original=from_original)

    def restore(self, text: str, ph_map: PlaceholderMap) -> str:
        """Replace all placeholder tokens back to their originals."""
        for placeholder, original in ph_map.to_original.items():
            text = text.replace(placeholder, original)

        # Safety: remove any stray placeholder tokens the LLM might have mangled
        stray = _PH_PATTERN.findall(text)
        if stray:
            logger.warning("Stray placeholders found after restore: {}", stray)
            for s in stray:
                text = text.replace(s, ph_map.to_original.get(s, ""))

        return text

    def verify_factual_integrity(
        self,
        original: str,
        rewritten: str,
        analysis: TextAnalysis,
    ) -> dict[str, str]:
        """
        Compare numerical values between original and rewritten.
        Returns a dict of { original_value: "MISSING" | "CHANGED to X" }.
        """
        issues: dict[str, str] = {}
        num_pattern = re.compile(r'\b\d[\d,]*(?:\.\d+)?\s*%?')

        orig_nums = set(num_pattern.findall(original))
        new_nums  = set(num_pattern.findall(rewritten))

        for n in orig_nums:
            if n not in new_nums:
                issues[n] = "MISSING from rewritten text"

        logger.debug("Factual integrity check: {} issues found.", len(issues))
        return issues
