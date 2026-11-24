"""
backend/quality/evaluator.py
─────────────────────────────
Multi-metric quality evaluation engine.

Scores the humanized output on eight dimensions (all 0.0–1.0):
  1. Semantic Similarity      – sentence-transformers cosine similarity
  2. Factual Preservation     – numerical value overlap
  3. Grammar Quality          – error rate from language-tool-python
  4. Readability Improvement  – Flesch Reading Ease delta (normalised)
  5. Repetition Reduction     – n-gram repetition before vs after
  6. Citation Integrity       – citation recall from original
  7. Sentence Variation       – standard deviation of sentence lengths
  8. Paragraph Coherence      – transition word density

Overall score = weighted average → 0–100 scale.

NOTE: We deliberately do NOT use "AI detector" scores as a primary metric.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import NamedTuple

import numpy as np
import textstat
from loguru import logger
from nltk.tokenize import sent_tokenize

from backend.core.schemas import QualityMetrics, TextAnalysis
from config.settings import settings

# ── Lazy singletons ──────────────────────────────────────────────────────────

_embedding_model = None
_embedding_unavailable = False


def _get_embedding_model():
    global _embedding_model, _embedding_unavailable
    if not settings.enable_local_transformers or _embedding_unavailable:
        return None  # fast Jaccard fallback – avoids slow first-request model loads
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(settings.semantic_model)
            logger.info("SentenceTransformer '{}' loaded.", settings.semantic_model)
        except Exception as exc:
            _embedding_unavailable = True
            logger.warning("Could not load SentenceTransformer: {}. Using fallback.", exc)
    return _embedding_model


_grammar_tool = None


def _get_grammar_tool():
    global _grammar_tool
    if _grammar_tool is None:
        try:
            import language_tool_python
            _grammar_tool = language_tool_python.LanguageTool("en-US")
            logger.info("LanguageTool loaded.")
        except Exception as exc:
            logger.warning("LanguageTool unavailable: {}. Grammar score will be estimated.", exc)
    return _grammar_tool


# ── Individual metric functions ───────────────────────────────────────────────

def _semantic_similarity(original: str, rewritten: str) -> float:
    model = _get_embedding_model()
    if model is None:
        # Fallback: Jaccard similarity on words
        orig_words = set(re.findall(r'\b\w+\b', original.lower()))
        new_words  = set(re.findall(r'\b\w+\b', rewritten.lower()))
        if not orig_words:
            return 1.0
        return len(orig_words & new_words) / len(orig_words | new_words)

    embeddings = model.encode([original[:5000], rewritten[:5000]])
    cos_sim = float(
        np.dot(embeddings[0], embeddings[1]) /
        (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-9)
    )
    return max(0.0, min(1.0, cos_sim))


def _factual_preservation(original: str, rewritten: str) -> float:
    num_pattern = re.compile(r'\b\d[\d,]*(?:\.\d+)?\s*%?')
    orig_nums = set(num_pattern.findall(original))
    new_nums  = set(num_pattern.findall(rewritten))
    if not orig_nums:
        return 1.0
    preserved = len(orig_nums & new_nums)
    return preserved / len(orig_nums)


def _grammar_heuristic(text: str) -> float:
    word_count = len(text.split())
    if word_count == 0:
        return 1.0
    # Heuristic: count sentence fragments and double spaces
    fragments = text.count("  ") + text.count(" ,") + text.count(" .")
    error_rate = min(fragments / word_count, 0.3)
    return 1.0 - error_rate


def _grammar_quality(text: str, enabled: bool = True) -> float:
    """Return 1.0 if perfect, lower for more errors per word."""
    if not enabled:
        return _grammar_heuristic(text)

    tool = _get_grammar_tool()
    word_count = len(text.split())
    if word_count == 0:
        return 1.0

    if tool is None:
        return _grammar_heuristic(text)

    try:
        matches = tool.check(text[:2000])
        # Ignore stylistic suggestions – only count genuine grammar errors
        hard_errors = [
            m for m in matches
            if m.ruleId not in {
                "WHITESPACE_RULE", "EN_QUOTES", "COMMA_PARENTHESIS_WHITESPACE",
                "DOUBLE_PUNCTUATION",
            }
        ]
        error_rate = len(hard_errors) / word_count
        return max(0.0, 1.0 - min(error_rate * 10, 1.0))
    except Exception:
        return 0.8  # graceful degradation


def _readability_improvement(original: str, rewritten: str) -> float:
    """
    Returns a score where 0.5 = neutral, >0.5 = improved, <0.5 = worse.
    Based on Flesch Reading Ease (higher = easier to read).
    """
    orig_score = textstat.flesch_reading_ease(original)
    new_score  = textstat.flesch_reading_ease(rewritten)
    delta      = new_score - orig_score   # positive = easier
    # Normalise: delta of +/-30 → ±0.5 around neutral 0.5
    normalised = 0.5 + (delta / 60.0)
    return max(0.0, min(1.0, normalised))


def _repetition_score(text: str) -> float:
    """Lower repetition → higher score (1.0 = no repeated n-grams)."""
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 6:
        return 1.0
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    total    = len(trigrams)
    counts   = Counter(trigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    rep_rate = repeated / total
    return max(0.0, 1.0 - rep_rate * 2)


def _repetition_reduction(original: str, rewritten: str) -> float:
    orig_rep  = _repetition_score(original)
    new_rep   = _repetition_score(rewritten)
    if orig_rep >= 1.0:
        return 1.0  # nothing to reduce
    # 1.0 = fully reduced, 0.5 = neutral, <0.5 = made it worse
    delta = new_rep - orig_rep
    return max(0.0, min(1.0, 0.5 + delta))


def _citation_integrity(original: str, rewritten: str, analysis: TextAnalysis) -> float:
    if not analysis.citations:
        return 1.0
    present = sum(1 for c in analysis.citations if c in rewritten)
    return present / len(analysis.citations)


def _sentence_variation(text: str) -> float:
    """Higher std deviation of sentence lengths → more natural variation."""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0.5
    lengths = [len(s.split()) for s in sentences]
    std_dev = statistics.stdev(lengths)
    # Ideal std_dev is around 8–15 for natural writing
    score   = min(std_dev / 12.0, 1.0)
    return score


def _paragraph_coherence(text: str) -> float:
    """
    Measure coherence via transition word density in paragraph openings.
    A good score means paragraphs connect logically – but not excessively.
    """
    COHESIVE = {
        "however", "but", "nevertheless", "yet", "although", "while",
        "this", "these", "such", "therefore", "as a result", "consequently",
        "similarly", "in contrast", "on the other hand", "for instance",
        "for example", "building on", "furthermore", "additionally",
    }
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return 1.0

    connected = 0
    for para in paragraphs[1:]:
        first_words = para.lower()[:60]
        if any(first_words.startswith(c) or f" {c} " in first_words for c in COHESIVE):
            connected += 1

    # Ideal: 40–80% of paragraphs have a transition
    ratio = connected / (len(paragraphs) - 1)
    if ratio < 0.2:
        return ratio * 2.5          # too few transitions
    elif ratio > 0.9:
        return 1.0 - (ratio - 0.9) * 5  # too many = AI pattern
    else:
        return 0.5 + ratio * 0.5    # good range


# ── Weights ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "semantic_similarity":    0.25,
    "factual_preservation":   0.20,
    "grammar_quality":        0.15,
    "readability_improvement":0.10,
    "repetition_reduction":   0.10,
    "citation_integrity":     0.10,
    "sentence_variation":     0.05,
    "paragraph_coherence":    0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ── Public class ──────────────────────────────────────────────────────────────

class QualityEvaluator:
    """Evaluate the quality of a humanized text against its original."""

    def evaluate(
        self,
        original: str,
        rewritten: str,
        analysis: TextAnalysis,
        skip_grammar: bool = False,
    ) -> QualityMetrics:
        logger.debug("Running quality evaluation …")

        sem_sim    = _semantic_similarity(original, rewritten)
        factual    = _factual_preservation(original, rewritten)
        grammar    = _grammar_quality(
            rewritten,
            enabled=settings.enable_grammar_check and not skip_grammar,
        )
        readab     = _readability_improvement(original, rewritten)
        rep_red    = _repetition_reduction(original, rewritten)
        cite_int   = _citation_integrity(original, rewritten, analysis)
        sent_var   = _sentence_variation(rewritten)
        para_coh   = _paragraph_coherence(rewritten)

        raw_scores = {
            "semantic_similarity":    sem_sim,
            "factual_preservation":   factual,
            "grammar_quality":        grammar,
            "readability_improvement":readab,
            "repetition_reduction":   rep_red,
            "citation_integrity":     cite_int,
            "sentence_variation":     sent_var,
            "paragraph_coherence":    para_coh,
        }

        overall = sum(raw_scores[k] * WEIGHTS[k] for k in WEIGHTS) * 100

        metrics = QualityMetrics(
            semantic_similarity=round(sem_sim, 3),
            factual_preservation=round(factual, 3),
            grammar_quality=round(grammar, 3),
            readability_improvement=round(readab - 0.5, 3),  # signed: positive = improved
            repetition_reduction=round(rep_red, 3),
            citation_integrity=round(cite_int, 3),
            sentence_variation=round(sent_var, 3),
            paragraph_coherence=round(para_coh, 3),
            overall_score=round(overall, 1),
        )

        logger.info("Quality scores: overall={:.1f} semantic={:.2f}", overall, sem_sim)
        return metrics
