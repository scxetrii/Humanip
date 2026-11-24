"""
backend/core/analyzer.py
────────────────────────
Pre-humanization NLP analysis engine.

Responsibilities
─────────────────
1. Tokenise and parse the input with spaCy.
2. Detect all categories of AI-writing problems:
   - Repetitive sentence structures / openers
   - Excessive / clichéd transition words
   - Unnatural / robotic vocabulary
   - Overly formal expressions
   - Redundant explanations
   - Weak inter-sentence connections
3. Extract elements that must be PRESERVED:
   - Citations   (APA, MLA, inline, footnote patterns)
   - Technical terms  (noun chunks with capital letters, domain jargon)
   - Named entities  (spaCy NER)
   - Numerical values / statistics
4. Compute readability metrics with textstat.
5. Return a TextAnalysis dataclass consumed by the humanization engine.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

import nltk
import spacy
import textstat
from loguru import logger
from nltk.tokenize import sent_tokenize, word_tokenize

from backend.core.schemas import (
    RepetitionIssue,
    TextAnalysis,
    ToneIssue,
)
from config.settings import settings

# ── Lazy singletons ──────────────────────────────────────────────────────────
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(settings.spacy_model)
            logger.info("spaCy model '{}' loaded.", settings.spacy_model)
        except OSError:
            logger.warning(
                "spaCy model '{}' not found – downloading en_core_web_sm …",
                settings.spacy_model,
            )
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


# ── Linguistic reference lists ───────────────────────────────────────────────

TRANSITION_OVERUSE = {
    "furthermore", "moreover", "additionally", "in addition",
    "in conclusion", "in summary", "to summarize", "it is worth noting",
    "it is important to note", "needless to say", "obviously",
    "clearly", "undeniably", "as a result", "consequently",
    "therefore", "thus", "hence", "accordingly", "subsequently",
    "notably", "significantly", "essentially", "ultimately",
    "firstly", "secondly", "thirdly", "lastly",
}

ROBOTIC_OPENERS = {
    "in today's world", "in today's society", "in modern society",
    "in recent years", "it is widely known", "it is well known",
    "it has been observed", "as we all know", "as mentioned earlier",
    "as previously stated", "as noted above", "as discussed",
    "this paper will", "this essay will", "this report will",
    "the purpose of this", "the aim of this",
}

UNNATURAL_PHRASES = {
    "delve into", "embark on", "leverage", "utilize",
    "in terms of", "with regards to", "with respect to",
    "it goes without saying", "needless to say",
    "at the end of the day", "in light of the fact",
    "due to the fact that", "in spite of the fact that",
    "it is imperative that", "it is crucial that", "it is essential that",
    "it should be noted that", "it must be acknowledged",
    "plays a crucial role", "plays a vital role",
    "it is of utmost importance", "of significant importance",
    "a wide range of", "a variety of", "a number of",
    "various factors", "numerous aspects",
}

OVERLY_FORMAL = {
    "aforementioned", "hereinafter", "henceforth", "heretofore",
    "notwithstanding", "inasmuch as", "insofar as", "therein",
    "thereto", "wherein", "whereas", "whereby", "heretofore",
    "forthwith", "hereunder",
}

GENERIC_INTRO_PHRASES = {
    "this is a topic", "this is an important", "this is a complex",
    "this is a critical", "in the following", "the following section",
    "as we explore", "let us explore", "let us examine",
    "in this section we will",
}

# ── Citation Patterns ────────────────────────────────────────────────────────

CITATION_PATTERNS = [
    # Any parenthetical citation with a year or "n.d.":
    # (Smith, 2020), (Tata Group, n.d.), (BBC News, 2008; Sorabjee, 2024)
    re.compile(
        r'\([^()]{0,300}?(?:\b(?:1[5-9]|20)\d{2}[a-z]?\b|\bn\.d\.)[^()]{0,200}?\)',
        re.IGNORECASE,
    ),
    # APA in-text: (Author, Year) or (Author & Author, Year)
    re.compile(r'\([A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)?(?:\s*et\s*al\.?)?,\s*\d{4}(?:,\s*pp?\.\s*[\d-]+)?\)'),
    # MLA in-text: (Author page)
    re.compile(r'\([A-Z][a-z]+\s+\d+\)'),
    # Numbered footnotes / endnotes: [1], [23]
    re.compile(r'\[\d+\]'),
    # DOI
    re.compile(r'doi:\s*10\.\d{4,}/\S+', re.IGNORECASE),
    # URL citations
    re.compile(r'https?://[^\s,;)]+'),
    # Author (Year) narrative citation
    re.compile(r'[A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\(\d{4}\)'),
]

# ── Numerical Value Pattern ───────────────────────────────────────────────────

NUMERIC_PATTERN = re.compile(
    r"""
    (?:
        \b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%  |   # percentage: 12.5%
        \b\d{1,3}(?:,\d{3})*(?:\.\d+)?           |   # integer / decimal
        \b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b  # written numbers
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ── Analysis helpers ─────────────────────────────────────────────────────────

def _extract_citations(text: str) -> list[str]:
    found: list[str] = []
    for pat in CITATION_PATTERNS:
        for match in pat.finditer(text):
            found.append(match.group())
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def _extract_numerical_values(text: str) -> list[str]:
    return list(dict.fromkeys(NUMERIC_PATTERN.findall(text)))


def _detect_repetitive_structures(sentences: list[str]) -> list[RepetitionIssue]:
    """Find n-gram phrases repeated across sentences."""
    issues: list[RepetitionIssue] = []
    # Check bigrams and trigrams at start of each sentence
    opener_ngrams: Counter = Counter()
    for sent in sentences:
        words = sent.lower().split()
        if len(words) >= 2:
            opener_ngrams[" ".join(words[:2])] += 1
        if len(words) >= 3:
            opener_ngrams[" ".join(words[:3])] += 1

    for phrase, count in opener_ngrams.items():
        if count >= 2 and phrase.split()[0] not in {"the", "a", "an", "this", "that"}:
            locations = [
                i for i, s in enumerate(sentences)
                if s.lower().startswith(phrase)
            ]
            issues.append(RepetitionIssue(phrase=phrase, count=count, locations=locations))

    # Also check full-phrase repetition inside the whole text
    words_all = re.findall(r'\b\w+\b', text_lower := " ".join(sentences).lower())
    all_trigrams: Counter = Counter(
        " ".join(words_all[i:i+3]) for i in range(len(words_all) - 2)
    )
    for phrase, count in all_trigrams.items():
        if count >= 3 and len(phrase) > 10:
            issues.append(
                RepetitionIssue(
                    phrase=phrase,
                    count=count,
                    locations=[m.start() for m in re.finditer(re.escape(phrase), text_lower)],
                )
            )
    return issues


def _detect_excessive_transitions(text_lower: str) -> list[str]:
    found = []
    for phrase in TRANSITION_OVERUSE:
        if phrase in text_lower:
            count = text_lower.count(phrase)
            if count >= 2:
                found.append(f"{phrase} ×{count}")
    return found


def _detect_unnatural_phrases(text_lower: str) -> list[str]:
    return [p for p in UNNATURAL_PHRASES if p in text_lower]


def _detect_formal_expressions(text_lower: str) -> list[str]:
    return [p for p in OVERLY_FORMAL if re.search(r'\b' + re.escape(p) + r'\b', text_lower)]


def _detect_robotic_openers(sentences: list[str]) -> list[str]:
    found = []
    for sent in sentences:
        low = sent.lower().strip()
        for opener in ROBOTIC_OPENERS:
            if low.startswith(opener):
                found.append(sent[:80])
                break
    return found


def _detect_generic_phrases(text_lower: str) -> list[str]:
    return [p for p in GENERIC_INTRO_PHRASES if p in text_lower]


def _detect_weak_connections(sentences: list[str]) -> list[str]:
    """Flag sentence pairs with no cohesive device between them."""
    cohesive_starters = {
        "however", "but", "yet", "although", "while", "this", "these",
        "such", "therefore", "as a result", "consequently", "similarly",
        "in contrast", "on the other hand", "for instance", "for example",
    }
    weak = []
    for i in range(1, len(sentences)):
        first_word = sentences[i].split()[0].lower() if sentences[i].split() else ""
        prev_ends_abruptly = not sentences[i - 1].strip().endswith((":", ";", ","))
        no_link = not any(sentences[i].lower().startswith(c) for c in cohesive_starters)
        if prev_ends_abruptly and no_link and len(sentences[i]) > 40:
            weak.append(sentences[i][:60] + "…")
    return weak[:5]   # limit to 5 examples


def _detect_redundant_explanations(sentences: list[str]) -> list[str]:
    """Find sentences that restate the previous sentence's core idea."""
    redundant = []
    for i in range(1, len(sentences)):
        prev_words = set(re.findall(r'\b\w{5,}\b', sentences[i - 1].lower()))
        curr_words = set(re.findall(r'\b\w{5,}\b', sentences[i].lower()))
        if prev_words and curr_words:
            overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
            if overlap > 0.55:
                redundant.append(sentences[i][:60] + "…")
    return redundant[:5]


def _ai_pattern_score(
    sentences: list[str],
    rep_structures: list,
    exc_transitions: list[str],
    unnatural: list[str],
    formal_expr: list[str],
    robotic_openers: list[str],
    generic: list[str],
    weak_connections: list[str],
    redundant: list[str],
) -> float:
    """Heuristic 0–1 score: higher means the text reads more AI-generated."""
    weighted_issues = (
        len(rep_structures) * 2.0
        + len(exc_transitions) * 1.0
        + len(unnatural) * 1.5
        + len(formal_expr) * 0.75
        + len(robotic_openers) * 1.5
        + len(generic) * 1.5
        + len(weak_connections) * 0.5
        + len(redundant) * 0.75
    )
    density = weighted_issues / max(len(sentences), 1)
    issue_component = min(density / 1.5, 1.0)

    lengths = [len(s.split()) for s in sentences if s.split()]
    if len(lengths) > 1:
        mean = sum(lengths) / len(lengths)
        variance = sum((n - mean) ** 2 for n in lengths) / len(lengths)
        cv = (variance ** 0.5) / mean if mean else 0.0
        uniformity = max(0.0, 1.0 - cv / 0.7)
    else:
        uniformity = 0.5

    return round(min(1.0, 0.75 * issue_component + 0.25 * uniformity), 3)


def _extract_technical_terms(doc: spacy.tokens.Doc) -> list[str]:
    """Extract noun chunks that are likely technical terms."""
    terms: list[str] = []
    for chunk in doc.noun_chunks:
        # Keep multi-word NPs that contain proper nouns or uppercase abbreviations
        if any(t.is_upper and len(t.text) > 1 for t in chunk) or \
                any(t.pos_ == "PROPN" for t in chunk):
            terms.append(chunk.text)
    # Also grab standalone abbreviations: 2+ capital letters
    terms += re.findall(r'\b[A-Z]{2,}\b', doc.text)
    cleaned: list[str] = []
    for term in dict.fromkeys(terms):
        stripped = term.strip()
        if len(stripped) < 2 or not re.search(r'\w', stripped):
            continue
        if stripped.islower():
            continue
        cleaned.append(term)
    return cleaned[:50]


def _named_entities(doc: spacy.tokens.Doc) -> list[str]:
    return list(dict.fromkeys(ent.text for ent in doc.ents))[:50]


def _readability_grade_label(score: float) -> str:
    """Convert Flesch-Kincaid grade level to a readable label."""
    if score <= 6:
        return "Elementary"
    elif score <= 8:
        return "Middle School"
    elif score <= 10:
        return "High School"
    elif score <= 13:
        return "College"
    else:
        return "Post-Graduate"


# ── Public API ────────────────────────────────────────────────────────────────

class TextAnalyzer:
    """Stateless analyser – call .analyze(text) and receive TextAnalysis."""

    def __init__(self) -> None:
        _ensure_nltk()
        self._nlp = _get_nlp()

    def analyze(self, text: str) -> TextAnalysis:
        logger.debug("Analyzing text ({} chars) …", len(text))

        # ── Tokenise ─────────────────────────────────────────────────────────
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences  = sent_tokenize(text)
        words      = word_tokenize(text)
        text_lower = text.lower()

        # ── spaCy parse ──────────────────────────────────────────────────────
        doc = self._nlp(text[:100_000])   # spaCy hard limit guard

        # ── Basic stats ──────────────────────────────────────────────────────
        word_count      = len([w for w in words if w.isalnum()])
        char_count      = len(text)
        sentence_count  = len(sentences)
        para_count      = len(paragraphs)
        avg_sent_len    = word_count / max(sentence_count, 1)

        readability_score = textstat.flesch_reading_ease(text)
        fk_grade          = textstat.flesch_kincaid_grade(text)

        # ── Issue detection ──────────────────────────────────────────────────
        rep_structures    = _detect_repetitive_structures(sentences)
        exc_transitions   = _detect_excessive_transitions(text_lower)
        unnatural         = _detect_unnatural_phrases(text_lower)
        formal_expr       = _detect_formal_expressions(text_lower)
        robotic_openers   = _detect_robotic_openers(sentences)
        generic           = _detect_generic_phrases(text_lower)
        weak_connections  = _detect_weak_connections(sentences)
        redundant         = _detect_redundant_explanations(sentences)

        # ── Preservation extraction ──────────────────────────────────────────
        citations    = _extract_citations(text)
        tech_terms   = _extract_technical_terms(doc)
        named_ents   = _named_entities(doc)
        numerics     = _extract_numerical_values(text)

        ai_pattern_score = _ai_pattern_score(
            sentences, rep_structures, exc_transitions, unnatural,
            formal_expr, robotic_openers, generic, weak_connections, redundant,
        )

        analysis = TextAnalysis(
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            paragraph_count=para_count,
            avg_sentence_length=round(avg_sent_len, 1),
            readability_score=round(readability_score, 1),
            readability_grade=_readability_grade_label(fk_grade),
            ai_pattern_score=ai_pattern_score,
            repetitive_structures=rep_structures,
            excessive_transitions=exc_transitions,
            unnatural_phrases=unnatural,
            overly_formal_expressions=formal_expr,
            repetitive_openers=robotic_openers,
            generic_phrases=generic,
            weak_connections=weak_connections,
            redundant_explanations=redundant,
            citations=citations,
            technical_terms=tech_terms,
            named_entities=named_ents,
            numerical_values=numerics,
        )

        logger.info(
            "Analysis complete: {} words, {} sentences, {} issues found.",
            word_count,
            sentence_count,
            (
                len(rep_structures)
                + len(exc_transitions)
                + len(unnatural)
                + len(formal_expr)
            ),
        )
        return analysis
