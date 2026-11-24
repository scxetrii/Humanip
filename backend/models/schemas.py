"""
backend/models/schemas.py
──────────────────────────
All Pydantic models used for API request/response validation and
internal data transfer objects (DTOs).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ════════════════════════════════════════════════════════════════════════
# Enumerations
# ════════════════════════════════════════════════════════════════════════

class WritingMode(str, Enum):
    NATURAL       = "natural"        # Everyday natural writing
    ACADEMIC      = "academic"       # Academic language, structured
    PROFESSIONAL  = "professional"   # Polished workplace communication
    SIMPLE        = "simple"         # Plain English, accessible
    RESEARCH      = "research"       # Dissertations, formal reports
    CONVERSATIONAL= "conversational" # Relaxed, dialogue-like
    PERSONAL      = "personal"       # First-person individual voice


class TransformIntensity(str, Enum):
    LIGHT    = "light"    # Minor polish – structural edits only
    BALANCED = "balanced" # Moderate rewrite – default
    STRONG   = "strong"   # Thorough rewrite – maximum humanization


class EnglishVariant(str, Enum):
    AMERICAN = "american"
    BRITISH  = "british"


class LLMProvider(str, Enum):
    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    GOOGLE     = "google"
    OLLAMA     = "ollama"
    CUSTOM     = "custom"


class RewriteScope(str, Enum):
    SENTENCE  = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT  = "document"


# ════════════════════════════════════════════════════════════════════════
# Text Analysis
# ════════════════════════════════════════════════════════════════════════

class TextStats(BaseModel):
    """Word-level and readability statistics for a block of text."""
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    flesch_reading_ease: float = 0.0        # 0–100; higher = easier
    flesch_kincaid_grade: float = 0.0       # US school grade level
    gunning_fog: float = 0.0
    automated_readability_index: float = 0.0


class AnalysisIssue(BaseModel):
    """A single writing-quality issue found in the input."""
    issue_type: str                         # e.g. "repetitive_opener"
    description: str
    severity: str = "medium"               # low | medium | high
    location: Optional[str] = None         # paragraph or sentence reference
    suggestion: Optional[str] = None


class TextAnalysis(BaseModel):
    """Full NLP pre-analysis of the input text."""
    stats: TextStats
    issues: list[AnalysisIssue] = []
    detected_citations: list[str] = []
    detected_technical_terms: list[str] = []
    repetitive_phrases: list[str] = []
    dominant_tone: str = "neutral"         # academic | formal | casual | …
    ai_pattern_score: float = 0.0          # 0–1; how "AI-like" the text reads
    language_detected: str = "en"


# ════════════════════════════════════════════════════════════════════════
# Quality Evaluation
# ════════════════════════════════════════════════════════════════════════

class FactChange(BaseModel):
    """Tracks a potential factual difference between original and rewritten."""
    original_fragment: str
    rewritten_fragment: str
    change_type: str    # "number_changed" | "citation_altered" | "claim_added"
    severity: str = "medium"


class QualityReport(BaseModel):
    """Scores produced by the quality evaluation engine."""
    semantic_similarity: float      # 0–1; cosine similarity via sentence-transformers
    factual_preservation: float     # 0–1; fact-check pass rate
    grammar_quality: float          # 0–1; grammar error rate inverted
    readability_improvement: float  # delta Flesch score / 100
    repetition_reduction: float     # 0–1; how much repetition was removed
    citation_integrity: float       # 1.0 if all citations unchanged, else <1
    sentence_variety: float         # 0–1; structural diversity score
    paragraph_coherence: float      # 0–1; local coherence (entity continuity)
    overall_score: float            # weighted composite
    fact_changes: list[FactChange] = []
    warnings: list[str] = []


# ════════════════════════════════════════════════════════════════════════
# Diff / Comparison
# ════════════════════════════════════════════════════════════════════════

class DiffSegment(BaseModel):
    """One segment of a word-level diff between original and rewritten text."""
    text: str
    change: str  # "equal" | "insert" | "delete" | "replace"


class TextDiff(BaseModel):
    segments: list[DiffSegment]
    added_words: int = 0
    removed_words: int = 0
    changed_sentences: int = 0


# ════════════════════════════════════════════════════════════════════════
# Humanization Request / Response
# ════════════════════════════════════════════════════════════════════════

class HumanizeRequest(BaseModel):
    """
    Payload sent by the client to trigger humanization.
    All preservation toggles default to True — the engine must
    actively opt out if explicitly disabled.
    """
    text: str = Field(..., min_length=10, max_length=50_000)
    mode: WritingMode = WritingMode.NATURAL
    intensity: TransformIntensity = TransformIntensity.BALANCED
    scope: RewriteScope = RewriteScope.DOCUMENT
    english_variant: EnglishVariant = EnglishVariant.AMERICAN

    # Preservation toggles
    preserve_citations: bool = True
    preserve_technical_terms: bool = True
    preserve_original_voice: bool = False  # only relevant in PERSONAL mode
    preserve_numbers: bool = True          # always enforced regardless

    # Provider override (None → use server default)
    llm_provider: Optional[LLMProvider] = None
    llm_model: Optional[str] = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return v.strip()


class HumanizeResponse(BaseModel):
    """Full response from a single humanization job."""
    request_id: str
    original_text: str
    humanized_text: str
    mode: WritingMode
    intensity: TransformIntensity
    provider_used: str
    model_used: str

    # Analysis of the original
    original_analysis: TextAnalysis

    # Stats on the output
    output_stats: TextStats

    # Quality evaluation
    quality: QualityReport

    # Word-level diff
    diff: TextDiff

    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: int = 0


# ════════════════════════════════════════════════════════════════════════
# Sentence & Paragraph Granular Rewrite
# ════════════════════════════════════════════════════════════════════════

class SentenceRewriteRequest(BaseModel):
    sentence: str = Field(..., min_length=5, max_length=2_000)
    mode: WritingMode = WritingMode.NATURAL
    intensity: TransformIntensity = TransformIntensity.BALANCED
    context: Optional[str] = None  # surrounding sentences for context


class SentenceRewriteResponse(BaseModel):
    original: str
    rewritten: str
    explanation: Optional[str] = None


class ParagraphRewriteRequest(BaseModel):
    paragraph: str = Field(..., min_length=10, max_length=5_000)
    mode: WritingMode = WritingMode.NATURAL
    intensity: TransformIntensity = TransformIntensity.BALANCED
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class ParagraphRewriteResponse(BaseModel):
    original: str
    rewritten: str
    issues_fixed: list[str] = []


# ════════════════════════════════════════════════════════════════════════
# Batch Processing
# ════════════════════════════════════════════════════════════════════════

class BatchItem(BaseModel):
    item_id: str
    text: str = Field(..., min_length=10, max_length=50_000)


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=20)
    mode: WritingMode = WritingMode.NATURAL
    intensity: TransformIntensity = TransformIntensity.BALANCED
    preserve_citations: bool = True
    preserve_technical_terms: bool = True


class BatchItemResult(BaseModel):
    item_id: str
    original_text: str
    humanized_text: str
    quality_score: float
    error: Optional[str] = None


class BatchResponse(BaseModel):
    batch_id: str
    results: list[BatchItemResult]
    total_items: int
    successful: int
    failed: int
    total_processing_time_ms: int


# ════════════════════════════════════════════════════════════════════════
# History
# ════════════════════════════════════════════════════════════════════════

class HistoryEntry(BaseModel):
    id: int
    request_id: str
    mode: str
    intensity: str
    provider: str
    model: str
    original_word_count: int
    output_word_count: int
    quality_score: float
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryListResponse(BaseModel):
    items: list[HistoryEntry]
    total: int
    page: int
    page_size: int


class HistoryDetailResponse(BaseModel):
    entry: HistoryEntry
    original_text: str
    humanized_text: str
    quality: QualityReport


# ════════════════════════════════════════════════════════════════════════
# LLM Provider Config (runtime, not persisted with secrets)
# ════════════════════════════════════════════════════════════════════════

class ProviderInfo(BaseModel):
    provider: LLMProvider
    model: str
    is_available: bool
    error: Optional[str] = None


class ProvidersListResponse(BaseModel):
    providers: list[ProviderInfo]
    default_provider: str


# ════════════════════════════════════════════════════════════════════════
# File Upload
# ════════════════════════════════════════════════════════════════════════

class FileUploadResponse(BaseModel):
    filename: str
    extracted_text: str
    word_count: int
    char_count: int
    file_type: str
    warnings: list[str] = []


# ════════════════════════════════════════════════════════════════════════
# Generic API Responses
# ════════════════════════════════════════════════════════════════════════

class SuccessResponse(BaseModel):
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
