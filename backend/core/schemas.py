"""
backend/core/schemas.py
───────────────────────
All Pydantic v2 request / response models for the humanizer application.
Shared by the FastAPI layer, the Streamlit frontend, and the internal engine.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class WritingMode(str, Enum):
    """The seven transformation personas the engine supports."""
    NATURAL      = "natural"        # everyday human prose
    ACADEMIC     = "academic"       # scholarly but not stiff
    PROFESSIONAL = "professional"   # polished workplace writing
    SIMPLE       = "simple"         # plain English, wide audience
    RESEARCH     = "research"       # dissertations, papers, reports
    CONVERSATIONAL = "conversational"  # spoken / chat-like
    PERSONAL     = "personal"       # preserve individual voice


class Intensity(str, Enum):
    """How aggressively the engine rewrites."""
    LIGHT    = "light"      # minimal changes, fix obvious issues
    BALANCED = "balanced"   # moderate restructuring (default)
    STRONG   = "strong"     # substantial rewrite


class EnglishVariant(str, Enum):
    AMERICAN = "american"
    BRITISH  = "british"


class RewriteGranularity(str, Enum):
    SENTENCE  = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT  = "document"


# ── Request Models ────────────────────────────────────────────────────────────

class HumanizeRequest(BaseModel):
    """Payload sent by the client to trigger a humanization run."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="The AI-generated text to humanize.",
    )
    mode: WritingMode = WritingMode.NATURAL
    intensity: Intensity = Intensity.BALANCED
    granularity: RewriteGranularity = RewriteGranularity.DOCUMENT

    # Preservation toggles
    preserve_citations: bool = True
    preserve_technical_terms: bool = True
    preserve_original_voice: bool = False

    # Style
    english_variant: EnglishVariant = EnglishVariant.AMERICAN

    # Optional session identifier (used for history)
    session_id: str | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        return v.strip()


class BatchHumanizeRequest(BaseModel):
    """Request to process multiple texts in one call."""
    items: list[HumanizeRequest] = Field(..., min_length=1, max_length=20)


# ── Analysis Models ──────────────────────────────────────────────────────────

class RepetitionIssue(BaseModel):
    phrase: str
    count: int
    locations: list[int]  # character offsets


class ToneIssue(BaseModel):
    sentence: str
    issue_type: str   # e.g. "overly_formal", "robotic_opener"
    suggestion: str


class TextAnalysis(BaseModel):
    """Pre-humanization analysis of the input text."""

    word_count: int
    char_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    readability_score: float        # Flesch Reading Ease
    readability_grade: str          # Flesch-Kincaid Grade level label

    # Issues found
    repetitive_structures: list[RepetitionIssue]
    excessive_transitions: list[str]
    unnatural_phrases: list[str]
    overly_formal_expressions: list[str]
    repetitive_openers: list[str]
    generic_phrases: list[str]
    weak_connections: list[str]
    redundant_explanations: list[str]

    # Extracted entities to preserve
    citations: list[str]
    technical_terms: list[str]
    named_entities: list[str]
    numerical_values: list[str]


# ── Response Models ───────────────────────────────────────────────────────────

class DiffSegment(BaseModel):
    """One segment of an inline diff between original and rewritten text."""
    text: str
    change_type: str   # "equal" | "added" | "removed" | "replaced"


class QualityMetrics(BaseModel):
    """Quality evaluation scores (all 0.0 – 1.0 unless noted)."""

    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    factual_preservation: float = Field(..., ge=0.0, le=1.0)
    grammar_quality: float = Field(..., ge=0.0, le=1.0)
    readability_improvement: float   # can be negative if worse
    repetition_reduction: float = Field(..., ge=0.0, le=1.0)
    citation_integrity: float = Field(..., ge=0.0, le=1.0)
    sentence_variation: float = Field(..., ge=0.0, le=1.0)
    paragraph_coherence: float = Field(..., ge=0.0, le=1.0)

    # Computed overall score (0–100)
    overall_score: float = Field(..., ge=0.0, le=100.0)


class HumanizeResponse(BaseModel):
    """Full response from one humanization run."""

    # Input echo
    original_text: str
    mode: WritingMode
    intensity: Intensity

    # Outputs
    humanized_text: str
    analysis: TextAnalysis
    quality: QualityMetrics
    diff: list[DiffSegment]

    # Metadata
    provider_used: str
    model_used: str
    processing_time_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    history_id: int | None = None


class BatchHumanizeResponse(BaseModel):
    results: list[HumanizeResponse]
    total_processing_time_ms: int


# ── History Models ────────────────────────────────────────────────────────────

class HistoryItem(BaseModel):
    id: int
    session_id: str | None
    original_snippet: str    # first 200 chars
    humanized_snippet: str
    mode: WritingMode
    intensity: Intensity
    overall_score: float
    created_at: datetime


class HistoryListResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


# ── Provider Models ───────────────────────────────────────────────────────────

class ProviderInfo(BaseModel):
    name: str
    is_configured: bool
    is_local: bool
    model: str
    description: str


class ProviderListResponse(BaseModel):
    providers: list[ProviderInfo]
    active_provider: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    provider: str
    db_connected: bool
    nlp_loaded: bool
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
