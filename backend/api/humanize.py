"""
backend/api/humanize.py
───────────────────────
Main humanization endpoints.

POST /api/v1/humanize          → single text humanization
POST /api/v1/humanize/batch    → batch processing (up to 20 texts)
POST /api/v1/analyze           → analysis only (no LLM call)
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend.core.analyzer import TextAnalyzer
from backend.core.engine import engine
from backend.core.schemas import (
    BatchHumanizeRequest,
    BatchHumanizeResponse,
    HumanizeRequest,
    HumanizeResponse,
    TextAnalysis,
)
from backend.db import database as db
from backend.providers.base import LLMError, ProviderRegistry
from loguru import logger

router = APIRouter()

_analyzer = TextAnalyzer()

_PROVIDER_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_provider_ready() -> None:
    """Fail fast with an actionable error instead of a confusing 502."""
    try:
        provider = ProviderRegistry.active()
    except LLMError as exc:
        raise HTTPException(503, detail=str(exc))

    if provider.is_configured:
        return

    key_var = _PROVIDER_KEY_VARS.get(provider.name)
    hint = (
        f" Set {key_var} in .env and restart the server."
        if key_var
        else f" Configure '{provider.name}' in .env and restart the server."
    )
    raise HTTPException(
        400,
        detail=f"Provider '{provider.name}' is not configured.{hint}",
    )


async def _persist(response: HumanizeResponse, session_id: str | None) -> None:
    """Background task: save a completed humanization to history."""
    try:
        record_id = await db.save_history(
            session_id=session_id,
            original_text=response.original_text,
            humanized_text=response.humanized_text,
            mode=response.mode.value,
            intensity=response.intensity.value,
            granularity="document",
            provider_used=response.provider_used,
            model_used=response.model_used,
            english_variant="american",
            quality=response.quality.model_dump(),
            analysis_summary={
                "word_count": response.analysis.word_count,
                "sentence_count": response.analysis.sentence_count,
                "readability_score": response.analysis.readability_score,
                "issues_found": (
                    len(response.analysis.unnatural_phrases)
                    + len(response.analysis.repetitive_structures)
                ),
            },
            processing_time_ms=response.processing_time_ms,
        )
        logger.debug("History saved: id={}", record_id)
    except Exception as exc:
        logger.warning("Failed to save history: {}", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/humanize",
    response_model=HumanizeResponse,
    summary="Humanize a single text",
)
async def humanize_text(
    request: HumanizeRequest,
    background_tasks: BackgroundTasks,
) -> HumanizeResponse:
    """
    Transform AI-generated text into natural human writing.

    - Select a **writing mode** (natural, academic, professional, …)
    - Choose transformation **intensity** (light / balanced / strong)
    - Toggle citation and technical term preservation
    - Optionally specify American or British English
    """
    _ensure_provider_ready()
    response = await engine.run(request)
    background_tasks.add_task(_persist, response, request.session_id)
    return response


@router.post(
    "/humanize/batch",
    response_model=BatchHumanizeResponse,
    summary="Humanize multiple texts (batch)",
)
async def humanize_batch(request: BatchHumanizeRequest) -> BatchHumanizeResponse:
    """Process up to 20 texts in one API call (processed concurrently)."""
    if len(request.items) > 20:
        raise HTTPException(422, "Batch size cannot exceed 20 items.")

    _ensure_provider_ready()
    results = await engine.run_batch(request.items)
    total_ms = sum(r.processing_time_ms for r in results)
    return BatchHumanizeResponse(results=results, total_processing_time_ms=total_ms)


@router.post(
    "/analyze",
    response_model=TextAnalysis,
    summary="Analyse text without rewriting",
)
async def analyze_text(payload: dict) -> TextAnalysis:
    """
    Run the NLP analysis pipeline on a text without calling the LLM.
    Returns detected issues, citations, technical terms, and readability scores.
    """
    text = payload.get("text", "")
    if not text or len(text.strip()) < 10:
        raise HTTPException(422, "Provide at least 10 characters of text.")

    return _analyzer.analyze(text)
