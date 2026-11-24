"""
backend/core/engine.py
──────────────────────
Central humanization orchestrator.

Flow:
  1. Receive a HumanizeRequest.
  2. Run TextAnalyzer → TextAnalysis.
  3. Apply pre-processing (citation extraction, placeholder injection).
  4. Build prompts (system + user) for the selected writing mode / intensity.
  5. Send to the active LLM provider.
  6. Post-process: restore preserved elements, enforce spelling variant.
  7. Run QualityEngine → QualityMetrics.
  8. Build diff.
  9. Persist to history (async).
 10. Return HumanizeResponse.

For SENTENCE granularity the engine rewrites each sentence individually
with surrounding context to maintain coherence.

For PARAGRAPH granularity the engine rewrites each paragraph individually.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from datetime import datetime

from loguru import logger
from nltk.tokenize import sent_tokenize

from backend.core.analyzer import TextAnalyzer
from backend.core.prompts import (
    build_sentence_prompt,
    build_system_prompt,
    build_user_prompt,
)
from backend.core.schemas import (
    DiffSegment,
    HumanizeRequest,
    HumanizeResponse,
    Intensity,
    QualityMetrics,
    RewriteGranularity,
    TextAnalysis,
    WritingMode,
)
from backend.preservation.guardian import PreservationGuardian
from backend.providers.base import ProviderRegistry
from backend.quality.evaluator import QualityEvaluator
from config.settings import settings

# ── Lazy singletons ──────────────────────────────────────────────────────────

_analyzer:  TextAnalyzer   | None = None
_quality:   QualityEvaluator | None = None
_guardian:  PreservationGuardian | None = None


def _get_analyzer() -> TextAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TextAnalyzer()
    return _analyzer


def _get_quality() -> QualityEvaluator:
    global _quality
    if _quality is None:
        _quality = QualityEvaluator()
    return _quality


def _get_guardian() -> PreservationGuardian:
    global _guardian
    if _guardian is None:
        _guardian = PreservationGuardian()
    return _guardian


# ── Spelling variant post-processing ─────────────────────────────────────────

_US_TO_UK = {
    "color": "colour", "flavor": "flavour", "honor": "honour",
    "humor": "humour", "labor": "labour", "neighbor": "neighbour",
    "recognize": "recognise", "organize": "organise",
    "analyze": "analyse", "apologize": "apologise",
    "realize": "realise", "specialize": "specialise",
    "center": "centre", "theater": "theatre",
    "traveling": "travelling", "canceled": "cancelled",
    "defense": "defence", "offense": "offence",
    "license": "licence",
}
_UK_TO_US = {v: k for k, v in _US_TO_UK.items()}


def _apply_spelling_variant(text: str, variant: str) -> str:
    mapping = _UK_TO_US if variant == "american" else _US_TO_UK
    for original, replacement in mapping.items():
        text = re.sub(r'\b' + re.escape(original) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


# ── Diff builder ──────────────────────────────────────────────────────────────

def _build_diff(original: str, rewritten: str) -> list[DiffSegment]:
    """
    Word-level diff between original and rewritten text.
    Returns a list of DiffSegment for the frontend to render.
    """
    import difflib
    orig_words = original.split()
    new_words  = rewritten.split()

    segments: list[DiffSegment] = []
    matcher = difflib.SequenceMatcher(None, orig_words, new_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(DiffSegment(
                text=" ".join(orig_words[i1:i2]) + " ",
                change_type="equal"
            ))
        elif tag == "replace":
            segments.append(DiffSegment(
                text=" ".join(orig_words[i1:i2]) + " ",
                change_type="removed"
            ))
            segments.append(DiffSegment(
                text=" ".join(new_words[j1:j2]) + " ",
                change_type="added"
            ))
        elif tag == "delete":
            segments.append(DiffSegment(
                text=" ".join(orig_words[i1:i2]) + " ",
                change_type="removed"
            ))
        elif tag == "insert":
            segments.append(DiffSegment(
                text=" ".join(new_words[j1:j2]) + " ",
                change_type="added"
            ))

    return segments


# ── Sentence-level rewriting ──────────────────────────────────────────────────

async def _rewrite_by_sentence(
    text: str,
    analysis: TextAnalysis,
    request: HumanizeRequest,
    system_prompt: str,
) -> str:
    """Rewrite each sentence individually with neighbouring context."""
    provider = ProviderRegistry.active()
    sentences = sent_tokenize(text)
    rewritten: list[str] = []

    # Chunk to avoid too many parallel calls
    sem = asyncio.Semaphore(3)

    async def _do_one(idx: int) -> str:
        async with sem:
            ctx_before = " ".join(sentences[max(0, idx - 2):idx])
            ctx_after  = " ".join(sentences[idx + 1:idx + 3])
            user_prompt = build_sentence_prompt(
                sentences[idx], ctx_before, ctx_after,
                request.mode, request.intensity
            )
            resp = await provider.complete(system_prompt, user_prompt, max_tokens=512)
            return resp.text

    tasks   = [_do_one(i) for i in range(len(sentences))]
    results = await asyncio.gather(*tasks)
    return " ".join(results)


# ── Paragraph-level rewriting ─────────────────────────────────────────────────

async def _rewrite_by_paragraph(
    text: str,
    analysis: TextAnalysis,
    request: HumanizeRequest,
    system_prompt: str,
) -> str:
    """Rewrite each paragraph individually."""
    provider   = ProviderRegistry.active()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    rewritten: list[str] = []

    for para in paragraphs:
        user_prompt = build_user_prompt(
            para, analysis,
            request.preserve_citations,
            request.preserve_technical_terms,
        )
        resp = await provider.complete(system_prompt, user_prompt)
        rewritten.append(resp.text)

    return "\n\n".join(rewritten)


# ════════════════════════════════════════════════════════════════════════════════
# Main Engine
# ════════════════════════════════════════════════════════════════════════════════

class HumanizationEngine:
    """
    Stateless orchestrator.  Instantiate once at startup and call .run() per request.
    All heavy singleton objects (analyzer, quality, guardian) are lazy-loaded.
    """

    async def run(self, request: HumanizeRequest) -> HumanizeResponse:
        t_start = time.perf_counter()
        logger.info(
            "Humanize request | mode={} intensity={} granularity={} len={}",
            request.mode, request.intensity, request.granularity, len(request.text)
        )

        # ── Step 1: Analyse ───────────────────────────────────────────────────
        analyzer = _get_analyzer()
        analysis = analyzer.analyze(request.text)

        # ── Step 2: Pre-process (extract & replace preserved elements) ────────
        guardian = _get_guardian()
        guarded_text, placeholder_map = guardian.protect(
            request.text, analysis,
            protect_citations=request.preserve_citations,
            protect_terms=request.preserve_technical_terms,
        )

        # ── Step 3: Build prompts ─────────────────────────────────────────────
        system_prompt = build_system_prompt(
            request.mode, request.intensity, request.english_variant
        )
        provider = ProviderRegistry.active()

        # ── Step 4: Call LLM (granularity-aware) ─────────────────────────────
        if request.granularity == RewriteGranularity.SENTENCE:
            raw_output = await _rewrite_by_sentence(
                guarded_text, analysis, request, system_prompt
            )
        elif request.granularity == RewriteGranularity.PARAGRAPH:
            raw_output = await _rewrite_by_paragraph(
                guarded_text, analysis, request, system_prompt
            )
        else:  # DOCUMENT
            user_prompt = build_user_prompt(
                guarded_text, analysis,
                request.preserve_citations,
                request.preserve_technical_terms,
            )
            llm_resp   = await provider.complete(system_prompt, user_prompt)
            raw_output = llm_resp.text

        # ── Step 5: Post-process ──────────────────────────────────────────────
        restored = guardian.restore(raw_output, placeholder_map)
        final    = _apply_spelling_variant(restored, request.english_variant.value)

        # ── Step 6: Quality evaluation ────────────────────────────────────────
        quality_engine = _get_quality()
        quality        = quality_engine.evaluate(request.text, final, analysis)

        # ── Step 7: Build diff ────────────────────────────────────────────────
        diff = _build_diff(request.text, final)

        # ── Step 8: Determine provider metadata ───────────────────────────────
        active_provider = ProviderRegistry.active()
        elapsed_ms      = int((time.perf_counter() - t_start) * 1000)

        response = HumanizeResponse(
            original_text=request.text,
            mode=request.mode,
            intensity=request.intensity,
            humanized_text=final,
            analysis=analysis,
            quality=quality,
            diff=diff,
            provider_used=active_provider.name,
            model_used=active_provider.model,
            processing_time_ms=elapsed_ms,
            created_at=datetime.utcnow(),
        )

        logger.info(
            "Humanize done | {}ms | quality={:.1f}",
            elapsed_ms,
            quality.overall_score,
        )
        return response

    async def run_batch(
        self, requests: list[HumanizeRequest]
    ) -> list[HumanizeResponse]:
        """Process multiple requests concurrently (max 3 at a time)."""
        sem = asyncio.Semaphore(3)

        async def _bounded(req: HumanizeRequest) -> HumanizeResponse:
            async with sem:
                return await self.run(req)

        return list(await asyncio.gather(*[_bounded(r) for r in requests]))


# ── Module-level singleton ────────────────────────────────────────────────────
engine = HumanizationEngine()
