"""
backend/api/humanize_stream.py
───────────────────────────────
Streaming humanization endpoint.

POST /api/v1/humanize/stream — Server-Sent Events stream

Why streaming matters for speed perception
───────────────────────────────────────────
A 5000-word document with 4 chunks at 4s each = 16s total.
Without streaming: the user stares at a spinner for 16s.
With streaming:    the first rewritten section appears at 4s.

SSE event format
─────────────────
Each event is JSON on a `data:` line followed by two newlines:

    data: {"type": "start", "total_chunks": 4, "word_count": 5000}
    data: {"type": "chunk", "index": 0, "text": "...", "total_chunks": 4,
           "completed": 1}
    data: {"type": "done",  "humanized_text": "...", "human_score": 82.5,
           "processing_time_ms": 6200, "quality": {...}, "diff": [...],
           "analysis": {...}}
    data: {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.api.humanize import _persist
from backend.core.engine import (
    _Deadline,
    _analysis_for_chunk,
    _apply_spelling_variant,
    _build_diff,
    _complete_ensuring_rewrite,
    _get_analyzer,
    _get_guardian,
    _get_quality,
    _neutral_quality,
    _split_into_chunks,
)
from backend.core.prompts import build_system_prompt, build_user_prompt
from backend.core.schemas import HumanizeRequest, HumanizeResponse
from backend.providers.base import (
    LLMConfigError,
    LLMError,
    ProviderRegistry,
)
from config.settings import settings

router = APIRouter()


def _sse(payload: dict) -> str:
    """Format one SSE event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_humanize(request: HumanizeRequest):
    """
    Core streaming generator.
    Yields SSE-formatted strings as each chunk completes.
    """
    t_start = time.perf_counter()

    try:
        provider = ProviderRegistry.active()
    except LLMError as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    if not provider.is_configured:
        yield _sse({
            "type": "error",
            "message": (
                f"Provider '{provider.name}' is not configured. "
                f"Set its API key in .env and restart the server."
            ),
        })
        return

    analyzer = _get_analyzer()
    guardian = _get_guardian()
    quality_engine = _get_quality()

    # ── Analyse + guard ───────────────────────────────────────────────────────
    analysis = analyzer.analyze(request.text)
    guarded_text, placeholder_map = guardian.protect(
        request.text, analysis,
        protect_citations=request.preserve_citations,
        protect_terms=request.preserve_technical_terms,
    )
    system_prompt = build_system_prompt(
        request.mode, request.intensity, request.english_variant
    )
    deadline = _Deadline(settings.humanize_timeout)

    # ── Split into chunks ─────────────────────────────────────────────────────
    chunks = _split_into_chunks(
        guarded_text,
        settings.chunk_max_words,
        settings.chunk_max_chars,
    )
    if not chunks:
        chunks = [guarded_text]

    total = len(chunks)
    yield _sse({"type": "start", "total_chunks": total,
                "word_count": analysis.word_count})

    # ── Process chunks (concurrent) ───────────────────────────────────────────
    sem = asyncio.Semaphore(max(1, settings.chunk_concurrency))
    rewritten: list[str | None] = [None] * total
    done_queue: asyncio.Queue[tuple[int, str | None, BaseException | None]] = (
        asyncio.Queue()
    )

    async def _process_chunk(index: int, chunk: str) -> None:
        async with sem:
            if deadline.remaining() < 2.0:
                await done_queue.put((index, chunk, None))
                return

            chunk_analysis = _analysis_for_chunk(analysis, chunk)
            user_prompt = build_user_prompt(
                chunk, chunk_analysis,
                request.preserve_citations,
                request.preserve_technical_terms,
            )
            words = len(chunk.split())
            max_tokens = max(512, min(settings.llm_max_tokens, int(words * 2.2) + 128))

            try:
                result = await _complete_ensuring_rewrite(
                    provider, system_prompt, user_prompt, chunk,
                    max_tokens=max_tokens, max_attempts=2,
                    deadline=deadline,
                    per_call_timeout=float(settings.chunk_timeout),
                )
            except LLMConfigError as exc:
                await done_queue.put((index, None, exc))
                return
            except LLMError as exc:
                logger.warning("Stream chunk {} failed ({}); keeping original", index, exc)
                result = chunk
            await done_queue.put((index, result, None))

    tasks = [
        asyncio.create_task(_process_chunk(i, chunk))
        for i, chunk in enumerate(chunks)
    ]

    # ── Stream results as they arrive ─────────────────────────────────────────
    completed = 0
    try:
        while completed < total:
            try:
                index, text, fatal = await asyncio.wait_for(
                    done_queue.get(),
                    timeout=settings.humanize_timeout + 10,
                )
            except asyncio.TimeoutError:
                break

            if fatal is not None:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                yield _sse({"type": "error", "message": str(fatal)})
                return

            assert text is not None
            rewritten[index] = text
            completed += 1

            chunk_restored = guardian.restore(text, placeholder_map)
            chunk_final = _apply_spelling_variant(
                chunk_restored, request.english_variant.value
            )

            yield _sse({
                "type": "chunk",
                "index": index,
                "text": chunk_final,
                "total_chunks": total,
                "completed": completed,
            })
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Fill any gaps (chunks that never completed)
    for i, value in enumerate(rewritten):
        if value is None:
            rewritten[i] = chunks[i]

    # ── Assemble final text ───────────────────────────────────────────────────
    raw_full = "\n\n".join(rewritten)
    restored = guardian.restore(raw_full, placeholder_map)
    final = _apply_spelling_variant(restored, request.english_variant.value)

    # ── Human-likeness of the rewritten text ──────────────────────────────────
    output_analysis = analyzer.analyze(final)
    human_score = round((1.0 - output_analysis.ai_pattern_score) * 100, 1)

    # ── Quality evaluation ────────────────────────────────────────────────────
    try:
        quality = await asyncio.wait_for(
            asyncio.to_thread(
                quality_engine.evaluate, request.text, final, analysis
            ),
            timeout=settings.quality_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("Quality evaluation timed out; using heuristic scores.")
        try:
            quality = await asyncio.wait_for(
                asyncio.to_thread(
                    quality_engine.evaluate, request.text, final, analysis, True
                ),
                timeout=settings.quality_timeout,
            )
        except asyncio.TimeoutError:
            quality = _neutral_quality()

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    diff = _build_diff(request.text, final)

    response = HumanizeResponse(
        original_text=request.text,
        mode=request.mode,
        intensity=request.intensity,
        humanized_text=final,
        analysis=analysis,
        quality=quality,
        diff=diff,
        human_score=human_score,
        provider_used=provider.name,
        model_used=provider.model,
        processing_time_ms=elapsed_ms,
    )
    asyncio.create_task(_persist(response, request.session_id))

    yield _sse({
        "type": "done",
        "humanized_text": final,
        "human_score": human_score,
        "processing_time_ms": elapsed_ms,
        "quality": quality.model_dump(),
        "diff": [segment.model_dump() for segment in diff],
        "analysis": analysis.model_dump(),
        "provider_used": provider.name,
        "model_used": provider.model,
    })


@router.post(
    "/humanize/stream",
    summary="Humanize with streaming (Server-Sent Events)",
)
async def humanize_stream(request_body: HumanizeRequest):
    """
    Stream humanization results chunk-by-chunk via SSE.
    The client receives each rewritten section as it completes,
    instead of waiting for the full document.
    """
    return StreamingResponse(
        _stream_humanize(request_body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Nginx: disable buffering
            "Connection": "keep-alive",
        },
    )
