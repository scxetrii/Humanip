"""
backend/api/health.py
─────────────────────
Health check and application status endpoints.
"""
from __future__ import annotations

import time

from fastapi import APIRouter

from backend.core.schemas import HealthResponse
from backend.providers.base import ProviderRegistry
from config.settings import settings

router = APIRouter()

_startup_time = time.perf_counter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """Returns the application health status and basic diagnostics."""
    # Test DB connectivity
    db_ok = True
    try:
        from backend.db.database import AsyncSessionLocal
        from sqlalchemy import text
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    # Test NLP is loaded
    nlp_ok = True
    try:
        import spacy
        spacy.load(settings.spacy_model)
    except Exception:
        nlp_ok = False

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        provider=ProviderRegistry.active().name,
        db_connected=db_ok,
        nlp_loaded=nlp_ok,
        uptime_seconds=round(time.perf_counter() - _startup_time, 1),
    )
