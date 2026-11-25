"""
backend/main.py
───────────────
FastAPI application entry point.

Registers:
  - Startup / shutdown lifecycle hooks (DB init, provider registration)
  - CORS middleware (for Streamlit frontend)
  - All API routers
  - Global exception handlers
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

from backend.db.database import init_db, prune_old_history
from backend.providers.base import (
    LLMConfigError,
    LLMError,
    LLMTimeoutError,
    ProviderRegistry,
)
from config.logging_config import configure_logging
from config.settings import settings

# ── Startup time ─────────────────────────────────────────────────────────────
_start_time = time.perf_counter()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    configure_logging()
    logger.info("═" * 60)
    logger.info(" {} v{} starting up …", settings.app_name, settings.app_version)
    logger.info("═" * 60)

    # ── Database ──────────────────────────────────────────────────────────────
    await init_db()
    await prune_old_history()

    # ── Import providers (triggers auto-registration) ─────────────────────────
    import backend.providers.openai_provider      # noqa: F401
    import backend.providers.anthropic_provider   # noqa: F401
    import backend.providers.other_providers      # noqa: F401

    logger.info(
        "Providers registered: {}", ProviderRegistry.all_names()
    )
    logger.info("Active provider: {}", settings.llm_provider)

    try:
        active_provider = ProviderRegistry.active()
    except LLMError as exc:
        logger.error("Active provider is unavailable: {}", exc)
    else:
        if not active_provider.is_configured:
            logger.warning(
                "Active provider '{}' is NOT configured. Humanization requests "
                "will fail until its API key is set in .env and the server is restarted.",
                active_provider.name,
            )

    logger.info("API listening on http://{}:{}", settings.api_host, settings.api_port)

    yield  # ── App running ──────────────────────────────────────────────────

    logger.info("Shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Humanip API – advanced AI text humanization that converts "
        "AI-generated text into natural, fluent human prose while preserving "
        "all factual content."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    if isinstance(exc, LLMTimeoutError):
        status_code, error_type = 504, "llm_timeout"
    elif isinstance(exc, LLMConfigError):
        status_code, error_type = 400, "provider_not_configured"
    else:
        status_code, error_type = 502, "llm_error"

    logger.error("LLM error ({}): {}", error_type, exc)
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exc), "error_type": error_type},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "validation_error"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on {}", request.url)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "error_type": "internal_error",
        },
    )


# ── Register routers ──────────────────────────────────────────────────────────

from backend.api import health, humanize, history, providers, upload  # noqa: E402
from backend.api.humanize_stream import router as stream_router  # noqa: E402

app.include_router(health.router,     prefix="",          tags=["Health"])
app.include_router(humanize.router,   prefix="/api/v1",   tags=["Humanize"])
app.include_router(stream_router,     prefix="/api/v1",   tags=["Streaming"])
app.include_router(history.router,    prefix="/api/v1",   tags=["History"])
app.include_router(providers.router,  prefix="/api/v1",   tags=["Providers"])
app.include_router(upload.router,     prefix="/api/v1",   tags=["Upload"])

# ── Serve frontend static files ───────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR / "static")),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(
                str(index),
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )
        return JSONResponse({"app": settings.app_name, "docs": "/docs"})
