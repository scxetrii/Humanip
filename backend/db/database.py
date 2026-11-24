"""
backend/db/database.py
──────────────────────
SQLAlchemy async database layer using SQLite.

Tables:
  - humanization_history  — every completed humanization run
  - sessions              — optional session metadata

CRUD functions are plain async functions (no class wrapper needed at this scale).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    String,
    Text,
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings

# ── Engine & Session factory ──────────────────────────────────────────────────

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.db_path}",
    echo=settings.db_echo,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class HumanizationRecord(Base):
    __tablename__ = "humanization_history"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(String(64), nullable=True, index=True)

    # Text (snippets to avoid huge DB)
    original_snippet  = Column(String(300))
    humanized_snippet = Column(String(300))
    original_text     = Column(Text)
    humanized_text    = Column(Text)

    # Settings used
    mode              = Column(String(32))
    intensity         = Column(String(16))
    granularity       = Column(String(16))
    provider_used     = Column(String(32))
    model_used        = Column(String(64))
    english_variant   = Column(String(16))

    # Quality metrics (stored as individual columns for quick filtering)
    overall_score        = Column(Float)
    semantic_similarity  = Column(Float)
    factual_preservation = Column(Float)
    grammar_quality      = Column(Float)
    readability_improvement = Column(Float)
    repetition_reduction    = Column(Float)

    # Processing
    processing_time_ms = Column(Integer)

    # Analysis summary (JSON blob)
    analysis_summary = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    deleted    = Column(Boolean, default=False)


# ── Init ──────────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables (idempotent – safe to call on every startup)."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialised at {}", settings.db_path)


# ── CRUD ──────────────────────────────────────────────────────────────────────

async def save_history(
    session_id: str | None,
    original_text: str,
    humanized_text: str,
    mode: str,
    intensity: str,
    granularity: str,
    provider_used: str,
    model_used: str,
    english_variant: str,
    quality: dict[str, Any],
    analysis_summary: dict[str, Any] | None,
    processing_time_ms: int,
) -> int:
    """Insert a new history record and return its id."""
    record = HumanizationRecord(
        session_id=session_id,
        original_snippet=original_text[:300],
        humanized_snippet=humanized_text[:300],
        original_text=original_text[:20_000],   # hard cap
        humanized_text=humanized_text[:20_000],
        mode=mode,
        intensity=intensity,
        granularity=granularity,
        provider_used=provider_used,
        model_used=model_used,
        english_variant=english_variant,
        overall_score=quality.get("overall_score"),
        semantic_similarity=quality.get("semantic_similarity"),
        factual_preservation=quality.get("factual_preservation"),
        grammar_quality=quality.get("grammar_quality"),
        readability_improvement=quality.get("readability_improvement"),
        repetition_reduction=quality.get("repetition_reduction"),
        analysis_summary=analysis_summary,
        processing_time_ms=processing_time_ms,
        created_at=datetime.utcnow(),
    )

    async with AsyncSessionLocal() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id


async def get_history(
    session_id: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[HumanizationRecord], int]:
    """
    Fetch paginated history for a session (or all if session_id is None).
    Returns (records, total_count).
    """
    async with AsyncSessionLocal() as session:
        base_q = select(HumanizationRecord).where(
            HumanizationRecord.deleted == False
        )
        if session_id:
            base_q = base_q.where(HumanizationRecord.session_id == session_id)

        total = await session.scalar(
            select(func.count()).select_from(base_q.subquery())
        )

        rows = await session.execute(
            base_q.order_by(HumanizationRecord.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return list(rows.scalars()), total or 0


async def get_history_item(record_id: int) -> HumanizationRecord | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(HumanizationRecord).where(
                HumanizationRecord.id == record_id,
                HumanizationRecord.deleted == False,
            )
        )
        return result.scalar_one_or_none()


async def soft_delete_history_item(record_id: int) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(HumanizationRecord).where(HumanizationRecord.id == record_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            return False
        record.deleted = True
        await session.commit()
        return True


async def delete_all_history(session_id: str | None = None) -> int:
    """Hard-delete all history (or session-scoped). Returns rows deleted."""
    async with AsyncSessionLocal() as session:
        q = delete(HumanizationRecord)
        if session_id:
            q = q.where(HumanizationRecord.session_id == session_id)
        result = await session.execute(q)
        await session.commit()
        return result.rowcount


async def prune_old_history(retention_days: int | None = None) -> int:
    """Delete records older than retention_days. Returns rows deleted."""
    days = retention_days or settings.history_retention_days
    cutoff = datetime.utcnow() - timedelta(days=days)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            delete(HumanizationRecord).where(
                HumanizationRecord.created_at < cutoff
            )
        )
        await session.commit()
        return result.rowcount
