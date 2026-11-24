"""
backend/models/database.py
───────────────────────────
SQLAlchemy ORM definitions and async database engine setup.
Uses SQLite via aiosqlite for fully async I/O.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    String, Text, JSON, func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings

settings = get_settings()


# ── Engine & Session Factory ──────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    connect_args={"check_same_thread": False},  # required for SQLite
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Base Model ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class HumanizationRecord(Base):
    """
    Persists every completed humanization job.
    Full text bodies are stored to allow history retrieval.
    Documents are kept private — the delete endpoint removes rows + blobs.
    """
    __tablename__ = "humanization_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    request_id      = Column(String(36), unique=True, nullable=False, index=True)

    # Settings used
    mode            = Column(String(32), nullable=False)
    intensity       = Column(String(16), nullable=False)
    scope           = Column(String(16), nullable=False, default="document")
    english_variant = Column(String(16), nullable=False, default="american")
    provider        = Column(String(32), nullable=False)
    model           = Column(String(64), nullable=False)

    # Preservation flags
    preserve_citations      = Column(Boolean, default=True)
    preserve_technical_terms= Column(Boolean, default=True)

    # Text (stored encrypted at rest in production — handled by DB layer)
    original_text   = Column(Text, nullable=False)
    humanized_text  = Column(Text, nullable=False)

    # Quick statistics (avoid full text re-parse on list views)
    original_word_count = Column(Integer, default=0)
    output_word_count   = Column(Integer, default=0)
    quality_score       = Column(Float, default=0.0)
    processing_time_ms  = Column(Integer, default=0)

    # Full quality report stored as JSON blob
    quality_json    = Column(JSON, nullable=True)

    # Timestamps
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())

    def quality_report_dict(self) -> dict:
        return self.quality_json or {}


class UploadRecord(Base):
    """
    Tracks uploaded files so they can be cleaned up and audited.
    Actual file content is stored only on disk; this table holds metadata.
    """
    __tablename__ = "upload_records"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    filename    = Column(String(255), nullable=False)
    file_type   = Column(String(16), nullable=False)   # txt | docx | pdf
    file_size   = Column(Integer, default=0)
    disk_path   = Column(String(512), nullable=False)
    word_count  = Column(Integer, default=0)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    deleted     = Column(Boolean, default=False)


class UndoStack(Base):
    """
    Supports undo/redo by keeping a lightweight ordered stack of text snapshots
    per session.  Snapshots are pruned to the last 10 per session automatically.
    """
    __tablename__ = "undo_stack"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(String(36), nullable=False, index=True)
    step        = Column(Integer, nullable=False)   # 0 = oldest
    text        = Column(Text, nullable=False)
    description = Column(String(128), nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())


# ── Database Lifecycle ────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables if they don't exist yet."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async session, commits or rolls back."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
