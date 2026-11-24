"""
config/settings.py
─────────────────
Centralised application settings loaded from environment variables
or a .env file.  Every module imports from here – never from os.environ
directly – so the whole app respects a single source of truth.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All application settings.  Values come from (in priority order):
    1. Real environment variables
    2. A .env file at the project root
    3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "Humanip"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 1

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_port: int = 8501

    # ── Database ──────────────────────────────────────────────────────────────
    db_path: Path = ROOT_DIR / "data" / "humanizer.db"
    db_echo: bool = False  # set True to log every SQL statement

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # Which provider to use: "openai" | "anthropic" | "cohere" | "ollama"
    llm_provider: Literal["openai", "anthropic", "cohere", "ollama"] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-20241022"

    # Cohere
    cohere_api_key: str = ""
    cohere_model: str = "command-r-plus"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── LLM Parameters ────────────────────────────────────────────────────────
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: int = 60            # seconds per attempt
    llm_max_retries: int = 2         # attempts before giving up
    llm_total_timeout: int = 90      # hard cap for the whole LLM call
    humanize_timeout: int = 110      # overall LLM budget for one humanization run

    # ── NLP ───────────────────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"   # or en_core_web_md / lg
    enable_local_transformers: bool = False
    local_transformer_model: str = "facebook/bart-large-cnn"

    # ── File Uploads ─────────────────────────────────────────────────────────
    upload_max_size_mb: int = 10
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".txt", ".docx", ".pdf"]
    )
    upload_dir: Path = ROOT_DIR / "data" / "uploads"

    # ── History ───────────────────────────────────────────────────────────────
    history_max_items: int = 500       # rows kept per user session
    history_retention_days: int = 30

    # ── Quality Engine ────────────────────────────────────────────────────────
    similarity_threshold: float = 0.75  # minimum acceptable semantic similarity
    semantic_model: str = "all-MiniLM-L6-v2"
    enable_grammar_check: bool = False  # LanguageTool download/startup is slow
    quality_timeout: int = 20           # seconds before falling back to heuristics

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-32-chars-min",
        min_length=16,
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Path = ROOT_DIR / "data" / "app.log"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_size_mb * 1024 * 1024

    @field_validator("db_path", "upload_dir", "log_file", mode="before")
    @classmethod
    def _ensure_parent_exists(cls, v: Path | str) -> Path:
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Convenience alias used throughout the codebase
settings = get_settings()
