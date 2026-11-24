"""
backend/config.py
─────────────────
Central application configuration loaded from environment variables or .env file.
All sensitive values (API keys) are read from env — never hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.  Values are populated from:
    1. Environment variables (highest priority)
    2. .env file in the project root
    3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "AI Text Humanizer"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = Field(default="change-me-in-production", min_length=16)

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/humanizer.db"

    # ── LLM Providers ─────────────────────────────────────────────────────────
    default_llm_provider: Literal[
        "openai", "anthropic", "google", "ollama", "custom"
    ] = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # Google
    google_api_key: str = ""
    google_model: str = "gemini-1.5-pro"

    # Ollama (local, no key needed)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Custom OpenAI-compatible
    custom_llm_base_url: str = "http://localhost:1234/v1"
    custom_llm_api_key: str = "lm-studio"
    custom_llm_model: str = "local-model"

    # ── File Upload ───────────────────────────────────────────────────────────
    max_upload_size_mb: int = 10
    allowed_extensions: str = "txt,docx,pdf"
    upload_dir: str = "./data/uploads"

    # ── NLP ───────────────────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"

    # ── Features ──────────────────────────────────────────────────────────────
    enable_local_models: bool = False
    enable_grammar_check: bool = True
    enable_history: bool = True
    log_level: str = "INFO"

    # ── Computed helpers ──────────────────────────────────────────────────────
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_ext_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        """Return the filesystem path for SQLite."""
        raw = self.database_url.replace("sqlite+aiosqlite:///", "")
        p = Path(raw)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
