"""
backend/providers/openai_provider.py
─────────────────────────────────────
OpenAI Chat Completions provider adapter.

Supports:
  - Any gpt-4o / gpt-4o-mini / gpt-3.5-turbo model.
  - Custom base_url for Azure OpenAI or compatible endpoints.
  - Async calls via the official openai Python SDK v1.x.
"""
from __future__ import annotations

from loguru import logger

from backend.providers.base import (
    BaseLLMProvider,
    LLMConfigError,
    ProviderRegistry,
    _status_code,
)
from config.settings import settings


class OpenAIProvider(BaseLLMProvider):

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return settings.openai_model

    @property
    def is_configured(self) -> bool:
        return bool(settings.openai_api_key)

    @property
    def description(self) -> str:
        return f"OpenAI – {settings.openai_model}"

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMConfigError(
                "openai package not installed – run: pip install openai"
            ) from exc

        if not self.is_configured:
            raise LLMConfigError("OPENAI_API_KEY is not set in environment / .env")

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout,
            max_retries=0,   # retries handled by BaseLLMProvider
        )

        models = [self.model]
        fallback = settings.openai_fallback_model.strip()
        if fallback and fallback != self.model:
            models.append(fallback)

        last_exc: Exception | None = None
        for model in models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
            except Exception as exc:
                last_exc = exc
                code = _status_code(exc)
                retryable = code is None or code in (408, 409, 425, 429) or code >= 500
                if model != models[-1] and retryable:
                    logger.warning(
                        "[OpenAI] model {} failed ({}); retrying with fallback {}",
                        model, code or type(exc).__name__, models[-1],
                    )
                    continue
                raise

            content = response.choices[0].message.content or ""
            logger.debug(
                "[OpenAI] model={} tokens used: prompt={} completion={}",
                model,
                response.usage.prompt_tokens if response.usage else "?",
                response.usage.completion_tokens if response.usage else "?",
            )
            return content

        raise last_exc  # pragma: no cover


# ── Auto-register ─────────────────────────────────────────────────────────────
ProviderRegistry.register(OpenAIProvider())
