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

from backend.providers.base import BaseLLMProvider, LLMConfigError, ProviderRegistry
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

        response = await client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        content = response.choices[0].message.content or ""
        logger.debug(
            "[OpenAI] tokens used: prompt={} completion={}",
            response.usage.prompt_tokens if response.usage else "?",
            response.usage.completion_tokens if response.usage else "?",
        )
        return content


# ── Auto-register ─────────────────────────────────────────────────────────────
ProviderRegistry.register(OpenAIProvider())
