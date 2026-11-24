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

import asyncio
import re
import time

from loguru import logger

from backend.providers.base import (
    BaseLLMProvider,
    LLMConfigError,
    LLMQuotaError,
    ProviderRegistry,
    _is_daily_quota_error,
    _status_code,
)
from config.settings import settings


class OpenAIProvider(BaseLLMProvider):

    _exhausted_until: dict[str, float] = {}

    def __init__(self) -> None:
        self._rate_lock: asyncio.Lock | None = None
        self._rate_loop: asyncio.AbstractEventLoop | None = None
        self._next_call_at = 0.0

    async def _throttle(self) -> None:
        interval = max(0.0, float(settings.llm_min_interval))
        if interval <= 0:
            return
        loop = asyncio.get_running_loop()
        if self._rate_lock is None or self._rate_loop is not loop:
            self._rate_lock = asyncio.Lock()
            self._rate_loop = loop
            self._next_call_at = 0.0
        async with self._rate_lock:
            start = max(time.monotonic(), self._next_call_at)
            self._next_call_at = start + interval
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _quota_cooldown(exc: BaseException) -> float:
        match = re.search(
            r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s",
            str(exc),
            re.IGNORECASE,
        )
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = float(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds + 60
        return 3600.0

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
        for candidate in settings.openai_fallback_model.split(","):
            candidate = candidate.strip()
            if candidate and candidate not in models:
                models.append(candidate)

        now = time.monotonic()
        models = [
            model for model in models
            if self._exhausted_until.get(model, 0.0) <= now
        ]
        if not models:
            raise LLMQuotaError(
                "All configured Gemini models have exhausted their daily "
                "free-tier quota. Add billing or raise the quota, then retry."
            )

        last_exc: Exception | None = None
        for index, model in enumerate(models):
            try:
                await self._throttle()
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
                daily_quota = _is_daily_quota_error(exc)
                if daily_quota:
                    cooldown = self._quota_cooldown(exc)
                    self._exhausted_until[model] = time.monotonic() + cooldown
                    logger.warning(
                        "[OpenAI] model {} daily quota exhausted; "
                        "skipping it for {:.0f}s",
                        model, cooldown,
                    )
                retryable = (
                    code is None
                    or code in (404, 408, 409, 425, 429)
                    or code >= 500
                )
                if index < len(models) - 1 and (retryable or daily_quota):
                    logger.warning(
                        "[OpenAI] model {} failed ({}); trying fallback chain: {}",
                        model,
                        code or type(exc).__name__,
                        ", ".join(models[index + 1:]),
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
