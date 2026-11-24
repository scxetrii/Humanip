"""
backend/providers/base.py
─────────────────────────
Abstract base class and registry for all LLM provider adapters.

Each concrete provider (OpenAI, Anthropic, Cohere, Ollama) subclasses
BaseLLMProvider and implements _call_api().  The ProviderRegistry keeps a
catalogue of available providers so the application can switch between them
at runtime without changing any other code.
"""
from __future__ import annotations

import abc
import time
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings


# ── Data carriers ─────────────────────────────────────────────────────────────

class LLMResponse:
    """Thin wrapper returned by every provider."""

    __slots__ = ("text", "model", "provider", "latency_ms", "raw")

    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        latency_ms: int,
        raw: Any = None,
    ) -> None:
        self.text       = text.strip()
        self.model      = model
        self.provider   = provider
        self.latency_ms = latency_ms
        self.raw        = raw   # original SDK response object, for debugging


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMProvider(abc.ABC):
    """
    All providers must implement:
      - name         (str property)
      - is_local     (bool property – True for Ollama etc.)
      - is_configured(bool property – keys present / model reachable)
      - _call_api()  (actual HTTP / SDK call, returns raw text)

    The public .complete() method handles retries, timing, and logging.
    """

    # ── Abstract interface ───────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def model(self) -> str:
        ...

    @property
    def is_local(self) -> bool:
        return False

    @property
    @abc.abstractmethod
    def is_configured(self) -> bool:
        ...

    @property
    def description(self) -> str:
        return f"{self.name} ({self.model})"

    @abc.abstractmethod
    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make the actual API call and return raw assistant text."""
        ...

    # ── Public interface ──────────────────────────────────────────────────────

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Call the provider with retry logic and timing.
        Raises LLMError if all retries fail.
        """
        t0 = time.perf_counter()
        temp    = temperature if temperature is not None else settings.llm_temperature
        tokens  = max_tokens  if max_tokens  is not None else settings.llm_max_tokens

        try:
            text = await self._call_api_with_retry(system_prompt, user_prompt, temp, tokens)
        except Exception as exc:
            raise LLMError(f"[{self.name}] All retries failed: {exc}") from exc

        latency = int((time.perf_counter() - t0) * 1000)
        logger.info("[{}] Response in {}ms ({} chars)", self.name, latency, len(text))
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            latency_ms=latency,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_api_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await self._call_api(system_prompt, user_prompt, temperature, max_tokens)


# ── Registry ──────────────────────────────────────────────────────────────────

class ProviderRegistry:
    """
    Maps provider names to instances.
    Providers register themselves at import time via ProviderRegistry.register().
    """

    _registry: dict[str, BaseLLMProvider] = {}

    @classmethod
    def register(cls, provider: BaseLLMProvider) -> None:
        cls._registry[provider.name] = provider
        logger.debug("Provider registered: {}", provider.name)

    @classmethod
    def get(cls, name: str) -> BaseLLMProvider:
        if name not in cls._registry:
            raise LLMError(
                f"Provider '{name}' is not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def active(cls) -> BaseLLMProvider:
        return cls.get(settings.llm_provider)

    @classmethod
    def all(cls) -> list[BaseLLMProvider]:
        return list(cls._registry.values())

    @classmethod
    def all_names(cls) -> list[str]:
        return list(cls._registry.keys())
