"""
backend/providers/anthropic_provider.py
────────────────────────────────────────
Anthropic Claude provider adapter.

Uses the official anthropic Python SDK (v0.40+) with async support.
Supports claude-3-5-haiku, claude-3-5-sonnet, claude-3-opus.
"""
from __future__ import annotations

from loguru import logger

from backend.providers.base import BaseLLMProvider, ProviderRegistry
from config.settings import settings


class AnthropicProvider(BaseLLMProvider):

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return settings.anthropic_model

    @property
    def is_configured(self) -> bool:
        return bool(settings.anthropic_api_key)

    @property
    def description(self) -> str:
        return f"Anthropic – {settings.anthropic_model}"

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package not installed – run: pip install anthropic") from exc

        if not self.is_configured:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment / .env")

        client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout,
            max_retries=0,
        )

        message = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from the first content block
        content = ""
        for block in message.content:
            if hasattr(block, "text"):
                content += block.text

        logger.debug(
            "[Anthropic] stop_reason={} input_tokens={} output_tokens={}",
            message.stop_reason,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        return content


ProviderRegistry.register(AnthropicProvider())
