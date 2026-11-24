"""
backend/providers/other_providers.py
─────────────────────────────────────
Cohere and Ollama provider adapters.

Cohere – uses the official cohere Python SDK (v5+).
Ollama – uses its local REST API (no extra SDK needed, plain httpx).
"""
from __future__ import annotations

import json

import httpx
from loguru import logger

from backend.providers.base import (
    BaseLLMProvider,
    LLMConfigError,
    ProviderRegistry,
)
from config.settings import settings


# ════════════════════════════════════════════════════════════════════════════════
# Cohere
# ════════════════════════════════════════════════════════════════════════════════

class CohereProvider(BaseLLMProvider):

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def model(self) -> str:
        return settings.cohere_model

    @property
    def is_configured(self) -> bool:
        return bool(settings.cohere_api_key)

    @property
    def description(self) -> str:
        return f"Cohere – {settings.cohere_model}"

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            import cohere
        except ImportError as exc:
            raise LLMConfigError(
                "cohere package not installed – run: pip install cohere"
            ) from exc

        if not self.is_configured:
            raise LLMConfigError("COHERE_API_KEY is not set in environment / .env")

        client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)

        response = await client.chat(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        content = ""
        for block in response.message.content:
            if hasattr(block, "text"):
                content += block.text

        logger.debug("[Cohere] finish_reason={}", response.finish_reason)
        return content


# ════════════════════════════════════════════════════════════════════════════════
# Ollama (local)
# ════════════════════════════════════════════════════════════════════════════════

class OllamaProvider(BaseLLMProvider):
    """
    Uses Ollama's /api/chat endpoint.
    Ollama must be running locally: https://ollama.com
    Pull a model first: `ollama pull llama3.2`
    """

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return settings.ollama_model

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_configured(self) -> bool:
        # Ollama is always "configured" – it's local.
        # We consider it configured when the model name is set.
        return bool(settings.ollama_model)

    @property
    def description(self) -> str:
        return f"Ollama (local) – {settings.ollama_model}"

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {settings.ollama_base_url}. "
                f"Is the Ollama server running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            raise RuntimeError(
                f"Ollama returned HTTP {exc.response.status_code}: {body}"
            ) from exc

        content = data.get("message", {}).get("content", "")
        logger.debug("[Ollama] model={} done_reason={}", self.model, data.get("done_reason"))
        return content


# ── Auto-register ─────────────────────────────────────────────────────────────
ProviderRegistry.register(CohereProvider())
ProviderRegistry.register(OllamaProvider())
