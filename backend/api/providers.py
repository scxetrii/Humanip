"""
backend/api/providers.py
─────────────────────────
Provider management endpoints.

GET  /api/v1/providers          → list all registered providers + their status
POST /api/v1/providers/switch   → change the active provider at runtime
GET  /api/v1/providers/active   → return the currently active provider info
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.schemas import ProviderInfo, ProviderListResponse
from backend.providers.base import LLMError, ProviderRegistry
from config.settings import settings

router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class SwitchProviderRequest(BaseModel):
    provider: str
    model: str | None = None   # optional override of the model for this provider


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/providers",
    response_model=ProviderListResponse,
    summary="List all available LLM providers",
)
async def list_providers() -> ProviderListResponse:
    """
    Returns every registered provider with its configuration status.
    Use this to present a provider selector in the UI.
    """
    providers = [
        ProviderInfo(
            name=p.name,
            is_configured=p.is_configured,
            is_local=p.is_local,
            model=p.model,
            description=p.description,
        )
        for p in ProviderRegistry.all()
    ]
    return ProviderListResponse(
        providers=providers,
        active_provider=settings.llm_provider,
    )


@router.get(
    "/providers/active",
    response_model=ProviderInfo,
    summary="Get active provider details",
)
async def get_active_provider() -> ProviderInfo:
    """Return information about the currently active LLM provider."""
    p = ProviderRegistry.active()
    return ProviderInfo(
        name=p.name,
        is_configured=p.is_configured,
        is_local=p.is_local,
        model=p.model,
        description=p.description,
    )


@router.post(
    "/providers/switch",
    summary="Switch the active LLM provider",
)
async def switch_provider(request: SwitchProviderRequest) -> dict:
    """
    Change which provider is used for future requests.
    The switch takes effect immediately for all subsequent /humanize calls.
    """
    try:
        provider = ProviderRegistry.get(request.provider)
    except LLMError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    if not provider.is_configured:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{request.provider}' is not configured. "
                f"Set the required API key in your .env file."
            ),
        )

    # Mutate the settings singleton so all subsequent ProviderRegistry.active() calls
    # return the new provider.  In a multi-worker setup, use Redis or a shared store.
    settings.llm_provider = request.provider  # type: ignore[misc]

    return {
        "switched_to": request.provider,
        "model": provider.model,
        "message": f"Active provider is now '{request.provider}' ({provider.model}).",
    }


@router.get(
    "/providers/{provider_name}/ping",
    summary="Test a provider's connectivity",
)
async def ping_provider(provider_name: str) -> dict:
    """
    Send a minimal test prompt to verify a provider is reachable and responding.
    Returns response latency in milliseconds.
    """
    try:
        provider = ProviderRegistry.get(provider_name)
    except LLMError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if not provider.is_configured:
        return {
            "provider": provider_name,
            "reachable": False,
            "reason": "Not configured (missing API key).",
        }

    try:
        result = await provider.complete(
            system_prompt="You are a test assistant.",
            user_prompt="Reply with the single word: OK",
            max_tokens=10,
        )
        return {
            "provider": provider_name,
            "reachable": True,
            "latency_ms": result.latency_ms,
            "response_preview": result.text[:20],
        }
    except Exception as exc:
        return {
            "provider": provider_name,
            "reachable": False,
            "reason": str(exc),
        }
