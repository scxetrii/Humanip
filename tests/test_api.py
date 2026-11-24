"""
tests/test_api.py
─────────────────
Integration tests for all FastAPI endpoints.
Uses httpx.AsyncClient with the ASGI app directly — no real HTTP server needed.

Run with:  pytest tests/test_api.py -v --asyncio-mode=auto
"""
from __future__ import annotations

import os
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Minimal env so the app doesn't crash on import ───────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_PROVIDER", "openai")

from backend.main import app


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """Yield an async ASGI test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_required_fields(self, client):
        resp = await client.get("/health")
        data = resp.json()
        for field in ("status", "version", "db_connected", "nlp_loaded"):
            assert field in data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_root_redirects_or_200(self, client):
        resp = await client.get("/")
        assert resp.status_code in (200, 307, 308)


# ── Providers ─────────────────────────────────────────────────────────────────

class TestProvidersEndpoint:
    @pytest.mark.asyncio
    async def test_list_providers_200(self, client):
        resp = await client.get("/api/v1/providers")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_providers_returns_list(self, client):
        resp = await client.get("/api/v1/providers")
        data = resp.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) >= 1

    @pytest.mark.asyncio
    async def test_provider_has_required_fields(self, client):
        resp = await client.get("/api/v1/providers")
        providers = resp.json()["providers"]
        for p in providers:
            for field in ("name", "is_configured", "is_local", "model"):
                assert field in p, f"Provider missing field '{field}'"

    @pytest.mark.asyncio
    async def test_active_provider_200(self, client):
        resp = await client.get("/api/v1/providers/active")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_switch_to_unknown_provider_404(self, client):
        resp = await client.post(
            "/api/v1/providers/switch",
            json={"provider": "nonexistent_provider_xyz"},
        )
        assert resp.status_code in (404, 400)

    @pytest.mark.asyncio
    async def test_ping_unconfigured_provider(self, client):
        resp = await client.get("/api/v1/providers/openai/ping")
        assert resp.status_code == 200
        data = resp.json()
        # Key must exist regardless of configured state
        assert "reachable" in data


# ── Analyze endpoint ──────────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    SAMPLE = "In conclusion, it is important to note that AI has changed many things."

    @pytest.mark.asyncio
    async def test_analyze_returns_200(self, client):
        resp = await client.post("/api/v1/analyze", json={"text": self.SAMPLE})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_analyze_returns_analysis_fields(self, client):
        resp = await client.post("/api/v1/analyze", json={"text": self.SAMPLE})
        data = resp.json()
        for field in ("word_count", "sentence_count", "citations", "readability_score"):
            assert field in data, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_analyze_too_short_text_422(self, client):
        resp = await client.post("/api/v1/analyze", json={"text": "Hi"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_empty_422(self, client):
        resp = await client.post("/api/v1/analyze", json={"text": ""})
        assert resp.status_code == 422


# ── History ───────────────────────────────────────────────────────────────────

class TestHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_history_list_200(self, client):
        resp = await client.get("/api/v1/history")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_history_has_pagination_fields(self, client):
        resp = await client.get("/api/v1/history")
        data = resp.json()
        for field in ("items", "total", "page", "page_size"):
            assert field in data

    @pytest.mark.asyncio
    async def test_history_items_is_list(self, client):
        resp = await client.get("/api/v1/history")
        assert isinstance(resp.json()["items"], list)

    @pytest.mark.asyncio
    async def test_history_nonexistent_id_404(self, client):
        resp = await client.get("/api/v1/history/999999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id_404(self, client):
        resp = await client.delete("/api/v1/history/999999")
        assert resp.status_code == 404


# ── Upload ────────────────────────────────────────────────────────────────────

class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_txt_200(self, client):
        content = b"This is a sample text file for testing the upload endpoint."
        resp = await client.post(
            "/api/v1/upload",
            files={"file": ("sample.txt", content, "text/plain")},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_txt_returns_text(self, client):
        content = b"Hello world. This is test content."
        resp = await client.post(
            "/api/v1/upload",
            files={"file": ("sample.txt", content, "text/plain")},
        )
        data = resp.json()
        assert "extracted_text" in data
        assert len(data["extracted_text"]) > 0

    @pytest.mark.asyncio
    async def test_upload_unsupported_format_422(self, client):
        resp = await client.post(
            "/api/v1/upload",
            files={"file": ("image.png", b"\x89PNG\r\n", "image/png")},
        )
        assert resp.status_code in (422, 415)

    @pytest.mark.asyncio
    async def test_download_txt_200(self, client):
        resp = await client.post(
            "/api/v1/download/txt",
            json={"text": "Hello world.", "filename": "output"},
        )
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_download_txt_no_text_422(self, client):
        resp = await client.post("/api/v1/download/txt", json={"text": ""})
        assert resp.status_code == 422


# ── Validation ────────────────────────────────────────────────────────────────

class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_humanize_missing_text_422(self, client):
        resp = await client.post("/api/v1/humanize", json={"mode": "natural"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_humanize_text_too_short_422(self, client):
        resp = await client.post(
            "/api/v1/humanize",
            json={"text": "Short", "mode": "natural"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_humanize_invalid_mode_422(self, client):
        resp = await client.post(
            "/api/v1/humanize",
            json={"text": "A" * 100, "mode": "invalid_mode_xyz"},
        )
        assert resp.status_code == 422
