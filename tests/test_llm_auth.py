"""Offline unit tests for Gemini Developer API and Vertex AI auth modes."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from src.agent.llm import GeminiPursuitLLM
from src.api.app import create_app
from src.config import Settings


def test_gemini_developer_api_key_ignores_ambient_vertex_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "stale-env-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    settings = Settings(
        llm_provider="gemini",
        google_genai_use_vertexai=False,
        gemini_api_key="TEST_GEMINI_KEY",
        gemini_primary_model="gemini-2.5-flash",
    )
    assert settings.use_vertexai is False
    assert settings.vertex_auth_mode == "api_key"

    adapter = GeminiPursuitLLM(settings)
    try:
        api_client = adapter.client._api_client
        assert api_client.vertexai is False
        assert api_client.api_key == "TEST_GEMINI_KEY"
        assert api_client._http_options.base_url == "https://generativelanguage.googleapis.com/"
        assert api_client._http_options.retry_options is not None
        assert api_client._http_options.retry_options.attempts == 4
        assert adapter._primary_model == "gemini-2.5-flash"
    finally:
        adapter.close()

    assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "stale-env-proj"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"


def test_express_mode_api_key_ignores_ambient_location_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "stale-env-proj")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings(
        llm_provider="vertex_gemini",
        google_api_key="TEST_EXPRESS_KEY",
        google_cloud_project="",
        google_cloud_location="global",
    )
    assert settings.use_vertexai is True
    assert settings.vertex_auth_mode == "api_key"

    adapter = GeminiPursuitLLM(settings)
    try:
        api_client = adapter.client._api_client
        assert api_client.vertexai is True
        assert api_client.api_key == "TEST_EXPRESS_KEY"
        assert api_client.project is None
        assert api_client.location is None
        assert api_client._http_options.base_url == "https://aiplatform.googleapis.com/"
        assert api_client._http_options.headers["x-goog-api-key"] == "TEST_EXPRESS_KEY"
    finally:
        adapter.close()

    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "us-central1"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "stale-env-proj"


def test_standard_gcp_api_key_with_explicit_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings(
        llm_provider="vertex_gemini",
        google_api_key="TEST_STD_KEY",
        google_cloud_project="my-gcp-proj",
        google_cloud_location="global",
    )
    adapter = GeminiPursuitLLM(settings)
    try:
        api_client = adapter.client._api_client
        assert api_client.vertexai is True
        assert api_client.api_key == "TEST_STD_KEY"
        assert api_client.project == "my-gcp-proj"
        assert api_client.location == "global"
        assert api_client._http_options.base_url == "https://aiplatform.googleapis.com/"
        assert api_client._http_options.headers["x-goog-api-key"] == "TEST_STD_KEY"
    finally:
        adapter.close()


def test_genai_client_retries_transient_ssl_connect_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings(
        llm_provider="gemini",
        google_genai_use_vertexai=False,
        gemini_api_key="TEST_RETRY_KEY",
    )
    adapter = GeminiPursuitLLM(settings)
    try:
        retryer = adapter.client._api_client._retry
        retryer.wait = lambda _state: 0  # avoid sleeping in unit tests
        attempts = {"count": 0}

        def flaky_call() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
            return "ok"

        assert retryer(flaky_call) == "ok"
        assert attempts["count"] == 3
    finally:
        adapter.close()


async def test_health_endpoint_reports_auth_mode_without_leaking_key(tmp_path: Path) -> None:
    secret_key = "SUPER_SECRET_GEMINI_KEY"
    test_settings = Settings(
        llm_provider="gemini",
        google_genai_use_vertexai=False,
        gemini_api_key=secret_key,
        gemini_primary_model="gemini-2.5-flash",
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "health_ckpt.db",
    )
    app = create_app(settings_override=test_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["llm_provider"] == "gemini"
            assert payload["backend"] == "gemini_api"
            assert payload["model"] == "gemini-2.5-flash"
            assert payload["auth_mode"] == "api_key"
            assert secret_key not in resp.text
