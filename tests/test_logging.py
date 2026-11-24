"""Tests for secret-safe CLI logging across API, graph, LLM, guardrails, and MCP store."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from src.api.app import create_app
from src.config import Settings
from src.logging_config import configure_logging


@pytest.fixture
def capture_pursuitguard_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Attach pytest's caplog handler directly to `src` when propagate=False."""
    pkg_logger = logging.getLogger("src")
    caplog.set_level(logging.INFO, logger="src")
    pkg_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        pkg_logger.removeHandler(caplog.handler)


def test_configure_logging_is_idempotent() -> None:
    try:
        pkg_logger = configure_logging("INFO")
        configure_logging("DEBUG")
        configure_logging("INFO")
        configure_logging("DEBUG")

        tagged = [h for h in pkg_logger.handlers if h.name == "pursuitguard_console"]
        assert len(tagged) == 1
        assert pkg_logger.propagate is False
        assert pkg_logger.level == logging.DEBUG
        assert tagged[0].level == logging.DEBUG
    finally:
        configure_logging("INFO")


async def test_end_to_end_run_emits_structured_logs(
    tmp_path: Path,
    capture_pursuitguard_logs: pytest.LogCaptureFixture,
) -> None:
    test_settings = Settings(
        llm_provider="mock",
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "log_ckpt.db",
        trace_log_path=tmp_path / "log_traces.jsonl",
    )
    app = create_app(settings_override=test_settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rfp_text = Path("data/rfps/tx_water_conveyance.md").read_text(encoding="utf-8")
            r1 = await client.post(
                "/api/threads/log-t1/analyze",
                json={"rfp_text": rfp_text, "workspace_id": "commercial_aec"},
            )
            assert r1.status_code == 200

            r2 = await client.post(
                "/api/threads/log-t1/resume",
                json={"decisions": {"clash_nar_10_PRJ-22-119": "use_closeout"}},
            )
            assert r2.status_code == 200

    logger_names = {rec.name for rec in capture_pursuitguard_logs.records}
    expected_loggers = {
        "src.api.app",
        "src.agent.llm",
        "src.agent.nodes",
        "src.agent.freshness",
        "src.agent.guardrails",
        "src.mcp_server.server",
        "src.mcp_server.store",
    }
    assert expected_loggers.issubset(logger_names)


async def test_logs_warn_when_api_key_set_in_mock_mode_and_never_leak_secrets(
    tmp_path: Path,
    capture_pursuitguard_logs: pytest.LogCaptureFixture,
) -> None:
    secret_key = "SUPER_SECRET_KEY_NEVER_LOG_12345"

    mock_settings = Settings(
        llm_provider="mock",
        google_api_key=secret_key,
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "mock_ckpt.db",
        trace_log_path=tmp_path / "mock_traces.jsonl",
    )
    mock_app = create_app(settings_override=mock_settings)
    async with mock_app.router.lifespan_context(mock_app):
        pass

    assert any(
        rec.levelno == logging.WARNING
        and "GEMINI_API_KEY/GOOGLE_API_KEY is set, but LLM_PROVIDER=mock" in rec.message
        for rec in capture_pursuitguard_logs.records
    )

    capture_pursuitguard_logs.clear()
    gemini_settings = Settings(
        llm_provider="gemini",
        google_genai_use_vertexai=False,
        gemini_api_key=secret_key,
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "gemini_ckpt.db",
        trace_log_path=tmp_path / "gemini_traces.jsonl",
    )
    gemini_app = create_app(settings_override=gemini_settings)
    async with gemini_app.router.lifespan_context(gemini_app):
        pass

    assert "gemini_developer_api" in capture_pursuitguard_logs.text
    assert "auth_mode=api_key" in capture_pursuitguard_logs.text
    assert secret_key not in capture_pursuitguard_logs.text

    capture_pursuitguard_logs.clear()
    vertex_settings = Settings(
        llm_provider="vertex_gemini",
        google_api_key=secret_key,
        google_cloud_project="",
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "vertex_ckpt.db",
        trace_log_path=tmp_path / "vertex_traces.jsonl",
    )
    vertex_app = create_app(settings_override=vertex_settings)
    async with vertex_app.router.lifespan_context(vertex_app):
        pass

    assert "auth_mode=api_key" in capture_pursuitguard_logs.text
    assert "express_projectless" in capture_pursuitguard_logs.text
    assert secret_key not in capture_pursuitguard_logs.text
