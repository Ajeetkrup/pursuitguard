"""FastAPI SSE endpoint and lifespan tests (ensuring clean Windows SQLite teardown)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from src.api.app import create_app
from src.config import Settings


def _parse_sse_events(body_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body_text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = "message"
        data_raw = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_raw += line.split(":", 1)[1].strip()
        if data_raw:
            events.append((ev_name, json.loads(data_raw)))
    return events


async def test_ui_serves_light_mode_html(tmp_path: Path) -> None:
    test_settings = Settings(
        llm_provider="mock",
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "ui_ckpt.db",
        trace_log_path=tmp_path / "ui_traces.jsonl",
    )
    app = create_app(settings_override=test_settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")
            assert "color-scheme: light" in resp.text
            assert "PursuitGuard" in resp.text
            assert "Plus Jakarta Sans" in resp.text
            assert "renderMarkdown(vSum," in resp.text
            assert "function parseRfpOverview(" in resp.text
            assert 'id="audit-dialog"' in resp.text
            assert 'closedby="any"' in resp.text
            assert 'id="copy-brief-btn"' in resp.text
            assert 'command="--copy"' in resp.text
            assert "function humanizeGuardrailNote(" in resp.text
            assert "function sortRequirementsForDisplay(" in resp.text
            assert "function resetQualificationView(" in resp.text
            assert "#0f1419" not in resp.text
            assert "prefers-color-scheme: dark" not in resp.text


async def test_fastapi_sse_analyze_resume_and_followup(tmp_path: Path) -> None:
    test_settings = Settings(
        llm_provider="mock",
        crm_db_path=Path("data/crm.db"),
        checkpoint_db_path=tmp_path / "api_ckpt.db",
        trace_log_path=tmp_path / "traces.jsonl",
    )
    app = create_app(settings_override=test_settings)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["llm_provider"] == "mock"

            rfps = await client.get("/api/rfps")
            assert rfps.status_code == 200
            assert len(rfps.json()) == 3

            rfp_text = Path("data/rfps/tx_water_conveyance.md").read_text(encoding="utf-8")
            r1 = await client.post(
                "/api/threads/api-t1/analyze",
                json={"rfp_text": rfp_text, "workspace_id": "commercial_aec"},
            )
            assert r1.status_code == 200
            evs1 = _parse_sse_events(r1.text)
            assert evs1[-1][0] == "paused"

            r2 = await client.post(
                "/api/threads/api-t1/resume",
                json={"decisions": {"clash_nar_10_PRJ-22-119": "use_closeout"}},
            )
            assert r2.status_code == 200
            evs2 = _parse_sse_events(r2.text)
            assert evs2[-1][0] == "complete"
            assert evs2[-1][1]["state"]["scorecard"]["verdict"] == "GO"

            r3 = await client.post(
                "/api/threads/api-t1/followup",
                json={"followup_input": "Raise minimum project size threshold to 35,000 sq ft"},
            )
            assert r3.status_code == 200
            evs3 = _parse_sse_events(r3.text)
            assert evs3[-1][0] == "complete"
            assert evs3[-1][1]["state"]["scorecard"]["verdict"] == "NO_GO"

            audit = await client.get("/api/threads/api-t1/audit")
            assert audit.status_code == 200
            assert audit.json()["raw_rfp_cleared"] is True
            assert (tmp_path / "traces.jsonl").exists()
