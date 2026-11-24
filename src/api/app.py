"""FastAPI application managing singleton store, AsyncSqliteSaver lifespan, and SSE endpoints."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.agent.graph import build_pursuit_graph, open_sqlite_checkpointer
from src.agent.llm import create_llm
from src.config import Settings, get_settings
from src.logging_config import configure_logging
from src.mcp_server.store import ReadOnlyPursuitStore
from src.observability import (
    build_langgraph_run_config,
    flush_langfuse,
    record_local_trace,
)

logger = logging.getLogger(__name__)

UI_HTML_PATH = Path(__file__).resolve().parent.parent / "ui" / "index.html"
RFPS_DIR = Path("data/rfps")


class AnalyzeRequest(BaseModel):
    rfp_text: str = Field(min_length=10)
    workspace_id: str = Field(default="commercial_aec")


class ResumeRequest(BaseModel):
    decisions: dict[str, str] = Field(default_factory=dict)
    workspace_id: str = Field(default="commercial_aec")


class FollowupRequest(BaseModel):
    followup_input: str = Field(min_length=3)
    workspace_id: str = Field(default="commercial_aec")


def _sse_event(event_name: str, payload: Any) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"


def _dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _serialize_state_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": values.get("workspace_id", "commercial_aec"),
        "raw_rfp_cleared": values.get("raw_rfp_text", "") == "",
        "pinned_requirements": [_dump(r) for r in values.get("pinned_requirements", [])],
        "what_if_assumptions": values.get("what_if_assumptions", []),
        "evidence_by_id": {k: _dump(v) for k, v in values.get("evidence_by_id", {}).items()},
        "discrepancies": [_dump(d) for d in values.get("discrepancies", [])],
        "unverifiable_criteria": values.get("unverifiable_criteria", []),
        "recent_turns": [_dump(t) for t in values.get("recent_turns", [])],
        "conversation_summary": values.get("conversation_summary", ""),
        "budget": _dump(values.get("budget") or {}),
        "hitl_decisions": values.get("hitl_decisions", {}),
        "scorecard": _dump(values.get("scorecard")),
        "audit_log": [_dump(a) for a in values.get("audit_log", [])],
    }


def create_app(settings_override: Settings | None = None, rfps_dir: Path = RFPS_DIR) -> FastAPI:
    """Create FastAPI app with clean lifespan management for SQLite store & checkpointer."""
    settings = settings_override or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        backend = (
            "mock"
            if settings.llm_provider == "mock"
            else ("vertex_ai" if settings.use_vertexai else "gemini_api")
        )
        project_label = settings.google_cloud_project.strip() or "<express/projectless>"
        logger.info(
            "PursuitGuard startup: backend=%s llm_provider=%s auth_mode=%s primary_model=%s light_model=%s "
            "location=%s project=%s crm_db=%s checkpoint_db=%s stale_crm_days=%d "
            "max_tool_calls=%d max_token_budget=%d",
            backend,
            settings.llm_provider,
            settings.vertex_auth_mode,
            settings.gemini_primary_model,
            settings.gemini_light_model,
            settings.google_cloud_location,
            project_label,
            settings.crm_db_path,
            settings.checkpoint_db_path,
            settings.stale_crm_days,
            settings.max_tool_calls,
            settings.max_token_budget,
        )
        if settings.llm_provider == "mock" and settings.resolved_google_api_key:
            logger.warning(
                "GEMINI_API_KEY/GOOGLE_API_KEY is set, but LLM_PROVIDER=mock — running in offline "
                "mock mode (no Gemini calls will be made). Set LLM_PROVIDER=gemini to enable live calls."
            )

        settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        store = ReadOnlyPursuitStore(settings.crm_db_path)
        llm = create_llm(settings)
        async with open_sqlite_checkpointer(str(settings.checkpoint_db_path)) as saver:
            graph = build_pursuit_graph(store=store, llm=llm, settings=settings, checkpointer=saver)
            app_instance.state.settings = settings
            app_instance.state.store = store
            app_instance.state.graph = graph
            try:
                yield
            finally:
                logger.info("PursuitGuard shutdown: closing SQLite store, LLM client, and checkpointer")
                if hasattr(llm, "close"):
                    llm.close()
                store.close()

    application = FastAPI(title="PursuitGuard — AEC Go/No-Go Pursuit Qualification Agent", lifespan=lifespan)

    async def _stream_graph(
        graph: Any,
        graph_input: Any,
        *,
        thread_id: str,
        workspace_id: str,
    ) -> AsyncIterator[str]:
        cfg = build_langgraph_run_config(
            thread_id=thread_id,
            workspace_id=workspace_id,
            settings=settings,
        )
        start = time.perf_counter()
        try:
            async for chunk in graph.astream(graph_input, config=cfg, stream_mode="updates"):
                for node_name, node_data in chunk.items():
                    if node_name == "__interrupt__":
                        intr_list = [getattr(item, "value", item) for item in node_data]
                        logger.debug(
                            "Graph __interrupt__ thread=%s count=%d",
                            thread_id,
                            len(intr_list),
                        )
                        yield _sse_event("interrupt", {"interrupts": intr_list})
                    else:
                        msg = (
                            node_data.get("status_message", "")
                            if isinstance(node_data, dict)
                            else ""
                        )
                        logger.debug(
                            "Graph node_update thread=%s node=%s status=%s",
                            thread_id,
                            node_name,
                            msg,
                        )
                        yield _sse_event(
                            "node_update",
                            {"node": node_name, "status_message": msg},
                        )

            snap = await graph.aget_state(cfg)
            serialized = _serialize_state_snapshot(snap.values)
            pending_interrupts = [
                getattr(intr, "value", intr)
                for task in snap.tasks
                for intr in task.interrupts
            ]

            elapsed_ms = (time.perf_counter() - start) * 1000
            if pending_interrupts:
                logger.warning(
                    "Graph paused for HITL thread=%s workspace=%s interrupts=%d elapsed_ms=%.1f",
                    thread_id,
                    workspace_id,
                    len(pending_interrupts),
                    elapsed_ms,
                )
                yield _sse_event(
                    "paused",
                    {"interrupts": pending_interrupts, "state": serialized},
                )
            else:
                scorecard = serialized.get("scorecard")
                budget = serialized.get("budget", {})
                verdict = scorecard.get("verdict") if scorecard else None
                logger.info(
                    "Graph complete thread=%s workspace=%s verdict=%s tool_calls=%s llm_calls=%s "
                    "input_tokens=%s output_tokens=%s pruned_tokens=%s elapsed_ms=%.1f",
                    thread_id,
                    workspace_id,
                    verdict,
                    budget.get("tool_calls", 0),
                    budget.get("llm_calls", 0),
                    budget.get("input_tokens", 0),
                    budget.get("output_tokens", 0),
                    budget.get("pruned_tokens", 0),
                    elapsed_ms,
                )
                record_local_trace(
                    trace_path=settings.trace_log_path,
                    thread_id=thread_id,
                    workspace_id=workspace_id,
                    verdict=verdict,
                    budget=budget,
                    audit_events=serialized.get("audit_log", []),
                )
                flush_langfuse(settings)
                yield _sse_event("complete", {"state": serialized})
        except Exception as exc:
            logger.exception(
                "Graph stream failed thread=%s workspace=%s",
                thread_id,
                workspace_id,
            )
            yield _sse_event("error", {"message": str(exc)})

    @application.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return UI_HTML_PATH.read_text(encoding="utf-8")

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        backend = (
            "mock"
            if settings.llm_provider == "mock"
            else ("vertex_ai" if settings.use_vertexai else "gemini_api")
        )
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "backend": backend,
            "model": settings.gemini_primary_model if settings.llm_provider != "mock" else "mock",
            "auth_mode": settings.vertex_auth_mode,
            "stale_crm_days": settings.stale_crm_days,
        }

    @application.get("/api/rfps")
    async def list_rfps() -> list[dict[str, str]]:
        if not rfps_dir.exists():
            return []
        items: list[dict[str, str]] = []
        for path in sorted(rfps_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            first_line = content.splitlines()[0].lstrip("# ").strip() if content else path.stem
            items.append({"id": path.stem, "title": first_line, "content": content})
        return items

    @application.post("/api/threads/{thread_id}/analyze")
    async def analyze_rfp(thread_id: str, body: AnalyzeRequest) -> StreamingResponse:
        graph = application.state.graph
        logger.info(
            "POST /analyze thread=%s workspace=%s rfp_chars=%d",
            thread_id,
            body.workspace_id,
            len(body.rfp_text),
        )
        initial_state = {
            "thread_id": thread_id,
            "workspace_id": body.workspace_id,
            "raw_rfp_text": body.rfp_text,
            "followup_input": "",
        }
        return StreamingResponse(
            _stream_graph(
                graph,
                initial_state,
                thread_id=thread_id,
                workspace_id=body.workspace_id,
            ),
            media_type="text/event-stream",
        )

    @application.post("/api/threads/{thread_id}/resume")
    async def resume_thread(thread_id: str, body: ResumeRequest) -> StreamingResponse:
        graph = application.state.graph
        logger.info(
            "POST /resume thread=%s workspace=%s decision_keys=%s",
            thread_id,
            body.workspace_id,
            sorted(body.decisions.keys()),
        )
        return StreamingResponse(
            _stream_graph(
                graph,
                Command(resume=body.decisions),
                thread_id=thread_id,
                workspace_id=body.workspace_id,
            ),
            media_type="text/event-stream",
        )

    @application.post("/api/threads/{thread_id}/followup")
    async def followup_thread(thread_id: str, body: FollowupRequest) -> StreamingResponse:
        graph = application.state.graph
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = await graph.aget_state(cfg)
        if not snap.values or not snap.values.get("pinned_requirements"):
            logger.warning("POST /followup thread=%s not ready (404)", thread_id)
            raise HTTPException(
                status_code=404,
                detail=f"Thread '{thread_id}' has no analyzed RFP state yet.",
            )
        ws = snap.values.get("workspace_id", body.workspace_id)
        logger.info(
            "POST /followup thread=%s workspace=%s followup_preview=%r",
            thread_id,
            ws,
            body.followup_input[:80],
        )
        return StreamingResponse(
            _stream_graph(
                graph,
                {"followup_input": body.followup_input},
                thread_id=thread_id,
                workspace_id=ws,
            ),
            media_type="text/event-stream",
        )

    @application.get("/api/threads/{thread_id}/audit")
    async def get_audit(thread_id: str) -> dict[str, Any]:
        graph = application.state.graph
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = await graph.aget_state(cfg)
        if not snap.values:
            logger.warning("GET /audit thread=%s not found (404)", thread_id)
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
        serialized = _serialize_state_snapshot(snap.values)
        logger.info(
            "GET /audit thread=%s workspace=%s raw_rfp_cleared=%s audit_events=%d",
            thread_id,
            serialized["workspace_id"],
            serialized["raw_rfp_cleared"],
            len(serialized["audit_log"]),
        )
        return serialized

    return application


app = create_app()
