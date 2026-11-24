"""Langfuse v3+ observability adapter with graceful local JSONL trace fallback."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)


def build_langgraph_run_config(
    *,
    thread_id: str,
    workspace_id: str,
    settings: Settings,
) -> dict[str, Any]:
    """Build LangGraph RunnableConfig with Langfuse v3+ CallbackHandler when configured."""
    metadata: dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "langfuse_user_id": workspace_id,
        "langfuse_tags": ["pursuitguard", workspace_id],
    }
    callbacks: list[Any] = []

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.langchain import CallbackHandler

            callbacks.append(CallbackHandler())
        except (ImportError, RuntimeError, ValueError) as exc:
            logger.warning("Langfuse CallbackHandler unavailable; using local traces: %s", exc)
            callbacks = []

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "metadata": metadata,
    }
    if callbacks:
        config["callbacks"] = callbacks
    logger.debug(
        "Built LangGraph run config thread=%s workspace=%s langfuse=%s",
        thread_id,
        workspace_id,
        bool(callbacks),
    )
    return config


def record_local_trace(
    *,
    trace_path: Path,
    thread_id: str,
    workspace_id: str,
    verdict: str | None,
    budget: dict[str, Any],
    audit_events: list[dict[str, Any]],
) -> None:
    """Append structured execution trace locally so runs are auditable offline."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "thread_id": thread_id,
        "workspace_id": workspace_id,
        "verdict": verdict,
        "budget": budget,
        "events": audit_events,
    }
    with trace_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info(
        "Recorded local trace path=%s thread=%s verdict=%s",
        trace_path,
        thread_id,
        verdict,
    )


def flush_langfuse(settings: Settings) -> None:
    """Flush pending Langfuse spans if credentials are configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return
    try:
        from langfuse import get_client

        get_client().flush()
        logger.debug("Flushed Langfuse spans")
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning("Langfuse flush failed: %s", exc)
