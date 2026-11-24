"""Sequential LangGraph StateGraph assembly for PursuitGuard."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.llm import PursuitLLM
from src.agent.models import (
    AuditEvent,
    BudgetState,
    FreshnessDiscrepancy,
    RequirementEvidence,
    RequirementItem,
    Scorecard,
    TurnRecord,
    WhatIfPatch,
)
from src.agent.nodes import make_nodes
from src.agent.state import PursuitState
from src.config import Settings
from src.mcp_server.store import ReadOnlyPursuitStore

logger = logging.getLogger(__name__)


def make_checkpoint_serde() -> JsonPlusSerializer:
    """Create allowlisted msgpack serializer for PursuitGuard Pydantic state models."""
    allowed = [
        ("src.agent.models", cls.__name__)
        for cls in (
            RequirementItem,
            RequirementEvidence,
            FreshnessDiscrepancy,
            BudgetState,
            TurnRecord,
            AuditEvent,
            WhatIfPatch,
            Scorecard,
        )
    ]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


@asynccontextmanager
async def open_sqlite_checkpointer(conn_string: str) -> AsyncIterator[AsyncSqliteSaver]:
    """Open AsyncSqliteSaver cleanly with allowlisted serializer and guaranteed close."""
    logger.info("Opening SQLite checkpointer db=%s", conn_string)
    async with aiosqlite.connect(conn_string) as conn:
        saver = AsyncSqliteSaver(conn, serde=make_checkpoint_serde())
        await saver.setup()
        try:
            yield saver
        finally:
            logger.debug("Closing SQLite checkpointer db=%s", conn_string)


def _route_after_entry(state: PursuitState) -> str:
    if state.get("followup_input") and state.get("pinned_requirements"):
        target = "apply_what_if_delta"
    else:
        target = "extract_requirements"
    logger.debug("Graph route after entry_router -> %s", target)
    return target


def _route_after_verify(state: PursuitState) -> str:
    target = "verify_requirements" if state.get("retry_queue") else "reconcile_freshness"
    logger.debug("Graph route after verify_requirements -> %s", target)
    return target


def _route_after_prune(state: PursuitState) -> str:
    hitl = state.get("hitl_decisions", {})
    discrepancies = state.get("discrepancies", [])
    unverifiable = state.get("unverifiable_criteria", [])
    pinned_map = {r.req_id: r for r in state.get("pinned_requirements", [])}

    needs_discrepancy = any(d.discrepancy_id not in hitl for d in discrepancies)
    needs_unverifiable = any(
        rid not in hitl and not (rid in pinned_map and pinned_map[rid].waived_by_what_if)
        for rid in unverifiable
    )
    target = "hitl_review_gate" if (needs_discrepancy or needs_unverifiable) else "emit_scorecard"
    logger.debug("Graph route after prune_context -> %s", target)
    return target


def build_pursuit_graph(
    *,
    store: ReadOnlyPursuitStore,
    llm: PursuitLLM,
    settings: Settings,
    checkpointer: Any = None,
):
    """Build and compile the 7-step sequential PursuitGuard StateGraph."""
    nodes = make_nodes(store=store, llm=llm, settings=settings)
    builder = StateGraph(PursuitState)

    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "entry_router")
    builder.add_conditional_edges(
        "entry_router",
        _route_after_entry,
        {
            "extract_requirements": "extract_requirements",
            "apply_what_if_delta": "apply_what_if_delta",
        },
    )
    builder.add_edge("extract_requirements", "verify_requirements")
    builder.add_edge("apply_what_if_delta", "verify_requirements")
    builder.add_conditional_edges(
        "verify_requirements",
        _route_after_verify,
        {
            "verify_requirements": "verify_requirements",
            "reconcile_freshness": "reconcile_freshness",
        },
    )
    builder.add_edge("reconcile_freshness", "prune_context")
    builder.add_conditional_edges(
        "prune_context",
        _route_after_prune,
        {
            "hitl_review_gate": "hitl_review_gate",
            "emit_scorecard": "emit_scorecard",
        },
    )
    builder.add_edge("hitl_review_gate", "emit_scorecard")
    builder.add_edge("emit_scorecard", END)

    compiled = builder.compile(checkpointer=checkpointer)
    logger.debug("Compiled PursuitGuard StateGraph checkpointer_enabled=%s", checkpointer is not None)
    return compiled
