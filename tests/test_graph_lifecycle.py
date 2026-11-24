"""Lifecycle tests for sequential LangGraph state, HITL interrupt/resume, budget caps, and invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from src.agent.graph import build_pursuit_graph, open_sqlite_checkpointer
from src.agent.llm import MockPursuitLLM
from src.config import Settings
from src.mcp_server import ReadOnlyPursuitStore


@pytest.fixture()
def store() -> ReadOnlyPursuitStore:
    s = ReadOnlyPursuitStore(Path("data/crm.db"))
    try:
        yield s
    finally:
        s.close()


async def test_interrupt_resume_and_whatif_persistence_across_restart(
    tmp_path: Path, store: ReadOnlyPursuitStore
) -> None:
    ckpt_path = str(tmp_path / "checkpoints.db")
    rfp_text = Path("data/rfps/tx_water_conveyance.md").read_text(encoding="utf-8")
    cfg = {"configurable": {"thread_id": "tx-water-1"}}

    # First graph instance runs until HITL interrupt on Barton Creek closeout clash
    async with open_sqlite_checkpointer(ckpt_path) as saver1:
        graph1 = build_pursuit_graph(store=store, llm=MockPursuitLLM(), settings=Settings(), checkpointer=saver1)
        paused = await graph1.ainvoke(
            {"raw_rfp_text": rfp_text, "workspace_id": "commercial_aec", "thread_id": "tx-water-1"},
            config=cfg,
        )
        assert paused["raw_rfp_text"] == ""
        assert paused["budget"].pruned_tokens > 0
        assert "__interrupt__" in paused
        disc_ids = [d["discrepancy_id"] for d in paused["__interrupt__"][0].value["discrepancies"]]
        assert "clash_nar_10_PRJ-22-119" in disc_ids

    # Brand new graph instance resumes cleanly from disk SQLite checkpoint
    async with open_sqlite_checkpointer(ckpt_path) as saver2:
        graph2 = build_pursuit_graph(store=store, llm=MockPursuitLLM(), settings=Settings(), checkpointer=saver2)
        resumed = await graph2.ainvoke(
            Command(resume={"clash_nar_10_PRJ-22-119": "use_closeout"}),
            config=cfg,
        )
        # Choosing newer closeout ($16.8M vs $14.2M) lifts combined water fee to $26.9M >= $25M -> GO
        assert resumed["scorecard"].verdict == "GO"

        # What-If follow-up skips extraction and checks only affected requirement
        calls_before = resumed["budget"].tool_calls
        fup = await graph2.ainvoke({"followup_input": "Raise minimum project size threshold to 35,000 sq ft"}, config=cfg)
        assert fup["budget"].tool_calls - calls_before == 1
        assert fup["scorecard"].verdict == "NO_GO"


async def test_mandatory_expired_se_cannot_emit_go_and_waiver_makes_conditional(
    tmp_path: Path, store: ReadOnlyPursuitStore
) -> None:
    ckpt_path = str(tmp_path / "ckpt_se.db")
    rfp_text = Path("data/rfps/federal_courthouse_seismic.md").read_text(encoding="utf-8")
    cfg = {"configurable": {"thread_id": "gsa-se-1"}}

    async with open_sqlite_checkpointer(ckpt_path) as saver:
        graph = build_pursuit_graph(store=store, llm=MockPursuitLLM(), settings=Settings(), checkpointer=saver)
        await graph.ainvoke(
            {"raw_rfp_text": rfp_text, "workspace_id": "commercial_aec", "thread_id": "gsa-se-1"},
            config=cfg,
        )
        resumed = await graph.ainvoke(
            Command(resume={"clash_nar_14_PRJ-23-041": "use_closeout"}),
            config=cfg,
        )
        assert resumed["scorecard"].verdict == "NO_GO"

        # What-If JV partner waiver transitions NO_GO -> CONDITIONAL_GO (never unconditional GO)
        waived = await graph.ainvoke(
            {"followup_input": "Assume a JV partner provides an active CA SE license"},
            config=cfg,
        )
        assert waived["scorecard"].verdict == "CONDITIONAL_GO"


async def test_hard_budget_cap_halts_execution(tmp_path: Path, store: ReadOnlyPursuitStore) -> None:
    ckpt_path = str(tmp_path / "ckpt_cap.db")
    rfp_text = Path("data/rfps/va_hospital_renovation.md").read_text(encoding="utf-8")
    tight_settings = Settings(max_tool_calls=1)
    cfg = {"configurable": {"thread_id": "cap-test"}}

    async with open_sqlite_checkpointer(ckpt_path) as saver:
        graph = build_pursuit_graph(store=store, llm=MockPursuitLLM(), settings=tight_settings, checkpointer=saver)
        out = await graph.ainvoke(
            {"raw_rfp_text": rfp_text, "workspace_id": "commercial_aec", "thread_id": "cap-test"},
            config=cfg,
        )
        assert out["budget"].budget_halted is True
        assert out["budget"].tool_calls == 1
