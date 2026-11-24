"""Typed LangGraph state for sequential PursuitGuard execution."""

from __future__ import annotations

from typing import TypedDict

from src.agent.models import (
    AuditEvent,
    BudgetState,
    FreshnessDiscrepancy,
    RequirementEvidence,
    RequirementItem,
    Scorecard,
    TurnRecord,
)


class PursuitState(TypedDict, total=False):
    """Clean, race-free sequential state persisted across AsyncSqliteSaver checkpoints."""

    thread_id: str
    workspace_id: str
    raw_rfp_text: str  # Cleared to "" right after extraction
    followup_input: str
    pinned_requirements: list[RequirementItem]
    target_req_ids: list[str]
    what_if_assumptions: list[str]
    evidence_by_id: dict[str, RequirementEvidence]
    discrepancies: list[FreshnessDiscrepancy]
    unverifiable_criteria: list[str]
    recent_turns: list[TurnRecord]  # Explicit 2-turn window
    conversation_summary: str
    budget: BudgetState
    hitl_decisions: dict[str, str]
    scorecard: Scorecard | None
    audit_log: list[AuditEvent]
    retry_queue: list[str]
    status_message: str
