"""Pydantic data models for PursuitGuard state, requirements, freshness, and scorecards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReqKind = Literal["crm_projects", "staff_credential", "narrative_scope", "client_history", "unmapped"]
EvalStatus = Literal["PASSED", "FAILED", "UNVERIFIABLE_NEEDS_HUMAN", "WAIVED_BY_HUMAN"]
Verdict = Literal["GO", "CONDITIONAL_GO", "NO_GO", "HITL_REQUIRED"]


class RequirementItem(BaseModel):
    """Single qualification requirement extracted from an RFP."""

    req_id: str
    title: str
    description: str
    kind: ReqKind
    is_mandatory: bool = True
    # Optional deterministic filter hints extracted from RFP text
    subsector: str | None = None
    state: str | None = None
    role: str | None = None
    min_count: int | None = None
    min_sqft: int | None = None
    min_total_fee_usd: float | None = None
    min_year: int | None = None
    leed_min: str | None = None
    credential_type: str | None = None
    client_name: str | None = None
    narrative_query: str | None = None
    waived_by_what_if: bool = False


class RequirementEvidence(BaseModel):
    """Verification outcome and provenance for a single requirement."""

    req_id: str
    status: EvalStatus
    tool_used: str
    summary: str
    matched_count: int = 0
    total_fee_usd: float = 0.0
    oldest_sync_at: str | None = None
    max_sync_age_days: int = 0
    citations: list[str] = Field(default_factory=list)
    retried_with_synonym: bool = False


class FreshnessDiscrepancy(BaseModel):
    """Detected mismatch, stale sync, or ambiguous tie between CRM and closeout sheets."""

    discrepancy_id: str
    kind: Literal["METRIC_CLASH", "AMBIGUOUS_ENTITY_MATCH", "STALE_CRM_SYNC"]
    closeout_doc_id: str | None = None
    closeout_title: str | None = None
    closeout_updated_at: str | None = None
    crm_project_id: str | None = None
    crm_title: str | None = None
    crm_last_synced_at: str | None = None
    match_score: float = 0.0
    candidate_matches: list[dict[str, Any]] = Field(default_factory=list)
    details: str


class BudgetState(BaseModel):
    """Sequential runtime cost and loop guard counters."""

    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    pruned_tokens: int = 0
    retries: int = 0
    budget_halted: bool = False


class TurnRecord(BaseModel):
    """Compact turn representation kept in explicit state window."""

    turn_index: int
    user_input: str
    verdict: Verdict | None = None
    summary: str


class AuditEvent(BaseModel):
    """Structured execution trail entry for audit panel and Langfuse inspection."""

    node: str
    action: str
    detail: str
    tokens_used: int = 0


class WhatIfPatch(BaseModel):
    """Structured delta produced for a follow-up What-If sensitivity turn."""

    summary: str
    target_req_ids: list[str] = Field(default_factory=list)
    updated_requirements: list[RequirementItem] = Field(default_factory=list)
    assumption_note: str | None = None


class Scorecard(BaseModel):
    """Final Go / Conditional Go / No-Go recommendation with deterministic guardrails."""

    verdict: Verdict
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    summary: str
    requirements_passed: int
    requirements_failed: int
    requirements_unverifiable: int
    requirements_waived: int
    discrepancy_count: int
    guardrail_notes: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
