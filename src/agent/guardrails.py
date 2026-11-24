"""Deterministic budget circuit breakers, freshness thresholds, and verdict invariants."""

from __future__ import annotations

import logging

from src.agent.models import (
    BudgetState,
    FreshnessDiscrepancy,
    RequirementEvidence,
    RequirementItem,
    Scorecard,
    Verdict,
)

logger = logging.getLogger(__name__)


def can_continue_budget(
    budget: BudgetState,
    *,
    max_tool_calls: int = 12,
    max_token_budget: int = 25000,
    next_calls: int = 1,
) -> tuple[bool, str | None]:
    """Check whether sequential execution remains within hard call and token caps."""
    if budget.tool_calls + next_calls > max_tool_calls:
        reason = f"Halted: tool call budget reached ({budget.tool_calls}/{max_tool_calls})."
        logger.warning(reason)
        return False, reason
    total_tokens = budget.input_tokens + budget.output_tokens
    if total_tokens >= max_token_budget:
        reason = f"Halted: token budget reached ({total_tokens}/{max_token_budget})."
        logger.warning(reason)
        return False, reason
    return True, None


def _apply_closeout_override(
    disc: FreshnessDiscrepancy,
    requirements: list[RequirementItem],
    updated_evidence: dict[str, RequirementEvidence],
) -> None:
    """Adjust CRM combined fee totals when a human picks newer closeout sheet figures."""
    for req in requirements:
        if req.kind != "crm_projects" or req.min_total_fee_usd is None:
            continue
        ev = updated_evidence.get(req.req_id)
        if ev is None:
            continue

        for cand in disc.candidate_matches:
            crm_fee = float(cand.get("crm_fee_usd") or 0.0)
            sheet_fee = float(cand.get("closeout_fee_usd") or 0.0)
            if sheet_fee <= crm_fee:
                continue

            adjusted_total = ev.total_fee_usd + (sheet_fee - crm_fee)
            ev.total_fee_usd = adjusted_total
            logger.info(
                "HITL closeout override applied: disc=%s req_id=%s adjusted_total=%.0f threshold=%.0f",
                disc.discrepancy_id,
                req.req_id,
                adjusted_total,
                req.min_total_fee_usd,
            )
            if adjusted_total >= req.min_total_fee_usd and (
                req.min_count is None or ev.matched_count >= req.min_count
            ):
                ev.status = "PASSED"
                ev.summary = (
                    f"Closeout override applied: combined fee ${adjusted_total:,.0f} "
                    f"meets ${req.min_total_fee_usd:,.0f} threshold."
                )
            break


def apply_hitl_resolutions(
    requirements: list[RequirementItem],
    evidence_by_id: dict[str, RequirementEvidence],
    discrepancies: list[FreshnessDiscrepancy],
    hitl_decisions: dict[str, str],
) -> tuple[dict[str, RequirementEvidence], list[str]]:
    """Apply deterministic human resolutions (closeout fee overrides, waivers, or rejections)."""
    updated_evidence = {k: v.model_copy(deep=True) for k, v in evidence_by_id.items()}
    notes: list[str] = []

    for req in requirements:
        ev = updated_evidence.get(req.req_id)
        decision = hitl_decisions.get(req.req_id)
        if ev is None or ev.status != "UNVERIFIABLE_NEEDS_HUMAN" or not decision:
            continue

        norm = decision.strip().lower()
        if norm in {"waive", "approve", "verified_offline"}:
            ev.status = "WAIVED_BY_HUMAN"
            ev.summary = f"{ev.summary} [Human resolution: {decision}]"
            notes.append(f"Requirement {req.req_id} verified/waived by human ({decision}).")
            logger.info("HITL resolution: req_id=%s waived decision=%s", req.req_id, decision)
        elif norm in {"reject", "fail"}:
            ev.status = "FAILED"
            ev.summary = f"{ev.summary} [Human marked failed]"
            notes.append(f"Requirement {req.req_id} marked FAILED by human.")
            logger.info("HITL resolution: req_id=%s marked FAILED decision=%s", req.req_id, decision)

    for disc in discrepancies:
        choice = hitl_decisions.get(disc.discrepancy_id)
        if not choice:
            continue
        notes.append(f"{disc.discrepancy_id} ({disc.kind}): human selected '{choice}'.")
        if disc.kind == "METRIC_CLASH" and choice.strip().lower() == "use_closeout":
            _apply_closeout_override(disc, requirements, updated_evidence)

    return updated_evidence, notes


def enforce_scorecard_invariants(
    proposed_summary: str,
    requirements: list[RequirementItem],
    evidence_by_id: dict[str, RequirementEvidence],
    discrepancies: list[FreshnessDiscrepancy],
    hitl_decisions: dict[str, str],
    what_if_assumptions: list[str],
    budget: BudgetState,
) -> Scorecard:
    """Enforce deterministic Go/No-Go rules so an LLM can never force GO over disqualifiers."""
    resolved_ev, resolution_notes = apply_hitl_resolutions(
        requirements=requirements,
        evidence_by_id=evidence_by_id,
        discrepancies=discrepancies,
        hitl_decisions=hitl_decisions,
    )

    passed = 0
    failed = 0
    unverifiable = 0
    waived = 0
    mandatory_failed = False
    mandatory_unverifiable = False
    has_waiver_or_whatif = bool(what_if_assumptions)
    all_citations: list[str] = []

    for req in requirements:
        ev = resolved_ev.get(req.req_id)
        if ev is None:
            unverifiable += 1
            if req.is_mandatory:
                mandatory_unverifiable = True
            continue

        all_citations.extend(ev.citations)
        if req.waived_by_what_if or ev.status == "WAIVED_BY_HUMAN":
            waived += 1
            has_waiver_or_whatif = True
        elif ev.status == "PASSED":
            passed += 1
        elif ev.status == "FAILED":
            failed += 1
            if req.is_mandatory:
                mandatory_failed = True
        else:
            unverifiable += 1
            if req.is_mandatory:
                mandatory_unverifiable = True

    unresolved_discrepancies = [
        d for d in discrepancies if d.discrepancy_id not in hitl_decisions
    ]
    guardrail_notes = list(resolution_notes)

    if budget.budget_halted:
        guardrail_notes.append("Execution reached hard budget cap before exhausting optional retries.")

    if mandatory_failed:
        verdict: Verdict = "NO_GO"
        confidence = "HIGH"
        guardrail_notes.append(
            "Mandatory disqualifier FAILED — deterministic guard prohibits GO."
        )
        logger.warning(
            "Guardrail enforced verdict=NO_GO (mandatory disqualifier failed; passed=%d failed=%d)",
            passed,
            failed,
        )
    elif mandatory_unverifiable or unresolved_discrepancies:
        verdict = "HITL_REQUIRED"
        confidence = "LOW"
        if mandatory_unverifiable:
            guardrail_notes.append(
                "Mandatory requirement remains UNVERIFIABLE without human waiver."
            )
        if unresolved_discrepancies:
            guardrail_notes.append(
                f"{len(unresolved_discrepancies)} freshness/entity discrepancy awaiting sign-off."
            )
        logger.warning(
            "Guardrail enforced verdict=HITL_REQUIRED (mandatory_unverifiable=%s unresolved_discrepancies=%d)",
            mandatory_unverifiable,
            len(unresolved_discrepancies),
        )
    elif has_waiver_or_whatif or failed > 0:
        verdict = "CONDITIONAL_GO"
        confidence = "MEDIUM"
        guardrail_notes.append(
            "Conditional verdict: relies on explicit human waiver or What-If assumption."
        )
    else:
        verdict = "GO"
        confidence = "HIGH"

    logger.info(
        "Scorecard invariants finalized verdict=%s confidence=%s passed=%d failed=%d unverifiable=%d waived=%d",
        verdict,
        confidence,
        passed,
        failed,
        unverifiable,
        waived,
    )
    dedup_citations = list(dict.fromkeys(all_citations))
    return Scorecard(
        verdict=verdict,
        confidence=confidence,
        summary=proposed_summary,
        requirements_passed=passed,
        requirements_failed=failed,
        requirements_unverifiable=unverifiable,
        requirements_waived=waived,
        discrepancy_count=len(discrepancies),
        guardrail_notes=guardrail_notes,
        citations=dedup_citations,
    )
