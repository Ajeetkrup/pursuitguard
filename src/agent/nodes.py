"""LangGraph node implementations for sequential PursuitGuard execution."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from mcp import Client

from src.agent.freshness import reconcile_closeouts_against_crm
from src.agent.guardrails import can_continue_budget, enforce_scorecard_invariants
from src.agent.llm import PursuitLLM
from src.agent.models import (
    AuditEvent,
    BudgetState,
    RequirementEvidence,
    RequirementItem,
    TurnRecord,
)
from src.agent.state import PursuitState
from src.config import Settings
from src.mcp_server.server import build_mcp_server, parse_mcp_result
from src.mcp_server.store import ReadOnlyPursuitStore

logger = logging.getLogger(__name__)


def _evaluate_tool_outcome(req: RequirementItem, tool_name: str, data: Any, retried: bool) -> RequirementEvidence:
    """Deterministically evaluate MCP tool results against RequirementItem thresholds."""
    if tool_name == "query_projects" and isinstance(data, dict):
        count = int(data.get("count", 0))
        total_fee = float(data.get("total_fee_usd", 0.0))
        rows = data.get("rows", [])
        citations = [f"{r['project_id']} ({r['title']})" for r in rows]
        count_ok = count >= (req.min_count or 1)
        fee_ok = req.min_total_fee_usd is None or total_fee >= req.min_total_fee_usd
        passed = count_ok and fee_ok
        summary = (
            f"Matched {count} CRM projects totaling ${total_fee:,.0f} "
            f"(oldest sync: {data.get('oldest_sync_at')})."
        )
        return RequirementEvidence(
            req_id=req.req_id,
            status="PASSED" if passed else "FAILED",
            tool_used=tool_name,
            summary=summary,
            matched_count=count,
            total_fee_usd=total_fee,
            oldest_sync_at=data.get("oldest_sync_at"),
            max_sync_age_days=int(data.get("max_sync_age_days", 0)),
            citations=citations,
            retried_with_synonym=retried,
        )

    if tool_name == "check_staff_credentials" and isinstance(data, dict):
        active_count = int(data.get("active_count", 0))
        expired_count = int(data.get("expired_count", 0))
        rows = data.get("rows", [])
        citations = [f"{r['license_number']} ({r['staff_name']}, exp {r['expires_on']})" for r in rows]
        passed = active_count >= 1
        summary = f"Found {active_count} active and {expired_count} expired {req.credential_type or ''} licenses in {req.state or 'any'}."
        return RequirementEvidence(
            req_id=req.req_id,
            status="PASSED" if passed else "FAILED",
            tool_used=tool_name,
            summary=summary,
            matched_count=active_count,
            oldest_sync_at=data.get("oldest_sync_at"),
            max_sync_age_days=int(data.get("max_sync_age_days", 0)),
            citations=citations,
            retried_with_synonym=retried,
        )

    if tool_name == "get_client_history" and isinstance(data, dict):
        count = int(data.get("count", 0))
        win_rate = float(data.get("win_rate", 0.0))
        pursuits = data.get("pursuits", [])
        citations = [f"{p['pursuit_id']}: {p['opportunity_name']} ({p['outcome']}, {p['debrief_score']})" for p in pursuits]
        return RequirementEvidence(
            req_id=req.req_id,
            status="PASSED" if count > 0 else "UNVERIFIABLE_NEEDS_HUMAN",
            tool_used=tool_name,
            summary=f"{count} historical pursuits ({int(win_rate * 100)}% win rate, avg score {data.get('avg_debrief_score')}).",
            matched_count=count,
            oldest_sync_at=data.get("oldest_sync_at"),
            citations=citations,
            retried_with_synonym=retried,
        )

    if tool_name == "search_project_narratives" and isinstance(data, list):
        won_docs = [d for d in data if str(d.get("status", "")).lower() == "won"]
        citations = [f"{d['doc_id']}: {d['title']} ({d['updated_at']})" for d in won_docs]
        passed = len(won_docs) >= 1
        return RequirementEvidence(
            req_id=req.req_id,
            status="PASSED" if passed else "FAILED",
            tool_used=tool_name,
            summary=f"Found {len(won_docs)} won narrative match(es) via FTS5 BM25.",
            matched_count=len(won_docs),
            citations=citations,
            retried_with_synonym=retried,
        )

    return RequirementEvidence(
        req_id=req.req_id,
        status="UNVERIFIABLE_NEEDS_HUMAN",
        tool_used=tool_name,
        summary="Tool returned unrecognized structure.",
        retried_with_synonym=retried,
    )


def make_nodes(store: ReadOnlyPursuitStore, llm: PursuitLLM, settings: Settings) -> dict[str, Any]:
    """Build node callables bound to singleton store, LLM, and settings."""

    def entry_router(state: PursuitState) -> dict[str, Any]:
        is_followup = bool(state.get("followup_input") and state.get("pinned_requirements"))
        mode = "what_if_delta" if is_followup else "initial_extraction"
        logger.info(
            "Node entry_router thread=%s workspace=%s mode=%s",
            state.get("thread_id", "<unknown>"),
            state.get("workspace_id", settings.default_workspace_id),
            mode,
        )
        return {
            "status_message": f"Routed turn to {mode}.",
            "audit_log": [
                *state.get("audit_log", []),
                AuditEvent(node="entry_router", action=mode, detail=state.get("followup_input", "New RFP")),
            ],
        }

    def extract_requirements(state: PursuitState, config: RunnableConfig | None = None) -> dict[str, Any]:
        raw_rfp = state.get("raw_rfp_text", "")
        budget = state.get("budget", BudgetState()).model_copy(deep=True)
        items, in_tok, out_tok = llm.extract_requirements(raw_rfp, config=config)
        budget.llm_calls += 1
        budget.input_tokens += in_tok
        budget.output_tokens += out_tok
        pruned_tokens = len(raw_rfp) // 4
        budget.pruned_tokens += pruned_tokens
        logger.info(
            "Node extract_requirements extracted=%d cleared_chars=%d tokens=%d",
            len(items),
            len(raw_rfp),
            in_tok + out_tok,
        )
        audit = [
            *state.get("audit_log", []),
            AuditEvent(
                node="extract_requirements",
                action="shred_and_clear_raw_rfp",
                detail=f"Extracted {len(items)} pinned criteria and cleared {len(raw_rfp)} chars of raw RFP text.",
                tokens_used=in_tok + out_tok,
            ),
        ]
        return {
            "raw_rfp_text": "",
            "pinned_requirements": items,
            "target_req_ids": [r.req_id for r in items],
            "what_if_assumptions": state.get("what_if_assumptions", []),
            "evidence_by_id": state.get("evidence_by_id", {}),
            "discrepancies": state.get("discrepancies", []),
            "unverifiable_criteria": [],
            "hitl_decisions": state.get("hitl_decisions", {}),
            "budget": budget,
            "audit_log": audit,
            "retry_queue": [],
            "status_message": f"Extracted {len(items)} requirements; raw RFP cleared.",
        }

    def apply_what_if_delta(state: PursuitState, config: RunnableConfig | None = None) -> dict[str, Any]:
        followup = state.get("followup_input", "")
        pinned = [r.model_copy(deep=True) for r in state.get("pinned_requirements", [])]
        budget = state.get("budget", BudgetState()).model_copy(deep=True)

        patch, in_tok, out_tok = llm.parse_what_if(followup, pinned, config=config)
        budget.llm_calls += 1
        budget.input_tokens += in_tok
        budget.output_tokens += out_tok

        updated_map = {r.req_id: r for r in patch.updated_requirements}
        new_pinned = [updated_map.get(r.req_id, r) for r in pinned]
        assumptions = [*state.get("what_if_assumptions", [])]
        if patch.assumption_note:
            assumptions.append(patch.assumption_note)

        # Clear stale HITL decision only for requirements explicitly modified by What-If
        hitl = dict(state.get("hitl_decisions", {}))
        for rid in patch.target_req_ids:
            hitl.pop(rid, None)

        logger.info(
            "Node apply_what_if_delta target_req_ids=%s total_assumptions=%d",
            patch.target_req_ids,
            len(assumptions),
        )
        audit = [
            *state.get("audit_log", []),
            AuditEvent(
                node="apply_what_if_delta",
                action="patch_criteria",
                detail=f"Updated {patch.target_req_ids}: {patch.summary}",
                tokens_used=in_tok + out_tok,
            ),
        ]
        return {
            "pinned_requirements": new_pinned,
            "target_req_ids": patch.target_req_ids,
            "what_if_assumptions": assumptions,
            "hitl_decisions": hitl,
            "budget": budget,
            "audit_log": audit,
            "retry_queue": [],
            "status_message": f"Patched {len(patch.target_req_ids)} requirement(s) for What-If turn.",
        }

    async def verify_requirements(state: PursuitState, config: RunnableConfig | None = None) -> dict[str, Any]:
        workspace_id = state.get("workspace_id", settings.default_workspace_id)
        pinned = state.get("pinned_requirements", [])
        req_map = {r.req_id: r for r in pinned}
        retry_queue = list(state.get("retry_queue", []))
        is_retry_pass = bool(retry_queue)
        ids_to_check = retry_queue if is_retry_pass else list(state.get("target_req_ids", []))
        logger.info(
            "Node verify_requirements start workspace=%s is_retry=%s ids=%s",
            workspace_id,
            is_retry_pass,
            ids_to_check,
        )

        evidence_by_id = dict(state.get("evidence_by_id", {}))
        budget = state.get("budget", BudgetState()).model_copy(deep=True)
        audit = list(state.get("audit_log", []))
        next_retry_queue: list[str] = []

        mcp_server = build_mcp_server(bound_workspace_id=workspace_id, store=store)
        async with Client(mcp_server) as client:
            for req_id in ids_to_check:
                req = req_map.get(req_id)
                if req is None:
                    continue

                if req.waived_by_what_if:
                    evidence_by_id[req_id] = RequirementEvidence(
                        req_id=req_id,
                        status="WAIVED_BY_HUMAN",
                        tool_used="what_if_assumption",
                        summary="Satisfied under explicit What-If partner/waiver assumption.",
                    )
                    logger.info("Node verify_requirements req_id=%s action=what_if_waived", req_id)
                    audit.append(
                        AuditEvent(
                            node="verify_requirements",
                            action="what_if_waived",
                            detail=f"{req_id} marked waived via What-If assumption.",
                        )
                    )
                    continue

                if req.kind == "unmapped":
                    evidence_by_id[req_id] = RequirementEvidence(
                        req_id=req_id,
                        status="UNVERIFIABLE_NEEDS_HUMAN",
                        tool_used="none",
                        summary="Criterion does not map to CRM views or narrative archive.",
                    )
                    logger.info(
                        "Node verify_requirements req_id=%s action=flagged_unverifiable",
                        req_id,
                    )
                    audit.append(
                        AuditEvent(
                            node="verify_requirements",
                            action="flagged_unverifiable",
                            detail=f"{req_id} ({req.title}) flagged UNVERIFIABLE_NEEDS_HUMAN.",
                        )
                    )
                    continue

                allowed, halt_reason = can_continue_budget(
                    budget,
                    max_tool_calls=settings.max_tool_calls,
                    max_token_budget=settings.max_token_budget,
                )
                if not allowed:
                    budget.budget_halted = True
                    evidence_by_id[req_id] = RequirementEvidence(
                        req_id=req_id,
                        status="UNVERIFIABLE_NEEDS_HUMAN",
                        tool_used="budget_guard",
                        summary=halt_reason or "Hard execution budget reached.",
                    )
                    logger.warning(
                        "Node verify_requirements req_id=%s budget_halted reason=%s",
                        req_id,
                        halt_reason,
                    )
                    audit.append(
                        AuditEvent(
                            node="verify_requirements",
                            action="budget_halted",
                            detail=halt_reason or "Budget cap reached.",
                        )
                    )
                    break

                retry_attempt = 1 if is_retry_pass else 0
                tool_name, tool_args, in_tok, out_tok = llm.plan_mcp_call(
                    req, retry_attempt=retry_attempt, config=config
                )
                budget.llm_calls += 1
                budget.tool_calls += 1
                budget.input_tokens += in_tok
                budget.output_tokens += out_tok

                raw_res = await client.call_tool(tool_name, tool_args)
                data = parse_mcp_result(raw_res)
                ev = _evaluate_tool_outcome(req, tool_name, data, retried=is_retry_pass)

                if (
                    tool_name == "search_project_narratives"
                    and ev.matched_count == 0
                    and not is_retry_pass
                    and budget.retries < settings.max_retries
                ):
                    budget.retries += 1
                    next_retry_queue.append(req_id)
                    logger.info(
                        "Node verify_requirements req_id=%s queued synonym retry (0 hits)",
                        req_id,
                    )
                    audit.append(
                        AuditEvent(
                            node="verify_requirements",
                            action="queue_synonym_retry",
                            detail=f"{req_id}: 0 narrative hits on '{tool_args.get('query')}'; retrying with broader synonyms.",
                            tokens_used=in_tok + out_tok,
                        )
                    )
                    continue

                evidence_by_id[req_id] = ev
                logger.info(
                    "Node verify_requirements req_id=%s tool=%s status=%s matched=%d",
                    req_id,
                    tool_name,
                    ev.status,
                    ev.matched_count,
                )
                audit.append(
                    AuditEvent(
                        node="verify_requirements",
                        action=f"mcp:{tool_name}",
                        detail=f"{req_id} -> {ev.status}: {ev.summary}",
                        tokens_used=in_tok + out_tok,
                    )
                )

        unverifiable = [
            rid
            for rid, ev in evidence_by_id.items()
            if ev.status == "UNVERIFIABLE_NEEDS_HUMAN"
        ]
        return {
            "evidence_by_id": evidence_by_id,
            "unverifiable_criteria": unverifiable,
            "retry_queue": next_retry_queue,
            "budget": budget,
            "audit_log": audit,
            "status_message": f"Verified {len(ids_to_check)} requirement(s) sequentially.",
        }

    def reconcile_freshness(state: PursuitState) -> dict[str, Any]:
        workspace_id = state.get("workspace_id", settings.default_workspace_id)
        pinned = state.get("pinned_requirements", [])
        project_scopes = {(r.state, r.subsector) for r in pinned if r.subsector}
        active_scopes = project_scopes or {
            (r.state, r.subsector) for r in pinned if r.state or r.subsector
        }
        crm_rows = store.list_projects_for_reconciliation(workspace_id)
        closeouts = store.list_closeout_sheets(workspace_id)
        discrepancies = reconcile_closeouts_against_crm(
            crm_projects=crm_rows,
            closeout_sheets=closeouts,
            active_scopes=active_scopes or None,
            stale_crm_days=settings.stale_crm_days,
        )
        logger.info(
            "Node reconcile_freshness workspace=%s closeouts=%d crm_rows=%d discrepancies=%d",
            workspace_id,
            len(closeouts),
            len(crm_rows),
            len(discrepancies),
        )
        audit = [
            *state.get("audit_log", []),
            AuditEvent(
                node="reconcile_freshness",
                action="fuzzy_reconcile",
                detail=f"Scanned {len(closeouts)} closeouts against {len(crm_rows)} CRM rows; detected {len(discrepancies)} discrepancy/stale flag(s).",
            ),
        ]
        return {
            "discrepancies": discrepancies,
            "audit_log": audit,
            "status_message": f"Found {len(discrepancies)} freshness/entity discrepancy flag(s).",
        }

    def prune_context(state: PursuitState, config: RunnableConfig | None = None) -> dict[str, Any]:
        recent_turns = list(state.get("recent_turns", []))
        summary = state.get("conversation_summary", "")
        budget = state.get("budget", BudgetState()).model_copy(deep=True)
        audit = list(state.get("audit_log", []))

        user_prompt = state.get("followup_input") or f"Initial RFP analysis ({len(state.get('pinned_requirements', []))} criteria)"
        turn_idx = (recent_turns[-1].turn_index + 1) if recent_turns else 1
        recent_turns.append(
            TurnRecord(
                turn_index=turn_idx,
                user_input=user_prompt,
                summary=state.get("status_message", ""),
            )
        )

        if len(recent_turns) > 2:
            dropped = recent_turns[:-2]
            recent_turns = recent_turns[-2:]
            summary, in_tok, out_tok = llm.summarize_turns(summary, dropped, config=config)
            budget.llm_calls += 1
            budget.input_tokens += in_tok
            budget.output_tokens += out_tok
            saved = sum(len(t.user_input) + len(t.summary) for t in dropped) // 4
            budget.pruned_tokens += saved
            logger.info(
                "Node prune_context turn=%d summarized_dropped=%d pruned_tokens_added=%d total_pruned=%d",
                turn_idx,
                len(dropped),
                saved,
                budget.pruned_tokens,
            )
            audit.append(
                AuditEvent(
                    node="prune_context",
                    action="summarize_and_window",
                    detail=f"Trimmed {len(dropped)} older turn(s) into rolling summary; retained last 2 turns + pinned requirements.",
                    tokens_used=in_tok + out_tok,
                )
            )
        else:
            logger.info(
                "Node prune_context turn=%d window_within_limit retained=%d",
                turn_idx,
                len(recent_turns),
            )
            audit.append(
                AuditEvent(
                    node="prune_context",
                    action="window_within_limit",
                    detail=f"Turn {turn_idx}: kept {len(recent_turns)} turn(s) and pinned requirements intact.",
                )
            )

        return {
            "recent_turns": recent_turns,
            "conversation_summary": summary,
            "budget": budget,
            "audit_log": audit,
        }

    def hitl_review_gate(state: PursuitState) -> dict[str, Any]:
        # Zero LLM calls before interrupt() so Command(resume=...) replay is fast & deterministic
        hitl = dict(state.get("hitl_decisions", {}))
        discrepancies = state.get("discrepancies", [])
        unverifiable = state.get("unverifiable_criteria", [])
        pinned_map = {r.req_id: r for r in state.get("pinned_requirements", [])}

        pending_discrepancies = [
            d.model_dump() for d in discrepancies if d.discrepancy_id not in hitl
        ]
        pending_unverifiable = [
            {
                "req_id": rid,
                "title": pinned_map[rid].title if rid in pinned_map else rid,
                "reason": state["evidence_by_id"][rid].summary,
            }
            for rid in unverifiable
            if rid not in hitl and not (rid in pinned_map and pinned_map[rid].waived_by_what_if)
        ]

        if not pending_discrepancies and not pending_unverifiable:
            logger.info("Node hitl_review_gate continuing with 0 pending HITL items")
            return {"hitl_decisions": hitl}

        pending_disc_ids = [d["discrepancy_id"] for d in pending_discrepancies]
        pending_req_ids = [u["req_id"] for u in pending_unverifiable]
        logger.warning(
            "Node hitl_review_gate pausing via interrupt(): discrepancies=%s unverifiable=%s",
            pending_disc_ids,
            pending_req_ids,
        )
        payload = {
            "discrepancies": pending_discrepancies,
            "unverifiable_criteria": pending_unverifiable,
            "instructions": (
                "Provide decisions keyed by discrepancy_id ('use_closeout', 'use_crm', or 'acknowledge') "
                "and/or unverifiable req_id ('waive' or 'reject')."
            ),
        }
        decisions = interrupt(payload)
        decision_map = (
            {str(k): str(v) for k, v in decisions.items()}
            if isinstance(decisions, dict)
            else {}
        )
        hitl.update(decision_map)
        recorded_keys = sorted(decision_map)
        logger.info(
            "Node hitl_review_gate resumed count=%d keys=%s",
            len(recorded_keys),
            recorded_keys,
        )

        return {
            "hitl_decisions": hitl,
            "audit_log": [
                *state.get("audit_log", []),
                AuditEvent(
                    node="hitl_review_gate",
                    action="human_resume",
                    detail=f"Recorded {len(recorded_keys)} HITL decision(s).",
                ),
            ],
        }

    def emit_scorecard(state: PursuitState, config: RunnableConfig | None = None) -> dict[str, Any]:
        requirements = state.get("pinned_requirements", [])
        evidence_by_id = state.get("evidence_by_id", {})
        discrepancies = state.get("discrepancies", [])
        hitl_decisions = state.get("hitl_decisions", {})
        assumptions = state.get("what_if_assumptions", [])
        budget = state.get("budget", BudgetState()).model_copy(deep=True)

        summary_text, in_tok, out_tok = llm.draft_scorecard_summary(
            requirements=requirements,
            evidence_by_id=evidence_by_id,
            discrepancies=discrepancies,
            what_if_assumptions=assumptions,
            config=config,
        )
        budget.llm_calls += 1
        budget.input_tokens += in_tok
        budget.output_tokens += out_tok

        scorecard = enforce_scorecard_invariants(
            proposed_summary=summary_text,
            requirements=requirements,
            evidence_by_id=evidence_by_id,
            discrepancies=discrepancies,
            hitl_decisions=hitl_decisions,
            what_if_assumptions=assumptions,
            budget=budget,
        )
        logger.info(
            "Node emit_scorecard verdict=%s confidence=%s passed=%d failed=%d unverifiable=%d waived=%d guardrail_notes=%d",
            scorecard.verdict,
            scorecard.confidence,
            scorecard.requirements_passed,
            scorecard.requirements_failed,
            scorecard.requirements_unverifiable,
            scorecard.requirements_waived,
            len(scorecard.guardrail_notes),
        )

        recent_turns = list(state.get("recent_turns", []))
        if recent_turns:
            latest = recent_turns[-1].model_copy(deep=True)
            latest.verdict = scorecard.verdict
            recent_turns[-1] = latest

        return {
            "scorecard": scorecard,
            "followup_input": "",
            "recent_turns": recent_turns,
            "budget": budget,
            "status_message": f"Final verdict: {scorecard.verdict}",
            "audit_log": [
                *state.get("audit_log", []),
                AuditEvent(
                    node="emit_scorecard",
                    action=f"verdict:{scorecard.verdict}",
                    detail=scorecard.summary,
                    tokens_used=in_tok + out_tok,
                ),
            ],
        }

    return {
        "entry_router": entry_router,
        "extract_requirements": extract_requirements,
        "apply_what_if_delta": apply_what_if_delta,
        "verify_requirements": verify_requirements,
        "reconcile_freshness": reconcile_freshness,
        "prune_context": prune_context,
        "hitl_review_gate": hitl_review_gate,
        "emit_scorecard": emit_scorecard,
    }
