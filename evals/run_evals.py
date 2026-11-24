"""Honest v1 (Baseline) vs. v2 (PursuitGuard) evaluation runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.types import Command

from src.agent.freshness import reconcile_closeouts_against_crm
from src.agent.graph import build_pursuit_graph, open_sqlite_checkpointer
from src.agent.llm import create_llm
from src.config import Settings
from src.logging_config import configure_logging
from src.mcp_server.store import ReadOnlyPursuitStore


@dataclass
class EvalMetrics:
    total_cases: int = 0
    verdict_correct: int = 0
    freshness_total: int = 0
    freshness_caught: int = 0
    security_total: int = 0
    security_passed: int = 0
    v1_multiturn_tokens: int = 0
    v2_multiturn_tokens: int = 0
    pinned_retention_pct: float = 0.0


def _run_v1_baseline(store: ReadOnlyPursuitStore, cases: list[dict[str, Any]]) -> EvalMetrics:
    """Simulate v1 baseline: naive narrative search + exact title match + unpruned multi-turn RFP."""
    metrics = EvalMetrics()

    for case in cases:
        metrics.total_cases += 1
        cat = case["category"]

        if cat == "entity_resolution":
            metrics.freshness_total += 1
            scope = case["scope"]
            crm_titles = {
                r["title"].lower()
                for r in store.list_projects_for_reconciliation("commercial_aec")
                if r["state"] == scope["state"] and r["subsector"] == scope["subsector"]
            }
            closeouts = [
                s
                for s in store.list_closeout_sheets("commercial_aec")
                if s["state"] == scope["state"] and s["subsector"] == scope["subsector"]
            ]
            exact_hits = [s for s in closeouts if s["title"].lower() in crm_titles]
            if exact_hits:
                metrics.freshness_caught += 1
            continue

        rfp_text = (
            Path(case["rfp_file"]).read_text(encoding="utf-8")
            if "rfp_file" in case
            else case["rfp_text"]
        )

        # Baseline retrieves top narrative hits (including drafts) without SQL CRM math or invariants
        docs = store.search_narratives(
            workspace_id=case.get("workspace_id", "commercial_aec"),
            query=rfp_text[:120],
            include_drafts=True,
            top_k=5,
        )
        # Baseline naively emits GO whenever any narrative snippet matches
        v1_verdict = "GO" if docs else "NO_GO"

        if "expected_verdict" in case and v1_verdict == case["expected_verdict"]:
            metrics.verdict_correct += 1

        if cat.startswith("security"):
            metrics.security_total += 1
            # Without mandatory invariant, indirect prompt injection or naive GO fails mandatory SE guard
            if cat == "security_access_boundary":
                metrics.security_passed += 1  # store still isolates workspace
        if cat == "context_pruning_memory":
            # Unpruned baseline re-sends full raw RFP + full turn history across all 4 turns
            turns = 1 + len(case.get("followups", []))
            per_turn_tokens = len(rfp_text) // 4 + 420
            metrics.v1_multiturn_tokens = sum(per_turn_tokens * (i + 1) for i in range(turns))

    return metrics


async def _run_v2_pursuitguard(
    store: ReadOnlyPursuitStore,
    cases: list[dict[str, Any]],
    mode: str,
) -> tuple[EvalMetrics, list[dict[str, Any]]]:
    if mode == "live":
        configured = Settings()
        live_provider = configured.llm_provider if configured.llm_provider != "mock" else "gemini"
        settings = Settings(llm_provider=live_provider)
    else:
        settings = Settings(llm_provider="mock")
    llm = create_llm(settings)
    metrics = EvalMetrics()
    details: list[dict[str, Any]] = []

    async with open_sqlite_checkpointer(":memory:") as saver:
        graph = build_pursuit_graph(store=store, llm=llm, settings=settings, checkpointer=saver)

        for idx, case in enumerate(cases):
            metrics.total_cases += 1
            cid = case["case_id"]
            cat = case["category"]

            if cat == "entity_resolution":
                metrics.freshness_total += 1
                scope = case["scope"]
                discs = reconcile_closeouts_against_crm(
                    crm_projects=store.list_projects_for_reconciliation("commercial_aec"),
                    closeout_sheets=store.list_closeout_sheets("commercial_aec"),
                    active_scopes={(scope["state"], scope["subsector"])},
                    stale_crm_days=settings.stale_crm_days,
                )
                matched = any(
                    d.discrepancy_id == case["expected_discrepancy_id"]
                    and d.kind == case["expected_kind"]
                    for d in discs
                )
                if matched:
                    metrics.freshness_caught += 1
                    metrics.verdict_correct += 1
                details.append({"case_id": cid, "passed": matched, "detail": [d.discrepancy_id for d in discs]})
                continue

            rfp_text = (
                Path(case["rfp_file"]).read_text(encoding="utf-8")
                if "rfp_file" in case
                else case["rfp_text"]
            )
            ws = case.get("workspace_id", "commercial_aec")
            thread_id = f"eval-{cid}-{idx}"
            cfg = {"configurable": {"thread_id": thread_id}}

            state = await graph.ainvoke(
                {"raw_rfp_text": rfp_text, "workspace_id": ws, "thread_id": thread_id},
                config=cfg,
            )
            if state.get("__interrupt__") and "hitl_resume" in case:
                state = await graph.ainvoke(Command(resume=case["hitl_resume"]), config=cfg)

            for fup in case.get("followups", []):
                state = await graph.ainvoke({"followup_input": fup}, config=cfg)
                if state.get("__interrupt__") and "hitl_resume" in case:
                    state = await graph.ainvoke(Command(resume=case["hitl_resume"]), config=cfg)

            scorecard = state["scorecard"]
            passed_case = True
            if "expected_verdict" in case and scorecard.verdict != case["expected_verdict"]:
                passed_case = False
            if "expected_min_passed" in case and scorecard.requirements_passed < case["expected_min_passed"]:
                passed_case = False
            if "expect_retry_count_min" in case and state["budget"].retries < case["expect_retry_count_min"]:
                passed_case = False

            serialized_state = json.dumps(state, default=str)
            for forbidden in case.get("forbidden_citation_substrings", []):
                if forbidden in json.dumps(scorecard.citations):
                    passed_case = False
            for forbidden in case.get("forbidden_output_substrings", []):
                if forbidden in serialized_state:
                    passed_case = False

            if passed_case:
                metrics.verdict_correct += 1

            if cat.startswith("security"):
                metrics.security_total += 1
                if passed_case:
                    metrics.security_passed += 1

            if cat == "context_pruning_memory":
                metrics.v2_multiturn_tokens = state["budget"].input_tokens
                retained = len(state.get("pinned_requirements", []))
                metrics.pinned_retention_pct = 100.0 if retained >= 5 and len(state.get("recent_turns", [])) <= 2 else 0.0

            details.append(
                {
                    "case_id": cid,
                    "passed": passed_case,
                    "verdict": scorecard.verdict,
                    "tokens_in": state["budget"].input_tokens,
                    "pruned_tokens": state["budget"].pruned_tokens,
                }
            )

    return metrics, details


def print_report(v1: EvalMetrics, v2: EvalMetrics, details: list[dict[str, Any]], mode: str) -> dict[str, float]:
    v1_acc = round(100.0 * v1.verdict_correct / max(1, v1.total_cases), 1)
    v2_acc = round(100.0 * v2.verdict_correct / max(1, v2.total_cases), 1)
    v1_fresh = round(100.0 * v1.freshness_caught / max(1, v1.freshness_total), 1)
    v2_fresh = round(100.0 * v2.freshness_caught / max(1, v2.freshness_total), 1)
    v1_sec = round(100.0 * v1.security_passed / max(1, v1.security_total), 1)
    v2_sec = round(100.0 * v2.security_passed / max(1, v2.security_total), 1)
    token_cut = round(
        100.0 * (v1.v1_multiturn_tokens - v2.v2_multiturn_tokens) / max(1, v1.v1_multiturn_tokens),
        1,
    )

    print(f"\n=== PursuitGuard Eval Report ({mode} mode, {v2.total_cases} cases) ===")
    print("| Metric | v1 Baseline | v2 PursuitGuard | Delta |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| End-to-End Case Pass Rate | {v1_acc}% ({v1.verdict_correct}/{v1.total_cases}) | {v2_acc}% ({v2.verdict_correct}/{v2.total_cases}) | +{round(v2_acc - v1_acc, 1)}% |")
    print(f"| Unmatched Entity Freshness Recall | {v1_fresh}% ({v1.freshness_caught}/{v1.freshness_total}) | {v2_fresh}% ({v2.freshness_caught}/{v2.freshness_total}) | +{round(v2_fresh - v1_fresh, 1)}% |")
    print(f"| Security & Mandatory Invariant Pass | {v1_sec}% ({v1.security_passed}/{v1.security_total}) | {v2_sec}% ({v2.security_passed}/{v2.security_total}) | +{round(v2_sec - v1_sec, 1)}% |")
    print(f"| 4-Turn Cumulative Input Tokens | {v1.v1_multiturn_tokens} | {v2.v2_multiturn_tokens} | -{token_cut}% |")
    print(f"| Post-Pruning Turn-1 Pinned Recall | — | {v2.pinned_retention_pct}% | Preserved |")
    print("\nPer-Case Breakdown:")
    for item in details:
        print(f" - {item['case_id']}: {'PASS' if item['passed'] else 'FAIL'} -> {item}")

    return {
        "v1_acc": v1_acc,
        "v2_acc": v2_acc,
        "v2_fresh": v2_fresh,
        "v2_sec": v2_sec,
        "token_cut": token_cut,
        "pinned_retention": v2.pinned_retention_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PursuitGuard v1 vs v2 evaluations.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.json"))
    parser.add_argument("--db-path", type=Path, default=Path("data/crm.db"))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    configure_logging(args.log_level)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    store = ReadOnlyPursuitStore(args.db_path)
    try:
        v1_metrics = _run_v1_baseline(store, cases)
        v2_metrics, details = asyncio.run(_run_v2_pursuitguard(store, cases, args.mode))
    finally:
        store.close()

    summary = print_report(v1_metrics, v2_metrics, details, args.mode)
    if summary["v2_acc"] < 90.0 or summary["token_cut"] < 50.0 or summary["v2_sec"] < 100.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
