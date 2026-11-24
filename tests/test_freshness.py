"""Tests for blocked fuzzy entity resolution, clash detection, and tie escalation."""

from __future__ import annotations

from pathlib import Path

from src.agent.freshness import reconcile_closeouts_against_crm
from src.mcp_server import ReadOnlyPursuitStore


def test_blocked_fuzzy_reconciliation_catches_clashes_and_ties() -> None:
    store = ReadOnlyPursuitStore(Path("data/crm.db"))
    try:
        crm_rows = store.list_projects_for_reconciliation("commercial_aec")
        closeouts = store.list_closeout_sheets("commercial_aec")

        acute_flags = reconcile_closeouts_against_crm(
            crm_projects=crm_rows,
            closeout_sheets=closeouts,
            active_scopes={("TX", "acute_care")},
            stale_crm_days=7,
        )
        by_id = {d.discrepancy_id: d for d in acute_flags}

        # Unmatched Houston Methodist closeout vs abbreviated PRJ-21-084 Methodist Hlth Ph2
        assert "clash_nar_01_PRJ-21-084" in by_id
        assert by_id["clash_nar_01_PRJ-21-084"].kind == "METRIC_CLASH"
        assert "21,400,000" in by_id["clash_nar_01_PRJ-21-084"].details

        # Memorial Hermann expansion sits within tie_margin of Ph1 & Ph2 -> escalates as tie
        assert "tie_nar_02" in by_id
        assert by_id["tie_nar_02"].kind == "AMBIGUOUS_ENTITY_MATCH"
        assert len(by_id["tie_nar_02"].candidate_matches) == 2

        # Water conveyance clash must NOT leak into acute_care scope
        assert "clash_nar_10_PRJ-22-119" not in by_id

        water_flags = reconcile_closeouts_against_crm(
            crm_projects=crm_rows,
            closeout_sheets=closeouts,
            active_scopes={("TX", "water_conveyance")},
            stale_crm_days=7,
        )
        assert any(d.discrepancy_id == "clash_nar_10_PRJ-22-119" for d in water_flags)
    finally:
        store.close()
