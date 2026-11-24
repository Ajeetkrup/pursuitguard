"""Blocked fuzzy entity resolution and CRM sync freshness reconciliation."""

from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz

from src.agent.models import FreshnessDiscrepancy

logger = logging.getLogger(__name__)

ABBREVIATIONS = {
    "hlth": "health",
    "ctr": "center",
    "crk": "creek",
    "stn": "station",
    "sf": "san francisco",
    "fed": "federal",
    "med": "medical",
    "ctrl": "central",
    "hist": "historic",
    "hosp": "hospital",
    "expan": "expansion",
    "surg": "surgical",
    "crit": "critical",
    "amb": "ambulatory",
    "tx": "texas",
    "ca": "california",
    "va": "va",
}

NOISE_TOKENS = {"closeout", "final", "sheet", "upgrade", "retrofit", "expansion", "tower", "west"}


def normalize_project_title(title: str, *, drop_noise: bool = False) -> str:
    """Normalize Deltek project codes and AEC abbreviations without discarding phase markers."""
    cleaned = re.sub(r"^prj-\d+-\d+\s*", "", title.strip().lower())
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    expanded = [ABBREVIATIONS.get(tok, tok) for tok in tokens]
    if drop_noise:
        expanded = [tok for tok in expanded if tok not in NOISE_TOKENS]
    return " ".join(expanded)


def _score_pair(sheet_title: str, crm_title: str) -> float:
    norm_sheet = normalize_project_title(sheet_title, drop_noise=True)
    norm_crm = normalize_project_title(crm_title, drop_noise=True)
    if not norm_sheet or not norm_crm:
        return 0.0
    token_set = fuzz.token_set_ratio(norm_sheet, norm_crm)
    token_sort = fuzz.token_sort_ratio(norm_sheet, norm_crm)
    return round(0.65 * token_set + 0.35 * token_sort, 1)


def reconcile_closeouts_against_crm(
    crm_projects: list[dict[str, Any]],
    closeout_sheets: list[dict[str, Any]],
    *,
    active_scopes: set[tuple[str | None, str | None]] | None = None,
    stale_crm_days: int = 7,
    match_threshold: float = 58.0,
    tie_margin: float = 10.0,
) -> list[FreshnessDiscrepancy]:
    """Match closeout sheets to Deltek CRM rows inside (state, subsector) blocks."""
    discrepancies: list[FreshnessDiscrepancy] = []
    matched_crm_ids: set[str] = set()

    def _in_active_scope(state: str, subsector: str) -> bool:
        if not active_scopes:
            return True
        for s_state, s_sub in active_scopes:
            if (s_state is None or s_state.upper() == state.upper()) and (
                s_sub is None or s_sub.lower() == subsector.lower()
            ):
                return True
        return False

    for sheet in closeout_sheets:
        s_state = sheet["state"]
        s_sub = sheet["subsector"]
        if not _in_active_scope(s_state, s_sub):
            continue

        # Block candidate pool strictly by (state, subsector)
        candidates = [
            row
            for row in crm_projects
            if row["state"].upper() == s_state.upper()
            and row["subsector"].lower() == s_sub.lower()
        ]
        if not candidates:
            continue

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in candidates:
            score = _score_pair(sheet["title"], row["title"])
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)

        top_score, top_row = scored[0]
        if top_score < match_threshold:
            logger.debug(
                "Closeout %s below match threshold (best_score=%.1f < %.1f)",
                sheet["doc_id"],
                top_score,
                match_threshold,
            )
            continue

        # Ambiguous tie check: if runner-up is also plausible and within tie_margin, escalate
        if len(scored) > 1 and (top_score - scored[1][0]) < tie_margin and scored[1][0] >= match_threshold:
            second_score, second_row = scored[1]
            logger.warning(
                "Ambiguous entity tie %s: '%s' (%.1f) vs '%s' (%.1f)",
                sheet["doc_id"],
                top_row["title"],
                top_score,
                second_row["title"],
                second_score,
            )
            discrepancies.append(
                FreshnessDiscrepancy(
                    discrepancy_id=f"tie_{sheet['doc_id']}",
                    kind="AMBIGUOUS_ENTITY_MATCH",
                    closeout_doc_id=sheet["doc_id"],
                    closeout_title=sheet["title"],
                    closeout_updated_at=sheet["updated_at"],
                    crm_project_id=top_row["project_id"],
                    crm_title=top_row["title"],
                    crm_last_synced_at=top_row["last_synced_at"],
                    match_score=top_score,
                    candidate_matches=[
                        {"project_id": top_row["project_id"], "title": top_row["title"], "score": top_score},
                        {"project_id": second_row["project_id"], "title": second_row["title"], "score": second_score},
                    ],
                    details=(
                        f"Closeout sheet '{sheet['title']}' closely matches both "
                        f"'{top_row['title']}' ({top_score}) and '{second_row['title']}' ({second_score}). "
                        "Human confirmation required instead of guessing."
                    ),
                )
            )
            continue

        matched_crm_ids.add(top_row["project_id"])
        sheet_newer = sheet["updated_at"] > top_row["last_synced_at"]
        fee_diff = (
            sheet.get("claimed_fee_usd") is not None
            and abs(float(sheet["claimed_fee_usd"]) - float(top_row["contract_value_usd"])) >= 1000.0
        )
        sqft_diff = (
            sheet.get("claimed_sq_ft") is not None
            and int(sheet["claimed_sq_ft"]) != int(top_row["sq_ft"])
        )

        if sheet_newer and (fee_diff or sqft_diff):
            logger.warning(
                "Freshness metric clash %s vs crm=%s (score=%.1f, sheet_date=%s, crm_date=%s)",
                sheet["doc_id"],
                top_row["project_id"],
                top_score,
                sheet["updated_at"],
                top_row["last_synced_at"],
            )
            discrepancies.append(
                FreshnessDiscrepancy(
                    discrepancy_id=f"clash_{sheet['doc_id']}_{top_row['project_id']}",
                    kind="METRIC_CLASH",
                    closeout_doc_id=sheet["doc_id"],
                    closeout_title=sheet["title"],
                    closeout_updated_at=sheet["updated_at"],
                    crm_project_id=top_row["project_id"],
                    crm_title=top_row["title"],
                    crm_last_synced_at=top_row["last_synced_at"],
                    match_score=top_score,
                    candidate_matches=[
                        {
                            "project_id": top_row["project_id"],
                            "crm_fee_usd": top_row["contract_value_usd"],
                            "closeout_fee_usd": sheet.get("claimed_fee_usd"),
                            "crm_sq_ft": top_row["sq_ft"],
                            "closeout_sq_ft": sheet.get("claimed_sq_ft"),
                        }
                    ],
                    details=(
                        f"Freshness clash: '{sheet['title']}' ({sheet['updated_at']}) reports "
                        f"${float(sheet.get('claimed_fee_usd') or 0):,.0f} / {sheet.get('claimed_sq_ft')} sq ft, "
                        f"whereas Deltek '{top_row['title']}' (synced {top_row['last_synced_at']}, "
                        f"{top_row.get('sync_age_days', '?')}d old) reports "
                        f"${float(top_row['contract_value_usd']):,.0f} / {top_row['sq_ft']} sq ft."
                    ),
                )
            )

    # Also surface any stale (>7d) active-scope CRM records without an explicit closeout clash
    for row in crm_projects:
        if row["project_id"] in matched_crm_ids:
            continue
        if not _in_active_scope(row["state"], row["subsector"]):
            continue
        if int(row.get("sync_age_days", 0)) > stale_crm_days:
            logger.info(
                "Stale CRM sync flagged project=%s age_days=%s threshold_days=%d",
                row["project_id"],
                row.get("sync_age_days"),
                stale_crm_days,
            )
            discrepancies.append(
                FreshnessDiscrepancy(
                    discrepancy_id=f"stale_{row['project_id']}",
                    kind="STALE_CRM_SYNC",
                    crm_project_id=row["project_id"],
                    crm_title=row["title"],
                    crm_last_synced_at=row["last_synced_at"],
                    match_score=100.0,
                    details=(
                        f"CRM record '{row['title']}' was last synced {row['last_synced_at']} "
                        f"({row.get('sync_age_days')}d old, exceeds {stale_crm_days}d threshold)."
                    ),
                )
            )

    return discrepancies
