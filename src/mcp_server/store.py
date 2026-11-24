"""Read-only SQLite store backing CRM queries and FTS5 narrative search."""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

LEED_RANK = {"none": 0, "silver": 1, "gold": 2, "platinum": 3}

logger = logging.getLogger(__name__)


def _sync_age_days(synced_iso: str, reference_date: date) -> int:
    return max(0, (reference_date - date.fromisoformat(synced_iso[:10])).days)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sanitize_fts_query(raw_query: str) -> str:
    tokens = [tok for tok in re.findall(r"[A-Za-z0-9]+", raw_query) if len(tok) >= 2]
    if not tokens:
        return ""
    return " OR ".join(f'"{tok}"' for tok in tokens[:16])


class ReadOnlyPursuitStore:
    """Singleton SQLite reader opened strictly in read-only mode."""

    def __init__(self, db_path: Path, reference_date: date = date(2026, 11, 24)) -> None:
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"CRM SQLite DB not found: {self.db_path}")
        self.reference_date = reference_date
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            uri, uri=True, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA query_only = ON;")
        logger.info(
            "Opened read-only SQLite store path=%s mode=ro query_only=ON",
            self.db_path,
        )

    def close(self) -> None:
        """Close the SQLite connection cleanly (important for Windows temp cleanup)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed read-only SQLite store path=%s", self.db_path)

    def execute_readonly(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only SQL statement (used internally and for security assertions)."""
        if self._conn is None:
            raise RuntimeError("SQLite store is closed.")
        cur = self._conn.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
        logger.debug("SQLite execute_readonly returned rows=%d", len(rows))
        return rows

    def query_projects(
        self,
        workspace_id: str,
        subsector: str | None = None,
        state: str | None = None,
        role: str | None = None,
        min_sqft: int | None = None,
        min_fee_usd: float | None = None,
        min_year: int | None = None,
        leed_min: str | None = None,
        search_text: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]

        if subsector:
            clauses.append("LOWER(subsector) = LOWER(?)")
            params.append(subsector.strip())
        if state:
            clauses.append("UPPER(state) = UPPER(?)")
            params.append(state.strip())
        if role:
            clauses.append("LOWER(role) = LOWER(?)")
            params.append(role.strip())
        if min_sqft is not None:
            clauses.append("sq_ft >= ?")
            params.append(int(min_sqft))
        if min_fee_usd is not None:
            clauses.append("contract_value_usd >= ?")
            params.append(float(min_fee_usd))
        if min_year is not None:
            clauses.append("completion_year >= ?")
            params.append(int(min_year))
        if leed_min:
            rank = LEED_RANK.get(leed_min.strip().lower())
            if rank is not None:
                clauses.append("leed_rank >= ?")
                params.append(rank)
        if search_text:
            clauses.append("LOWER(title) LIKE LOWER(?)")
            params.append(f"%{search_text.strip()}%")

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT project_id, title, subsector, state, role, sq_ft,
                   contract_value_usd, completion_year, leed_tier, last_synced_at
            FROM v_projects
            WHERE {where_sql}
            ORDER BY completion_year DESC, contract_value_usd DESC
        """
        rows = self.execute_readonly(sql, tuple(params))
        for row in rows:
            row["sync_age_days"] = _sync_age_days(row["last_synced_at"], self.reference_date)

        total_fee = sum(float(r["contract_value_usd"]) for r in rows)
        avg_sqft = round(sum(int(r["sq_ft"]) for r in rows) / len(rows), 1) if rows else 0.0
        sync_dates = [r["last_synced_at"] for r in rows]
        ages = [r["sync_age_days"] for r in rows]

        return {
            "count": len(rows),
            "total_fee_usd": total_fee,
            "avg_sqft": avg_sqft,
            "oldest_sync_at": min(sync_dates) if sync_dates else None,
            "newest_sync_at": max(sync_dates) if sync_dates else None,
            "max_sync_age_days": max(ages) if ages else 0,
            "rows": rows,
        }

    def check_staff_credentials(
        self,
        workspace_id: str,
        credential_type: str | None = None,
        state: str | None = None,
        active_only: bool = True,
    ) -> dict[str, Any]:
        clauses = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]

        if credential_type:
            clauses.append("UPPER(credential_type) = UPPER(?)")
            params.append(credential_type.strip())
        if state:
            clauses.append("UPPER(state) = UPPER(?)")
            params.append(state.strip())

        where_sql = " AND ".join(clauses)
        all_rows = self.execute_readonly(
            f"""
            SELECT credential_id, staff_name, credential_type, state,
                   license_number, expires_on, is_active, last_synced_at
            FROM v_staff_credentials
            WHERE {where_sql}
            ORDER BY is_active DESC, expires_on DESC
            """,
            tuple(params),
        )
        for row in all_rows:
            row["is_active"] = bool(row["is_active"])
            row["sync_age_days"] = _sync_age_days(row["last_synced_at"], self.reference_date)

        active_count = sum(1 for r in all_rows if r["is_active"])
        expired_count = sum(1 for r in all_rows if not r["is_active"])
        returned_rows = [r for r in all_rows if r["is_active"]] if active_only else all_rows
        sync_dates = [r["last_synced_at"] for r in all_rows]
        ages = [r["sync_age_days"] for r in all_rows]

        return {
            "active_count": active_count,
            "expired_count": expired_count,
            "oldest_sync_at": min(sync_dates) if sync_dates else None,
            "max_sync_age_days": max(ages) if ages else 0,
            "rows": returned_rows,
        }

    def get_client_history(self, workspace_id: str, client_name: str) -> dict[str, Any]:
        rows = self.execute_readonly(
            """
            SELECT pursuit_id, client_name, opportunity_name, pursuit_year,
                   outcome, debrief_score, debrief_notes, last_synced_at
            FROM v_client_pursuits
            WHERE workspace_id = ? AND LOWER(client_name) LIKE LOWER(?)
            ORDER BY pursuit_year DESC
            """,
            (workspace_id, f"%{client_name.strip()}%"),
        )
        for row in rows:
            row["sync_age_days"] = _sync_age_days(row["last_synced_at"], self.reference_date)

        win_count = sum(1 for r in rows if r["outcome"].lower() == "won")
        loss_count = sum(1 for r in rows if r["outcome"].lower() == "lost")
        total = len(rows)
        avg_score = round(sum(float(r["debrief_score"]) for r in rows) / total, 1) if total else 0.0
        sync_dates = [r["last_synced_at"] for r in rows]

        return {
            "client_query": client_name,
            "count": total,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": round(win_count / total, 2) if total else 0.0,
            "avg_debrief_score": avg_score,
            "oldest_sync_at": min(sync_dates) if sync_dates else None,
            "pursuits": rows,
        }

    def search_narratives(
        self,
        workspace_id: str,
        query: str,
        subsector: str | None = None,
        include_drafts: bool = False,
        top_k: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return []

        clauses = ["m.workspace_id = ?"]
        params: list[Any] = [fts_query, workspace_id]

        if subsector:
            clauses.append("LOWER(m.subsector) = LOWER(?)")
            params.append(subsector.strip())
        if not include_drafts:
            clauses.append("LOWER(m.status) = 'won'")

        where_sql = " AND ".join(clauses)
        rows = self.execute_readonly(
            f"""
            SELECT m.doc_id, m.doc_type, m.title, m.subsector, m.state, m.status,
                   m.updated_at, m.claimed_fee_usd, m.claimed_sq_ft, m.completion_year,
                   m.source_file, m.body, m.embedding_json,
                   bm25(narratives_fts) AS bm25_rank
            FROM narratives_fts
            JOIN narratives_meta m ON m.doc_id = narratives_fts.doc_id
            WHERE narratives_fts MATCH ? AND {where_sql}
            ORDER BY CASE WHEN LOWER(m.status) = 'won' THEN 0 ELSE 1 END, bm25_rank ASC
            LIMIT ?
            """,
            (*params, max(top_k * 3, 15)),
        )
        if not rows:
            return []

        # Reciprocal Rank Fusion over BM25 rank + optional cached embedding cosine rank
        bm25_scores: dict[str, float] = {
            row["doc_id"]: 1.0 / (60.0 + idx) for idx, row in enumerate(rows)
        }

        dense_scores: dict[str, float] = {}
        if query_embedding:
            scored_dense: list[tuple[str, float]] = []
            for row in rows:
                emb_raw = row.get("embedding_json")
                if emb_raw:
                    vec = json.loads(emb_raw)
                    if len(vec) == len(query_embedding):
                        scored_dense.append((row["doc_id"], _cosine_similarity(query_embedding, vec)))
            scored_dense.sort(key=lambda item: item[1], reverse=True)
            for idx, (doc_id, _) in enumerate(scored_dense):
                dense_scores[doc_id] = 1.0 / (60.0 + idx)

        results: list[dict[str, Any]] = []
        for row in rows:
            doc_id = row["doc_id"]
            rrf_score = bm25_scores.get(doc_id, 0.0) + dense_scores.get(doc_id, 0.0)
            results.append(
                {
                    "doc_id": doc_id,
                    "doc_type": row["doc_type"],
                    "title": row["title"],
                    "subsector": row["subsector"],
                    "state": row["state"],
                    "status": row["status"],
                    "updated_at": row["updated_at"],
                    "claimed_fee_usd": row["claimed_fee_usd"],
                    "claimed_sq_ft": row["claimed_sq_ft"],
                    "completion_year": row["completion_year"],
                    "source_file": row["source_file"],
                    "excerpt": row["body"][:450],
                    "rrf_score": round(rrf_score, 5),
                }
            )
        results.sort(key=lambda item: (item["status"] != "won", -item["rrf_score"]))
        final_results = results[:top_k]
        logger.debug(
            "FTS narrative search fts_query=%r raw_hits=%d dense_rrf=%s returned=%d",
            fts_query,
            len(rows),
            bool(query_embedding),
            len(final_results),
        )
        return final_results

    def list_closeout_sheets(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return closeout sheets scoped to workspace for freshness reconciliation."""
        return self.execute_readonly(
            """
            SELECT doc_id, title, subsector, state, status, updated_at,
                   claimed_fee_usd, claimed_sq_ft, completion_year, source_file
            FROM narratives_meta
            WHERE workspace_id = ? AND doc_type = 'closeout_sheet' AND LOWER(status) = 'won'
            ORDER BY updated_at DESC
            """,
            (workspace_id,),
        )

    def list_projects_for_reconciliation(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return canonical CRM rows scoped to workspace for freshness reconciliation."""
        rows = self.execute_readonly(
            """
            SELECT project_id, title, subsector, state, role, sq_ft,
                   contract_value_usd, completion_year, last_synced_at
            FROM v_projects
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        for row in rows:
            row["sync_age_days"] = _sync_age_days(row["last_synced_at"], self.reference_date)
        return rows
