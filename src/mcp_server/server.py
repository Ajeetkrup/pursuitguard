"""Official MCP Python SDK v2 read-only CRM server with closure-bound workspace."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from src.logging_config import configure_logging
from src.mcp_server.store import ReadOnlyPursuitStore

logger = logging.getLogger(__name__)


def _non_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _try_parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def build_mcp_server(bound_workspace_id: str, store: ReadOnlyPursuitStore) -> MCPServer:
    """Create a read-only MCP server with workspace_id locked in a server-side closure."""
    mcp = MCPServer("deltek-readonly-mcp")
    logger.debug("Building read-only MCP server bound_workspace=%s", bound_workspace_id)

    @mcp.tool()
    def query_projects(
        subsector: str | None = None,
        state: str | None = None,
        role: str | None = None,
        min_sqft: int | None = None,
        min_fee_usd: float | None = None,
        min_year: int | None = None,
        leed_min: str | None = None,
        search_text: str | None = None,
    ) -> dict[str, Any]:
        """Query CRM projects with exact count, total fee, average sq ft, and sync freshness."""
        res = store.query_projects(
            workspace_id=bound_workspace_id,
            subsector=subsector,
            state=state,
            role=role,
            min_sqft=min_sqft,
            min_fee_usd=min_fee_usd,
            min_year=min_year,
            leed_min=leed_min,
            search_text=search_text,
        )
        filters = _non_none(
            subsector=subsector,
            state=state,
            role=role,
            min_sqft=min_sqft,
            min_fee_usd=min_fee_usd,
            min_year=min_year,
            leed_min=leed_min,
            search_text=search_text,
        )
        logger.info(
            "MCP tool=query_projects workspace=%s filters=%s count=%d total_fee_usd=%.0f",
            bound_workspace_id,
            filters,
            res["count"],
            res["total_fee_usd"],
        )
        return res

    @mcp.tool()
    def check_staff_credentials(
        credential_type: str | None = None,
        state: str | None = None,
        active_only: bool = True,
    ) -> dict[str, Any]:
        """Verify active and expired PE, SE, AIA, or LEED_AP staff credentials."""
        res = store.check_staff_credentials(
            workspace_id=bound_workspace_id,
            credential_type=credential_type,
            state=state,
            active_only=active_only,
        )
        filters = _non_none(
            credential_type=credential_type,
            state=state,
            active_only=active_only,
        )
        logger.info(
            "MCP tool=check_staff_credentials workspace=%s filters=%s active=%d expired=%d",
            bound_workspace_id,
            filters,
            res["active_count"],
            res["expired_count"],
        )
        return res

    @mcp.tool()
    def get_client_history(client_name: str) -> dict[str, Any]:
        """Retrieve historical pursuit outcomes, win rate, and debrief notes for a client."""
        res = store.get_client_history(
            workspace_id=bound_workspace_id,
            client_name=client_name,
        )
        logger.info(
            "MCP tool=get_client_history workspace=%s client=%r count=%d win_rate=%.2f",
            bound_workspace_id,
            client_name,
            res["count"],
            res["win_rate"],
        )
        return res

    @mcp.tool()
    def search_project_narratives(
        query: str,
        subsector: str | None = None,
        include_drafts: bool = False,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Hybrid FTS5 BM25 search over won project narratives (excludes drafts by default)."""
        res = store.search_narratives(
            workspace_id=bound_workspace_id,
            query=query,
            subsector=subsector,
            include_drafts=include_drafts,
            top_k=top_k,
        )
        filters = _non_none(
            query=query[:80],
            subsector=subsector,
            include_drafts=include_drafts,
            top_k=top_k,
        )
        logger.info(
            "MCP tool=search_project_narratives workspace=%s filters=%s hits=%d",
            bound_workspace_id,
            filters,
            len(res),
        )
        return res

    return mcp


def parse_mcp_result(result: Any) -> Any:
    """Decode an MCP SDK v2 CallToolResult into plain Python data."""
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    content = getattr(result, "content", None) or []
    texts = [getattr(block, "text", "") for block in content if getattr(block, "text", None)]
    if not texts:
        return None
    if len(texts) == 1:
        return _try_parse_json(texts[0].strip())
    return [_try_parse_json(raw) for raw in texts]


def main() -> None:
    """Run the read-only CRM MCP server over stdio."""
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    db_path = Path(os.getenv("CRM_DB_PATH", "data/crm.db"))
    workspace_id = os.getenv("PURSUITGUARD_WORKSPACE_ID", "commercial_aec")
    logger.info("Starting stdio MCP server db=%s workspace=%s", db_path, workspace_id)
    store = ReadOnlyPursuitStore(db_path=db_path)
    try:
        server = build_mcp_server(bound_workspace_id=workspace_id, store=store)
        server.run()
    finally:
        logger.info("Stopping stdio MCP server workspace=%s", workspace_id)
        store.close()


if __name__ == "__main__":
    main()
