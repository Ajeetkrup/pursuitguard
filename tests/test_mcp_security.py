"""Security tests for SQLite read-only driver enforcement and closure-bound MCP isolation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mcp import Client

from src.mcp_server import ReadOnlyPursuitStore, build_mcp_server, parse_mcp_result


@pytest.fixture()
def crm_store() -> ReadOnlyPursuitStore:
    store = ReadOnlyPursuitStore(Path("data/crm.db"))
    try:
        yield store
    finally:
        store.close()


def test_sqlite_driver_blocks_all_mutations(crm_store: ReadOnlyPursuitStore) -> None:
    with pytest.raises(sqlite3.OperationalError):
        crm_store.execute_readonly("DELETE FROM projects WHERE project_id = 'PRJ-21-084'")
    with pytest.raises(sqlite3.OperationalError):
        crm_store.execute_readonly(
            "UPDATE projects SET contract_value_usd = 99999999 WHERE project_id = 'PRJ-21-084'"
        )


async def test_mcp_schema_hides_workspace_and_isolates_defense(crm_store: ReadOnlyPursuitStore) -> None:
    server = build_mcp_server(bound_workspace_id="commercial_aec", store=crm_store)
    async with Client(server) as client:
        listed = await client.list_tools()
        assert len(listed.tools) == 4
        for tool in listed.tools:
            props = tool.input_schema.get("properties", {})
            assert "workspace_id" not in props, f"workspace_id leaked in {tool.name}"

        # Querying with no filters must never expose restricted_defense rows
        proj_res = parse_mcp_result(await client.call_tool("query_projects", {}))
        titles = [r["title"] for r in proj_res["rows"]]
        assert not any("Belvoir" in t or "Cheyenne" in t or "Norfolk" in t for t in titles)

        # Classified defense document (nar_20) and confidential margins must NEVER leak to commercial_aec
        secret_res = parse_mcp_result(
            await client.call_tool(
                "search_project_narratives",
                {"query": "SCIF EMP 94000000"},
            )
        )
        assert secret_res == []

        belvoir_res = parse_mcp_result(
            await client.call_tool(
                "search_project_narratives",
                {"query": "Fort Belvoir Classified Command Facility"},
            )
        )
        assert not any(doc["doc_id"] == "nar_20" for doc in belvoir_res)
        assert not any("18.4%" in doc["excerpt"] or "94,000,000" in doc["excerpt"] for doc in belvoir_res)

        # Default include_drafts=False excludes unapproved 2026 drafts
        acute_res = parse_mcp_result(
            await client.call_tool(
                "search_project_narratives",
                {"query": "acute care ICU renovation", "include_drafts": False},
            )
        )
        assert all(doc["status"] == "won" for doc in acute_res)
        assert not any(doc["doc_id"] in {"nar_06", "nar_07", "nar_09"} for doc in acute_res)


def _walk_schema_for_freeform_objects(node: object, path: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" and not node.get("properties"):
            issues.append(path)
        for key, val in node.items():
            issues.extend(_walk_schema_for_freeform_objects(val, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            issues.extend(_walk_schema_for_freeform_objects(item, f"{path}[{idx}]"))
    return issues


def test_vertex_structured_output_schemas_have_no_freeform_objects() -> None:
    from src.agent.llm import _ExtractedList, _ToolChoice
    from src.agent.models import WhatIfPatch

    for schema_cls in (_ExtractedList, _ToolChoice, WhatIfPatch):
        issues = _walk_schema_for_freeform_objects(schema_cls.model_json_schema())
        assert issues == [], f"{schema_cls.__name__} contains untyped object schemas: {issues}"

    choice = _ToolChoice(
        tool_name="check_staff_credentials",
        credential_type="SE",
        state="CA",
        active_only=True,
        min_sqft=50000,  # belongs to query_projects and must be stripped
    )
    assert choice.to_tool_args() == {"credential_type": "SE", "state": "CA", "active_only": True}
