"""Read-only SQLite CRM store and official MCP SDK v2 server."""

from src.mcp_server.server import build_mcp_server, parse_mcp_result
from src.mcp_server.store import ReadOnlyPursuitStore

__all__ = ["ReadOnlyPursuitStore", "build_mcp_server", "parse_mcp_result"]
