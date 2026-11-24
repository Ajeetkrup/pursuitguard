"""FastAPI backend with SSE streaming for PursuitGuard."""

from src.api.app import app, create_app

__all__ = ["app", "create_app"]
