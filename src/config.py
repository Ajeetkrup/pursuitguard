"""Runtime settings for PursuitGuard."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["gemini", "vertex_gemini", "mock"] = Field(default="mock")
    google_genai_use_vertexai: bool = Field(default=False)
    google_genai_use_enterprise: bool = Field(default=True)
    google_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    google_cloud_project: str = Field(default="")
    google_cloud_location: str = Field(default="global")
    gemini_primary_model: str = Field(default="gemini-2.5-flash")
    gemini_light_model: str = Field(default="gemini-2.5-flash")
    gemini_embed_model: str = Field(default="gemini-embedding-2")

    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    crm_db_path: Path = Field(default=Path("data/crm.db"))
    checkpoint_db_path: Path = Field(default=Path("data/checkpoints.db"))
    trace_log_path: Path = Field(default=Path("data/traces.jsonl"))
    stale_crm_days: int = Field(default=7, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_token_budget: int = Field(default=25000, ge=1000)
    max_retries: int = Field(default=1, ge=0)
    default_workspace_id: str = Field(default="commercial_aec")
    log_level: str = Field(default="INFO")

    @property
    def resolved_google_api_key(self) -> str:
        """Resolve Gemini Developer API or Google Cloud API key from settings or environment."""
        return (
            self.gemini_api_key.strip()
            or self.google_api_key.strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        )

    @property
    def use_vertexai(self) -> bool:
        """Whether live Gemini calls target Vertex AI instead of the Gemini Developer API."""
        if self.llm_provider == "mock":
            return False
        if self.llm_provider == "vertex_gemini":
            return True
        return self.google_genai_use_vertexai

    @property
    def vertex_auth_mode(self) -> str:
        """Return active authentication mode ('mock', 'api_key', or 'adc')."""
        if self.llm_provider == "mock":
            return "mock"
        return "api_key" if self.resolved_google_api_key else "adc"


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
