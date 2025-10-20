# serving/settings.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Where your chain reads its YAML config from
    app_cfg_path: str = Field(
        default_factory=lambda: (
            os.getenv("APP_CFG_PATH")
            or os.getenv("APP_CONFIG")
            or str(Path(__file__).resolve().parent.parent / "config" / "app.yaml")
        )
    )

    # DB + feature flags (env-first)
    db_dsn: Optional[str] = Field(default=os.getenv("PGVECTOR_DSN"))
    langsmith_tracing: bool = Field(default=os.getenv("LANGSMITH_TRACING", "false").lower() in {"1","true","yes"})
    langsmith_project: str = Field(default=os.getenv("LANGSMITH_PROJECT", "IntelSent"))
    openai_api_key: Optional[SecretStr] = Field(default=os.getenv("OPENAI_API_KEY"))

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    def redacted(self) -> Dict[str, Any]:
        return {
            "app_cfg_path": self.app_cfg_path,
            "db_dsn": ("set" if self.db_dsn else None),
            "langsmith_tracing": self.langsmith_tracing,
            "langsmith_project": self.langsmith_project,
            "openai_api_key": ("set" if self.openai_api_key else None),
        }

