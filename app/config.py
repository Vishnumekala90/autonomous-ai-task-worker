"""Environment-based configuration. Secrets are SecretStr so they never leak into repr/logs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # LLM: auto | openai | gemini | scripted
    llm_provider: str = "auto"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.0-flash"

    # Agent policy
    approval_threshold: float = 100_000.0
    max_retries: int = 3
    max_steps: int = 30
    retry_backoff_seconds: float = 0.5

    # Controlled local environment
    database_path: Path = ROOT_DIR / "data" / "worker.db"
    invoices_dir: Path = ROOT_DIR / "data" / "invoices"
    enable_fault_injection: bool = True
    log_level: str = "INFO"

    @field_validator("database_path", "invoices_dir")
    @classmethod
    def _anchor_relative_paths(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT_DIR / value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        return None if value == "" else value

    def resolved_provider(self) -> str:
        """Pick the concrete provider. 'auto' prefers a real LLM and falls back to the offline stand-in."""
        provider = self.llm_provider.lower()
        if provider != "auto":
            return provider
        if self.openai_api_key.get_secret_value():
            return "openai"
        if self.gemini_api_key.get_secret_value():
            return "gemini"
        return "scripted"


@lru_cache
def get_settings() -> Settings:
    return Settings()
