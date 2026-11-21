"""Application settings.

Every setting has a default that works for local development. To override one, set
``HANDBOOK_RAG_<NAME>`` in the environment or in a ``.env`` file in the working directory.
Code reads settings through :func:`get_settings` and never from ``os.environ`` directly,
so there is one place that lists everything the application can be configured with.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the application."""

    model_config = SettingsConfigDict(
        env_prefix="HANDBOOK_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        # A .env file also holds values meant for other tools, such as API credentials
        # read by an SDK. Ignore them here instead of failing on unknown keys.
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for the handbook snapshot and other local data. Ignored by git.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Minimum severity of log messages to emit.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
