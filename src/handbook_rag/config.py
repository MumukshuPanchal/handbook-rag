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

    handbook_repo_url: str = Field(
        default="https://gitlab.com/gitlab-com/content-sites/handbook.git",
        description="Git repository the handbook is downloaded from.",
    )
    handbook_commit: str = Field(
        default="e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8",
        description="Exact commit of the handbook to index. Changing it changes the corpus.",
    )
    handbook_content_path: str = Field(
        default="content/handbook",
        description="Directory inside the repository that holds the handbook pages.",
    )
    handbook_licence: str = Field(
        default="CC-BY-SA-4.0",
        description="SPDX identifier of the licence the handbook text is published under.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
