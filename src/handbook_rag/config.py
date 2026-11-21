"""Application settings.

Every setting has a default that works for local development. To override one, set
``HANDBOOK_RAG_<NAME>`` in the environment or in a ``.env`` file in the working directory.
Code reads settings through :func:`get_settings` and never from ``os.environ`` directly,
so there is one place that lists everything the application can be configured with.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from handbook_rag.ingest.chunker import ChunkingConfig


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
    handbook_base_url: str = Field(
        default="https://handbook.gitlab.com/handbook/",
        description="Public address of the handbook, used to build a link to each page.",
    )
    min_page_chars: int = Field(
        default=200,
        gt=0,
        description="Pages with less cleaned text than this are skipped as stubs.",
    )

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model. Its tokenizer also measures chunk sizes.",
    )
    embedding_max_tokens: int = Field(
        default=512,
        gt=0,
        description="Longest input the embedding model reads. Text beyond it is ignored.",
    )
    chunk_max_tokens: int = Field(
        default=320, gt=0, description="Token budget for one chunk, before its context header."
    )
    chunk_min_tokens: int = Field(
        default=60, gt=0, description="Pieces smaller than this are merged into a neighbour."
    )
    chunk_overlap_tokens: int = Field(
        default=40, ge=0, description="How much of a piece may be repeated in the next one."
    )

    embedding_query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description=(
            "Instruction placed before a question when it is embedded. The default is the "
            "one the BGE models were trained with. Chunks are embedded without it."
        ),
    )
    embedding_batch_size: int = Field(
        default=16,
        gt=0,
        description=(
            "How many texts the embedding model processes at once. Larger is not always "
            "faster: memory use grows with batch size and text length, and on a laptop a "
            "batch of 64 long chunks was five times slower than 16."
        ),
    )
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Address of the Qdrant vector store."
    )
    search_limit: int = Field(default=5, gt=0, description="How many chunks a search returns.")

    @property
    def chunking(self) -> ChunkingConfig:
        """The chunking settings as the chunker takes them."""
        return ChunkingConfig(
            max_tokens=self.chunk_max_tokens,
            min_tokens=self.chunk_min_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )

    @property
    def collection_name(self) -> str:
        """The vector store collection for this corpus, embedding model and chunking.

        Changing any of the three gives a new collection, so vectors produced under
        different settings are never mixed in one index.
        """
        model = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.lower()).strip("-")
        return f"handbook-{self.handbook_commit[:8]}-{model}-{self.chunking.version}"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
