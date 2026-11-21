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

from pydantic import AliasChoices, Field, SecretStr
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
        # Let a field that has an alias also be set by its own name in code and tests.
        populate_by_name=True,
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
    sparse_model: str = Field(
        default="Qdrant/bm25", description="Model that builds the sparse keyword vectors."
    )
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Address of the Qdrant vector store."
    )
    search_limit: int = Field(default=5, gt=0, description="How many chunks a search returns.")
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid",
        description=(
            "How chunks are retrieved: by meaning (dense), by keyword (sparse), or both "
            "merged (hybrid). Hybrid is the default; ADR 0011 gives the measurements "
            "and the reasoning."
        ),
    )
    fusion_candidates: int = Field(
        default=30,
        gt=0,
        description="In hybrid mode, how many candidates each search contributes before merging.",
    )
    rerank: bool = Field(
        default=False,
        description="Whether retrieved candidates are re-scored by a cross-encoder before use.",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        description="Cross-encoder model that scores a question against each candidate chunk.",
    )
    rerank_candidates: int = Field(
        default=30, gt=0, description="How many candidates the first search hands to the reranker."
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description=(
            "Reciprocal rank fusion constant. Larger values flatten the advantage of top ranks."
        ),
    )

    embedding_device: str | None = Field(
        default=None,
        description=(
            "Device the embedding model runs on, such as 'cpu', 'cuda' or 'mps'. Unset: a "
            "GPU when there is one, which is what indexing wants. The HTTP API uses the "
            "CPU unless this is set, because it embeds one question at a time."
        ),
    )
    api_host: str = Field(
        default="127.0.0.1",
        description=(
            "Address the HTTP API listens on. The default accepts connections from this "
            "machine only; the API has no authentication. Inside a container it has to "
            "be 0.0.0.0."
        ),
    )
    api_port: int = Field(default=8000, gt=0, lt=65536, description="Port of the HTTP API.")
    tracing: bool = Field(
        default=False,
        description=(
            "Whether each request is recorded as a trace and sent to the trace backend. "
            "Start a local backend with `make traces`."
        ),
    )
    otlp_endpoint: str = Field(
        default="http://localhost:4318",
        description="Base address of the OTLP/HTTP receiver that traces are sent to.",
    )
    service_name: str = Field(
        default="handbook-rag", description="The name this application has in the trace backend."
    )
    trace_content: bool = Field(
        default=False,
        description=(
            "Whether traces include the text of questions and the message and stack "
            "trace of errors. Off by default: a trace then holds ids, scores, sizes and "
            "timings, and an error by its kind only. Switch on for local debugging."
        ),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        # Read from the standard variable name, without this application's prefix.
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
        description=(
            "Credential for the Claude API. Keep it in .env, which git ignores. When it is "
            "unset, the SDK looks for a credential in its own standard places."
        ),
    )
    answer_model: str = Field(
        default="claude-haiku-5-5", description="Claude model that writes the answers."
    )
    answer_max_tokens: int = Field(
        default=4000,
        gt=0,
        description="Upper limit on the tokens the model may produce for one answer.",
    )
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description=(
            "How much reasoning the model spends per answer. Higher is slower and costs more."
        ),
    )
    answer_context_chunks: int = Field(
        default=5, gt=0, description="How many retrieved chunks the model is given to answer from."
    )

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
        """The vector store collection for this corpus, these models and this chunking.

        Changing any of them gives a new collection, so vectors produced under different
        settings are never mixed in one index.
        """

        def slug(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

        return (
            f"handbook-{self.handbook_commit[:8]}-{slug(self.embedding_model)}"
            f"-{slug(self.sparse_model)}-{self.chunking.version}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
