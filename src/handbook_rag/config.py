"""Application settings.

Every setting has a default that works for local development. To override one, set
``HANDBOOK_RAG_<NAME>`` in the environment or in a ``.env`` file in the working directory.
Code reads settings through :func:`get_settings` and never from ``os.environ`` directly,
so there is one place that lists everything the application can be configured with.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from handbook_rag.ingest.chunker import ChunkingConfig

# The providers that can write answers, by how their model names start.
ANSWER_MODEL_PREFIXES = ("gemini-", "claude-")


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
    cache_mode: Literal["off", "exact", "semantic"] = Field(
        default="semantic",
        description=(
            "Whether an answer stored earlier is served again without a request to the "
            "answer model. off: never. exact: to the same question, where case, spacing "
            "and the question marks, exclamation marks and full stops at its end do not "
            "count. semantic: also to a question that a second model, the verifier, "
            "judges to mean the same. That serves more and is the risky tier: a question "
            "that only looks the same will sometimes get another question's answer. It "
            "also loads a third model, which is downloaded the first time it is needed: "
            "by the service when it starts, and by `ask` at the first question that has a "
            "stored one near it. The default is the project owner's decision, and ADR "
            "0020 records it with the measured risk beside it."
        ),
    )
    cache_threshold: float = Field(
        default=0.91,
        gt=0.0,
        le=1.0,
        description=(
            "How sure the verifier must be, in its weaker direction, before a stored "
            "answer is served to a question worded differently. Higher is stricter. The "
            "default is fitted on labelled pairs of questions. In the saved run "
            "evals/runs/cache/as-built the measurement's rule gave 0.8669, and every "
            "threshold from there to 0.9131 serves that run exactly the same. The "
            "default is the top of that range, to two places: it costs nothing on the "
            "run, and it stands further above the highest score at which the run holds "
            "a wrong serve, 0.854. Where in the range it stands was chosen with every "
            "pair in view, so nothing tests that choice. That run's other half and its "
            "other splits are the only evidence of how the rule's threshold behaves, "
            "and there is none about real questions."
        ),
    )
    cache_candidates: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "How many of the nearest stored questions the verifier is asked about. Each "
            "one is read twice. The default is the number the measurement used; it is a "
            "setting, and nothing chose it from the pairs."
        ),
    )
    cache_floor: float = Field(
        default=0.7,
        ge=-1.0,
        le=1.0,
        description=(
            "The cosine below which a stored question is no candidate at all, so that "
            "the verifier does not read questions that have little to do with each "
            "other. It saves work, and does nothing for correctness. The default was "
            "read from the labelled pairs with every pair in view: in the three saved "
            "runs no candidate under a cosine of 0.76 scored above 0.13, and with a "
            "floor of 0.7 every question is served what it is served with the "
            "measurement's own 0.5. The wrong serve that scores highest is at a cosine "
            "of 0.789 in the last two runs and of 0.962 in the first: above it each "
            "time."
        ),
    )
    cache_verifier_model: str = Field(
        default="cross-encoder/nli-deberta-v3-small",
        description=(
            "Cross-encoder that checks whether a stored question and a new one imply each "
            "other, the second check the answer cache makes before it serves a stored "
            "answer to a question worded differently. It must be a model trained for "
            "natural language inference whose configuration names one output "
            "'entailment'. Loaded when the cache's mode is semantic, and by "
            "`handbook-rag eval-cache`, which measures the check."
        ),
    )

    embedding_device: str | None = Field(
        default=None,
        description=(
            "Device the embedding model runs on, such as 'cpu', 'cuda' or 'mps'. Unset: a "
            "GPU when there is one, which is what indexing wants. The HTTP API and the MCP "
            "server use the CPU unless this is set, because they embed one question at a time."
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
    api_max_answers_in_flight: int = Field(
        default=4,
        gt=0,
        description=(
            "How many answers the HTTP API writes at once. Each one holds a thread while "
            "the model writes. A request beyond the limit gets 503 with a Retry-After "
            "header. Kept small: a free Gemini key allowed 15 requests a minute when this "
            "was written, so more answers at once would mostly wait on that limit."
        ),
    )
    api_demo: bool = Field(
        default=True,
        description=(
            "Whether the HTTP API serves the demo page at / and its two files under "
            "/static/. Switched off, all three answer 404. Like the rest of the API the "
            "page has no login, and every question asked on it spends the answer model's "
            "allowance (ADR 0016)."
        ),
    )
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
            "timings, and an error by its kind only. The MCP server's log of a failed "
            "search follows it too. Switch on for local debugging."
        ),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        # Read from the provider's standard variable name. This application's prefix is
        # not needed, though the prefixed name works too.
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
        description=(
            "Credential for the Gemini API, needed when the answer model is a Gemini "
            "model. Google AI Studio gives one free of charge. Keep it in .env, which git "
            "ignores."
        ),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        # Read from the provider's standard variable name. This application's prefix is
        # not needed, though the prefixed name works too.
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
        description=(
            "Credential for the Claude API, used when the answer model is a Claude model. "
            "Keep it in .env, which git ignores. When it is unset, the SDK looks for a "
            "credential in its own standard places."
        ),
    )
    answer_model: str = Field(
        default="gemini-3.5-flash-lite",
        description=(
            "Model that writes the answers. Its name selects the provider: gemini-... is "
            "served by the Gemini API, claude-... by the Claude API. It must be an exact "
            "model name: an alias such as gemini-flash-latest is rejected, because it "
            "moves to a new model without notice and results would stop being comparable."
        ),
    )
    answer_max_tokens: int = Field(
        default=4000,
        gt=0,
        description="Upper limit on the tokens the model may produce for one answer.",
    )
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        default="low",
        description=(
            "How much reasoning a Claude model spends per answer. Higher is slower and "
            "costs more. A Gemini model uses its own default."
        ),
    )
    answer_context_chunks: int = Field(
        default=5, gt=0, description="How many retrieved chunks the model is given to answer from."
    )
    # Both limits are in seconds and at most ten minutes. Nobody waits that long for an
    # answer, and the bound turns away a value written in milliseconds by mistake, as
    # the Gemini SDK counts them.
    answer_timeout_seconds: float = Field(
        default=20.0,
        gt=0,
        le=600,
        description=(
            "How long one request to the answer model may be silent: to connect, and "
            "between two pieces of the reply. On our side it limits silence, not the "
            "whole reply. The Gemini SDK also sends it to the API with each request, and "
            "whether the API then ends a reply that runs longer in all was not measured. "
            "The default is above the slowest reply of the answer evaluation, 17.1 s for "
            "the whole of it, and about twice the time within which 95% arrived, 10.1 s. "
            "An evaluation keeps the 60 s its runs were made with (ADR 0018)."
        ),
    )
    answer_deadline_seconds: float = Field(
        default=30.0,
        gt=0,
        le=600,
        description=(
            "How long the answer model may be tried before its first text, counted from "
            "its first request, after the search: no new attempt starts after it, and a "
            "pause before one is cut short. A request "
            "already sent is not cut off by it, and ends at its own time limit if it goes "
            "silent. It may be shorter than that limit: then only quick failures are "
            "tried again. The default leaves room for a request that times out and one "
            "more attempt (ADR 0018)."
        ),
    )
    answer_fallback_model: str | None = Field(
        default=None,
        description=(
            "A second model, asked once when the answer model has given up: it was "
            "overloaded, went silent, reached its limit or used up its allowance for the "
            "day. Not when the answer model is wrongly configured: that is for whoever "
            "runs the service to fix. It is named like the answer model, and its "
            "provider's key must be in the settings when the service starts: "
            "GEMINI_API_KEY for a Gemini model, ANTHROPIC_API_KEY for a Claude model. "
            "Unset or empty: no fallback."
        ),
    )
    eval_requests_per_minute: float = Field(
        default=10.0,
        gt=0,
        le=600,
        description=(
            "The most requests per minute an answer evaluation sends to a model. Keep it "
            "under the limit of the key in use, with room for your own questions: a free "
            "Gemini key allowed 15 a minute for the default model when this was written."
        ),
    )

    judge_model: str = Field(
        default="gemini-3.5-flash-lite",
        description=(
            "Model that judges answers in an evaluation. It must be an exact Gemini model "
            "name. The default is the answer model, because it is the only one a free key "
            "allows enough requests for. A model judging its own answers tends to be "
            "lenient, which is why the judge is checked against labelled answers first."
        ),
    )
    judge_max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Upper limit on the tokens the judge may produce for one answer.",
    )

    @field_validator("judge_model")
    @classmethod
    def _judge_model_is_an_exact_gemini_model(cls, value: str) -> str:
        if not value.startswith("gemini-") or value == "gemini-":
            raise ValueError(f"must be a Gemini model (gemini-...), not {value!r}")
        if value.endswith("-latest"):
            raise ValueError(
                f"must be an exact model name: {value!r} is an alias that moves to a new "
                "model without notice, and verdicts would stop being comparable"
            )
        return value

    @field_validator("answer_model")
    @classmethod
    def _answer_model_names_a_provider(cls, value: str) -> str:
        return _exact_answer_model(value)

    @field_validator("answer_fallback_model")
    @classmethod
    def _fallback_model_names_a_provider(cls, value: str | None) -> str | None:
        # An empty value is what the example file holds, and means no fallback.
        if value is None or value == "":
            return None
        return _exact_answer_model(value)

    @model_validator(mode="after")
    def _fallback_is_another_model(self) -> Self:
        if self.answer_fallback_model == self.answer_model:
            raise ValueError(
                "answer_fallback_model must be another model than answer_model: asked "
                "again, the same model would only spend another request of the same allowance"
            )
        return self

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
        return self.collection_for(self.handbook_commit)

    def collection_for(self, commit: str) -> str:
        """The collection that holds ``commit`` indexed with these models and this chunking.

        Updating the index to a newer commit reads the older commit's collection, and it
        has to be the one made with the same models and chunking as the new one.
        """
        return (
            f"handbook-{commit[:8]}-{collection_slug(self.embedding_model)}"
            f"-{collection_slug(self.sparse_model)}-{self.chunking.version}"
        )


def _exact_answer_model(value: str) -> str:
    """The name of a model that can write answers, or an error that says what is wrong."""
    if not any(value.startswith(prefix) and value != prefix for prefix in ANSWER_MODEL_PREFIXES):
        raise ValueError(
            f"must be a Gemini model (gemini-...) or a Claude model (claude-...), not {value!r}"
        )
    if value.endswith("-latest"):
        raise ValueError(
            f"must be an exact model name: {value!r} is an alias that moves to a new "
            "model without notice, and results would stop being comparable"
        )
    return value


def collection_slug(name: str) -> str:
    """A model's name as a collection name holds it: lower case, with dashes."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()
