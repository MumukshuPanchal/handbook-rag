"""Build the pipeline's parts from the application settings.

Each stage is defined by a small interface. This module is the one place that decides
which implementation stands behind each interface, so swapping one is a change here and
nowhere else.
"""

import hashlib
from typing import Any, Literal

import anthropic
from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient

from handbook_rag.config import Settings
from handbook_rag.evals.judging import Criterion, Judge, JudgedCase
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT
from handbook_rag.generate.prompt import SYSTEM_PROMPT
from handbook_rag.index.embedder import Embedder, SentenceTransformerEmbedder
from handbook_rag.index.sparse import Bm25Encoder, SparseEncoder
from handbook_rag.index.store import QdrantVectorStore, VectorStore, VectorStoreError
from handbook_rag.retrieve.dense import DenseRetriever, Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.rerank import CrossEncoderReranker, RerankingRetriever
from handbook_rag.retrieve.sparse import SparseRetriever

RetrievalMode = Literal["dense", "sparse", "hybrid"]

# How long the client waits in each phase of a request to an answer model: for the
# connection, and for each next part of the reply.
MODEL_TIMEOUT_SECONDS = 60.0
# How often a request to Gemini is sent at most, and which replies are worth another
# attempt: the server was busy or failed. A connection that fails or times out before
# the reply starts counts as an attempt too. Each attempt can take longer than one time
# limit, and the SDK pauses between attempts, so the worst case is a few minutes.
# A rate limit (429) is not retried. On a free key it means waiting a minute or a day,
# which trying again a second later cannot fix.
GEMINI_ATTEMPTS = 3
GEMINI_RETRY_STATUS_CODES = [408, 500, 502, 503, 504]


def make_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def make_store(
    settings: Settings, *, check: bool = True, timeout: float = 60.0
) -> QdrantVectorStore:
    """Connect to Qdrant, failing with a clear message if it is not running.

    A command wants to know at once that the store is missing, so by default it is
    asked. A long-running service passes ``check=False``: it should start without the
    store, report that in its health check, and work again when the store is back. It
    also passes a shorter ``timeout``, so a store that hangs does not hold a request for
    a minute.
    """
    client = QdrantClient(url=settings.qdrant_url, timeout=int(timeout))
    if check:
        try:
            client.get_collections()
        except Exception as error:
            raise VectorStoreError(
                f"Cannot reach Qdrant at {settings.qdrant_url}. Start it with `make qdrant`."
            ) from error
    return QdrantVectorStore(client, settings.collection_name)


def make_memory_store(collection: str) -> QdrantVectorStore:
    """A vector store held in this process's memory, for a small index that is thrown away.

    It needs no server, which is what lets the retrieval gate run in CI.
    """
    return QdrantVectorStore(QdrantClient(":memory:"), collection)


def make_sparse_encoder(settings: Settings) -> Bm25Encoder:
    return Bm25Encoder(settings.sparse_model)


def make_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.rerank_model)


def describe_retrieval(
    settings: Settings, mode: RetrievalMode | None = None, rerank: bool | None = None
) -> dict[str, object]:
    """The settings that decide what ``make_retriever`` returns, for recording with a run.

    Only what the configuration uses is listed: keyword search has no embedding model,
    and fusion settings mean nothing outside hybrid search.
    """
    mode = mode or settings.retrieval_mode
    reranked = settings.rerank if rerank is None else rerank
    described: dict[str, object] = {
        "handbook_commit": settings.handbook_commit,
        "chunking": settings.chunking.version,
        "collection": settings.collection_name,
        "retrieval_mode": mode,
    }
    if mode in ("dense", "hybrid"):
        described["embedding_model"] = settings.embedding_model
    if mode in ("sparse", "hybrid"):
        described["sparse_model"] = settings.sparse_model
    if mode == "hybrid":
        described["fusion_candidates"] = settings.fusion_candidates
        described["rrf_k"] = settings.rrf_k
    described["rerank"] = reranked
    if reranked:
        described["rerank_model"] = settings.rerank_model
        described["rerank_candidates"] = settings.rerank_candidates
    return described


def make_retriever(
    settings: Settings,
    mode: RetrievalMode | None = None,
    rerank: bool | None = None,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    sparse_encoder: SparseEncoder | None = None,
) -> Retriever:
    """Build the retriever for ``mode`` and ``rerank``, or for the values in the settings.

    Only the models a configuration needs are loaded: keyword search never loads the
    embedding model, and nothing loads the reranker unless reranking is on.

    A caller that has just built an index passes the ``store``, ``embedder`` and
    ``sparse_encoder`` it used, so the retriever searches that index and no model is
    loaded a second time. Whatever is not passed is built from the settings.
    """
    mode = mode or settings.retrieval_mode
    if store is None:
        store = make_store(settings)
    retriever: Retriever
    if mode == "dense":
        retriever = DenseRetriever(embedder or make_embedder(settings), store)
    elif mode == "sparse":
        retriever = SparseRetriever(sparse_encoder or make_sparse_encoder(settings), store)
    else:
        retriever = HybridRetriever(
            DenseRetriever(embedder or make_embedder(settings), store),
            SparseRetriever(sparse_encoder or make_sparse_encoder(settings), store),
            candidates=settings.fusion_candidates,
            rrf_k=settings.rrf_k,
        )
    if settings.rerank if rerank is None else rerank:
        retriever = RerankingRetriever(
            retriever, make_reranker(settings), candidates=settings.rerank_candidates
        )
    return retriever


def make_generator(settings: Settings, *, attempts: int | None = None) -> Generator:
    """Build the answer generator for the configured model.

    The model's name decides the provider: ``gemini-...`` is served by the Gemini API and
    ``claude-...`` by the Claude API. The settings accept no other name.

    A key is passed to its client explicitly. It is never exported into the process
    environment, where other tools could pick it up.

    ``attempts`` is how often one request may be sent, the first time included. Left
    out, a failed request is tried again a few times. An evaluation passes 1: it then
    knows how many requests it has sent, and it tries a failed question again itself.
    """
    if attempts is not None and attempts < 1:
        raise ValueError("A request has to be sent at least once.")
    if settings.answer_model.startswith("gemini-"):
        return _make_gemini_generator(settings, attempts or GEMINI_ATTEMPTS)
    return _make_claude_generator(settings, attempts)


def describe_generation(settings: Settings) -> dict[str, object]:
    """The settings that decide what an answer looks like, for recording with a run.

    The instructions are recorded as a short hash. A reworded instruction is a different
    system, and two runs are comparable only if they name the same one.
    """
    gemini = settings.answer_model.startswith("gemini-")
    instructions = MARKER_SYSTEM_PROMPT if gemini else SYSTEM_PROMPT
    described: dict[str, object] = {
        "answer_model": settings.answer_model,
        "answer_max_tokens": settings.answer_max_tokens,
        "answer_context_chunks": settings.answer_context_chunks,
        "citations": "numbered markers" if gemini else "from the provider's API",
        "instructions": hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16],
    }
    if not gemini:
        described["answer_effort"] = settings.answer_effort
    return described


def _make_gemini_generator(settings: Settings, attempts: int) -> GeminiGenerator:
    return GeminiGenerator(
        _gemini_client(settings, attempts),
        model=settings.answer_model,
        max_tokens=settings.answer_max_tokens,
    )


def _gemini_client(settings: Settings, attempts: int) -> genai.Client:
    key = settings.gemini_api_key
    if key is None or not key.get_secret_value():
        raise GenerationError(
            "No Gemini API key was found. Create a free key in Google AI Studio and put it "
            "in a .env file in the project folder, as GEMINI_API_KEY.",
            reason="configuration",
        )
    return genai.Client(
        api_key=key.get_secret_value(),
        http_options=genai_types.HttpOptions(
            timeout=int(MODEL_TIMEOUT_SECONDS * 1000),  # the SDK counts in milliseconds
            retry_options=genai_types.HttpRetryOptions(
                attempts=attempts, http_status_codes=GEMINI_RETRY_STATUS_CODES
            ),
        ),
    )


def make_judge[C: JudgedCase](
    settings: Settings, criterion: Criterion[C], *, attempts: int | None = None
) -> Judge[C]:
    """Build the judge for one criterion.

    ``attempts`` is how often one request may be sent, as for :func:`make_generator`. An
    evaluation passes 1.
    """
    if attempts is not None and attempts < 1:
        raise ValueError("A request has to be sent at least once.")
    return criterion.judge(
        _gemini_client(settings, attempts or GEMINI_ATTEMPTS),
        model=settings.judge_model,
        max_tokens=settings.judge_max_tokens,
    )


def describe_judge[C: JudgedCase](settings: Settings, criterion: Criterion[C]) -> dict[str, object]:
    """The settings that decide a verdict, for recording with the verdicts.

    The instructions are recorded as a short hash, as for an answer, and so is the layout
    of a request. Verdicts made under reworded instructions, or from a request laid out
    another way, come from a different judge, and its agreement with the labels has to
    be measured again.
    """
    return {
        "judge_model": settings.judge_model,
        "judge_max_tokens": settings.judge_max_tokens,
        "judge_instructions": hashlib.sha256(criterion.instructions.encode("utf-8")).hexdigest()[
            :16
        ],
        "judge_request": criterion.request_fingerprint(),
    }


def _make_claude_generator(settings: Settings, attempts: int | None) -> ClaudeGenerator:
    key = settings.anthropic_api_key
    options: dict[str, Any] = {"timeout": MODEL_TIMEOUT_SECONDS}
    if attempts is not None:
        # The SDK counts the tries after the first one.
        options["max_retries"] = attempts - 1
    # Without a key in the settings, the SDK looks in its own standard places.
    if key is not None and key.get_secret_value():
        options["api_key"] = key.get_secret_value()
    client = anthropic.Anthropic(**options)
    return ClaudeGenerator(
        client,
        model=settings.answer_model,
        max_tokens=settings.answer_max_tokens,
        effort=settings.answer_effort,
    )
