"""Build the pipeline's parts from the application settings.

Each stage is defined by a small interface. This module is the one place that decides
which implementation stands behind each interface, so swapping one is a change here and
nowhere else.
"""

import hashlib
import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import anthropic
from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient

from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.rules import ANSWER_CACHE_VERSION
from handbook_rag.cache.store import QdrantAnswerStore, collection_name
from handbook_rag.cache.verify import EntailmentVerifier, LazyVerifier
from handbook_rag.config import Settings, collection_slug
from handbook_rag.evals.judging import Criterion, Judge, JudgedCase
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT, build_numbered_prompt
from handbook_rag.generate.prompt import SYSTEM_PROMPT, build_user_content
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.index.embedder import Embedder, SentenceTransformerEmbedder
from handbook_rag.index.sparse import Bm25Encoder, SparseEncoder
from handbook_rag.index.store import QdrantVectorStore, VectorStore, VectorStoreError
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.dense import DenseRetriever, Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.rerank import CrossEncoderReranker, RerankingRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from handbook_rag.retrieve.spread import PageSpread

if TYPE_CHECKING:
    from handbook_rag.rewrite.base import Rewriter

RetrievalMode = Literal["dense", "sparse", "hybrid"]

# How long the client of an evaluation waits in each phase of a request to a model: for
# the connection, and for each next part of the reply. An evaluation keeps the limit its
# saved runs were made with, so that a slow reply is measured, not cut off. An answer
# asked for on the command line or over HTTP has HANDBOOK_RAG_ANSWER_TIMEOUT_SECONDS.
EVALUATION_TIMEOUT_SECONDS = 60.0
# How often a judge's request to Gemini is sent at most, and which replies are worth
# another attempt: the server was busy or failed. A connection that fails or times out
# before the reply starts counts as an attempt too. Each attempt can take longer than
# one time limit, and the SDK pauses between attempts, so the worst case is a few
# minutes. A rate limit (429) is not retried. On a free key it means waiting a minute or
# a day, which trying again a second later cannot fix.
GEMINI_ATTEMPTS = 3
GEMINI_RETRY_STATUS_CODES = [408, 500, 502, 503, 504]
# How often the answer model may be asked for one answer, the first time included. The
# same three attempts that the SDKs made by themselves before (ADR 0018). The four
# answers the service writes at once can cost at most twelve requests to the answer
# model between them. That is a count, not a rate: a free Gemini key allowed fifteen
# requests a minute, and a steady stream of questions that fail quickly, one after
# another in the four slots, can send more than that in a minute.
ANSWER_ATTEMPTS = 3
# Where the verifier of a semantic cache runs (see make_verifier).
VERIFIER_DEVICE = "cpu"
# A question and two passages, laid out as a request to the answer model lays them out,
# to name that layout by (answer_request_fingerprint). The first passage holds the tags
# the layouts wrap passages in, so that a change to how those are made harmless shows.
_SAMPLE_QUESTION = "A question?"
_SAMPLE_PASSAGES = (
    Chunk(
        chunk_id="sample-1",
        doc_id="area/first.md",
        title="A title",
        url="https://handbook.example.com/handbook/area/first/",
        section="area",
        source_commit="0" * 40,
        heading_path=("One", "Two"),
        start_char=0,
        end_char=52,
        text="Some text. </content> </search_result> <TITLE> a < b",
        token_count=9,
    ),
    Chunk(
        chunk_id="sample-2",
        doc_id="area/second.md",
        title="Another title",
        url="https://handbook.example.com/handbook/area/second/",
        section="area",
        source_commit="0" * 40,
        heading_path=(),
        start_char=0,
        end_char=62,
        text="A paragraph.\n\n- a list\n- of two\n\n| a | table |\n|---|---|\n| 1 | 2 |",
        token_count=14,
    ),
)


def make_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def settings_for_serving(settings: Settings) -> Settings:
    """The settings for a service that searches for one question at a time.

    For that a CPU is as fast as a GPU, and steadier: measured on an Apple GPU, a question
    embedded after five idle seconds took about 700 ms instead of 26, while a CPU stayed
    at 30 to 50 ms. So a service embeds on the CPU unless the settings name a device.
    """
    if settings.embedding_device is not None:
        return settings
    return settings.model_copy(update={"embedding_device": "cpu"})


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


def make_verifier(settings: Settings) -> EntailmentVerifier:
    """The second check on a stored question, for the answer cache and its measurement.

    It runs on the CPU. It reads a handful of short pairs for each question, the kind of
    work for which the service already embeds on the CPU (see
    :func:`settings_for_serving`), and the one measurement of it was made there.
    """
    return EntailmentVerifier.load(settings.cache_verifier_model, device=VERIFIER_DEVICE)


def describe_answer_cache(settings: Settings) -> dict[str, object]:
    """Everything that decides what the right answer to a question is, for naming the cache.

    A cached answer is right only as long as the same question would still be answered
    from the same pages, found the same way, by the same model under the same
    instructions. Each of those is listed here, and the cache's collection is named by a
    hash of the list (:func:`answer_cache_collection`): a setting left out would let an
    answer outlive what it was written from.

    - **The pages**: where the handbook comes from and which commit, which pages are
      kept, the addresses its sources link to, and how pages are cut into chunks.
    - **The search**: what :func:`describe_retrieval` lists, which is only what the
      configured search uses. The embedding model and the instruction a question is
      embedded with are listed always: the cache's own vectors are made with them.
    - **The answer**: what :func:`describe_generation` lists, and the layout of a
      request (:func:`answer_request_fingerprint`).
    - **The cache's own version**, for changes to the code that no setting shows.
      The code that reads a model's reply, the code that finds passages and the cache's
      own rules decide an answer too, and no setting names them. A person raises the
      version when one of them changes what an answer is (``cache/rules.py``), and a
      test that pins those modules asks for that decision.

    What is not listed decides only whether an answer is served or how fast: the cache's
    mode, threshold, candidates, floor and verifier, the time limits, the fallback
    model, whose answers are never stored, and the devices. One question's section is
    not a setting. It is part of each entry (``cache/answers.py``).
    """
    return {
        "cache_version": ANSWER_CACHE_VERSION,
        "handbook_repo_url": settings.handbook_repo_url,
        "handbook_content_path": settings.handbook_content_path,
        "handbook_base_url": settings.handbook_base_url,
        "min_page_chars": settings.min_page_chars,
        **describe_retrieval(settings),
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        **describe_generation(settings),
        "request": answer_request_fingerprint(settings),
    }


def answer_cache_collection(settings: Settings) -> str:
    """The collection that holds the answers cached under these settings.

    A change to anything :func:`describe_answer_cache` lists gives another name. The
    answers stored before are then not served under the new settings. They stay in
    their old collection, and are served again when the setting is put back, unless
    `handbook-rag cache-clear --all` removed them or `handbook-rag index` changed the
    index in between, which empties every collection of the commit.
    """
    described = json.dumps(describe_answer_cache(settings), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(described.encode("utf-8")).hexdigest()[:16]
    return collection_name(answer_cache_commit(settings), digest)


def answer_cache_commit(settings: Settings) -> str:
    """The handbook commit as the names of its answer collections hold it: its first eight
    characters, for a person to read, and for finding every collection of the commit."""
    return collection_slug(settings.handbook_commit[:8])


def answer_request_fingerprint(settings: Settings) -> str:
    """A short hash of how a request to the answer model is laid out, apart from its instructions.

    It covers the tags passages are wrapped in, their numbering and how their own text
    is kept from closing a tag, for the provider the settings name. An answer written
    from a request laid out another way is an answer of another system.
    """
    laid_out: object
    if settings.answer_model.startswith("gemini-"):
        laid_out = build_numbered_prompt(_SAMPLE_QUESTION, _SAMPLE_PASSAGES)
    else:
        laid_out = build_user_content(_SAMPLE_QUESTION, _SAMPLE_PASSAGES)
    encoded = json.dumps(laid_out, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def make_answer_store(settings: Settings, store: QdrantVectorStore) -> QdrantAnswerStore:
    """Where the answers cached under these settings are kept: a collection of its own,
    beside the handbook's, reached through the same connection."""
    return QdrantAnswerStore(store.client, answer_cache_collection(settings))


def make_answer_cache(
    settings: Settings,
    *,
    store: QdrantVectorStore,
    embedder: Embedder,
    background: bool = True,
) -> AnswerCache | None:
    """The answer cache the settings ask for, or ``None`` when its mode is ``off``.

    It embeds a question with ``embedder``, which is the retriever's own, so the cache's
    vectors are the ones a search would compare. In the mode ``semantic`` it also has
    the verifier, which is loaded when it is first asked: a command that finds no stored
    question near its own never loads it, and a service asks it once when it starts
    (``AnswerCache.warm_up``).

    Answers are stored from a thread of their own unless ``background`` is off.
    """
    if settings.cache_mode == "off":
        return None
    verifier = None
    if settings.cache_mode == "semantic":
        # Looked up when it is called, so that a test's stand-in is the one that is built.
        verifier = LazyVerifier(lambda: make_verifier(settings), name=settings.cache_verifier_model)
    return AnswerCache(
        make_answer_store(settings, store),
        embedder=embedder,
        verifier=verifier,
        threshold=settings.cache_threshold,
        candidates=settings.cache_candidates,
        floor=settings.cache_floor,
        background=background,
    )


def make_retriever_and_cache(
    settings: Settings, store: QdrantVectorStore
) -> tuple[Retriever, AnswerCache | None]:
    """The retriever, and the answer cache that shares its embedding model and its store.

    This is what the ``ask`` command builds, and the service when it has an answer
    model. An evaluation builds its retriever with :func:`make_retriever`, and never a
    cache: it measures the model, and while it runs a cache cannot be built at all
    (``measuring.py``).

    With keyword search and the cache switched off, no embedding model is loaded.
    """
    embedder: Embedder | None = None
    if settings.retrieval_mode != "sparse" or settings.cache_mode != "off":
        embedder = make_embedder(settings)
    retriever = make_retriever(settings, store=store, embedder=embedder)
    if embedder is None:
        return retriever, None
    return retriever, make_answer_cache(settings, store=store, embedder=embedder)


def describe_retrieval(
    settings: Settings,
    mode: RetrievalMode | None = None,
    rerank: bool | None = None,
    *,
    spread: PageSpread | None = None,
) -> dict[str, object]:
    """The settings that decide what ``make_retriever`` returns, for recording with a run.

    Only what the configuration uses is listed: keyword search has no embedding model,
    and fusion settings mean nothing outside hybrid search. The rule that spreads results
    over pages is listed only when it is on, so a run of the default search is described
    as it was before the rule existed.
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
        if spread is not None:
            described["spread_per_page"] = spread.per_page
            described["spread_within"] = spread.within
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
    spread: PageSpread | None = None,
) -> Retriever:
    """Build the retriever for ``mode`` and ``rerank``, or for the values in the settings.

    Only the models a configuration needs are loaded: keyword search never loads the
    embedding model, and nothing loads the reranker unless reranking is on.

    A caller that has just built an index passes the ``store``, ``embedder`` and
    ``sparse_encoder`` it used, so the retriever searches that index and no model is
    loaded a second time. Whatever is not passed is built from the settings.

    ``spread`` is a variant, and off unless it is passed: the merged list of hybrid
    search is spread over pages (see :mod:`handbook_rag.retrieve.spread`). No setting
    turns it on. It is refused for a search that merges nothing, and with reranking,
    which would put the results in another order again.
    """
    mode = mode or settings.retrieval_mode
    reranked = settings.rerank if rerank is None else rerank
    if spread is not None and (mode != "hybrid" or reranked):
        raise ValueError(
            "Results are spread over pages in hybrid search without reranking only, "
            f"not in {mode} search{' with reranking' if reranked else ''}."
        )
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
            spread=spread,
        )
    if reranked:
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

    ``attempts`` is how often the answer model may be asked for one answer, the first
    time included. Left out, it is :data:`ANSWER_ATTEMPTS`. The model's own generator is
    then wrapped in a :class:`RetryingGenerator`, which asks again when that may help,
    within the deadline in the settings, and asks the fallback model when the settings
    name one. Its SDK sends each request once, with the time limit in the settings.

    An evaluation passes 1. It gets the model's own generator, as it always has: no
    wrapper, no fallback and no deadline, the SDK sending each request once, and the
    time limit its saved runs were made with. It then knows how many requests it has
    sent, and it tries a failed question again itself.

    A missing key is found here when the settings are the only place it can be: the
    Gemini key of a Gemini answer model, and the key of the fallback model, whatever its
    provider. A service then finds it when it starts, not at the first failure. A Claude
    answer model may find its credential in the SDK's own places, so a missing one is not
    seen here: its first request fails as a ``configuration`` error instead. Building
    sends no request, so a key that is present but wrong is found only by a request.
    """
    if attempts is not None and attempts < 1:
        raise ValueError("A request has to be sent at least once.")
    if attempts == 1:
        return _model_generator(
            settings,
            settings.answer_model,
            timeout_seconds=EVALUATION_TIMEOUT_SECONDS,
            for_evaluation=True,
        )
    answer_model = _model_generator(
        settings, settings.answer_model, timeout_seconds=settings.answer_timeout_seconds
    )
    fallback: Callable[[], Generator] | None = None
    if settings.answer_fallback_model is not None:
        _check_fallback_key(settings, settings.answer_fallback_model)
        # Built when it is first needed, which on most days is never.
        fallback = partial(
            _model_generator,
            settings,
            settings.answer_fallback_model,
            timeout_seconds=settings.answer_timeout_seconds,
        )
    return RetryingGenerator(
        answer_model,
        attempts=attempts or ANSWER_ATTEMPTS,
        deadline_seconds=settings.answer_deadline_seconds,
        fallback=fallback,
    )


def make_rewriter(settings: Settings) -> "Rewriter":
    """The model that rewrites a follow-up into a question that stands alone (issue 79).

    It is the answer model, used a second time for a much smaller request, so its name
    decides the provider as for an answer. Built as an evaluation builds the answer
    model: its SDK sends each request once, and waits
    :data:`EVALUATION_TIMEOUT_SECONDS` at most for the connection and for each next part
    of a reply, so that a slow reply is measured and not cut off. Nothing asks again:
    the rewriting step makes one request, and an evaluation tries a failed case again
    itself. A service that rewrites before it answers will need a time limit of its own,
    which belongs to the change that puts the step in front of ``POST /ask``.

    A missing Gemini key is found here, as for the answer model. The Claude client's own
    span is switched off: the step opens a span for its request itself.
    """
    # Imported here and not with the rest: nothing else in this module rewrites, and the
    # rewriting package reaches into private names of the answer models' modules, which a
    # process that only answers has no reason to load.
    from handbook_rag.rewrite.claude import ClaudeRewriter
    from handbook_rag.rewrite.gemini import GeminiRewriter

    if settings.answer_model.startswith("gemini-"):
        return GeminiRewriter(
            _gemini_client(settings, 1, timeout_seconds=EVALUATION_TIMEOUT_SECONDS),
            model=settings.answer_model,
        )
    options: dict[str, Any] = {
        "timeout": EVALUATION_TIMEOUT_SECONDS,
        "max_retries": 0,
        "open_telemetry": False,
    }
    key = settings.anthropic_api_key
    if key is not None and key.get_secret_value():
        options["api_key"] = key.get_secret_value()
    return ClaudeRewriter(
        anthropic.Anthropic(**options), model=settings.answer_model, effort=settings.answer_effort
    )


def describe_generation(settings: Settings) -> dict[str, object]:
    """The settings that decide what an answer looks like, for recording with a run.

    The instructions are recorded as a short hash. A reworded instruction is a different
    system, and two runs are comparable only if they name the same one.
    """
    gemini = settings.answer_model.startswith("gemini-")
    instructions = answer_instructions(settings)
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


def answer_instructions(settings: Settings) -> str:
    """The instructions the configured answer model is sent with every question.

    A Gemini model cites by numbered markers, so its instructions say how to write them.
    """
    if settings.answer_model.startswith("gemini-"):
        return MARKER_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _model_generator(
    settings: Settings, model: str, *, timeout_seconds: float, for_evaluation: bool = False
) -> Generator:
    """The generator of one model, named as the settings name a model.

    Its SDK sends each request once, and waits ``timeout_seconds`` at most for the
    connection and for each next part of a reply. Whoever asks again does it in our own
    code, where each request is counted and traced.
    """
    if model.startswith("gemini-"):
        return GeminiGenerator(
            _gemini_client(settings, 1, timeout_seconds=timeout_seconds),
            model=model,
            max_tokens=settings.answer_max_tokens,
        )
    return _make_claude_generator(
        settings, model, timeout_seconds=timeout_seconds, for_evaluation=for_evaluation
    )


def _check_fallback_key(settings: Settings, model: str) -> None:
    """Fail now if the fallback model's provider has no key in the settings.

    A Claude model that writes the answers may find its credential in the SDK's own
    places. A fallback may not, because whether the SDK would find one cannot be known
    without building its client, and the fallback is built only when it is needed.
    """
    gemini = model.startswith("gemini-")
    key = settings.gemini_api_key if gemini else settings.anthropic_api_key
    if key is None or not key.get_secret_value():
        variable = "GEMINI_API_KEY" if gemini else "ANTHROPIC_API_KEY"
        raise GenerationError(
            f"The fallback model {model} needs {variable}, and none was found. Put it in a "
            ".env file in the project folder, or unset HANDBOOK_RAG_ANSWER_FALLBACK_MODEL.",
            reason="configuration",
        )


def _gemini_client(
    settings: Settings, attempts: int, *, timeout_seconds: float = EVALUATION_TIMEOUT_SECONDS
) -> genai.Client:
    key = settings.gemini_api_key
    if key is None or not key.get_secret_value():
        raise GenerationError(
            "No Gemini API key was found. Create a free key in Google AI Studio and put it "
            "in a .env file in the project folder, as GEMINI_API_KEY.",
            reason="configuration",
        )
    # The SDK counts in whole milliseconds, and reads 0 as no time limit at all.
    milliseconds = max(1, round(timeout_seconds * 1000))
    return genai.Client(
        api_key=key.get_secret_value(),
        http_options=genai_types.HttpOptions(
            timeout=milliseconds,
            retry_options=genai_types.HttpRetryOptions(
                attempts=attempts, http_status_codes=GEMINI_RETRY_STATUS_CODES
            ),
        ),
    )


def make_judge[C: JudgedCase](
    settings: Settings, criterion: Criterion[C], *, attempts: int | None = None
) -> Judge[C]:
    """Build the judge for one criterion.

    ``attempts`` is how often its SDK may send one request, the first time included.
    Left out, it is :data:`GEMINI_ATTEMPTS`. An evaluation passes 1, as it does to
    :func:`make_generator`.
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

    A later version of a judge also records its number, so that a saved run says in
    plain words which version judged. Version 1 records none. Every run saved before
    there was a second version holds exactly the four settings below. If version 1
    described itself in any other words, none of those runs could be continued, or
    quoted beside a score, as a run of the judge the code builds today.

    The two hashes are what tells one judge from another. The number is there for the
    reader (see :func:`handbook_rag.evals.judging.judge_version`).
    """
    described: dict[str, object] = {
        "judge_model": settings.judge_model,
        "judge_max_tokens": settings.judge_max_tokens,
        "judge_instructions": hashlib.sha256(criterion.instructions.encode("utf-8")).hexdigest()[
            :16
        ],
        "judge_request": criterion.request_fingerprint(),
    }
    if criterion.version != 1:
        described["judge_version"] = criterion.version
    return described


def _make_claude_generator(
    settings: Settings, model: str, *, timeout_seconds: float, for_evaluation: bool
) -> ClaudeGenerator:
    key = settings.anthropic_api_key
    # The SDK counts the tries after the first one, so none means one request.
    options: dict[str, Any] = {"timeout": timeout_seconds, "max_retries": 0}
    if not for_evaluation:
        # Each request already has a span of our own (generate/retrying.py). The SDK's
        # own span for the same request would count its tokens a second time. An
        # evaluation does not trace, and keeps its client as its runs had it.
        options["open_telemetry"] = False
    # Without a key in the settings, the SDK looks in its own standard places.
    if key is not None and key.get_secret_value():
        options["api_key"] = key.get_secret_value()
    client = anthropic.Anthropic(**options)
    return ClaudeGenerator(
        client,
        model=model,
        max_tokens=settings.answer_max_tokens,
        effort=settings.answer_effort,
    )
