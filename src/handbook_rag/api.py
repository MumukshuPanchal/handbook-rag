"""The HTTP API: the assistant as a service that other programs can call.

Two endpoints so far: ``POST /search`` returns the passages most relevant to a question,
and ``GET /health`` says whether the service can do its job. ``POST /ask``, which
answers with citations, arrives with the answering step.

The application is built by :func:`create_app`, a factory. Tests build one with stand-in
parts; ``handbook-rag serve`` builds the real one.
"""

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from handbook_rag import __version__, wiring
from handbook_rag.config import Settings, get_settings
from handbook_rag.index.store import (
    CollectionMissingError,
    ScoredChunk,
    VectorStore,
    VectorStoreError,
)
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import configure_tracing, record_query, record_results, stage, trace_id

# Limits on what a caller may send. A question longer than this is not a question, and
# the embedding model would silently ignore most of it.
MAX_QUERY_CHARS = 500
MAX_LIMIT = 20
# The largest request body that is read. A valid request is far smaller.
MAX_BODY_BYTES = 8_192
# How long one call to the vector store may take before the request gives up.
STORE_TIMEOUT_SECONDS = 10.0
# Searched once when the application starts, so that the models are loaded and have run.
WARM_UP_QUERY = "warm up"
# Where a caller finds the id of the request's trace, on success and on failure.
TRACE_HEADER = "X-Trace-Id"

UNAVAILABLE = "The search index is unavailable. Try again later."
NOT_BUILT = "The index has not been built. Run `handbook-rag index`."
EMPTY = "The index is empty. Run `handbook-rag index`."
UNREACHABLE = "The vector store cannot be reached."

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """A question to search the handbook for."""

    # A misspelt field is rejected instead of being ignored without a word.
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
    ] = Field(description="The question, in plain language.")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=MAX_LIMIT,
        # Strict: `true` and "3" are mistakes by the caller, not numbers.
        strict=True,
        description="How many passages to return. Default: the service's search limit.",
    )
    section: str | None = Field(
        default=None,
        max_length=100,
        description="Search only one handbook section, such as 'engineering'.",
    )


class Passage(BaseModel):
    """One retrieved chunk of the handbook."""

    rank: int = Field(description="Position in the results, starting at 1.")
    score: float = Field(description="Higher is more relevant. Comparable within one response.")
    doc_id: str = Field(description="The page, as its path in the handbook.")
    title: str
    url: str = Field(description="Where the page can be read.")
    section: str
    heading_path: list[str] = Field(description="The headings above the passage, outermost first.")
    text: str

    @classmethod
    def from_result(cls, rank: int, result: ScoredChunk) -> "Passage":
        chunk = result.chunk
        return cls(
            rank=rank,
            score=result.score,
            doc_id=chunk.doc_id,
            title=chunk.title,
            url=chunk.url,
            section=chunk.section,
            heading_path=list(chunk.heading_path),
            text=chunk.text,
        )


class SearchResponse(BaseModel):
    """The passages found for a question, best first."""

    results: list[Passage]
    mode: Literal["dense", "sparse", "hybrid"] = Field(description="How the passages were found.")
    reranked: bool
    took_ms: float = Field(description="Time spent searching, in milliseconds.")
    trace_id: str | None = Field(
        description=(
            "The id of this request's trace, when tracing is on. Quote it in a bug report. "
            f"It is also sent in the {TRACE_HEADER} header, on failures too."
        )
    )


class Health(BaseModel):
    """Whether the service can answer requests."""

    status: Literal["ok", "unavailable"]
    version: str
    chunks: int | None = Field(description="How many chunks the index holds, if it can be reached.")
    detail: str | None = None


def _is_health_check(scope: MutableMapping[str, Any]) -> bool:
    """Whether a request is a health probe, which is not worth a trace.

    Something asks for /health every few seconds. Traced, those requests would bury the
    ones made by people.
    """
    return scope.get("path") == "/health"


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    store: VectorStore | None = None,
) -> FastAPI:
    """Build the application.

    Whatever is not passed is built from the settings when the application starts.
    Tests pass a ``retriever`` and a ``store`` made of stand-in parts.
    """
    settings = settings or get_settings()
    # A service embeds one question at a time. For that a CPU is as fast as a GPU, and
    # steadier: measured on an Apple GPU, a question embedded after five idle seconds
    # took about 700 ms instead of 26, while a CPU stayed at 30 to 50 ms.
    serving = settings
    if settings.embedding_device is None:
        serving = settings.model_copy(update={"embedding_device": "cpu"})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything slow happens once, here: loading the models and running them for
        # the first time. No request pays for start-up.
        provider = configure_tracing(settings)
        # The store is not asked whether it is there: the service starts without it,
        # /health says what is wrong, and searches work as soon as it is back.
        app.state.store = (
            store
            if store is not None
            else wiring.make_store(serving, check=False, timeout=STORE_TIMEOUT_SECONDS)
        )
        app.state.retriever = (
            retriever
            if retriever is not None
            else wiring.make_retriever(serving, store=app.state.store)
        )
        with suppress(VectorStoreError), stage("warm_up"):
            app.state.retriever.retrieve(WARM_UP_QUERY, limit=1)
        try:
            yield
        finally:
            if provider is not None:
                provider.shutdown()

    app = FastAPI(
        title="Handbook assistant",
        version=__version__,
        summary="Search a company handbook by meaning and by keyword.",
        lifespan=lifespan,
        telemetry={
            "exclude": _is_health_check,
            # The framework's own log records of a rejected request include the
            # caller's input, so they are switched off.
            "logs": False,
        },
    )

    @app.middleware("http")
    async def refuse_large_bodies(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Turn away a request that announces a body far larger than any valid one."""
        announced = request.headers.get("content-length", "")
        if announced.isdigit() and int(announced) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"The request body is larger than {MAX_BODY_BYTES} bytes."},
            )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def explain_without_echoing(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """Say which field is wrong and why, without sending the caller's input back.

        The default response repeats the rejected value, however large it is.
        """
        problems = [
            {"type": problem["type"], "loc": problem["loc"], "msg": problem["msg"]}
            for problem in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": problems})

    @app.get("/health", responses={503: {"model": Health}})
    def health(response: Response) -> Health:
        """Say whether the service can do its job: is the index there, and does it hold anything?"""
        problem: str | None = None
        chunks: int | None = None
        try:
            chunks = app.state.store.count()
        except CollectionMissingError:
            problem = NOT_BUILT
        except Exception as error:
            # The cause can name internal addresses, so it goes to the log only.
            logger.warning("health check failed: %s: %s", type(error).__name__, error)
            problem = UNREACHABLE
        if chunks == 0:
            problem = EMPTY
        if problem is not None:
            response.status_code = 503
            return Health(status="unavailable", version=__version__, chunks=chunks, detail=problem)
        return Health(status="ok", version=__version__, chunks=chunks)

    @app.post("/search", responses={503: {"description": "The vector store is unavailable."}})
    def search(request: SearchRequest, response: Response) -> SearchResponse:
        """Return the passages of the handbook most relevant to a question, best first."""
        started = time.perf_counter()
        with stage("search", mode=settings.retrieval_mode) as span:
            traced = trace_id(span)
            trace_headers = {TRACE_HEADER: traced} if traced is not None else None
            record_query(span, request.query)
            try:
                results = app.state.retriever.retrieve(
                    request.query,
                    limit=request.limit or settings.search_limit,
                    section=request.section,
                )
            except VectorStoreError as error:
                # A dependency is down: that is not the caller's fault (4xx) and not a
                # bug here (500). 503 tells the caller that trying again later may work.
                # The cause goes to the log, not to the caller: it can name internal
                # addresses. The trace records the kind of error.
                logger.warning("search failed, trace %s: %s", traced, error)
                raise HTTPException(
                    status_code=503, detail=UNAVAILABLE, headers=trace_headers
                ) from error
            record_results(span, results)
        if traced is not None:
            response.headers[TRACE_HEADER] = traced
        return SearchResponse(
            results=[Passage.from_result(rank, result) for rank, result in enumerate(results, 1)],
            mode=settings.retrieval_mode,
            reranked=settings.rerank,
            took_ms=round((time.perf_counter() - started) * 1000, 1),
            trace_id=traced,
        )

    return app
