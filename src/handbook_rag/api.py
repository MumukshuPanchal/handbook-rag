"""The HTTP API: the assistant as a service that other programs can call.

Two endpoints so far: ``POST /search`` returns the passages most relevant to a question,
and ``GET /health`` says whether the service can do its job. ``POST /ask``, which
answers with citations, arrives with the answering step.

The application is built by :func:`create_app`, a factory. Tests build one with stand-in
parts; ``handbook-rag serve`` builds the real one.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from handbook_rag import __version__, wiring
from handbook_rag.config import Settings, get_settings
from handbook_rag.index.store import ScoredChunk, VectorStore, VectorStoreError
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import configure_tracing, record_query, record_results, stage, trace_id

# Limits on what a caller may send. A question longer than this is not a question, and
# the embedding model would silently ignore most of it.
MAX_QUERY_CHARS = 500
MAX_LIMIT = 20
# Searched once when the application starts, so that no caller meets a cold model.
WARM_UP_QUERY = "warm up"
UNAVAILABLE = "The search index is unavailable. Try again later."

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
        description="The id of this request's trace, when tracing is on. Quote it in a bug report."
    )


class Health(BaseModel):
    """Whether the service can answer requests."""

    status: Literal["ok", "unavailable"]
    version: str
    chunks: int | None = Field(description="How many chunks the index holds, if it can be reached.")
    detail: str | None = None


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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything slow happens once, here: connecting to the vector store and loading
        # the models. A request must never pay for start-up.
        provider = configure_tracing(settings)
        app.state.store = store if store is not None else wiring.make_store(settings)
        app.state.retriever = (
            retriever
            if retriever is not None
            else wiring.make_retriever(settings, store=app.state.store)
        )
        # A model's first run is far slower than the rest: 1.7 s against 0.06 s when
        # measured. That is paid here too. If the index is not built yet the service
        # still starts, and /health says what is wrong.
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
    )

    @app.get("/health", responses={503: {"model": Health}})
    def health(response: Response) -> Health:
        """Say whether the service can do its job: is the index there, and does it hold anything?"""
        try:
            chunks = app.state.store.count()
        except Exception as error:
            response.status_code = 503
            return Health(
                status="unavailable",
                version=__version__,
                chunks=None,
                detail=f"The vector store cannot be queried: {type(error).__name__}",
            )
        if chunks == 0:
            response.status_code = 503
            return Health(
                status="unavailable",
                version=__version__,
                chunks=0,
                detail="The index is empty. Run `handbook-rag index`.",
            )
        return Health(status="ok", version=__version__, chunks=chunks)

    @app.post("/search", responses={503: {"description": "The vector store is unavailable."}})
    def search(request: SearchRequest) -> SearchResponse:
        """Return the passages of the handbook most relevant to a question, best first."""
        started = time.perf_counter()
        with stage("search", mode=settings.retrieval_mode) as span:
            traced = trace_id(span)
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
                # The cause goes to the log and the trace, not to the caller: it can
                # name internal addresses.
                logger.warning("search failed, trace %s: %s", traced, error)
                raise HTTPException(status_code=503, detail=UNAVAILABLE) from error
            record_results(span, results)
        return SearchResponse(
            results=[Passage.from_result(rank, result) for rank, result in enumerate(results, 1)],
            mode=settings.retrieval_mode,
            reranked=settings.rerank,
            took_ms=round((time.perf_counter() - started) * 1000, 1),
            trace_id=traced,
        )

    return app
