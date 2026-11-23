"""The HTTP API: the assistant as a service that other programs can call.

Three endpoints: ``POST /ask`` answers a question with citations, as one JSON response
or as server-sent events while the model writes; ``POST /search`` returns the passages
most relevant to a question; and ``GET /health`` says whether the service can do its job.
Why ``/ask`` works the way it does is in ADR 0015. Unless it is switched off, the service
also serves a demo page at ``/`` that asks ``/ask`` (see ``demo.py`` and ADR 0016).

The application is built by :func:`create_app`, a factory. Tests build one with stand-in
parts; ``handbook-rag serve`` builds the real one.
"""

import json
import logging
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.types import Receive, Scope, Send

from handbook_rag import __version__, demo, wiring
from handbook_rag.config import Settings, get_settings
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer
from handbook_rag.index.store import (
    CollectionMissingError,
    ScoredChunk,
    VectorStore,
    VectorStoreError,
)
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import answer_question
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.streaming import (
    AnswerRun,
    Failed,
    Finished,
    OnText,
    OnTrace,
    Slots,
    Text,
    server_sent_event,
)
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

# What a caller who is turned away because answers are full should wait, in seconds. A
# typical answer took two to three seconds when measured, and 95% took under ten.
BUSY_RETRY_AFTER_SECONDS = 5
# The longest wait a Retry-After header asks for. A provider's per-minute limit needs
# far less, and a caller told to wait for days would simply not come back.
MAX_RETRY_AFTER_SECONDS = 3600
# The headers of a stream. A proxy that holds a response back until it is complete
# would hold back every event; these two ask caches and proxies not to.
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

UNAVAILABLE = "The search index is unavailable. Try again later."
NOT_BUILT = "The index has not been built. Run `handbook-rag index`."
EMPTY = "The index is empty. Run `handbook-rag index`."
UNREACHABLE = "The vector store cannot be reached."
# What a caller of /ask is told when no answer can be given. Fixed words: the messages
# of the errors behind them name files, settings and the provider's own wording.
NOT_CONFIGURED = "Answering is not configured on this server."
BUSY = "The service is writing as many answers as it can. Try again in a few seconds."
MODEL_RATE_LIMITED = "The answer model's rate limit was reached. Try again shortly."
MODEL_QUOTA = "The answer model's allowance for today is used up. Try again tomorrow."
MODEL_UNAVAILABLE = "The answer model is unavailable. Try again later."
NOT_FINISHED = "The answer could not be finished."

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


class AskRequest(BaseModel):
    """A question to answer from the handbook."""

    # A misspelt field is rejected instead of being ignored without a word.
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
    ] = Field(description="The question, in plain language.")
    section: str | None = Field(
        default=None,
        max_length=100,
        description="Answer only from one handbook section, such as 'engineering'.",
    )
    stream: bool = Field(
        default=False,
        # Strict: "true" and 1 are mistakes by the caller, not yes.
        strict=True,
        description=(
            "Send the answer as server-sent events while the model writes it, instead of "
            "one JSON response at the end."
        ),
    )


class Source(BaseModel):
    """A passage the answer cites, under the number the answer gives it."""

    number: int = Field(description="The number in the answer's markers: 1 for [1].")
    doc_id: str = Field(description="The page, as its path in the handbook.")
    title: str
    url: str = Field(description="Where the page can be read.")
    section: str
    heading_path: list[str] = Field(description="The headings above the passage, outermost first.")
    text: str

    @classmethod
    def from_chunk(cls, number: int, chunk: Chunk) -> "Source":
        return cls(
            number=number,
            doc_id=chunk.doc_id,
            title=chunk.title,
            url=chunk.url,
            section=chunk.section,
            heading_path=list(chunk.heading_path),
            text=chunk.text,
        )


class AnswerSegment(BaseModel):
    """A run of the answer's text, with the sources it rests on."""

    text: str
    sources: list[int] = Field(
        description="The numbers of the sources this text rests on. Empty when it cites none."
    )


class Usage(BaseModel):
    """The tokens of the request to the model."""

    input_tokens: int = Field(
        description="Tokens the model read, not counting any served from the provider's cache."
    )
    output_tokens: int = Field(description="Tokens the model wrote, its reasoning included.")


class AskResponse(BaseModel):
    """An answer, with the passages it cites."""

    answer: str = Field(
        description=(
            "The answer, with a numbered marker such as [1] after each statement a source "
            "supports. A reply that is not an answer carries no markers."
        )
    )
    answered: bool = Field(
        description=(
            "False when the assistant declined, the provider withheld the reply, or the "
            "reply did not come to a proper end."
        )
    )
    segments: list[AnswerSegment] = Field(description="The answer in order, piece by piece.")
    sources: list[Source] = Field(
        description="The cited passages, numbered as in the answer. Empty when it is not an answer."
    )
    model: str = Field(description="The model that wrote the reply.")
    stop_reason: str = Field(
        description="Why the model stopped: end_turn, max_tokens, refusal, no_context or another."
    )
    usage: Usage
    took_ms: float = Field(description="Time from the request to the whole answer, in ms.")
    first_text_ms: float | None = Field(
        description=(
            "Time from the request to the first words of the answer, in milliseconds. "
            "Null when no text came."
        )
    )
    trace_id: str | None = Field(
        description=(
            "The id of this request's trace, when tracing is on. Quote it in a bug report. "
            f"It is also sent in the {TRACE_HEADER} header, on failures too."
        )
    )


class Problem(BaseModel):
    """Why a request could not be served, in fixed words."""

    detail: str


class Health(BaseModel):
    """Whether the service can answer requests."""

    status: Literal["ok", "unavailable"]
    version: str
    chunks: int | None = Field(description="How many chunks the index holds, if it can be reached.")
    answering: bool = Field(
        description=(
            "Whether an answer model is configured, so that POST /ask can answer. Found out "
            "from the settings, without a request to the model."
        )
    )
    detail: str | None = None


# How /ask describes its two forms of success in the API's documentation.
ASK_SUCCESS = (
    "The answer. By default one JSON object, described below. With `stream` true, a "
    "stream of server-sent events (`text/event-stream`): each piece of text as the model "
    'writes it, as `event: text` with `data: {"text": "..."}`, then one `event: answer` '
    "whose data is the same JSON object. If the answer fails after it has begun, the "
    'stream ends with one `event: error` with `data: {"detail": "..."}` instead.'
)
ASK_STREAM_EXAMPLE = (
    "event: text\n"
    'data: {"text":"New parents get sixteen weeks of paid leave."}\n'
    "\n"
    "event: answer\n"
    'data: {"answer":"New parents get sixteen weeks of paid leave.[1]","answered":true,'
    '"segments":[{"text":"New parents get sixteen weeks of paid leave.","sources":[1]}],'
    '"sources":[{"number":1,"doc_id":"people/leave.md","title":"Leave",'
    '"url":"https://handbook.example.com/handbook/people/leave/","section":"people",'
    '"heading_path":["Parental leave"],"text":"New parents get sixteen weeks of paid leave."}],'
    '"model":"gemini-3.5-flash-lite","stop_reason":"end_turn",'
    '"usage":{"input_tokens":1200,"output_tokens":20},"took_ms":1500.0,"first_text_ms":1300.0,'
    '"trace_id":null}\n'
    "\n"
)
ASK_UNAVAILABLE = (
    "No answer could be started, or none could be finished before anything was sent: the "
    "vector store or the answer model is unavailable, the model's limit was reached, "
    "answering is not configured, or the service is writing as many answers as it can. "
    "The body says which, in fixed words. This is also the response when `stream` is true "
    "and the failure came before the first text."
)


def _json_line(value: object) -> str:
    """JSON on one line, as compact as the framework writes it, with text left in UTF-8.

    Both forms of an answer are written through here, so this is the one place that
    makes sure the text can be sent. A model can write a lone half of a surrogate pair,
    which UTF-8 cannot encode; it becomes a question mark instead of failing the
    response, or a stream halfway through.
    """
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _respond(answer: Answer, *, started: float, run: AnswerRun) -> AskResponse:
    """The response to a question, from the answer and the run that wrote it."""
    # A reply that is not an answer lists no sources, even if the model attached one to
    # it. The command line does the same. Its markers would then point at nothing.
    sources = answer.sources if answer.answered else ()
    numbers = {chunk.chunk_id: number for number, chunk in enumerate(sources, start=1)}
    first_text = None if run.first_text_at is None else _milliseconds(run.first_text_at - started)
    return AskResponse(
        answer=answer.text_with_markers if answer.answered else answer.text,
        answered=answer.answered,
        segments=[
            AnswerSegment(
                text=segment.text,
                sources=sorted(
                    {
                        numbers[citation.chunk.chunk_id]
                        for citation in segment.citations
                        if citation.chunk.chunk_id in numbers
                    }
                ),
            )
            for segment in answer.segments
        ],
        sources=[Source.from_chunk(number, chunk) for number, chunk in enumerate(sources, 1)],
        model=answer.model,
        stop_reason=answer.stop_reason,
        usage=Usage(
            input_tokens=answer.usage.input_tokens, output_tokens=answer.usage.output_tokens
        ),
        took_ms=_milliseconds(time.perf_counter() - started),
        first_text_ms=first_text,
        trace_id=run.trace_id,
    )


def _milliseconds(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _refusal(error: BaseException) -> tuple[str, dict[str, str]] | None:
    """What a caller is told about a failure: fixed words and any headers to go with them.

    ``None`` means a fault in this service, which a caller is not told about in words.
    """
    if isinstance(error, VectorStoreError):
        return UNAVAILABLE, {}
    if not isinstance(error, GenerationError):
        return None
    if error.reason == "rate_limit":
        if error.retry_after is None:
            return MODEL_RATE_LIMITED, {}
        # The header takes whole seconds. Rounded up, so a caller who waits that long
        # does not come back a moment too early, and never more than an hour.
        wait = min(MAX_RETRY_AFTER_SECONDS, max(1, math.ceil(error.retry_after)))
        return MODEL_RATE_LIMITED, {"Retry-After": str(wait)}
    if error.reason == "quota":
        return MODEL_QUOTA, {}
    return MODEL_UNAVAILABLE, {}


def _trace_headers(run: AnswerRun) -> dict[str, str]:
    return {TRACE_HEADER: run.trace_id} if run.trace_id is not None else {}


class EventStream(StreamingResponse):
    """A stream of server-sent events that stops its answer when it ends, however it ends.

    When the caller goes away, the server's next attempt to send fails, or the work of
    sending is cancelled. Either way the ``finally`` below runs and asks the worker to
    stop, so the model does not go on writing an answer that nobody reads. The event
    source itself is not relied on for this: a generator that stops being iterated is
    closed only when it is garbage-collected.
    """

    media_type = "text/event-stream"

    def __init__(
        self,
        content: AsyncIterator[bytes],
        *,
        on_close: Callable[[], None],
        headers: Mapping[str, str],
    ) -> None:
        super().__init__(content, headers=headers)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._on_close()


async def _events(run: AnswerRun, first: Text | Finished, started: float) -> AsyncIterator[bytes]:
    """The events of a stream that has begun: text, then the answer or an error."""
    event: Text | Finished | Failed = first
    while isinstance(event, Text):
        yield server_sent_event("text", _json_line({"text": event.text}))
        event = await run.next()
    if isinstance(event, Finished):
        body = _respond(event.answer, started=started, run=run)
        yield server_sent_event("answer", _json_line(body.model_dump(mode="json")))
        return
    refusal = _refusal(event.error)
    if refusal is None:
        logger.error("answer failed after it began, trace %s", run.trace_id, exc_info=event.error)
        detail = NOT_FINISHED
    else:
        logger.warning("answer failed after it began, trace %s: %s", run.trace_id, event.error)
        detail = refusal[0]
    yield server_sent_event("error", _json_line({"detail": detail}))


def _answer_model(settings: Settings) -> Generator | None:
    """The answer model, or ``None`` when the settings do not allow one, such as without a key.

    Searching needs no answer model, so the service starts without one and says so.
    """
    try:
        return wiring.make_generator(settings)
    except GenerationError as error:
        logger.warning("answering is not configured: %s", error)
        return None


# The addresses whose requests are not worth a trace: health probes, and the demo page
# and its two files.
UNTRACED_PATHS = frozenset({"/health", *demo.FILES})


def _is_untraced(scope: MutableMapping[str, Any]) -> bool:
    """Whether a request is to /health or to one of the demo page's three addresses.

    Something asks for /health every few seconds, and a page load is not a question.
    Traced, those requests would bury the ones made by people. The address alone decides,
    so a request to one of them is not traced whatever its method, and also while the
    page is switched off and the address answers 404. A question asked on the page is a
    request to /ask, and is traced like any other.
    """
    return scope.get("path") in UNTRACED_PATHS


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    store: VectorStore | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    """Build the application.

    Whatever is not passed is built from the settings when the application starts.
    Tests pass a ``retriever``, a ``store`` and a ``generator`` made of stand-in parts.
    """
    settings = settings or get_settings()
    # A service embeds one question at a time, which a CPU does as fast as a GPU.
    serving = wiring.settings_for_serving(settings)

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
        # Building the answer model sends no request. So a missing key is found out here,
        # and a wrong one only by the first question.
        app.state.generator = generator if generator is not None else _answer_model(settings)
        app.state.answer_slots = Slots(settings.api_max_answers_in_flight)
        try:
            yield
        finally:
            if provider is not None:
                provider.shutdown()

    app = FastAPI(
        title="Handbook assistant",
        version=__version__,
        summary="Answer questions from a company handbook with cited passages, and search it.",
        lifespan=lifespan,
        telemetry={
            "exclude": _is_untraced,
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
        # Without an answer model the service still searches, so it is not unhealthy.
        answering = app.state.generator is not None
        if problem is not None:
            response.status_code = 503
            return Health(
                status="unavailable",
                version=__version__,
                chunks=chunks,
                answering=answering,
                detail=problem,
            )
        return Health(status="ok", version=__version__, chunks=chunks, answering=answering)

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

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={
            200: {
                "description": ASK_SUCCESS,
                "content": {
                    "text/event-stream": {
                        "schema": {"type": "string"},
                        "example": ASK_STREAM_EXAMPLE,
                    }
                },
            },
            503: {
                "model": Problem,
                "description": ASK_UNAVAILABLE,
                "headers": {
                    "Retry-After": {
                        "description": (
                            "Seconds to wait before asking again, when that is known: the "
                            "service was full, or the model's per-minute limit was reached."
                        ),
                        "schema": {"type": "integer"},
                    },
                    TRACE_HEADER: {
                        "description": "The id of the request's trace, when it was traced.",
                        "schema": {"type": "string"},
                    },
                },
            },
        },
    )
    async def ask(request: AskRequest) -> Response:
        """Answer a question from the handbook, citing the passages the answer rests on.

        The answer is written in a worker thread. The request waits for the first thing
        that happens there, and only then decides what to send: a failure before any
        text is an HTTP error, also when a stream was asked for.
        """
        started = time.perf_counter()
        generator: Generator | None = app.state.generator
        if generator is None:
            raise HTTPException(status_code=503, detail=NOT_CONFIGURED)
        slots: Slots = app.state.answer_slots
        if not slots.take():
            raise HTTPException(
                status_code=503,
                detail=BUSY,
                headers={"Retry-After": str(BUSY_RETRY_AFTER_SECONDS)},
            )
        retriever: Retriever = app.state.retriever

        def write(on_text: OnText, on_trace: OnTrace) -> Answer:
            return answer_question(
                request.query,
                retriever=retriever,
                generator=generator,
                limit=settings.answer_context_chunks,
                section=request.section,
                on_text=on_text,
                on_trace=on_trace,
            )

        run = AnswerRun.start(write, slots)
        try:
            event = await run.next()
            if request.stream and not isinstance(event, Failed):
                # From here the stream owns the run, and stops it when it ends.
                return EventStream(
                    _events(run, event, started),
                    on_close=run.stop,
                    headers={**STREAM_HEADERS, **_trace_headers(run)},
                )
            while isinstance(event, Text):
                event = await run.next()
        except BaseException:
            # Cancelled while waiting, as when the server shuts down: nobody will read
            # the answer, so the model need not finish it.
            run.stop()
            raise
        if isinstance(event, Finished):
            body = _respond(event.answer, started=started, run=run)
            return Response(
                content=_json_line(body.model_dump(mode="json")),
                media_type="application/json",
                headers=_trace_headers(run),
            )
        refusal = _refusal(event.error)
        if refusal is None:
            # A fault in this service: raised again, so the server answers 500 and logs it.
            raise event.error
        detail, headers = refusal
        # The cause goes to the log, not to the caller: it can name files, settings and
        # internal addresses. The trace records the kind of error.
        logger.warning("answer failed, trace %s: %s", run.trace_id, event.error)
        raise HTTPException(
            status_code=503, detail=detail, headers={**headers, **_trace_headers(run)} or None
        ) from event.error

    if settings.api_demo:
        demo.add_demo_page(app)

    return app
