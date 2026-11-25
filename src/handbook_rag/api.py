"""The HTTP API: the assistant as a service that other programs can call.

Three endpoints: ``POST /ask`` answers a question with citations, as one JSON response
or as server-sent events while the model writes; ``POST /search`` returns the passages
most relevant to a question; and ``GET /health`` says whether the service can do its job.
Why ``/ask`` works the way it does is in ADR 0015. Unless it is switched off, the service
also serves a demo page at ``/`` that asks ``/ask`` (see ``demo.py`` and ADR 0016).

``/ask`` serves an answer stored earlier when the answer cache holds one for the
question, and says so in the answer (``cache/answers.py`` and ADR 0020).

``/ask`` takes the earlier questions of a conversation too. The service keeps no
conversation: the caller sends them with each question. A question with earlier
questions is first rewritten so that it stands alone, and every reply says which
question was searched for (``rewrite/``, ``pipeline.py`` and ADR 0024).

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
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.config import Settings, get_settings
from handbook_rag.generate.base import FailureReason, GenerationError, Generator
from handbook_rag.generate.models import Answer
from handbook_rag.index.store import (
    CollectionMissingError,
    QdrantVectorStore,
    ScoredChunk,
    VectorStore,
    VectorStoreError,
)
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import FollowUps, SearchedFor, answer_question
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.rewrite.models import Outcome
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
# How many earlier questions /ask accepts with a question. The service reads the latest
# few (HANDBOOK_RAG_FOLLOW_UP_EARLIER_QUESTIONS) and drops the rest, so a caller may send
# its whole conversation up to this bound.
MAX_EARLIER_QUESTIONS = 10
# The largest request body that is read. It leaves room for a question and every earlier
# question at their longest, written in characters of four bytes each, as some scripts are.
MAX_BODY_BYTES = 32_768
# How long one call to the vector store may take before the request gives up.
STORE_TIMEOUT_SECONDS = 10.0
# Searched once when the application starts, so that the models are loaded and have run.
WARM_UP_QUERY = "warm up"
# How long a service that shuts down waits for answers that are still being stored.
CACHE_CLOSE_SECONDS = 5.0
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
MODEL_TIMED_OUT = "The answer model did not answer in time. Try again later."
NOT_FINISHED = "The answer could not be finished."

logger = logging.getLogger(__name__)


class CacheStartError(RuntimeError):
    """The answer cache's second model could not be loaded when the service started."""


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
    earlier: list[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS),
        ]
    ] = Field(
        default_factory=list,
        max_length=MAX_EARLIER_QUESTIONS,
        description=(
            "The earlier questions of the conversation, oldest first. With at least one, "
            "the question is first rewritten so that it stands alone, with one more "
            "request to the answer model, and the rewritten question is the one searched "
            "for and answered. The latest few are read and the oldest dropped. Send each "
            "earlier question as it was searched for, the `searched_for.question` of its "
            "reply, so that a further follow-up builds on questions that already stand "
            "alone. Questions only: each item is a string, and an answer, or an object "
            "with one, is refused. The service keeps no conversation of its own."
        ),
    )
    cache: bool = Field(
        default=True,
        # Strict: "false" and 0 are mistakes by the caller, not no.
        strict=True,
        description=(
            "False asks the answer model, also when the service holds a stored answer "
            "for the question. The new answer takes the stored one's place if it is one "
            "the cache keeps. A reply that was cut off, for one, is not kept, and the "
            "stored answer then stays. It has no effect on a service whose cache is "
            "switched off."
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


class CheckNote(BaseModel):
    """A check of the reply in code that fired, and what it did to the reply."""

    check: Literal["address_in_closing_line", "cites_nothing", "mostly_a_copy"] = Field(
        description=(
            "Which check: a closing line that cites nothing held a web or email address; "
            "the reply cited no passage; most of the reply repeats its passages word for word."
        )
    )
    action: Literal["dropped_line", "declined", "marked"] = Field(
        description=(
            "What it did: left that closing line out, declined in place of the reply, or "
            "left the reply as it was and marked it."
        )
    )
    note: str = Field(
        description="What to tell the reader, in fixed words. Show it with the answer."
    )


class Usage(BaseModel):
    """The tokens of the request to the model."""

    input_tokens: int = Field(
        description="Tokens the model read, not counting any served from the provider's cache."
    )
    output_tokens: int = Field(description="Tokens the model wrote, its reasoning included.")


class CachedAnswer(BaseModel):
    """Where an answer came from that was not written for this request."""

    tier: Literal["exact", "semantic"] = Field(
        description=(
            "exact: the answer was stored for this very question, where case, spacing and "
            "the punctuation at its end do not count. semantic: it was stored for a "
            "question that a second model judged to mean the same. That judgement can be "
            "wrong, so compare `question` with what you asked."
        )
    )
    question: str = Field(description="The stored question the answer was written for.")
    similarity: float | None = Field(
        description=(
            "The cosine of the two questions' embeddings, for a semantic match. Null for "
            "an exact one."
        )
    )
    score: float | None = Field(
        description=(
            "How far each question implies the other by the second model, in its weaker "
            "direction, from 0 to 1. It reached the service's threshold. Null for an "
            "exact match."
        )
    )
    stored_at: str = Field(description="When the answer was stored, in UTC (ISO 8601).")


class SearchedQuestion(BaseModel):
    """The question that was searched for and answered, and how it came to be."""

    asked: str = Field(description="The question as it was sent.")
    question: str = Field(
        description=(
            "The question that was looked up in the cache, searched for and answered. "
            "Send it as an earlier question with the next follow-up."
        )
    )
    outcome: Outcome = Field(
        description=(
            "How the rewriting step came to the question. not_asked: no earlier question "
            "was sent, every earlier question sent held no letter or digit, or follow-ups "
            "are switched off, so no model was asked. unchanged: "
            "the model gave the question back. rewritten: the model gave another question, "
            "which passed the step's checks. unusable: the model's reply did not pass "
            "them. failed: the request failed or ran out of time. After unusable and "
            "failed the question is the one asked, searched for as it was typed: a "
            "follow-up then finds little."
        )
    )
    failure: FailureReason | None = Field(
        description="Why the request failed, in one fixed word, after failed. Null otherwise."
    )
    earlier: int = Field(
        description="How many earlier questions the step read. 0 when it did not run."
    )
    requests: int = Field(
        description="How many requests to a model the step made: 0 or 1. Not counted in `requests`."
    )
    usage: Usage
    took_ms: float = Field(description="How long the step took, in milliseconds.")


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
    checks: list[CheckNote] = Field(
        description=(
            "The checks of the reply in code that fired, in the order they were made. The "
            "answer is the reply as they left it, so show each note with it. Empty when "
            "none fired."
        )
    )
    model: str = Field(description="The model that wrote the reply.")
    stop_reason: str = Field(
        description="Why the model stopped: end_turn, max_tokens, refusal, no_context or another."
    )
    usage: Usage
    requests: int = Field(
        description=(
            "How many requests to a model the answer took, failed ones included. 0 when "
            "nothing was found to answer from, so that no model was asked, and 0 when "
            "the answer came from the cache. The rewriting step's request, when there was "
            "one, is counted in `searched_for.requests` and not here."
        )
    )
    fallback: bool = Field(
        description="Whether the fallback model wrote the reply, because the answer model gave up."
    )
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
    cached: CachedAnswer | None = Field(
        description=(
            "Null when the model wrote the answer for this request. Otherwise the answer "
            "is one stored earlier: `model` and `stop_reason` are those of the stored "
            "answer, `usage` is zero and `requests` is 0, because no model was asked."
        )
    )
    searched_for: SearchedQuestion = Field(
        description=(
            "The question that was searched for and answered. It differs from the one "
            "asked when a follow-up was rewritten."
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
    'stream ends with one `event: error` with `data: {"detail": "..."}` instead. An '
    "answer that comes from the cache is one `event: text` with the whole text, then "
    "the `event: answer`. Before the first `event: text`, one `event: searched` "
    "carries the question searched for, as the answer's `searched_for` field."
)
ASK_STREAM_EXAMPLE = (
    "event: searched\n"
    'data: {"asked":"How many weeks of parental leave do new parents get?",'
    '"question":"How many weeks of parental leave do new parents get?","outcome":"not_asked",'
    '"failure":null,"earlier":0,"requests":0,"usage":{"input_tokens":0,"output_tokens":0},'
    '"took_ms":0.0}\n'
    "\n"
    "event: text\n"
    'data: {"text":"New parents get sixteen weeks of paid leave."}\n'
    "\n"
    "event: answer\n"
    'data: {"answer":"New parents get sixteen weeks of paid leave.[1]","answered":true,'
    '"segments":[{"text":"New parents get sixteen weeks of paid leave.","sources":[1]}],'
    '"sources":[{"number":1,"doc_id":"people/leave.md","title":"Leave",'
    '"url":"https://handbook.example.com/handbook/people/leave/","section":"people",'
    '"heading_path":["Parental leave"],"text":"New parents get sixteen weeks of paid leave."}],'
    '"checks":[],"model":"gemini-3.5-flash-lite","stop_reason":"end_turn",'
    '"usage":{"input_tokens":1200,"output_tokens":20},"requests":1,"fallback":false,'
    '"took_ms":1500.0,"first_text_ms":1300.0,"trace_id":null,"cached":null,'
    '"searched_for":{"asked":"How many weeks of parental leave do new parents get?",'
    '"question":"How many weeks of parental leave do new parents get?","outcome":"not_asked",'
    '"failure":null,"earlier":0,"requests":0,"usage":{"input_tokens":0,"output_tokens":0},'
    '"took_ms":0.0}}\n'
    "\n"
)
ASK_UNAVAILABLE = (
    "No answer could be started, or none could be finished before anything was sent: the "
    "vector store or the answer model is unavailable, the model did not answer in time, "
    "the model's limit was reached, answering is not configured, or the service is "
    "writing as many answers as it can. The body says which, in fixed words. This is also "
    "the response when `stream` is true and the failure came before the first text."
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


def _searched_question(searched: SearchedFor) -> SearchedQuestion:
    """What a caller is told about the question searched for. The reason a reply was not
    usable is not sent: it can quote the model's reply, and the outcome says enough."""
    rewrite = searched.rewrite
    usage = rewrite.reply.usage if rewrite.reply is not None else None
    return SearchedQuestion(
        asked=searched.asked,
        question=searched.question,
        outcome=rewrite.outcome,
        failure=rewrite.failure.reason if rewrite.failure is not None else None,
        earlier=searched.earlier,
        requests=rewrite.requests,
        usage=Usage(
            input_tokens=usage.input_tokens if usage is not None else 0,
            output_tokens=usage.output_tokens if usage is not None else 0,
        ),
        took_ms=_milliseconds(rewrite.seconds),
    )


def _respond(
    answer: Answer, *, started: float, run: AnswerRun, searched: SearchedFor
) -> AskResponse:
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
        checks=[
            CheckNote.model_validate(
                {"check": check.name, "action": check.action, "note": check.note}
            )
            for check in answer.checks
        ],
        model=answer.model,
        stop_reason=answer.stop_reason,
        usage=Usage(
            input_tokens=answer.usage.input_tokens, output_tokens=answer.usage.output_tokens
        ),
        requests=answer.requests,
        fallback=answer.fallback,
        took_ms=_milliseconds(time.perf_counter() - started),
        first_text_ms=first_text,
        trace_id=run.trace_id,
        cached=None
        if answer.cached is None
        else CachedAnswer(
            tier=answer.cached.tier,
            question=answer.cached.question,
            similarity=answer.cached.similarity,
            score=answer.cached.score,
            stored_at=answer.cached.stored_at,
        ),
        searched_for=_searched_question(searched),
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
    if error.reason == "timeout":
        return MODEL_TIMED_OUT, {}
    return MODEL_UNAVAILABLE, {}


def _for_the_log(error: BaseException) -> str:
    """What the service's log says about a failure: its message, and, when the fallback
    model failed too, why the answer model had given up before it."""
    if isinstance(error, GenerationError) and error.before_fallback is not None:
        return f"{error} (before the fallback, the answer model failed: {error.before_fallback})"
    return str(error)


def _log_a_failed_rewrite(searched: list[SearchedFor], run: AnswerRun) -> None:
    """Log the rewriting step's failure, as the failure of an answer is logged.

    The reply carries one fixed word for it, and a follow-up is then searched for as it
    was typed. Without this, a step that is set up wrong, with a key the provider refuses
    or a model it does not know, would turn every follow-up into one searched for as
    typed and leave no trace of why. The question is not logged, as no question is.
    """
    if not searched or searched[0].rewrite.failure is None:
        return
    failure = searched[0].rewrite.failure
    logger.warning(
        "a follow-up was searched for as typed, because rewriting it failed (%s), trace %s: %s",
        failure.reason,
        run.trace_id,
        _for_the_log(failure),
    )


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


async def _events(
    run: AnswerRun, first: Text | Finished, started: float, searched: SearchedFor
) -> AsyncIterator[bytes]:
    """The events of a stream that has begun: the question searched for, text, then the
    answer or an error.

    The question searched for is known before the search, and goes first, so that a page
    can show it while the answer is written. It is sent once the first text or the answer
    is there and not sooner: until then a failure is still an HTTP error.
    """
    yield server_sent_event(
        "searched", _json_line(_searched_question(searched).model_dump(mode="json"))
    )
    event: Text | Finished | Failed = first
    while isinstance(event, Text):
        yield server_sent_event("text", _json_line({"text": event.text}))
        event = await run.next()
    if isinstance(event, Finished):
        body = _respond(event.answer, started=started, run=run, searched=searched)
        yield server_sent_event("answer", _json_line(body.model_dump(mode="json")))
        return
    refusal = _refusal(event.error)
    if refusal is None:
        logger.error("answer failed after it began, trace %s", run.trace_id, exc_info=event.error)
        detail = NOT_FINISHED
    else:
        logger.warning(
            "answer failed after it began, trace %s: %s", run.trace_id, _for_the_log(event.error)
        )
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


def _follow_ups(settings: Settings) -> FollowUps | None:
    """The rewriting step of the service, or ``None`` when follow-ups are off or cannot be
    built. Without it a question with earlier questions is searched for as typed, and the
    reply says that the step did not run."""
    try:
        return wiring.make_follow_ups(settings)
    except GenerationError as error:
        logger.warning("follow-up questions are not rewritten: %s", error)
        return None


def _warm_up(cache: AnswerCache, settings: Settings) -> None:
    """Run the cache's models once, or say which settings to change.

    Only the second model can fail this: a failure of the embedding model here is logged
    by the cache and no more. A service that cannot load a model it is set to use does
    not start. The error names the settings, because the model's own error names none.
    """
    try:
        cache.warm_up()
    except Exception as error:
        raise CacheStartError(
            f"The answer cache's second model, {settings.cache_verifier_model}, could not be "
            f"loaded: {type(error).__name__}: {error}. HANDBOOK_RAG_CACHE_MODE=semantic "
            "needs it. Its files are downloaded the first time, so check the network, and "
            "the model's name in HANDBOOK_RAG_CACHE_VERIFIER_MODEL. To start without it, "
            "set HANDBOOK_RAG_CACHE_MODE to exact or to off."
        ) from error


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
    cache: AnswerCache | None = None,
    follow_ups: FollowUps | None = None,
) -> FastAPI:
    """Build the application.

    Whatever is not passed is built from the settings when the application starts.
    Tests pass a ``retriever``, a ``store`` and a ``generator`` made of stand-in parts.

    The rewriting step uses the answer model, so a caller that passes a ``generator``
    passes the ``follow_ups`` that go with it, or has none. Otherwise it is built from the
    settings, when there is an answer model and follow-ups are on.

    The answer cache is built together with the retriever, because it embeds a question
    with the retriever's own model and keeps its answers in the retriever's store. So a
    caller that passes a ``retriever`` passes the ``cache`` that goes with it, or has
    none. A service without an answer model builds none either: it has nothing to keep.
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
        # Building the answer model sends no request. So a missing key is found out here,
        # and a wrong one only by the first question.
        app.state.generator = generator if generator is not None else _answer_model(settings)
        if generator is not None:
            app.state.follow_ups = follow_ups
        else:
            app.state.follow_ups = (
                _follow_ups(settings) if app.state.generator is not None else None
            )
        held = app.state.store
        if retriever is not None:
            app.state.retriever, app.state.cache = retriever, cache
        elif (
            cache is None
            and app.state.generator is not None
            and isinstance(held, QdrantVectorStore)
        ):
            # Only a service that can answer has a use for stored answers.
            app.state.retriever, app.state.cache = wiring.make_retriever_and_cache(serving, held)
        else:
            app.state.retriever = wiring.make_retriever(serving, store=held)
            app.state.cache = cache
        with suppress(VectorStoreError), stage("warm_up"):
            app.state.retriever.retrieve(WARM_UP_QUERY, limit=1)
        if app.state.cache is not None:
            # The collection the answers are kept in is made now, before any question.
            # Made by the first answer that is stored, it would be made while questions
            # are already being looked up. A store that is not there is logged, and the
            # service starts without it, as it does for its index.
            app.state.cache.prepare()
            # In the mode `semantic` this runs both of the cache's models once: it embeds a
            # question, and it loads the verifier and asks it. The first time a process
            # runs a model is slow, and no question waits for that.
            with stage("cache.warm_up", mode=app.state.cache.mode):
                _warm_up(app.state.cache, settings)
        app.state.answer_slots = Slots(settings.api_max_answers_in_flight)
        try:
            yield
        finally:
            if app.state.cache is not None:
                app.state.cache.close(CACHE_CLOSE_SECONDS)
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
        answers: AnswerCache | None = app.state.cache
        rewriting: FollowUps | None = app.state.follow_ups
        # Set by the worker before the search, and so before any text or answer is read.
        searched: list[SearchedFor] = []

        def write(on_text: OnText, on_trace: OnTrace) -> Answer:
            return answer_question(
                request.query,
                retriever=retriever,
                generator=generator,
                limit=settings.answer_context_chunks,
                section=request.section,
                on_text=on_text,
                on_trace=on_trace,
                cache=answers,
                fresh=not request.cache,
                earlier=request.earlier,
                follow_ups=rewriting,
                on_searched=searched.append,
            )

        run = AnswerRun.start(write, slots)
        try:
            event = await run.next()
            # By the first event the step has run, if it ran at all.
            _log_a_failed_rewrite(searched, run)
            if request.stream and not isinstance(event, Failed):
                # From here the stream owns the run, and stops it when it ends.
                return EventStream(
                    _events(run, event, started, searched[0]),
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
            body = _respond(event.answer, started=started, run=run, searched=searched[0])
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
        logger.warning("answer failed, trace %s: %s", run.trace_id, _for_the_log(event.error))
        raise HTTPException(
            status_code=503, detail=detail, headers={**headers, **_trace_headers(run)} or None
        ) from event.error

    if settings.api_demo:
        demo.add_demo_page(app)

    return app
