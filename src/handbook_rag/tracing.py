"""Tracing: one trace per request, one span per stage.

A trace is the record of one request. It is made of spans: named, timed steps that nest
inside each other, so a trace reads like a call tree with a duration on every line. When
a search is slow or returns the wrong passage, the trace shows which stage is
responsible.

The pipeline is instrumented with OpenTelemetry, the vendor-neutral standard (ADR 0008).
Each stage opens its span with :func:`stage`. Until :func:`configure_tracing` is called,
OpenTelemetry hands out spans that record nothing, so the instrumentation needs no
backend and costs next to nothing in tests, evaluation runs and indexing.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from handbook_rag import __version__
from handbook_rag.config import Settings

if TYPE_CHECKING:
    # The vector store opens spans itself, so it cannot be imported here at run time.
    # An answer holds chunks from the store, so the same goes for it, and for the cache.
    from handbook_rag.cache.answers import Lookup
    from handbook_rag.generate.models import Answer
    from handbook_rag.index.store import ScoredChunk

TRACER_NAME = "handbook_rag"
ATTRIBUTE_PREFIX = "handbook_rag"

# Names from OpenTelemetry's semantic conventions for generative AI, for the span of a
# request to a language model. A backend built for language-model applications knows
# them. They are written out here instead of imported, because the package that defines
# them keeps them in a private module that is still changing. tests/test_tracing.py
# compares each with that package, so a renamed or retired name is noticed.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
# How long sending a batch of spans may take, retries included. When no backend is
# running, this is how long a command waits before it gives up on its trace.
EXPORT_TIMEOUT_SECONDS = 3.0

# Process-wide state, set once by ``configure_tracing``. Tests replace both.
_provider: TracerProvider | None = None
_record_content = False
# What ``stage`` uses while tracing is off. It hands out spans that record nothing.
_OFF = trace.NoOpTracerProvider()


def configure_tracing(
    settings: Settings, exporter: SpanExporter | None = None
) -> TracerProvider | None:
    """Start recording spans and sending them to the trace backend in the settings.

    Returns ``None`` and changes nothing when tracing is switched off. Otherwise returns
    the provider: call its ``shutdown()`` before the process exits, or the last spans
    are lost, because they are sent in batches in the background.
    """
    global _provider, _record_content
    if not settings.tracing:
        return None
    if exporter is None:
        # Imported here so that commands which never trace do not load the exporter.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(
            endpoint=f"{settings.otlp_endpoint.rstrip('/')}/v1/traces",
            timeout=EXPORT_TIMEOUT_SECONDS,
        )
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.service_name, "service.version": __version__}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    # Also OpenTelemetry's process-wide default, so that a library which traces itself
    # adds its spans to the same trace. Our own spans never rely on that default: see
    # ``stage``.
    trace.set_tracer_provider(provider)
    _provider = provider
    _record_content = settings.trace_content
    return provider


@contextmanager
def tracing_session(settings: Settings) -> Iterator[None]:
    """Trace what runs inside, and send the remaining spans on the way out."""
    global _provider, _record_content
    before = (_provider, _record_content)
    provider = configure_tracing(settings)
    try:
        yield
    finally:
        if provider is not None:
            provider.shutdown()
            _provider, _record_content = before


@contextmanager
def stage(
    name: str, *, kind: SpanKind = SpanKind.INTERNAL, **attributes: str | int | float | bool | None
) -> Iterator[Span]:
    """Open a span for one stage of the pipeline.

    Spans opened inside it become its children. An exception that leaves the block marks
    the span as failed, then carries on to the caller. Attribute names are given without
    the ``handbook_rag.`` prefix.

    What is kept of an exception depends on the content setting. An error message can
    quote the input that caused it, and a stack trace holds file paths from this machine.
    So by default only the kind of exception is recorded; with content switched on, its
    message and stack trace are recorded too.
    """
    # Never OpenTelemetry's process-wide default: once a session has ended, that default
    # still points at the session's provider, and tracing would stay half switched on.
    provider = _provider if _provider is not None else _OFF
    tracer = trace.get_tracer(TRACER_NAME, __version__, tracer_provider=provider)
    with tracer.start_as_current_span(
        name, kind=kind, record_exception=False, set_status_on_exception=False
    ) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"{ATTRIBUTE_PREFIX}.{key}", value)
        try:
            yield span
        except Exception as error:
            kind_of_error = f"{type(error).__module__}.{type(error).__qualname__}"
            span.set_status(Status(StatusCode.ERROR, kind_of_error))
            if _record_content:
                span.record_exception(error)
            else:
                span.add_event("exception", {"exception.type": kind_of_error})
            raise


def record_query(span: Span, query: str) -> None:
    """Note the question on a span: always its length, its text only if content is on.

    Questions can hold personal or confidential details, and traces are often kept
    longer and read by more people than the application's own data. So the default
    records what is needed to see that something was asked, and no more.
    """
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.query.chars", len(query))
    if _record_content:
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.query.text", query)


@contextmanager
def model_call(
    *,
    provider: str,
    operation: str,
    model: str,
    passages: int,
    attempt: int = 1,
    fallback: bool = False,
) -> Iterator[Span]:
    """Open the span of one request to a language model.

    The span is named and described as OpenTelemetry's conventions for generative AI
    say, so that a backend built for language-model applications can read it: its name
    is the operation and the model asked for, and it is a client span, because the model
    runs at the provider. What the conventions have no name for, such as how many
    passages the model was given, carries the project's prefix. A failed request marks
    the span as failed, as in any stage.

    An answer can take more than one request. ``attempt`` is this request's place among
    them, counted from 1 across both models, and ``fallback`` says whether it went to
    the fallback model.
    """
    with stage(
        f"{operation} {model}", kind=SpanKind.CLIENT, attempt=attempt, fallback=fallback
    ) as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
        span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.passages.given", passages)
        yield span


def record_failure(span: Span, reason: str) -> None:
    """Note why a model call failed, in the fixed words of a generation error's reason.

    A reason is one of a handful of words chosen in advance, such as "timeout" or
    "quota", never the error's own message. So it is always recorded, and a trace shows
    why an attempt was or was not followed by another.
    """
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.failure", reason)


def record_requests(span: Span, *, requests: int, fallback: bool) -> None:
    """Note on the span of an answer how many requests to a model it took, and whether
    the fallback model was asked. Recorded whether the answer was written or not."""
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.requests", requests)
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.fallback", fallback)


def record_reply(span: Span, answer: "Answer") -> None:
    """Note what a model call returned, without its text.

    The model that served the reply, the tokens, why it stopped, whether it counts as an
    answer, which passages it cites, and how long the first words took. These are names,
    ids and counts, so they are always recorded.
    """
    usage = answer.usage
    span.set_attribute(GEN_AI_RESPONSE_MODEL, answer.model)
    # A list in the conventions. The reason is in the words this application uses for
    # both providers, such as "end_turn" or "max_tokens".
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [answer.stop_reason])
    # Every token the model read, those served from the provider's cache included. The
    # application's own count leaves the cached ones out, because they are billed at
    # another price, so the cache's share is recorded on its own below.
    span.set_attribute(
        GEN_AI_USAGE_INPUT_TOKENS,
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens,
    )
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
    if usage.cache_read_input_tokens:
        span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, usage.cache_read_input_tokens)
    if usage.cache_creation_input_tokens:
        span.set_attribute(
            GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, usage.cache_creation_input_tokens
        )
    if answer.seconds_to_first_text is not None:
        # Measured to the first words of the answer. A model that reasons before it
        # answers sends other pieces first, so this can be later than its first piece.
        span.set_attribute(GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, answer.seconds_to_first_text)
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.answered", answer.answered)
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.passages.cited", len(answer.sources))
    # In the order the answer numbers its sources, so [1] in the answer is the first id.
    # With the search span's ids, this tells whether the right passage was cited.
    span.set_attribute(
        f"{ATTRIBUTE_PREFIX}.passages.cited_chunk_ids",
        [chunk.chunk_id for chunk in answer.sources],
    )


def record_answer(span: Span, answer: "Answer") -> None:
    """Note how a question ended: whether it was answered, and its text if content is on.

    The answer is recorded with its citation markers, so a trace shows which statement
    rests on which passage. Like a question, its text is left out by default: it can
    repeat personal or confidential details from the question.
    """
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.answered", answer.answered)
    if answer.checks:
        # Which checks of the reply in code fired, by name: never the words they took out.
        span.set_attribute(
            f"{ATTRIBUTE_PREFIX}.reply_checks", tuple(check.name for check in answer.checks)
        )
    if _record_content:
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.answer.text", answer.text_with_markers)


def record_cache_lookup(span: Span, lookup: "Lookup") -> None:
    """Note what the answer cache did with a question: the tier, hit or miss, the scores.

    Tiers, counts and scores are always recorded. With them a trace shows why a stored
    answer was served or was not: how near each candidate was, and what the verifier
    said of it in each direction. The stored questions themselves are questions that
    people asked, so they are recorded only if content is on.
    """
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.cache.read", lookup.read)
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.cache.hit", lookup.answer is not None)
    if lookup.tier is not None:
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.cache.tier", lookup.tier)
    if lookup.failure is not None:
        # The kind of error only, as for any failed stage.
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.cache.failure", lookup.failure)
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.cache.candidates", len(lookup.checked))
    if not lookup.checked:
        return
    # Nearest first, so the same place in each list is the same candidate.
    span.set_attribute(
        f"{ATTRIBUTE_PREFIX}.cache.similarities",
        [round(candidate.similarity, 6) for candidate in lookup.checked],
    )
    verified = [candidate.entails for candidate in lookup.checked if candidate.entails is not None]
    if len(verified) == len(lookup.checked):
        span.set_attribute(
            f"{ATTRIBUTE_PREFIX}.cache.entails_forward",
            [round(entails.forward, 6) for entails in verified],
        )
        span.set_attribute(
            f"{ATTRIBUTE_PREFIX}.cache.entails_backward",
            [round(entails.backward, 6) for entails in verified],
        )
    if _record_content:
        span.set_attribute(
            f"{ATTRIBUTE_PREFIX}.cache.questions",
            [candidate.question for candidate in lookup.checked],
        )


def trace_id(span: Span) -> str | None:
    """The id to look a trace up by in the backend, or ``None`` when nothing is recorded."""
    if not span.is_recording():
        return None
    return format(span.get_span_context().trace_id, "032x")


def record_results(span: Span, results: Sequence["ScoredChunk"]) -> None:
    """Note what a stage returned: which chunks, from which pages, with which scores.

    Ids and scores are enough to explain a ranking, and each id leads back to the chunk's
    text in the index. The text itself is not copied into the trace.
    """
    if not span.is_recording():
        return
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.results.count", len(results))
    span.set_attribute(
        f"{ATTRIBUTE_PREFIX}.results.chunk_ids", [result.chunk.chunk_id for result in results]
    )
    span.set_attribute(
        f"{ATTRIBUTE_PREFIX}.results.doc_ids", [result.chunk.doc_id for result in results]
    )
    span.set_attribute(
        f"{ATTRIBUTE_PREFIX}.results.scores", [round(result.score, 6) for result in results]
    )
