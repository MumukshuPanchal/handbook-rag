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
from opentelemetry.trace import Span

from handbook_rag import __version__
from handbook_rag.config import Settings

if TYPE_CHECKING:
    # The vector store opens spans itself, so it cannot be imported here at run time.
    from handbook_rag.index.store import ScoredChunk

TRACER_NAME = "handbook_rag"
ATTRIBUTE_PREFIX = "handbook_rag"
# How long sending a batch of spans may take, retries included. When no backend is
# running, this is how long a command waits before it gives up on its trace.
EXPORT_TIMEOUT_SECONDS = 3.0

# Process-wide state, set once by ``configure_tracing``. Tests replace both.
_provider: TracerProvider | None = None
_record_content = False


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
    # Also the process-wide default, so that libraries which trace themselves, such as
    # the model SDK, add their spans to the same trace.
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
def stage(name: str, **attributes: str | int | float | bool | None) -> Iterator[Span]:
    """Open a span for one stage of the pipeline.

    Spans opened inside it become its children. An exception that leaves the block is
    recorded on the span and marks it as failed, then carries on to the caller.
    Attribute names are given without the ``handbook_rag.`` prefix.
    """
    tracer = trace.get_tracer(TRACER_NAME, __version__, tracer_provider=_provider)
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(f"{ATTRIBUTE_PREFIX}.{key}", value)
        yield span


def record_query(span: Span, query: str) -> None:
    """Note the question on a span: always its length, its text only if content is on.

    Questions can hold personal or confidential details, and traces are often kept
    longer and read by more people than the application's own data. So the default
    records what is needed to see that something was asked, and no more.
    """
    span.set_attribute(f"{ATTRIBUTE_PREFIX}.query.chars", len(query))
    if _record_content:
        span.set_attribute(f"{ATTRIBUTE_PREFIX}.query.text", query)


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
