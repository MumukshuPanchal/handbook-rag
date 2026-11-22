from pathlib import Path
from typing import ClassVar

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag import tracing, wiring
from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore, ScoredChunk, VectorStoreError
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.wiring import RetrievalMode
from tests.support import (
    FakeEmbedder,
    FakeReranker,
    FakeSparseEncoder,
    count_words,
    make_chunk,
    write_snapshot,
)

runner = CliRunner()

LEAVE = "New parents get sixteen weeks of paid leave."
LAPTOP = "Expense a laptop up to one thousand dollars."


def build(mode: RetrievalMode = "hybrid", *, rerank: bool = False) -> Retriever:
    """A retriever over two chunks in an in-memory index, with stand-in models."""
    chunks = [make_chunk(LEAVE), make_chunk(LAPTOP, doc_id="finance/expenses.md")]
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("traced")
    index_chunks(chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    return wiring.make_retriever(
        Settings(), mode, rerank, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )


def paths(exporter: InMemorySpanExporter) -> list[str]:
    """Every recorded span as the names from the root of its trace down to it, in start order."""
    finished = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in finished}

    def path(span: ReadableSpan) -> str:
        names = [span.name]
        while span.parent is not None and span.parent.span_id in by_id:
            span = by_id[span.parent.span_id]
            names.append(span.name)
        return " > ".join(reversed(names))

    return [path(span) for span in sorted(finished, key=lambda span: span.start_time or 0)]


def named(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = [span for span in exporter.get_finished_spans() if span.name == name]
    return span


def attribute(span: ReadableSpan, name: str) -> object:
    assert span.attributes is not None
    return span.attributes[f"handbook_rag.{name}"]


def test_a_hybrid_search_is_one_trace_with_a_span_for_every_stage(
    spans: InMemorySpanExporter,
) -> None:
    build().retrieve("how many weeks of paid leave", limit=1)

    assert paths(spans) == [
        "retrieve.hybrid",
        "retrieve.hybrid > retrieve.dense",
        "retrieve.hybrid > retrieve.dense > embed.query",
        "retrieve.hybrid > retrieve.dense > vector_store.query",
        "retrieve.hybrid > retrieve.sparse",
        "retrieve.hybrid > retrieve.sparse > vector_store.query",
        "retrieve.hybrid > fuse",
    ]
    assert len({span.context.trace_id for span in spans.get_finished_spans()}) == 1


def test_spans_say_what_each_stage_returned(spans: InMemorySpanExporter) -> None:
    results = build().retrieve("how many weeks of paid leave", limit=2)

    whole = named(spans, "retrieve.hybrid")
    assert attribute(whole, "limit") == 2
    assert attribute(whole, "results.count") == 2
    assert attribute(whole, "results.chunk_ids") == tuple(r.chunk.chunk_id for r in results)
    assert attribute(whole, "results.doc_ids") == ("people/leave.md", "finance/expenses.md")
    assert attribute(whole, "results.scores") == tuple(round(r.score, 6) for r in results)
    fuse = named(spans, "fuse")
    from_dense = attribute(named(spans, "retrieve.dense"), "results.count")
    from_sparse = attribute(named(spans, "retrieve.sparse"), "results.count")
    assert attribute(fuse, "fuse.from_dense") == from_dense == 2
    assert attribute(fuse, "fuse.from_sparse") == from_sparse
    assert attribute(fuse, "fuse.in_both") == from_sparse, "every keyword match is a dense match"
    assert attribute(fuse, "rrf_k") == 60
    assert attribute(named(spans, "embed.query"), "model") == "fake-embedder"
    assert attribute(named(spans, "retrieve.sparse"), "query.terms") == 5
    stores = [span for span in spans.get_finished_spans() if span.name == "vector_store.query"]
    assert sorted(str(attribute(span, "vector")) for span in stores) == ["dense", "sparse"]
    for span in stores:
        assert span.attributes is not None
        assert span.attributes["db.system.name"] == "qdrant"
        assert span.attributes["db.collection.name"] == "traced"


class Fixed:
    """A retriever that always returns the same chunks."""

    def __init__(self, *texts: str) -> None:
        self._results = [ScoredChunk(make_chunk(text), 1.0) for text in texts]

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        return self._results[:limit]


def test_the_merge_span_counts_what_each_search_found_and_what_both_found(
    spans: InMemorySpanExporter,
) -> None:
    hybrid = HybridRetriever(Fixed("a", "b", "c"), Fixed("c", "d"), candidates=30, rrf_k=60)

    hybrid.retrieve("anything", limit=5)

    fuse = named(spans, "fuse")
    assert attribute(fuse, "fuse.from_dense") == 3
    assert attribute(fuse, "fuse.from_sparse") == 2
    assert attribute(fuse, "fuse.in_both") == 1


def test_the_question_itself_is_left_out_of_a_trace_unless_content_is_switched_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = "how many weeks of paid leave"

    build().retrieve(question, limit=1)
    recorded = [key for span in spans.get_finished_spans() for key in span.attributes or {}]
    values = [
        value for span in spans.get_finished_spans() for value in (span.attributes or {}).values()
    ]

    assert "handbook_rag.query.text" not in recorded
    assert question not in values
    assert attribute(named(spans, "retrieve.hybrid"), "query.chars") == len(question)

    spans.clear()
    monkeypatch.setattr(tracing, "_record_content", True)
    build().retrieve(question, limit=1)

    assert attribute(named(spans, "retrieve.hybrid"), "query.text") == question


def test_reranking_adds_its_own_span_around_the_model_call(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wiring, "make_reranker", lambda _settings: FakeReranker())

    results = build("dense", rerank=True).retrieve("how many weeks of paid leave", limit=1)

    assert paths(spans) == [
        "retrieve.reranked",
        "retrieve.reranked > retrieve.dense",
        "retrieve.reranked > retrieve.dense > embed.query",
        "retrieve.reranked > retrieve.dense > vector_store.query",
        "retrieve.reranked > rerank",
    ]
    rerank = named(spans, "rerank")
    assert attribute(rerank, "model") == "fake-reranker"
    assert attribute(rerank, "candidates") == 2
    assert attribute(named(spans, "retrieve.reranked"), "results.chunk_ids") == (
        results[0].chunk.chunk_id,
    )


def test_a_failing_stage_is_marked_as_failed_and_so_is_every_stage_around_it(
    spans: InMemorySpanExporter,
) -> None:
    never_indexed = wiring.make_memory_store("missing")
    retriever = wiring.make_retriever(
        Settings(), "dense", store=never_indexed, embedder=FakeEmbedder()
    )

    with pytest.raises(VectorStoreError, match="does not exist yet"):
        retriever.retrieve("anything", limit=3)

    failed = named(spans, "vector_store.query")
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.status.description == "handbook_rag.index.store.VectorStoreError"
    assert failed.kind is SpanKind.CLIENT, "a call to another system"
    (event,) = failed.events
    assert event.name == "exception"
    assert dict(event.attributes or {}) == {
        "exception.type": "handbook_rag.index.store.VectorStoreError"
    }, "the kind of error only: no message and no stack trace"
    assert named(spans, "retrieve.dense").status.status_code is StatusCode.ERROR
    assert named(spans, "embed.query").status.status_code is not StatusCode.ERROR


def test_an_error_message_reaches_a_trace_only_when_content_is_switched_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(question: str) -> None:
        with pytest.raises(ValueError, match="cannot embed"), tracing.stage("embed.query"):
            raise ValueError(f"cannot embed {question!r}")

    fail("my salary is 90,000")
    (quiet,) = named(spans, "embed.query").events
    spans.clear()
    monkeypatch.setattr(tracing, "_record_content", True)
    fail("my salary is 90,000")
    (full,) = named(spans, "embed.query").events

    assert "salary" not in str(dict(quiet.attributes or {}))
    assert (quiet.attributes or {})["exception.type"] == "builtins.ValueError"
    assert "salary" in str((full.attributes or {})["exception.message"])
    assert "exception.stacktrace" in (full.attributes or {})


def test_a_question_that_cannot_be_searched_still_leaves_a_span(
    spans: InMemorySpanExporter,
) -> None:
    assert build("sparse").retrieve("the of is", limit=3) == []
    assert build("dense").retrieve("   ", limit=3) == []

    only_stop_words = named(spans, "retrieve.sparse")
    assert attribute(only_stop_words, "query.terms") == 0
    assert attribute(only_stop_words, "results.count") == 0
    assert attribute(named(spans, "retrieve.dense"), "results.count") == 0
    assert paths(spans) == ["retrieve.sparse", "retrieve.dense"], "no model or store was called"


def test_nothing_is_recorded_until_tracing_is_configured() -> None:
    assert tracing.configure_tracing(Settings()) is None, "tracing is off by default"

    with tracing.stage("search") as span:
        results = build().retrieve("how many weeks of paid leave", limit=1)
        # Checked while the span is open: a span that has ended reports the same.
        assert span.is_recording() is False
        assert tracing.trace_id(span) is None

    assert len(results) == 1


def test_our_spans_never_use_the_process_wide_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a session, OpenTelemetry's own default still points at that session's provider.

    If ``stage`` fell back to it, tracing would stay half on: spans recorded, trace ids
    printed, nothing sent.
    """
    left_behind = TracerProvider()
    left_behind.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: left_behind)
    monkeypatch.setattr(tracing, "_provider", None)

    with tracing.stage("search") as span:
        assert span.is_recording() is False
        assert tracing.trace_id(span) is None


class RecordingExporter(InMemorySpanExporter):
    """Stands in for the exporter that sends spans over the network."""

    created: ClassVar[list["RecordingExporter"]] = []

    def __init__(self, endpoint: str, timeout: float) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.timeout = timeout
        RecordingExporter.created.append(self)


@pytest.fixture
def exporters(monkeypatch: pytest.MonkeyPatch) -> list[RecordingExporter]:
    """Tracing switched on, with spans kept in memory instead of sent anywhere."""
    RecordingExporter.created = []
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", RecordingExporter
    )
    # The process-wide default can be set only once, so tests leave it alone.
    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", lambda _provider: None)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_record_content", False)
    return RecordingExporter.created


def test_configured_tracing_sends_spans_to_the_endpoint_in_the_settings(
    exporters: list[RecordingExporter],
) -> None:
    settings = Settings(
        tracing=True, otlp_endpoint="http://collector:4318/", service_name="assistant"
    )

    provider = tracing.configure_tracing(settings)
    assert isinstance(provider, TracerProvider)
    with tracing.stage("search", mode="hybrid") as span:
        recorded_id = tracing.trace_id(span)
    provider.shutdown()

    (exporter,) = exporters
    assert exporter.endpoint == "http://collector:4318/v1/traces"
    assert 0 < exporter.timeout <= 5, "a missing backend must not hold a command up for long"
    (sent,) = exporter.get_finished_spans()
    assert sent.name == "search"
    assert sent.resource.attributes["service.name"] == "assistant"
    assert recorded_id == format(sent.context.trace_id, "032x")
    assert attribute(sent, "mode") == "hybrid"


def test_a_tracing_session_sends_what_was_recorded_and_then_switches_tracing_off(
    exporters: list[RecordingExporter],
) -> None:
    settings = Settings(tracing=True, trace_content=True)

    with tracing.tracing_session(settings), tracing.stage("search") as span:
        tracing.record_query(span, "a question")

    (exporter,) = exporters
    (sent,) = exporter.get_finished_spans()
    assert attribute(sent, "query.text") == "a question"
    with tracing.stage("afterwards") as later:
        assert later.is_recording() is False
    assert tracing._record_content is False


def test_a_session_that_ends_in_an_error_still_sends_its_spans_and_switches_off(
    exporters: list[RecordingExporter],
) -> None:
    session = tracing.tracing_session(Settings(tracing=True))

    with pytest.raises(RuntimeError, match="boom"), session, tracing.stage("search"):
        raise RuntimeError("boom")

    (exporter,) = exporters
    (sent,) = exporter.get_finished_spans()
    assert sent.status.status_code is StatusCode.ERROR
    assert tracing._provider is None
    with tracing.stage("afterwards") as later:
        assert later.is_recording() is False


def test_a_session_with_tracing_off_does_nothing(exporters: list[RecordingExporter]) -> None:
    with tracing.tracing_session(Settings()), tracing.stage("search") as span:
        assert span.is_recording() is False

    assert exporters == []


@pytest.fixture
def indexed_handbook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A two-page handbook, indexed in memory with stand-in models, for the command line."""
    commit = "a" * 40
    pages = {"people/leave.md": f"## Parental leave\n\n{LEAVE}", "finance/expenses.md": LAPTOP}
    write_snapshot(tmp_path / "data", commit, pages)
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)
    client = QdrantClient(":memory:")
    monkeypatch.setattr(wiring, "make_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
    monkeypatch.setattr(
        wiring, "make_store", lambda settings: QdrantVectorStore(client, settings.collection_name)
    )
    assert runner.invoke(app, ["index"]).exit_code == 0


@pytest.mark.usefixtures("indexed_handbook")
def test_a_traced_search_prints_the_id_of_its_trace(
    exporters: list[RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_TRACING", "true")
    get_settings.cache_clear()

    result = runner.invoke(app, ["search", "how many weeks of paid leave", "--limit", "1"])

    assert result.exit_code == 0, result.output
    (exporter,) = exporters
    sent = exporter.get_finished_spans()
    (root,) = [span for span in sent if span.parent is None]
    assert root.name == "search"
    assert attribute(root, "mode") == "hybrid"
    assert attribute(root, "results.count") == 1
    assert f"trace {format(root.context.trace_id, '032x')}" in result.output
    assert {span.context.trace_id for span in sent} == {root.context.trace_id}
    assert "retrieve.hybrid" in {span.name for span in sent}
    # Content is off unless the settings switch it on, so the question is not in the trace.
    recorded = {key: value for span in sent for key, value in (span.attributes or {}).items()}
    assert "handbook_rag.query.text" not in recorded
    assert "how many weeks of paid leave" not in recorded.values()


@pytest.mark.usefixtures("indexed_handbook")
def test_a_search_is_not_traced_unless_tracing_is_switched_on(
    exporters: list[RecordingExporter],
) -> None:
    result = runner.invoke(app, ["search", "how many weeks of paid leave"])

    assert result.exit_code == 0, result.output
    assert "trace" not in result.output
    assert exporters == []


@pytest.mark.usefixtures("indexed_handbook")
def test_a_failed_search_is_traced_as_failed_and_still_reports_its_trace(
    exporters: list[RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_TRACING", "true")
    # Other chunk sizes mean another collection, and that one was never built.
    monkeypatch.setenv("HANDBOOK_RAG_CHUNK_MAX_TOKENS", "200")
    get_settings.cache_clear()

    result = runner.invoke(app, ["search", "how many weeks of paid leave"])

    assert result.exit_code == 1
    assert "does not exist yet" in result.output
    (exporter,) = exporters
    (root,) = [span for span in exporter.get_finished_spans() if span.parent is None]
    assert root.status.status_code is StatusCode.ERROR
    assert f"trace {format(root.context.trace_id, '032x')}" in result.output
