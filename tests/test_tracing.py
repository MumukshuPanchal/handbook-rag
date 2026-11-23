import ast
import inspect
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.trace import SpanKind, StatusCode
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag import tracing, wiring
from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore, ScoredChunk, VectorStoreError
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import answer_question
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.wiring import RetrievalMode
from tests.support import (
    FakeEmbedder,
    FakeReranker,
    FakeSparseEncoder,
    StandInModel,
    count_words,
    everything_recorded,
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
    assert failed.status.description == "handbook_rag.index.store.CollectionMissingError"
    assert failed.kind is SpanKind.CLIENT, "a call to another system"
    (event,) = failed.events
    assert event.name == "exception"
    assert dict(event.attributes or {}) == {
        "exception.type": "handbook_rag.index.store.CollectionMissingError"
    }, "the kind of error only: no message and no stack trace"
    # What went wrong underneath is kept as well, again by kind only.
    assert (failed.attributes or {})["error.type"] == "builtins.ValueError"
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


# The span of a model call is named by OpenTelemetry's conventions for generative AI.
# The names are written out in our code, so these tests compare them with the package
# that defines the conventions: a name it renames or retires makes them fail.


def notes_on_names(body: Sequence[ast.stmt]) -> dict[str, str]:
    """Each name a block of the conventions' source defines, with the note written under it.

    Python does not keep a note under a constant, so it is read from the source. The
    note says when a name is deprecated, and what replaced it.
    """
    notes: dict[str, str] = {}
    for statement, following in zip(body, [*body[1:], None], strict=True):
        value = statement.value if isinstance(statement, ast.AnnAssign | ast.Assign) else None
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        note = ""
        if isinstance(following, ast.Expr) and isinstance(following.value, ast.Constant):
            note = str(following.value.value)
        notes[value.value] = note
    return notes


def retired(note: str) -> bool:
    """Whether a note says that a name was replaced by another or removed.

    "Moved" is not retirement: the package says that of every name in this module,
    because the conventions for generative AI are now kept in a repository of their own.
    """
    return "Replaced by" in note or "Removed" in note


CONVENTIONS = ast.parse(inspect.getsource(gen_ai))


def notes_on_values(enum: str) -> dict[str, str]:
    """The values one list of allowed values in the conventions defines, with their notes."""
    (body,) = [
        node.body
        for node in CONVENTIONS.body
        if isinstance(node, ast.ClassDef) and node.name == enum
    ]
    return notes_on_names(body)


def test_retired_names_are_recognised_as_retired() -> None:
    """The check below is only worth something if it can fail."""
    notes = notes_on_names(CONVENTIONS.body)

    assert retired(notes["gen_ai.system"]), "replaced by gen_ai.provider.name"
    assert retired(notes["gen_ai.usage.prompt_tokens"]), "replaced by gen_ai.usage.input_tokens"
    assert retired(notes["gen_ai.prompt"]), "removed"
    assert retired(notes_on_values("GenAiSystemValues")["gemini"]), "replaced by gcp.gemini"
    assert not retired(notes["gen_ai.provider.name"])


def test_our_names_for_a_model_call_are_the_current_names_of_the_installed_conventions() -> None:
    ours = {name: value for name, value in vars(tracing).items() if name.startswith("GEN_AI_")}
    notes = notes_on_names(CONVENTIONS.body)

    assert len(ours) == 10
    for name, value in ours.items():
        assert getattr(gen_ai, name) == value, f"{name} is spelt differently in the conventions"
        assert not retired(notes[value]), f"{value}: {notes[value]}"


@pytest.mark.parametrize(
    ("generator", "provider", "operation"),
    [
        (GeminiGenerator, "GCP_GEMINI", "GENERATE_CONTENT"),
        (ClaudeGenerator, "ANTHROPIC", "CHAT"),
    ],
)
def test_each_provider_and_operation_is_named_as_the_installed_conventions_name_it(
    generator: type[GeminiGenerator | ClaudeGenerator], provider: str, operation: str
) -> None:
    providers = notes_on_values("GenAiProviderNameValues")
    operations = notes_on_values("GenAiOperationNameValues")

    assert generator.provider == getattr(gen_ai.GenAiProviderNameValues, provider).value
    assert generator.operation == getattr(gen_ai.GenAiOperationNameValues, operation).value
    assert not retired(providers[generator.provider])
    assert not retired(operations[generator.operation])


def test_every_conventional_name_a_model_call_records_is_current(
    spans: InMemorySpanExporter,
) -> None:
    """Also catches a name typed into the code without going through the constants."""

    def everything(question: str, chunks: Sequence[Chunk]) -> Answer:
        return Answer(
            question=question,
            segments=(Segment("Sixteen weeks.", (Citation(chunks[0]),)),),
            answered=True,
            model="stand-in-model",
            stop_reason="end_turn",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=3,
                cache_creation_input_tokens=2,
            ),
            seconds_to_first_text=0.5,
        )

    answer_question(
        "how long is parental leave",
        retriever=Fixed(LEAVE),
        generator=Scripted(everything),
        limit=1,
    )

    notes = notes_on_names(CONVENTIONS.body)
    recorded = {
        key
        for span in spans.get_finished_spans()
        for key in span.attributes or {}
        if key.startswith("gen_ai.")
    }
    assert len(recorded) == 10, "every name the code knows, in one span"
    for key in recorded:
        assert key in notes, f"{key} is not a name of the conventions"
        assert not retired(notes[key]), f"{key}: {notes[key]}"


# The command line traces an answer as it traces a search.


class Scripted(StandInModel):
    """An answer model that writes what it is told to, and notes whether a trace was open."""

    def __init__(
        self,
        replies: Callable[[str, Sequence[Chunk]], Answer] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._replies = replies
        self._error = error

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        if self._error is not None:
            raise self._error
        if self._replies is not None:
            return self._replies(question, chunks)
        if on_text is not None:
            on_text("Sixteen weeks of paid leave.")
        return Answer(
            question=question,
            segments=(Segment("Sixteen weeks of paid leave.", (Citation(chunks[0]),)),),
            answered=True,
            model="stand-in-model",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1200, output_tokens=40),
        )


def build_outside_a_trace(model: Scripted, built: list[bool]) -> Callable[[Settings], Scripted]:
    """A stand-in for building the answer model that notes whether a span was open then."""

    def build(_settings: Settings) -> Scripted:
        built.append(trace.get_current_span().is_recording())
        return model

    return build


@pytest.mark.usefixtures("indexed_handbook")
def test_a_traced_answer_prints_the_id_of_its_trace_at_the_end(
    exporters: list[RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_TRACING", "true")
    get_settings.cache_clear()
    built: list[bool] = []
    monkeypatch.setattr(wiring, "make_generator", build_outside_a_trace(Scripted(), built))

    result = runner.invoke(app, ["ask", "how many weeks of paid leave"])

    assert result.exit_code == 0, result.output
    (exporter,) = exporters
    sent = exporter.get_finished_spans()
    (root,) = [span for span in sent if span.parent is None]
    assert root.name == "answer"
    assert {span.context.trace_id for span in sent} == {root.context.trace_id}
    assert {"search", "retrieve.hybrid", "chat stand-in-model"} <= {span.name for span in sent}
    assert result.output.splitlines()[-1] == f"trace {format(root.context.trace_id, '032x')}"
    assert built == [False], "the model is built before the trace begins"
    recorded = everything_recorded(sent)
    assert "handbook_rag.query.text" not in recorded
    assert "how many weeks of paid leave" not in recorded
    assert "Sixteen weeks of paid leave" not in recorded, "nor the answer"


@pytest.mark.usefixtures("indexed_handbook")
def test_an_answer_is_not_traced_unless_tracing_is_switched_on(
    exporters: list[RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wiring, "make_generator", lambda _settings: Scripted())

    result = runner.invoke(app, ["ask", "how many weeks of paid leave"])

    assert result.exit_code == 0, result.output
    assert "trace" not in result.output
    assert exporters == []


@pytest.mark.usefixtures("indexed_handbook")
def test_a_failed_answer_is_traced_as_failed_and_still_reports_its_trace(
    exporters: list[RecordingExporter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_TRACING", "true")
    get_settings.cache_clear()
    limited = GenerationError("The Gemini API rate limit was reached.", reason="rate_limit")
    monkeypatch.setattr(wiring, "make_generator", lambda _settings: Scripted(error=limited))

    result = runner.invoke(app, ["ask", "how many weeks of paid leave"])

    assert result.exit_code == 1
    assert "error: The Gemini API rate limit was reached." in result.output
    (exporter,) = exporters
    sent = exporter.get_finished_spans()
    (root,) = [span for span in sent if span.parent is None]
    assert root.status.status_code is StatusCode.ERROR
    (call,) = [span for span in sent if span.name == "chat stand-in-model"]
    assert call.status.status_code is StatusCode.ERROR
    assert f"trace {format(root.context.trace_id, '032x')}" in result.output
