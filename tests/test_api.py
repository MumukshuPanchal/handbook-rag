import asyncio
import itertools
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr
from starlette.types import Message
from typer.testing import CliRunner

from handbook_rag import __version__, api, streaming, wiring
from handbook_rag.api import (
    ASK_STREAM_EXAMPLE,
    BUSY,
    BUSY_RETRY_AFTER_SECONDS,
    EMPTY,
    MAX_BODY_BYTES,
    MAX_LIMIT,
    MAX_QUERY_CHARS,
    MODEL_QUOTA,
    MODEL_RATE_LIMITED,
    MODEL_TIMED_OUT,
    MODEL_UNAVAILABLE,
    NOT_BUILT,
    NOT_CONFIGURED,
    TRACE_HEADER,
    UNAVAILABLE,
    UNREACHABLE,
    WARM_UP_QUERY,
    AskResponse,
    _is_untraced,
    create_app,
)
from handbook_rag.cli import app as cli
from handbook_rag.config import Settings
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import (
    QdrantVectorStore,
    ScoredChunk,
    VectorStore,
    VectorStoreError,
)
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.streaming import (
    THREAD_NAME,
    AnswerRun,
    CallerGone,
    Failed,
    Finished,
    Slots,
    Text,
    server_sent_event,
)
from tests.support import (
    FakeEmbedder,
    FakeSparseEncoder,
    StandInModel,
    everything_recorded,
    make_chunk,
)

LEAVE = "New parents get sixteen weeks of paid leave."
LAPTOP = "Expense a laptop up to one thousand dollars."


def indexed_store() -> tuple[QdrantVectorStore, FakeEmbedder, FakeSparseEncoder]:
    """Two chunks in an in-memory index, with the stand-in models that indexed them."""
    chunks = [
        make_chunk(LEAVE, heading_path=("Parental leave",)),
        make_chunk(LAPTOP, doc_id="finance/expenses.md", section="finance", title="Expenses"),
    ]
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("api")
    index_chunks(chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    return store, embedder, sparse_encoder


@pytest.fixture
def client() -> Iterator[TestClient]:
    """The API over a two-chunk index, with stand-in models and no server to start."""
    settings = Settings()
    store, embedder, sparse_encoder = indexed_store()
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )
    with TestClient(create_app(settings, retriever=retriever, store=store)) as test_client:
        yield test_client


def test_search_returns_the_passages_best_first_with_where_they_come_from(
    client: TestClient,
) -> None:
    response = client.post("/search", json={"query": "how many weeks of paid leave"})

    assert response.status_code == 200
    body = response.json()
    assert [passage["rank"] for passage in body["results"]] == [1, 2]
    best = body["results"][0]
    assert best["text"] == LEAVE
    assert best["doc_id"] == "people/leave.md"
    assert best["title"] == "Leave"
    assert best["section"] == "people"
    assert best["heading_path"] == ["Parental leave"]
    assert best["url"] == "https://handbook.example.com/handbook/people/leave/"
    assert best["score"] > body["results"][1]["score"]
    assert body["mode"] == "hybrid"
    assert body["reranked"] is False
    assert body["took_ms"] >= 0
    assert body["trace_id"] is None, "tracing is off by default"


def test_search_respects_the_limit_and_the_section(client: TestClient) -> None:
    question = "how many weeks of paid leave"

    one = client.post("/search", json={"query": question, "limit": 1}).json()
    finance = client.post("/search", json={"query": question, "section": "finance"}).json()
    nowhere = client.post("/search", json={"query": question, "section": "legal"})

    assert [passage["text"] for passage in one["results"]] == [LEAVE]
    assert [passage["doc_id"] for passage in finance["results"]] == ["finance/expenses.md"]
    assert nowhere.status_code == 200, "finding nothing is an answer, not an error"
    assert nowhere.json()["results"] == []


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "query"),
        ({"query": ""}, "query"),
        ({"query": "   "}, "query"),
        ({"query": "x" * (MAX_QUERY_CHARS + 1)}, "query"),
        ({"query": "leave", "limit": 0}, "limit"),
        ({"query": "leave", "limit": MAX_LIMIT + 1}, "limit"),
        ({"query": "leave", "limit": "many"}, "limit"),
        ({"query": "leave", "limit": "3"}, "limit"),
        ({"query": "leave", "limit": True}, "limit"),
        ({"query": "leave", "limit": 2.5}, "limit"),
        ({"query": "leave", "top_k": 3}, "top_k"),
    ],
)
def test_search_rejects_a_bad_request_and_names_the_field(
    client: TestClient, body: dict[str, Any], field: str
) -> None:
    response = client.post("/search", json=body)

    assert response.status_code == 422
    assert [error["loc"][-1] for error in response.json()["detail"]] == [field]


def test_a_rejected_request_is_not_sent_back_to_the_caller(client: TestClient) -> None:
    secret = "my salary is 90,000 " * 30

    response = client.post("/search", json={"query": secret, "limit": 5})

    assert response.status_code == 422
    assert "salary" not in response.text
    (problem,) = response.json()["detail"]
    assert set(problem) == {"type", "loc", "msg"}


def test_a_body_far_larger_than_any_valid_request_is_refused_unread(client: TestClient) -> None:
    response = client.post("/search", json={"query": "x", "padding": "y" * MAX_BODY_BYTES})

    assert response.status_code == 413
    assert "larger than" in response.json()["detail"]


def test_the_default_number_of_results_comes_from_the_settings() -> None:
    settings = Settings(search_limit=1)
    store, embedder, sparse_encoder = indexed_store()
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )

    with TestClient(create_app(settings, retriever=retriever, store=store)) as client:
        default = client.post("/search", json={"query": "paid leave"}).json()
        asked = client.post("/search", json={"query": "paid leave", "limit": 2}).json()

    assert len(default["results"]) == 1
    assert len(asked["results"]) == 2


def test_a_search_is_traced_from_the_http_request_down_and_a_health_probe_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web framework traces requests itself, through OpenTelemetry's process-wide tracer.

    Health probes arrive every few seconds. Traced, they would bury the requests made by
    people, so they are left out.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: provider)
    monkeypatch.setattr("handbook_rag.tracing._provider", provider)
    store, embedder, sparse_encoder = indexed_store()
    retriever = wiring.make_retriever(
        Settings(), store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )

    with TestClient(create_app(Settings(), retriever=retriever, store=store)) as client:
        exporter.clear()  # drop the trace of the search that warmed the service up
        assert client.get("/health").status_code == 200
        body = client.post("/search", json={"query": "paid leave"}).json()

    finished = exporter.get_finished_spans()
    (root,) = [span for span in finished if span.parent is None]
    assert root.name == "POST /search"
    assert {span.context.trace_id for span in finished} == {root.context.trace_id}
    assert body["trace_id"] == format(root.context.trace_id, "032x")
    assert {"search", "retrieve.hybrid"} <= {span.name for span in finished}
    assert not [span.name for span in finished if "health" in span.name]
    assert _is_untraced({"type": "http", "path": "/search"}) is False


def test_search_accepts_a_question_at_the_length_limit_and_strips_spaces(
    client: TestClient,
) -> None:
    longest = client.post("/search", json={"query": "x" * MAX_QUERY_CHARS, "limit": MAX_LIMIT})
    padded = client.post("/search", json={"query": "  how many weeks of paid leave  "})

    assert longest.status_code == 200
    assert padded.json()["results"][0]["text"] == LEAVE


class StoreDown:
    """A retriever whose vector store has gone away."""

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        raise VectorStoreError("Cannot reach Qdrant at http://localhost:6333.")


def test_search_answers_503_when_the_vector_store_is_down_without_saying_where_it_is(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store, _, _ = indexed_store()
    with TestClient(create_app(Settings(), retriever=StoreDown(), store=store)) as client:
        response = client.post("/search", json={"query": "paid leave"})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
    assert "localhost:6333" not in response.text
    assert TRACE_HEADER not in response.headers, "tracing is off"
    assert "Cannot reach Qdrant at http://localhost:6333." in caplog.text, "the cause is logged"


def test_health_reports_the_size_of_the_index(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "chunks": 2,
        "answering": False,
        "detail": None,
    }


def health_of(store: VectorStore) -> tuple[int, dict[str, Any]]:
    with TestClient(create_app(Settings(), retriever=StoreDown(), store=store)) as client:
        response = client.get("/health")
    return response.status_code, response.json()


def test_health_says_which_of_three_things_is_wrong(caplog: pytest.LogCaptureFixture) -> None:
    empty = wiring.make_memory_store("empty")
    empty.ensure_collection(FakeEmbedder().dimension)
    never_built = wiring.make_memory_store("missing")
    nothing_listening = wiring.make_store(
        Settings(qdrant_url="http://127.0.0.1:1"), check=False, timeout=2
    )

    assert health_of(empty) == (
        503,
        {
            "status": "unavailable",
            "version": __version__,
            "chunks": 0,
            "answering": False,
            "detail": EMPTY,
        },
    )
    assert health_of(never_built) == (
        503,
        {
            "status": "unavailable",
            "version": __version__,
            "chunks": None,
            "answering": False,
            "detail": NOT_BUILT,
        },
    )
    status, body = health_of(nothing_listening)
    assert (status, body["detail"], body["chunks"]) == (503, UNREACHABLE, None)
    assert "127.0.0.1" not in str(body), "the address of the store is not for callers"
    assert "health check failed" in caplog.text, "the cause is logged instead"


def test_the_service_starts_while_the_vector_store_is_down_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wiring, "make_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
    settings = Settings(qdrant_url="http://127.0.0.1:1")

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "paid leave"})

    assert (health.status_code, health.json()["detail"]) == (503, UNREACHABLE)
    assert (search.status_code, search.json()) == (503, {"detail": UNAVAILABLE})


def test_a_start_up_failure_that_is_not_the_stores_is_not_swallowed() -> None:
    store, _, _ = indexed_store()
    broken = Recording(RuntimeError("the model file is corrupt"))

    with (
        pytest.raises(RuntimeError, match="the model file is corrupt"),
        TestClient(create_app(Settings(), retriever=broken, store=store)),
    ):
        pass


def test_the_api_embeds_on_the_cpu_unless_the_settings_name_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, embedder, sparse_encoder = indexed_store()
    devices: list[str | None] = []

    def make_embedder(settings: Settings) -> FakeEmbedder:
        devices.append(settings.embedding_device)
        return embedder

    monkeypatch.setattr(wiring, "make_embedder", make_embedder)
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: sparse_encoder)

    with TestClient(create_app(Settings(), store=store)):
        pass
    with TestClient(create_app(Settings(embedding_device="mps"), store=store)):
        pass

    assert devices == ["cpu", "mps"]


def test_a_traced_request_returns_the_id_of_its_trace_and_leaves_the_question_out(
    client: TestClient, spans: InMemorySpanExporter
) -> None:
    question = "how many weeks of paid leave"
    spans.clear()  # drop the trace of the search that warmed the service up

    response = client.post("/search", json={"query": question})
    body = response.json()

    finished = spans.get_finished_spans()
    (root,) = [span for span in finished if span.parent is None]
    assert root.name == "search"
    assert body["trace_id"] == format(root.context.trace_id, "032x")
    assert response.headers[TRACE_HEADER] == body["trace_id"]
    assert {span.context.trace_id for span in finished} == {root.context.trace_id}
    assert "retrieve.hybrid" in {span.name for span in finished}
    recorded = [value for span in finished for value in (span.attributes or {}).values()]
    assert question not in recorded


def test_a_failed_request_is_traced_as_failed(spans: InMemorySpanExporter) -> None:
    store, _, _ = indexed_store()
    with TestClient(create_app(Settings(), retriever=StoreDown(), store=store)) as client:
        response = client.post("/search", json={"query": "paid leave"})

    (root,) = [span for span in spans.get_finished_spans() if span.name == "search"]
    assert root.status.is_ok is False
    # The failure a caller reports is the one they most need an id for.
    assert response.headers[TRACE_HEADER] == format(root.context.trace_id, "032x")


class Recording:
    """A retriever that notes what it was asked, and can be told to fail."""

    def __init__(self, error: Exception | None = None) -> None:
        self.asked: list[str] = []
        self._error = error

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        self.asked.append(query)
        if self._error is not None:
            raise self._error
        return []


def test_the_first_caller_does_not_meet_a_cold_model() -> None:
    store, _, _ = indexed_store()
    retriever = Recording()

    with TestClient(create_app(Settings(), retriever=retriever, store=store)) as client:
        assert retriever.asked == [WARM_UP_QUERY], "one search ran before any request"
        client.post("/search", json={"query": "paid leave"})

    assert retriever.asked == [WARM_UP_QUERY, "paid leave"]


def test_the_service_starts_without_an_index_and_says_so_in_health() -> None:
    never_built = wiring.make_memory_store("missing")
    retriever = Recording(VectorStoreError("Collection missing does not exist yet."))

    with TestClient(create_app(Settings(), retriever=retriever, store=never_built)) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "paid leave"})

    assert (health.status_code, health.json()["detail"]) == (503, NOT_BUILT)
    assert (search.status_code, search.json()) == (503, {"detail": UNAVAILABLE})


def test_the_models_are_loaded_once_at_start_up_and_not_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, embedder, sparse_encoder = indexed_store()
    built: list[str] = []

    def make_embedder(_settings: Settings) -> FakeEmbedder:
        built.append("embedder")
        return embedder

    def make_sparse_encoder(_settings: Settings) -> FakeSparseEncoder:
        built.append("sparse encoder")
        return sparse_encoder

    def make_store(_settings: Settings, **_options: Any) -> QdrantVectorStore:
        built.append("store")
        return store

    monkeypatch.setattr(wiring, "make_embedder", make_embedder)
    monkeypatch.setattr(wiring, "make_sparse_encoder", make_sparse_encoder)
    monkeypatch.setattr(wiring, "make_store", make_store)

    application = create_app(Settings())
    assert built == [], "nothing is loaded until the application starts"
    with TestClient(application) as client:
        for _ in range(3):
            assert client.post("/search", json={"query": "paid leave"}).status_code == 200
        assert client.get("/health").status_code == 200

    assert sorted(built) == ["embedder", "sparse encoder", "store"]


def test_the_api_documents_itself(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Handbook assistant"
    assert schema["info"]["version"] == __version__
    assert set(schema["paths"]) == {"/health", "/search", "/ask"}
    query = schema["components"]["schemas"]["SearchRequest"]["properties"]["query"]
    assert query["maxLength"] == MAX_QUERY_CHARS
    assert client.get("/docs").status_code == 200


def test_serve_runs_the_api_where_the_settings_say(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "uvicorn.run", lambda target, **options: calls.append({"target": target, **options})
    )
    monkeypatch.setenv("HANDBOOK_RAG_API_PORT", "8123")

    default = CliRunner().invoke(cli, ["serve"])
    chosen = CliRunner().invoke(cli, ["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert default.exit_code == 0, default.output
    assert chosen.exit_code == 0, chosen.output
    assert calls == [
        {
            "target": "handbook_rag.api:create_app",
            "factory": True,
            "host": "127.0.0.1",
            "port": 8123,
        },
        {"target": "handbook_rag.api:create_app", "factory": True, "host": "0.0.0.0", "port": 9000},
    ]


# POST /ask. The answer model is a stand-in that writes its answer piece by piece, as the
# real generators do, from the worker thread the answer runs in.

QUESTION = "how many weeks of paid leave"
# How long a test waits for a worker thread before it calls the wait a failure.
WAIT = 5.0


class Writing(StandInModel):
    """A stand-in answer model that passes its answer on piece by piece through on_text.

    Each piece cites passages by their place among the ones it is given. It can fail
    before its first piece or after its last, or stop before one of its pieces (the
    second, unless told otherwise) until the test lets it go on.
    """

    def __init__(
        self,
        *pieces: tuple[str, tuple[int, ...]],
        answered: bool = True,
        fail_before: BaseException | None = None,
        fail_after: BaseException | None = None,
        hold: threading.Event | None = None,
        hold_at: int = 1,
    ) -> None:
        self.pieces = pieces
        self.answered = answered
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.hold = hold
        self.hold_at = hold_at
        self.asked: list[tuple[str, list[Chunk]]] = []
        self.written: list[str] = []
        self.interrupted: BaseException | None = None
        self.started = threading.Event()
        self.finished = threading.Event()

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.asked.append((question, list(chunks)))
        self.started.set()
        try:
            if self.fail_before is not None:
                raise self.fail_before
            for index, (text, _cited) in enumerate(self.pieces):
                if index == self.hold_at and self.hold is not None:
                    assert self.hold.wait(WAIT), "the test never let the model go on"
                assert on_text is not None
                try:
                    on_text(text)
                except BaseException as error:
                    self.interrupted = error
                    raise
                self.written.append(text)
            if self.fail_after is not None:
                raise self.fail_after
            return Answer(
                question=question,
                segments=tuple(
                    Segment(text, tuple(Citation(chunks[place]) for place in cited))
                    for text, cited in self.pieces
                ),
                answered=self.answered,
                model="stand-in-model-002",
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=1200, output_tokens=40),
                seconds=1.0,
                seconds_to_first_text=0.5,
            )
        finally:
            self.finished.set()


def answering(
    model: Generator | None,
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
) -> FastAPI:
    """The API over the two-chunk index, answering with the given stand-in model."""
    settings = settings or Settings()
    store, embedder, sparse_encoder = indexed_store()
    if retriever is None:
        retriever = wiring.make_retriever(
            settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
        )
    return create_app(settings, retriever=retriever, store=store, generator=model)


def answers_ended() -> bool:
    """Wait for the worker thread of every answer to end, and say whether they all did."""
    for thread in threading.enumerate():
        if thread.name == THREAD_NAME:
            thread.join(WAIT)
    return not [thread for thread in threading.enumerate() if thread.name == THREAD_NAME]


def slots_in_use(application: FastAPI) -> int:
    slots: Slots = application.state.answer_slots
    return slots.in_use


@pytest.fixture
def ticking_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clock that moves on one second each time it is read, so every timing is known.

    An answer reads it three times: when the request arrives, when the first text is
    read, and when the answer is complete.
    """
    ticks = itertools.count()
    clock = SimpleNamespace(perf_counter=lambda: float(next(ticks)))
    monkeypatch.setattr(api, "time", clock)
    monkeypatch.setattr(streaming, "time", clock)


LEAVE_SOURCE = {
    "doc_id": "people/leave.md",
    "title": "Leave",
    "url": "https://handbook.example.com/handbook/people/leave/",
    "section": "people",
    "heading_path": ["Parental leave"],
    "text": LEAVE,
}
LAPTOP_SOURCE = {
    "doc_id": "finance/expenses.md",
    "title": "Expenses",
    "url": "https://handbook.example.com/handbook/finance/expenses/",
    "section": "finance",
    "heading_path": [],
    "text": LAPTOP,
}


@pytest.mark.usefixtures("ticking_clock")
def test_ask_numbers_the_sources_in_the_order_the_answer_first_cites_them() -> None:
    # The search ranks the leave passage first; the answer cites the laptop passage first.
    model = Writing(
        ("A laptop is expensed up to one thousand dollars.", (1,)),
        (" Parents get sixteen weeks.", (0, 1)),
        (" Ask the People team.", ()),
    )
    with TestClient(answering(model)) as client:
        response = client.post("/ask", json={"query": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert TRACE_HEADER not in response.headers, "tracing is off"
    assert response.json() == {
        "answer": (
            "A laptop is expensed up to one thousand dollars.[1]"
            " Parents get sixteen weeks.[1][2] Ask the People team."
        ),
        "answered": True,
        "segments": [
            {"text": "A laptop is expensed up to one thousand dollars.", "sources": [1]},
            {"text": " Parents get sixteen weeks.", "sources": [1, 2]},
            {"text": " Ask the People team.", "sources": []},
        ],
        "sources": [{"number": 1, **LAPTOP_SOURCE}, {"number": 2, **LEAVE_SOURCE}],
        "model": "stand-in-model-002",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1200, "output_tokens": 40},
        "requests": 1,
        "fallback": False,
        "took_ms": 2000.0,
        "first_text_ms": 1000.0,
        "trace_id": None,
    }
    ((asked, chunks),) = model.asked
    assert asked == QUESTION
    assert [chunk.text for chunk in chunks] == [LEAVE, LAPTOP], "ranked as the search found them"


def test_a_reply_that_is_not_an_answer_lists_no_sources_and_carries_no_markers() -> None:
    # The model declined, and still attached a citation to the sentence.
    model = Writing((NOT_FOUND_MESSAGE, (0,)), answered=False)
    with TestClient(answering(model)) as client:
        body = client.post("/ask", json={"query": QUESTION}).json()

    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["answered"] is False
    assert body["segments"] == [{"text": NOT_FOUND_MESSAGE, "sources": []}]
    assert body["sources"] == []


@pytest.mark.usefixtures("ticking_clock")
def test_an_answer_without_any_text_has_no_time_to_its_first_text() -> None:
    with TestClient(answering(Writing(answered=False))) as client:
        body = client.post("/ask", json={"query": QUESTION}).json()

    assert (body["answer"], body["segments"], body["first_text_ms"]) == ("", [], None)
    assert body["took_ms"] == 1000.0


def test_ask_gives_the_model_as_many_passages_as_the_settings_say_from_the_section_asked() -> None:
    model = Writing(("Sixteen weeks.", (0,)))

    with TestClient(answering(model, Settings(answer_context_chunks=1))) as client:
        client.post("/ask", json={"query": QUESTION})
        client.post("/ask", json={"query": f"  {QUESTION}  ", "section": "finance"})

    (first, second) = model.asked
    assert [chunk.text for chunk in first[1]] == [LEAVE]
    assert second[0] == QUESTION, "spaces around the question are taken off"
    assert [chunk.text for chunk in second[1]] == [LAPTOP]


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "query"),
        ({"query": ""}, "query"),
        ({"query": "   "}, "query"),
        ({"query": "x" * (MAX_QUERY_CHARS + 1)}, "query"),
        ({"query": "leave", "section": "x" * 101}, "section"),
        ({"query": "leave", "stream": "true"}, "stream"),
        ({"query": "leave", "stream": 1}, "stream"),
        ({"query": "leave", "stream": None}, "stream"),
        ({"query": "leave", "limit": 3}, "limit"),
    ],
)
def test_ask_rejects_a_bad_request_and_names_the_field(body: dict[str, Any], field: str) -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(answering(model)) as client:
        response = client.post("/ask", json=body)

    assert response.status_code == 422
    assert [error["loc"][-1] for error in response.json()["detail"]] == [field]
    assert model.asked == [], "nothing was asked of the model"


def test_ask_accepts_a_question_at_the_length_limit_and_refuses_a_body_far_larger() -> None:
    with TestClient(answering(Writing(("Sixteen weeks.", (0,))))) as client:
        longest = client.post("/ask", json={"query": "x" * MAX_QUERY_CHARS, "stream": False})
        huge = client.post("/ask", json={"query": "x", "padding": "y" * MAX_BODY_BYTES})

    assert longest.status_code == 200
    assert huge.status_code == 413


def answer_event(stream: bytes) -> bytes:
    """The data of the last event of a stream, which is the answer."""
    head, data = stream.removesuffix(b"\n\n").rsplit(b"\ndata: ", 1)
    assert head.endswith(b"event: answer")
    return data


@pytest.mark.usefixtures("ticking_clock")
def test_a_stream_sends_each_piece_as_it_is_written_then_the_answer_byte_for_byte() -> None:
    # A line break and a letter outside ASCII: JSON stays on one line, and text is UTF-8.
    model = Writing(("Sixteen weeks.", (0,)), ("\nCafé visits need no approval.", ()))
    with TestClient(answering(model)) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert TRACE_HEADER not in response.headers, "tracing is off"
    assert (
        response.content
        == (
            'event: text\ndata: {"text":"Sixteen weeks."}\n\n'
            'event: text\ndata: {"text":"\\nCafé visits need no approval."}\n\n'
            "event: answer\n"
            'data: {"answer":"Sixteen weeks.[1]\\nCafé visits need no approval.","answered":true,'
            '"segments":[{"text":"Sixteen weeks.","sources":[1]},'
            '{"text":"\\nCafé visits need no approval.","sources":[]}],'
            '"sources":[{"number":1,"doc_id":"people/leave.md","title":"Leave",'
            '"url":"https://handbook.example.com/handbook/people/leave/","section":"people",'
            '"heading_path":["Parental leave"],'
            '"text":"New parents get sixteen weeks of paid leave."}],'
            '"model":"stand-in-model-002","stop_reason":"end_turn",'
            '"usage":{"input_tokens":1200,"output_tokens":40},"requests":1,"fallback":false,'
            '"took_ms":2000.0,"first_text_ms":1000.0,"trace_id":null}\n\n'
        ).encode()
    )


def test_text_that_utf8_cannot_encode_is_replaced_instead_of_failing_either_form() -> None:
    # A lone half of a surrogate pair is a valid Python string and not valid UTF-8.
    model = Writing(("Sixteen weeks \ud800.", (0,)))
    with TestClient(answering(model)) as client:
        plain = client.post("/ask", json={"query": QUESTION})
        streamed = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert plain.status_code == 200
    assert plain.json()["answer"] == "Sixteen weeks ?.[1]"
    assert streamed.status_code == 200
    assert streamed.content.startswith(b'event: text\ndata: {"text":"Sixteen weeks ?."}\n\n')
    assert json.loads(answer_event(streamed.content))["answer"] == "Sixteen weeks ?.[1]"


@pytest.mark.usefixtures("ticking_clock")
def test_the_answer_at_the_end_of_a_stream_is_exactly_the_response_without_one() -> None:
    model = Writing(("Sixteen weeks.", (0,)), (" A laptop too.", (1,)))
    with TestClient(answering(model)) as client:
        plain = client.post("/ask", json={"query": QUESTION})
        streamed = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert answer_event(streamed.content) == plain.content


@pytest.mark.usefixtures("ticking_clock")
def test_a_stream_whose_answer_was_finished_before_it_began_holds_just_the_answer() -> None:
    with TestClient(answering(Writing(answered=False))) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert response.status_code == 200
    assert response.content.startswith(b"event: answer\ndata: {")
    assert response.content.count(b"event: ") == 1
    assert json.loads(answer_event(response.content))["first_text_ms"] is None


@pytest.mark.parametrize("stream", [False, True])
def test_a_store_that_is_down_is_a_503_before_anything_is_sent_even_for_a_stream(
    stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(answering(model, retriever=StoreDown())) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"detail": UNAVAILABLE}
    assert "localhost:6333" not in response.text
    assert "Cannot reach Qdrant at http://localhost:6333." in caplog.text, "the cause is logged"
    assert model.asked == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_that_fails_before_any_text_is_a_503_that_keeps_the_cause_to_the_log(
    stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "The Gemini API rejected the credential. Check GEMINI_API_KEY in .env."
    model = Writing(fail_before=GenerationError(secret, reason="configuration"))
    with TestClient(answering(model)) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})

    assert response.status_code == 503
    assert response.json() == {"detail": MODEL_UNAVAILABLE}
    assert "Retry-After" not in response.headers
    assert ".env" not in response.text
    assert secret in caplog.text


@pytest.mark.parametrize(
    ("error", "detail", "retry_after"),
    [
        (GenerationError("x", reason="rate_limit", retry_after=36.2), MODEL_RATE_LIMITED, "37"),
        (GenerationError("x", reason="rate_limit", retry_after=36.0), MODEL_RATE_LIMITED, "36"),
        # The header takes whole seconds, and never says to come back at once.
        (GenerationError("x", reason="rate_limit", retry_after=0.2), MODEL_RATE_LIMITED, "1"),
        (GenerationError("x", reason="rate_limit", retry_after=0.0), MODEL_RATE_LIMITED, "1"),
        # Never more than an hour, however long the provider asks for.
        (GenerationError("x", reason="rate_limit", retry_after=3600.0), MODEL_RATE_LIMITED, "3600"),
        (GenerationError("x", reason="rate_limit", retry_after=3600.5), MODEL_RATE_LIMITED, "3600"),
        (GenerationError("x", reason="rate_limit", retry_after=1e300), MODEL_RATE_LIMITED, "3600"),
        # A wait that is not a number of seconds is no wait, and no fault of the service.
        (GenerationError("x", reason="rate_limit", retry_after=math.inf), MODEL_RATE_LIMITED, None),
        (GenerationError("x", reason="rate_limit", retry_after=math.nan), MODEL_RATE_LIMITED, None),
        (GenerationError("x", reason="rate_limit", retry_after=-5.0), MODEL_RATE_LIMITED, None),
        (GenerationError("x", reason="rate_limit"), MODEL_RATE_LIMITED, None),
        (GenerationError("x", reason="quota"), MODEL_QUOTA, None),
        (GenerationError("x", reason="quota", retry_after=36.0), MODEL_QUOTA, None),
        (GenerationError("x", reason="configuration"), MODEL_UNAVAILABLE, None),
        (GenerationError("x", reason="unavailable"), MODEL_UNAVAILABLE, None),
        (GenerationError("x", reason="timeout"), MODEL_TIMED_OUT, None),
        (GenerationError("x", reason="timeout", retry_after=5.0), MODEL_TIMED_OUT, None),
    ],
)
def test_the_models_limits_are_a_503_that_says_which_limit_and_when_to_come_back(
    error: GenerationError, detail: str, retry_after: str | None
) -> None:
    """503 and not 429: the limit is the service's own, with its provider, not the caller's."""
    with TestClient(answering(Writing(fail_before=error))) as client:
        response = client.post("/ask", json={"query": QUESTION})

    assert response.status_code == 503
    assert response.json() == {"detail": detail}
    assert response.headers.get("Retry-After") == retry_after


def test_a_model_that_fails_after_its_first_text_ends_the_stream_with_an_error_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lost = GenerationError("Could not reach the Gemini API. Check the network.")
    model = Writing(("Sixteen weeks.", (0,)), fail_after=lost)
    with TestClient(answering(model)) as client:
        streamed = client.post("/ask", json={"query": QUESTION, "stream": True})
        plain = client.post("/ask", json={"query": QUESTION})

    assert streamed.status_code == 200, "the stream had begun"
    assert streamed.content == (
        b'event: text\ndata: {"text":"Sixteen weeks."}\n\n'
        b'event: error\ndata: {"detail":"The answer model is unavailable. Try again later."}\n\n'
    )
    assert "answer failed after it began" in caplog.text
    assert "Check the network" in caplog.text
    # Without a stream nothing has been sent yet, so the failure is a status.
    assert (plain.status_code, plain.json()) == (503, {"detail": MODEL_UNAVAILABLE})


def test_a_fault_of_our_own_after_the_first_text_ends_the_stream_with_fixed_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Writing(("Sixteen weeks.", (0,)), fail_after=RuntimeError("a bug at /srv/app.py"))
    with TestClient(answering(model)) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert response.content.endswith(
        b'event: error\ndata: {"detail":"The answer could not be finished."}\n\n'
    )
    assert b"/srv/app.py" not in response.content
    (record,) = [r for r in caplog.records if "answer failed after it began" in r.getMessage()]
    assert record.levelname == "ERROR"
    assert record.exc_info is not None, "with the stack trace, for whoever runs the service"


@pytest.mark.parametrize("stream", [False, True])
def test_a_fault_of_our_own_before_any_text_is_a_server_error_and_frees_its_slot(
    stream: bool,
) -> None:
    application = answering(Writing(fail_before=RuntimeError("a bug")))

    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})
    with (
        TestClient(application) as client,
        pytest.raises(RuntimeError, match="a bug"),
    ):
        client.post("/ask", json={"query": QUESTION, "stream": stream})

    assert response.status_code == 500
    assert answers_ended()
    assert slots_in_use(application) == 0


def test_without_an_api_key_the_service_still_searches_and_says_answering_is_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with TestClient(answering(None)) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": QUESTION})
        ask = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert (health.status_code, health.json()["answering"]) == (200, False)
    assert search.status_code == 200
    assert (ask.status_code, ask.json()) == (503, {"detail": NOT_CONFIGURED})
    assert "answering is not configured: No Gemini API key was found" in caplog.text


def test_health_says_answering_is_on_when_a_model_is_configured_without_asking_it() -> None:
    with TestClient(answering(Writing())) as passed:
        given = passed.get("/health").json()
    settings = Settings(gemini_api_key=SecretStr("a-key-that-is-never-sent"))
    with TestClient(answering(None, settings)) as built:
        from_settings = built.get("/health").json()

    assert given["answering"] is True
    assert from_settings["answering"] is True, "built from the key, with no request"


def test_a_fallback_model_without_its_key_is_found_when_the_service_starts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        gemini_api_key=SecretStr("a-key-that-is-never-sent"),
        answer_fallback_model="claude-haiku-5-5",
    )
    with TestClient(answering(None, settings)) as client:
        health = client.get("/health").json()
        ask = client.post("/ask", json={"query": QUESTION})

    assert health["answering"] is False
    assert (ask.status_code, ask.json()) == (503, {"detail": NOT_CONFIGURED})
    assert "The fallback model claude-haiku-5-5 needs ANTHROPIC_API_KEY" in caplog.text


def with_fallback(answer_model: Writing, fallback: Writing) -> RetryingGenerator:
    """The two stand-in models in the wrapper the service uses, with no real pauses."""
    return RetryingGenerator(
        answer_model,
        attempts=3,
        deadline_seconds=30.0,
        fallback=lambda: fallback,
        sleep=lambda _seconds: None,
    )


@pytest.mark.parametrize("stream", [False, True])
def test_the_answer_says_how_many_requests_it_took_and_that_the_fallback_wrote_it(
    stream: bool,
) -> None:
    used_up = GenerationError("Today's limit is used up.", reason="quota")
    generator = with_fallback(Writing(fail_before=used_up), Writing(("Sixteen weeks.", (0,))))
    with TestClient(answering(generator)) as client:
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})

    body = json.loads(answer_event(response.content)) if stream else response.json()
    assert response.status_code == 200
    assert (body["requests"], body["fallback"]) == (2, True)
    assert body["answer"] == "Sixteen weeks.[1]"


def test_an_answer_that_was_tried_again_counts_every_request_and_no_fallback() -> None:
    busy = GenerationError("The model is overloaded.")
    model = Writing(("Sixteen weeks.", (0,)))
    flaky = [busy, busy]

    class Busy(Writing):
        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            if flaky:
                raise flaky.pop()
            return super().generate(question, chunks, on_text=on_text)

    generator = with_fallback(Busy(("Sixteen weeks.", (0,))), model)
    with TestClient(answering(generator)) as client:
        body = client.post("/ask", json={"query": QUESTION}).json()

    assert (body["requests"], body["fallback"]) == (3, False)
    assert model.asked == [], "the fallback was not needed"


def test_when_the_fallback_fails_too_the_log_says_why_the_answer_model_gave_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    used_up = GenerationError("Today's limit for gemini-3.5-flash-lite is used up.", reason="quota")
    down = GenerationError("The Claude API returned an error (529): Overloaded")
    generator = with_fallback(Writing(fail_before=used_up), Writing(fail_before=down))
    with TestClient(answering(generator)) as client:
        response = client.post("/ask", json={"query": QUESTION})

    # The caller hears about the fallback's failure, the latest news.
    assert (response.status_code, response.json()) == (503, {"detail": MODEL_UNAVAILABLE})
    assert "used up" not in response.text
    assert (
        "The Claude API returned an error (529): Overloaded (before the fallback, the answer "
        "model failed: Today's limit for gemini-3.5-flash-lite is used up.)"
    ) in caplog.text


def test_a_model_that_did_not_answer_in_time_has_a_503_of_its_own() -> None:
    silent = GenerationError("The Gemini API did not answer in time.", reason="timeout")
    with TestClient(answering(Writing(fail_before=silent))) as client:
        response = client.post("/ask", json={"query": QUESTION})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The answer model did not answer in time. Try again later."
    }
    assert MODEL_TIMED_OUT != MODEL_UNAVAILABLE
    assert "Retry-After" not in response.headers


def test_a_question_beyond_the_answers_in_flight_is_turned_away_until_one_ends() -> None:
    hold = threading.Event()
    model = Writing(("Sixteen weeks.", (0,)), (" Ask HR.", ()), hold=hold)
    application = answering(model, Settings(api_max_answers_in_flight=1))
    first: list[Response] = []

    with TestClient(application) as client:
        asking = threading.Thread(
            target=lambda: first.append(client.post("/ask", json={"query": QUESTION}))
        )
        asking.start()
        try:
            assert model.started.wait(WAIT)
            turned_away = client.post("/ask", json={"query": QUESTION, "stream": True})
            health = client.get("/health")
        finally:
            hold.set()
            asking.join(WAIT)
        again = client.post("/ask", json={"query": QUESTION})

    assert turned_away.status_code == 503
    assert turned_away.json() == {"detail": BUSY}
    assert turned_away.headers["Retry-After"] == str(BUSY_RETRY_AFTER_SECONDS)
    assert health.status_code == 200, "a full house does not stop the rest of the service"
    assert len(model.asked) == 2, "the question turned away never reached the model"
    assert [response.status_code for response in first] == [200]
    assert again.status_code == 200
    assert answers_ended()
    assert slots_in_use(application) == 0


def ask_scope(body: bytes, spec_version: str) -> dict[str, Any]:
    """An HTTP request to /ask as a server hands it to the application."""
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "root_path": "",
        "path": "/ask",
        "raw_path": b"/ask",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
    }


async def leave_after_the_first_text(application: FastAPI, spec_version: str) -> list[Message]:
    """Ask for a stream and go away once the first text has arrived.

    How a server tells the application depends on the version of the interface it
    speaks: by a disconnect message when asked (before 2.4, which is what the installed
    uvicorn speaks), or by failing to send (from 2.4 on).
    """
    body = json.dumps({"query": QUESTION, "stream": True}).encode()
    sent: list[Message] = []
    got_text = asyncio.Event()
    asked = False

    async def receive() -> Message:
        nonlocal asked
        if not asked:
            asked = True
            return {"type": "http.request", "body": body, "more_body": False}
        await got_text.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            got_text.set()
            if spec_version == "2.4":
                raise OSError("the caller has gone")

    with suppress(OSError):
        await application(ask_scope(body, spec_version), receive, send)
    return sent


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_when_the_caller_goes_away_mid_stream_the_model_call_is_stopped_and_its_slot_freed(
    spec_version: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="handbook_rag.streaming")
    hold = threading.Event()
    model = Writing(("Sixteen weeks.", (0,)), (" Ask HR.", ()), (" Then rest.", ()), hold=hold)
    application = answering(model)

    with TestClient(application):
        sent = asyncio.run(leave_after_the_first_text(application, spec_version))
        # The model is still held after its first piece. Now it may go on, and it is
        # stopped when it hands over its next piece.
        hold.set()
        assert model.finished.wait(WAIT)
        assert answers_ended()

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b'event: text\ndata: {"text":"Sixteen weeks."}\n\n'
    assert isinstance(model.interrupted, CallerGone)
    assert model.written == ["Sixteen weeks."], "nothing was written after the caller left"
    assert slots_in_use(application) == 0
    assert "the caller went away" in caplog.text


def trace_id_of_ask(response: Response, stream: bool) -> str | None:
    body = json.loads(answer_event(response.content)) if stream else response.json()
    traced = body["trace_id"]
    assert traced is None or isinstance(traced, str)
    return traced


@pytest.mark.parametrize("stream", [False, True])
def test_an_answer_is_one_trace_from_the_http_request_down_also_when_it_streams(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer is written in a worker thread. Its spans still belong to the request."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: provider)
    monkeypatch.setattr("handbook_rag.tracing._provider", provider)

    with TestClient(answering(Writing(("Sixteen weeks.", (0,))))) as client:
        exporter.clear()  # drop the trace of the search that warmed the service up
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})
        assert answers_ended()

    finished = exporter.get_finished_spans()
    (root,) = [span for span in finished if span.parent is None]
    assert root.name == "POST /ask"
    assert {span.context.trace_id for span in finished} == {root.context.trace_id}
    assert {"answer", "search", "retrieve.hybrid", "chat stand-in-model"} <= {
        span.name for span in finished
    }
    traced = format(root.context.trace_id, "032x")
    assert trace_id_of_ask(response, stream) == traced
    assert response.headers[TRACE_HEADER] == traced


@pytest.mark.parametrize("stream", [False, True])
def test_a_traced_answer_returns_the_id_of_its_trace_and_leaves_the_question_out(
    stream: bool, spans: InMemorySpanExporter
) -> None:
    with TestClient(answering(Writing(("Sixteen weeks.", (0,))))) as client:
        spans.clear()
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})
        assert answers_ended()

    finished = spans.get_finished_spans()
    (root,) = [span for span in finished if span.parent is None]
    assert root.name == "answer"
    assert trace_id_of_ask(response, stream) == format(root.context.trace_id, "032x")
    assert response.headers[TRACE_HEADER] == format(root.context.trace_id, "032x")
    recorded = everything_recorded(finished)
    assert QUESTION not in recorded
    assert "Sixteen weeks." not in recorded
    assert QUESTION not in sent_back(response)


def sent_back(response: Response) -> str:
    """Everything a response tells its caller: its headers and its body."""
    headers = "\n".join(f"{name}: {value}" for name, value in response.headers.items())
    return f"{headers}\n{response.text}"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("after_text", [False, True])
def test_a_failed_answer_keeps_the_question_the_answer_and_the_errors_message_to_itself(
    stream: bool, after_text: bool, spans: InMemorySpanExporter
) -> None:
    secret = "Could not read /Users/someone/handbook/.env: check GEMINI_API_KEY."
    question = "my salary is 90,000: how many weeks of paid leave do I get?"
    error = GenerationError(secret)
    model = (
        Writing(("Parents get sixteen weeks.", (0,)), fail_after=error)
        if after_text
        else Writing(fail_before=error)
    )
    with TestClient(answering(model)) as client:
        spans.clear()
        response = client.post("/ask", json={"query": question, "stream": stream})
        assert answers_ended()

    recorded = everything_recorded(spans.get_finished_spans())
    for part in (question, "salary is 90,000", secret, "/Users/someone", "GEMINI_API_KEY"):
        assert part not in recorded
        assert part not in sent_back(response)
    assert "Parents get sixteen weeks" not in recorded
    assert "handbook_rag.generate.base.GenerationError" in recorded, "the kind of error only"


@pytest.mark.parametrize("stream", [False, True])
def test_a_failed_answer_is_traced_as_failed_and_its_503_carries_the_trace_id(
    stream: bool, spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    model = Writing(fail_before=GenerationError("x", reason="rate_limit", retry_after=36.0))
    with TestClient(answering(model)) as client:
        spans.clear()
        response = client.post("/ask", json={"query": QUESTION, "stream": stream})
        assert answers_ended()

    (root,) = [span for span in spans.get_finished_spans() if span.parent is None]
    (call,) = [span for span in spans.get_finished_spans() if span.name == "chat stand-in-model"]
    traced = format(root.context.trace_id, "032x")
    assert root.status.status_code is StatusCode.ERROR
    assert call.status.status_code is StatusCode.ERROR
    assert response.status_code == 503
    assert response.headers[TRACE_HEADER] == traced
    assert response.headers["Retry-After"] == "36"
    assert f"answer failed, trace {traced}" in caplog.text


def test_the_api_documents_both_forms_of_an_answer_and_why_it_may_be_refused(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    ask = schema["paths"]["/ask"]["post"]
    schemas = schema["components"]["schemas"]

    assert set(ask["responses"]) == {"200", "422", "503"}
    success = ask["responses"]["200"]
    assert success["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AskResponse"
    }
    assert "event: answer" in success["description"]
    assert "event: error" in success["description"]
    example = success["content"]["text/event-stream"]["example"]
    assert example == ASK_STREAM_EXAMPLE
    # The example is a stream that keeps to the format, ending in a valid answer.
    AskResponse.model_validate_json(answer_event(example.encode()))
    assert example.startswith('event: text\ndata: {"text":')
    refused = ask["responses"]["503"]
    assert refused["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Problem"
    }
    assert set(refused["headers"]) == {"Retry-After", TRACE_HEADER}
    request = schemas["AskRequest"]
    assert set(request["properties"]) == {"query", "section", "stream"}
    assert request["required"] == ["query"]
    assert request["additionalProperties"] is False
    assert request["properties"]["query"]["maxLength"] == MAX_QUERY_CHARS
    assert request["properties"]["stream"]["type"] == "boolean"
    assert request["properties"]["stream"]["default"] is False
    assert set(schemas["AskResponse"]["properties"]) == {
        "answer",
        "answered",
        "segments",
        "sources",
        "model",
        "stop_reason",
        "usage",
        "requests",
        "fallback",
        "took_ms",
        "first_text_ms",
        "trace_id",
    }
    assert schemas["Health"]["properties"]["answering"]["type"] == "boolean"


# The parts that turn an answer written in a thread into events a request awaits.

ANSWER = Answer(
    question=QUESTION,
    segments=(Segment("Sixteen weeks."),),
    answered=True,
    model="stand-in-model",
    stop_reason="end_turn",
)


def test_events_written_before_the_first_read_are_all_read_in_order() -> None:
    slots = Slots(1)

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        on_trace("a" * 32)
        on_text("Sixteen ")
        on_text("weeks.")
        return ANSWER

    async def scenario() -> tuple[AnswerRun, list[object]]:
        assert slots.take()
        run = AnswerRun.start(write, slots)
        assert answers_ended(), "the worker has finished before anything is read"
        return run, [await run.next() for _ in range(3)]

    run, events = asyncio.run(scenario())

    assert events == [Text("Sixteen "), Text("weeks."), Finished(ANSWER)]
    assert run.trace_id == "a" * 32
    assert run.first_text_at is not None
    assert slots.in_use == 0


def test_a_failure_in_the_worker_is_handed_to_the_reader_as_the_last_event() -> None:
    slots = Slots(1)
    bug = ZeroDivisionError("a bug")

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        on_text("Sixteen ")
        raise bug

    async def scenario() -> list[object]:
        assert slots.take()
        run = AnswerRun.start(write, slots)
        return [await run.next(), await run.next()]

    assert asyncio.run(scenario()) == [Text("Sixteen "), Failed(bug)]
    assert answers_ended()
    assert slots.in_use == 0


@pytest.mark.parametrize("stop", [SystemExit(3), KeyboardInterrupt()])
def test_even_an_exit_raised_in_the_worker_reaches_the_reader_and_frees_its_slot(
    stop: BaseException,
) -> None:
    """Not only errors: without a last event the request would wait for ever."""
    slots = Slots(1)

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        raise stop

    async def scenario() -> object:
        assert slots.take()
        run = AnswerRun.start(write, slots)
        return await asyncio.wait_for(run.next(), WAIT)

    assert asyncio.run(scenario()) == Failed(stop)
    assert answers_ended()
    assert slots.in_use == 0


class SlowToGiveBack(Slots):
    """Slots that take a moment to give one back, as a busy machine might."""

    def give_back(self) -> None:
        time.sleep(0.2)
        super().give_back()


def test_the_slot_is_free_by_the_time_the_last_event_is_read() -> None:
    """A caller who has read the answer and asks again must not find the service full."""
    slots = SlowToGiveBack(1)

    async def scenario() -> tuple[object, int]:
        assert slots.take()
        run = AnswerRun.start(lambda _on_text, _on_trace: ANSWER, slots)
        last = await asyncio.wait_for(run.next(), WAIT)
        # Read at once: no waiting for the worker thread to end.
        return last, slots.in_use

    assert asyncio.run(scenario()) == (Finished(ANSWER), 0)
    assert answers_ended()


def test_a_worker_thread_does_not_keep_the_process_alive() -> None:
    """A model that never answers must not stop the server from shutting down."""
    slots = Slots(1)
    go_on = threading.Event()

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        assert go_on.wait(WAIT)
        return ANSWER

    async def scenario() -> list[threading.Thread]:
        assert slots.take()
        run = AnswerRun.start(write, slots)
        workers = [thread for thread in threading.enumerate() if thread.name == THREAD_NAME]
        go_on.set()
        await run.next()
        return workers

    (worker,) = asyncio.run(scenario())

    assert worker.daemon is True
    assert answers_ended()


def test_a_run_that_is_stopped_abandons_its_answer_at_the_next_text_and_frees_its_slot() -> None:
    slots = Slots(1)
    go_on = threading.Event()
    raised: list[BaseException] = []

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        on_text("Sixteen ")
        assert go_on.wait(WAIT)
        try:
            on_text("weeks.")
        except BaseException as error:
            raised.append(error)
            raise
        return ANSWER

    async def scenario() -> list[object]:
        assert slots.take()
        run = AnswerRun.start(write, slots)
        first = await run.next()
        run.stop()
        go_on.set()
        return [first, await run.next()]

    first, last = asyncio.run(scenario())

    assert first == Text("Sixteen ")
    assert isinstance(last, Failed)
    assert isinstance(last.error, CallerGone)
    assert [type(error) for error in raised] == [CallerGone]
    assert answers_ended()
    assert slots.in_use == 0


def test_a_worker_whose_reader_has_gone_still_ends_quietly_and_frees_its_slot() -> None:
    """The event loop can close while a worker is still writing, as when a server stops."""
    slots = Slots(1)
    go_on = threading.Event()
    crashed: list[threading.ExceptHookArgs] = []

    def write(on_text: Callable[[str], None], on_trace: Callable[[str], None]) -> Answer:
        assert go_on.wait(WAIT)
        on_text("Sixteen weeks.")
        return ANSWER

    async def scenario() -> None:
        assert slots.take()
        AnswerRun.start(write, slots)

    previous = threading.excepthook
    threading.excepthook = crashed.append
    try:
        asyncio.run(scenario())  # the loop is closed when this returns
        go_on.set()
        assert answers_ended()
    finally:
        threading.excepthook = previous

    assert crashed == []
    assert slots.in_use == 0


def test_a_slot_is_given_back_when_the_worker_cannot_be_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = Slots(1)

    def refuse(_thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    async def scenario() -> None:
        assert slots.take()
        with monkeypatch.context() as patched:
            patched.setattr(threading.Thread, "start", refuse)
            AnswerRun.start(lambda _on_text, _on_trace: ANSWER, slots)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        asyncio.run(scenario())
    assert slots.in_use == 0


def test_slots_are_taken_until_none_is_left_and_given_back_one_by_one() -> None:
    slots = Slots(2)

    assert [slots.take(), slots.take(), slots.take()] == [True, True, False]
    assert slots.in_use == 2
    slots.give_back()
    assert slots.in_use == 1
    assert slots.take() is True
    assert slots.take() is False


def test_a_server_sent_event_is_its_name_its_data_and_a_blank_line() -> None:
    assert (
        server_sent_event("text", '{"text":"é"}') == 'event: text\ndata: {"text":"é"}\n\n'.encode()
    )
    for broken in ("two\nlines", "two\rlines"):
        with pytest.raises(ValueError, match="single line"):
            server_sent_event("text", broken)


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_cancelled_while_it_waits_for_its_answer_stops_the_model_call(
    stream: bool,
) -> None:
    """As when the server shuts down while the model is still thinking.

    A stream is decided on only once the first text is there, so until then both forms
    wait in the same place.
    """
    hold = threading.Event()
    model = Writing(("Sixteen weeks.", (0,)), hold=hold, hold_at=0)
    application = answering(model)
    body = json.dumps({"query": QUESTION, "stream": stream}).encode()

    async def cancel_while_waiting() -> None:
        asked = False

        async def receive() -> Message:
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()  # the caller never goes away by itself
            raise AssertionError("unreachable")

        async def send(_message: Message) -> None:
            raise AssertionError("nothing is sent before the answer exists")

        request = asyncio.create_task(application(ask_scope(body, "2.4"), receive, send))
        assert await asyncio.to_thread(model.started.wait, WAIT)
        request.cancel()
        with suppress(asyncio.CancelledError):
            await request

    with TestClient(application):
        asyncio.run(cancel_while_waiting())
        hold.set()
        assert model.finished.wait(WAIT)
        assert answers_ended()

    assert isinstance(model.interrupted, CallerGone)
    assert model.written == []
    assert slots_in_use(application) == 0
