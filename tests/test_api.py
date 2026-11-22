from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typer.testing import CliRunner

from handbook_rag import __version__, wiring
from handbook_rag.api import (
    EMPTY,
    MAX_BODY_BYTES,
    MAX_LIMIT,
    MAX_QUERY_CHARS,
    NOT_BUILT,
    TRACE_HEADER,
    UNAVAILABLE,
    UNREACHABLE,
    WARM_UP_QUERY,
    _is_health_check,
    create_app,
)
from handbook_rag.cli import app as cli
from handbook_rag.config import Settings
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import (
    QdrantVectorStore,
    ScoredChunk,
    VectorStore,
    VectorStoreError,
)
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

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
    assert _is_health_check({"type": "http", "path": "/search"}) is False


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
    assert response.json() == {"status": "ok", "version": __version__, "chunks": 2, "detail": None}


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
        {"status": "unavailable", "version": __version__, "chunks": 0, "detail": EMPTY},
    )
    assert health_of(never_built) == (
        503,
        {"status": "unavailable", "version": __version__, "chunks": None, "detail": NOT_BUILT},
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
    assert set(schema["paths"]) == {"/health", "/search"}
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
