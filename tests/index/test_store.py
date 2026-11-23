import dataclasses
from typing import Any, cast

import httpx
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import CollectionMissingError, QdrantVectorStore, VectorStoreError
from tests.support import make_chunk

NORTH = [1.0, 0.0, 0.0]
EAST = [0.0, 1.0, 0.0]
MOSTLY_NORTH = [0.9, 0.1, 0.0]
NO_WORDS = SparseVector([], [])
LAPTOP = SparseVector([1, 2], [1.0, 1.0])
LEAVE = SparseVector([3, 4], [1.0, 1.0])


@pytest.fixture
def client() -> QdrantClient:
    """Qdrant's embedded in-memory mode: the same API as the server, with no server."""
    return QdrantClient(":memory:")


@pytest.fixture
def store(client: QdrantClient) -> QdrantVectorStore:
    store = QdrantVectorStore(client, "test-collection")
    store.ensure_collection(dimension=3)
    return store


def test_new_collection_is_empty(store: QdrantVectorStore) -> None:
    assert store.count() == 0


def test_search_returns_the_closest_chunks_first(store: QdrantVectorStore) -> None:
    north, east, mostly_north = make_chunk("north"), make_chunk("east"), make_chunk("mostly north")
    store.upsert([north, east, mostly_north], [NORTH, EAST, MOSTLY_NORTH], [NO_WORDS] * 3)

    results = store.search(NORTH, limit=2)

    assert [r.chunk.text for r in results] == ["north", "mostly north"]
    assert results[0].score > results[1].score


def test_stored_chunk_comes_back_unchanged(store: QdrantVectorStore) -> None:
    original = make_chunk("Take 25 days.", heading_path=("Leave", "Annual leave"))
    store.upsert([original], [NORTH], [NO_WORDS])

    (result,) = store.search(NORTH, limit=1)

    assert result.chunk == original
    assert isinstance(result.chunk.heading_path, tuple)


def test_search_can_be_limited_to_one_section(store: QdrantVectorStore) -> None:
    people = make_chunk("leave policy", section="people")
    finance = make_chunk("expense policy", section="finance", doc_id="finance/expenses.md")
    store.upsert([people, finance], [NORTH, MOSTLY_NORTH], [LEAVE, LAPTOP])

    results = store.search(NORTH, limit=5, section="finance")

    assert [r.chunk.section for r in results] == ["finance"]


def test_upserting_the_same_id_replaces_the_chunk(store: QdrantVectorStore) -> None:
    chunk = make_chunk("same content")
    store.upsert([chunk], [NORTH], [NO_WORDS])
    store.upsert([chunk], [EAST], [NO_WORDS])

    assert store.count() == 1
    assert store.search(EAST, limit=1)[0].score == pytest.approx(1.0)


def test_existing_ids_reports_only_stored_chunks(store: QdrantVectorStore) -> None:
    stored, missing = make_chunk("stored"), make_chunk("missing")
    store.upsert([stored], [NORTH], [NO_WORDS])

    assert store.existing_ids([stored.chunk_id, missing.chunk_id]) == {stored.chunk_id}


def test_delete_except_removes_everything_not_listed(store: QdrantVectorStore) -> None:
    keep, drop = make_chunk("keep"), make_chunk("drop")
    store.upsert([keep, drop], [NORTH, EAST], [NO_WORDS] * 2)

    removed = store.delete_except({keep.chunk_id})

    assert removed == 1
    assert store.existing_ids([keep.chunk_id, drop.chunk_id]) == {keep.chunk_id}


def test_ensure_collection_is_safe_to_repeat(store: QdrantVectorStore) -> None:
    store.upsert([make_chunk("kept")], [NORTH], [NO_WORDS])

    store.ensure_collection(dimension=3)

    assert store.count() == 1


def test_collection_built_for_another_model_is_rejected(client: QdrantClient) -> None:
    QdrantVectorStore(client, "shared").ensure_collection(dimension=3)

    with pytest.raises(VectorStoreError, match="size 3"):
        QdrantVectorStore(client, "shared").ensure_collection(dimension=8)


def test_every_chunk_needs_a_vector(store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="exactly one dense and one sparse"):
        store.upsert([make_chunk("a"), make_chunk("b")], [NORTH], [NO_WORDS, NO_WORDS])
    with pytest.raises(ValueError, match="exactly one dense and one sparse"):
        store.upsert([make_chunk("a")], [NORTH], [])


def test_searching_a_collection_that_was_never_built_says_how_to_build_it(
    client: QdrantClient,
) -> None:
    with pytest.raises(VectorStoreError, match="handbook-rag index"):
        QdrantVectorStore(client, "never-built").search(NORTH, limit=1)


def test_sparse_search_matches_shared_words_and_ignores_the_dense_vector(
    store: QdrantVectorStore,
) -> None:
    laptop = make_chunk("laptop expense", doc_id="finance/expenses.md", section="finance")
    leave = make_chunk("parental leave")
    store.upsert([laptop, leave], [NORTH, NORTH], [LAPTOP, LEAVE])

    results = store.search_sparse(SparseVector([1], [1.0]), limit=5)

    assert [r.chunk.text for r in results] == ["laptop expense"]


def test_sparse_search_ranks_more_matching_words_higher(store: QdrantVectorStore) -> None:
    one_word = make_chunk("laptop only")
    two_words = make_chunk("laptop expense")
    store.upsert([one_word, two_words], [NORTH, NORTH], [SparseVector([1], [1.0]), LAPTOP])

    results = store.search_sparse(LAPTOP, limit=5)

    assert [r.chunk.text for r in results] == ["laptop expense", "laptop only"]


def test_sparse_search_can_be_limited_to_one_section(store: QdrantVectorStore) -> None:
    finance = make_chunk("laptop expense", doc_id="finance/expenses.md", section="finance")
    people = make_chunk("laptop loan")
    store.upsert([finance, people], [NORTH, NORTH], [LAPTOP, LAPTOP])

    results = store.search_sparse(LAPTOP, limit=5, section="people")

    assert [r.chunk.section for r in results] == ["people"]


class CountingClient:
    """Passes every call on to a real client and counts them by name."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return attribute(*args, **kwargs)

        return counted


def test_a_search_asks_the_store_once(client: QdrantClient) -> None:
    QdrantVectorStore(client, "test-collection").ensure_collection(dimension=3)
    counting = CountingClient(client)
    store = QdrantVectorStore(cast(QdrantClient, counting), "test-collection")
    store.upsert([make_chunk("north")], [NORTH], [LAPTOP])
    counting.calls.clear()

    store.search(NORTH, limit=1)
    store.search_sparse(LAPTOP, limit=1)

    assert counting.calls == ["query_points", "query_points"]


def bad_response(status: int, body: bytes = b"") -> UnexpectedResponse:
    """An answer from the store with an error status, as its client raises it."""
    return UnexpectedResponse(status, "reason", body, httpx.Headers())


def test_a_store_that_does_not_answer_is_reported_as_unavailable_and_not_asked_again(
    client: QdrantClient,
) -> None:
    QdrantVectorStore(client, "test-collection").ensure_collection(dimension=3)
    counting = CountingClient(client)
    store = QdrantVectorStore(cast(QdrantClient, counting), "test-collection")

    def no_answer(*_args: Any, **_kwargs: Any) -> None:
        counting.calls.append("query_points")
        raise ResponseHandlingException(ConnectionError("connection reset"))

    counting.query_points = no_answer  # type: ignore[attr-defined]

    with pytest.raises(VectorStoreError, match="did not answer") as raised:
        store.search(NORTH, limit=1)

    assert "ConnectionError" in str(raised.value)
    assert "connection reset" in str(raised.value)
    assert isinstance(raised.value.__cause__, ResponseHandlingException)
    assert not isinstance(raised.value, CollectionMissingError)
    # A store that hangs must not be waited for twice.
    assert counting.calls == ["query_points"]


def test_a_store_that_answers_with_a_server_error_is_reported_as_unavailable(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_args: Any, **_kwargs: Any) -> None:
        raise bad_response(503)

    monkeypatch.setattr(client, "query_points", failing)

    with pytest.raises(VectorStoreError, match="failed with status 503"):
        store.search_sparse(LAPTOP, limit=1)


def test_a_request_the_store_rejects_is_a_bug_and_is_not_dressed_up_as_an_outage(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = bad_response(400, b"Vector dimension error: expected dim: 384, got 64")

    def wrong_request(*_args: Any, **_kwargs: Any) -> None:
        raise rejected

    monkeypatch.setattr(client, "query_points", wrong_request)

    with pytest.raises(UnexpectedResponse) as raised:
        store.search(NORTH, limit=1)

    assert raised.value is rejected


def test_a_missing_collection_is_its_own_kind_of_error_on_a_server_too(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_found(*_args: Any, **_kwargs: Any) -> None:
        raise bad_response(404, b"Collection `test-collection` doesn't exist!")

    monkeypatch.setattr(client, "query_points", not_found)
    monkeypatch.setattr(client, "count", not_found)

    with pytest.raises(CollectionMissingError, match="handbook-rag index"):
        store.search(NORTH, limit=1)
    with pytest.raises(CollectionMissingError, match="does not exist yet"):
        store.count()


def test_counting_a_collection_that_was_never_built_says_so(client: QdrantClient) -> None:
    never_built = QdrantVectorStore(client, "never-built")

    with pytest.raises(CollectionMissingError, match="never-built does not exist yet"):
        never_built.count()


def test_an_error_that_is_not_the_stores_doing_is_passed_on_unchanged(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def first(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("the first failure")

    def second(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("the second failure")

    monkeypatch.setattr(client, "query_points", first)
    with pytest.raises(TypeError, match="the first failure"):
        store.search(NORTH, limit=1)

    # Even when asking whether the collection exists fails too, the first error is the
    # one that is reported.
    monkeypatch.setattr(client, "collection_exists", second)
    with pytest.raises(TypeError, match="the first failure"):
        store.search(NORTH, limit=1)


def test_stored_chunks_come_back_by_id_and_an_unknown_id_is_left_out(
    store: QdrantVectorStore,
) -> None:
    stored = make_chunk("Take 25 days.", heading_path=("Leave", "Annual leave"))
    store.upsert([stored], [NORTH], [LEAVE])

    found = store.stored_chunks([stored.chunk_id, make_chunk("never stored").chunk_id])

    assert found == {stored.chunk_id: stored}


def test_points_come_back_with_their_payload_and_both_vectors(store: QdrantVectorStore) -> None:
    chunk = make_chunk("laptop expense")
    store.upsert([chunk], [MOSTLY_NORTH], [SparseVector([7, 3], [0.5, 2.0])])

    (point,) = store.points([chunk.chunk_id, make_chunk("never stored").chunk_id]).values()

    assert point.point_id == chunk.chunk_id
    assert point.chunk == chunk
    assert point.payload["text"] == "laptop expense"
    # The store keeps a dense vector at length 1, and a keyword vector in word order.
    length = (0.9**2 + 0.1**2) ** 0.5
    assert point.dense == pytest.approx([0.9 / length, 0.1 / length, 0.0], abs=1e-6)
    assert point.sparse == SparseVector([3, 7], [2.0, 0.5])


def test_a_point_without_words_comes_back_with_an_empty_keyword_vector(
    store: QdrantVectorStore,
) -> None:
    chunk = make_chunk("the of a")
    store.upsert([chunk], [NORTH], [NO_WORDS])

    assert store.points([chunk.chunk_id])[chunk.chunk_id].sparse == NO_WORDS


def record(vector: Any) -> models.Record:
    chunk = make_chunk("north")
    payload = {key: value for key, value in dataclasses.asdict(chunk).items() if key != "chunk_id"}
    return models.Record(id=chunk.chunk_id, payload=payload, vector=vector)


def test_a_keyword_vector_the_store_leaves_out_is_read_as_empty(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "retrieve", lambda *_a, **_k: [record({"dense": NORTH})])

    (point,) = store.points([make_chunk("north").chunk_id]).values()

    assert (point.dense, point.sparse) == (NORTH, NO_WORDS)


@pytest.mark.parametrize(
    "vector", [None, NORTH, {"sparse": models.SparseVector(indices=[], values=[])}]
)
def test_a_point_without_a_named_dense_vector_is_reported(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch, vector: Any
) -> None:
    monkeypatch.setattr(client, "retrieve", lambda *_a, **_k: [record(vector)])

    with pytest.raises(VectorStoreError, match="has no dense vector named 'dense'"):
        store.points([make_chunk("north").chunk_id])


@pytest.fixture
def small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read two points at a time, so that reading has to go on from batch to batch."""
    monkeypatch.setattr("handbook_rag.index.store._BATCH", 2)


@pytest.mark.usefixtures("small_batches")
def test_every_point_and_chunk_is_read_across_batches(store: QdrantVectorStore) -> None:
    chunks = [make_chunk(f"chunk {n}", doc_id=f"page{n % 2}.md") for n in range(5)]
    store.upsert(chunks, [NORTH] * 5, [NO_WORDS] * 5)
    ids = [chunk.chunk_id for chunk in chunks]

    assert sorted(store.point_ids()) == sorted(ids)
    assert sorted(store.scroll_chunks(), key=lambda c: c.text) == chunks
    assert set(store.stored_chunks(ids)) == set(ids)
    assert set(store.points(ids)) == set(ids)


@pytest.mark.usefixtures("small_batches")
def test_reading_chunks_can_be_limited_to_some_pages(store: QdrantVectorStore) -> None:
    chunks = [make_chunk(f"chunk {n}", doc_id=f"page{n % 3}.md") for n in range(6)]
    store.upsert(chunks, [NORTH] * 6, [NO_WORDS] * 6)

    some = list(store.scroll_chunks({"page0.md", "page2.md"}))

    assert sorted(c.text for c in some) == ["chunk 0", "chunk 2", "chunk 3", "chunk 5"]
    assert list(store.scroll_chunks(set())) == []


def test_another_collection_is_reached_through_the_same_connection(
    store: QdrantVectorStore, client: QdrantClient
) -> None:
    other = store.with_collection("other-collection")

    assert other.collection == "other-collection"
    assert not other.exists()
    other.ensure_collection(dimension=3)
    assert other.exists()
    assert store.collection_names() == ["other-collection", "test-collection"]
    assert QdrantVectorStore(client, "other-collection").count() == 0


def test_reading_the_chunks_of_no_page_asks_the_store_nothing(
    store: QdrantVectorStore, client: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A filter that matches any of no values is not something every store accepts.
    def no_call(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the store was asked")

    monkeypatch.setattr(client, "scroll", no_call)

    assert list(store.scroll_chunks(set())) == []
