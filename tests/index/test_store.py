import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import QdrantVectorStore, VectorStoreError
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
