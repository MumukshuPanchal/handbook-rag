import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore
from tests.support import FakeEmbedder, make_chunk

CHUNKS = [
    make_chunk("Parental leave is sixteen weeks.", heading_path=("Parental leave",)),
    make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md"),
    make_chunk(
        "Report a security incident in the incident channel.", doc_id="security/incidents.md"
    ),
]


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(":memory:"), "test-collection")


def test_first_run_embeds_and_stores_every_chunk(store: QdrantVectorStore) -> None:
    embedder = FakeEmbedder()

    report = index_chunks(CHUNKS, embedder=embedder, store=store)

    assert (report.total, report.embedded, report.already_indexed, report.removed) == (3, 3, 0, 0)
    assert store.count() == 3
    assert len(embedder.embedded) == 3


def test_chunks_are_embedded_with_their_context_header(store: QdrantVectorStore) -> None:
    embedder = FakeEmbedder()

    index_chunks(CHUNKS, embedder=embedder, store=store)

    assert "Leave > Parental leave\n\nParental leave is sixteen weeks." in embedder.embedded


def test_second_run_does_no_embedding(store: QdrantVectorStore) -> None:
    embedder = FakeEmbedder()
    index_chunks(CHUNKS, embedder=embedder, store=store)
    embedder.embedded.clear()

    report = index_chunks(CHUNKS, embedder=embedder, store=store)

    assert (report.embedded, report.already_indexed, report.removed) == (0, 3, 0)
    assert embedder.embedded == []
    assert store.count() == 3


def test_only_new_chunks_are_embedded_and_stale_ones_are_removed(
    store: QdrantVectorStore,
) -> None:
    embedder = FakeEmbedder()
    index_chunks(CHUNKS, embedder=embedder, store=store)
    embedder.embedded.clear()
    new_chunk = make_chunk("Book travel through the travel portal.", doc_id="finance/travel.md")

    report = index_chunks([CHUNKS[0], new_chunk], embedder=embedder, store=store)

    assert (report.total, report.embedded, report.already_indexed, report.removed) == (2, 1, 1, 2)
    assert embedder.embedded == [new_chunk.text_for_embedding]
    assert store.existing_ids([c.chunk_id for c in CHUNKS]) == {CHUNKS[0].chunk_id}


def test_progress_is_reported_until_done(store: QdrantVectorStore) -> None:
    progress: list[tuple[int, int]] = []

    index_chunks(
        CHUNKS,
        embedder=FakeEmbedder(),
        store=store,
        batch_size=2,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(2, 3), (3, 3)]
