import dataclasses
from typing import Any

import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

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

    report = index_chunks(
        CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store
    )

    assert (report.total, report.embedded, report.already_indexed, report.removed) == (3, 3, 0, 0)
    assert store.count() == 3
    assert len(embedder.embedded) == 3


def test_chunks_are_embedded_with_their_context_header(store: QdrantVectorStore) -> None:
    embedder = FakeEmbedder()

    index_chunks(CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store)

    assert "Leave > Parental leave\n\nParental leave is sixteen weeks." in embedder.embedded


def test_second_run_does_no_embedding(store: QdrantVectorStore) -> None:
    embedder = FakeEmbedder()
    index_chunks(CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store)
    embedder.embedded.clear()

    report = index_chunks(
        CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store
    )

    assert (report.embedded, report.already_indexed, report.removed) == (0, 3, 0)
    assert embedder.embedded == []
    assert store.count() == 3


def test_only_new_chunks_are_embedded_and_stale_ones_are_removed(
    store: QdrantVectorStore,
) -> None:
    embedder = FakeEmbedder()
    index_chunks(CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store)
    embedder.embedded.clear()
    new_chunk = make_chunk("Book travel through the travel portal.", doc_id="finance/travel.md")

    report = index_chunks(
        [CHUNKS[0], new_chunk], embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store
    )

    assert (report.total, report.embedded, report.already_indexed, report.removed) == (2, 1, 1, 2)
    assert embedder.embedded == [new_chunk.text_for_embedding]
    assert set(store.point_ids()) == {CHUNKS[0].chunk_id, new_chunk.chunk_id}


def test_progress_is_reported_until_done(store: QdrantVectorStore) -> None:
    progress: list[tuple[int, int]] = []

    index_chunks(
        CHUNKS,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        store=store,
        batch_size=2,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(2, 3), (3, 3)]


def test_chunks_are_stored_with_keyword_vectors_too(store: QdrantVectorStore) -> None:
    sparse = FakeSparseEncoder()
    index_chunks(CHUNKS, embedder=FakeEmbedder(), sparse_encoder=sparse, store=store)

    results = store.search_sparse(sparse.encode_query("laptop"), limit=3)

    assert [r.chunk.doc_id for r in results] == ["finance/expenses.md"]


def test_a_chunk_stored_under_its_id_with_another_title_is_embedded_again(
    store: QdrantVectorStore,
) -> None:
    # Within one collection the id stays when only the title changes, for example when
    # the way titles are read is changed. Skipping by id alone would keep the old one.
    embedder = FakeEmbedder()
    index_chunks(CHUNKS, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store)
    embedder.embedded.clear()
    retitled = dataclasses.replace(CHUNKS[0], title="Time off")

    report = index_chunks(
        [retitled, *CHUNKS[1:]], embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store
    )

    assert retitled.chunk_id == CHUNKS[0].chunk_id
    assert (report.embedded, report.already_indexed, report.replaced, report.removed) == (
        1,
        2,
        1,
        0,
    )
    assert embedder.embedded == [retitled.text_for_embedding]
    assert store.stored_chunks([retitled.chunk_id]) == {retitled.chunk_id: retitled}


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://handbook.example.org/handbook/people/leave/"},
        {"section": "benefits"},
        {"heading_path": ("Leave",)},
        {"source_commit": "b" * 40},
        {"token_count": 99},
    ],
    ids=lambda change: next(iter(change)),
)
def test_a_chunk_stored_with_any_other_detail_is_stored_again(
    store: QdrantVectorStore, change: dict[str, Any]
) -> None:
    index_chunks(CHUNKS, embedder=FakeEmbedder(), sparse_encoder=FakeSparseEncoder(), store=store)
    changed = dataclasses.replace(CHUNKS[0], **change)

    report = index_chunks(
        [changed, *CHUNKS[1:]],
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        store=store,
    )

    assert (report.embedded, report.replaced, report.already_indexed) == (1, 1, 2)
    assert store.stored_chunks([changed.chunk_id]) == {changed.chunk_id: changed}


def test_a_run_without_reuse_reuses_nothing_and_times_its_embedding(
    store: QdrantVectorStore,
) -> None:
    report = index_chunks(
        CHUNKS, embedder=FakeEmbedder(), sparse_encoder=FakeSparseEncoder(), store=store
    )

    assert (report.reused, report.replaced) == (0, 0)
    assert 0 < report.embed_seconds <= report.seconds
    assert report.reuse_seconds < report.embed_seconds
