"""Updating the index to a newer commit by reusing the vectors of the older one.

The two commits below are small handbooks made to hold every case an update has to get
right. The first test checks that they do, so a change to the chunker that quietly
removed a case would show up there and not as a test that passes for the wrong reason.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from qdrant_client import QdrantClient

from handbook_rag.config import Settings
from handbook_rag.index.compare import compare_collections
from handbook_rag.index.indexer import IndexReport, Reuse, index_chunks, text_key
from handbook_rag.index.reuse import (
    PageChanges,
    PreviousIndex,
    ReuseError,
    compare_pages,
    find_reuse_source,
    read_previous_index,
)
from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import QdrantVectorStore, ReadableStore, VectorStoreError
from handbook_rag.ingest.chunker import ChunkingConfig, chunk_document
from handbook_rag.ingest.loader import load_documents
from handbook_rag.ingest.models import Chunk
from tests.support import ExactEmbedder, FakeSparseEncoder, count_words, make_chunk

OLD_COMMIT, NEW_COMMIT = "a" * 40, "b" * 40
# Small enough that a short page is cut into several chunks and a table into pieces.
CONFIG = ChunkingConfig(max_tokens=30, min_tokens=4, overlap_tokens=0)

TRAVEL = (
    "---\ntitle: Trips\n---\n\n## Travel\n\n"
    "Book every trip through the travel portal so that finance can see it.\n\n"
    "Choose the cheapest direct flight that arrives on the day you need to be there.\n\n"
    "Keep every receipt and attach it to your expense report within thirty days.\n"
)
WORKFLOW = (
    "---\ntitle: Workflow\n---\n\nThis page describes how we work.\n\n"
    "## Reviews\n\nEvery merge request needs one approval from a maintainer.\n\n"
    "## Releases\n\nWe release on the third Thursday of every month.\n"
)
LIMITS = (
    "---\ntitle: Limits\n---\n\n## Spending limits\n\n| Item | Limit |\n| --- | --- |\n"
    + "".join(f"| Item {n} | {n}00 dollars |\n" for n in range(1, 9))
)
RATES = (
    "---\ntitle: Rates\n---\n\nRates are reviewed every year.\n\n"
    "| Country | Rate |\n| --- | --- |\n"
    + "".join(f"| Country {n} | {n}0 dollars a day |\n" for n in range(1, 9))
)
OVERVIEW = (
    "---\ntitle: Overview\n---\n\n"
    "This area of the handbook collects the pages that describe how the team works.\n"
)
OLD_PAGES = {
    "people/leave.md": (
        "---\ntitle: Leave\n---\n\n## Parental leave\n\nNew parents get sixteen weeks of paid "
        "leave.\n\n## Sick leave\n\nTake the time you need and tell your manager.\n"
    ),
    "finance/expenses.md": (
        "---\ntitle: Expenses\n---\n\n## Laptops\n\nExpense a laptop up to one thousand dollars.\n"
    ),
    "people/travel.md": TRAVEL,
    "engineering/workflow.md": WORKFLOW,
    "finance/limits.md": LIMITS,
    "finance/rates.md": RATES,
    "a/_index.md": OVERVIEW,
    "b/_index.md": OVERVIEW,
    "legal/retired.md": (
        "---\ntitle: Old policy\n---\n\nThis policy was retired and replaced by the new one.\n"
    ),
}
NEW_PAGES = {
    **{page: text for page, text in OLD_PAGES.items() if page != "legal/retired.md"},
    "finance/expenses.md": OLD_PAGES["finance/expenses.md"].replace("Expenses", "Spending"),
    "people/travel.md": TRAVEL.replace("## Travel", "## Voyage"),
    "engineering/workflow.md": WORKFLOW.replace("how we work.", "how we work, and who decides."),
    "finance/limits.md": LIMITS.replace("| Item | Limit |", "| Item | Bound |"),
    "finance/rates.md": RATES.replace("every year.", "every year by finance."),
    "c/_index.md": OVERVIEW,
    "security/incidents.md": (
        "---\ntitle: Incidents\n---\n\nReport a security incident in the incident channel.\n"
    ),
}


def chunk_pages(root: Path, pages: dict[str, str], commit: str) -> list[Chunk]:
    """Load and chunk a small handbook as indexing would, with a word for a token."""
    for relative, text in pages.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    documents = load_documents(
        root, source_commit=commit, base_url="https://handbook.example.com/handbook/", min_chars=20
    ).documents
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, count_tokens=count_words, config=CONFIG)
    ]


def on_page(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.doc_id == doc_id]


def build(
    store: QdrantVectorStore, chunks: list[Chunk], reuse: Reuse | None = None
) -> tuple[IndexReport, ExactEmbedder]:
    embedder = ExactEmbedder()
    report = index_chunks(
        chunks, embedder=embedder, sparse_encoder=FakeSparseEncoder(), store=store, reuse=reuse
    )
    return report, embedder


@dataclass(frozen=True)
class Update:
    previous: PreviousIndex
    changes: PageChanges
    report: IndexReport
    embedder: ExactEmbedder


def update(old: ReadableStore, store: QdrantVectorStore, chunks: list[Chunk]) -> Update:
    """Index ``chunks`` into ``store``, taking what vectors it can from ``old``."""
    previous = read_previous_index(old)
    report, embedder = build(store, chunks, Reuse(old, previous.points))
    return Update(previous, compare_pages(previous, chunks), report, embedder)


@dataclass(frozen=True)
class Commits:
    old: list[Chunk]
    new: list[Chunk]
    old_store: QdrantVectorStore
    full: QdrantVectorStore
    updated: QdrantVectorStore
    update: Update


@pytest.fixture
def commits(tmp_path: Path) -> Commits:
    """The older commit indexed, and the newer one indexed twice: in full, and by update."""
    client = QdrantClient(":memory:")
    old = chunk_pages(tmp_path / "old", OLD_PAGES, OLD_COMMIT)
    new = chunk_pages(tmp_path / "new", NEW_PAGES, NEW_COMMIT)
    old_store = QdrantVectorStore(client, "handbook-old")
    build(old_store, old)
    full = QdrantVectorStore(client, "handbook-full")
    build(full, new)
    updated = QdrantVectorStore(client, "handbook-updated")
    return Commits(old, new, old_store, full, updated, update(old_store, updated, new))


def test_the_two_commits_hold_every_case_an_update_has_to_get_right(commits: Commits) -> None:
    old_ids = {chunk.chunk_id for chunk in commits.old}
    old_texts = {chunk.text_for_embedding for chunk in commits.old}

    # A title changed in the front matter: the id stays, the embedded text does not.
    (retitled,) = on_page(commits.new, "finance/expenses.md")
    assert retitled.chunk_id in old_ids
    assert retitled.text_for_embedding not in old_texts
    # A heading renamed to a word of the same length: the chunk under it that does not
    # hold the heading's line keeps its id, and its embedded text changes.
    renamed = on_page(commits.new, "people/travel.md")[1]
    assert renamed.heading_path == ("Voyage",)
    assert renamed.chunk_id in old_ids
    assert renamed.text_for_embedding not in old_texts
    # Edited at the top: everything below moved, so it has new ids and the same text.
    moved = on_page(commits.new, "engineering/workflow.md")[1:]
    assert len(moved) == 2
    assert all(chunk.chunk_id not in old_ids for chunk in moved)
    assert all(chunk.text_for_embedding in old_texts for chunk in moved)
    # A table whose header was renamed: a later piece covers the same span of the page
    # as before, and repeats the new header, so its text is not that span.
    new_piece = on_page(commits.new, "finance/limits.md")[1]
    old_piece = on_page(commits.old, "finance/limits.md")[1]
    assert (new_piece.start_char, new_piece.end_char) == (old_piece.start_char, old_piece.end_char)
    assert new_piece.text.startswith("| Item | Bound |\n")
    assert old_piece.text.startswith("| Item | Limit |\n")
    # A table that only moved: its later pieces repeat the header and have new ids.
    pieces = on_page(commits.new, "finance/rates.md")[1:]
    assert len(pieces) == 4
    assert all(piece.text.startswith("| Country | Rate |\n") for piece in pieces)
    assert all(piece.chunk_id not in old_ids for piece in pieces)
    assert all(piece.text_for_embedding in old_texts for piece in pieces)
    # The same embedded text on two pages, and on a third page that is new.
    shared = {on_page(commits.new, f"{area}/_index.md")[0].text_for_embedding for area in "abc"}
    assert len(shared) == 1
    # A page added, a page removed, and a page that did not change.
    assert not on_page(commits.old, "security/incidents.md")
    assert not on_page(commits.new, "legal/retired.md")
    assert [c.text_for_embedding for c in on_page(commits.new, "people/leave.md")] == [
        c.text_for_embedding for c in on_page(commits.old, "people/leave.md")
    ]


def test_an_updated_index_equals_a_full_index_of_the_newer_commit(commits: Commits) -> None:
    comparison = compare_collections(commits.full, commits.updated, tolerance=0.0)

    assert comparison.differences() == []
    assert comparison.compared == len(commits.new)
    assert comparison.largest_dense_difference == 0.0
    assert comparison.largest_sparse_difference == 0.0


def test_only_chunks_whose_embedded_text_is_new_are_embedded(commits: Commits) -> None:
    old_texts = {chunk.text_for_embedding for chunk in commits.old}
    new_texts = sorted(
        chunk.text_for_embedding
        for chunk in commits.new
        if chunk.text_for_embedding not in old_texts
    )

    assert sorted(commits.update.embedder.embedded) == new_texts
    assert commits.update.report.embedded == len(new_texts) == 10
    assert commits.update.report.reused == len(commits.new) - 10


def stored_vector(store: QdrantVectorStore, chunk: Chunk) -> list[float]:
    return store.points([chunk.chunk_id])[chunk.chunk_id].dense


def test_a_chunk_that_only_moved_takes_the_vectors_of_its_older_self(commits: Commits) -> None:
    moved = on_page(commits.new, "engineering/workflow.md")[1]
    (before,) = [c for c in commits.old if c.text_for_embedding == moved.text_for_embedding]

    assert moved.text_for_embedding not in commits.update.embedder.embedded
    assert stored_vector(commits.updated, moved) == stored_vector(commits.old_store, before)


@pytest.mark.parametrize(
    ("doc_id", "position"),
    [
        pytest.param("finance/expenses.md", 0, id="title changed in the front matter"),
        pytest.param("people/travel.md", 1, id="heading renamed to a word of the same length"),
        pytest.param("finance/limits.md", 1, id="table header renamed above a later piece"),
    ],
)
def test_a_chunk_whose_embedded_text_changed_is_embedded_again(
    commits: Commits, doc_id: str, position: int
) -> None:
    changed = on_page(commits.new, doc_id)[position]

    assert changed.text_for_embedding in commits.update.embedder.embedded
    assert stored_vector(commits.updated, changed) == ExactEmbedder().embed_query(
        changed.text_for_embedding
    )


def test_the_same_text_on_two_pages_lends_its_vectors_to_a_third(commits: Commits) -> None:
    copies = [on_page(commits.new, f"{area}/_index.md")[0] for area in "abc"]
    vector = ExactEmbedder().embed_query(copies[0].text_for_embedding)

    assert copies[0].text_for_embedding not in commits.update.embedder.embedded
    assert [stored_vector(commits.updated, copy) for copy in copies] == [vector] * 3
    # One of the two older copies lends its vectors to all three. Either would do; the
    # first one read is kept, and the store reads in the order of ids, so every run of
    # the same update takes the same one.
    lender = commits.update.previous.points[text_key(copies[0].text_for_embedding)]
    older_copies = on_page(commits.old, "a/_index.md") + on_page(commits.old, "b/_index.md")
    assert lender == min(copy.chunk_id for copy in older_copies)


def test_every_payload_comes_from_the_newer_commit(commits: Commits) -> None:
    stored = commits.updated.stored_chunks([chunk.chunk_id for chunk in commits.new])

    assert stored == {chunk.chunk_id: chunk for chunk in commits.new}
    assert {chunk.source_commit for chunk in stored.values()} == {NEW_COMMIT}


def test_a_page_that_moved_to_another_address_keeps_its_vectors_and_takes_its_new_payload(
    tmp_path: Path,
) -> None:
    client = QdrantClient(":memory:")
    old = chunk_pages(
        tmp_path / "old", {"people/leave.md": OLD_PAGES["people/leave.md"]}, OLD_COMMIT
    )
    new = chunk_pages(
        tmp_path / "new", {"benefits/leave.md": OLD_PAGES["people/leave.md"]}, NEW_COMMIT
    )
    old_store = QdrantVectorStore(client, "old")
    build(old_store, old)

    done = update(old_store, QdrantVectorStore(client, "new"), new)

    assert (done.report.reused, done.report.embedded) == (2, 0)
    stored = QdrantVectorStore(client, "new").stored_chunks([c.chunk_id for c in new])
    assert {c.doc_id for c in stored.values()} == {"benefits/leave.md"}
    assert {c.section for c in stored.values()} == {"benefits"}
    assert {c.url for c in stored.values()} == {
        "https://handbook.example.com/handbook/benefits/leave/"
    }
    assert done.changes == PageChanges(
        added=["benefits/leave.md"], changed=[], removed=["people/leave.md"], unchanged=[]
    )


def test_pages_are_counted_as_added_changed_removed_or_unchanged(commits: Commits) -> None:
    assert commits.update.changes == PageChanges(
        added=["c/_index.md", "security/incidents.md"],
        changed=[
            "engineering/workflow.md",
            "finance/expenses.md",
            "finance/limits.md",
            "finance/rates.md",
            "people/travel.md",
        ],
        removed=["legal/retired.md"],
        unchanged=["a/_index.md", "b/_index.md", "people/leave.md"],
    )


def test_the_report_counts_add_up(commits: Commits) -> None:
    report, changes = commits.update.report, commits.update.changes
    old_pages = {chunk.doc_id for chunk in commits.old}
    new_pages = {chunk.doc_id for chunk in commits.new}

    assert report.total == len(commits.new) == 21
    assert report.reused + report.embedded + report.already_indexed == report.total
    assert (report.reused, report.embedded, report.already_indexed) == (11, 10, 0)
    assert (report.removed, report.replaced) == (0, 0)
    assert len(changes.added) + len(changes.changed) + len(changes.unchanged) == len(new_pages)
    assert len(changes.removed) + len(changes.changed) + len(changes.unchanged) == len(old_pages)
    assert commits.update.previous.chunks == len(commits.old)
    assert set(commits.update.previous.pages) == old_pages


def test_nothing_is_embedded_when_only_the_commit_changed(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    old = chunk_pages(tmp_path / "old", OLD_PAGES, OLD_COMMIT)
    new = chunk_pages(tmp_path / "new", OLD_PAGES, NEW_COMMIT)
    old_store = QdrantVectorStore(client, "old")
    build(old_store, old)

    done = update(old_store, QdrantVectorStore(client, "new"), new)

    assert done.embedder.embedded == []
    assert (done.report.reused, done.report.embedded) == (len(new), 0)
    assert done.changes.unchanged == sorted(OLD_PAGES)
    assert done.changes.added == done.changes.changed == done.changes.removed == []


class ReadOnlyClient:
    """Passes reading calls on to a real client, and fails the test on anything else."""

    READS = frozenset({"scroll", "retrieve", "count", "collection_exists", "get_collection"})

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name not in self.READS:
            raise AssertionError(f"the older collection was asked to {name}")
        self.calls.append(name)
        return getattr(self._client, name)


def every_point(store: QdrantVectorStore) -> dict[str, Any]:
    points = store.points(sorted(store.point_ids()))
    return {point_id: dataclasses.asdict(point) for point_id, point in points.items()}


def test_the_older_collection_is_only_read(commits: Commits) -> None:
    client = QdrantClient(":memory:")
    old_store = QdrantVectorStore(client, "old")
    build(old_store, commits.old)
    before = every_point(old_store)
    guarded = ReadOnlyClient(client)
    read_only = QdrantVectorStore(cast(QdrantClient, guarded), "old")

    done = update(read_only, QdrantVectorStore(client, "new"), commits.new)

    assert done.report.reused == 11
    assert set(guarded.calls) == {"scroll", "retrieve"}
    assert every_point(old_store) == before


def test_a_point_is_never_reused_on_its_id_alone(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    (chunk,) = chunk_pages(
        tmp_path, {"finance/expenses.md": NEW_PAGES["finance/expenses.md"]}, NEW_COMMIT
    )
    # An older point stored under the very same id, embedded from another title.
    older = dataclasses.replace(chunk, title="Expenses", source_commit=OLD_COMMIT)
    old_store = QdrantVectorStore(client, "old")
    build(old_store, [older])
    assert older.chunk_id == chunk.chunk_id

    done = update(old_store, QdrantVectorStore(client, "new"), [chunk])

    assert (done.report.reused, done.report.embedded) == (0, 1)
    assert done.embedder.embedded == [chunk.text_for_embedding]
    assert stored_vector(QdrantVectorStore(client, "new"), chunk) == ExactEmbedder().embed_query(
        chunk.text_for_embedding
    )


def leave_page_read_then_changed(
    tmp_path: Path, change: str
) -> tuple[list[Chunk], IndexReport, ExactEmbedder]:
    """The leave page indexed and read, then changed by someone else, then updated from.

    Its first chunk's point is left alone. Its second is given another title or other
    headings, or deleted, after the older collection was read and before its vectors
    are fetched.
    """
    client = QdrantClient(":memory:")
    old = chunk_pages(tmp_path, {"people/leave.md": OLD_PAGES["people/leave.md"]}, OLD_COMMIT)
    old_store = QdrantVectorStore(client, "old")
    build(old_store, old)
    previous = read_previous_index(old_store)
    if change == "deleted":
        old_store.delete_except({old[0].chunk_id})
    else:
        if change == "retitled":
            changed = dataclasses.replace(old[1], title="Something else")
        else:
            changed = dataclasses.replace(old[1], heading_path=("Something else",))
        text = changed.text_for_embedding
        old_store.upsert(
            [changed],
            [ExactEmbedder().embed_query(text)],
            [FakeSparseEncoder().encode_query(text)],
        )
    newer = [dataclasses.replace(chunk, source_commit=NEW_COMMIT) for chunk in old]
    report, embedder = build(
        QdrantVectorStore(client, "new"), newer, Reuse(old_store, previous.points)
    )
    return newer, report, embedder


@pytest.mark.parametrize("change", ["retitled", "reheaded", "deleted"])
def test_a_point_that_changed_after_it_was_read_is_not_reused(tmp_path: Path, change: str) -> None:
    newer, report, embedder = leave_page_read_then_changed(tmp_path, change)

    assert (report.reused, report.embedded) == (1, 1)
    assert embedder.embedded == [newer[1].text_for_embedding]


def test_the_token_count_stored_is_the_newer_chunks_not_the_lenders() -> None:
    client = QdrantClient(":memory:")
    old_store, new_store = QdrantVectorStore(client, "old"), QdrantVectorStore(client, "new")
    build(old_store, [dataclasses.replace(LEAVE, token_count=99)])
    newer = dataclasses.replace(LEAVE, source_commit=NEW_COMMIT)

    done = update(old_store, new_store, [newer])

    assert done.report.reused == 1
    assert new_store.stored_chunks([newer.chunk_id]) == {newer.chunk_id: newer}


def test_an_update_removes_what_an_earlier_run_left_that_the_newer_commit_lacks(
    tmp_path: Path,
) -> None:
    client = QdrantClient(":memory:")
    old = chunk_pages(tmp_path, {"people/leave.md": OLD_PAGES["people/leave.md"]}, OLD_COMMIT)
    old_store, new_store = QdrantVectorStore(client, "old"), QdrantVectorStore(client, "new")
    build(old_store, old)
    build(new_store, [make_chunk("Left behind by an earlier run.", doc_id="gone.md")])
    newer = [dataclasses.replace(chunk, source_commit=NEW_COMMIT) for chunk in old]

    done = update(old_store, new_store, newer)

    assert (done.report.reused, done.report.removed) == (2, 1)
    assert sorted(new_store.point_ids()) == sorted(chunk.chunk_id for chunk in newer)


def test_a_lent_point_without_a_keyword_vector_gets_one_computed_from_its_text() -> None:
    client = QdrantClient(":memory:")
    old_store, new_store = QdrantVectorStore(client, "old"), QdrantVectorStore(client, "new")
    old_store.ensure_collection(ExactEmbedder.dimension)
    text = LEAVE.text_for_embedding
    old_store.upsert([LEAVE], [ExactEmbedder().embed_query(text)], [SparseVector([], [])])
    newer = dataclasses.replace(LEAVE, source_commit=NEW_COMMIT)

    done = update(old_store, new_store, [newer])

    assert (done.report.reused, done.report.embedded) == (1, 0)
    stored = new_store.points([newer.chunk_id])[newer.chunk_id].sparse
    expected = FakeSparseEncoder().encode_query(text)
    assert expected.indices
    # The store keeps a keyword vector in word order.
    assert dict(zip(stored.indices, stored.values, strict=True)) == dict(
        zip(expected.indices, expected.values, strict=True)
    )


def test_reuse_reports_its_progress_batch_by_batch(commits: Commits) -> None:
    client = QdrantClient(":memory:")
    previous = read_previous_index(commits.old_store)
    progress: list[tuple[int, int]] = []

    index_chunks(
        commits.new,
        embedder=ExactEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        store=QdrantVectorStore(client, "new"),
        batch_size=4,
        reuse=Reuse(commits.old_store, previous.points),
        on_reuse_progress=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(4, 11), (8, 11), (11, 11)]


def test_an_interrupted_update_carries_on_where_it_stopped(commits: Commits) -> None:
    # What is already stored in the new collection, as it should be, is left alone.
    again = update(commits.old_store, commits.updated, commits.new)

    assert (again.report.already_indexed, again.report.reused, again.report.embedded) == (21, 0, 0)
    assert again.embedder.embedded == []


def test_reading_the_older_collection_can_be_limited_to_some_pages(commits: Commits) -> None:
    some = read_previous_index(commits.old_store, pages={"people/leave.md", "legal/retired.md"})
    none = read_previous_index(commits.old_store, pages=set())

    assert set(some.pages) == {"people/leave.md", "legal/retired.md"}
    assert some.chunks == 3
    assert len(some.points) == 3
    assert (none.chunks, none.points, none.pages) == (0, {}, {})


LEAVE = make_chunk("Take 25 days.", heading_path=("Annual leave",))


def pages_after(newer: Chunk) -> PageChanges:
    """How the leave page counts when its one chunk becomes ``newer``."""
    store = QdrantVectorStore(QdrantClient(":memory:"), "old")
    build(store, [LEAVE])
    return compare_pages(read_previous_index(store), [newer])


def test_a_page_whose_chunks_differ_only_in_their_commit_is_unchanged() -> None:
    assert pages_after(dataclasses.replace(LEAVE, source_commit=NEW_COMMIT)) == PageChanges(
        added=[], changed=[], removed=[], unchanged=["people/leave.md"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"chunk_id": "another id"},
        {"title": "Holidays"},
        {"url": "https://handbook.example.com/handbook/holidays/"},
        {"section": "benefits"},
        {"heading_path": ("Holidays",)},
        {"start_char": 1},
        {"end_char": 14},
        {"text": "Take 30 days."},
        {"token_count": 4},
    ],
    ids=lambda change: next(iter(change)),
)
def test_a_page_is_changed_when_anything_its_chunk_is_stored_with_differs(
    change: dict[str, Any],
) -> None:
    assert pages_after(dataclasses.replace(LEAVE, **change)).changed == ["people/leave.md"]


# Finding the older collection, and refusing one made with other models or chunking.


def test_a_page_that_gained_a_chunk_is_changed() -> None:
    store = QdrantVectorStore(QdrantClient(":memory:"), "old")
    build(store, [LEAVE])
    gained = make_chunk("Carry over five days.", heading_path=("Annual leave",))

    changes = compare_pages(read_previous_index(store), [LEAVE, gained])

    assert changes == PageChanges(added=[], changed=["people/leave.md"], removed=[], unchanged=[])


def test_an_older_point_that_is_not_a_whole_chunk_stops_the_update() -> None:
    client = QdrantClient(":memory:")
    old_store = QdrantVectorStore(client, "old")
    build(old_store, [LEAVE])
    client.delete_payload("old", keys=["text"], points=[LEAVE.chunk_id])

    with pytest.raises(VectorStoreError, match=f"Point {LEAVE.chunk_id} in old has no 'text'"):
        read_previous_index(old_store)


@pytest.fixture
def store() -> QdrantVectorStore:
    """The store, as `handbook-rag index` would connect to it for the newer commit."""
    return QdrantVectorStore(
        QdrantClient(":memory:"), Settings(handbook_commit=NEW_COMMIT).collection_name
    )


def test_the_older_commits_collection_is_found_by_its_commit(store: QdrantVectorStore) -> None:
    older = store.with_collection(Settings().collection_for(OLD_COMMIT))
    older.ensure_collection(dimension=8)

    found = find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), OLD_COMMIT)

    assert (
        found.collection == older.collection == Settings(handbook_commit=OLD_COMMIT).collection_name
    )


@pytest.mark.parametrize(
    "commit",
    [OLD_COMMIT[:12], OLD_COMMIT + "0", OLD_COMMIT.upper()],
    ids=["short", "long", "upper"],
)
def test_a_commit_that_is_not_a_full_hash_is_refused(store: QdrantVectorStore, commit: str) -> None:
    with pytest.raises(ReuseError, match="not a full commit hash"):
        find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), commit)


def test_the_collection_being_built_is_refused_as_a_source(store: QdrantVectorStore) -> None:
    store.ensure_collection(dimension=8)

    with pytest.raises(ReuseError, match="which is the collection being built"):
        find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), NEW_COMMIT)


def test_a_commit_with_no_collection_at_all_is_refused(store: QdrantVectorStore) -> None:
    with pytest.raises(ReuseError) as refused:
        find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), OLD_COMMIT)

    assert "has no collection made with these models and this chunking" in str(refused.value)
    assert "holds no collection of that commit" in str(refused.value)


@pytest.mark.parametrize(
    ("older", "named"),
    [
        pytest.param(
            {"embedding_model": "BAAI/bge-base-en-v1.5"},
            "another embedding model (these settings use BAAI/bge-small-en-v1.5)",
            id="embedding model",
        ),
        pytest.param(
            {"embedding_model": "BAAI/bge-small-en-v1.5b"},
            "another embedding model (these settings use BAAI/bge-small-en-v1.5)",
            id="embedding model whose name starts like this one",
        ),
        pytest.param(
            {"sparse_model": "Qdrant/bm42"},
            "another keyword model (these settings use Qdrant/bm25)",
            id="keyword model",
        ),
        pytest.param(
            {"chunk_max_tokens": 200},
            "another chunking (these settings use v1-max320-min60-overlap40)",
            id="chunking",
        ),
    ],
)
def test_vectors_made_with_another_model_or_chunking_are_refused_by_name(
    store: QdrantVectorStore, older: dict[str, Any], named: str
) -> None:
    made_otherwise = store.with_collection(
        Settings(handbook_commit=OLD_COMMIT, **older).collection_name
    )
    made_otherwise.ensure_collection(dimension=8)
    # A collection of another commit is not mentioned.
    store.with_collection(Settings(handbook_commit="c" * 40).collection_name).ensure_collection(8)

    with pytest.raises(ReuseError) as refused:
        find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), OLD_COMMIT)

    message = str(refused.value)
    assert "another model's vector for the same text means something else" in message
    assert f"The store holds {made_otherwise.collection}, made with {named}." in message
    assert "c" * 8 not in message


@pytest.mark.parametrize(
    "rename",
    [
        pytest.param(lambda name: f"{name}-full-check", id="a word added at the end"),
        pytest.param(
            lambda name: name.replace("-qdrant-bm25-", "-extra-qdrant-bm25-"),
            id="a word added in the middle",
        ),
    ],
)
def test_a_collection_of_that_commit_named_otherwise_is_not_said_to_have_other_settings(
    store: QdrantVectorStore, rename: Callable[[str], str]
) -> None:
    # Such a name shows the same models and chunking, and may hide anything; so the
    # refusal says only what the name shows.
    odd = rename(Settings().collection_for(OLD_COMMIT))
    store.with_collection(odd).ensure_collection(dimension=8)

    with pytest.raises(ReuseError) as refused:
        find_reuse_source(store, Settings(handbook_commit=NEW_COMMIT), OLD_COMMIT)

    message = str(refused.value)
    assert (
        f"The store holds {odd}, which is named otherwise, so what it was made with cannot be "
        "read from its name." in message
    )
    assert "made with another" not in message
