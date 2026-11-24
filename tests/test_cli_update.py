"""The commands that move the index to a newer commit of the handbook (ADR 0019).

A second snapshot is downloaded beside the first, the newer commit is indexed into a
collection of its own by reusing what it can of the older one, and two indexes can be
compared. The older snapshot and the older collection are never changed.
"""

import dataclasses
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.index.indexer import IndexReport
from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import QdrantVectorStore
from tests.support import (
    FakeEmbedder,
    FakeSparseEncoder,
    Upstream,
    count_words,
    make_chunk,
    write_snapshot,
)

runner = CliRunner()

OLD, NEW = "a" * 40, "b" * 40
NO_WORDS = SparseVector([], [])
OLD_PAGES = {
    "people/leave.md": (
        '---\ntitle: "Leave"\n---\n\n'
        "## Parental leave\n\nNew parents get sixteen weeks of paid leave."
    ),
    "finance/expenses.md": (
        '---\ntitle: "Expenses"\n---\n\n## Laptops\n\nExpense a laptop up to one thousand dollars.'
    ),
    "legal/retired.md": (
        '---\ntitle: "Old policy"\n---\n\nThis policy was retired and replaced by the new one.'
    ),
}
NEW_PAGES = {
    "people/leave.md": OLD_PAGES["people/leave.md"],
    "finance/expenses.md": OLD_PAGES["finance/expenses.md"].replace("Expenses", "Spending"),
    "security/incidents.md": (
        '---\ntitle: "Incidents"\n---\n\nReport a security incident in the incident channel.'
    ),
}


@dataclass
class Handbook:
    """Two snapshots on disk, stand-in models, and one vector store held in memory."""

    embedder: FakeEmbedder
    client: QdrantClient
    root: Path
    monkeypatch: pytest.MonkeyPatch

    def use(self, commit: str) -> None:
        """Point the settings at one of the two snapshots, as its environment would."""
        data_dir = self.root / ("data" if commit == OLD else "data-next")
        self.monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(data_dir))
        self.monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
        get_settings.cache_clear()

    def collection(self, commit: str) -> QdrantVectorStore:
        """The collection of ``commit`` made with the default models and chunking.

        Built without reading the environment, so a test that changes a model in the
        environment still finds the collection the older commit was indexed into.
        """
        defaults = Settings.model_construct(handbook_commit=commit)
        return QdrantVectorStore(self.client, defaults.collection_name)

    def collections(self) -> list[str]:
        return sorted(item.name for item in self.client.get_collections().collections)


@pytest.fixture
def handbook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Handbook:
    write_snapshot(tmp_path / "data", OLD, OLD_PAGES)
    write_snapshot(tmp_path / "data-next", NEW, NEW_PAGES)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)
    embedder, client = FakeEmbedder(), QdrantClient(":memory:")
    monkeypatch.setattr("handbook_rag.wiring.make_embedder", lambda _settings: embedder)
    monkeypatch.setattr(
        "handbook_rag.wiring.make_sparse_encoder", lambda _settings: FakeSparseEncoder()
    )
    monkeypatch.setattr(
        "handbook_rag.wiring.make_store",
        lambda settings: QdrantVectorStore(client, settings.collection_name),
    )
    built = Handbook(embedder, client, tmp_path, monkeypatch)
    built.use(OLD)
    return built


def indexed_at_the_older_commit(handbook: Handbook) -> dict[str, Any]:
    """Index the older commit, then switch to the newer one. Return the older points."""
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0, result.output
    handbook.embedder.embedded.clear()
    handbook.use(NEW)
    return every_point(handbook.collection(OLD))


def every_point(store: QdrantVectorStore) -> dict[str, Any]:
    points = store.points(sorted(store.point_ids()))
    return {point_id: dataclasses.asdict(point) for point_id, point in points.items()}


# Downloading a newer commit beside the pinned one.


def snapshot_at(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, data_dir: Path, commit: str
) -> Any:
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_REPO_URL", upstream.url)
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    get_settings.cache_clear()
    return runner.invoke(app, ["snapshot"])


def leave_page(data_dir: Path) -> str:
    return (
        data_dir / "handbook" / "repo" / "content" / "handbook" / "people" / "leave.md"
    ).read_text()


def test_snapshot_refuses_to_replace_a_snapshot_of_another_commit(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    assert snapshot_at(monkeypatch, upstream, data_dir, upstream.first_commit).exit_code == 0
    manifest = (data_dir / "handbook" / "manifest.json").read_text()

    refused = snapshot_at(monkeypatch, upstream, data_dir, upstream.second_commit)

    assert refused.exit_code == 1
    assert f"holds the snapshot of commit {upstream.first_commit[:12]}" in refused.output
    assert "set HANDBOOK_RAG_DATA_DIR to another directory" in refused.output
    assert "run again with --replace" in refused.output
    assert (data_dir / "handbook" / "manifest.json").read_text() == manifest
    assert "25 days" in leave_page(data_dir)


def test_snapshot_replaces_a_snapshot_of_another_commit_when_asked(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    snapshot_at(monkeypatch, upstream, data_dir, upstream.first_commit)
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", upstream.second_commit)
    get_settings.cache_clear()

    replaced = runner.invoke(app, ["snapshot", "--replace"])

    assert replaced.exit_code == 0, replaced.output
    assert json.loads((data_dir / "handbook" / "manifest.json").read_text())["commit"] == (
        upstream.second_commit
    )
    assert "30 days" in leave_page(data_dir)


def test_a_newer_commit_is_downloaded_beside_the_first_without_touching_it(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    snapshot_at(monkeypatch, upstream, tmp_path / "data", upstream.first_commit)
    again = snapshot_at(monkeypatch, upstream, tmp_path / "data", upstream.first_commit)
    first = sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "data").rglob("*.md"))

    beside = snapshot_at(monkeypatch, upstream, tmp_path / "data-next", upstream.second_commit)

    assert again.exit_code == 0, again.output
    assert beside.exit_code == 0, beside.output
    assert "25 days" in leave_page(tmp_path / "data")
    assert "30 days" in leave_page(tmp_path / "data-next")
    assert (
        sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "data").rglob("*.md"))
        == first
    )


# Indexing a newer commit by reusing the vectors of an older one.


def test_index_reuses_the_older_commits_vectors_and_reports_what_changed(
    handbook: Handbook,
) -> None:
    older = indexed_at_the_older_commit(handbook)

    result = runner.invoke(app, ["index", "--reuse-from", OLD])

    assert result.exit_code == 0, result.output
    old_name = handbook.collection(OLD).collection
    assert f"reuse from  {old_name}, which is only read" in result.output
    assert "read        3 stored chunks of 3 pages" in result.output
    assert "chunks      3 in the collection" in result.output
    assert "pages       1 added, 1 changed, 1 removed, 1 unchanged" in result.output
    assert "reused      1 chunks' vectors" in result.output
    assert "embedded    2 new" in result.output
    assert "reused      1 of 1 (100%)" in result.output
    assert "embedded    2 of 2 (100%)" in result.output
    assert "skipped     0 already stored" in result.output
    assert "chunking" in result.output
    assert "reading the older collection" in result.output
    assert sorted(handbook.embedder.embedded) == [
        "Incidents\n\nReport a security incident in the incident channel.",
        "Spending > Laptops\n\n## Laptops\n\nExpense a laptop up to one thousand dollars.",
    ]
    assert every_point(handbook.collection(OLD)) == older
    assert handbook.collections() == sorted([old_name, handbook.collection(NEW).collection])
    found = runner.invoke(app, ["search", "laptop", "--limit", "1"])
    assert "Spending > Laptops" in found.output


def test_index_without_reuse_says_nothing_about_reuse(handbook: Handbook) -> None:
    result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, result.output
    for word in ("reuse", "pages ", "replaced", "chunking"):
        assert word not in result.output


@pytest.mark.parametrize(
    ("setting", "value", "named"),
    [
        ("HANDBOOK_RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5", "another embedding model"),
        ("HANDBOOK_RAG_SPARSE_MODEL", "Qdrant/bm42", "another keyword model"),
        ("HANDBOOK_RAG_CHUNK_MAX_TOKENS", "200", "another chunking"),
    ],
)
def test_index_refuses_to_reuse_vectors_made_with_other_models_or_chunking(
    handbook: Handbook, monkeypatch: pytest.MonkeyPatch, setting: str, value: str, named: str
) -> None:
    older = indexed_at_the_older_commit(handbook)
    monkeypatch.setenv(setting, value)
    get_settings.cache_clear()

    result = runner.invoke(app, ["index", "--reuse-from", OLD])

    assert result.exit_code == 1
    assert "another model's vector for the same text means something else" in result.output
    assert f"made with {named}" in result.output
    assert handbook.embedder.embedded == []
    assert handbook.collections() == [handbook.collection(OLD).collection]
    assert every_point(handbook.collection(OLD)) == older


def test_index_refuses_to_reuse_vectors_from_the_collection_it_is_building(
    handbook: Handbook,
) -> None:
    result = runner.invoke(app, ["index", "--reuse-from", OLD])

    assert result.exit_code == 1
    assert "which is the collection being built" in result.output
    assert handbook.collections() == []


def test_index_stores_again_a_chunk_whose_details_changed_under_the_same_id(
    handbook: Handbook, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["index"])
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_BASE_URL", "https://handbook.example.org/handbook/")
    get_settings.cache_clear()

    again = runner.invoke(app, ["index"])

    assert again.exit_code == 0, again.output
    assert "embedded    3 new" in again.output
    assert "replaced    3 stored under the same id with other details" in again.output
    found = runner.invoke(app, ["search", "parental leave", "--limit", "1"])
    assert "https://handbook.example.org/handbook/people/leave/" in found.output


# Checking reuse on a slice, and comparing two collections.


def slice_command(
    root: Path, *extra: str, evidence: Sequence[str] = tuple(sorted(OLD_PAGES))
) -> list[str]:
    """The check-reuse command line, with a golden set whose evidence is on these pages.

    By default the evidence spans the three pages of the older commit.
    """
    rows = [
        {
            "id": f"q{number}",
            "question": f"Question {number}?",
            "reference_answer": "An answer.",
            "evidence": [{"doc_id": doc_id, "quote": "a quote"}],
            "type": "answerable",
            "origin": "hand-written",
        }
        for number, doc_id in enumerate(evidence, start=1)
    ]
    dataset = root / "questions.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = root / "gate.json"
    config.write_text(json.dumps({"distractor_pages": 0, "seed": 1, "minimums": {"hit@5": 0.5}}))
    return [
        "check-reuse",
        "--reuse-from",
        OLD,
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        *extra,
    ]


def test_check_reuse_indexes_the_slice_twice_in_memory_and_finds_the_two_the_same(
    handbook: Handbook,
) -> None:
    older = indexed_at_the_older_commit(handbook)

    result = runner.invoke(app, slice_command(handbook.root))

    assert result.exit_code == 0, result.output
    assert f"pages       2 of 3 at commit {NEW[:12]}, 2 chunks" in result.output
    assert "full        2 embedded" in result.output
    assert "3 chunks of the same pages" in result.output
    assert "pages       0 added, 1 changed, 1 removed, 1 unchanged" in result.output
    assert "reused      1 chunks' vectors, 1 embedded" in result.output
    assert "compared    2 points that both collections hold" in result.output
    assert "ids         0 missing from the second, 0 only in the second" in result.output
    assert "vectors     0 differ by more than 0.0001; largest difference" in result.output
    assert "Reuse check passed" in result.output
    # Nothing was written to the store server: it holds the older collection only, as it was.
    assert handbook.collections() == [handbook.collection(OLD).collection]
    assert every_point(handbook.collection(OLD)) == older


class DriftingEmbedder(FakeEmbedder):
    """An embedding model whose vectors moved since the older index was built."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = super().embed_documents(texts)
        for vector in vectors:
            vector[0] += 0.05
            length = math.sqrt(sum(value * value for value in vector))
            vector[:] = [value / length for value in vector]
        return vectors


def test_check_reuse_fails_and_names_what_differs_beyond_the_tolerance(
    handbook: Handbook, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed_at_the_older_commit(handbook)
    monkeypatch.setattr("handbook_rag.wiring.make_embedder", lambda _settings: DriftingEmbedder())

    strict = runner.invoke(app, slice_command(handbook.root))
    lenient = runner.invoke(app, slice_command(handbook.root, "--tolerance", "0.5"))

    assert strict.exit_code == 1
    assert "vectors     1 differ by more than 0.0001" in strict.output
    assert "    dense    " in strict.output
    assert "Reuse check failed" in strict.output
    assert lenient.exit_code == 0, lenient.output
    assert "Reuse check passed" in lenient.output


def test_check_reuse_refuses_an_older_commit_that_has_no_collection(handbook: Handbook) -> None:
    handbook.use(NEW)

    result = runner.invoke(app, slice_command(handbook.root))

    assert result.exit_code == 1
    assert "holds no collection of that commit" in result.output


def test_compare_collections_names_the_points_that_differ(handbook: Handbook) -> None:
    indexed_at_the_older_commit(handbook)
    runner.invoke(app, ["index"])
    old_name, new_name = handbook.collection(OLD).collection, handbook.collection(NEW).collection

    same = runner.invoke(app, ["compare-collections", old_name, old_name])
    different = runner.invoke(app, ["compare-collections", old_name, new_name])

    assert same.exit_code == 0, same.output
    assert "compared    3 points" in same.output
    assert "payloads    0 fields differ" in same.output
    assert different.exit_code == 1
    assert f"{new_name} against {old_name}" in different.output
    # Two chunks keep their ids: the leave page's, which only came from another commit,
    # and the expenses page's, whose title changed and so its vectors too.
    ids = {chunk.doc_id: chunk.chunk_id for chunk in handbook.collection(OLD).scroll_chunks()}
    leave, expenses = ids["people/leave.md"], ids["finance/expenses.md"]
    assert "compared    2 points" in different.output
    assert "payloads    3 fields differ" in different.output
    assert f"payload  {leave}: source_commit differs" in different.output
    assert f"payload  {expenses}: source_commit differs" in different.output
    assert f"payload  {expenses}: title differs" in different.output
    assert f"dense    {expenses}: differs by up to" in different.output
    assert f"sparse   {expenses}: differs by up to 1" in different.output
    assert f"{leave}: differs" not in different.output
    assert "ids         1 missing from the second, 1 only in the second" in different.output
    assert f"in {old_name} only" in different.output
    assert f"in {new_name} only" in different.output


def test_compare_collections_explains_a_collection_that_does_not_exist(handbook: Handbook) -> None:
    result = runner.invoke(app, ["compare-collections", "handbook-nothing", "handbook-none"])

    assert result.exit_code == 1
    assert "Collection handbook-nothing does not exist" in result.output
    assert handbook.collections() == []


def test_compare_collections_explains_a_second_collection_that_does_not_exist(
    handbook: Handbook,
) -> None:
    runner.invoke(app, ["index"])
    old_name = handbook.collection(OLD).collection

    result = runner.invoke(app, ["compare-collections", old_name, "handbook-none"])

    assert result.exit_code == 1
    assert "Collection handbook-none does not exist" in result.output


def test_compare_collections_lists_twenty_differences_and_counts_the_rest(
    handbook: Handbook,
) -> None:
    full, empty = handbook.collection(OLD), handbook.collection(NEW)
    for store in (full, empty):
        store.ensure_collection(dimension=FakeEmbedder.dimension)
    chunks = [make_chunk(f"chunk number {n}") for n in range(25)]
    full.upsert(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]), [NO_WORDS] * 25)

    result = runner.invoke(app, ["compare-collections", full.collection, empty.collection])

    assert result.exit_code == 1
    assert "ids         25 missing from the second, 0 only in the second" in result.output
    assert result.output.count("    missing  ") == 20
    assert "    and 5 more" in result.output


def test_after_a_reuse_the_rate_counts_only_the_time_spent_embedding(
    handbook: Handbook, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexed_at_the_older_commit(handbook)
    # Copying 40 reused vectors took most of the run; embedding 20 chunks took 2 seconds.
    timed = IndexReport(
        total=60,
        embedded=20,
        already_indexed=0,
        removed=0,
        seconds=50.0,
        reused=40,
        reuse_seconds=47.0,
        embed_seconds=2.0,
    )
    monkeypatch.setattr("handbook_rag.cli.index_chunks", lambda *_args, **_kwargs: timed)

    result = runner.invoke(app, ["index", "--reuse-from", OLD])

    assert result.exit_code == 0, result.output
    assert "embedded    20 new (10 per second)" in result.output
    assert "reused      40 chunks' vectors, in 47.0 seconds" in result.output
    assert "time        50.0 seconds" in result.output
    assert "reusing 47.0, embedding 2.0" in result.output


# What the review of these commands found.


@pytest.mark.parametrize("manifest", [None, '{"commit": null}'], ids=["no manifest", "null"])
def test_snapshot_refuses_to_replace_another_commit_whose_manifest_names_none(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path, manifest: str | None
) -> None:
    # As after a replacement that failed halfway: the clone still says what is there.
    data_dir = tmp_path / "data"
    snapshot_at(monkeypatch, upstream, data_dir, upstream.first_commit)
    manifest_path = data_dir / "handbook" / "manifest.json"
    if manifest is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(manifest)

    refused = snapshot_at(monkeypatch, upstream, data_dir, upstream.second_commit)

    assert refused.exit_code == 1
    assert f"holds the snapshot of commit {upstream.first_commit[:12]}" in refused.output
    assert "25 days" in leave_page(data_dir)


def test_snapshot_explains_a_manifest_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    snapshot_at(monkeypatch, upstream, data_dir, upstream.first_commit)
    (data_dir / "handbook" / "manifest.json").write_text("not json")

    refused = snapshot_at(monkeypatch, upstream, data_dir, upstream.second_commit)

    assert refused.exit_code == 1
    assert isinstance(refused.exception, SystemExit)
    assert "manifest.json cannot be read" in refused.output
    assert "25 days" in leave_page(data_dir)


def broken_older_point(handbook: Handbook) -> str:
    """Strip the title from the payload of one point of the older collection. Its id."""
    older = handbook.collection(OLD)
    point_id = sorted(older.point_ids())[0]
    handbook.client.delete_payload(older.collection, keys=["title"], points=[point_id])
    return point_id


def test_index_names_a_stored_point_that_is_not_a_whole_chunk(handbook: Handbook) -> None:
    runner.invoke(app, ["index"])
    point_id = broken_older_point(handbook)

    result = runner.invoke(app, ["index"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    older = handbook.collection(OLD).collection
    assert f"Point {point_id} in {older} has no 'title' in its payload" in result.output


def test_an_update_from_an_older_point_that_is_not_a_whole_chunk_stops_before_writing(
    handbook: Handbook,
) -> None:
    indexed_at_the_older_commit(handbook)
    point_id = broken_older_point(handbook)

    result = runner.invoke(app, ["index", "--reuse-from", OLD])

    assert result.exit_code == 1
    older = handbook.collection(OLD).collection
    assert f"Point {point_id} in {older} has no 'title' in its payload" in result.output
    assert handbook.collections() == [older]


def test_check_reuse_that_reused_nothing_fails_and_says_so(handbook: Handbook) -> None:
    handbook.use(NEW)
    handbook.collection(OLD).ensure_collection(FakeEmbedder.dimension)

    result = runner.invoke(app, slice_command(handbook.root))

    assert result.exit_code == 1
    assert "reused      0 chunks' vectors" in result.output
    assert "no vector was reused, so nothing about reuse was checked" in result.output
    assert "Reuse check passed" not in result.output


def test_check_reuse_reads_only_the_pages_of_the_slice_from_the_older_collection(
    handbook: Handbook,
) -> None:
    indexed_at_the_older_commit(handbook)
    command = slice_command(handbook.root, evidence=["legal/retired.md", "people/leave.md"])

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert "2 chunks of the same pages" in result.output
    assert "pages       0 added, 0 changed, 1 removed, 1 unchanged" in result.output


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-0.1"])
@pytest.mark.parametrize(
    "command",
    [["compare-collections", "first", "second"], ["check-reuse", "--reuse-from", OLD]],
    ids=["compare-collections", "check-reuse"],
)
def test_a_tolerance_that_would_make_the_comparison_meaningless_is_refused(
    handbook: Handbook, command: list[str], tolerance: str
) -> None:
    result = runner.invoke(app, [*command, "--tolerance", tolerance])

    assert result.exit_code == 2
    assert "must be a finite number of at least 0" in result.output
    assert handbook.embedder.embedded == []


def test_compare_collections_applies_the_tolerance_it_is_given(handbook: Handbook) -> None:
    indexed_at_the_older_commit(handbook)
    runner.invoke(app, ["index"])
    old_name, new_name = handbook.collection(OLD).collection, handbook.collection(NEW).collection

    strict = runner.invoke(app, ["compare-collections", old_name, new_name])
    lenient = runner.invoke(app, ["compare-collections", old_name, new_name, "--tolerance", "2"])

    assert "vectors     2 differ by more than 0.0001" in strict.output
    assert "vectors     0 differ by more than 2" in lenient.output


def test_without_reuse_the_rate_counts_the_whole_run(
    handbook: Handbook, monkeypatch: pytest.MonkeyPatch
) -> None:
    timed = IndexReport(
        total=60, embedded=20, already_indexed=40, removed=0, seconds=10.0, embed_seconds=2.0
    )
    monkeypatch.setattr("handbook_rag.cli.index_chunks", lambda *_args, **_kwargs: timed)

    result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, result.output
    assert "embedded    20 new (2 per second)" in result.output
