import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag import __version__
from handbook_rag.cli import app
from handbook_rag.index.store import QdrantVectorStore
from tests.support import FakeEmbedder, Upstream, count_words, write_snapshot

runner = CliRunner()


def test_version_flag_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_running_without_arguments_lists_the_commands() -> None:
    result = runner.invoke(app, [])

    assert "snapshot" in result.output


def test_snapshot_command_downloads_and_reports(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_REPO_URL", upstream.url)
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", upstream.first_commit)

    result = runner.invoke(app, ["snapshot"])

    assert result.exit_code == 0, result.output
    assert "3 Markdown files" in result.output
    assert upstream.first_commit[:12] in result.output
    assert (data_dir / "handbook" / "manifest.json").is_file()


def test_snapshot_command_fails_clearly_on_a_bad_pin(
    monkeypatch: pytest.MonkeyPatch, upstream: Upstream, tmp_path: Path
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_REPO_URL", upstream.url)
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", "main")

    result = runner.invoke(app, ["snapshot"])

    assert result.exit_code == 1
    assert "error:" in result.output
    assert "full commit hash" in result.output


@pytest.fixture
def word_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count words instead of downloading the embedding model's tokenizer."""
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)
    monkeypatch.setattr("handbook_rag.cli.token_counter", lambda _model: count_words)


@pytest.mark.usefixtures("word_tokens")
def test_chunks_command_summarises_and_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    page = "## Limits\n\nYou can expense a laptop up to one thousand dollars with approval."
    write_snapshot(tmp_path / "data", commit, {"finance/expenses.md": page, "stub.md": "Moved."})
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    export = tmp_path / "out" / "chunks.jsonl"

    result = runner.invoke(app, ["chunks", "--export", str(export)])

    assert result.exit_code == 0, result.output
    assert "1 loaded" in result.output
    assert "1 too little text" in result.output
    assert "0 chunks exceed" in result.output
    (line,) = export.read_text().splitlines()
    exported = json.loads(line)
    assert exported["doc_id"] == "finance/expenses.md"
    assert exported["heading_path"] == ["Limits"]
    assert exported["text"] == page


@pytest.mark.usefixtures("word_tokens")
def test_chunks_command_explains_a_missing_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))

    result = runner.invoke(app, ["chunks"])

    assert result.exit_code == 1
    assert "handbook-rag snapshot" in result.output


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeEmbedder:
    """A snapshot on disk, a fake embedding model and an in-memory vector store."""
    commit = "a" * 40
    write_snapshot(
        tmp_path / "data",
        commit,
        {
            "people/leave.md": (
                '---\ntitle: "Leave"\n---\n\n'
                "## Parental leave\n\nNew parents get sixteen weeks of paid leave."
            ),
            "finance/expenses.md": "## Laptops\n\nExpense a laptop up to one thousand dollars.",
        },
    )
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)

    embedder = FakeEmbedder()
    client = QdrantClient(":memory:")
    monkeypatch.setattr("handbook_rag.cli.make_embedder", lambda _settings: embedder)
    monkeypatch.setattr(
        "handbook_rag.cli.make_store",
        lambda settings: QdrantVectorStore(client, settings.collection_name),
    )
    return embedder


def test_index_then_search_finds_the_right_page(fake_pipeline: FakeEmbedder) -> None:
    indexed = runner.invoke(app, ["index"])

    assert indexed.exit_code == 0, indexed.output
    assert "2 in the collection" in indexed.output
    assert "2 new" in indexed.output

    found = runner.invoke(app, ["search", "how many weeks of parental leave", "--limit", "1"])

    assert found.exit_code == 0, found.output
    assert "Leave > Parental leave" in found.output
    assert "https://handbook.gitlab.com/handbook/people/leave/" in found.output
    assert "expenses" not in found.output


def test_indexing_twice_embeds_nothing_the_second_time(fake_pipeline: FakeEmbedder) -> None:
    runner.invoke(app, ["index"])
    fake_pipeline.embedded.clear()

    again = runner.invoke(app, ["index"])

    assert again.exit_code == 0, again.output
    assert "0 new" in again.output
    assert "2 already stored" in again.output
    assert fake_pipeline.embedded == []


def test_search_can_be_limited_to_a_section(fake_pipeline: FakeEmbedder) -> None:
    runner.invoke(app, ["index"])

    found = runner.invoke(app, ["search", "parental leave", "--section", "finance"])

    assert found.exit_code == 0, found.output
    assert "finance/expenses" in found.output
    assert "people/leave" not in found.output


@pytest.mark.usefixtures("fake_pipeline")
def test_search_before_indexing_says_what_to_do() -> None:
    result = runner.invoke(app, ["search", "anything"])

    assert result.exit_code == 1
    assert "handbook-rag index" in result.output


def test_commands_explain_when_the_vector_store_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_QDRANT_URL", "http://127.0.0.1:9")

    for command in (["index"], ["search", "anything"]):
        result = runner.invoke(app, command)

        assert result.exit_code == 1
        assert "make qdrant" in result.output
