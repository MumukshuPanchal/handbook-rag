import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from handbook_rag import __version__
from handbook_rag.cli import app
from tests.support import Upstream, count_words, write_snapshot

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
