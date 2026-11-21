import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag import __version__
from handbook_rag.cli import app
from handbook_rag.index.store import QdrantVectorStore
from tests.support import (
    FakeEmbedder,
    FakeReranker,
    FakeSparseEncoder,
    Upstream,
    count_words,
    write_snapshot,
)

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
    monkeypatch.setattr("handbook_rag.wiring.make_embedder", lambda _settings: embedder)
    monkeypatch.setattr(
        "handbook_rag.wiring.make_sparse_encoder", lambda _settings: FakeSparseEncoder()
    )
    monkeypatch.setattr("handbook_rag.wiring.make_reranker", lambda _settings: FakeReranker())
    monkeypatch.setattr(
        "handbook_rag.wiring.make_store",
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


def test_eval_retrieval_scores_the_golden_set_and_writes_the_run(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "how many weeks of paid parental leave do new parents get",
                "reference_answer": "Sixteen weeks.",
                "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
                "type": "answerable",
                "origin": "hand-written",
            }
        )
        + "\n"
    )
    output = tmp_path / "runs"

    result = runner.invoke(
        app, ["eval-retrieval", "--dataset", str(dataset), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "1 scored, 0 failed, 0 unanswerable" in result.output
    assert "Hit@5       1.000" in result.output
    assert "Hit@5 on hand-written questions: 1.000 (n=1)" in result.output
    (row,) = (output / "baseline" / "results.jsonl").read_text().splitlines()
    assert json.loads(row)["grade"]["hit@1"] == 1.0


@pytest.mark.usefixtures("fake_pipeline")
def test_eval_retrieval_explains_a_missing_or_broken_golden_set(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["eval-retrieval", "--dataset", str(tmp_path / "none.jsonl")])
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"id": "q001"}\n')

    malformed = runner.invoke(app, ["eval-retrieval", "--dataset", str(broken)])

    assert missing.exit_code == 1
    assert "golden set not found" in missing.output
    assert malformed.exit_code == 1
    assert "broken.jsonl:1" in malformed.output


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_works_in_every_retrieval_mode(fake_pipeline: FakeEmbedder, mode: str) -> None:
    runner.invoke(app, ["index"])

    found = runner.invoke(app, ["search", "parental leave weeks", "--limit", "1", "--mode", mode])

    assert found.exit_code == 0, found.output
    assert "people/leave" in found.output


def test_eval_retrieval_reports_the_mode_it_ran_in(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q001",
                "question": "how many weeks of paid parental leave do new parents get",
                "reference_answer": "Sixteen weeks.",
                "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
                "type": "answerable",
                "origin": "hand-written",
            }
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        [
            "eval-retrieval",
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "runs"),
            "--variant",
            "v1",
            "--mode",
            "hybrid",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "mode        hybrid" in result.output
    assert (tmp_path / "runs" / "v1" / "results.jsonl").is_file()


def test_search_can_rerank_its_results(fake_pipeline: FakeEmbedder) -> None:
    runner.invoke(app, ["index"])

    found = runner.invoke(
        app, ["search", "laptop expense dollars", "--limit", "1", "--mode", "hybrid", "--rerank"]
    )

    assert found.exit_code == 0, found.output
    assert "finance/expenses" in found.output
    assert "1. 3.000" in found.output


def test_golden_candidates_samples_passages_worth_asking_about(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    # The fixture's pages are too short to write a question from, so add one that is not.
    prose = "Team members submit an expense report within thirty days of the purchase. " * 14
    write_snapshot(tmp_path / "data", "a" * 40, {"legal/policy.md": f"## Policy\n\n{prose}"})
    output = tmp_path / "golden" / "candidates.jsonl"

    result = runner.invoke(app, ["golden-candidates", str(output), "--count", "5", "--seed", "1"])

    assert result.exit_code == 0, result.output
    assert "Sampled 1 passages from 1 sections (seed 1)" in result.output
    (candidate,) = [json.loads(line) for line in output.read_text().splitlines()]
    assert candidate["candidate_id"] == "c001"
    assert candidate["doc_id"] == "legal/policy.md"
    assert "thirty days" in candidate["text"]


def golden_rows(quote: str) -> str:
    rows = [
        {
            "id": "q001",
            "question": "How long is parental leave?",
            "reference_answer": "Sixteen weeks, paid.",
            "evidence": [{"doc_id": "people/leave.md", "quote": quote}],
            "type": "answerable",
            "origin": "hand-written",
        },
        {
            "id": "q002",
            "question": "What is the dress code in the office on Mars?",
            "type": "unanswerable",
            "origin": "hand-written",
            "unanswerable_kind": "false-premise",
        },
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


@pytest.mark.usefixtures("fake_pipeline")
def test_golden_check_summarises_a_set_whose_labels_match_the_corpus(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))

    result = runner.invoke(app, ["golden-check", "--dataset", str(dataset)])

    assert result.exit_code == 0, result.output
    assert "Golden set: 2 questions" in result.output
    assert "1 answerable, 1 unanswerable" in result.output
    assert "declining 1 false-premise" in result.output
    assert "every quote was found in its page, and on no other page" in result.output


@pytest.mark.usefixtures("fake_pipeline")
def test_golden_check_fails_when_a_quote_is_not_in_its_page(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("twenty weeks of paid leave"))

    result = runner.invoke(app, ["golden-check", "--dataset", str(dataset)])

    assert result.exit_code == 1
    assert "1 broken" in result.output
    assert "q001: quote not found in people/leave.md" in result.output


@pytest.mark.usefixtures("fake_pipeline")
def test_golden_review_writes_a_page_with_the_evidence_marked(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    output = tmp_path / "review" / "page.html"

    result = runner.invoke(
        app, ["golden-review", "--dataset", str(dataset), "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert "Review page for 2 questions (2 waiting for review)" in result.output
    assert "<mark>sixteen weeks of paid leave</mark>" in output.read_text()


@pytest.mark.usefixtures("fake_pipeline")
def test_golden_review_defaults_to_the_data_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))

    result = runner.invoke(app, ["golden-review", "--dataset", str(dataset)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "golden-review.html").is_file()
