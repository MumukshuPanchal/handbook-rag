import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from handbook_rag import __version__, wiring
from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.index.store import QdrantVectorStore, ScoredChunk, VectorStoreError
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.dense import Retriever
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
    assert "Retrieval evaluation: current" in result.output
    (row,) = (output / "current" / "results.jsonl").read_text().splitlines()
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
    assert "by origin: hand-written" in result.output


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
    assert "Review page for 2 questions (2 not yet verified by a person)" in result.output
    assert "<mark>sixteen weeks of paid leave</mark>" in output.read_text()


@pytest.mark.usefixtures("fake_pipeline")
def test_golden_review_defaults_to_the_data_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))

    result = runner.invoke(app, ["golden-review", "--dataset", str(dataset)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "golden-review.html").is_file()


def test_eval_compare_reports_the_difference_between_two_runs(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    runs = tmp_path / "runs"
    common = ["eval-retrieval", "--dataset", str(dataset), "--output", str(runs)]
    runner.invoke(
        app, [*common, "--variant", "baseline", "--mode", "dense", "--description", "Dense."]
    )
    runner.invoke(app, [*common, "--variant", "v1", "--mode", "hybrid", "--description", "Hybrid."])

    result = runner.invoke(app, ["eval-compare", "baseline", "v1", "--runs", str(runs)])

    assert result.exit_code == 0, result.output
    assert "v1 against baseline, on the same 1 questions" in result.output
    assert "baseline    Dense." in result.output
    assert "v1          Hybrid." in result.output
    assert "Hit@5" in result.output
    assert "+0.000 (+0.000 to +0.000)" in result.output
    assert "unclear" in result.output
    summary = json.loads((runs / "v1" / "summary.json").read_text())
    assert summary["config"]["retrieval_mode"] == "hybrid"
    assert summary["config"]["rerank"] is False


def test_eval_compare_explains_a_run_that_does_not_exist(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval-compare", "baseline", "v9", "--runs", str(tmp_path)])

    assert result.exit_code == 1
    assert "no run found" in result.output


def test_eval_table_prints_one_row_per_run(fake_pipeline: FakeEmbedder, tmp_path: Path) -> None:
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    runs = tmp_path / "runs"
    common = ["eval-retrieval", "--dataset", str(dataset), "--output", str(runs)]
    runner.invoke(
        app, [*common, "--variant", "baseline", "--mode", "dense", "--description", "Dense."]
    )
    runner.invoke(app, [*common, "--variant", "v1", "--mode", "hybrid", "--description", "Hybrid."])

    table = runner.invoke(
        app, ["eval-table", "--runs", str(runs), "--metric", "hit@5", "--metric", "mrr@10"]
    )
    unknown = runner.invoke(app, ["eval-table", "--runs", str(runs), "--metric", "bleu"])
    empty = runner.invoke(app, ["eval-table", "--runs", str(tmp_path / "nowhere")])

    assert table.exit_code == 0, table.output
    lines = table.output.splitlines()
    assert lines[0].startswith("| Run | What it is | Questions | Hit@5 (95% interval) | MRR@10 |")
    assert lines[2].startswith("| baseline | Dense. | 1 | 1.000 (1.000 to 1.000) | 1.000 |")
    assert lines[3].startswith("| v1 | Hybrid. | 1 | 1.000 (1.000 to 1.000) | 1.000 |")
    assert unknown.exit_code == 1
    assert "no run has the metric: bleu" in unknown.output
    assert empty.exit_code == 1
    assert "no runs found" in empty.output


def run_summary(runs: Path, variant: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((runs / variant / "summary.json").read_text())
    return loaded


def test_eval_retrieval_describes_a_run_by_its_mode_unless_told_otherwise(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    runs = tmp_path / "runs"
    common = ["eval-retrieval", "--dataset", str(dataset), "--output", str(runs)]

    default = runner.invoke(app, common)
    runner.invoke(app, [*common, "--variant", "v1", "--mode", "sparse", "--rerank"])

    assert default.exit_code == 0, default.output
    assert "mode        hybrid\n" in default.output
    assert run_summary(runs, "current")["description"] == (
        "Dense and BM25 merged with reciprocal rank fusion."
    )
    assert run_summary(runs, "current")["config"]["retrieval_mode"] == "hybrid"
    assert run_summary(runs, "current")["config"]["rerank"] is False
    assert run_summary(runs, "v1")["description"] == (
        "Keyword search only (BM25). Then reranked by a cross-encoder."
    )


def test_eval_retrieval_without_a_name_never_writes_the_baseline(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    """The committed run named ``baseline`` is the dense run every variant is compared with.

    The default retrieval mode is no longer dense, so a run that is given no name must
    not land there: it would silently replace the baseline with a different system.
    """
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    runs = tmp_path / "runs"
    common = ["eval-retrieval", "--dataset", str(dataset), "--output", str(runs)]
    runner.invoke(app, [*common, "--variant", "baseline", "--mode", "dense"])
    baseline_before = (runs / "baseline" / "summary.json").read_text()

    unnamed = runner.invoke(app, common)

    assert unnamed.exit_code == 0, unnamed.output
    assert sorted(path.name for path in runs.iterdir() if path.is_dir()) == [
        "baseline",
        "current",
    ]
    assert (runs / "baseline" / "summary.json").read_text() == baseline_before
    assert run_summary(runs, "baseline")["config"]["retrieval_mode"] == "dense"


def test_eval_retrieval_takes_its_mode_and_reranking_from_the_settings(
    fake_pipeline: FakeEmbedder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The settings are read once and then cached, so they are set before any command runs.
    monkeypatch.setenv("HANDBOOK_RAG_RETRIEVAL_MODE", "sparse")
    monkeypatch.setenv("HANDBOOK_RAG_RERANK", "true")
    reranker = FakeReranker()
    monkeypatch.setattr("handbook_rag.wiring.make_reranker", lambda _settings: reranker)
    runner.invoke(app, ["index"])
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    runs = tmp_path / "runs"
    common = ["eval-retrieval", "--dataset", str(dataset), "--output", str(runs)]

    from_settings = runner.invoke(app, common)
    reranked_by_the_first_run = len(reranker.calls)
    without_reranking = runner.invoke(app, [*common, "--variant", "v1", "--no-rerank"])

    # What ran, and not only what was written down: keyword search never embeds the
    # question, and the reranker is called once for the one answerable question.
    assert fake_pipeline.queries == []
    assert reranked_by_the_first_run == 1
    assert len(reranker.calls) == 1, "--no-rerank must not call the reranker"
    assert from_settings.exit_code == 0, from_settings.output
    assert "mode        sparse, reranked with" in from_settings.output
    assert run_summary(runs, "current")["description"] == (
        "Keyword search only (BM25). Then reranked by a cross-encoder."
    )
    assert run_summary(runs, "current")["config"]["retrieval_mode"] == "sparse"
    assert run_summary(runs, "current")["config"]["rerank"] is True
    assert without_reranking.exit_code == 0, without_reranking.output
    assert "mode        sparse\n" in without_reranking.output
    assert run_summary(runs, "v1")["description"] == "Keyword search only (BM25)."
    assert run_summary(runs, "v1")["config"]["retrieval_mode"] == "sparse"
    assert run_summary(runs, "v1")["config"]["rerank"] is False


def gate_command(
    tmp_path: Path, minimums: dict[str, float], *, distractor_pages: int = 0
) -> list[str]:
    """The eval-gate command line for the two-page handbook of ``fake_pipeline``.

    The seed differs from the number of extra pages on purpose, so that swapping the two
    would change which pages are indexed.
    """
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(golden_rows("sixteen weeks of paid leave"))
    config = tmp_path / "gate.json"
    config.write_text(
        json.dumps({"distractor_pages": distractor_pages, "seed": 5, "minimums": minimums})
    )
    return ["eval-gate", "--dataset", str(dataset), "--config", str(config)]


def test_eval_gate_indexes_only_its_slice_in_memory_and_passes(
    fake_pipeline: FakeEmbedder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_server(_settings: Settings) -> QdrantVectorStore:
        raise VectorStoreError("the gate must not need a vector store server")

    monkeypatch.setattr("handbook_rag.wiring.make_store", no_server)

    result = runner.invoke(app, gate_command(tmp_path, {"hit@5": 1.0, "mrr@10": 1.0}))

    assert result.exit_code == 0, result.output
    assert "pages       1 of 2: 1 with evidence, 0 without (seed 5)" in result.output
    assert "chunks      1, indexed in memory" in result.output
    assert "mode        hybrid\n" in result.output
    assert "1 scored, 0 failed, 1 unanswerable" in result.output
    assert "Hit@5       1.000  minimum 1.000  ok" in result.output
    assert "MRR@10      1.000  minimum 1.000  ok" in result.output
    assert "Gate passed." in result.output
    (embedded,) = fake_pipeline.embedded
    assert "sixteen weeks" in embedded, "only the page with evidence is embedded"
    assert len(fake_pipeline.queries) == 1, "the default retriever, hybrid, embeds the question"


def test_eval_gate_adds_the_extra_pages_its_settings_ask_for(
    fake_pipeline: FakeEmbedder, tmp_path: Path
) -> None:
    command = gate_command(tmp_path, {"hit@5": 1.0}, distractor_pages=1)

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert "pages       2 of 2: 1 with evidence, 1 without (seed 5)" in result.output
    assert len(fake_pipeline.embedded) == 2


def test_eval_gate_scores_the_retriever_the_settings_choose(
    fake_pipeline: FakeEmbedder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The settings are read once and then cached, so they are set before the command runs.
    monkeypatch.setenv("HANDBOOK_RAG_RETRIEVAL_MODE", "sparse")
    monkeypatch.setenv("HANDBOOK_RAG_RERANK", "true")
    reranker = FakeReranker()
    monkeypatch.setattr("handbook_rag.wiring.make_reranker", lambda _settings: reranker)
    command = gate_command(tmp_path, {"hit@5": 1.0}, distractor_pages=1)

    result = runner.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert "mode        sparse, reranked with" in result.output
    # What ran, and not only what was printed: keyword search never embeds the question,
    # and the reranker is called once for the one answerable question.
    assert fake_pipeline.queries == []
    assert len(reranker.calls) == 1


class WorstFirst:
    """A retriever with a ranking bug: it returns what it found in reverse order."""

    def __init__(self, base: Retriever) -> None:
        self._base = base

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        return list(reversed(self._base.retrieve(query, limit=limit, section=section)))


class NothingFound:
    """A retriever that has stopped finding anything."""

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        return []


class Unreachable:
    """A retriever whose vector store has gone away."""

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        raise VectorStoreError("connection refused")


@pytest.mark.usefixtures("fake_pipeline")
def test_eval_gate_fails_when_ranking_gets_worse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = wiring.make_retriever

    def broken(*args: Any, **kwargs: Any) -> Retriever:
        return WorstFirst(real(*args, **kwargs))

    monkeypatch.setattr("handbook_rag.wiring.make_retriever", broken)
    command = gate_command(tmp_path, {"hit@5": 1.0, "mrr@10": 0.9}, distractor_pages=1)

    result = runner.invoke(app, command)

    assert result.exit_code == 1
    assert "Hit@5       1.000  minimum 1.000  ok" in result.output
    assert "MRR@10      0.500  minimum 0.900  BELOW" in result.output
    assert "Gate failed:" in result.output
    assert "mrr@10 is 0.500, below the minimum of 0.900" in result.output
    assert "Gate passed" not in result.output


@pytest.mark.usefixtures("fake_pipeline")
def test_eval_gate_fails_when_a_question_cannot_be_searched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "handbook_rag.wiring.make_retriever", lambda *_args, **_kwargs: Unreachable()
    )

    result = runner.invoke(app, gate_command(tmp_path, {"hit@5": 0.5}))

    assert result.exit_code == 1
    assert "0 scored, 1 failed" in result.output
    assert "1 of 1 questions could not be searched; the first was q001" in result.output
    assert "VectorStoreError: connection refused" in result.output
    assert "Gate passed" not in result.output


@pytest.mark.usefixtures("fake_pipeline")
def test_eval_gate_names_the_questions_it_did_not_find(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "handbook_rag.wiring.make_retriever", lambda *_args, **_kwargs: NothingFound()
    )

    result = runner.invoke(app, gate_command(tmp_path, {"hit@5": 1.0}))

    assert result.exit_code == 1
    assert "Hit@5       0.000  minimum 1.000  BELOW" in result.output
    assert "not found in the top five: q001" in result.output
    assert "hit@5 is 0.000, below the minimum of 1.000" in result.output


@pytest.mark.usefixtures("fake_pipeline")
def test_eval_gate_explains_missing_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    command = gate_command(tmp_path, {"hit@5": 1.0})

    no_settings = runner.invoke(app, [*command[:3], "--config", str(tmp_path / "none.json")])
    no_questions = runner.invoke(app, ["eval-gate", "--dataset", "none.jsonl", *command[3:]])
    get_settings.cache_clear()
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "empty"))
    no_snapshot = runner.invoke(app, command)

    assert no_settings.exit_code == 1
    assert "gate settings not found" in no_settings.output
    assert no_questions.exit_code == 1
    assert "golden set not found" in no_questions.output
    assert no_snapshot.exit_code == 1
    assert "handbook-rag snapshot" in no_snapshot.output


class MissingModelFile:
    """An embedding model whose files are not in the local cache, as when working offline."""

    name = "missing-model"
    dimension = 64

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise FileNotFoundError("model.safetensors is not in the local cache")

    def embed_query(self, text: str) -> list[float]:
        raise FileNotFoundError("model.safetensors is not in the local cache")


@pytest.mark.usefixtures("fake_pipeline")
def test_a_missing_model_file_is_not_reported_as_a_missing_golden_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("handbook_rag.wiring.make_embedder", lambda _settings: MissingModelFile())

    result = runner.invoke(app, gate_command(tmp_path, {"hit@5": 1.0}))

    assert result.exit_code == 1
    assert "golden set not found" not in result.output
    assert isinstance(result.exception, FileNotFoundError)
    assert "model.safetensors" in str(result.exception)


class CannedGenerator:
    """Answers from the first chunk it is given, citing it."""

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        if not chunks:
            return Answer(question, (Segment(NOT_FOUND_MESSAGE),), False, "fake", "no_context")
        text = "Sixteen weeks of paid leave."
        if on_text is not None:
            on_text(text)
        return Answer(
            question=question,
            segments=(Segment(text, (Citation(chunks[0], chunks[0].text),)),),
            answered=True,
            model="claude-haiku-5-5",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=3000, output_tokens=400),
            seconds=1.2,
        )


def test_ask_prints_the_answer_its_sources_and_the_cost(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder
) -> None:
    monkeypatch.setattr("handbook_rag.wiring.make_generator", lambda _settings: CannedGenerator())
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["ask", "how many weeks of parental leave"])

    assert result.exit_code == 0, result.output
    assert "Sixteen weeks of paid leave." in result.output
    assert "[1] Leave > Parental leave" in result.output
    assert "https://handbook.gitlab.com/handbook/people/leave/" in result.output
    assert "3,000 tokens in" in result.output
    assert "$0.00050" in result.output


def test_ask_reports_a_generation_failure_clearly(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder
) -> None:
    class FailingGenerator:
        def generate(self, *_args: object, **_kwargs: object) -> Answer:
            raise GenerationError("No Claude API credential was found.")

    monkeypatch.setattr("handbook_rag.wiring.make_generator", lambda _settings: FailingGenerator())
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["ask", "anything about parental leave"])

    assert result.exit_code == 1
    assert "No Claude API credential" in result.output


def test_ask_without_a_key_says_where_to_put_one(fake_pipeline: FakeEmbedder) -> None:
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["ask", "how many weeks of parental leave"])

    assert result.exit_code == 1
    assert "error: No Gemini API key was found" in result.output
    assert "GEMINI_API_KEY" in result.output
    assert "Traceback" not in result.output


class FixedReply:
    """A generator that gives one prepared reply, citing the first chunk it is handed."""

    def __init__(self, text: str, *, answered: bool, stop_reason: str) -> None:
        self._text = text
        self._answered = answered
        self._stop_reason = stop_reason

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        if on_text is not None:
            on_text(self._text)
        return Answer(
            question=question,
            segments=(Segment(self._text, (Citation(chunks[0]),)),),
            answered=self._answered,
            model="gemini-3.5-flash-lite",
            stop_reason=self._stop_reason,
        )


def ask_with(monkeypatch: pytest.MonkeyPatch, reply: FixedReply) -> str:
    monkeypatch.setattr("handbook_rag.wiring.make_generator", lambda _settings: reply)
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["ask", "how many weeks of parental leave"])
    assert result.exit_code == 0, result.output
    return result.output


def test_ask_lists_no_sources_for_a_reply_that_is_not_an_answer(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder
) -> None:
    # The model declined, and still attached a citation to the sentence.
    output = ask_with(
        monkeypatch, FixedReply(NOT_FOUND_MESSAGE, answered=False, stop_reason="end_turn")
    )

    assert NOT_FOUND_MESSAGE in output
    assert "Sources" not in output
    assert "Treat it with caution" not in output
    assert "stopped early" not in output


def test_ask_prints_no_dollar_figure_for_a_model_without_a_known_price(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder
) -> None:
    output = ask_with(
        monkeypatch, FixedReply("Sixteen weeks.", answered=True, stop_reason="end_turn")
    )

    assert "Sources" in output
    assert "gemini-3.5-flash-lite | 0 tokens in" in output
    assert "$" not in output


@pytest.mark.parametrize(
    ("stop_reason", "note"),
    [
        ("max_tokens", "cut off at the token limit"),
        ("refusal", "withheld by the model's provider"),
        ("other", "stopped early (other)"),
        ("unknown", "stopped early (unknown)"),
    ],
)
def test_ask_says_when_the_reply_did_not_come_to_a_proper_end(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder, stop_reason: str, note: str
) -> None:
    reply = FixedReply(
        "Sixteen weeks. And", answered=stop_reason == "max_tokens", stop_reason=stop_reason
    )

    assert note in ask_with(monkeypatch, reply)


@pytest.mark.parametrize("stop_reason", ["end_turn", "no_context"])
def test_ask_adds_no_note_when_the_reply_ended_normally(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder, stop_reason: str
) -> None:
    reply = FixedReply(NOT_FOUND_MESSAGE, answered=False, stop_reason=stop_reason)

    output = ask_with(monkeypatch, reply)

    assert "stopped early" not in output
    assert "withheld" not in output
    assert "cut off" not in output


# eval-answers: a real run sends one request per question, so these use a stand-in model.

ANSWER_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "q001",
        "question": "how many weeks of parental leave",
        "reference_answer": "Sixteen weeks.",
        "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
        "type": "answerable",
        "origin": "model-drafted",
    },
    {
        "id": "u001",
        "question": "how much is the parking allowance",
        "type": "unanswerable",
        "origin": "model-drafted",
        "unanswerable_kind": "near-miss",
    },
]


class EvalModel:
    """Answers from the first chunk, declines anything about parking, and counts its calls."""

    def __init__(self, *, error: GenerationError | None = None) -> None:
        self.questions: list[str] = []
        self.built_with: list[dict[str, Any]] = []
        self._error = error

    def build(self, _settings: Settings, **options: Any) -> "EvalModel":
        self.built_with.append(options)
        return self

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        declined = "parking" in question
        text = NOT_FOUND_MESSAGE if declined else "Sixteen weeks of paid leave."
        return Answer(
            question=question,
            segments=(Segment(text, () if declined else (Citation(chunks[0]),)),),
            answered=not declined,
            model="fake",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1800, output_tokens=90),
            seconds=2.0,
            seconds_to_first_text=1.0,
        )


@pytest.fixture
def answer_run(
    monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakeEmbedder, tmp_path: Path
) -> list[str]:
    """An indexed fake handbook, a two-question golden set, and no waiting between requests."""
    golden = tmp_path / "questions.jsonl"
    golden.write_text("".join(json.dumps(question) + "\n" for question in ANSWER_QUESTIONS))
    monkeypatch.setattr("handbook_rag.evals.answers.time.sleep", lambda _seconds: None)
    assert runner.invoke(app, ["index"]).exit_code == 0
    return ["eval-answers", "--dataset", str(golden), "--output", str(tmp_path / "runs")]


def test_eval_answers_asks_every_question_once_and_writes_the_run(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    model = EvalModel()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, [*answer_run, "--variant", "baseline"])

    assert result.exit_code == 0, result.output
    assert sorted(model.questions) == sorted(q["question"] for q in ANSWER_QUESTIONS)
    # One attempt per request, so the run knows how many it sent.
    assert model.built_with == [{"attempts": 1}]
    assert "2 in the golden set, 0 already answered, 2 to ask now" in result.output
    assert "at most 10 requests a minute, so at least 1 min" in result.output
    assert "2 scored, 0 failed, 0 not asked yet" in result.output
    assert "at most 2 now, 2 for this run in all" in result.output
    assert "Right call                   1.000  (95% interval 1.000 to 1.000, n=2)" in result.output
    assert "Declined, near-miss questions: 1.000 (n=1)" in result.output
    assert "Answered, evidence given: 1.000 (n=1)" in result.output
    assert "median 2.0 s" in result.output
    assert "first words after 1.0 s" in result.output
    assert "1,800 in and 90 out per question" in result.output
    assert "Run the same command again" not in result.output

    run_dir = tmp_path / "runs" / "baseline"
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["description"] == "Answers by gemini-3.5-flash-lite from 5 passages."
    assert summary["config"]["answer_model"] == "gemini-3.5-flash-lite"
    assert summary["config"]["retrieval_mode"] == "hybrid"
    assert summary["attempts"] == 2
    assert json.loads((run_dir / "run.json").read_text())["attempts"] == 2


def test_eval_answers_continues_a_run_and_does_not_ask_twice(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    model = EvalModel()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    first = runner.invoke(app, [*answer_run, "--max-requests", "1"])

    # A limit the caller set is a planned stop, not a failure.
    assert first.exit_code == 0, first.output
    assert len(model.questions) == 1
    assert "1 to ask now" in first.output
    assert "1 scored, 0 failed, 1 not asked yet" in first.output
    assert "stopped: the limit of 1 requests for this run was reached" in first.output
    assert "Run the same command again to continue this run." in first.output

    second = runner.invoke(app, answer_run)

    assert second.exit_code == 0, second.output
    assert len(model.questions) == 2
    assert len(set(model.questions)) == 2
    assert "1 already answered, 1 to ask now" in second.output
    assert "at most 1 now, 2 for this run in all" in second.output

    third = runner.invoke(app, answer_run)

    assert third.exit_code == 0, third.output
    assert len(model.questions) == 2
    assert "2 already answered, 0 to ask now" in third.output
    assert "requests a minute" not in third.output
    assert "2 scored, 0 failed, 0 not asked yet" in third.output


def test_eval_answers_does_not_continue_a_run_made_with_another_model(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str]
) -> None:
    model = EvalModel()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)
    assert runner.invoke(app, answer_run).exit_code == 0

    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_MODEL", "gemini-9-other")
    get_settings.cache_clear()
    loaded: list[str] = []
    real_retriever = wiring.make_retriever

    def loading_retriever(settings: Settings) -> Retriever:
        loaded.append("retriever")
        return real_retriever(settings)

    monkeypatch.setattr("handbook_rag.wiring.make_retriever", loading_retriever)
    refused = runner.invoke(app, answer_run)

    assert refused.exit_code == 1
    assert "was made with other settings (answer_model)" in refused.output
    assert len(model.questions) == 2
    # It was refused before any model was loaded, which on a real run takes a while.
    assert loaded == []

    restarted = runner.invoke(app, [*answer_run, "--restart"])

    assert restarted.exit_code == 0, restarted.output
    assert len(model.questions) == 4
    assert "0 already answered, 2 to ask now" in restarted.output


def test_eval_answers_fails_clearly_when_the_model_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    model = EvalModel(error=GenerationError("Today's limit is used up."))
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, answer_run)

    assert result.exit_code == 1
    assert "no question was answered" in result.output
    assert "Today's limit is used up." in result.output
    # Each question was tried, and tried once more. Nothing was scored or written as a result.
    assert len(model.questions) == 3
    assert not (tmp_path / "runs" / "current" / "summary.json").exists()
    assert json.loads((tmp_path / "runs" / "current" / "run.json").read_text())["attempts"] == 3


def test_eval_answers_reports_a_question_that_kept_failing(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    class FailsOnParking(EvalModel):
        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            if "parking" in question:
                self.questions.append(question)
                raise GenerationError("The Gemini API sent a reply that could not be read.")
            return super().generate(question, chunks, on_text=on_text)

    model = FailsOnParking()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, answer_run)

    assert result.exit_code == 1
    assert "1 scored, 1 failed, 0 not asked yet" in result.output
    # Each of the two tries is shown as it fails, and the question is listed at the end.
    assert result.output.count("failed  u001  The Gemini API sent a reply that could not") == 2
    assert "failed: u001: The Gemini API sent a reply that could not be read." in result.output
    assert "Run the same command again to continue this run." in result.output
    errors = (tmp_path / "runs" / "current" / "errors.jsonl").read_text()
    assert json.loads(errors)["question_id"] == "u001"


def test_eval_answers_without_a_key_says_where_to_put_one_and_leaves_no_run_behind(
    answer_run: list[str], tmp_path: Path
) -> None:
    result = runner.invoke(app, answer_run)

    assert result.exit_code == 1
    assert "GEMINI_API_KEY" in result.output
    assert not (tmp_path / "runs").exists()


def test_eval_answers_shows_the_cause_when_a_limit_ended_a_run_that_answered_nothing(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str]
) -> None:
    model = EvalModel(error=GenerationError("Today's limit is used up."))
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, [*answer_run, "--max-requests", "1"])

    assert result.exit_code == 1
    assert "no question was answered" in result.output
    assert "the limit of 1 requests for this run was reached" in result.output
    # The reason the one request failed is shown as well, not only the limit.
    assert "Today's limit is used up." in result.output


def test_eval_answers_counts_a_request_before_it_is_sent(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    class Interrupted(EvalModel):
        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            if self.questions:
                raise KeyboardInterrupt  # Ctrl-C during the second request
            return super().generate(question, chunks, on_text=on_text)

    model = Interrupted()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, answer_run)

    assert result.exit_code != 0
    run_dir = tmp_path / "runs" / "current"
    # Both requests are counted, and the reply that arrived is kept.
    assert json.loads((run_dir / "run.json").read_text())["attempts"] == 2
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 1


def test_eval_answers_reports_a_reply_that_cites_a_passage_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    class CitesElsewhere(EvalModel):
        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            stranger = dataclasses.replace(chunks[0], chunk_id="a-chunk-that-was-not-given")
            return super().generate(question, [stranger], on_text=on_text)

    model = CitesElsewhere()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, answer_run)

    # A short message and a failing exit, not a traceback.
    assert result.exit_code == 1
    assert "cites a passage the model was not given (a-chunk-that-was-not-given)" in result.output
    assert not isinstance(result.exception, ValueError)
    # Every request that was sent is counted, the one whose reply could not be kept too.
    attempts = json.loads((tmp_path / "runs" / "current" / "run.json").read_text())["attempts"]
    assert attempts == len(model.questions) >= 1


def test_eval_compare_names_a_score_it_could_not_compare(tmp_path: Path) -> None:
    for name, grade in (("baseline", {"answered": 1.0}), ("v1", {"fully_cited": 1.0})):
        run_dir = tmp_path / name
        run_dir.mkdir()
        row = {
            "prompt_id": "q1",
            "status": "ok",
            "fingerprint": "label-q1",
            "grade": {"right_call": 1.0, **grade},
        }
        (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
        (run_dir / "summary.json").write_text(json.dumps({"description": name}))

    result = runner.invoke(app, ["eval-compare", "baseline", "v1", "--runs", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (
        "not compared, because no question has the score in both runs: answered, fully_cited"
        in result.output
    )


def test_eval_answers_describes_the_run_by_the_passages_it_really_gave(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_CONTEXT_CHUNKS", "1")
    get_settings.cache_clear()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", EvalModel().build)

    assert runner.invoke(app, answer_run).exit_code == 0

    summary = json.loads((tmp_path / "runs" / "current" / "summary.json").read_text())
    assert summary["description"] == "Answers by gemini-3.5-flash-lite from 1 passages."
    assert summary["config"]["answer_context_chunks"] == 1


def test_answer_runs_can_be_compared_and_tabled_like_retrieval_runs(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    class AnswersEverything(EvalModel):
        """A worse assistant: it answers the question about parking too."""

        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            return super().generate(question.replace("parking", "car"), chunks, on_text=on_text)

    runs = str(tmp_path / "runs")
    monkeypatch.setattr("handbook_rag.wiring.make_generator", EvalModel().build)
    assert runner.invoke(app, [*answer_run, "--variant", "baseline"]).exit_code == 0
    monkeypatch.setattr("handbook_rag.wiring.make_generator", AnswersEverything().build)
    assert runner.invoke(app, [*answer_run, "--variant", "v1"]).exit_code == 0
    # A third run that stopped before it reached the unanswerable question.
    assert (
        runner.invoke(app, [*answer_run, "--variant", "v2", "--max-requests", "1"]).exit_code == 0
    )

    compared = runner.invoke(app, ["eval-compare", "baseline", "v1", "--runs", runs])

    assert compared.exit_code == 0, compared.output
    assert "v1 against baseline, on the same 2 questions" in compared.output
    lines = {line.split()[0]: line for line in compared.output.splitlines() if line.strip()}
    assert "-0.500" in lines["Right"]
    # A score that applies to one of the two questions says so.
    assert lines["Declines"].endswith("(on 1 question)")
    assert lines["Answers"].endswith("(on 1 question)")
    assert "(on" not in lines["Right"]

    # A run that stopped early did not score the same questions, and is refused as before.
    partial = runner.invoke(app, ["eval-compare", "baseline", "v2", "--runs", runs])
    assert partial.exit_code == 1
    assert "did not score the same questions" in partial.output

    table = runner.invoke(
        app, ["eval-table", "--runs", runs, "--metric", "right_call", "--metric", "declined"]
    )

    assert table.exit_code == 0, table.output
    rows = table.output.splitlines()
    assert rows[0].startswith(
        "| Run | What it is | Questions | Right call (95% interval) | Declines"
    )
    assert rows[2].startswith(
        "| baseline | Answers by gemini-3.5-flash-lite from 5 passages. | 2 | 1.000"
    )
    assert "| 1.000 |" in rows[2]
    assert "| 0.000 |" in rows[3]
    everything = runner.invoke(app, ["eval-table", "--runs", runs])
    assert everything.exit_code == 0, everything.output
    assert len(everything.output.splitlines()) == 5


def test_eval_answers_explains_a_missing_golden_set(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    model = EvalModel()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)

    result = runner.invoke(app, ["eval-answers", "--dataset", str(tmp_path / "missing.jsonl")])

    assert result.exit_code == 1
    assert "golden set not found" in result.output
    assert model.questions == []


def test_eval_answers_stops_on_a_broken_label_before_any_request(
    monkeypatch: pytest.MonkeyPatch, answer_run: list[str], tmp_path: Path
) -> None:
    model = EvalModel()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)
    broken = {
        **ANSWER_QUESTIONS[0],
        "evidence": [{"doc_id": "people/leave.md", "quote": "twenty weeks of paid leave"}],
    }
    golden = tmp_path / "broken.jsonl"
    golden.write_text(json.dumps(broken) + "\n")

    result = runner.invoke(app, ["eval-answers", "--dataset", str(golden)])

    assert result.exit_code == 1
    assert "quote not found in people/leave.md" in result.output
    assert model.questions == []
