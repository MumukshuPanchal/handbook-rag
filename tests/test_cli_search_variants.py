"""The variants of hybrid search that ``eval-retrieval`` can measure (issue 78).

A variant is asked for on the command line of one run: another number of candidates,
another fusion constant, or the rule that spreads the merged list over pages. These tests
hold three things. Without such an option the command builds and records the search as
it always did. With one, the option reaches the search and the run says so. And no other
command can ask for a variant: the gate, a search and an answer use the default.

The stand-in handbook has a page of six short chunks about leave and a page of one chunk
about laptops. The question needs a passage of each page, and shares no word with the
second one, so the default search returns the six chunks of the first page before it.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from handbook_rag import wiring
from handbook_rag.cli import app
from handbook_rag.config import get_settings
from handbook_rag.index.store import QdrantVectorStore
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.spread import PageSpread
from tests.support import (
    FakeEmbedder,
    FakeReranker,
    FakeSparseEncoder,
    count_words,
    write_snapshot,
)
from tests.test_cli_eval_cache import COLOUR_CODES

runner = CliRunner()

VARIANT_OPTIONS = {"--fusion-candidates", "--rrf-k", "--spread-per-page", "--spread-within"}
LEAVE = (
    '---\ntitle: "Leave"\n---\n\n'
    "## Parental leave\n\nNew parents get sixteen weeks of paid parental leave.\n\n"
    "## Blocks\n\nPaid parental leave weeks can be split in two blocks.\n\n"
    "## Notice\n\nParental leave weeks need notice in writing.\n\n"
    "## Return\n\nAfter parental leave weeks parents return part time.\n\n"
    "## Records\n\nPaid leave weeks are recorded for new parents.\n\n"
    "## Cover\n\nA colleague covers the weeks of parental leave.\n"
)
EXPENSES = "## Laptops\n\nThe laptop limit is one thousand dollars.\n"
QUESTION = {
    "id": "t001",
    "question": (
        "how many weeks of paid parental leave do new parents get "
        "and what may they spend on a computer"
    ),
    "reference_answer": "Sixteen weeks, and one thousand dollars.",
    "evidence": [
        {"doc_id": "people/leave.md", "quote": "sixteen weeks"},
        {"doc_id": "finance/expenses.md", "quote": "one thousand dollars"},
    ],
    "type": "answerable",
    "origin": "model-drafted",
}
# What a run of the default search records, in this order, since before the variants.
DEFAULT_CONFIG = [
    "handbook_commit",
    "chunking",
    "collection",
    "retrieval_mode",
    "embedding_model",
    "sparse_model",
    "fusion_candidates",
    "rrf_k",
    "rerank",
]


class Stand:
    """The stand-in handbook on disk, and what the commands built to search it."""

    def __init__(self, tmp_path: Path) -> None:
        self.runs = tmp_path / "runs"
        self.dataset = tmp_path / "two.jsonl"
        self.dataset.write_text(json.dumps(QUESTION) + "\n")
        self.embedder = FakeEmbedder()
        self.built: list[dict[str, Any]] = []
        """The options each hybrid search was built with, in the order they were built."""
        self.asked_with: list[dict[str, Any]] = []
        """The keyword arguments of each call that asked the wiring for a retriever."""

    def run(self, variant: str, *options: str) -> Any:
        return runner.invoke(
            app,
            [
                "eval-retrieval",
                "--dataset",
                str(self.dataset),
                "--output",
                str(self.runs),
                "--variant",
                variant,
                *options,
            ],
        )

    def summary(self, variant: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads((self.runs / variant / "summary.json").read_text())
        return loaded

    def row(self, variant: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads((self.runs / variant / "results.jsonl").read_text())
        return loaded

    def retrieved(self, variant: str) -> list[dict[str, Any]]:
        line = json.loads((self.runs / variant / "retrieved.jsonl").read_text())
        found: list[dict[str, Any]] = line["retrieved"]
        return found


@pytest.fixture
def stand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Stand:
    """An indexed stand-in handbook whose first page is cut into six chunks."""
    commit = "a" * 40
    write_snapshot(
        tmp_path / "data", commit, {"people/leave.md": LEAVE, "finance/expenses.md": EXPENSES}
    )
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    # One section of a page is one chunk.
    monkeypatch.setenv("HANDBOOK_RAG_CHUNK_MAX_TOKENS", "12")
    monkeypatch.setenv("HANDBOOK_RAG_CHUNK_MIN_TOKENS", "1")
    monkeypatch.setenv("HANDBOOK_RAG_CHUNK_OVERLAP_TOKENS", "0")
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)

    stand = Stand(tmp_path)
    client = QdrantClient(":memory:")
    monkeypatch.setattr("handbook_rag.wiring.make_embedder", lambda _settings: stand.embedder)
    monkeypatch.setattr(
        "handbook_rag.wiring.make_sparse_encoder", lambda _settings: FakeSparseEncoder()
    )
    monkeypatch.setattr("handbook_rag.wiring.make_reranker", lambda _settings: FakeReranker())
    monkeypatch.setattr(
        "handbook_rag.wiring.make_store",
        lambda settings: QdrantVectorStore(client, settings.collection_name),
    )
    hybrid, make_retriever = HybridRetriever, wiring.make_retriever

    def noting_hybrid(dense: Retriever, sparse: Retriever, **options: Any) -> Retriever:
        stand.built.append(options)
        return hybrid(dense, sparse, **options)

    def noting_make_retriever(*arguments: Any, **options: Any) -> Retriever:
        stand.asked_with.append(options)
        return make_retriever(*arguments, **options)

    monkeypatch.setattr("handbook_rag.wiring.HybridRetriever", noting_hybrid)
    monkeypatch.setattr("handbook_rag.wiring.make_retriever", noting_make_retriever)

    indexed = runner.invoke(app, ["index"])
    assert indexed.exit_code == 0, indexed.output
    assert "7 in the collection" in indexed.output
    return stand


def test_eval_retrieval_without_a_variant_builds_and_records_the_search_as_before(
    stand: Stand,
) -> None:
    result = stand.run("plain")

    assert result.exit_code == 0, result.output
    assert stand.built == [{"candidates": 30, "rrf_k": 60, "spread": None}]
    assert "\n  variant " not in result.output
    summary = stand.summary("plain")
    assert list(summary["config"]) == DEFAULT_CONFIG
    assert (summary["config"]["fusion_candidates"], summary["config"]["rrf_k"]) == (30, 60)
    assert summary["description"] == "Dense and BM25 merged with reciprocal rank fusion."
    # The six chunks of the first page come before the one chunk of the second.
    assert stand.row("plain")["evidence_ranks"] == [1, 7]
    assert [item["doc_id"] for item in stand.retrieved("plain")] == [
        *["people/leave.md"] * 6,
        "finance/expenses.md",
    ]
    assert stand.row("plain")["grade"]["all@5"] == 0.0


def test_eval_retrieval_spreads_the_results_over_pages_when_a_run_asks_for_it(
    stand: Stand,
) -> None:
    plain = stand.run("plain")
    two = stand.run("two", "--spread-per-page", "2", "--spread-within", "5")
    one = stand.run("one", "--spread-per-page", "1", "--spread-within", "5")

    assert (plain.exit_code, two.exit_code, one.exit_code) == (0, 0, 0), two.output + one.output
    # The second page's chunk takes the place of the first page's third chunk, or second.
    assert stand.row("plain")["evidence_ranks"] == [1, 7]
    assert stand.row("two")["evidence_ranks"] == [1, 3]
    assert stand.row("one")["evidence_ranks"] == [1, 2]
    assert [stand.row(run)["grade"]["all@5"] for run in ("plain", "two", "one")] == [0.0, 1.0, 1.0]
    before = stand.retrieved("plain")
    for run in ("two", "one"):
        after = stand.retrieved(run)
        # The same seven chunks, each with the score the merge gave it, and the chunks
        # of the first page in the order they had.
        assert sorted(item["chunk_id"] for item in after) == sorted(
            item["chunk_id"] for item in before
        )
        assert {item["chunk_id"]: item["score"] for item in after} == {
            item["chunk_id"]: item["score"] for item in before
        }
        assert [item["chunk_id"] for item in after if item["doc_id"] == "people/leave.md"] == [
            item["chunk_id"] for item in before[:6]
        ]
    assert "  variant     at most 2 chunks of a page among the first 5 results\n" in two.output
    assert "  variant     at most 1 chunk of a page among the first 5 results\n" in one.output
    config = stand.summary("two")["config"]
    assert list(config) == [*DEFAULT_CONFIG[:-1], "spread_per_page", "spread_within", "rerank"]
    assert (config["spread_per_page"], config["spread_within"]) == (2, 5)
    assert (config["fusion_candidates"], config["rrf_k"]) == (30, 60)
    assert stand.summary("two")["description"] == (
        "Dense and BM25 merged with reciprocal rank fusion. "
        "Variant: at most 2 chunks of a page among the first 5 results."
    )
    assert [options["spread"] for options in stand.built] == [
        None,
        PageSpread(per_page=2, within=5),
        PageSpread(per_page=1, within=5),
    ]


def test_eval_retrieval_hands_the_candidates_and_the_constant_of_a_variant_to_the_merge(
    stand: Stand,
) -> None:
    variant = stand.run("variant", "--fusion-candidates", "12", "--rrf-k", "7")
    plain = stand.run("plain")
    # As few candidates as the evaluation asks for results is the least that can be measured.
    fewest = stand.run("fewest", "--fusion-candidates", "10")

    assert (variant.exit_code, plain.exit_code, fewest.exit_code) == (0, 0, 0), variant.output
    # The variant is of its one run: the next run is built from the settings again, and
    # the settings every other command reads were not changed.
    assert stand.built == [
        {"candidates": 12, "rrf_k": 7, "spread": None},
        {"candidates": 30, "rrf_k": 60, "spread": None},
        {"candidates": 10, "rrf_k": 60, "spread": None},
    ]
    assert (get_settings().fusion_candidates, get_settings().rrf_k) == (30, 60)
    config = stand.summary("variant")["config"]
    assert list(config) == DEFAULT_CONFIG
    assert (config["fusion_candidates"], config["rrf_k"]) == (12, 7)
    assert stand.summary("variant")["description"] == (
        "Dense and BM25 merged with reciprocal rank fusion. "
        "Variant: 12 candidates from each search; fusion constant 7."
    )
    assert "  variant     12 candidates from each search; fusion constant 7\n" in variant.output
    # The constant reached the merge: the chunk both searches put first scores 1/(7+1)
    # from each of them, and 1/(60+1) in the default search.
    assert stand.retrieved("variant")[0]["score"] == pytest.approx(2 / 8)
    assert stand.retrieved("plain")[0]["score"] == pytest.approx(2 / 61)


@pytest.mark.parametrize(
    ("options", "named"),
    [
        (["--spread-per-page", "1"], "--spread-within"),
        (["--spread-within", "5"], "--spread-per-page"),
        (["--mode", "dense", "--rrf-k", "10"], "dense"),
        (["--mode", "sparse", "--fusion-candidates", "50"], "sparse"),
        (["--mode", "dense", "--spread-per-page", "1", "--spread-within", "5"], "dense"),
        (["--rerank", "--spread-per-page", "1", "--spread-within", "5"], "--no-rerank"),
        (["--fusion-candidates", "0"], "--fusion-candidates"),
        # The evaluation asks for ten results, so nine candidates would be measured as ten.
        (["--fusion-candidates", "9"], "--fusion-candidates"),
        # And the reranker takes thirty merged results, so twenty would be measured as thirty.
        (["--rerank", "--fusion-candidates", "20"], "--fusion-candidates"),
        (["--rrf-k", "0"], "--rrf-k"),
        (["--spread-per-page", "0", "--spread-within", "5"], "--spread-per-page"),
        (["--spread-per-page", "1", "--spread-within", "0"], "--spread-within"),
    ],
)
def test_eval_retrieval_refuses_a_variant_it_could_not_measure_before_any_work(
    stand: Stand, options: list[str], named: str
) -> None:
    result = stand.run("refused", *options)

    assert result.exit_code == 2, result.output
    assert named in COLOUR_CODES.sub("", result.output)
    assert not stand.runs.exists()
    assert stand.built == []
    assert stand.embedder.queries == []


def test_a_variant_is_refused_when_the_settings_choose_a_search_it_does_not_fit(
    stand: Stand, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The settings are read once and then cached, so they are set before a command reads them.
    get_settings.cache_clear()
    monkeypatch.setenv("HANDBOOK_RAG_RERANK", "true")
    spread = ("--spread-per-page", "1", "--spread-within", "5")

    reranked = stand.run("refused", *spread)
    without_reranking = stand.run("spread", *spread, "--no-rerank")

    assert reranked.exit_code == 2, reranked.output
    assert "--no-rerank" in COLOUR_CODES.sub("", reranked.output)
    assert without_reranking.exit_code == 0, without_reranking.output
    assert stand.row("spread")["evidence_ranks"] == [1, 2]

    get_settings.cache_clear()
    monkeypatch.setenv("HANDBOOK_RAG_RETRIEVAL_MODE", "sparse")
    keyword_search = stand.run("refused", "--rrf-k", "10", "--no-rerank")

    assert keyword_search.exit_code == 2, keyword_search.output
    assert "sparse" in COLOUR_CODES.sub("", keyword_search.output)
    assert not (stand.runs / "refused").exists()


def test_no_command_but_eval_retrieval_can_ask_for_a_variant() -> None:
    group = get_command(app)
    assert isinstance(group, TyperGroup)
    commands = group.commands
    options = {
        name: {option for parameter in command.params for option in parameter.opts}
        for name, command in commands.items()
    }

    assert options["eval-retrieval"] >= VARIANT_OPTIONS
    for name in ("search", "ask", "serve", "mcp", "eval-gate", "eval-answers", "eval-adversarial"):
        assert name in options
    for name, of_command in options.items():
        if name != "eval-retrieval":
            assert not VARIANT_OPTIONS & of_command, name
    # Each of the four is off unless it is given.
    defaults = {
        parameter.name: parameter.default
        for parameter in commands["eval-retrieval"].params
        if VARIANT_OPTIONS & set(parameter.opts)
    }
    assert defaults == {
        "fusion_candidates": None,
        "rrf_k": None,
        "spread_per_page": None,
        "spread_within": None,
    }


def test_the_gate_and_a_search_build_the_default_search(stand: Stand, tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"distractor_pages": 0, "seed": 5, "minimums": {"hit@5": 1.0}}))

    gated = runner.invoke(
        app, ["eval-gate", "--dataset", str(stand.dataset), "--config", str(gate)]
    )
    searched = runner.invoke(app, ["search", "weeks of paid parental leave", "--limit", "5"])

    assert gated.exit_code == 0, gated.output
    assert "Gate passed." in gated.output
    assert searched.exit_code == 0, searched.output
    # Neither asks the wiring for the rule, and both get thirty candidates from each
    # search, the constant 60 and no rule.
    assert len(stand.asked_with) == 2
    assert all("spread" not in options for options in stand.asked_with)
    assert stand.built == [{"candidates": 30, "rrf_k": 60, "spread": None}] * 2
    assert searched.output.count("people/leave") == 5
