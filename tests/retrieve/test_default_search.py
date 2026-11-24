"""The default search returns what it returned before the search variants were added.

Issue 78 added variants of hybrid search to measure on questions that need two passages:
a rule that spreads the merged list over pages, another number of candidates, another
fusion constant. They are options, and each is off unless a run asks for it. These tests
hold the default to that.

The rankings below were written down from the code as it was before the options existed
(commit 20d237f, "feat: the two-passage questions read a second time, and measured on the
33 confirmed"), on a small stand-in index. This file uses nothing the options added, so
it runs against that commit as it is, and gives the same result there.

The stand-in models are the ones of ``tests/support.py``. The vector store sorts by
score, and how it orders two equal scores is its own business, so every score a pinned
result rests on is checked to stand apart from the others.
"""

import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore, ScoredChunk
from handbook_rag.retrieve.dense import DenseRetriever, Retriever
from handbook_rag.retrieve.fusion import reciprocal_rank_fusion
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

ROOT = Path(__file__).parents[2]
PAGES = {
    "people/leave.md": [
        "New parents get sixteen weeks of paid parental leave.",
        "Paid parental leave weeks can be split in two blocks.",
        "Parental leave weeks need notice in writing.",
        "Sick days are recorded by the manager.",
    ],
    "people/germany.md": [
        "Germany: parental leave follows the statutory allowance.",
        "Germany has thirty vacation days.",
    ],
    "finance/expenses.md": [
        "New laptops for parents are paid by the company.",
        "Travel expenses need a receipt from the hotel.",
    ],
    "support/errors.md": [
        "Error ZX81 means the license key expired.",
        "Restart the runner after an error.",
        "A license key is renewed every year by the owner of the account.",
    ],
    "security/tiers.md": [
        "The runner is a tier two system.",
        "Tier one systems hold customer data.",
    ],
}
# For each question, the first eight results of the default search: the chunk's page, its
# text and its merged score. Copied from the output of commit 20d237f.
BEFORE = {
    "paid parental leave weeks for new parents in germany": [
        ("people/leave.md", "New parents get sixteen weeks of paid parental leave.", 0.032522),
        ("people/leave.md", "Paid parental leave weeks can be split in two blocks.", 0.032266),
        ("finance/expenses.md", "New laptops for parents are paid by the company.", 0.032002),
        ("people/leave.md", "Parental leave weeks need notice in writing.", 0.03125),
        ("people/germany.md", "Germany: parental leave follows the statutory allowance.", 0.030769),
        ("people/germany.md", "Germany has thirty vacation days.", 0.030303),
        ("people/leave.md", "Sick days are recorded by the manager.", 0.029631),
        ("finance/expenses.md", "Travel expenses need a receipt from the hotel.", 0.014925),
    ],
    "error zx81 license key expired restart runner": [
        ("support/errors.md", "Error ZX81 means the license key expired.", 0.032787),
        ("support/errors.md", "Restart the runner after an error.", 0.032258),
        (
            "support/errors.md",
            "A license key is renewed every year by the owner of the account.",
            0.031746,
        ),
        ("security/tiers.md", "The runner is a tier two system.", 0.03101),
        ("security/tiers.md", "Tier one systems hold customer data.", 0.015625),
        ("finance/expenses.md", "Travel expenses need a receipt from the hotel.", 0.015152),
        ("people/leave.md", "New parents get sixteen weeks of paid parental leave.", 0.014925),
        ("people/leave.md", "Paid parental leave weeks can be split in two blocks.", 0.014706),
    ],
    "which tier one systems hold customer data and is the runner one of them": [
        ("security/tiers.md", "Tier one systems hold customer data.", 0.032787),
        ("security/tiers.md", "The runner is a tier two system.", 0.032002),
        ("support/errors.md", "Restart the runner after an error.", 0.032002),
        (
            "support/errors.md",
            "A license key is renewed every year by the owner of the account.",
            0.015625,
        ),
        ("support/errors.md", "Error ZX81 means the license key expired.", 0.015385),
        ("people/leave.md", "Sick days are recorded by the manager.", 0.015152),
        ("people/germany.md", "Germany: parental leave follows the statutory allowance.", 0.014925),
        ("finance/expenses.md", "New laptops for parents are paid by the company.", 0.014706),
    ],
}
PINNED = 8
# Two scores closer than this could change places on another machine.
APART = 0.001

Parts = tuple[QdrantVectorStore, FakeEmbedder, FakeSparseEncoder]


def stand_in_index() -> Parts:
    """The stand-in index: thirteen chunks of five pages, and the models that built it."""
    chunks = [
        make_chunk(text, doc_id=doc_id, section=doc_id.split("/")[0], title=doc_id)
        for doc_id, texts in PAGES.items()
        for text in texts
    ]
    store = wiring.make_memory_store("default-search")
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    index_chunks(chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    return store, embedder, sparse_encoder


@pytest.fixture
def parts() -> Parts:
    return stand_in_index()


def default_search(parts: Parts) -> Retriever:
    """The retriever the settings choose when nothing is set, as every command builds it."""
    store, embedder, sparse_encoder = parts
    return wiring.make_retriever(
        Settings(), store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )


def shown(results: list[ScoredChunk]) -> list[tuple[str, str, float]]:
    return [(found.chunk.doc_id, found.chunk.text, round(found.score, 6)) for found in results]


def test_the_default_search_returns_what_it_returned_before_the_variants_existed(
    parts: Parts,
) -> None:
    retriever = default_search(parts)

    assert isinstance(retriever, HybridRetriever)
    for question, before in BEFORE.items():
        assert shown(retriever.retrieve(question, limit=PINNED)) == before, question
        # Asked for five, as an answer asks, it returns the first five of those.
        assert shown(retriever.retrieve(question, limit=5)) == before[:5], question


def test_no_pinned_result_rests_on_two_scores_that_could_change_places(parts: Parts) -> None:
    store, embedder, sparse_encoder = parts
    searches: dict[str, Retriever] = {
        "dense": DenseRetriever(embedder, store),
        "sparse": SparseRetriever(sparse_encoder, store),
    }

    for question, before in BEFORE.items():
        found = {name: search.retrieve(question, limit=30) for name, search in searches.items()}
        for name, results in found.items():
            scores = sorted(result.score for result in results if result.score > 0)
            gaps = [high - low for low, high in itertools.pairwise(scores)]
            assert min(gaps) > APART, (question, name)
        # A chunk that shares no word with the question scores zero with the stand-in
        # embedding model, and those chunks are equal among themselves. They come last in
        # its list, keyword search does not return them, and none of them is pinned.
        above_zero = {result.chunk.text for result in found["dense"] if result.score > 0}
        assert {text for _, text, _ in before} <= above_zero, question
        assert {result.chunk.text for result in found["sparse"]} <= above_zero, question


def test_the_default_search_is_the_two_lists_merged_and_nothing_after_that(parts: Parts) -> None:
    store, embedder, sparse_encoder = parts
    dense, sparse = DenseRetriever(embedder, store), SparseRetriever(sparse_encoder, store)
    retriever = default_search(parts)

    for question in BEFORE:
        # Thirty candidates from each search and the constant 60, as the settings have them.
        merged = reciprocal_rank_fusion(
            [dense.retrieve(question, limit=30), sparse.retrieve(question, limit=30)],
            limit=PINNED,
            k=60,
        )
        assert retriever.retrieve(question, limit=PINNED) == merged, question


def test_a_rule_that_spreads_results_over_pages_would_change_every_pinned_ranking() -> None:
    # What makes the first test a test of "the rule is off": in each ranking three of the
    # first five results are chunks of one page, so a limit of one or two for each page
    # would move a chunk.
    for question, before in BEFORE.items():
        pages = Counter(doc_id for doc_id, _, _ in before[:5])
        assert max(pages.values()) == 3, question


def test_the_settings_of_the_default_search_are_those_the_committed_runs_recorded() -> None:
    settings = Settings()
    described = wiring.describe_retrieval(settings)

    assert (settings.retrieval_mode, settings.rerank) == ("hybrid", False)
    assert (settings.fusion_candidates, settings.rrf_k) == (30, 60)
    # The golden set's hybrid run and the two-passage run of the default search say what
    # they were made with. A run of the default search made today says the same, key for
    # key and in the same order.
    for run in ("retrieval/v2", "two-passage/hybrid"):
        summary = json.loads((ROOT / "evals" / "runs" / run / "summary.json").read_text("utf-8"))
        assert list(summary["config"].items()) == list(described.items()), run
