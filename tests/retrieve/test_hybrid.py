import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore, ScoredChunk
from handbook_rag.retrieve.dense import DenseRetriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.")
EXPENSES = make_chunk(
    "Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md", section="finance"
)
ERROR_CODE = make_chunk(
    "Error ZX81 means the license key expired.", doc_id="support/errors.md", section="support"
)


@pytest.fixture
def store() -> QdrantVectorStore:
    store = QdrantVectorStore(QdrantClient(":memory:"), "test-collection")
    index_chunks(
        [LEAVE, EXPENSES, ERROR_CODE],
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        store=store,
    )
    return store


def test_sparse_retriever_finds_an_exact_term(store: QdrantVectorStore) -> None:
    retriever = SparseRetriever(FakeSparseEncoder(), store)

    results = retriever.retrieve("what does ZX81 mean", limit=3)

    assert results[0].chunk == ERROR_CODE


def test_sparse_retriever_returns_nothing_for_a_blank_or_stop_word_question(
    store: QdrantVectorStore,
) -> None:
    retriever = SparseRetriever(FakeSparseEncoder(), store)

    assert retriever.retrieve("  ", limit=3) == []
    assert retriever.retrieve("the of a is", limit=3) == []


def test_sparse_retriever_respects_the_section_filter(store: QdrantVectorStore) -> None:
    retriever = SparseRetriever(FakeSparseEncoder(), store)

    results = retriever.retrieve("laptop leave", limit=3, section="finance")

    assert [r.chunk for r in results] == [EXPENSES]


class FixedRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.limits: list[int] = []
        self.sections: list[str | None] = []

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        self.limits.append(limit)
        self.sections.append(section)
        return self._results[:limit]


def test_hybrid_puts_a_chunk_both_searches_agree_on_first() -> None:
    dense = FixedRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.8)])
    sparse = FixedRetriever([ScoredChunk(ERROR_CODE, 9.0), ScoredChunk(EXPENSES, 7.0)])

    results = HybridRetriever(dense, sparse).retrieve("q", limit=3)

    assert results[0].chunk == EXPENSES
    assert {r.chunk.doc_id for r in results} == {
        "people/leave.md",
        "finance/expenses.md",
        "support/errors.md",
    }


def test_hybrid_asks_each_search_for_more_candidates_than_it_returns() -> None:
    dense, sparse = FixedRetriever([]), FixedRetriever([])

    HybridRetriever(dense, sparse, candidates=30).retrieve("q", limit=5, section="finance")

    assert dense.limits == [30]
    assert sparse.limits == [30]
    assert dense.sections == sparse.sections == ["finance"]


def test_hybrid_never_asks_for_fewer_candidates_than_the_limit() -> None:
    dense, sparse = FixedRetriever([]), FixedRetriever([])

    HybridRetriever(dense, sparse, candidates=10).retrieve("q", limit=50)

    assert dense.limits == [50]


def test_hybrid_end_to_end_finds_exact_terms_and_paraphrases(store: QdrantVectorStore) -> None:
    retriever = HybridRetriever(
        DenseRetriever(FakeEmbedder(), store), SparseRetriever(FakeSparseEncoder(), store)
    )

    assert retriever.retrieve("ZX81", limit=1)[0].chunk == ERROR_CODE
    assert retriever.retrieve("sixteen weeks paid leave for parents", limit=1)[0].chunk == LEAVE


def thirty_each() -> tuple[FixedRetriever, FixedRetriever]:
    """Two searches of thirty results that share one chunk, which both of them rank last."""
    shared = ScoredChunk(make_chunk("both"), 0.1)
    dense = [ScoredChunk(make_chunk(f"dense {rank}"), 1.0 - 0.01 * rank) for rank in range(1, 30)]
    sparse = [ScoredChunk(make_chunk(f"sparse {rank}"), 9.0 - 0.1 * rank) for rank in range(1, 30)]
    return FixedRetriever([*dense, shared]), FixedRetriever([*sparse, shared])


def test_with_the_default_constant_a_chunk_of_both_lists_comes_before_any_chunk_of_one() -> None:
    dense, sparse = thirty_each()

    as_set = HybridRetriever(dense, sparse).retrieve("q", limit=3)
    with_a_low_constant = HybridRetriever(dense, sparse, rrf_k=5).retrieve("q", limit=10)

    # Thirtieth in both lists is 2/90, and first in one list is 1/61.
    assert as_set[0].chunk.text == "both"
    assert as_set[0].score == pytest.approx(2 / 90)
    assert as_set[1].score == pytest.approx(1 / 61)
    # With the constant 5 the first results of each search come through: 1/6 against 2/35.
    assert {found.chunk.text for found in with_a_low_constant[:2]} == {"dense 1", "sparse 1"}
    assert with_a_low_constant[0].score == pytest.approx(1 / 6)
    assert "both" not in [found.chunk.text for found in with_a_low_constant]


def test_a_chunk_beyond_the_candidates_of_both_searches_is_not_returned() -> None:
    dense, sparse = thirty_each()

    with_ten = HybridRetriever(dense, sparse, candidates=10).retrieve("q", limit=10)
    with_thirty = HybridRetriever(dense, sparse, candidates=30).retrieve("q", limit=10)

    assert dense.limits == [10, 30]
    assert "both" not in [found.chunk.text for found in with_ten]
    assert with_thirty[0].chunk.text == "both"
