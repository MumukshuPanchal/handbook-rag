import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore
from handbook_rag.retrieve.dense import DenseRetriever
from tests.support import FakeEmbedder, make_chunk

LEAVE = make_chunk("Parental leave is sixteen weeks of paid leave.")
EXPENSES = make_chunk(
    "Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md", section="finance"
)
INCIDENTS = make_chunk(
    "Report a security incident in the incident channel.",
    doc_id="security/incidents.md",
    section="security",
)


@pytest.fixture
def retriever() -> DenseRetriever:
    embedder = FakeEmbedder()
    store = QdrantVectorStore(QdrantClient(":memory:"), "test-collection")
    index_chunks([LEAVE, EXPENSES, INCIDENTS], embedder=embedder, store=store)
    return DenseRetriever(embedder, store)


def test_most_relevant_chunk_comes_first(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("how many weeks of parental leave", limit=3)

    assert results[0].chunk == LEAVE
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_limit_caps_the_number_of_results(retriever: DenseRetriever) -> None:
    assert len(retriever.retrieve("laptop expense", limit=1)) == 1


def test_section_filter_is_passed_through(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("parental leave", limit=3, section="security")

    assert [r.chunk for r in results] == [INCIDENTS]


def test_blank_question_returns_nothing(retriever: DenseRetriever) -> None:
    assert retriever.retrieve("   ", limit=3) == []
