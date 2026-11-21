import pytest
from qdrant_client import QdrantClient

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.index.store import QdrantVectorStore
from handbook_rag.retrieve.dense import DenseRetriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.rerank import RerankingRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from tests.support import FakeEmbedder, FakeReranker, FakeSparseEncoder


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the real models and store with stand-ins, recording which were built."""
    built: list[str] = []
    client = QdrantClient(":memory:")

    def embedder(_settings: Settings) -> FakeEmbedder:
        built.append("embedder")
        return FakeEmbedder()

    def sparse_encoder(_settings: Settings) -> FakeSparseEncoder:
        built.append("sparse")
        return FakeSparseEncoder()

    def reranker(_settings: Settings) -> FakeReranker:
        built.append("reranker")
        return FakeReranker()

    monkeypatch.setattr(wiring, "make_embedder", embedder)
    monkeypatch.setattr(wiring, "make_sparse_encoder", sparse_encoder)
    monkeypatch.setattr(wiring, "make_reranker", reranker)
    monkeypatch.setattr(
        wiring, "make_store", lambda settings: QdrantVectorStore(client, settings.collection_name)
    )
    return built


def test_dense_mode_loads_only_the_embedding_model(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(), "dense"), DenseRetriever)
    assert loaded == ["embedder"]


def test_sparse_mode_loads_only_the_keyword_encoder(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(), "sparse"), SparseRetriever)
    assert loaded == ["sparse"]


def test_hybrid_mode_loads_both(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(), "hybrid"), HybridRetriever)
    assert sorted(loaded) == ["embedder", "sparse"]


def test_mode_defaults_to_the_setting(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(retrieval_mode="sparse")), SparseRetriever)
    assert isinstance(wiring.make_retriever(Settings()), DenseRetriever)


def test_reranking_wraps_the_chosen_retriever_only_when_asked(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(), "dense", rerank=True), RerankingRetriever)
    assert "reranker" in loaded

    loaded.clear()
    assert isinstance(wiring.make_retriever(Settings(), "dense", rerank=False), DenseRetriever)
    assert "reranker" not in loaded


def test_reranking_defaults_to_the_setting(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(rerank=True)), RerankingRetriever)
    assert isinstance(wiring.make_retriever(Settings(rerank=True), rerank=False), DenseRetriever)


def test_a_run_description_lists_only_what_the_configuration_uses() -> None:
    settings = Settings()

    dense = wiring.describe_retrieval(settings, "dense", False)
    sparse = wiring.describe_retrieval(settings, "sparse", False)
    hybrid = wiring.describe_retrieval(settings, "hybrid", True)

    assert dense["retrieval_mode"] == "dense"
    assert dense["embedding_model"] == settings.embedding_model
    assert dense["rerank"] is False
    assert "sparse_model" not in dense
    assert "rrf_k" not in dense
    assert "rerank_model" not in dense
    assert "embedding_model" not in sparse
    assert sparse["sparse_model"] == settings.sparse_model
    assert hybrid["rrf_k"] == settings.rrf_k
    assert hybrid["rerank_model"] == settings.rerank_model
    assert hybrid["handbook_commit"] == settings.handbook_commit
    assert hybrid["collection"] == settings.collection_name


def test_a_run_description_defaults_to_the_settings() -> None:
    described = wiring.describe_retrieval(Settings(retrieval_mode="sparse", rerank=True))

    assert described["retrieval_mode"] == "sparse"
    assert described["rerank"] is True
