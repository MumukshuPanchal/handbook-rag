import pytest
from qdrant_client import QdrantClient

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.index.store import QdrantVectorStore
from handbook_rag.retrieve.dense import DenseRetriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from tests.support import FakeEmbedder, FakeSparseEncoder


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

    monkeypatch.setattr(wiring, "make_embedder", embedder)
    monkeypatch.setattr(wiring, "make_sparse_encoder", sparse_encoder)
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
