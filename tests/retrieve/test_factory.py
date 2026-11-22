import pytest
from qdrant_client import QdrantClient

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import QdrantVectorStore, VectorStoreError
from handbook_rag.retrieve.dense import DenseRetriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.rerank import RerankingRetriever
from handbook_rag.retrieve.sparse import SparseRetriever
from tests.support import FakeEmbedder, FakeReranker, FakeSparseEncoder, make_chunk


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
    assert isinstance(wiring.make_retriever(Settings()), HybridRetriever)


def test_reranking_wraps_the_chosen_retriever_only_when_asked(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(), "dense", rerank=True), RerankingRetriever)
    assert "reranker" in loaded

    loaded.clear()
    assert isinstance(wiring.make_retriever(Settings(), "dense", rerank=False), DenseRetriever)
    assert "reranker" not in loaded


def test_reranking_defaults_to_the_setting(loaded: list[str]) -> None:
    assert isinstance(wiring.make_retriever(Settings(rerank=True)), RerankingRetriever)
    assert isinstance(wiring.make_retriever(Settings(rerank=True), rerank=False), HybridRetriever)


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


def test_a_retriever_built_from_existing_parts_searches_that_index_and_loads_nothing(
    loaded: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_server(_settings: Settings) -> QdrantVectorStore:
        raise VectorStoreError("no server should be contacted")

    monkeypatch.setattr(wiring, "make_store", no_server)
    chunk = make_chunk("New parents get sixteen weeks of paid leave.")
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("scratch")
    index_chunks([chunk], embedder=embedder, sparse_encoder=sparse_encoder, store=store)

    retriever = wiring.make_retriever(
        Settings(), "hybrid", store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )

    assert isinstance(retriever, HybridRetriever)
    assert loaded == []
    (found,) = retriever.retrieve("how many weeks of paid leave", limit=5)
    assert found.chunk.chunk_id == chunk.chunk_id


def test_parts_that_are_not_passed_are_built_from_the_settings(loaded: list[str]) -> None:
    store = wiring.make_memory_store("scratch")

    assert isinstance(wiring.make_retriever(Settings(), "dense", store=store), DenseRetriever)
    assert loaded == ["embedder"]


def test_two_memory_stores_do_not_share_their_contents() -> None:
    chunk = make_chunk("New parents get sixteen weeks of paid leave.")
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    first, second = wiring.make_memory_store("scratch"), wiring.make_memory_store("scratch")

    index_chunks([chunk], embedder=embedder, sparse_encoder=sparse_encoder, store=first)
    second.ensure_collection(embedder.dimension)

    assert first.existing_ids([chunk.chunk_id]) == {chunk.chunk_id}
    assert second.existing_ids([chunk.chunk_id]) == set()


def test_a_service_can_have_its_store_without_asking_whether_it_is_there() -> None:
    nothing_listening = Settings(qdrant_url="http://127.0.0.1:1")

    with pytest.raises(VectorStoreError, match="Cannot reach Qdrant"):
        wiring.make_store(nothing_listening)

    store = wiring.make_store(nothing_listening, check=False, timeout=2)
    assert store.collection == nothing_listening.collection_name


def test_the_embedding_model_runs_on_the_device_in_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, object]] = []

    def embedder(model: str, **options: object) -> FakeEmbedder:
        built.append(options)
        return FakeEmbedder()

    monkeypatch.setattr(wiring, "SentenceTransformerEmbedder", embedder)

    wiring.make_embedder(Settings())
    wiring.make_embedder(Settings(embedding_device="cpu"))

    assert [options["device"] for options in built] == [None, "cpu"]
