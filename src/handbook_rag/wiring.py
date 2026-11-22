"""Build the pipeline's parts from the application settings.

Each stage is defined by a small interface. This module is the one place that decides
which implementation stands behind each interface, so swapping one is a change here and
nowhere else.
"""

from typing import Literal

from qdrant_client import QdrantClient

from handbook_rag.config import Settings
from handbook_rag.index.embedder import Embedder, SentenceTransformerEmbedder
from handbook_rag.index.sparse import Bm25Encoder, SparseEncoder
from handbook_rag.index.store import QdrantVectorStore, VectorStore, VectorStoreError
from handbook_rag.retrieve.dense import DenseRetriever, Retriever
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.rerank import CrossEncoderReranker, RerankingRetriever
from handbook_rag.retrieve.sparse import SparseRetriever

RetrievalMode = Literal["dense", "sparse", "hybrid"]


def make_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        batch_size=settings.embedding_batch_size,
    )


def make_store(settings: Settings) -> QdrantVectorStore:
    """Connect to Qdrant, failing with a clear message if it is not running."""
    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    try:
        client.get_collections()
    except Exception as error:
        raise VectorStoreError(
            f"Cannot reach Qdrant at {settings.qdrant_url}. Start it with `make qdrant`."
        ) from error
    return QdrantVectorStore(client, settings.collection_name)


def make_memory_store(collection: str) -> QdrantVectorStore:
    """A vector store held in this process's memory, for a small index that is thrown away.

    It needs no server, which is what lets the retrieval gate run in CI.
    """
    return QdrantVectorStore(QdrantClient(":memory:"), collection)


def make_sparse_encoder(settings: Settings) -> Bm25Encoder:
    return Bm25Encoder(settings.sparse_model)


def make_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.rerank_model)


def describe_retrieval(
    settings: Settings, mode: RetrievalMode | None = None, rerank: bool | None = None
) -> dict[str, object]:
    """The settings that decide what ``make_retriever`` returns, for recording with a run.

    Only what the configuration uses is listed: keyword search has no embedding model,
    and fusion settings mean nothing outside hybrid search.
    """
    mode = mode or settings.retrieval_mode
    reranked = settings.rerank if rerank is None else rerank
    described: dict[str, object] = {
        "handbook_commit": settings.handbook_commit,
        "chunking": settings.chunking.version,
        "collection": settings.collection_name,
        "retrieval_mode": mode,
    }
    if mode in ("dense", "hybrid"):
        described["embedding_model"] = settings.embedding_model
    if mode in ("sparse", "hybrid"):
        described["sparse_model"] = settings.sparse_model
    if mode == "hybrid":
        described["fusion_candidates"] = settings.fusion_candidates
        described["rrf_k"] = settings.rrf_k
    described["rerank"] = reranked
    if reranked:
        described["rerank_model"] = settings.rerank_model
        described["rerank_candidates"] = settings.rerank_candidates
    return described


def make_retriever(
    settings: Settings,
    mode: RetrievalMode | None = None,
    rerank: bool | None = None,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    sparse_encoder: SparseEncoder | None = None,
) -> Retriever:
    """Build the retriever for ``mode`` and ``rerank``, or for the values in the settings.

    Only the models a configuration needs are loaded: keyword search never loads the
    embedding model, and nothing loads the reranker unless reranking is on.

    A caller that has just built an index passes the ``store``, ``embedder`` and
    ``sparse_encoder`` it used, so the retriever searches that index and no model is
    loaded a second time. Whatever is not passed is built from the settings.
    """
    mode = mode or settings.retrieval_mode
    if store is None:
        store = make_store(settings)
    retriever: Retriever
    if mode == "dense":
        retriever = DenseRetriever(embedder or make_embedder(settings), store)
    elif mode == "sparse":
        retriever = SparseRetriever(sparse_encoder or make_sparse_encoder(settings), store)
    else:
        retriever = HybridRetriever(
            DenseRetriever(embedder or make_embedder(settings), store),
            SparseRetriever(sparse_encoder or make_sparse_encoder(settings), store),
            candidates=settings.fusion_candidates,
            rrf_k=settings.rrf_k,
        )
    if settings.rerank if rerank is None else rerank:
        retriever = RerankingRetriever(
            retriever, make_reranker(settings), candidates=settings.rerank_candidates
        )
    return retriever
