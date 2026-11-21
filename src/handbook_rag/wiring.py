"""Build the pipeline's parts from the application settings.

Each stage is defined by a small interface. This module is the one place that decides
which implementation stands behind each interface, so swapping one is a change here and
nowhere else.
"""

from typing import Literal

from qdrant_client import QdrantClient

from handbook_rag.config import Settings
from handbook_rag.index.embedder import SentenceTransformerEmbedder
from handbook_rag.index.sparse import Bm25Encoder
from handbook_rag.index.store import QdrantVectorStore, VectorStoreError
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


def make_sparse_encoder(settings: Settings) -> Bm25Encoder:
    return Bm25Encoder(settings.sparse_model)


def make_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.rerank_model)


def make_retriever(
    settings: Settings, mode: RetrievalMode | None = None, rerank: bool | None = None
) -> Retriever:
    """Build the retriever for ``mode`` and ``rerank``, or for the values in the settings.

    Only the models a configuration needs are loaded: keyword search never loads the
    embedding model, and nothing loads the reranker unless reranking is on.
    """
    mode = mode or settings.retrieval_mode
    store = make_store(settings)
    retriever: Retriever
    if mode == "dense":
        retriever = DenseRetriever(make_embedder(settings), store)
    elif mode == "sparse":
        retriever = SparseRetriever(make_sparse_encoder(settings), store)
    else:
        retriever = HybridRetriever(
            DenseRetriever(make_embedder(settings), store),
            SparseRetriever(make_sparse_encoder(settings), store),
            candidates=settings.fusion_candidates,
            rrf_k=settings.rrf_k,
        )
    if settings.rerank if rerank is None else rerank:
        retriever = RerankingRetriever(
            retriever, make_reranker(settings), candidates=settings.rerank_candidates
        )
    return retriever
