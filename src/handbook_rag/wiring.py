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


def make_retriever(settings: Settings, mode: RetrievalMode | None = None) -> Retriever:
    """Build the retriever for ``mode``, or for the mode in the settings.

    Only the models a mode needs are loaded: keyword search never loads the embedding
    model, and search by meaning never loads the keyword encoder.
    """
    mode = mode or settings.retrieval_mode
    store = make_store(settings)
    if mode == "dense":
        return DenseRetriever(make_embedder(settings), store)
    if mode == "sparse":
        return SparseRetriever(make_sparse_encoder(settings), store)
    return HybridRetriever(
        DenseRetriever(make_embedder(settings), store),
        SparseRetriever(make_sparse_encoder(settings), store),
        candidates=settings.fusion_candidates,
        rrf_k=settings.rrf_k,
    )
