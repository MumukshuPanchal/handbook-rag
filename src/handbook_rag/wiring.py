"""Build the pipeline's parts from the application settings.

Each stage is defined by a small interface. This module is the one place that decides
which implementation stands behind each interface, so swapping one is a change here and
nowhere else.
"""

from qdrant_client import QdrantClient

from handbook_rag.config import Settings
from handbook_rag.index.embedder import SentenceTransformerEmbedder
from handbook_rag.index.store import QdrantVectorStore, VectorStoreError


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
