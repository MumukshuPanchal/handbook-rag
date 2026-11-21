"""Dense retrieval: search by meaning.

The question is embedded with the same model as the chunks, and the chunks whose vectors
are closest to it are returned. This finds passages that use different words from the
question, which keyword search cannot do.
"""

from typing import Protocol

from handbook_rag.index.embedder import Embedder
from handbook_rag.index.store import ScoredChunk, VectorStore


class Retriever(Protocol):
    """Anything that can find chunks relevant to a question."""

    def retrieve(
        self, query: str, *, limit: int, section: str | None = None
    ) -> list[ScoredChunk]: ...


class DenseRetriever:
    """Retrieve by embedding similarity."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        if not query.strip():
            return []
        vector = self._embedder.embed_query(query)
        return self._store.search(vector, limit=limit, section=section)
