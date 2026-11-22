"""Dense retrieval: search by meaning.

The question is embedded with the same model as the chunks, and the chunks whose vectors
are closest to it are returned. This finds passages that use different words from the
question, which keyword search cannot do.
"""

from typing import Protocol

from handbook_rag.index.embedder import Embedder
from handbook_rag.index.store import ScoredChunk, VectorStore
from handbook_rag.tracing import record_query, record_results, stage


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
        with stage("retrieve.dense", limit=limit, section=section) as span:
            record_query(span, query)
            results: list[ScoredChunk] = []
            if query.strip():
                with stage("embed.query", model=self._embedder.name):
                    vector = self._embedder.embed_query(query)
                results = self._store.search(vector, limit=limit, section=section)
            record_results(span, results)
            return results
