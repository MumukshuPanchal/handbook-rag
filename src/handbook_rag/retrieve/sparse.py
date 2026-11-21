"""Sparse retrieval: search by keyword with BM25."""

from handbook_rag.index.sparse import SparseEncoder
from handbook_rag.index.store import ScoredChunk, VectorStore


class SparseRetriever:
    """Retrieve the chunks that share the question's words, weighted by BM25."""

    def __init__(self, encoder: SparseEncoder, store: VectorStore) -> None:
        self._encoder = encoder
        self._store = store

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        if not query.strip():
            return []
        vector = self._encoder.encode_query(query)
        if not vector.indices:
            return []  # the question contained only stop words
        return self._store.search_sparse(vector, limit=limit, section=section)
