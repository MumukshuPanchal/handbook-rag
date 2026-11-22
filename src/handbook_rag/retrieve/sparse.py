"""Sparse retrieval: search by keyword with BM25."""

from handbook_rag.index.sparse import SparseEncoder
from handbook_rag.index.store import ScoredChunk, VectorStore
from handbook_rag.tracing import record_query, record_results, stage


class SparseRetriever:
    """Retrieve the chunks that share the question's words, weighted by BM25."""

    def __init__(self, encoder: SparseEncoder, store: VectorStore) -> None:
        self._encoder = encoder
        self._store = store

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        with stage("retrieve.sparse", limit=limit, section=section) as span:
            record_query(span, query)
            results: list[ScoredChunk] = []
            if query.strip():
                vector = self._encoder.encode_query(query)
                # Zero means the question contained only stop words, so nothing can match.
                span.set_attribute("handbook_rag.query.terms", len(vector.indices))
                if vector.indices:
                    results = self._store.search_sparse(vector, limit=limit, section=section)
            record_results(span, results)
            return results
