"""Hybrid retrieval: search by meaning and by keyword, then merge the two rankings."""

from handbook_rag.index.store import ScoredChunk
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Run two retrievers and fuse their results with reciprocal rank fusion.

    Each retriever is asked for more candidates than the caller wants, because a chunk
    that is, say, eighth in both lists can deserve a place in the final top five.
    """

    def __init__(
        self, dense: Retriever, sparse: Retriever, *, candidates: int = 30, rrf_k: int = 60
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._candidates = candidates
        self._rrf_k = rrf_k

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        depth = max(limit, self._candidates)
        rankings = [
            self._dense.retrieve(query, limit=depth, section=section),
            self._sparse.retrieve(query, limit=depth, section=section),
        ]
        return reciprocal_rank_fusion(rankings, limit=limit, k=self._rrf_k)
