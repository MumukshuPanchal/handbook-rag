"""Hybrid retrieval: search by meaning and by keyword, then merge the two rankings."""

from handbook_rag.index.store import ScoredChunk
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.fusion import reciprocal_rank_fusion
from handbook_rag.tracing import record_query, record_results, stage


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
        with stage("retrieve.hybrid", limit=limit, section=section, candidates=depth) as span:
            record_query(span, query)
            dense = self._dense.retrieve(query, limit=depth, section=section)
            sparse = self._sparse.retrieve(query, limit=depth, section=section)
            with stage("fuse", rrf_k=self._rrf_k) as fusing:
                fused = reciprocal_rank_fusion([dense, sparse], limit=limit, k=self._rrf_k)
                if fusing.is_recording():
                    # How much the two searches agree: chunks that both of them returned.
                    shared = {r.chunk.chunk_id for r in dense} & {r.chunk.chunk_id for r in sparse}
                    fusing.set_attribute("handbook_rag.fuse.from_dense", len(dense))
                    fusing.set_attribute("handbook_rag.fuse.from_sparse", len(sparse))
                    fusing.set_attribute("handbook_rag.fuse.in_both", len(shared))
            record_results(span, fused)
            return fused
