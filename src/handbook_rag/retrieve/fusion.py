"""Combine several ranked lists into one."""

from collections import defaultdict
from collections.abc import Sequence

from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], *, limit: int, k: int = 60
) -> list[ScoredChunk]:
    """Merge ranked lists using only each chunk's position in every list.

    Dense and keyword search score on different scales: one is a cosine similarity
    between 0 and 1, the other an unbounded BM25 score. Adding them would let one drown
    out the other. Reciprocal rank fusion ignores the scores and gives each chunk
    ``1 / (k + rank)`` for every list it appears in, then sums. A chunk ranked well by
    both lists beats one ranked first by only one.

    ``k`` damps the advantage of the very top ranks. 60 is the value from the original
    paper and the usual default.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk.chunk_id] += 1.0 / (k + rank)
            chunks.setdefault(result.chunk.chunk_id, result.chunk)
    # Ties are broken by id so the order never depends on dictionary internals.
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [ScoredChunk(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered[:limit]]
