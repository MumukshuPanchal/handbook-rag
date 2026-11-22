"""Reranking: re-score retrieved candidates by reading the question and each chunk together.

An embedding model reads the question and a chunk separately and compares two vectors.
That is fast enough to search tens of thousands of chunks, but each text is summarised
without knowing what the other says. A *cross-encoder* reads the question and one chunk
as a single input, so it can weigh every word of one against the other. It is more
accurate and far too slow to run over a whole corpus, which is why it is applied only to
the few dozen candidates that the first search returns.
"""

from collections.abc import Sequence
from typing import Protocol

from handbook_rag.index.store import ScoredChunk
from handbook_rag.local_models import load_local_first, quiet_model_libraries
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import record_query, record_results, stage


class Reranker(Protocol):
    """Anything that can score how well each text answers a question."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """A cross-encoder model run locally with sentence-transformers."""

    def __init__(self, model_name: str, *, batch_size: int = 16, device: str | None = None) -> None:
        # Imported here because loading PyTorch takes seconds.
        quiet_model_libraries()
        from sentence_transformers import CrossEncoder

        self._model = load_local_first(CrossEncoder, model_name, device=device)
        self._name = model_name
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class RerankingRetriever:
    """Wrap a retriever: fetch more candidates than needed, rerank them, keep the best."""

    def __init__(self, base: Retriever, reranker: Reranker, *, candidates: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        with stage("retrieve.reranked", limit=limit, section=section) as span:
            record_query(span, query)
            found = self._base.retrieve(query, limit=max(limit, self._candidates), section=section)
            results: list[ScoredChunk] = []
            if found:
                with stage("rerank", model=self._reranker.name, candidates=len(found)):
                    scores = self._reranker.score(
                        query, [result.chunk.text_for_embedding for result in found]
                    )
                # sorted() is stable, so candidates with equal scores keep their first-stage order.
                reranked = sorted(zip(found, scores, strict=True), key=lambda pair: -pair[1])
                results = [ScoredChunk(result.chunk, score) for result, score in reranked[:limit]]
            record_results(span, results)
            return results
