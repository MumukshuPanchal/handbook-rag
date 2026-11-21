"""Turn text into sparse keyword vectors with BM25.

A dense embedding captures meaning but can miss an exact term: a product name, an
acronym, an error code. A sparse vector has one position per word in the vocabulary and
is non-zero only for the words a text contains, so matching it is keyword search.

BM25 is the standard way to weight those words. A word counts for more when it is rare
across all documents (inverse document frequency), counts for less each additional time
it appears in one document (saturation), and is adjusted for document length so long
texts do not win simply by containing more words. Here the per-document part is computed
when a chunk is encoded, and the vector store applies the rarity weighting at query time.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Word ids and their weights. Every other position is zero."""

    indices: list[int]
    values: list[float]


class SparseEncoder(Protocol):
    """Anything that can turn chunks and questions into comparable sparse vectors."""

    @property
    def name(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


class Bm25Encoder:
    """BM25 weights from fastembed: lower-casing, stop-word removal and English stemming."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        # Imported here so commands that never search by keyword do not load it.
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(vector.indices.tolist(), vector.values.tolist())
            for vector in self._model.embed(list(texts))
        ]

    def encode_query(self, text: str) -> SparseVector:
        vector = next(iter(self._model.query_embed(text)))
        return SparseVector(vector.indices.tolist(), vector.values.tolist())
