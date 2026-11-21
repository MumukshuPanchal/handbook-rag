"""Store chunk vectors and search them.

The vector store keeps, for every chunk, its vector and its metadata (the *payload*).
Searching means finding the stored vectors closest to a question's vector.
"""

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from handbook_rag.index.sparse import SparseVector
from handbook_rag.ingest.models import Chunk

_DENSE = "dense"
_SPARSE = "sparse"
_BATCH = 1000


class VectorStoreError(RuntimeError):
    """The vector store cannot be used as configured."""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A search result: a chunk and how similar it is to the question (higher is closer)."""

    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """What the rest of the pipeline needs from a vector store."""

    def ensure_collection(self, dimension: int) -> None: ...

    def existing_ids(self, chunk_ids: Sequence[str]) -> set[str]: ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> None: ...

    def delete_except(self, keep: set[str]) -> int: ...

    def count(self) -> int: ...

    def search(
        self, vector: Sequence[float], *, limit: int, section: str | None = None
    ) -> list[ScoredChunk]: ...

    def search_sparse(
        self, vector: SparseVector, *, limit: int, section: str | None = None
    ) -> list[ScoredChunk]: ...


class QdrantVectorStore:
    """One Qdrant collection holding the chunks of one corpus configuration."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it is missing, or check that it fits the model."""
        if self._client.collection_exists(self._collection):
            vectors = self._client.get_collection(self._collection).config.params.vectors
            existing = vectors[_DENSE].size if isinstance(vectors, dict) else None
            if existing != dimension:
                raise VectorStoreError(
                    f"Collection {self._collection} stores vectors of size {existing}, but "
                    f"the embedding model produces size {dimension}."
                )
            return
        self._client.create_collection(
            self._collection,
            vectors_config={
                _DENSE: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
            },
            # The IDF modifier makes the store weight each word by how rare it is across
            # the collection. That is the half of BM25 that depends on the whole corpus.
            # Vector names are fixed when a collection is created and cannot be added later.
            sparse_vectors_config={
                _SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        # Index the fields we filter on, so a filtered search does not scan every chunk.
        for field in ("section", "doc_id"):
            self._client.create_payload_index(
                self._collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def existing_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Which of these chunk ids are already stored."""
        found: set[str] = set()
        for batch in _batches(list(chunk_ids), _BATCH):
            records = self._client.retrieve(
                self._collection, ids=batch, with_payload=False, with_vectors=False
            )
            found.update(str(record.id) for record in records)
        return found

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector],
    ) -> None:
        """Insert chunks, replacing any stored chunk that has the same id."""
        if not len(chunks) == len(vectors) == len(sparse_vectors):
            raise ValueError("Each chunk needs exactly one dense and one sparse vector.")
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    _DENSE: list(vector),
                    _SPARSE: models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=_to_payload(chunk),
            )
            for chunk, vector, sparse in zip(chunks, vectors, sparse_vectors, strict=True)
        ]
        for batch in _batches(points, 256):
            self._client.upsert(self._collection, points=batch, wait=True)

    def delete_except(self, keep: set[str]) -> int:
        """Remove every stored chunk whose id is not in ``keep``. Returns how many."""
        stale: list[str] = []
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                limit=_BATCH,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale.extend(str(r.id) for r in records if str(r.id) not in keep)
            if offset is None:
                break
        for batch in _batches(stale, _BATCH):
            self._client.delete(
                self._collection, points_selector=models.PointIdsList(points=list(batch)), wait=True
            )
        return len(stale)

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def search(
        self, vector: Sequence[float], *, limit: int, section: str | None = None
    ) -> list[ScoredChunk]:
        """Return the ``limit`` chunks closest in meaning to ``vector``, best first."""
        return self._query(list(vector), _DENSE, limit, section)

    def search_sparse(
        self, vector: SparseVector, *, limit: int, section: str | None = None
    ) -> list[ScoredChunk]:
        """Return the ``limit`` chunks that best match the question's words, best first."""
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return self._query(query, _SPARSE, limit, section)

    def _query(
        self, query: list[float] | models.SparseVector, using: str, limit: int, section: str | None
    ) -> list[ScoredChunk]:
        if not self._client.collection_exists(self._collection):
            raise VectorStoreError(
                f"Collection {self._collection} does not exist yet. "
                "Run `handbook-rag index` to build it."
            )
        query_filter = None
        if section is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="section", match=models.MatchValue(value=section))]
            )
        response = self._client.query_points(
            self._collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=_from_payload(str(point.id), point.payload or {}), score=point.score)
            for point in response.points
        ]


def _to_payload(chunk: Chunk) -> dict[str, Any]:
    payload = asdict(chunk)
    del payload["chunk_id"]  # stored as the point id
    payload["heading_path"] = list(chunk.heading_path)
    return payload


def _from_payload(chunk_id: str, payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=payload["doc_id"],
        title=payload["title"],
        url=payload["url"],
        section=payload["section"],
        source_commit=payload["source_commit"],
        heading_path=tuple(payload["heading_path"]),
        start_char=payload["start_char"],
        end_char=payload["end_char"],
        text=payload["text"],
        token_count=payload["token_count"],
    )


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
