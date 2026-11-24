"""Store chunk vectors and search them.

The vector store keeps, for every chunk, its vector and its metadata (the *payload*).
Searching means finding the stored vectors closest to a question's vector.
"""

from collections.abc import Collection, Iterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast

from opentelemetry.trace import SpanKind
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from handbook_rag.index.sparse import SparseVector
from handbook_rag.ingest.models import Chunk
from handbook_rag.tracing import stage

_DENSE = "dense"
_SPARSE = "sparse"
_BATCH = 1000


class VectorStoreError(RuntimeError):
    """The vector store is unavailable, or does not hold what was asked for."""


class CollectionMissingError(VectorStoreError):
    """The collection has not been built yet. The store itself is fine."""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A search result: a chunk and how similar it is to the question (higher is closer)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class StoredPoint:
    """One point as the store holds it: its payload and both of its vectors."""

    point_id: str
    collection: str
    payload: dict[str, Any]
    dense: list[float]
    sparse: SparseVector

    @property
    def chunk(self) -> Chunk:
        """The chunk the payload describes, which says what the vectors were made from.

        Read only when asked for: comparing two collections compares payloads of any
        shape, while taking vectors from a point needs a whole chunk.
        """
        return _whole_chunk(self.point_id, self.payload, self.collection)


class ReadableStore(Protocol):
    """Reading a collection and nothing else.

    Code that only has to read a collection, such as the older index that an update
    takes vectors from, is given this, so it cannot write to that collection by mistake.
    """

    @property
    def collection(self) -> str: ...

    def point_ids(self) -> Iterator[str]: ...

    def scroll_chunks(self, pages: Collection[str] | None = None) -> Iterator[Chunk]: ...

    def points(self, ids: Sequence[str]) -> dict[str, StoredPoint]: ...


class VectorStore(Protocol):
    """What the rest of the pipeline needs from a vector store."""

    def ensure_collection(self, dimension: int) -> None: ...

    def stored_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

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

    def with_collection(self, collection: str) -> "QdrantVectorStore":
        """Another collection in the same store, reached through the same connection."""
        return QdrantVectorStore(self._client, collection)

    def exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def collection_names(self) -> list[str]:
        """Every collection in the store, this one or not, in alphabetical order."""
        return sorted(item.name for item in self._client.get_collections().collections)

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

    def stored_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        """The chunks stored under these ids, as their payloads describe them."""
        found: dict[str, Chunk] = {}
        for batch in _batches(list(chunk_ids), _BATCH):
            records = self._client.retrieve(
                self._collection, ids=batch, with_payload=True, with_vectors=False
            )
            for record in records:
                found[str(record.id)] = _whole_chunk(
                    str(record.id), record.payload or {}, self._collection
                )
        return found

    def point_ids(self) -> Iterator[str]:
        """The id of every point in the collection."""
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                limit=_BATCH,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            yield from (str(record.id) for record in records)
            if offset is None:
                break

    def scroll_chunks(self, pages: Collection[str] | None = None) -> Iterator[Chunk]:
        """Every stored chunk, or only those of these pages, without their vectors.

        Vectors are left out because they are most of what a point weighs: reading a
        whole collection's payloads is quick, and its vectors are fetched only for the
        points that are needed, a batch at a time.
        """
        if pages is not None and not pages:
            return
        query_filter = None
        if pages is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=sorted(pages)))]
            )
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                self._collection,
                scroll_filter=query_filter,
                limit=_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                yield _whole_chunk(str(record.id), record.payload or {}, self._collection)
            if offset is None:
                break

    def points(self, ids: Sequence[str]) -> dict[str, StoredPoint]:
        """These points with their payloads and vectors. An id that is not stored is left out."""
        found: dict[str, StoredPoint] = {}
        for batch in _batches(list(ids), _BATCH):
            records = self._client.retrieve(
                self._collection, ids=batch, with_payload=True, with_vectors=True
            )
            for record in records:
                found[str(record.id)] = self._to_point(record)
        return found

    def _to_point(self, record: models.Record) -> StoredPoint:
        vectors = record.vector
        dense = vectors.get(_DENSE) if isinstance(vectors, dict) else None
        if not isinstance(dense, list):
            raise VectorStoreError(
                f"Point {record.id} in {self._collection} has no dense vector named {_DENSE!r}."
            )
        # A text with no word worth keeping has an empty keyword vector, which a store
        # may leave out of its answer altogether.
        sparse = vectors.get(_SPARSE) if isinstance(vectors, dict) else None
        if not isinstance(sparse, models.SparseVector):
            sparse = models.SparseVector(indices=[], values=[])
        return StoredPoint(
            point_id=str(record.id),
            collection=self._collection,
            payload=record.payload or {},
            dense=[float(value) for value in cast(list[float], dense)],
            sparse=SparseVector(list(sparse.indices), list(sparse.values)),
        )

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
        # Every id is read before anything is deleted, so deleting cannot disturb the
        # reading.
        stale = [point_id for point_id in self.point_ids() if point_id not in keep]
        for batch in _batches(stale, _BATCH):
            self._client.delete(
                self._collection, points_selector=models.PointIdsList(points=list(batch)), wait=True
            )
        return len(stale)

    def count(self) -> int:
        try:
            return self._client.count(self._collection, exact=True).count
        except Exception as error:
            explained = self._explain(error)
            if explained is None:
                raise
            raise explained from error

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
        with stage(
            "vector_store.query", kind=SpanKind.CLIENT, vector=using, limit=limit, section=section
        ) as span:
            # The span kind and these names follow the OpenTelemetry conventions for
            # database calls, so a trace backend can recognise this span as one.
            span.set_attribute("db.system.name", "qdrant")
            span.set_attribute("db.collection.name", self._collection)
            span.set_attribute("db.operation.name", "query")
            query_filter = None
            if section is not None:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(key="section", match=models.MatchValue(value=section))
                    ]
                )
            try:
                response = self._client.query_points(
                    self._collection,
                    query=query,
                    using=using,
                    limit=limit,
                    query_filter=query_filter,
                    with_payload=True,
                )
            except Exception as error:
                # What went wrong underneath, in the standard attribute for it. The span
                # itself is marked with the kind of error that leaves this method.
                span.set_attribute("error.type", f"{type(error).__module__}.{type(error).__name__}")
                if isinstance(error, UnexpectedResponse) and error.status_code is not None:
                    span.set_attribute("db.response.status_code", str(error.status_code))
                explained = self._explain(error)
                if explained is None:
                    raise
                raise explained from error
            span.set_attribute("handbook_rag.results.count", len(response.points))
            return [
                ScoredChunk(
                    chunk=_from_payload(str(point.id), point.payload or {}), score=point.score
                )
                for point in response.points
            ]

    def _explain(self, error: Exception) -> VectorStoreError | None:
        """Decide what a failed call to the store means.

        An unavailable store and a wrong request must not look alike: the first is a
        reason to try again later, the second is a bug that trying again will not fix.
        So only what is the store's doing becomes a ``VectorStoreError``. ``None`` means
        the store was fine and the caller should let the original error through.

        Whether the collection exists is asked only here, after a failure, and never
        after a timeout. Asking before every query cost each search a second round trip
        to the store, which was found by reading traces.
        """
        if isinstance(error, ResponseHandlingException):
            # No answer at all: the store is down, unreachable or too slow. It is not
            # asked anything more, which could take as long again.
            cause = type(error.source).__name__
            return VectorStoreError(f"The vector store did not answer ({cause}): {error.source}")
        if isinstance(error, UnexpectedResponse):
            if error.status_code == 404:
                return self._missing()
            if error.status_code is not None and error.status_code >= 500:
                return VectorStoreError(f"The vector store failed with status {error.status_code}.")
            return None  # another 4xx: the store is fine and the request was wrong
        # Qdrant's in-memory mode raises plain errors, so it has to be asked.
        with suppress(Exception):
            if not self._client.collection_exists(self._collection):
                return self._missing()
        return None

    def _missing(self) -> CollectionMissingError:
        return CollectionMissingError(
            f"Collection {self._collection} does not exist yet. "
            "Run `handbook-rag index` to build it."
        )


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


def _whole_chunk(chunk_id: str, payload: dict[str, Any], collection: str) -> Chunk:
    """The chunk a stored point describes, for indexing, which stops on a broken one.

    A bare KeyError would name the missing field and not the point or the collection to
    look at. A search keeps the bare error: there a broken point is a fault in this
    service, and the API reports every error of the store as the store being unavailable.
    """
    try:
        return _from_payload(chunk_id, payload)
    except KeyError as error:
        raise VectorStoreError(
            f"Point {chunk_id} in {collection} has no {error.args[0]!r} in its payload, "
            "so it does not describe a whole chunk."
        ) from error


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
