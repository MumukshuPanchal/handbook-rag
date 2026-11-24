"""Where the answer cache keeps its answers: a Qdrant collection of its own.

It sits beside the handbook's collection in the same store, so nothing new has to be
run. A point is one stored answer. Its vector is the embedding of the question the
answer was written for. Its payload is the answer with the passages it cites, whole, so
that serving it needs nothing from the handbook's collection.

The collection is made before the first question is looked up: by a service when it
starts, and by a command before it asks (:meth:`QdrantAnswerStore.ensure`). Making it
takes the store seconds on a busy machine, and a collection that the first write made
would be made while questions were already being looked up: for those seconds a lookup
finds nothing, or fails. A service that started without its store makes the collection
with its first write after all, and has those seconds once.

Emptying the cache removes the answers and keeps the collection, so that a service that
runs under the same settings does not meet a collection that is being made again. Only a
collection left behind under other settings is removed whole. Reading a collection that
is not there finds nothing, which is not a failure.
"""

import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from opentelemetry.trace import Span, SpanKind
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from handbook_rag.generate.models import Citation, Segment
from handbook_rag.ingest.models import Chunk
from handbook_rag.tracing import stage

COLLECTION_PREFIX = "answers-"
"""How the name of every collection of cached answers starts, whatever settings it is for."""
# A name as collection_name makes it: the prefix, a commit as a collection name holds it,
# and sixteen hexadecimal digits for the settings.
_NAME = re.compile(rf"{COLLECTION_PREFIX}(?P<commit>[a-z0-9-]*)-[0-9a-f]{{16}}")
# The two fields every search by meaning filters on, and the kind of index each has.
_INDEXED = {
    "scope": models.PayloadSchemaType.KEYWORD,
    "by_meaning": models.PayloadSchemaType.BOOL,
}


@dataclass(frozen=True, slots=True)
class Entry:
    """One stored answer, and the question it was written for."""

    entry_id: str
    """The point's id: made from ``comparable`` and ``scope``, so the same question asked
    of the same part of the handbook has one entry, and a newer answer replaces it."""
    question: str
    """The question as it was asked. A reader of the cached answer is shown it."""
    comparable: str
    """The question as the exact tier compares it."""
    scope: str
    """Which part of the handbook the answer was written from: all of it, or one section."""
    segments: tuple[Segment, ...]
    answered: bool
    model: str
    stop_reason: str
    stored_at: str
    by_meaning: bool
    """Whether the entry may be served to a question worded differently. Decided when it
    is stored (``cache/answers.py``), and filtered on by the store, so that an entry that
    may not be served takes no candidate's place."""


@dataclass(frozen=True, slots=True)
class Held:
    """How much a cache holds."""

    answers: int
    """Every stored reply."""
    declines: int
    """Those that are not an answer. They are served to the very same question only."""
    by_meaning: int
    """Those that may be served to a question worded differently."""


class AnswerStore(Protocol):
    """What the answer cache needs of the place its answers are kept."""

    def ensure(self, dimension: int) -> None:
        """Make the place ready for vectors of this size, before anything is looked up."""
        ...

    def get(self, entry_id: str) -> Entry | None:
        """The entry stored under this id, or ``None``."""
        ...

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        """The entries of one scope that may be served by meaning, nearest first.

        Each comes with the cosine of its question's embedding and ``vector``. At most
        ``limit`` are returned, and none below ``floor``.
        """
        ...

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        """Store an entry, in place of any entry with the same id."""
        ...


class QdrantAnswerStore:
    """One Qdrant collection holding the cached answers of one configuration."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection
        self._made = False
        # Answers are stored from a thread of their own, and two processes may share a store.
        self._making = threading.Lock()

    @property
    def collection(self) -> str:
        return self._collection

    def get(self, entry_id: str) -> Entry | None:
        with self._call("retrieve"):
            try:
                records = self._client.retrieve(
                    self._collection, ids=[entry_id], with_payload=True, with_vectors=False
                )
            except Exception as error:
                if self._nothing_stored(error):
                    return None
                raise
        return _entry(str(records[0].id), records[0].payload or {}) if records else None

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        with self._call("query", limit=limit) as span:
            try:
                response = self._client.query_points(
                    self._collection,
                    query=list(vector),
                    limit=limit,
                    # Inside the search, so that an entry that may not be served, or one
                    # written from another part of the handbook, takes no nearer one's place.
                    query_filter=models.Filter(
                        must=[
                            _is("scope", scope),
                            _is("by_meaning", True),
                        ]
                    ),
                    score_threshold=floor,
                    with_payload=True,
                )
            except Exception as error:
                if self._nothing_stored(error):
                    return []
                raise
            span.set_attribute("handbook_rag.results.count", len(response.points))
        return [
            (_entry(str(point.id), point.payload or {}), point.score) for point in response.points
        ]

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        point = models.PointStruct(id=entry.entry_id, vector=list(vector), payload=_payload(entry))
        try:
            # Nothing to do when the collection was made before, as it should have been.
            self.ensure(len(vector))
            self._client.upsert(self._collection, points=[point], wait=True)
        except Exception as error:
            # Only a collection that has gone is worth a second try. Any other failure,
            # a refused point for one, would fail the same way again.
            if not self._nothing_stored(error):
                raise
            # The collection was removed after this store had made it, as
            # `cache-clear --all` does to the collections of other settings. It is made
            # again, and the answer is stored.
            self._made = False
            self.ensure(len(vector))
            self._client.upsert(self._collection, points=[point], wait=True)

    def held(self) -> Held:
        """How many answers the collection holds. Nothing, when it was never made."""

        def count(*conditions: models.FieldCondition) -> int:
            only = models.Filter(must=list(conditions)) if conditions else None
            return self._client.count(self._collection, count_filter=only, exact=True).count

        try:
            return Held(
                answers=count(),
                declines=count(_is("answered", False)),
                by_meaning=count(_is("by_meaning", True)),
            )
        except Exception as error:
            if self._nothing_stored(error):
                return Held(0, 0, 0)
            raise

    def clear(self) -> int:
        """Remove every stored answer and keep the collection. Returns how many there were.

        The collection stays, with its indexes. A service that runs on this collection
        goes on looking questions up and storing answers, and does not meet a collection
        that is being made again.
        """
        removed = self.held().answers
        if removed:
            self._client.delete(
                self._collection,
                # A filter without a condition holds for every point.
                points_selector=models.FilterSelector(filter=models.Filter()),
                wait=True,
            )
        return removed

    def remove(self) -> int:
        """Remove the collection with everything in it. Returns how many answers it held.

        For a collection that was left behind under other settings and is read by nothing
        any more. A service that does still run under those settings makes the collection
        again with the next answer it stores.
        """
        removed = self.held().answers
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._made = False
        return removed

    def ensure(self, dimension: int) -> None:
        """Make the collection and its two indexes, whichever of them is missing.

        Asked once for each store, then remembered. When everything is there, that is one
        question to the store. A collection that exists without an index, because the
        process that was making it ended in between, gets the index now.
        """
        if self._made:
            return
        with self._making:
            if self._made:
                return
            indexed = self._indexed_fields()
            if indexed is None:
                try:
                    self._client.create_collection(
                        self._collection,
                        vectors_config=models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        ),
                    )
                except Exception:
                    # Another process may have made it in the same moment.
                    if not self._client.collection_exists(self._collection):
                        raise
                indexed = set()
            for field, kind in _INDEXED.items():
                if field not in indexed:
                    self._client.create_payload_index(
                        self._collection, field_name=field, field_schema=kind
                    )
            self._made = True

    def _indexed_fields(self) -> set[str] | None:
        """The payload fields the collection has an index for. ``None``: no collection."""
        try:
            described = self._client.get_collection(self._collection)
        except Exception as error:
            if self._nothing_stored(error):
                return None
            raise
        return set(described.payload_schema or {})

    def _nothing_stored(self, error: Exception) -> bool:
        """Whether a failed call means only that the collection does not exist.

        That is the state of a cache nothing was stored in yet, or one that was emptied,
        and it is no failure. A store that did not answer at all is not asked anything
        more, which could take as long again.
        """
        if isinstance(error, ResponseHandlingException):
            return False
        if isinstance(error, UnexpectedResponse):
            return error.status_code == 404
        # Qdrant's in-memory mode raises plain errors, so it has to be asked.
        with suppress(Exception):
            return not self._client.collection_exists(self._collection)
        return False

    @contextmanager
    def _call(self, operation: str, **attributes: int) -> Iterator[Span]:
        """The span of one read, named and described as the handbook's own searches are."""
        with stage(f"vector_store.{operation}", kind=SpanKind.CLIENT, **attributes) as span:
            span.set_attribute("db.system.name", "qdrant")
            span.set_attribute("db.collection.name", self._collection)
            span.set_attribute("db.operation.name", operation)
            yield span


def collection_name(commit: str, settings: str) -> str:
    """The name of the collection for one handbook commit under one set of settings.

    ``commit`` is the start of the commit as a collection name can hold it, for a person
    to read, and ``settings`` sixteen hexadecimal digits that stand for everything else
    that decides an answer (``wiring.answer_cache_collection``).
    """
    return f"{COLLECTION_PREFIX}{commit}-{settings}"


def answer_collections(client: QdrantClient, *, commit: str | None = None) -> list[str]:
    """The collections of cached answers in a store, for whatever settings they were made.

    Only a name of the form :func:`collection_name` gives is taken for one. A collection
    that something else made is left alone, also when its name starts the same way.
    With ``commit``, only the collections of that handbook commit.
    """
    found = []
    for item in client.get_collections().collections:
        named = _NAME.fullmatch(item.name)
        if named is not None and commit in (None, named["commit"]):
            found.append(item.name)
    return sorted(found)


def _is(field: str, value: str | bool) -> models.FieldCondition:
    return models.FieldCondition(key=field, match=models.MatchValue(value=value))


def _payload(entry: Entry) -> dict[str, Any]:
    """An entry as the store holds it. Each cited passage is written once, whole."""
    chunks: dict[str, dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []
    for segment in entry.segments:
        cited: list[dict[str, Any]] = []
        for citation in segment.citations:
            chunk = citation.chunk
            chunks.setdefault(
                chunk.chunk_id, {**asdict(chunk), "heading_path": list(chunk.heading_path)}
            )
            cited.append({"chunk_id": chunk.chunk_id, "cited_text": citation.cited_text})
        segments.append({"text": segment.text, "citations": cited})
    return {
        "question": entry.question,
        "comparable": entry.comparable,
        "scope": entry.scope,
        "answered": entry.answered,
        "by_meaning": entry.by_meaning,
        "stored_at": entry.stored_at,
        "model": entry.model,
        "stop_reason": entry.stop_reason,
        "segments": segments,
        "chunks": chunks,
    }


def _entry(entry_id: str, payload: dict[str, Any]) -> Entry:
    """The entry a payload describes. A payload of another shape is an error, which the
    cache treats as it treats a store that is down: the question is answered afresh."""
    chunks = {
        chunk_id: Chunk(**{**chunk, "heading_path": tuple(chunk["heading_path"])})
        for chunk_id, chunk in payload["chunks"].items()
    }
    return Entry(
        entry_id=entry_id,
        question=str(payload["question"]),
        comparable=str(payload["comparable"]),
        scope=str(payload["scope"]),
        segments=tuple(
            Segment(
                str(segment["text"]),
                tuple(
                    Citation(chunks[cited["chunk_id"]], cited["cited_text"])
                    for cited in segment["citations"]
                ),
            )
            for segment in payload["segments"]
        ),
        answered=bool(payload["answered"]),
        model=str(payload["model"]),
        stop_reason=str(payload["stop_reason"]),
        stored_at=str(payload["stored_at"]),
        by_meaning=bool(payload["by_meaning"]),
    )
