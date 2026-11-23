"""Embed chunks and store them, doing only the work that is still needed."""

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from handbook_rag.index.embedder import Embedder
from handbook_rag.index.sparse import SparseEncoder
from handbook_rag.index.store import ReadableStore, StoredPoint, VectorStore
from handbook_rag.ingest.models import Chunk

Progress = Callable[[int, int], None]
"""Called with how many items are done and how many there are in all."""


def text_key(text: str) -> str:
    """A short key for the text a vector is embedded from, to look the vector up by."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Reuse:
    """An older collection whose vectors may be used instead of embedding again.

    ``points`` maps the key of an embedded text (:func:`text_key`) to a point of
    ``source`` that was embedded from that text. It is a guide only: each point is read
    again together with its vectors, and used only if it still holds that text.
    """

    source: ReadableStore
    points: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one indexing run did."""

    total: int
    embedded: int
    already_indexed: int
    removed: int
    seconds: float
    reused: int = 0
    """Chunks whose vectors were taken from an older collection instead of embedded."""
    replaced: int = 0
    """Chunks stored under their id with other details, such as an older title."""
    reuse_seconds: float = 0.0
    embed_seconds: float = 0.0


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder,
    sparse_encoder: SparseEncoder,
    store: VectorStore,
    batch_size: int = 256,
    on_progress: Progress | None = None,
    reuse: Reuse | None = None,
    on_reuse_progress: Progress | None = None,
) -> IndexReport:
    """Make the store hold exactly these chunks.

    Indexing is idempotent. A chunk that is already stored as it is now is skipped
    without being embedded again, so running the command twice in a row does no work the
    second time. Chunks in the store that are no longer produced are removed.

    With ``reuse``, a chunk that has to be stored takes its vectors from the older
    collection when a point there was embedded from exactly the same text, and is
    embedded only otherwise. The older collection is only read. Each stored payload
    comes from the chunk, never from the point the vectors were taken from.
    """
    started = time.perf_counter()
    store.ensure_collection(embedder.dimension)

    # A chunk's id is made from its page, its position and its text, but what is
    # embedded also holds the page's title and the headings above the chunk, and what
    # is stored also holds the page's address and section. Those can change while the
    # id stays: a new handbook address in the settings, a change to how titles are
    # read, a heading renamed to a word of the same length. So a chunk is skipped only
    # when the point stored under its id describes exactly this chunk.
    stored = store.stored_chunks([chunk.chunk_id for chunk in chunks])
    pending = [chunk for chunk in chunks if stored.get(chunk.chunk_id) != chunk]
    replaced = sum(chunk.chunk_id in stored for chunk in pending)

    reuse_started = time.perf_counter()
    missing, reused = pending, 0
    if reuse is not None:
        missing, reused = _store_reused(pending, reuse, store, batch_size, on_reuse_progress)
    reuse_seconds = time.perf_counter() - reuse_started

    # Embedding is fastest when texts in a batch have similar lengths, because a batch
    # is padded to its longest text.
    missing.sort(key=lambda chunk: chunk.token_count, reverse=True)

    embed_started = time.perf_counter()
    done = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        texts = [chunk.text_for_embedding for chunk in batch]
        store.upsert(batch, embedder.embed_documents(texts), sparse_encoder.encode_documents(texts))
        done += len(batch)
        if on_progress is not None:
            on_progress(done, len(missing))
    embed_seconds = time.perf_counter() - embed_started

    removed = store.delete_except({chunk.chunk_id for chunk in chunks})
    return IndexReport(
        total=len(chunks),
        embedded=len(missing),
        already_indexed=len(chunks) - len(pending),
        removed=removed,
        seconds=time.perf_counter() - started,
        reused=reused,
        replaced=replaced,
        reuse_seconds=reuse_seconds,
        embed_seconds=embed_seconds,
    )


def _store_reused(
    chunks: Sequence[Chunk],
    reuse: Reuse,
    store: VectorStore,
    batch_size: int,
    on_progress: Progress | None,
) -> tuple[list[Chunk], int]:
    """Store every chunk whose vectors the older collection holds. Return the rest.

    The match is on the text that is embedded, not on the id. A chunk that only moved
    down its page has a new id and the same text, and its vectors are still right. A
    chunk under a renamed title can keep its id while its text has changed, and its
    vectors are not.
    """
    keys = {chunk.chunk_id: text_key(chunk.text_for_embedding) for chunk in chunks}
    lendable = [chunk for chunk in chunks if keys[chunk.chunk_id] in reuse.points]
    missing = [chunk for chunk in chunks if keys[chunk.chunk_id] not in reuse.points]
    stored = 0
    for start in range(0, len(lendable), batch_size):
        batch = lendable[start : start + batch_size]
        wanted = [reuse.points[keys[chunk.chunk_id]] for chunk in batch]
        found = reuse.source.points(sorted(set(wanted)))
        usable: list[tuple[Chunk, StoredPoint]] = []
        for chunk, point_id in zip(batch, wanted, strict=True):
            point = found.get(point_id)
            # The older collection was read a while ago. Checking the text again, on
            # the point as read with its vectors, means a vector is never taken on a
            # key or an id alone, even if that collection changed in the meantime.
            if point is not None and point.chunk.text_for_embedding == chunk.text_for_embedding:
                usable.append((chunk, point))
            else:
                missing.append(chunk)
        if usable:
            store.upsert(
                [chunk for chunk, _ in usable],
                [point.dense for _, point in usable],
                [point.sparse for _, point in usable],
            )
        stored += len(usable)
        if on_progress is not None:
            on_progress(start + len(batch), len(lendable))
    return missing, stored
