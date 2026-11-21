"""Embed chunks and store them, doing only the work that is still needed."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from handbook_rag.index.embedder import Embedder
from handbook_rag.index.store import VectorStore
from handbook_rag.ingest.models import Chunk


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one indexing run did."""

    total: int
    embedded: int
    already_indexed: int
    removed: int
    seconds: float


def index_chunks(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder,
    store: VectorStore,
    batch_size: int = 256,
    on_progress: Callable[[int, int], None] | None = None,
) -> IndexReport:
    """Make the store hold exactly these chunks.

    Indexing is idempotent. A chunk id is derived from the chunk's content and position,
    so a chunk that is already stored is skipped without being embedded again, and
    running the command twice in a row does no work the second time. Chunks in the store
    that are no longer produced are removed.
    """
    started = time.perf_counter()
    store.ensure_collection(embedder.dimension)

    stored = store.existing_ids([chunk.chunk_id for chunk in chunks])
    missing = [chunk for chunk in chunks if chunk.chunk_id not in stored]
    # Embedding is fastest when texts in a batch have similar lengths, because a batch
    # is padded to its longest text.
    missing.sort(key=lambda chunk: chunk.token_count, reverse=True)

    done = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = embedder.embed_documents([chunk.text_for_embedding for chunk in batch])
        store.upsert(batch, vectors)
        done += len(batch)
        if on_progress is not None:
            on_progress(done, len(missing))

    removed = store.delete_except({chunk.chunk_id for chunk in chunks})
    return IndexReport(
        total=len(chunks),
        embedded=len(missing),
        already_indexed=len(chunks) - len(missing),
        removed=removed,
        seconds=time.perf_counter() - started,
    )
