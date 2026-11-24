"""Update the index to a newer commit of the handbook without embedding everything again.

The newer commit gets a collection of its own, as every commit does, and the older
collection is only read. Most of a handbook does not change from one day to the next, so
most chunks of the newer commit were embedded before, from exactly the same text. Their
vectors are taken from the older collection, and only the rest is embedded (ADR 0019).

"The same text" is the text that is embedded: the page's title and the headings above
the chunk, then the chunk. It is not the chunk's id. An older point has no record of that
text as such, but its payload holds everything it is built from, so it is built again
from the payload, by the same code that built it when the point was embedded.
"""

import dataclasses
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from handbook_rag.config import Settings, collection_slug
from handbook_rag.index.indexer import text_key
from handbook_rag.index.store import QdrantVectorStore, ReadableStore
from handbook_rag.ingest.models import Chunk

_FULL_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
_NAMED_OTHERWISE = "which is named otherwise, so what it was made with cannot be read from its name"
# The end of a collection's name in the form ChunkingConfig.version gives it.
_SETTINGS_NAME = re.compile(r"(?P<models>.+)-(?P<chunking>v\d+-max\d+-min\d+-overlap\d+)")


class ReuseError(ValueError):
    """Vectors cannot be reused from where they were asked for."""


def find_reuse_source(
    store: QdrantVectorStore, settings: Settings, commit: str
) -> QdrantVectorStore:
    """The collection of an older commit that ``store``'s collection may take vectors from.

    It is named by its commit, and its collection follows from that commit with the
    models and the chunking of these settings. A collection made with other ones can
    therefore not be asked for. When the one that would fit does not exist, the error
    says which collections that commit does have, and how each differs.
    """
    if not _FULL_COMMIT_HASH.fullmatch(commit):
        raise ReuseError(
            f"{commit!r} is not a full commit hash. Name the older commit by its "
            "40-character hash, as the snapshot is pinned."
        )
    name = settings.collection_for(commit)
    if name == store.collection:
        raise ReuseError(
            f"Commit {commit[:12]} is indexed in {name}, which is the collection being "
            "built. Vectors are reused from the collection of an older commit, which is "
            "only read. Name another commit, or run without --reuse-from."
        )
    source = store.with_collection(name)
    if source.exists():
        return source

    reason = (
        f"Commit {commit[:12]} has no collection made with these models and this chunking "
        f"({name}), so there are no vectors to reuse. Vectors are reused only from the same "
        "embedding model, keyword model and chunking: another model's vector for the same "
        "text means something else, and another chunking cuts the pages into other texts."
    )
    prefix = f"handbook-{commit[:8]}-"
    others = [other for other in store.collection_names() if other.startswith(prefix)]
    for other in others:
        reason += f" The store holds {other}, {_described(other.removeprefix(prefix), settings)}."
    if not others:
        reason += " The store holds no collection of that commit."
    raise ReuseError(reason)


def _described(rest: str, settings: Settings) -> str:
    """What the name of a collection of the same commit shows it was made with.

    ``rest`` is its name after the commit. A name these settings' form would give is the
    embedding model and the keyword model, then the chunking version, joined by dashes.
    Only what such a name shows is said. A name of another form, such as one with a
    word added at the end, says nothing certain about the models or the chunking.
    """
    named = _SETTINGS_NAME.fullmatch(rest)
    if named is None:
        return _NAMED_OTHERWISE
    models, chunking = named.group("models"), named.group("chunking")
    differences: list[str] = []
    embedding = collection_slug(settings.embedding_model)
    # The embedding model alone is the form the names had before keyword vectors.
    if models != embedding and not models.startswith(f"{embedding}-"):
        differences.append(f"embedding model (these settings use {settings.embedding_model})")
    if not models.endswith(f"-{collection_slug(settings.sparse_model)}"):
        differences.append(f"keyword model (these settings use {settings.sparse_model})")
    if chunking != settings.chunking.version:
        differences.append(f"chunking (these settings use {settings.chunking.version})")
    if not differences:
        return _NAMED_OTHERWISE
    return f"made with another {' and '.join(differences)}"


@dataclass(frozen=True, slots=True)
class PreviousIndex:
    """What an older collection holds, read once before anything is written."""

    collection: str
    chunks: int
    points: dict[str, str]
    """For each embedded text, by its key, one point that was embedded from it."""
    pages: dict[str, frozenset[str]]
    """For each page, a fingerprint of each of its chunks, apart from the commit."""
    seconds: float


def read_previous_index(
    source: ReadableStore, *, pages: Collection[str] | None = None
) -> PreviousIndex:
    """Read the payloads of an older collection, or of only these pages in it.

    Only payloads are read here. Vectors are fetched later, a batch at a time, for the
    points that are reused, so a whole collection's vectors are never held at once.
    """
    started = time.perf_counter()
    points: dict[str, str] = {}
    entries: defaultdict[str, set[str]] = defaultdict(set)
    count = 0
    for chunk in source.scroll_chunks(pages):
        # Two pages can hold the same text under the same title. Either point's vectors
        # will do; the first one read is kept, and the store reads in the order of ids.
        points.setdefault(text_key(chunk.text_for_embedding), chunk.chunk_id)
        entries[chunk.doc_id].add(_page_entry(chunk))
        count += 1
    return PreviousIndex(
        collection=source.collection,
        chunks=count,
        points=points,
        pages={doc_id: frozenset(found) for doc_id, found in entries.items()},
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class PageChanges:
    """How the pages of a newer commit's index differ from those of an older one.

    A page is *changed* when what the index holds for it differs: a chunk was added,
    removed or moved, or a chunk's text, title, headings, address or section is not
    the same. The commit each chunk came from is not compared, since it always differs.
    A page whose file changed only where cleaning removes the text, such as an HTML
    comment, is unchanged, because its index entries are the same.
    """

    added: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: list[str]


def compare_pages(previous: PreviousIndex, chunks: Sequence[Chunk]) -> PageChanges:
    """Compare the pages of an older collection with the chunks of a newer commit."""
    entries: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        entries[chunk.doc_id].add(_page_entry(chunk))
    before, after = set(previous.pages), set(entries)
    both = before & after
    return PageChanges(
        added=sorted(after - before),
        changed=sorted(page for page in both if previous.pages[page] != entries[page]),
        removed=sorted(before - after),
        unchanged=sorted(page for page in both if previous.pages[page] == entries[page]),
    )


def _page_entry(chunk: Chunk) -> str:
    """A fingerprint of everything a chunk is stored with, except the commit it came from."""
    fields = dataclasses.asdict(chunk)
    del fields["source_commit"]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()
