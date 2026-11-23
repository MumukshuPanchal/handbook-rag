"""The two data shapes that flow through ingestion: a page, and a retrievable piece of it."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """One handbook page after cleaning."""

    doc_id: str
    """Path of the page relative to the handbook content directory. Stable across runs."""
    title: str
    url: str
    section: str
    """Top-level handbook area, such as ``engineering``. Usable as a search filter."""
    source_commit: str
    text: str
    """The cleaned page text. Chunk offsets refer to this string."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable piece of a document."""

    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    source_commit: str
    heading_path: tuple[str, ...]
    """Headings above the chunk, outermost first."""
    start_char: int
    end_char: int
    """The chunk covers ``document.text[start_char:end_char]``."""
    text: str
    """Usually exactly that slice. A piece cut from the middle of a long table also
    starts with the table's header rows, so it still makes sense on its own."""
    token_count: int

    @property
    def context_header(self) -> str:
        """Where this chunk sits: the page title followed by its heading path."""
        return " > ".join((self.title, *self.heading_path))

    @property
    def text_for_embedding(self) -> str:
        """The text to embed: the context header, then the chunk.

        A chunk such as "Up to 16 weeks, fully paid" means little by itself. Prefixing
        "Leave > Parental leave" tells the embedding model what it is about.

        A stored point holds no copy of this text: it is built again from the point's
        payload when an update decides whether the point's vectors can be reused. So a
        change to how it is built must come with a new chunking version
        (``ChunkingConfig.version``). Otherwise vectors embedded from the old form would
        be taken for vectors of the new one, by an update and by indexing alike.
        """
        return f"{self.context_header}\n\n{self.text}"
