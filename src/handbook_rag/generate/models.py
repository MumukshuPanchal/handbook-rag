"""What the generator returns: an answer, the passages it cites, and what it cost."""

from dataclasses import dataclass, field

from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk


@dataclass(frozen=True, slots=True)
class Citation:
    """A passage the model relied on for part of its answer."""

    chunk: Chunk
    cited_text: str
    """The exact text of the cited passage, as returned by the API."""


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of answer text, with the passages that support it (possibly none)."""

    text: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by the API for one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Answer:
    """The assistant's reply to one question."""

    question: str
    segments: tuple[Segment, ...]
    answered: bool
    """False when the assistant said the handbook does not contain the answer."""
    model: str
    stop_reason: str
    usage: TokenUsage = TokenUsage()
    seconds: float = 0.0
    seconds_to_first_text: float | None = None
    retrieved: tuple[ScoredChunk, ...] = field(default=())
    """The chunks the model was given, in ranked order."""

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)

    @property
    def sources(self) -> tuple[Chunk, ...]:
        """The distinct chunks cited, in the order they are first cited."""
        seen: dict[str, Chunk] = {}
        for segment in self.segments:
            for citation in segment.citations:
                seen.setdefault(citation.chunk.chunk_id, citation.chunk)
        return tuple(seen.values())

    @property
    def text_with_markers(self) -> str:
        """The answer with a numbered marker such as [1] after each supported statement."""
        numbers = {chunk.chunk_id: number for number, chunk in enumerate(self.sources, start=1)}
        parts: list[str] = []
        for segment in self.segments:
            cited = sorted({numbers[c.chunk.chunk_id] for c in segment.citations})
            parts.append(segment.text + "".join(f"[{number}]" for number in cited))
        return "".join(parts)
