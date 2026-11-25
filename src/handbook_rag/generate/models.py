"""What the generator returns: an answer, the passages it cites, and what it cost."""

from dataclasses import dataclass, field
from typing import Literal

from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk


@dataclass(frozen=True, slots=True)
class Citation:
    """A passage the model relied on for part of its answer."""

    chunk: Chunk
    cited_text: str | None = None
    """The exact words the model relied on, when the provider's API reports them.

    ``None`` when the model named the passage by its number: the citation then covers the
    whole passage.
    """


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
class CacheHit:
    """Where an answer came from that was not written for this question now.

    The answer cache (``cache/answers.py``) served it. Whoever shows the answer shows
    this too: an answer matched by meaning was written for another wording, and a reader
    who sees that wording can tell when the match was wrong.
    """

    tier: Literal["exact", "semantic"]
    """``exact``: stored for this very question. ``semantic``: stored for a question
    that a second model judged to mean the same."""
    question: str
    """The stored question the answer was written for, as it was asked then."""
    stored_at: str
    """When the answer was stored: a time in UTC, in ISO 8601."""
    similarity: float | None = None
    """The cosine of the two questions' embeddings. ``None`` for an exact match, which is
    found by a key and compares no embeddings."""
    score: float | None = None
    """The verifier's weaker direction, which reached the threshold. ``None`` for an
    exact match, which is not verified."""


@dataclass(frozen=True, slots=True)
class ReplyCheck:
    """A check of the reply in code that fired, and what it did (``generate/checks.py``).

    A change to a reply is never silent: whoever shows the answer shows ``note`` with it.
    """

    name: str
    """Which check: one of ``generate.checks.CHECKS``."""
    action: str
    """What it did: ``dropped_line``, ``dropped_words``, ``declined`` or ``marked``."""
    note: str
    """What the reader is told, in fixed words."""
    removed: str = ""
    """The model's words that the check took out, for an evaluation to read. Never shown
    to the asker, and empty when the reply was only marked."""


@dataclass(frozen=True, slots=True)
class Answer:
    """The assistant's reply to one question."""

    question: str
    segments: tuple[Segment, ...]
    answered: bool
    """False when the reply is not an answer: the assistant said the handbook does not
    contain it, the provider withheld it, or it did not come to a proper end."""
    model: str
    stop_reason: str
    usage: TokenUsage = TokenUsage()
    seconds: float = 0.0
    """How long the request that wrote the reply took. Failed requests before it are not
    counted in."""
    seconds_to_first_text: float | None = None
    """Seconds from the request to the first words the model wrote."""
    seconds_to_first_shown: float | None = None
    """Seconds from the request to the first words passed on to the reader, or that would
    be passed on when nobody listens. The same as ``seconds_to_first_text`` under
    instructions that stream text as it comes; later under those that hold text back until
    a citation follows it (``checks.HeldBack``), and that wait is a cost of the checks.
    ``None`` when no text was passed on."""
    retrieved: tuple[ScoredChunk, ...] = field(default=())
    """The chunks the model was given, in ranked order."""
    requests: int = 1
    """How many requests to a model the answer took, failed ones included. 0 when the
    assistant declined without asking a model, because nothing was found, and 0 when the
    answer came from the cache."""
    fallback: bool = False
    """Whether the fallback model wrote the reply, because the answer model had given up."""
    cached: CacheHit | None = None
    """Set when the answer cache served the answer. The model, the stop reason and the
    text are then those of the stored answer; the tokens and the time are zero, because
    no model was asked; and no passages were retrieved."""
    checks: tuple[ReplyCheck, ...] = ()
    """The checks of the reply in code that fired, in the order they were made. The
    segments are the reply as the checks left it. Empty when none fired, and for every
    reply written under instructions that have no checks."""
    model_text: str = ""
    """What the model wrote, when the generator shows the fixed declining sentence in its
    place: a reply that was nothing but the fixed form for a part the passages do not
    answer (``base.read_the_gap_form_as_a_decline``). That is the model declining in other
    words, not a check, so no note goes with it. Empty for every other reply; the words a
    check took out are kept by the check."""

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
