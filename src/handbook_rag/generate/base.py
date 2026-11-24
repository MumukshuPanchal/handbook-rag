"""What every answer generator shares: the interface, its error, and how declining works.

A generator is the part that talks to a language model. There is one per provider, and
the rest of the application only ever sees the interface defined here.
"""

import math
import re
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from handbook_rag.generate.models import Answer, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk

FailureReason = Literal["rate_limit", "quota", "configuration", "unavailable", "timeout"]
"""What kind of failure stopped a request, which decides what may help.

- ``rate_limit``: too many requests in a short time. Trying again soon may work.
- ``quota``: the allowance for the day is used up. Trying again today will not work.
- ``configuration``: no credential, a rejected one, a model that cannot be used, or a
  request the provider refuses as it was sent. Nothing works until the settings or the
  code change.
- ``timeout``: the API went silent for longer than the client's time limit, while
  connecting or between two pieces of a reply.
- ``unavailable``: anything else. The API failed, could not be reached, or sent a reply
  that could not be read.
"""


class GenerationError(RuntimeError):
    """The model could not be reached or refused the request.

    The message is for the person who runs the application: it can name settings and
    files. A caller of the HTTP API is told only the ``reason``, in fixed words, and
    when to try again if that is known.

    The error also says what the answer had cost when it failed. Those attributes are
    filled in on its way out, by the code that sent the request and the code that
    searched, so the generators themselves never set them.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: FailureReason = "unavailable",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: FailureReason = reason
        # A wait that is infinite, not a number or negative is no wait at all. It is
        # dropped here, so that no reader of the error has to check for it.
        usable = retry_after is not None and math.isfinite(retry_after) and retry_after >= 0
        self.retry_after = retry_after if usable else None
        """Seconds to wait before trying again, when the provider said so."""
        self.requests = 0
        """How many requests to a model the answer had made when it failed, this one included."""
        self.fallback = False
        """Whether this is the failure of the fallback model, tried after the answer model."""
        self.before_fallback: GenerationError | None = None
        """Why the answer model gave up, when this is the failure of the fallback model."""
        self.retrieved: tuple[ScoredChunk, ...] = ()
        """The passages the search had found, so that a caller can show them without
        searching again. Empty when the failure came before a search."""


class Generator(Protocol):
    """Anything that can write an answer to a question from a set of chunks.

    Besides writing, a generator says who serves its model and which model it asks for.
    The pipeline needs both for the trace of a model call, also when the call fails and
    there is no reply to read them from.
    """

    @property
    def provider(self) -> str:
        """Who serves the model, by the name OpenTelemetry's conventions give it."""
        ...

    @property
    def operation(self) -> str:
        """The kind of request made, by the name OpenTelemetry's conventions give it."""
        ...

    @property
    def model(self) -> str:
        """The model asked for. The reply names the model that served it."""
        ...

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer: ...


def decline_without_context(
    question: str, model: str, on_text: Callable[[str], None] | None
) -> Answer:
    """The answer when nothing was retrieved.

    There is nothing to ground an answer in, so the assistant says so itself, without
    spending a model call.
    """
    if on_text is not None:
        on_text(NOT_FOUND_MESSAGE)
    return Answer(
        question=question,
        segments=(Segment(NOT_FOUND_MESSAGE),),
        answered=False,
        model=model,
        stop_reason="no_context",
        requests=0,
    )


def is_answered(segments: Sequence[Segment], *, refused: bool) -> bool:
    """Whether the assistant gave an answer, as opposed to declining or being stopped.

    ``refused`` is the generator's own signal that the reply was withheld or did not
    come to a proper end. Otherwise the reply is a decline in two cases:

    - It is the fixed sentence and nothing else, even if the model attached a citation
      to it.
    - It contains the fixed sentence and cites nothing. An answer that covers part of
      the question may use the sentence for the part it could not find, and still cites
      what it did find.

    Only the words are compared. Capitals, the full stop, quotation marks and Markdown
    emphasis around the sentence make no difference.
    """
    if refused:
        return False
    said = _words("".join(segment.text for segment in segments))
    if not said:
        return False
    declined = _words(NOT_FOUND_MESSAGE)
    if said == declined:
        return False
    has_citations = any(segment.citations for segment in segments)
    return has_citations or declined not in said


def holds_the_decline(segments: Sequence[Segment]) -> bool:
    """Whether a reply says, of the question or of a part of it, that the handbook lacks it.

    True for a decline, and also for an answer that covers part of the question and
    uses the fixed sentence for the rest. The words are compared as in
    :func:`is_answered`.
    """
    return _words(NOT_FOUND_MESSAGE) in _words("".join(segment.text for segment in segments))


def plain_words(text: str) -> str:
    """A text as its words alone, as a reply is compared with the fixed sentence.

    For whoever looks for other phrases in a reply (``cache/rules.py``), so that case,
    punctuation and Markdown count for as little there as they do here.
    """
    return _words(text)


def _words(text: str) -> str:
    """The words of a text in lower case, single spaces between them, nothing else.

    A word is letters and digits of any script, so an answer in Japanese has words too.
    """
    return " ".join(re.findall(r"[^\W_]+", text.lower()))
