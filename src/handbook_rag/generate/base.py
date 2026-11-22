"""What every answer generator shares: the interface, its error, and how declining works.

A generator is the part that talks to a language model. There is one per provider, and
the rest of the application only ever sees the interface defined here.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

from handbook_rag.generate.models import Answer, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.ingest.models import Chunk


class GenerationError(RuntimeError):
    """The model could not be reached or refused the request."""


class Generator(Protocol):
    """Anything that can write an answer to a question from a set of chunks."""

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
    """
    if refused:
        return False
    said = "".join(segment.text for segment in segments).strip().lower()
    if not said:
        return False
    declined = NOT_FOUND_MESSAGE.rstrip(".").lower()
    if said.rstrip(".") == declined:
        return False
    has_citations = any(segment.citations for segment in segments)
    return has_citations or declined not in said
