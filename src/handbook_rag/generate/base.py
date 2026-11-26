"""What every answer generator shares: the interface, its error, and how declining works.

A generator is the part that talks to a language model. There is one per provider, and
the rest of the application only ever sees the interface defined here.
"""

import math
import re
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from handbook_rag.generate.models import Answer, Segment
from handbook_rag.generate.prompt import GAP_FORM, NOT_FOUND_MESSAGE
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
    come to a proper end. Otherwise the reply is a decline in three cases:

    - It is the fixed sentence and nothing else, even if the model attached a citation
      to it.
    - Each citation it carries follows the fixed sentence (:func:`cites_the_decline`),
      whatever else it holds: what it cites is that the handbook lacks the answer, as in
      "I could not find this in the handbook. [1]" with another line after it.
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
    cited = [segment for segment in segments if segment.citations and _words(segment.text)]
    if cited and all(cites_the_decline(segment) for segment in cited):
        return False
    has_citations = any(segment.citations for segment in segments)
    return has_citations or declined not in said


def cites_the_decline(segment: Segment) -> bool:
    """Whether a segment carries a citation that follows the fixed declining sentence.

    A citation closes a segment, so the words just before it are the end of the segment.
    Compared as in :func:`is_answered`.
    """
    words, declined = _words(segment.text), _words(NOT_FOUND_MESSAGE)
    return bool(segment.citations) and (words == declined or words.endswith(f" {declined}"))


def holds_the_decline(segments: Sequence[Segment]) -> bool:
    """Whether a reply says, of the question or of a part of it, that the handbook lacks it.

    True for a decline, and also for an answer that covers part of the question and
    uses the fixed sentence for the rest. The words are compared as in
    :func:`is_answered`.
    """
    return _words(NOT_FOUND_MESSAGE) in _words("".join(segment.text for segment in segments))


def holds_the_gap_form(segments: Sequence[Segment]) -> bool:
    """Whether a reply holds the fixed form for a part of the question the passages do not
    answer (``prompt.GAP_FORM``), as version 2 of the instructions asks for it.

    The words are compared as in :func:`is_answered`. A reply written without the form,
    as every reply under version 1 was, is left to the list of phrases in
    ``cache/rules.py``.
    """
    return f" {_words(GAP_FORM)} " in f" {_words(''.join(s.text for s in segments))} "


def only_the_gap_form(text: str) -> bool:
    """Whether a text is nothing but pieces in the fixed form for a part the passages do not
    answer, so that it states no answer at all.

    A list marker at the start of a line is left out first. The text is then cut at line
    breaks; after a full stop, a question mark, an exclamation mark, a colon or a
    semicolon; and before "but". Every piece that holds a word must begin with the form,
    so that "… do not say more than this: new parents get sixteen weeks" and "… do not say
    who pays, but leave is sixteen weeks" are not taken for it. Words are compared as in
    :func:`is_answered`. A piece cut in a wrong place (after "e.g.") does not begin with the
    form, and the text is then not taken for one: it stays a reply for the checks to read,
    which errs towards a note.

    A claim joined into the same piece by any other word or mark ("… do not say who pays,
    and leave is sixteen weeks", "although", a dash), or written straight after a full stop
    with no white space before it ("… who pays.New parents get …", as a reply reads once its
    citation markers are taken out), is not cut off, and the text is taken for the form
    alone. That is a limit: such a reply is a decline, shown as the fixed declining sentence
    when it cites nothing and in its own words with no source when it carried markers, with
    no note, and its words are kept for an evaluation to read.

    The text is read without its citation markers: a marker does not cut it.
    """
    form = _words(GAP_FORM)
    pieces = _SENTENCE_BREAK.split(_LIST_MARKER.sub("", text.strip()))
    sentences = [_words(piece) for piece in pieces]
    spoken = [sentence for sentence in sentences if sentence]
    return bool(spoken) and all(
        sentence == form or sentence.startswith(form + " ") for sentence in spoken
    )


def read_the_gap_form_as_a_decline(
    segments: Sequence[Segment], *, answered: bool
) -> tuple[tuple[Segment, ...], bool, str]:
    """A reply that is nothing but the fixed form for a missing part, read as the decline it is.

    Instructions that give the form (``prompt.GAP_FORM_VERSIONS``) ask for it after an
    answer to the part the passages do answer. A model may write it alone, for a question
    the passages do not answer at all, in place of the fixed declining sentence. That says
    what the sentence says, so such a reply is not an answer, and no source is listed
    under it, whether or not it carries citation markers. Returns the segments, whether
    the reply is an answer, and the model's own words when the reply was read so (empty
    otherwise).

    - **It cites nothing**: it is shown as the fixed declining sentence, with no note,
      since nothing of it was passed on while it was written.
    - **It carries citation markers**: it is shown in the model's own words, without the
      markers, which would point at no source. A citation lets the text before it be
      passed on while the reply is written (``checks.HeldBack``), so its words may already
      have reached the reader, and they are not taken back.

    The words are what :func:`only_the_gap_form` takes for the form alone, which includes a
    claim joined into a piece by "and", "although" or a dash: such a reply is counted as a
    decline though it states something. Where it carried markers, the claim stays in the
    words shown. A reply that is not an answer, or that holds anything besides the form,
    is returned as it is.
    """
    text = "".join(segment.text for segment in segments)
    if not answered or not only_the_gap_form(text):
        return tuple(segments), answered, ""
    if any(segment.citations for segment in segments):
        return tuple(Segment(segment.text) for segment in segments), False, text.strip()
    return (Segment(NOT_FOUND_MESSAGE),), False, text.strip()


def plain_words(text: str) -> str:
    """A text as its words alone, as a reply is compared with the fixed sentence.

    For whoever looks for other phrases in a reply (``cache/rules.py``), so that case,
    punctuation and Markdown count for as little there as they do here.
    """
    return _words(text)


# Where one piece of a reply ends and the next begins, for finding the fixed form alone: a
# line break; a full stop, a question mark, an exclamation mark, a colon or a semicolon
# followed by white space, also after a digit ("rule 2. New parents ..."); or "but",
# with a comma before it or not.
_SENTENCE_BREAK = re.compile(r"\n+|(?<=[.!?:;])\s+|,?\s+but\s+", re.IGNORECASE)
# A list marker at the start of a line: a bullet, or a number with a stop or a bracket.
# Taken out before the text is cut, so that the stop of "1." does not end a piece.
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+", re.MULTILINE)


def _words(text: str) -> str:
    """The words of a text in lower case, single spaces between them, nothing else.

    A word is letters and digits of any script, so an answer in Japanese has words too.
    """
    return " ".join(re.findall(r"[^\W_]+", text.lower()))
