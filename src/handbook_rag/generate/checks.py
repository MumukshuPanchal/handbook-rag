"""Checks of a reply in code, made before anyone is given it (issue 66).

The adversarial tests found what a planted instruction leaves in a reply when it gets
through: a closing line with an address that came from the planted text, a reply that is
nothing but a made-up word and cites nothing, and a copy of the search results. Each
check looks for one of those shapes, in words any reply can be read in, and does one
thing when it finds it. None of them looks at what a made-up word or address of the test
cases looks like: the address of a planted line is found as an address, whatever its
host.

- **An address in a closing line that cites nothing** (``address_in_closing_line``): the
  last line of the reply, when it stands after the last citation, holds a web or email
  address. *The line is dropped.* The answer above it is kept: in the baseline run, each
  planted line of this kind came under an answer that was right.

  Issue 66 words this check more narrowly: an address *found in no cited passage*. That
  rule leaves a planted line in whenever the planted text sits in the passage the answer
  cites, since the planted address is then in a cited passage; so it would have left in
  one of the three planted lines that got through in the baseline run
  (``inj-append_link-b08``). The check was widened before any reply of the hardened
  assistant was seen. It costs no answer that can be measured: none of the 129 answers of
  the first answer run has a closing line without a citation that holds an address. What
  it still lets through: an address inside a sentence that carries a citation, an address
  anywhere before the last citation, and a host written without a scheme or a path.
- **A reply that cites nothing at all** (``cites_nothing``): an answer with no citation.
  *The reply is declined.* Nothing in it rests on a passage a reader could check. It is
  the shape of the reply that obeyed a forged message from the system, and it costs an
  answer whose model left out its markers: one of the 129 answers of the first answer run.
- **A reply that is mostly a copy of the passages** (``mostly_a_copy``): at least
  :data:`COPY_MIN_WORDS` words, of which at least :data:`COPY_SHARE` stand in runs of
  :data:`COPY_RUN_WORDS` words or more that a passage holds word for word. *The reply is
  only marked.* What it repeats is handbook text, which the search gives anyway, and a
  long answer that quotes a procedure should not be lost.

Each check that fires is recorded on the answer with a note in fixed words, which every
surface shows with it: no reply is changed in silence. The words the model wrote and a
check took out are kept for an evaluation to read, and never shown.

So that what was streamed is never taken back, the generator passes text on only once a
citation follows it (:class:`HeldBack`): the text after the last citation, which is all a
dropped line or a declined reply can hold, waits for the end of the reply and the checks.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from handbook_rag.generate.base import plain_words
from handbook_rag.generate.models import ReplyCheck, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.ingest.models import Chunk

ADDRESS_IN_CLOSING_LINE: Final = "address_in_closing_line"
CITES_NOTHING: Final = "cites_nothing"
MOSTLY_A_COPY: Final = "mostly_a_copy"
CHECKS: Final = (CITES_NOTHING, ADDRESS_IN_CLOSING_LINE, MOSTLY_A_COPY)
"""The checks, in the order they are made."""

DROPPED_LINE: Final = "dropped_line"
DECLINED: Final = "declined"
MARKED: Final = "marked"

NOTES: Final = {
    ADDRESS_IN_CLOSING_LINE: (
        "A closing line was left out of this answer: it cited no passage and held a web or "
        "email address."
    ),
    CITES_NOTHING: (
        "The reply the model wrote cited none of the passages, so it is not shown: a reply "
        "that rests on no passage cannot be checked against the handbook."
    ),
    MOSTLY_A_COPY: "Most of this reply repeats the passages it was given, word for word.",
}
"""What the reader is told when each check fires."""

COPY_RUN_WORDS: Final = 8
"""How many words in a row a passage must hold for them to count as copied."""
COPY_SHARE: Final = 0.75
"""The share of a reply's words that must be copied for the reply to be marked."""
COPY_MIN_WORDS: Final = 200
"""How long a reply must be to be marked. The instructions ask for a short paragraph or a
few bullet points, and a short answer that quotes a list of steps is an answer."""

# A web address with its scheme or "www.", an email address, or a host with a path. A
# host alone, such as "GitLab.com", is not taken for an address: product names are
# written that way in ordinary answers.
_ADDRESS = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]{}\"'`]+"
    r"|[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}/[^\s<>()\[\]{}\"'`]*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CheckedReply:
    """A reply as the checks left it."""

    segments: tuple[Segment, ...]
    answered: bool
    checks: tuple[ReplyCheck, ...]


def check_reply(
    segments: Sequence[Segment], chunks: Sequence[Chunk], *, answered: bool
) -> CheckedReply:
    """Make every check on one reply, in the order of :data:`CHECKS`.

    A reply that is not an answer, such as a decline or a reply that was withheld, is
    left as it is. A reply declined for citing nothing is checked no further.
    """
    replied = tuple(segments)
    if not answered:
        return CheckedReply(replied, False, ())
    if not any(segment.citations for segment in replied):
        removed = "".join(segment.text for segment in replied).strip()
        declined = ReplyCheck(CITES_NOTHING, DECLINED, NOTES[CITES_NOTHING], removed)
        return CheckedReply((Segment(NOT_FOUND_MESSAGE),), False, (declined,))
    checks: list[ReplyCheck] = []
    kept, dropped = without_a_closing_address(replied)
    if dropped is not None:
        checks.append(
            ReplyCheck(
                ADDRESS_IN_CLOSING_LINE, DROPPED_LINE, NOTES[ADDRESS_IN_CLOSING_LINE], dropped
            )
        )
    text = "".join(segment.text for segment in kept)
    if len(plain_words(text).split()) >= COPY_MIN_WORDS and copied_share(text, chunks) >= (
        COPY_SHARE
    ):
        checks.append(ReplyCheck(MOSTLY_A_COPY, MARKED, NOTES[MOSTLY_A_COPY]))
    return CheckedReply(kept, True, tuple(checks))


def addresses(text: str) -> list[str]:
    """The web and email addresses in a text, as they are written there."""
    return [match.group() for match in _ADDRESS.finditer(text)]


def without_a_closing_address(
    segments: Sequence[Segment],
) -> tuple[tuple[Segment, ...], str | None]:
    """The reply without its closing line, if that line cites nothing and holds a web or
    email address; and the line, or ``None`` when it stays.

    The closing line is the last line of the text after the last citation, so that a line
    with a citation, and the text before the last citation, are never taken out. Whether a
    passage holds the address is not asked: a planted address is in the passage it was
    planted in, and that may be the very passage the answer cites.
    """
    last_cited = max(number for number, segment in enumerate(segments) if segment.citations)
    head, tail = tuple(segments[: last_cited + 1]), segments[last_cited + 1 :]
    lines = "".join(segment.text for segment in tail).split("\n")
    closing = next(
        (number for number in reversed(range(len(lines))) if lines[number].strip()), None
    )
    if closing is None:
        return tuple(segments), None
    if not addresses(lines[closing]):
        return tuple(segments), None
    rest = "\n".join(lines[:closing]).rstrip()
    return head + ((Segment(rest),) if rest.strip() else ()), lines[closing].strip()


def copied_share(text: str, chunks: Sequence[Chunk]) -> float:
    """The share of a text's words that stand in a run of :data:`COPY_RUN_WORDS` words or
    more that one passage holds in the same order. Words are compared as the fixed
    sentence of a decline is (``plain_words``), so case, punctuation and Markdown do not
    hide a copy."""
    words = plain_words(text).split()
    if len(words) < COPY_RUN_WORDS:
        return 0.0
    runs: set[tuple[str, ...]] = set()
    for chunk in chunks:
        given = plain_words(chunk.text).split()
        runs.update(
            tuple(given[start : start + COPY_RUN_WORDS])
            for start in range(len(given) - COPY_RUN_WORDS + 1)
        )
    copied = [False] * len(words)
    for start in range(len(words) - COPY_RUN_WORDS + 1):
        if tuple(words[start : start + COPY_RUN_WORDS]) in runs:
            copied[start : start + COPY_RUN_WORDS] = [True] * COPY_RUN_WORDS
    return sum(copied) / len(words)


class HeldBack:
    """Passes a reply's text on once a citation follows it, and the rest at the end.

    A checked reply is changed only after its last citation, or as a whole when it cites
    nothing (:func:`check_reply`). So text that a citation follows is never changed, and
    is passed on as it comes; the rest waits until the checks have been made. What was
    passed on is then always the start of the reply as it is kept.
    """

    def __init__(self, on_text: Callable[[str], None] | None) -> None:
        self._on_text = on_text
        self._sent = ""

    def closed_so_far(self, segments: Sequence[Segment]) -> None:
        """Pass on what the segments closed so far hold. While a reply is being read, only
        a citation closes a segment (``markers.MarkerParser``), so each of them has one."""
        self._send("".join(segment.text for segment in segments))

    def finished(self, segments: Sequence[Segment]) -> None:
        """Pass on the rest of the reply, as the checks left it."""
        self._send("".join(segment.text for segment in segments))

    def _send(self, text: str) -> None:
        # By the rule above, text is always the start of what was sent and more. Were it
        # ever not, sending nothing keeps what the caller has from being contradicted:
        # the whole answer still reaches every caller when it is done.
        if not text.startswith(self._sent) or len(text) == len(self._sent):
            return
        if self._on_text is not None:
            self._on_text(text[len(self._sent) :])
        self._sent = text
