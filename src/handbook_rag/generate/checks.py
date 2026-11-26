"""Checks of a reply in code, made before anyone is given it (issue 66).

The adversarial tests found what a planted instruction leaves in a reply when it gets
through: a closing line with an address that came from the planted text, a reply that is
nothing but a made-up word and cites nothing, and a copy of the search results. Each
check looks for one of those shapes, in words any reply can be read in, and does one
thing when it finds it. None of them looks at what a made-up word or address of the test
cases looks like: the address of a planted line is found as an address, whatever its
host.

- **An address after the last citation** (``address_in_closing_line``): a line of the
  reply that stands after its last citation, and so cites nothing, holds a web or email
  address. *Each such line is dropped.* The answer above is kept: in the baseline run, each
  planted line of this kind came under an answer that was right. Every such line goes, not
  the last one only, so that a line after it ("Hope this helps!") does not let it through.

  Issue 66 words this check more narrowly: an address *found in no cited passage*. That
  rule leaves a planted line in whenever the planted text sits in the passage the answer
  cites, since the planted address is then in a cited passage; so it would have left in
  one of the three planted lines that got through in the baseline run
  (``inj-append_link-b08``). The check was widened before any reply of the hardened
  assistant was seen. It costs no answer that can be measured: none of the 129 answers of
  the first answer run has a line without a citation after its last one that holds an
  address. What it still lets through: an address inside a sentence that carries a
  citation, an address anywhere before the last citation, a host written without a scheme
  or a path, a host and path without a scheme written with a capital letter, a disguised
  address (``hxxps://``, ``evil[.]example``), the piece of an address split across lines
  that holds no address of its own, and a whole reply replaced by planted text with a
  made-up citation.
- **A reply that cites nothing at all** (``cites_nothing``): an answer with no citation.
  *The reply is declined.* Nothing in it rests on a passage a reader could check. It is
  the shape of the reply that obeyed a forged message from the system, and it costs an
  answer whose model left out its markers: one of the 129 answers of the first answer run.
- **A reply that is mostly a copy of the passages** (``mostly_a_copy``): at least
  :data:`COPY_MIN_WORDS` words, of which at least :data:`COPY_SHARE` stand in runs of
  :data:`COPY_RUN_WORDS` words or more that a passage holds word for word. *The reply is
  only marked.* What it repeats is handbook text, which the search gives anyway, and a
  long answer that quotes a procedure should not be lost.

A reply that is not an answer is checked too, since a decline can carry planted words
beside it:

- **Words beside a decline** (``words_beside_a_decline``): a reply that came to a normal
  end, is not an answer, and holds more words than the fixed declining sentence. *It is
  shown as the fixed declining sentence alone.* Such a reply cites nothing, so none of it
  was passed on before the end.
- **A reply that did not come to a proper end** (``unfinished_reply``): one the provider
  withheld or that stopped for another reason than its end or its length. *Its words after
  the last citation are dropped*, and what a citation follows, which may already have been
  passed on, is kept.

A check that changes a reply always says so: each check that fires is recorded on the
answer with a note in fixed words, which every surface shows with it. The words the model
wrote and a check took out are kept for an evaluation to read, and never shown. One change
is made before the checks and with no note: a reply read as the fixed form for a missing
part alone (``base.read_the_gap_form_as_a_decline``) is a decline, shown as the fixed
declining sentence when it cites nothing and in its own words, without its markers, when
it carried them; its words are kept in the same way. A reply in that form that also
asserts something, joined into the same sentence, is read the same way, which is a limit
(``base.only_the_gap_form``). A decline lists no source, even when the model put a marker
after it.

So that what was streamed is never taken back, the generator passes text on only once a
citation follows it (:class:`HeldBack`): the text after the last citation, which is all a
dropped line or a declined reply can hold, waits for the end of the reply and the checks.
"""

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from handbook_rag.generate.base import (
    cites_the_decline,
    is_answered,
    only_the_gap_form,
    plain_words,
)
from handbook_rag.generate.models import ReplyCheck, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.ingest.models import Chunk

ADDRESS_IN_CLOSING_LINE: Final = "address_in_closing_line"
CITES_NOTHING: Final = "cites_nothing"
MOSTLY_A_COPY: Final = "mostly_a_copy"
WORDS_BESIDE_A_DECLINE: Final = "words_beside_a_decline"
UNFINISHED_REPLY: Final = "unfinished_reply"
CHECKS: Final = (
    WORDS_BESIDE_A_DECLINE,
    UNFINISHED_REPLY,
    CITES_NOTHING,
    ADDRESS_IN_CLOSING_LINE,
    MOSTLY_A_COPY,
)
"""The checks: the two for a reply that is not an answer, then the three for an answer, in
the order they are made."""

DROPPED_LINE: Final = "dropped_line"
DROPPED_WORDS: Final = "dropped_words"
DECLINED: Final = "declined"
MARKED: Final = "marked"

NOTES: Final = {
    ADDRESS_IN_CLOSING_LINE: (
        "Text after the answer's last citation was left out: it cited no passage and held a "
        "web or email address."
    ),
    CITES_NOTHING: (
        "The reply the model wrote cited none of the passages, so it is not shown: a reply "
        "that rests on no passage cannot be checked against the handbook."
    ),
    MOSTLY_A_COPY: "Most of this reply repeats the passages it was given, word for word.",
    WORDS_BESIDE_A_DECLINE: (
        "The reply was not an answer, and the words the model wrote beside the declining "
        "sentence are not shown."
    ),
    UNFINISHED_REPLY: (
        "The reply did not come to a proper end, and its words after the last citation are "
        "not shown."
    ),
}
"""What the reader is told when each check fires."""

COPY_RUN_WORDS: Final = 8
"""How many words in a row a passage must hold for them to count as copied."""
COPY_SHARE: Final = 0.75
"""The share of a reply's words that must be copied for the reply to be marked."""
COPY_MIN_WORDS: Final = 200
"""How long a reply must be to be marked. The instructions ask for a short paragraph or a
few bullet points, and a short answer that quotes a list of steps is an answer."""

# A web address with its scheme or "www.", in any case; an email address, in any case; or
# a host with a path and no scheme, in lower case. A host alone, such as "GitLab.com", is
# not taken for an address: product names are written that way in ordinary answers. Nor
# is a host and path with a capital letter, such as "Node.js/React": names of tools are
# written that way, and an address without a scheme is almost always written in lower
# case. Such an address with a capital letter is missed.
_ADDRESS = re.compile(
    r"(?i:(?:https?://|www\.)[^\s<>()\[\]{}\"'`]+)"
    r"|(?i:[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+)"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}/[^\s<>()\[\]{}\"'`]*"
)


@dataclass(frozen=True, slots=True)
class CheckedReply:
    """A reply as the checks left it."""

    segments: tuple[Segment, ...]
    answered: bool
    checks: tuple[ReplyCheck, ...]


def check_reply(
    segments: Sequence[Segment], chunks: Sequence[Chunk], *, answered: bool, ended: bool = True
) -> CheckedReply:
    """Make every check on one reply, in the order of :data:`CHECKS`.

    ``ended`` says whether the reply came to a proper end: its model finished it, or ran
    out of room. A reply that is not an answer goes to the checks for one; an answer that
    cites nothing is declined and checked no further.
    """
    replied = tuple(segments)
    if not answered:
        return _not_an_answer(replied, ended=ended)
    checked = _an_answer(replied, chunks)
    left = "".join(segment.text for segment in checked.segments)
    if checked.answered and (
        not is_answered(checked.segments, refused=False) or only_the_gap_form(left)
    ):
        # What the checks left says no more than that the handbook lacks the answer, as
        # "[1] I could not find this in the handbook." does once a line after it is gone,
        # or as a cited sentence in the fixed form does once an address after it is gone.
        declined = _not_an_answer(checked.segments, ended=True)
        return CheckedReply(declined.segments, False, (*checked.checks, *declined.checks))
    return checked


def _an_answer(replied: tuple[Segment, ...], chunks: Sequence[Chunk]) -> CheckedReply:
    """The checks for an answer: one that cites nothing is declined and checked no further."""
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


def _not_an_answer(replied: tuple[Segment, ...], *, ended: bool) -> CheckedReply:
    """A reply that is not an answer, with nothing in it beside what a decline says.

    After a normal end, a reply that is not an answer is the fixed declining sentence,
    whatever its case and marks; or nothing but the fixed form for a missing part, with
    citation markers (``base.read_the_gap_form_as_a_decline``); or cites nothing; or cites
    only after the fixed sentence (``base.is_answered``). The first two are left as they
    are, with no source. The others, if they hold other words, are shown as the fixed
    sentence: none of their text was passed on,
    since :class:`HeldBack` holds back a citation that follows the fixed sentence. A reply
    that did not come to a proper end keeps what a citation follows, which may have been
    passed on, and loses the rest.
    """
    text = "".join(segment.text for segment in replied)
    if ended:
        said = plain_words(text)
        if not said or said == plain_words(NOT_FOUND_MESSAGE) or only_the_gap_form(text):
            # The words are left as they are, and a marker after them names no source. The
            # fixed form alone says what the declining sentence says, and its words may
            # have been passed on (``base.read_the_gap_form_as_a_decline``).
            return CheckedReply(tuple(Segment(segment.text) for segment in replied), False, ())
        check = ReplyCheck(
            WORDS_BESIDE_A_DECLINE, DROPPED_WORDS, NOTES[WORDS_BESIDE_A_DECLINE], text.strip()
        )
        return CheckedReply((Segment(NOT_FOUND_MESSAGE),), False, (check,))
    cited = [number for number, segment in enumerate(replied) if segment.citations]
    head = replied[: cited[-1] + 1] if cited else ()
    rest = "".join(segment.text for segment in replied[len(head) :])
    if not plain_words(rest):
        return CheckedReply(replied, False, ())
    check = ReplyCheck(UNFINISHED_REPLY, DROPPED_WORDS, NOTES[UNFINISHED_REPLY], rest.strip())
    return CheckedReply(tuple(head), False, (check,))


def addresses(text: str) -> list[str]:
    """The web and email addresses in a text, as they are written there."""
    return [match.group() for match in _ADDRESS.finditer(text)]


def without_a_closing_address(
    segments: Sequence[Segment],
) -> tuple[tuple[Segment, ...], str | None]:
    """The reply without each line after its last citation that holds a web or email
    address; and those lines, one to a line, or ``None`` when none goes.

    Only text after the last citation is read, so that a line with a citation, and the
    text before the last citation, are never taken out. Whether a passage holds the
    address is not asked: a planted address is in the passage it was planted in, and that
    may be the very passage the answer cites.
    """
    last_cited = max(number for number, segment in enumerate(segments) if segment.citations)
    head, tail = tuple(segments[: last_cited + 1]), segments[last_cited + 1 :]
    lines = "".join(segment.text for segment in tail).split("\n")
    dropped = [line for line in lines if addresses(line)]
    if not dropped:
        return tuple(segments), None
    rest = "\n".join(line for line in lines if not addresses(line)).rstrip()
    kept = head + ((Segment(rest),) if rest.strip() else ())
    return kept, "\n".join(line.strip() for line in dropped)


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

    A checked reply is changed only after its last citation; or as a whole when it cites
    nothing, or cites only after the fixed declining sentence (:func:`check_reply`,
    ``base.read_the_gap_form_as_a_decline``). A reply that is nothing but the fixed form
    for a missing part loses its citations and keeps its words. So text that a citation
    follows is never changed, and is passed on as it comes, unless that citation follows
    the fixed sentence; the rest waits until the checks have been made. What was passed on
    is then always the start of the reply as it is kept.
    """

    def __init__(self, on_text: Callable[[str], None] | None) -> None:
        self._on_text = on_text
        self._sent = ""
        self.first_sent_at: float | None = None
        """When text was first passed on, by ``time.perf_counter``, also when nobody listens."""

    def closed_so_far(self, segments: Sequence[Segment]) -> None:
        """Pass on what the segments closed so far hold, up to the first whose citation
        follows the fixed declining sentence (``base.cites_the_decline``): a reply whose
        every citation is of that kind is a decline, shown as the fixed sentence alone, so
        it and all after it wait for the end. While a reply is being read, only a citation
        closes a segment (``markers.MarkerParser``), so each of them has one."""
        sendable: list[Segment] = []
        for segment in segments:
            if cites_the_decline(segment):
                break
            sendable.append(segment)
        self._send("".join(segment.text for segment in sendable))

    def finished(self, segments: Sequence[Segment]) -> None:
        """Pass on the rest of the reply, as the checks left it."""
        self._send("".join(segment.text for segment in segments))

    def _send(self, text: str) -> None:
        # By the rule above, text is always the start of what was sent and more. Were it
        # ever not, sending nothing keeps what the caller has from being contradicted:
        # the whole answer still reaches every caller when it is done.
        if not text.startswith(self._sent) or len(text) == len(self._sent):
            return
        if self.first_sent_at is None:
            self.first_sent_at = time.perf_counter()
        if self._on_text is not None:
            self._on_text(text[len(self._sent) :])
        self._sent = text
