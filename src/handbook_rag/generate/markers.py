"""Citations by numbered marker, for a model whose API has no citation feature.

The passages are numbered in the prompt. The model is asked to put the number of each
passage that supports a statement right after it, such as ``[2]``. This module writes
that prompt and turns the markers in the reply back into citations.

This is the portable method: it works with any model. What it cannot give is what a
built-in citation feature gives (ADR 0006): the exact words the model relied on. Here a
citation names a whole passage, and code can check only that the number belongs to a
passage the model was really given (ADR 0013).
"""

import re
from collections.abc import Sequence

from handbook_rag.generate.models import Citation, Segment
from handbook_rag.generate.prompt import SYSTEM_PROMPT
from handbook_rag.ingest.models import Chunk

MARKER_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """\
- Each search result has a number. After every statement, put the number of each search \
result that supports it in square brackets, like [2] or [1][3]. Use only the numbers of \
search results you were given, and do not add a list of sources at the end.
"""
)

# A marker is short by definition: a few spaces, then up to ten numbers of up to three
# digits in square brackets. The limits are what keeps the parser's work per piece of
# text small, however long the text is or however oddly a model behaves: no pattern
# below can run on over thousands of characters.
_SPACES = r"[ \t]{0,8}"
_NUMBERS = r"\d{1,3}(?:[ \t]{0,2},[ \t]{0,2}\d{1,3}){0,9}"
_LONGEST_NUMBERS = 3 + 9 * (2 + 1 + 2 + 3)
# What needs a decision in a piece of model output: a run of backticks, a line break, or
# a marker with the spaces before it, as in " [2]" or " [1, 3]".
_TOKEN = re.compile(
    rf"(?P<ticks>`+)|(?P<newline>\n)|(?P<marker>{_SPACES}\[(?P<numbers>{_NUMBERS})\])"
)
# The end of a piece of text that may still become a marker once more text arrives:
# spaces, then perhaps an opening bracket with numbers and no closing bracket yet.
_UNFINISHED = re.compile(rf"{_SPACES}(?:\[[\d, \t]{{0,{_LONGEST_NUMBERS}}})?\Z")
_LONGEST_UNFINISHED = 8 + 1 + _LONGEST_NUMBERS
# Punctuation that closes the statement before it: in "...leave [1]. Next", the full
# stop belongs to the statement that was just cited, not to the next one.
_CLOSING_PUNCTUATION = ".,;:!?)"
# A backtick run this long or longer opens a code block, which runs over several lines.
# A shorter run opens code inside a line.
_FENCE = 3
# The start of one of the tags this module wraps passages in, however it is spaced.
_OUR_TAGS = re.compile(r"<(?=\s*/?\s*(?:search_results?|title|content)\b)", re.IGNORECASE)


def build_numbered_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    """Build the user message: the passages, numbered from 1, then the question.

    Each passage sits in its own tagged block, apart from the question and from our
    instructions, which keeps a clear line between what we tell the model to do and the
    text it should only read.
    """
    results = [
        f'<search_result number="{number}">\n'
        f"<title>{_inert(chunk.context_header)}</title>\n"
        f"<content>\n{_inert(chunk.text)}\n</content>\n"
        "</search_result>"
        for number, chunk in enumerate(chunks, start=1)
    ]
    return (
        "<search_results>\n" + "\n".join(results) + f"\n</search_results>\n\nQuestion: {question}"
    )


def _inert(text: str) -> str:
    """Stop handbook text from opening or closing one of our own tags.

    Handbook pages are written by many people, so a passage is data we do not control. A
    passage holding ``</search_result>`` could otherwise end its own block early and make
    the text after it look like something other than a passage.
    """
    return _OUR_TAGS.sub("&lt;", text)


class MarkerParser:
    """Separates answer text from citation markers, as the text arrives in pieces.

    Feed it the model's output piece by piece. Each call returns the answer text that is
    now certain, with the markers taken out, ready to show to a reader. When the model
    has stopped, ``finish`` returns what was still held back, and ``segments`` holds the
    answer as runs of text with the passages that support each run.

    Three rules decide what a marker is:

    - **Every number in it names a passage that was sent.** Brackets around any other
      number are left in the text exactly as the model wrote them, so a citation the
      model made up stays visible instead of being silently removed.
    - **It is not inside code.** Between backticks, ``argv[1]`` and ``[1, 2, 3]`` are
      part of the answer.
    - **It is short**, as described above.

    A piece can end in the middle of a marker, so text that could still become one is
    held back until the next piece settles it. The result does not depend on where the
    stream is cut.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = chunks
        self._held = ""  # text that may still turn into a marker
        self._open = ""  # text of the segment being written
        self._closed: list[Segment] = []
        self._just_cited = False
        self._ticks = 0  # length of the run of backticks the text read so far ends with
        self._code = 0  # length of the backtick run that opened the code we are in, or 0

    def feed(self, piece: str) -> str:
        """Take the next piece of model output. Returns the answer text now known."""
        buffer = self._held + piece
        tail = _UNFINISHED.search(buffer, max(0, len(buffer) - _LONGEST_UNFINISHED))
        cut = len(buffer) if tail is None else tail.start()
        self._held = buffer[cut:]
        return self._read(buffer[:cut])

    def finish(self) -> str:
        """Take in what was held back. Call once, when the model has stopped."""
        shown = self._read(self._held)
        self._held = ""
        if self._open:
            self._closed.append(Segment(self._open))
            self._open = ""
        return shown

    @property
    def segments(self) -> tuple[Segment, ...]:
        """The answer so far. Complete only after ``finish``."""
        return tuple(self._closed)

    def _read(self, text: str) -> str:
        """Take in text in which every marker is whole. Returns it without its markers."""
        shown: list[str] = []
        position = 0
        for token in _TOKEN.finditer(text):
            before = text[position : token.start()]
            if before:
                self._end_backtick_run()
                shown.append(self._write(before))
            position = token.end()
            if token.group("ticks") is not None:
                # Backticks are ordinary text. What they mean depends on how long the
                # run is, and it may go on in the next piece, so it is only counted here.
                self._ticks += len(token.group())
                shown.append(self._write(token.group()))
                continue
            self._end_backtick_run()
            if token.group("newline") is not None:
                if 0 < self._code < _FENCE:
                    # Code inside a line ends with the line. Without this, one stray
                    # backtick would hide every marker in the rest of the answer.
                    self._code = 0
                shown.append(self._write(token.group()))
                continue
            cited = [] if self._code else self._passages(token.group("numbers"))
            if cited:
                self._cite(cited)
            else:
                shown.append(self._write(token.group()))  # not a citation: it stays as text
        rest = text[position:]
        if rest:
            self._end_backtick_run()
            shown.append(self._write(rest))
        return "".join(shown)

    def _end_backtick_run(self) -> None:
        """Other text followed a run of backticks: the run opens code, closes it, or neither."""
        run, self._ticks = self._ticks, 0
        if run == 0:
            return
        if self._code == 0:
            self._code = run
        elif self._code == run:
            self._code = 0
        # A run of another length inside code is part of the code.

    def _passages(self, numbers: str) -> list[int]:
        """The passages a marker names, each once. None at all unless every number names one."""
        wanted: list[int] = []
        for number in map(int, re.findall(r"\d+", numbers)):
            if not 1 <= number <= len(self._chunks):
                return []
            if number not in wanted:
                wanted.append(number)
        return wanted

    def _write(self, text: str) -> str:
        """Add answer text. Returns what was added."""
        if self._just_cited and not self._open:
            rest = text.lstrip(_CLOSING_PUNCTUATION)
            closing = text[: len(text) - len(rest)]
            if closing:
                last = self._closed[-1]
                self._closed[-1] = Segment(last.text + closing, last.citations)
            self._open = rest
            # Closing punctuation may itself arrive in two pieces, as in "[1])" then ".".
            self._just_cited = not rest
        else:
            self._open += text
            self._just_cited = False
        return text

    def _cite(self, numbers: Sequence[int]) -> None:
        """Attach passages to the statement that was just written."""
        citations = tuple(Citation(self._chunks[number - 1]) for number in numbers)
        if self._open.strip() or not self._closed:
            self._closed.append(Segment(self._open, citations))
            self._open = ""
        else:
            # Another marker for the statement before, as in "[1][3]" or "[1]. [3]".
            last = self._closed[-1]
            known = {citation.chunk.chunk_id for citation in last.citations}
            new = tuple(c for c in citations if c.chunk.chunk_id not in known)
            self._closed[-1] = Segment(last.text + self._open, last.citations + new)
            self._open = ""
        self._just_cited = True
