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

# A marker, with the spaces before it: " [2]" or " [1, 3]".
_MARKER = re.compile(r"[ \t]*\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
# The end of a piece of text that may still become a marker once more text arrives:
# spaces, then perhaps an opening bracket with numbers and no closing bracket yet.
_UNFINISHED = re.compile(r"[ \t]*(?:\[[\d,\s]*)?\Z")
# Punctuation that closes the statement before it: in "...leave [1]. Next", the full
# stop belongs to the statement that was just cited, not to the next one.
_CLOSING_PUNCTUATION = ".,;:!?)"
# The tags this module wraps passages in.
_OUR_TAGS = re.compile(r"<(/?(?:search_results?|title|content)\b)", re.IGNORECASE)


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
    return _OUR_TAGS.sub(r"&lt;\1", text)


class MarkerParser:
    """Separates answer text from citation markers, as the text arrives in pieces.

    Feed it the model's output piece by piece. Each call returns the answer text that is
    now certain, with the markers taken out, ready to show to a reader. When the model
    has stopped, ``finish`` returns what was still held back, and ``segments`` holds the
    answer as runs of text with the passages that support each run.

    A piece can end in the middle of a marker, so text that could still become one is
    held back until the next piece settles it.

    A marker counts as a citation only when it names a passage that was sent. Brackets
    around any other number are left in the text exactly as the model wrote them, so a
    citation the model made up stays visible instead of being silently removed.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = chunks
        self._held = ""  # text that may still turn into a marker
        self._open = ""  # text of the segment being written
        self._closed: list[Segment] = []
        self._just_cited = False

    def feed(self, piece: str) -> str:
        """Take the next piece of model output. Returns the answer text now known."""
        buffer = self._held + piece
        tail = _UNFINISHED.search(buffer)
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
        for match in _MARKER.finditer(text):
            cited = self._passages(match.group(1))
            if not cited:
                continue  # names no passage we sent: not a citation, so it stays as text
            shown.append(self._write(text[position : match.start()]))
            self._cite(cited)
            position = match.end()
        shown.append(self._write(text[position:]))
        return "".join(shown)

    def _passages(self, numbers: str) -> list[int]:
        """The numbers in a marker that name a passage we sent, each one once."""
        wanted: list[int] = []
        for number in map(int, re.findall(r"\d+", numbers)):
            if 1 <= number <= len(self._chunks) and number not in wanted:
                wanted.append(number)
        return wanted

    def _write(self, text: str) -> str:
        """Add answer text. Returns what was added."""
        if not text:
            return ""
        if self._just_cited and not self._open:
            rest = text.lstrip(_CLOSING_PUNCTUATION)
            closing = text[: len(text) - len(rest)]
            if closing:
                last = self._closed[-1]
                self._closed[-1] = Segment(last.text + closing, last.citations)
            self._open = rest
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
            self._closed[-1] = Segment(last.text, last.citations + new)
        self._just_cited = True
