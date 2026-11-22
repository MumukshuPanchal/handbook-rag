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

# Everything the parser has to recognise has a largest size. That is what keeps its work
# per piece of text small, however long the text is or however oddly a model behaves: no
# pattern below can run on over thousands of characters, and only a couple of hundred
# characters are ever held back.

# A marker: a few spaces, then up to ten numbers of up to three digits in square brackets.
_SPACES = r"[ \t]{0,8}"
_NUMBERS = r"\d{1,3}(?:[ \t]{0,2},[ \t]{0,2}\d{1,3}){0,9}"
_LONGEST_NUMBERS = 3 + 9 * (2 + 1 + 2 + 3)
_LONGEST_UNFINISHED = 8 + 1 + _LONGEST_NUMBERS
# Backticks in a row that can still open or close code. A longer run is ordinary text.
_LONGEST_RUN = 16
# A run of this many backticks or more opens a code block when it starts a line.
_FENCE = 3
# What may follow the backticks that open a code block on their line: a word like
# "python", or a longer label. Together with the backticks it stays under the size below.
_LONGEST_INFO = 180
# Code inside a line, from its first backtick to its last.
_LONGEST_CODE = 200
# How a line may begin before the backticks of a code block: indentation, perhaps a list marker.
_LONGEST_INDENT = 24

# What needs a decision in ordinary text: a run of backticks, a line break, or a marker
# with the spaces before it, as in " [2]" or " [1, 3]".
_TOKEN = re.compile(
    rf"(?P<ticks>`+)|(?P<newline>\n)|(?P<marker>{_SPACES}\[(?P<numbers>{_NUMBERS})\])"
)
# The end of a piece of text that may still become a marker once more text arrives:
# spaces, then perhaps an opening bracket with numbers and no closing bracket yet.
_UNFINISHED = re.compile(rf"{_SPACES}(?:\[[\d, \t]{{0,{_LONGEST_NUMBERS}}})?\Z")
_TICKS = re.compile(r"`+")
_INDENT = re.compile(r"[ \t]+")
# The start of a line that a code block may still open on, and one it can open on now.
_MAY_START_BLOCK = re.compile(r"[ \t]*(?:[-*+]|\d{1,3}[.)]?)?[ \t]*")
_STARTS_BLOCK = re.compile(r"[ \t]*(?:(?:[-*+]|\d{1,3}[.)])[ \t]+)?")
# Punctuation that closes the statement before it: in "...leave [1]. Next", the full
# stop belongs to the statement that was just cited, not to the next one.
_CLOSING_PUNCTUATION = ".,;:!?)"
# The start of one of the tags this module wraps passages in, however it is spaced.
_OUR_TAGS = re.compile(r"<(?=[\s/]*(?:search_results?|title|content)\b)", re.IGNORECASE)


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
      part of the answer. Code is recognised in the two forms Markdown writes with
      backticks: text between two equal runs of backticks on one line, and the lines
      between two lines that start with three or more. A backtick with no partner is an
      ordinary character.
    - **It is short**, as described above.

    This follows Markdown closely, not completely. Code marked in another way (an
    indented block, a block fenced with ``~~~``, a block inside a quotation) is read as
    ordinary text, and a line that starts with enough backticks closes a block even when
    more text follows on it.

    The parser cannot know what a piece of text is until it has seen enough of it: a
    piece can end in the middle of a marker, or after a backtick whose partner has not
    arrived. Text like that is held back until the next piece settles it. The result
    does not depend on where the stream is cut.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = chunks
        self._pending = ""  # text received and not yet decided
        self._open = ""  # text of the segment being written
        self._closed: list[Segment] = []
        self._just_cited = False
        self._fence = 0  # length of the backtick run that opened the code block we are in
        # How the current line begins, for as long as a code block could still open on
        # it. ``None`` once the line holds anything else.
        self._line: str | None = ""
        # The text so far ends in backticks that have been dealt with. More backticks
        # straight after them belong to the same run and are ordinary text.
        self._run_settled = False

    def feed(self, piece: str) -> str:
        """Take the next piece of model output. Returns the answer text now known."""
        self._pending += piece
        return self._decide(final=False)

    def finish(self) -> str:
        """Take in what was held back. Call once, when the model has stopped."""
        shown = self._decide(final=True)
        if self._open:
            self._closed.append(Segment(self._open))
            self._open = ""
        return shown

    @property
    def segments(self) -> tuple[Segment, ...]:
        """The answer so far. Complete only after ``finish``."""
        return tuple(self._closed)

    def _decide(self, *, final: bool) -> str:
        """Work through the pending text for as far as it can be told what it is."""
        text = self._pending
        shown: list[str] = []
        position = 0
        while position < len(text):
            step = self._block_step if self._fence else self._prose_step
            end = step(text, position, shown, final=final)
            if end == position:
                break  # not enough text yet to tell what comes next
            position = end
        self._pending = text[position:]
        return "".join(shown)

    def _prose_step(self, text: str, position: int, shown: list[str], *, final: bool) -> int:
        """Decide the next stretch of ordinary text. Returns where it ends."""
        token = _TOKEN.search(text, position)
        if token is None:
            cut = len(text)
            if not final:
                # The very end may still become a marker, so it waits.
                tail = _UNFINISHED.search(text, max(position, cut - _LONGEST_UNFINISHED))
                cut = cut if tail is None else tail.start()
            if cut > position:
                self._show(text[position:cut], shown)
            return cut
        if token.start() > position:
            self._show(text[position : token.start()], shown)
            return token.start()
        if token.group("ticks") is not None:
            return self._backticks_step(text, token.start(), token.end(), shown, final=final)
        if token.group("newline") is not None:
            self._show(token.group(), shown)
            return token.end()
        cited = self._passages(token.group("numbers"))
        if cited:
            # A citation is taken out of the text, so it leaves no trace: the line can
            # still open a code block as if the marker had not been there, and backticks
            # after it still continue a run already found too long to mark code.
            self._cite(cited)
        else:
            self._show(token.group(), shown)  # not a citation: it stays as text
        return token.end()

    def _backticks_step(
        self, text: str, start: int, end: int, shown: list[str], *, final: bool
    ) -> int:
        """Decide what a run of backticks in ordinary text is: it opens a code block,
        it begins code inside the line, or it is an ordinary character."""
        run = end - start
        if self._run_settled or run > _LONGEST_RUN:
            self._show(text[start:end], shown)
            self._run_settled = True
            return end

        if run >= _FENCE and self._line is not None and _STARTS_BLOCK.fullmatch(self._line):
            after = text[end : end + _LONGEST_INFO + 1]
            info, line_ended, _ = after.partition("\n")
            if "`" not in info and len(info) <= _LONGEST_INFO:
                if not line_ended and not final:
                    return start  # the rest of the line decides, and it is not here yet
                self._show(text[start:end], shown)
                self._fence = run
                return end

        # Code inside the line ends at the next run of the same length, on this line and
        # within reach. Without one, the backticks are ordinary characters.
        reach = start + _LONGEST_CODE
        line_end = text.find("\n", end, reach)
        for other in _TICKS.finditer(text, end, reach if line_end == -1 else line_end):
            whole = _TICKS.match(text, other.start())
            assert whole is not None
            if whole.end() > reach:
                break
            if whole.end() == len(text) and not final:
                return start  # this run may go on in the next piece
            if whole.end() - whole.start() == run:
                self._show(text[start : whole.end()], shown)
                return whole.end()
        if line_end == -1 and len(text) < reach and not final:
            return start  # the partner may still arrive
        self._show(text[start:end], shown)
        return end

    def _block_step(self, text: str, position: int, shown: list[str], *, final: bool) -> int:
        """Decide the next stretch of a code block. Returns where it ends."""
        if self._line is not None and not self._line.strip(" \t"):
            # Only indentation so far on this line, so backticks here may close the block.
            indent = _INDENT.match(text, position)
            if indent is not None:
                self._show(indent.group(), shown)
                return indent.end()
            ticks = _TICKS.match(text, position)
            if ticks is not None:
                run = ticks.end() - position
                if run >= self._fence:
                    self._show(ticks.group(), shown)
                    self._fence = 0
                    self._run_settled = True
                    return ticks.end()
                if ticks.end() == len(text) and not final:
                    return position  # the run may still grow long enough to close
                self._show(ticks.group(), shown)
                return ticks.end()
        newline = text.find("\n", position)
        end = len(text) if newline == -1 else newline + 1
        self._show(text[position:end], shown)
        return end

    def _show(self, text: str, shown: list[str]) -> None:
        """Add text to the answer, and note what it means for the line it is on."""
        shown.append(self._write(text))
        self._run_settled = False
        if "\n" in text:
            self._line = ""
            text = text.rsplit("\n", 1)[1]
        if self._line is not None:
            line = self._line + text
            fits = len(line) <= _LONGEST_INDENT and _MAY_START_BLOCK.fullmatch(line)
            self._line = line if fits else None

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
        """Add answer text to the segments. Returns what was added."""
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
