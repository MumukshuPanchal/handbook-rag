"""Split a cleaned page into chunks that follow the page's own structure.

How it works, in order:

1. A Markdown parser finds the page's blocks: headings, paragraphs, lists, tables and
   code. Working with whole blocks means a chunk never starts in the middle of a table
   row or a code block unless that single block is too large to keep whole.
2. Blocks are grouped into sections. A section is a heading and the content below it,
   up to the next heading.
3. Sections are packed into chunks up to a token budget. A level-1 or level-2 heading
   starts a new chunk, so a chunk does not mix two major topics. Very small pieces are
   merged forward instead, so no chunk is only a heading or a single line.
4. A section larger than the budget is cut at block boundaries, or at line boundaries
   inside a single block that is itself too large. Each piece starts by repeating the
   last units of the piece before it, as many as fit in the overlap budget, so text at
   a cut keeps some of its context. Whole units are repeated, never half a sentence.

Every chunk records the headings above it and its character offsets in the page.
"""

import hashlib
import itertools
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from markdown_it import MarkdownIt

from handbook_rag.ingest.models import Chunk, Document

TokenCounter = Callable[[Sequence[str]], list[int]]
"""Counts tokens for several texts at once, as the embedding model would see them."""

_MARKDOWN = MarkdownIt("commonmark").enable("table")
_BLOCK_KINDS = {
    "paragraph_open": "paragraph",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "table_open": "table",
    "fence": "code",
    "code_block": "code",
}
_INLINE_MARKUP = re.compile(r"[*_`]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
# A heading at this level or above starts a new chunk.
_TOPIC_HEADING_LEVEL = 2
# A chunk may exceed the budget by this factor when that avoids leaving a tiny fragment.
_TAIL_MERGE_ALLOWANCE = 1.25
_ID_NAMESPACE = uuid.UUID("6f1d0c4e-5a3b-4f0e-9c2a-7b8d1e2f3a4b")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """How pages are split. Changing any value changes chunk identifiers."""

    max_tokens: int = 320
    min_tokens: int = 60
    overlap_tokens: int = 40

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens < self.max_tokens:
            raise ValueError("min_tokens must be positive and smaller than max_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")

    @property
    def version(self) -> str:
        return f"v1-max{self.max_tokens}-min{self.min_tokens}-overlap{self.overlap_tokens}"


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str
    start: int
    end: int
    level: int = 0
    title: str = ""


@dataclass(frozen=True, slots=True)
class _Atom:
    """The smallest unit packed into a chunk: a whole block, or part of an oversized one."""

    start: int
    end: int
    tokens: int
    table_header: str = ""
    table_header_tokens: int = 0


@dataclass(slots=True)
class _Section:
    level: int
    path: tuple[str, ...]
    blocks: list[_Block]
    tokens: int = 0


@dataclass(frozen=True, slots=True)
class _Piece:
    """The atoms that will become one chunk, and the headings above them."""

    atoms: list[_Atom]
    path: tuple[str, ...]

    @property
    def tokens(self) -> int:
        return _piece_tokens(self.atoms)


def chunk_document(
    document: Document, *, count_tokens: TokenCounter, config: ChunkingConfig
) -> list[Chunk]:
    """Split one document into chunks. The same input always gives the same chunks."""
    text = document.text
    blocks = _parse_blocks(text)
    if not blocks:
        return []
    block_tokens = dict(
        zip(blocks, count_tokens([text[b.start : b.end] for b in blocks]), strict=True)
    )
    sections = _group_into_sections(blocks, block_tokens)

    pieces: list[_Piece] = []
    pending: list[_Section] = []

    def pending_tokens() -> int:
        return sum(s.tokens for s in pending)

    def flush() -> None:
        pending_blocks = [b for s in pending for b in s.blocks]
        # A heading with nothing under it is not worth a chunk of its own.
        if any(b.kind != "heading" for b in pending_blocks):
            atoms = [_Atom(b.start, b.end, block_tokens[b]) for b in pending_blocks]
            _add_piece(pieces, _Piece(atoms, _common_path([s.path for s in pending])), config)
        pending.clear()

    for section in sections:
        if section.tokens > config.max_tokens:
            # Keep a small leftover, such as a parent heading, attached to what follows.
            carried: list[_Block] = []
            if pending_tokens() < config.min_tokens:
                carried = [b for s in pending for b in s.blocks]
                pending.clear()
            flush()
            atoms = [
                atom
                for block in carried + section.blocks
                for atom in _atoms_for(block, block_tokens[block], text, count_tokens, config)
            ]
            for packed in _pack_with_overlap(atoms, config):
                _add_piece(pieces, _Piece(packed, section.path), config)
            continue
        starts_new_topic = 0 < section.level <= _TOPIC_HEADING_LEVEL
        if pending and (
            pending_tokens() + section.tokens > config.max_tokens
            or (starts_new_topic and pending_tokens() >= config.min_tokens)
        ):
            flush()
        pending.append(section)
    flush()

    drafts = [(*_render(piece.atoms, text), piece.path) for piece in pieces]
    token_counts = count_tokens([draft[2] for draft in drafts])
    return [
        Chunk(
            chunk_id=_chunk_id(document.doc_id, start, end, config.version, chunk_text),
            doc_id=document.doc_id,
            title=document.title,
            url=document.url,
            section=document.section,
            source_commit=document.source_commit,
            heading_path=path,
            start_char=start,
            end_char=end,
            text=chunk_text,
            token_count=tokens,
        )
        for (start, end, chunk_text, path), tokens in zip(drafts, token_counts, strict=True)
    ]


def _piece_tokens(atoms: Sequence[_Atom]) -> int:
    """Size of a piece: its atoms, plus the table header it will be prefixed with."""
    if not atoms:
        return 0
    return sum(atom.tokens for atom in atoms) + atoms[0].table_header_tokens


def _add_piece(pieces: list[_Piece], piece: _Piece, config: ChunkingConfig) -> None:
    """Add a piece, folding a very small one into the piece before it.

    A fragment of a few tokens is useless to retrieve by itself. Joining it to its
    neighbour may exceed the budget slightly, which is allowed up to a fixed margin.
    """
    if pieces and piece.tokens < config.min_tokens:
        previous = pieces[-1]
        new_atoms = [atom for atom in piece.atoms if atom not in previous.atoms]
        merged_tokens = previous.tokens + sum(atom.tokens for atom in new_atoms)
        if merged_tokens <= config.max_tokens * _TAIL_MERGE_ALLOWANCE:
            path = _common_path([previous.path, piece.path])
            pieces[-1] = _Piece(previous.atoms + new_atoms, path)
            return
    pieces.append(piece)


def _parse_blocks(text: str) -> list[_Block]:
    """Find the top-level blocks of a page with their character offsets."""
    line_starts = [0]
    for match in re.finditer("\n", text):
        line_starts.append(match.end())
    line_starts.append(len(text))

    tokens = _MARKDOWN.parse(text)
    blocks: list[_Block] = []
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None or token.nesting == -1:
            continue
        start = line_starts[token.map[0]]
        end = start + len(
            text[start : line_starts[min(token.map[1], len(line_starts) - 1)]].rstrip()
        )
        if end <= start:
            continue
        if token.type == "heading_open":
            title = _INLINE_MARKUP.sub("", tokens[index + 1].content).strip()
            blocks.append(_Block("heading", start, end, level=int(token.tag[1]), title=title))
        else:
            blocks.append(_Block(_BLOCK_KINDS.get(token.type, "other"), start, end))
    return blocks


def _group_into_sections(blocks: list[_Block], block_tokens: dict[_Block, int]) -> list[_Section]:
    """Group blocks under the heading they belong to, tracking the path of headings above."""
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind == "heading":
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.title))
            sections.append(_Section(block.level, tuple(t for _, t in heading_stack), [block]))
        elif sections:
            sections[-1].blocks.append(block)
        else:
            sections.append(_Section(0, (), [block]))
    for section in sections:
        section.tokens = sum(block_tokens[b] for b in section.blocks)
    return sections


def _common_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    """The headings shared by every section in a chunk."""
    common = paths[0]
    for path in paths[1:]:
        length = 0
        for ours, theirs in zip(common, path, strict=False):
            if ours != theirs:
                break
            length += 1
        common = common[:length]
    return common


def _atoms_for(
    block: _Block, tokens: int, text: str, count_tokens: TokenCounter, config: ChunkingConfig
) -> list[_Atom]:
    """Return the block as one atom, or cut it into line-sized atoms if it is too large."""
    if tokens <= config.max_tokens:
        return [_Atom(block.start, block.end, tokens)]

    spans = _line_spans(text, block.start, block.end)
    header, header_tokens = "", 0
    if block.kind == "table" and len(spans) > 2:
        candidate = text[spans[0][0] : spans[1][1]]
        (candidate_tokens,) = count_tokens([candidate])
        # Repeating the header is only worth it if it leaves most of the budget for rows.
        if candidate_tokens <= config.max_tokens // 3:
            header, header_tokens = candidate, candidate_tokens
    budget = config.max_tokens - header_tokens

    atoms: list[_Atom] = []
    for line_number, (start, end) in enumerate(spans):
        (line_tokens,) = count_tokens([text[start:end]])
        if line_tokens <= budget:
            parts = [(start, end, line_tokens)]
        else:
            parts = _split_long_line(text, start, end, count_tokens, budget)
        if header and line_number < 2:
            atoms.extend(_Atom(s, e, t) for s, e, t in parts)
        else:
            atoms.extend(_Atom(s, e, t, header, header_tokens) for s, e, t in parts)
    return atoms


def _line_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Offsets of each non-empty line between ``start`` and ``end``."""
    spans: list[tuple[int, int]] = []
    position = start
    for line in text[start:end].split("\n"):
        if line.strip():
            spans.append((position, position + len(line)))
        position += len(line) + 1
    return spans


def _split_long_line(
    text: str, start: int, end: int, count_tokens: TokenCounter, max_tokens: int
) -> list[tuple[int, int, int]]:
    """Cut one over-long line into parts within budget.

    Tries sentence ends first, then spaces. Text with no spaces at all, such as a long
    encoded string, is cut into fixed-size runs of characters as a last resort.
    """
    units = [(start, end)]
    for separator in (_SENTENCE_END, _WHITESPACE):
        counts = count_tokens([text[a:b] for a, b in units])
        refined: list[tuple[int, int]] = []
        for (a, b), unit_tokens in zip(units, counts, strict=True):
            if unit_tokens <= max_tokens:
                refined.append((a, b))
                continue
            cuts = [a, *(a + match.end() for match in separator.finditer(text[a:b])), b]
            refined.extend((x, y) for x, y in itertools.pairwise(cuts) if x < y)
        units = refined

    sized: list[tuple[int, int, int]] = []
    for (a, b), unit_tokens in zip(units, count_tokens([text[a:b] for a, b in units]), strict=True):
        if unit_tokens <= max_tokens:
            sized.append((a, b, unit_tokens))
        else:
            sized.extend(_split_by_characters(text, a, b, unit_tokens, count_tokens, max_tokens))

    parts: list[tuple[int, int, int]] = []
    for a, b, unit_tokens in sized:
        if parts and parts[-1][2] + unit_tokens <= max_tokens:
            parts[-1] = (parts[-1][0], b, parts[-1][2] + unit_tokens)
        else:
            parts.append((a, b, unit_tokens))
    return [(a, a + len(text[a:b].rstrip()), t) for a, b, t in parts if text[a:b].strip()]


def _split_by_characters(
    text: str, start: int, end: int, tokens: int, count_tokens: TokenCounter, max_tokens: int
) -> list[tuple[int, int, int]]:
    """Cut text that has no natural break into equal runs of characters within budget."""
    window = max(1, int((end - start) * max_tokens / tokens * 0.9))
    while True:
        spans = [(a, min(a + window, end)) for a in range(start, end, window)]
        counts = count_tokens([text[a:b] for a, b in spans])
        if max(counts) <= max_tokens or window == 1:
            return [(a, b, t) for (a, b), t in zip(spans, counts, strict=True)]
        window = max(1, window // 2)


def _pack_with_overlap(atoms: list[_Atom], config: ChunkingConfig) -> list[list[_Atom]]:
    """Pack atoms into pieces within budget, repeating a little of each piece in the next."""
    pieces: list[list[_Atom]] = []
    current: list[_Atom] = []
    for atom in atoms:
        if current and _piece_tokens(current) + atom.tokens > config.max_tokens:
            pieces.append(current)
            overlap: list[_Atom] = []
            for previous in reversed(current):
                if sum(a.tokens for a in overlap) + previous.tokens > config.overlap_tokens:
                    break
                overlap.insert(0, previous)
            if _piece_tokens([*overlap, atom]) > config.max_tokens:
                overlap = []
            current = overlap
        current.append(atom)
    if current:
        pieces.append(current)
    return pieces


def _render(atoms: list[_Atom], text: str) -> tuple[int, int, str]:
    """Turn a piece into its offsets and text."""
    start, end = atoms[0].start, atoms[-1].end
    body = text[start:end]
    header = atoms[0].table_header
    return start, end, f"{header}\n{body}" if header else body


def _chunk_id(doc_id: str, start: int, end: int, version: str, text: str) -> str:
    """A stable identifier that changes whenever the chunk's position or content does.

    Including the text matters for indexing: a chunk is skipped when its id is already
    stored, so two different texts must never share an id.
    """
    content = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}|{start}|{end}|{version}|{content}"))
