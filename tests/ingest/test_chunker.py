"""Chunker tests use a fake token counter: one whitespace-separated word is one token.

That keeps the numbers in each test easy to check by eye, and means the tests need no
model download. The budget used here is tiny for the same reason.
"""

import itertools
from collections.abc import Sequence

import pytest

from handbook_rag.ingest.chunker import ChunkingConfig, chunk_document, split_into_blocks
from handbook_rag.ingest.models import Chunk, Document
from tests.support import count_words

CONFIG = ChunkingConfig(max_tokens=20, min_tokens=5, overlap_tokens=4)
ALLOWED_MAX = 25  # the budget plus the margin allowed when absorbing a tiny fragment


def words(count: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(count)) + "."


def make_document(text: str) -> Document:
    return Document(
        doc_id="people/leave.md",
        title="Leave",
        url="https://handbook.example.com/handbook/people/leave/",
        section="people",
        source_commit="a" * 40,
        text=text,
    )


def chunk(text: str, config: ChunkingConfig = CONFIG) -> list[Chunk]:
    return chunk_document(make_document(text), count_tokens=count_words, config=config)


def test_short_page_becomes_one_chunk() -> None:
    text = f"## Leave\n\n{words(8)}"

    (only,) = chunk(text)

    assert only.text == text
    assert (only.start_char, only.end_char) == (0, len(text))
    assert only.heading_path == ("Leave",)
    assert only.token_count == 10
    assert only.title == "Leave"
    assert only.section == "people"


def test_major_heading_starts_a_new_chunk() -> None:
    chunks = chunk(f"## Sick leave\n\n{words(7)}\n\n## Parental leave\n\n{words(7)}")

    assert [c.heading_path for c in chunks] == [("Sick leave",), ("Parental leave",)]
    assert chunks[1].text.startswith("## Parental leave")


def test_subsections_are_packed_together_within_the_budget() -> None:
    text = f"## Leave types\n\n### Sick\n\n{words(4)}\n\n### Parental\n\n{words(4)}"

    (only,) = chunk(text)

    assert only.heading_path == ("Leave types",)
    assert "### Sick" in only.text
    assert "### Parental" in only.text


def test_subsections_split_when_they_do_not_fit_together() -> None:
    chunks = chunk(f"## Leave types\n\n### Sick\n\n{words(12)}\n\n### Parental\n\n{words(12)}")

    assert [c.heading_path for c in chunks] == [("Leave types",), ("Leave types", "Parental")]


def test_tiny_section_is_merged_into_the_next_one() -> None:
    (only,) = chunk(f"## Note\n\nShort.\n\n## Details\n\n{words(8)}")

    assert "Short." in only.text
    assert "## Details" in only.text
    assert only.heading_path == ()


def test_tiny_last_section_joins_the_chunk_before_it() -> None:
    (only,) = chunk(f"## Policy\n\n{words(16)}\n\n## Contact\n\nAsk HR.")

    assert only.text.endswith("Ask HR.")
    assert only.token_count <= ALLOWED_MAX


def test_heading_path_follows_nesting() -> None:
    text = (
        f"## Benefits\n\n{words(12)}\n\n### Health\n\n{words(14)}\n\n"
        f"#### Dental\n\n{words(14)}\n\n## Travel\n\n{words(12)}"
    )

    assert [c.heading_path for c in chunk(text)] == [
        ("Benefits",),
        ("Benefits", "Health"),
        ("Benefits", "Health", "Dental"),
        ("Travel",),
    ]


def test_inline_markup_is_removed_from_heading_paths() -> None:
    (only,) = chunk(f"## **Paid** `time` _off_\n\n{words(8)}")

    assert only.heading_path == ("Paid time off",)


def test_oversized_section_is_split_into_overlapping_pieces() -> None:
    paragraphs = [words(6, word=f"p{n}w") for n in range(10)]
    text = "## Policy\n\n" + "\n\n".join(paragraphs)
    one_paragraph_of_overlap = ChunkingConfig(max_tokens=20, min_tokens=5, overlap_tokens=6)

    chunks = chunk(text, one_paragraph_of_overlap)

    assert len(chunks) >= 3
    assert all(c.token_count <= ALLOWED_MAX for c in chunks)
    assert all(c.heading_path == ("Policy",) for c in chunks)
    assert chunks[0].text.startswith("## Policy")
    for paragraph in paragraphs:
        assert any(paragraph in c.text for c in chunks), "no content may be lost"
    for earlier, later in itertools.pairwise(chunks):
        assert later.start_char < earlier.end_char, "neighbouring pieces must overlap"
        assert later.start_char > earlier.start_char
        repeated = text[later.start_char : earlier.end_char]
        assert len(repeated.split()) <= one_paragraph_of_overlap.overlap_tokens


def test_pieces_do_not_overlap_when_no_whole_unit_fits_the_overlap_budget() -> None:
    paragraphs = [words(6, word=f"p{n}w") for n in range(10)]
    text = "## Policy\n\n" + "\n\n".join(paragraphs)

    chunks = chunk(text)  # the default test budget allows 4 tokens of overlap

    for earlier, later in itertools.pairwise(chunks):
        assert later.start_char >= earlier.end_char
    for paragraph in paragraphs:
        assert sum(paragraph in c.text for c in chunks) == 1


def test_parent_heading_stays_attached_to_an_oversized_section() -> None:
    text = "## Policy\n\n### Details\n\n" + "\n\n".join(words(6, word=f"p{n}w") for n in range(8))

    chunks = chunk(text)

    assert chunks[0].text.startswith("## Policy\n\n### Details")
    assert all(c.text.strip() != "## Policy" for c in chunks)


def test_code_block_that_fits_is_never_cut() -> None:
    code = "```bash\n" + "\n".join(f"echo line {n}" for n in range(4)) + "\n```"
    text = f"## Setup\n\n{words(12)}\n\n{code}\n\n{words(12)}"

    chunks = chunk(text)

    assert any(code in c.text for c in chunks)
    assert all(c.text.count("```") % 2 == 0 for c in chunks)


def test_heading_inside_a_code_block_is_not_a_heading() -> None:
    (only,) = chunk(f"## Real heading\n\n{words(6)}\n\n```\n## not a heading\n```")

    assert only.heading_path == ("Real heading",)


def test_long_table_is_split_by_rows_and_every_piece_keeps_the_header() -> None:
    header = "| Item | Limit |\n|------|-------|"
    rows = [f"| item{n} | {n}00 dollars |" for n in range(12)]
    text = f"## Limits\n\n{header}\n" + "\n".join(rows)
    document = make_document(text)

    chunks = chunk_document(document, count_tokens=count_words, config=CONFIG)

    assert len(chunks) >= 3
    for piece in chunks:
        assert header in piece.text
        assert all(line.startswith("|") for line in piece.text.splitlines() if "item" in line)
    for row in rows:
        assert any(row in c.text for c in chunks)
    later = chunks[1]
    assert later.text.startswith(header)
    assert not document.text[later.start_char : later.end_char].startswith(header)


def test_very_long_line_is_cut_at_sentence_ends() -> None:
    sentences = [words(7, word=f"s{n}w") for n in range(8)]
    text = "## Rules\n\n" + " ".join(sentences)

    chunks = chunk(text)

    assert len(chunks) >= 3
    assert all(c.token_count <= ALLOWED_MAX for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)


def test_offsets_point_back_into_the_document() -> None:
    text = (
        f"Intro {words(8)}\n\n## A\n\n{words(14)}\n\n- one\n- two\n- three\n\n"
        f"### B\n\n" + "\n\n".join(words(6, word=f"q{n}w") for n in range(7))
    )
    document = make_document(text)

    chunks = chunk_document(document, count_tokens=count_words, config=CONFIG)

    assert chunks
    for piece in chunks:
        assert document.text[piece.start_char : piece.end_char] == piece.text
        assert 0 <= piece.start_char < piece.end_char <= len(text)
    assert [c.start_char for c in chunks] == sorted(c.start_char for c in chunks)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    text = "## A\n\n" + "\n\n".join(words(6, word=f"p{n}w") for n in range(9))

    first, second = chunk(text), chunk(text)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_changing_the_configuration_changes_chunk_ids() -> None:
    text = f"## A\n\n{words(8)}"
    other = ChunkingConfig(max_tokens=30, min_tokens=5, overlap_tokens=4)

    assert chunk(text)[0].chunk_id != chunk(text, other)[0].chunk_id


def test_heading_with_nothing_under_it_is_not_a_chunk() -> None:
    assert chunk("## Only a heading") == []
    assert [c.text for c in chunk(f"## Real\n\n{words(8)}\n\n## Empty")] == [
        f"## Real\n\n{words(8)}"
    ]


def test_empty_document_gives_no_chunks() -> None:
    assert chunk("") == []


def test_text_for_embedding_starts_with_the_context_header() -> None:
    chunks = chunk(f"## Benefits\n\n{words(12)}\n\n### Health\n\n{words(14)}")

    assert chunks[1].context_header == "Leave > Benefits > Health"
    assert chunks[1].text_for_embedding == f"Leave > Benefits > Health\n\n{chunks[1].text}"


@pytest.mark.parametrize(
    ("max_tokens", "min_tokens", "overlap_tokens"),
    [(10, 10, 0), (10, 0, 0), (10, 5, 10), (10, 5, -1)],
)
def test_invalid_configuration_is_rejected(
    max_tokens: int, min_tokens: int, overlap_tokens: int
) -> None:
    with pytest.raises(ValueError, match="tokens"):
        ChunkingConfig(max_tokens=max_tokens, min_tokens=min_tokens, overlap_tokens=overlap_tokens)


def test_text_with_no_spaces_is_still_cut_to_fit() -> None:
    def count_characters(texts: Sequence[str]) -> list[int]:
        return [len(text) for text in texts]

    config = ChunkingConfig(max_tokens=50, min_tokens=10, overlap_tokens=0)
    blob = "x" * 400
    document = make_document(f"## Log\n\n{blob}")

    chunks = chunk_document(document, count_tokens=count_characters, config=config)

    assert len(chunks) >= 8
    assert all(c.token_count <= 50 * 1.25 for c in chunks)
    assert sum(c.text.count("x") for c in chunks) == 400


def test_table_header_counts_against_the_budget_of_every_piece() -> None:
    header = "| Item | Limit |\n| --- | --- |"
    rows = [f"| item{n} | {n}00 dollars |" for n in range(12)]
    text = f"## Limits\n\n{header}\n" + "\n".join(rows)

    chunks = chunk(text, ChunkingConfig(max_tokens=30, min_tokens=5, overlap_tokens=0))

    assert len(chunks) >= 3
    assert all(c.token_count <= 30 for c in chunks)
    assert all(header in c.text for c in chunks)


def test_table_header_too_large_to_repeat_is_not_repeated() -> None:
    # 14 tokens: it fits in a chunk, but repeating it would use most of a 20-token budget.
    wide_header = "| Item name here | Spending limit here |\n| --- | --- |"
    rows = [f"| item{n} | {n}00 dollars |" for n in range(12)]
    text = f"## Limits\n\n{wide_header}\n" + "\n".join(rows)

    chunks = chunk(text)

    assert sum(wide_header in c.text for c in chunks) == 1
    assert all(c.token_count <= ALLOWED_MAX for c in chunks)
    for row in rows:
        assert any(row in c.text for c in chunks)


def test_chunk_id_changes_when_the_text_changes_in_place() -> None:
    before = chunk(f"## A\n\n{words(8, word='old')}")
    after = chunk(f"## A\n\n{words(8, word='new')}")

    assert (before[0].start_char, before[0].end_char) == (after[0].start_char, after[0].end_char)
    assert before[0].chunk_id != after[0].chunk_id


def test_split_into_blocks_keeps_each_heading_with_the_block_below_it() -> None:
    text = (
        "## Leave\n\nTake time off.\n\n- one\n- two\n\n"
        "### Sick\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    )

    assert split_into_blocks(text) == [
        "## Leave\n\nTake time off.",
        "- one\n- two",
        "### Sick\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
    ]


def test_split_into_blocks_keeps_a_trailing_heading() -> None:
    assert split_into_blocks("Some text.\n\n## See also") == ["Some text.\n\n## See also"]


def test_split_into_blocks_of_a_lone_heading_or_nothing() -> None:
    assert split_into_blocks("## Only a heading") == ["## Only a heading"]
    assert split_into_blocks("") == []
    assert split_into_blocks("   \n") == []
