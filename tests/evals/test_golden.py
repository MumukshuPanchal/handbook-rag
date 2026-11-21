from collections import Counter

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.golden import (
    check_against_corpus,
    is_good_source,
    sample_source_chunks,
    word_overlap,
)
from handbook_rag.ingest.models import Chunk, Document

PROSE = "Team members submit an expense report within thirty days of the purchase. " * 8


def chunk(doc_id: str, text: str = PROSE, tokens: int = 200) -> Chunk:
    return Chunk(
        chunk_id=f"id-{doc_id}-{len(text)}",
        doc_id=doc_id,
        title="T",
        url="https://example.com/",
        section=doc_id.split("/")[0],
        source_commit="a" * 40,
        heading_path=(),
        start_char=0,
        end_char=len(text),
        text=text,
        token_count=tokens,
    )


def test_prose_of_a_reasonable_length_is_a_good_source() -> None:
    assert is_good_source(chunk("finance/expenses.md"))


def test_tables_code_and_tiny_or_huge_chunks_are_not_good_sources() -> None:
    table = "\n".join(["| Item | Limit |", "| --- | --- |"] + ["| Laptop | 1000 |"] * 30)

    assert not is_good_source(chunk("a/table.md", table))
    assert not is_good_source(chunk("a/code.md", PROSE + "\n```bash\necho hi\n```"))
    assert not is_good_source(chunk("a/short.md", tokens=40))
    assert not is_good_source(chunk("a/long.md", tokens=400))
    assert not is_good_source(chunk("a/thin.md", "## Heading\n\nSee the page."))


def corpus_of(sizes: dict[str, int]) -> list[Chunk]:
    return [chunk(f"{section}/page{n}.md") for section, size in sizes.items() for n in range(size)]


def test_sample_covers_every_section_and_favours_large_ones_less_than_proportionally() -> None:
    chunks = corpus_of({"engineering": 400, "finance": 25, "legal": 4})

    sampled = sample_source_chunks(chunks, count=30, seed=1)

    per_section = Counter(c.section for c in sampled)
    assert set(per_section) == {"engineering", "finance", "legal"}
    assert per_section["engineering"] > per_section["finance"] > per_section["legal"] >= 1
    assert per_section["engineering"] / len(sampled) < 400 / 429


def test_sample_is_reproducible_and_takes_one_chunk_per_page() -> None:
    chunks = corpus_of({"engineering": 50, "finance": 20})
    chunks += [chunk("finance/page0.md", PROSE + " More.")]

    first = sample_source_chunks(chunks, count=20, seed=3)
    again = sample_source_chunks(chunks, count=20, seed=3)
    other = sample_source_chunks(chunks, count=20, seed=4)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in again]
    assert [c.chunk_id for c in first] != [c.chunk_id for c in other]
    assert len({c.doc_id for c in first}) == len(first)
    assert len(first) <= 20


def test_sample_of_nothing_is_empty() -> None:
    assert sample_source_chunks([], count=10, seed=1) == []
    assert sample_source_chunks([chunk("a/tiny.md", tokens=10)], count=10, seed=1) == []


def test_word_overlap_measures_how_much_a_question_copies_its_evidence() -> None:
    quote = "Team members submit an expense report within thirty days of the purchase."

    copied = word_overlap("When do team members submit an expense report?", quote)
    paraphrased = word_overlap("What is the deadline for filing reimbursement claims?", quote)

    assert copied == 1.0
    assert paraphrased < 0.2
    assert word_overlap("What is it?", quote) == 0.0


def question(question_id: str, text: str, quote: str | None) -> GoldenQuestion:
    return GoldenQuestion.model_validate(
        {
            "id": question_id,
            "question": text,
            "reference_answer": "Thirty days." if quote else "",
            "evidence": [{"doc_id": "finance/expenses.md", "quote": quote}] if quote else [],
            "type": "answerable" if quote else "unanswerable",
            "origin": "model-drafted",
            "unanswerable_kind": None if quote else "out-of-scope",
        }
    )


def test_check_summarises_the_set_and_flags_copied_wording() -> None:
    page = Document("finance/expenses.md", "T", "https://example.com/", "finance", "a" * 40, PROSE)
    questions = [
        question(
            "q1",
            "When do team members submit an expense report?",
            "Team members submit an expense report",
        ),
        question(
            "q2", "What is the deadline for filing reimbursement claims?", "within thirty days"
        ),
        question("q3", "What is the dress code on Mars?", None),
    ]

    report = check_against_corpus(questions, [page])

    assert report.total == 3
    assert report.by_type == {"answerable": 2, "unanswerable": 1}
    assert report.by_section == {"finance": 2}
    assert report.by_unanswerable_kind == {"out-of-scope": 1}
    assert report.problems == []
    assert report.high_overlap == ["q1"]
    assert 0.3 < report.mean_overlap < 0.7


def test_check_reports_a_quote_that_another_page_repeats() -> None:
    page = Document("finance/expenses.md", "T", "https://example.com/", "finance", "a" * 40, PROSE)
    copy = Document("finance/archive.md", "T", "https://example.com/", "finance", "a" * 40, PROSE)
    other = Document("people/leave.md", "T", "https://example.com/", "people", "a" * 40, "Leave.")

    report = check_against_corpus(
        [question("q1", "When are expense reports due?", "within thirty days")],
        [page, copy, other],
    )

    assert report.problems == [
        "q1: quote also appears word for word in finance/archive.md, so the label is ambiguous"
    ]


def test_check_reports_a_quote_that_is_not_in_its_page() -> None:
    page = Document("finance/expenses.md", "T", "https://example.com/", "finance", "a" * 40, PROSE)

    report = check_against_corpus([question("q1", "When?", "within ninety days")], [page])

    assert len(report.problems) == 1
    assert "quote not found" in report.problems[0]
