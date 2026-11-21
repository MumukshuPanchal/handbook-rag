from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.review import render_review_page
from handbook_rag.ingest.models import Document

PAGE = Document(
    doc_id="people/leave.md",
    title="Leave & <time off>",
    url="https://handbook.example.com/handbook/people/leave/?a=1&b=2",
    section="people",
    source_commit="a" * 40,
    text=(
        "## Parental leave\n\n<script>alert(1)</script> New parents get sixteen weeks "
        "of paid leave. Tell your manager early."
    ),
)
ANSWERABLE = GoldenQuestion.model_validate(
    {
        "id": "q001",
        "question": "How long is parental leave for <new> parents?",
        "reference_answer": "Sixteen weeks, paid.",
        "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
        "type": "answerable",
        "origin": "model-drafted",
    }
)
UNANSWERABLE = GoldenQuestion.model_validate(
    {
        "id": "q002",
        "question": "How do I book a desk in the Lisbon office?",
        "type": "unanswerable",
        "origin": "hand-written",
        "review": "verified",
        "unanswerable_kind": "false-premise",
        "note": "The company has no offices.",
    }
)


def test_page_shows_each_question_with_its_evidence_marked_in_context() -> None:
    page = render_review_page([ANSWERABLE, UNANSWERABLE], [PAGE])

    assert "2 questions: 1 answerable, 1 unanswerable" in page
    assert "1 pending, 1 verified" in page
    assert '<section id="q001">' in page
    assert "Sixteen weeks, paid." in page
    assert "New parents get <mark>sixteen weeks of paid leave</mark>. Tell your manager" in page
    assert "q002 · unanswerable · false-premise · hand-written · review verified" in page
    assert "The company has no offices." in page


def test_handbook_text_cannot_add_markup_to_the_page() -> None:
    page = render_review_page([ANSWERABLE], [PAGE])

    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "for &lt;new&gt; parents?" in page
    assert "Leave &amp; &lt;time off&gt;" in page
    assert 'href="https://handbook.example.com/handbook/people/leave/?a=1&amp;b=2"' in page


def test_long_pages_are_shown_as_an_excerpt_around_the_evidence() -> None:
    long_page = Document(
        PAGE.doc_id,
        PAGE.title,
        PAGE.url,
        PAGE.section,
        PAGE.source_commit,
        "start " + "before " * 200 + "sixteen weeks of paid leave" + " after" * 200 + " end",
    )

    page = render_review_page([ANSWERABLE], [long_page], context_chars=50)

    assert "…" in page
    assert "start" not in page.split("<section")[1]
    assert " end" not in page.split("<section")[1]
    assert "<mark>sixteen weeks of paid leave</mark>" in page


def test_a_label_that_no_longer_matches_the_corpus_is_shown_as_a_problem() -> None:
    rewritten = Document(PAGE.doc_id, "Leave", PAGE.url, "people", "a" * 40, "Rewritten.")

    stale_quote = render_review_page([ANSWERABLE], [rewritten])
    missing_page = render_review_page([ANSWERABLE], [])

    assert "The quote was not found in this page." in stale_quote
    assert "Page not in the corpus: people/leave.md" in missing_page
