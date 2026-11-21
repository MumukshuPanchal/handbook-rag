"""Render the golden set as one page a person can read and check.

A golden set is only as good as its review. The questions live in a JSON Lines file,
which is easy for code and tiring for people, so this module turns it into a page that
shows each question next to the passage it was written from, with the evidence marked.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from html import escape

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.ingest.models import Document

_STYLE = """
:root { color-scheme: light dark; }
body { font: 16px/1.55 system-ui, sans-serif; max-width: 52rem; margin: 2rem auto;
  padding: 0 1rem; }
h1 { font-size: 1.6rem; }
section { border-top: 1px solid #8884; padding: 1.2rem 0; }
h2 { font-size: 1.15rem; margin: 0.4rem 0; }
.tags { font-size: 0.8rem; opacity: 0.75; letter-spacing: 0.02em; }
.label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; opacity: 0.7;
  margin: 0.9rem 0 0.2rem; }
.passage { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.92rem;
  padding: 0.7rem 0.9rem; border-left: 3px solid #8886; background: #8881; margin: 0.3rem 0; }
mark { background: #ffd54a; color: #000; padding: 0 0.1em; }
.source { font-size: 0.9rem; margin: 0.2rem 0; }
.missing { color: #c62828; font-weight: 600; }
ol li { margin: 0.25rem 0; }
"""

_CHECKS_ANSWERABLE = (
    "The question reads like something an employee would ask, and makes sense on its own.",
    "The reference answer is correct and says only what the passage says.",
    "The marked evidence contains the answer.",
    "You would not expect a different page to answer the same question.",
)
_CHECKS_UNANSWERABLE = (
    "The question reads like something an employee would ask.",
    "You agree the handbook should not be able to answer it, for the reason given.",
)


def render_review_page(
    questions: Sequence[GoldenQuestion],
    documents: Sequence[Document],
    *,
    context_chars: int = 350,
) -> str:
    """Build a self-contained HTML page for reviewing the golden set.

    Every piece of text is escaped before it is put in the page. Handbook text is data
    from a public website, and must never be able to add markup or scripts of its own.
    """
    pages = {document.doc_id: document for document in documents}
    review: Counter[str] = Counter(question.review for question in questions)
    types: Counter[str] = Counter(question.type for question in questions)
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Golden set review</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>Golden set review</h1>",
        f"<p>{len(questions)} questions: {_counts(types)}. Review status: {_counts(review)}.</p>",
        '<p class="label">For a question the handbook answers, check that</p>',
        _numbered(_CHECKS_ANSWERABLE),
        '<p class="label">For a question the handbook cannot answer, check that</p>',
        _numbered(_CHECKS_UNANSWERABLE),
        "<p>Note the id of every question that fails a check. A question that passes gets "
        "<code>&quot;review&quot;: &quot;verified&quot;</code> in the golden set file; one "
        "that fails is fixed or removed.</p>",
    ]
    parts.extend(_question_section(question, pages, context_chars) for question in questions)
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _question_section(
    question: GoldenQuestion, pages: dict[str, Document], context_chars: int
) -> str:
    tags = [question.type, question.origin, f"review {question.review}"]
    if question.unanswerable_kind is not None:
        tags.insert(1, question.unanswerable_kind)
    lines = [
        f'<section id="{escape(question.id, quote=True)}">',
        f'<div class="tags">{escape(question.id)} · {escape(" · ".join(tags))}</div>',
        f"<h2>{escape(question.question)}</h2>",
    ]
    if question.reference_answer:
        lines.append('<p class="label">Reference answer</p>')
        lines.append(f"<p>{escape(question.reference_answer)}</p>")
    for evidence in question.evidence:
        page = pages.get(evidence.doc_id)
        if page is None:
            lines.append(
                f'<p class="missing">Page not in the corpus: {escape(evidence.doc_id)}</p>'
            )
            continue
        link = f'<a href="{escape(page.url, quote=True)}">{escape(page.title)}</a>'
        lines.append('<p class="label">Evidence</p>')
        lines.append(f'<p class="source">{link} · <code>{escape(evidence.doc_id)}</code></p>')
        start = page.text.find(evidence.quote)
        if start < 0:
            lines.append('<p class="missing">The quote was not found in this page.</p>')
            lines.append(f'<div class="passage">{escape(evidence.quote)}</div>')
            continue
        end = start + len(evidence.quote)
        before = page.text[max(0, start - context_chars) : start]
        after = page.text[end : end + context_chars]
        lines.append(
            f'<div class="passage">{"…" if start > context_chars else ""}{escape(before)}'
            f"<mark>{escape(evidence.quote)}</mark>{escape(after)}"
            f"{'…' if end + context_chars < len(page.text) else ''}</div>"
        )
    if question.note:
        lines.append('<p class="label">Note</p>')
        lines.append(f"<p>{escape(question.note)}</p>")
    lines.append("</section>")
    return "\n".join(lines)


def _counts(counter: Mapping[str, int]) -> str:
    return escape(", ".join(f"{count} {name}" for name, count in sorted(counter.items())))


def _numbered(items: Sequence[str]) -> str:
    return "<ol>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ol>"
