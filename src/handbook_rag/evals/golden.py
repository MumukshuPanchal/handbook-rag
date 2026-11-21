"""Tools for building and checking the golden set.

Building a golden set has three steps: pick passages to write questions from, write the
questions, and check them. Picking and checking are code, so they are repeatable. Writing
is done by a model or a person following ``evals/golden/drafting-guidelines.md``, and is
then reviewed by a person.
"""

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from handbook_rag.evals.dataset import DatasetError, GoldenQuestion, locate_evidence
from handbook_rag.ingest.models import Chunk, Document

_WORD = re.compile(r"[a-z0-9]+")
# Words too common to say anything about whether a question copies its source.
_COMMON_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)


def is_good_source(chunk: Chunk) -> bool:
    """Whether a chunk has enough plain prose to write a clear factual question from."""
    lines = [line for line in chunk.text.splitlines() if line.strip()]
    table_lines = sum(line.lstrip().startswith("|") for line in lines)
    prose_characters = sum(len(line) for line in lines if not line.lstrip().startswith(("|", "#")))
    return (
        120 <= chunk.token_count <= 330
        and table_lines <= len(lines) / 2
        and "```" not in chunk.text
        and prose_characters >= 400
    )


def sample_source_chunks(chunks: Sequence[Chunk], *, count: int, seed: int) -> list[Chunk]:
    """Pick chunks to write questions from, spread across the handbook's sections.

    A purely random sample would mirror the corpus, where one section holds a third of
    all pages. Each section instead gets a share proportional to the square root of its
    size: large sections still get more questions, small ones are not left out. At most
    one chunk is taken per page, so the questions cover many pages. The same seed always
    gives the same sample.
    """
    generator = random.Random(seed)
    by_section: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if is_good_source(chunk):
            by_section[chunk.section].append(chunk)
    if not by_section:
        return []

    weights = {section: math.sqrt(len(members)) for section, members in by_section.items()}
    total_weight = sum(weights.values())
    picked: list[Chunk] = []
    for section in sorted(by_section):
        quota = max(1, round(count * weights[section] / total_weight))
        one_per_page: dict[str, Chunk] = {}
        for chunk in generator.sample(by_section[section], len(by_section[section])):
            one_per_page.setdefault(chunk.doc_id, chunk)
        picked.extend(list(one_per_page.values())[:quota])
    generator.shuffle(picked)
    return picked[:count]


def word_overlap(question: str, quote: str) -> float:
    """Share of the question's meaningful words that also appear in its evidence quote.

    A high value means the question reuses the passage's own wording. Such a question is
    easy for keyword matching and says little about search by meaning.
    """
    question_words = set(_WORD.findall(question.lower())) - _COMMON_WORDS
    if not question_words:
        return 0.0
    quote_words = set(_WORD.findall(quote.lower()))
    return len(question_words & quote_words) / len(question_words)


@dataclass(frozen=True, slots=True)
class GoldenSetReport:
    """What a check of the golden set against the corpus found."""

    total: int
    by_type: Counter[str]
    by_origin: Counter[str]
    by_review: Counter[str]
    by_section: Counter[str]
    by_unanswerable_kind: Counter[str]
    problems: list[str]
    """Labels that cannot be trusted: a missing page, a quote that is not in its page, or
    a quote that also appears on another page."""
    mean_overlap: float
    overlap_by_origin: dict[str, float]
    """The same share, per question origin. A question written without seeing the passage
    should share far fewer words with it."""
    high_overlap: list[str]
    """Ids of questions that share most of their words with their evidence."""


def check_against_corpus(
    questions: Sequence[GoldenQuestion],
    documents: Sequence[Document],
    *,
    high_overlap_threshold: float = 0.8,
) -> GoldenSetReport:
    """Verify every label against the corpus and summarise the set."""
    pages = {document.doc_id: document for document in documents}
    problems: list[str] = []
    overlaps: dict[str, float] = {}
    for question in questions:
        try:
            locate_evidence(question, pages)
        except DatasetError as error:
            problems.append(str(error))
        for evidence in question.evidence:
            # The handbook repeats some text on several pages. A retriever that returns
            # the other copy has found the answer, yet would be scored as a miss.
            copies = [
                document.doc_id
                for document in documents
                if document.doc_id != evidence.doc_id and evidence.quote in document.text
            ]
            if copies:
                more = f" and {len(copies) - 1} more" if len(copies) > 1 else ""
                problems.append(
                    f"{question.id}: quote also appears word for word in {copies[0]}{more}, "
                    "so the label is ambiguous"
                )
        if question.evidence:
            overlaps[question.id] = max(
                word_overlap(question.question, evidence.quote) for evidence in question.evidence
            )
    overlap_values: dict[str, list[float]] = defaultdict(list)
    for question in questions:
        if question.id in overlaps:
            overlap_values[question.origin].append(overlaps[question.id])
    return GoldenSetReport(
        total=len(questions),
        by_type=Counter(q.type for q in questions),
        by_origin=Counter(q.origin for q in questions),
        by_review=Counter(q.review for q in questions),
        by_section=Counter(q.section for q in questions if q.type == "answerable"),
        by_unanswerable_kind=Counter(
            q.unanswerable_kind for q in questions if q.unanswerable_kind is not None
        ),
        problems=problems,
        mean_overlap=sum(overlaps.values()) / len(overlaps) if overlaps else 0.0,
        overlap_by_origin={
            origin: sum(values) / len(values) for origin, values in sorted(overlap_values.items())
        },
        high_overlap=sorted(i for i, value in overlaps.items() if value >= high_overlap_threshold),
    )
