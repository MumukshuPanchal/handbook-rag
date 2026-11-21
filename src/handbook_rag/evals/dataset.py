"""The golden set: test questions with known answers and the passages that support them.

Evidence is recorded as a page and an exact quote from it, never as a chunk id. Chunk ids
change whenever chunking settings change; a quote does not. At evaluation time the quote
is located in the page, and a retrieved chunk counts as relevant when it covers it.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from handbook_rag.ingest.models import Document


class DatasetError(ValueError):
    """The golden set file is malformed or does not match the corpus."""


class Evidence(BaseModel):
    """A passage that answers the question: a page and an exact quote from its text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class GoldenQuestion(BaseModel):
    """One test question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = ""
    evidence: tuple[Evidence, ...] = ()
    type: Literal["answerable", "unanswerable"]
    """Whether the handbook contains the answer. Unanswerable questions test declining."""
    origin: Literal["hand-written", "model-drafted"]
    """Who wrote the question. Results are reported per origin, because questions drafted
    by a model tend to reuse the wording of their source passage."""
    review: Literal["pending", "verified"] = "pending"
    """Whether a person has checked the question, the answer and the evidence."""

    @property
    def section(self) -> str:
        """The handbook section of the first piece of evidence, or ``none``."""
        if not self.evidence:
            return "none"
        parts = self.evidence[0].doc_id.split("/")
        return parts[0] if len(parts) > 1 else "handbook"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Where a piece of evidence sits in a page: character offsets in the cleaned text."""

    doc_id: str
    start: int
    end: int


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Read a golden set from a JSON Lines file, one question per line."""
    questions: list[GoldenQuestion] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(GoldenQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as error:
            raise DatasetError(f"{path}:{number}: {error}") from error
    problems = check_golden_set(questions)
    if problems:
        raise DatasetError(f"{path}: " + "; ".join(problems[:5]))
    return questions


def check_golden_set(questions: Sequence[GoldenQuestion]) -> list[str]:
    """Problems that can be found without looking at the corpus."""
    problems: list[str] = []
    for question_id, count in Counter(q.id for q in questions).items():
        if count > 1:
            problems.append(f"id {question_id} is used {count} times")
    normalised = Counter(" ".join(q.question.lower().split()) for q in questions)
    for text, count in normalised.items():
        if count > 1:
            problems.append(f"question appears {count} times: {text[:60]!r}")
    for q in questions:
        if q.type == "answerable" and not q.evidence:
            problems.append(f"{q.id}: an answerable question needs evidence")
        if q.type == "answerable" and not q.reference_answer.strip():
            problems.append(f"{q.id}: an answerable question needs a reference answer")
        if q.type == "unanswerable" and q.evidence:
            problems.append(f"{q.id}: an unanswerable question cannot have evidence")
    return problems


def locate_evidence(question: GoldenQuestion, documents: dict[str, Document]) -> list[EvidenceSpan]:
    """Find each quote in its page. Raises if a page or a quote is missing.

    A missing quote means the corpus changed, or the evidence was not copied exactly.
    Either way the label can no longer be trusted, so this fails loudly instead of
    silently scoring the question as a miss.
    """
    spans: list[EvidenceSpan] = []
    for evidence in question.evidence:
        document = documents.get(evidence.doc_id)
        if document is None:
            raise DatasetError(f"{question.id}: page {evidence.doc_id} is not in the corpus")
        start = document.text.find(evidence.quote)
        if start < 0:
            raise DatasetError(
                f"{question.id}: quote not found in {evidence.doc_id}: {evidence.quote[:60]!r}"
            )
        spans.append(EvidenceSpan(evidence.doc_id, start, start + len(evidence.quote)))
    return spans
