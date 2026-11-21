import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.dataset import (
    DatasetError,
    EvidenceSpan,
    GoldenQuestion,
    load_golden_set,
    locate_evidence,
)
from handbook_rag.ingest.models import Document

PAGE = Document(
    doc_id="people/leave.md",
    title="Leave",
    url="https://handbook.example.com/handbook/people/leave/",
    section="people",
    source_commit="a" * 40,
    text="## Parental leave\n\nNew parents get sixteen weeks of paid leave.",
)
ANSWERABLE: dict[str, Any] = {
    "id": "q001",
    "question": "How long is parental leave?",
    "reference_answer": "Sixteen weeks, paid.",
    "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
    "type": "answerable",
    "origin": "model-drafted",
}
UNANSWERABLE: dict[str, Any] = {
    "id": "q002",
    "question": "What is the office dress code on Mars?",
    "type": "unanswerable",
    "origin": "hand-written",
    "unanswerable_kind": "out-of-scope",
    "note": "The company has no office on Mars.",
}


def write_lines(path: Path, *rows: dict[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_loads_answerable_and_unanswerable_questions(tmp_path: Path) -> None:
    first, second = load_golden_set(write_lines(tmp_path / "q.jsonl", ANSWERABLE, UNANSWERABLE))

    assert first.id == "q001"
    assert first.evidence[0].quote == "sixteen weeks of paid leave"
    assert first.section == "people"
    assert first.review == "pending"
    assert first.unanswerable_kind is None
    assert second.type == "unanswerable"
    assert second.evidence == ()
    assert second.section == "none"
    assert second.unanswerable_kind == "out-of-scope"
    assert second.note == "The company has no office on Mars."


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(ANSWERABLE) + "\n\n" + json.dumps(UNANSWERABLE) + "\n")

    assert len(load_golden_set(path)) == 2


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"type": "maybe"}, "type"),
        ({"origin": "unknown"}, "origin"),
        ({"unexpected_field": 1}, "unexpected_field"),
        ({"question": ""}, "question"),
        ({"evidence": []}, "needs evidence"),
        ({"reference_answer": " "}, "needs a reference answer"),
        ({"unanswerable_kind": "near-miss"}, "only an unanswerable question"),
    ],
)
def test_malformed_question_is_rejected_with_its_location(
    tmp_path: Path, change: dict[str, Any], expected: str
) -> None:
    path = write_lines(tmp_path / "q.jsonl", {**ANSWERABLE, **change})

    with pytest.raises(DatasetError, match=expected):
        load_golden_set(path)


def test_invalid_json_reports_the_line(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(ANSWERABLE) + "\n{not json\n")

    with pytest.raises(DatasetError, match=r"q\.jsonl:2"):
        load_golden_set(path)


def test_duplicate_ids_and_duplicate_questions_are_rejected(tmp_path: Path) -> None:
    same_id = write_lines(tmp_path / "a.jsonl", ANSWERABLE, {**UNANSWERABLE, "id": "q001"})
    same_text = write_lines(
        tmp_path / "b.jsonl",
        ANSWERABLE,
        {**ANSWERABLE, "id": "q009", "question": "how long is  PARENTAL leave?"},
    )

    with pytest.raises(DatasetError, match="used 2 times"):
        load_golden_set(same_id)
    with pytest.raises(DatasetError, match="appears 2 times"):
        load_golden_set(same_text)


def test_unanswerable_question_cannot_carry_evidence(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "q.jsonl", {**UNANSWERABLE, "evidence": ANSWERABLE["evidence"]})

    with pytest.raises(DatasetError, match="cannot have evidence"):
        load_golden_set(path)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"unanswerable_kind": None}, "needs unanswerable_kind"),
        ({"unanswerable_kind": "too-hard"}, "unanswerable_kind"),
    ],
)
def test_unanswerable_question_must_say_why_it_cannot_be_answered(
    tmp_path: Path, change: dict[str, Any], expected: str
) -> None:
    path = write_lines(tmp_path / "q.jsonl", {**UNANSWERABLE, **change})

    with pytest.raises(DatasetError, match=expected):
        load_golden_set(path)


def test_evidence_is_located_by_its_quote() -> None:
    question = GoldenQuestion.model_validate(ANSWERABLE)

    (span,) = locate_evidence(question, {PAGE.doc_id: PAGE})

    assert span == EvidenceSpan("people/leave.md", 35, 62)
    assert PAGE.text[span.start : span.end] == "sixteen weeks of paid leave"


def test_missing_page_or_quote_fails_loudly() -> None:
    question = GoldenQuestion.model_validate(ANSWERABLE)
    rewritten_page = dataclasses.replace(PAGE, text="The policy was rewritten.")

    with pytest.raises(DatasetError, match="not in the corpus"):
        locate_evidence(question, {})
    with pytest.raises(DatasetError, match="quote not found"):
        locate_evidence(question, {PAGE.doc_id: rewritten_page})


def test_fingerprint_changes_only_when_something_a_score_depends_on_changes() -> None:
    original = GoldenQuestion.model_validate(ANSWERABLE).fingerprint

    def fingerprint(**change: Any) -> str:
        return GoldenQuestion.model_validate({**ANSWERABLE, **change}).fingerprint

    assert len(original) == 16
    assert fingerprint() == original
    assert fingerprint(review="verified", note="Checked on the page.", id="q777") == original
    assert fingerprint(question="How many weeks is parental leave?") != original
    assert fingerprint(reference_answer="Sixteen weeks.") != original
    assert (
        fingerprint(evidence=[{"doc_id": "people/leave.md", "quote": "sixteen weeks"}]) != original
    )
