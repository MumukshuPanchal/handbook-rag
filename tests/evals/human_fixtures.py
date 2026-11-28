"""Small labelled sets in the layout of the real ones, for the tests of the person's sample.

Ninety questions. Each has a real answer in both sets. In the first set an answer to an odd
question has two statements and one to an even question has one, all labelled supported;
in the second set every real answer is labelled correct. Copies with one planted fault are
written on the first 24 questions, three of each of the eight kinds of the first set, and on
questions 41 to 56, two of each kind of the second. A copy's text says nothing of its kind.

The disputes the drawing looks for first: statement 2 of question 9 is labelled
``unclear``; the readers split on statement 1 of question 11; the saved check of version 1
of the judge of support differs from the set on statement 1 of question 13, and version 2 on
statement 1 of question 15; the judge of correctness differs on the real answer to question
60 and on the copy of question 41. One real statement is labelled ``no_claim`` (question 5)
and one unsupported (question 7), as in the real set. No label here is anybody's: the tests
made them.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.evals.correctness import CorrectnessCase, load_labelled_correctness
from handbook_rag.evals.human_sample import (
    CORRECT_PART,
    SAVED_CHECKS,
    SUPPORT_PART,
    Sets,
    find_disputes,
    load_saved_checks,
    reader_splits,
)
from handbook_rag.evals.judge import SupportCase
from handbook_rag.evals.judging import LabelledCase
from handbook_rag.evals.support import load_labelled_cases, passage_hash
from handbook_rag.ingest.build import Corpus
from handbook_rag.ingest.models import Chunk
from tests.support import make_chunk

QUESTIONS = 90
SUPPORT_KINDS = (
    "number",
    "name",
    "added_detail",
    "scope",
    "negation",
    "uncited",
    "wrong_passage",
    "other_question",
)
CORRECT_KINDS = (
    "number",
    "name",
    "negation",
    "scope",
    "contradiction",
    "other_question",
    "no_answer",
    "omission",
)
SUPPORT_PATH = Path("evals/judge/support/cases.jsonl")
CORRECT_PATH = Path("evals/judge/correct/cases.jsonl")
# Where a saved check differs from the set: (part, case, statement) -> the judge's verdict.
DIFFERS = {
    "support_v1": {("q013", 1): "unsupported"},
    "support_v2": {("q015", 1): "unsupported"},
    "correct": {("q060", 1): "partial", ("q041-v1", 1): "correct"},
}


def question(number: int) -> str:
    return f"How many days of leave do people in team {number} get?"


def case_id(number: int) -> str:
    return f"q{number:03d}"


def chunks() -> list[Chunk]:
    """Two passages for each question, the second for a citation of the wrong passage."""
    return [
        make_chunk(
            f"People in team {number} get {number + 10} days{' (other)' * extra}.",
            doc_id=f"teams/{number}-{extra}.md",
            title=f"Team {number}",
            heading_path=("Leave",),
        )
        for number in range(1, QUESTIONS + 1)
        for extra in (0, 1)
    ]


def corpus() -> Corpus:
    return Corpus(
        documents=[],
        chunks=chunks(),
        skipped=Counter(),
        chunking_version="v-test",
        source_commit="a" * 40,
    )


def _passages(number: int, by_doc: dict[str, Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "number": place,
            "chunk_id": by_doc[f"teams/{number}-{extra}.md"].chunk_id,
            "text_sha256": passage_hash(by_doc[f"teams/{number}-{extra}.md"].text),
        }
        for place, extra in ((1, 0), (2, 1))
    ]


def _agreed(label: str) -> dict[str, str]:
    return {"first": label, "second": label}


def _real_statements(number: int) -> list[dict[str, Any]]:
    statements: list[dict[str, Any]] = [
        {
            "number": 1,
            "text": f"People in team {number} get {number + 10} days.",
            "cites": [1],
            "expected": "supported",
            "readers": _agreed("supported"),
        }
    ]
    if number % 2:
        statements.append(
            {
                "number": 2,
                "text": f"That is the rule for team {number}.",
                "cites": [1],
                "expected": "supported",
                "readers": _agreed("supported"),
            }
        )
    return statements


def support_rows() -> list[dict[str, Any]]:
    by_doc = {chunk.doc_id: chunk for chunk in chunks()}
    rows: list[dict[str, Any]] = []
    for number in range(1, QUESTIONS + 1):
        statements = _real_statements(number)
        if number == 5:
            statements[1].update(
                text="The handbook says no more.", expected="no_claim", readers=_agreed("no_claim")
            )
        if number == 7:
            statements[1].update(cites=[], expected="unsupported", readers=_agreed("unsupported"))
        if number == 9:
            statements[1].update(
                expected="unclear",
                readers={"first": "supported", "second": "supported", "third": "unsupported"},
            )
        if number == 11:
            statements[0]["readers"] = {
                "first": "supported",
                "second": "unsupported",
                "third": "supported",
            }
        rows.append(
            {
                "case": case_id(number),
                "question_id": case_id(number),
                "kind": "real",
                "question": question(number),
                "passages": _passages(number, by_doc),
                "statements": statements,
            }
        )
    for place, kind in enumerate(SUPPORT_KINDS):
        for copy in range(3):
            number = 1 + place * 3 + copy
            statements = _real_statements(number)
            statements[0] = {
                "number": 1,
                "text": f"People in team {number} get {number + 99} days.",
                "cites": [] if kind == "uncited" else [2] if kind == "wrong_passage" else [1],
                "expected": "unsupported",
                "readers": _agreed("unsupported"),
                "planted": kind,
                "change": "a stand-in change",
            }
            rows.append(
                {
                    "case": f"{case_id(number)}-v1",
                    "question_id": case_id(number),
                    "kind": kind,
                    "question": question(number),
                    "passages": _passages(number, by_doc),
                    "statements": statements,
                }
            )
    return rows


def correct_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "case": case_id(number),
            "question_id": case_id(number),
            "kind": "real",
            "question": question(number),
            "reference": f"{number + 10} days.",
            "answer": f"People in team {number} get {number + 10} days.",
            "expected": "correct",
            "readers": _agreed("correct"),
        }
        for number in range(1, QUESTIONS + 1)
    ]
    for place, kind in enumerate(CORRECT_KINDS):
        for copy in range(2):
            number = 41 + place * 2 + copy
            label = "partial" if kind == "omission" else "incorrect"
            rows.append(
                {
                    "case": f"{case_id(number)}-v1",
                    "question_id": case_id(number),
                    "kind": kind,
                    "question": question(number),
                    "reference": f"{number + 10} days.",
                    "answer": f"People in team {number} get {number + 77} days.",
                    "expected": label,
                    "readers": _agreed(label),
                }
            )
    return rows


SUPPORT_RUBRIC = "# Rubric: is a statement backed?\n\nCheck each statement.\n"
CORRECT_RUBRIC = "A note for people.\n\n---\n\nYou check answers against a reference answer.\n"


def write_sets(root: Path) -> None:
    """The two sets and the two rubrics, under ``root`` as in the project."""
    for path, rows in ((SUPPORT_PATH, support_rows()), (CORRECT_PATH, correct_rows())):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (root / "evals/judge/rubric.md").write_text(SUPPORT_RUBRIC, encoding="utf-8")
    (root / "evals/judge/correct/rubric.md").write_text(CORRECT_RUBRIC, encoding="utf-8")


def load_cases(
    root: Path,
) -> tuple[list[LabelledCase[SupportCase]], list[LabelledCase[CorrectnessCase]]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks()}
    return (
        load_labelled_cases(root / SUPPORT_PATH, by_id),
        load_labelled_correctness(root / CORRECT_PATH, by_id),
    )


def write_saved_checks(
    root: Path, *, leave_out: str | None = None, without: tuple[str, ...] = ()
) -> None:
    """Saved checks of the three judges: the set's label everywhere but in :data:`DIFFERS`.

    A label ``unclear`` gets ``supported``. ``leave_out`` names a case no check has a verdict
    on; ``without`` names checks that are not saved at all.
    """
    support, correct = load_cases(root)
    for check in SAVED_CHECKS:
        if check.short in without:
            continue
        cases: list[LabelledCase[Any]] = list(support if check.part == SUPPORT_PART else correct)
        directory = root / check.run_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps({"config": {"judge_model": f"stand-in {check.short}"}, "attempts": 1})
        )
        lines = []
        for labelled in cases:
            if labelled.case.case_id == leave_out:
                continue
            verdicts = [
                {
                    "number": number,
                    "verdict": DIFFERS[check.short].get(
                        (labelled.case.case_id, number),
                        "supported" if expected == "unclear" else expected,
                    ),
                    "reason": f"stand-in reason {check.short}",
                }
                for number, expected in sorted(labelled.expected.items())
            ]
            lines.append(
                json.dumps(
                    {
                        "case": labelled.case.case_id,
                        "fingerprint": labelled.case.fingerprint,
                        "verdicts": verdicts,
                        "model": "stand-in",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "seconds": 0.0,
                    }
                )
            )
        (directory / "judgements.jsonl").write_text("\n".join(lines) + "\n")


def load_sets(root: Path) -> Sets:
    """The sets as the commands load them, with the disputes the saved checks and readers hold."""
    support, correct = load_cases(root)
    splits = {
        SUPPORT_PART: reader_splits(root / SUPPORT_PATH),
        CORRECT_PART: reader_splits(root / CORRECT_PATH),
    }
    return Sets(support, correct, find_disputes(support, correct, load_saved_checks(root), splits))


def make_project(root: Path) -> Sets:
    """Sets, rubrics and saved checks under ``root``, and the sets loaded from them."""
    write_sets(root)
    write_saved_checks(root)
    return load_sets(root)
