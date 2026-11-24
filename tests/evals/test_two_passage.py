import json
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.two_passage import (
    KINDS,
    WhyTwo,
    check_two_passage_set,
    kinds_of,
    load_reasons,
)


def question(question_id: str = "t001", **changes: Any) -> GoldenQuestion:
    row: dict[str, Any] = {
        "id": question_id,
        "question": f"How does leave compare with the laptop limit ({question_id})?",
        "reference_answer": "Sixteen weeks, and one thousand dollars.",
        "evidence": [
            {"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"},
            {"doc_id": "finance/expenses.md", "quote": "one thousand dollars"},
        ],
        "type": "answerable",
        "origin": "model-drafted",
    }
    return GoldenQuestion.model_validate({**row, **changes})


def reason(question_id: str = "t001", kind: str = "comparison") -> WhyTwo:
    return WhyTwo.model_validate(
        {
            "id": question_id,
            "kind": kind,
            "only_first": "Only the leave is known.",
            "only_second": "Only the laptop limit is known.",
        }
    )


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_reasons_are_loaded_by_question_id(tmp_path: Path) -> None:
    path = write(tmp_path / "why.jsonl", [reason("t001").model_dump(), reason("t002").model_dump()])
    path.write_text(path.read_text() + "\n")  # a blank line at the end is skipped

    reasons = load_reasons(path)

    assert list(reasons) == ["t001", "t002"]
    assert reasons["t002"].only_second == "Only the laptop limit is known."


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"kind": "trivia"}, "kind"),
        ({"only_first": ""}, "only_first"),
        ({"only_second": ""}, "only_second"),
        ({"note": "not a field"}, "note"),
    ],
)
def test_a_reason_that_does_not_say_what_it_must_is_rejected_with_its_line(
    tmp_path: Path, change: dict[str, str], complaint: str
) -> None:
    rows = [reason("t001").model_dump(), {**reason("t002").model_dump(), **change}]
    path = write(tmp_path / "why.jsonl", rows)

    with pytest.raises(DatasetError, match=rf"why\.jsonl:2: (.|\n)*{complaint}"):
        load_reasons(path)


def test_two_reasons_for_one_question_and_broken_json_are_rejected(tmp_path: Path) -> None:
    twice = write(tmp_path / "twice.jsonl", [reason().model_dump(), reason().model_dump()])
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n")

    with pytest.raises(DatasetError, match=r"twice\.jsonl:2: id t001 is used twice"):
        load_reasons(twice)
    with pytest.raises(DatasetError, match=r"broken\.jsonl:1"):
        load_reasons(broken)


def test_a_set_of_two_passage_questions_with_a_reason_for_each_has_no_problems() -> None:
    questions = [question("t001"), question("t002")]
    reasons = {"t001": reason("t001"), "t002": reason("t002", "process")}

    assert check_two_passage_set(questions, reasons) == []
    assert kinds_of(questions, reasons) == {"comparison": 1, "process": 1}


def test_the_kinds_are_the_four_the_card_describes() -> None:
    assert KINDS == ("comparison", "rule-and-exception", "process", "term-and-rule")


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        (
            {"evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks"}]},
            "t001: 1 pieces of evidence, where two are needed",
        ),
        (
            {
                "evidence": [
                    {"doc_id": "people/leave.md", "quote": "sixteen weeks"},
                    {"doc_id": "people/leave.md", "quote": "paid leave"},
                ]
            },
            "t001: both passages are on people/leave.md",
        ),
        (
            {
                "evidence": [
                    {"doc_id": "a.md", "quote": "one"},
                    {"doc_id": "b.md", "quote": "two"},
                    {"doc_id": "c.md", "quote": "three"},
                ]
            },
            "t001: 3 pieces of evidence, where two are needed",
        ),
        ({"evidence_mode": "any"}, "t001: evidence_mode any says that one passage is enough"),
        (
            {
                "type": "unanswerable",
                "evidence": [],
                "reference_answer": "",
                "unanswerable_kind": "near-miss",
            },
            "t001: a two-passage question has an answer in the handbook",
        ),
    ],
)
def test_a_question_that_is_not_a_two_passage_question_is_a_problem(
    changes: dict[str, Any], problem: str
) -> None:
    assert check_two_passage_set([question(**changes)], {"t001": reason()}) == [problem]


def test_a_question_without_a_reason_and_a_reason_without_a_question_are_problems() -> None:
    problems = check_two_passage_set([question("t001")], {"t009": reason("t009")})

    assert problems == [
        "t001: no reason says why it needs both passages",
        "t009: a reason for a question that is not in the set",
    ]
