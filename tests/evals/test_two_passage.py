import json
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.two_passage import (
    KINDS,
    ReadingLine,
    SecondReading,
    WhyTwo,
    check_two_passage_set,
    kinds_of,
    load_reasons,
    load_second_reading,
    reading_of,
    second_reading_of,
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


CONFIRMS = {"both": "complete", "first": "partial", "second": "partial"}
FIRST_SUFFICES = {"both": "complete", "first": "complete", "second": "partial"}


def read_again(question_id: str = "t001", labels: dict[str, str] = CONFIRMS) -> WhyTwo:
    """A reason with the labels of a second reading, and what they come to."""
    confirmed = SecondReading.model_validate(labels).confirms
    row = {**reason(question_id).model_dump(), "second_reading": labels, "confirmed": confirmed}
    return WhyTwo.model_validate(row)


@pytest.mark.parametrize(
    ("both", "first", "second", "confirms"),
    [
        ("complete", "partial", "partial", True),
        ("complete", "none", "partial", True),
        ("complete", "none", "none", True),
        ("complete", "complete", "partial", False),
        ("complete", "partial", "complete", False),
        ("partial", "partial", "partial", False),
        ("none", "none", "none", False),
    ],
)
def test_a_reading_confirms_when_both_passages_suffice_and_neither_does_alone(
    both: str, first: str, second: str, confirms: bool
) -> None:
    labels = {"both": both, "first": first, "second": second}

    assert SecondReading.model_validate(labels).confirms is confirms


def test_a_reason_keeps_the_labels_of_its_second_reading_and_what_they_come_to() -> None:
    assert read_again().confirmed is True
    assert read_again(labels=FIRST_SUFFICES).confirmed is False
    assert reason().second_reading is None
    assert reason().confirmed is None


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"confirmed": True}, "confirmed is given, and no second reading"),
        ({"second_reading": CONFIRMS}, "a second reading is given, and confirmed is not"),
        (
            {"second_reading": CONFIRMS, "confirmed": False},
            "confirmed is false, and the labels of the second reading say true",
        ),
        (
            {"second_reading": FIRST_SUFFICES, "confirmed": True},
            "confirmed is true, and the labels of the second reading say false",
        ),
        ({"second_reading": {**CONFIRMS, "first": "mostly"}, "confirmed": True}, "first"),
        ({"second_reading": {"both": "complete"}, "confirmed": False}, "first"),
    ],
)
def test_a_confirmed_that_its_labels_contradict_is_rejected_with_its_line(
    tmp_path: Path, change: dict[str, Any], complaint: str
) -> None:
    path = write(tmp_path / "why.jsonl", [{**reason().model_dump(exclude_none=True), **change}])

    with pytest.raises(DatasetError, match=rf"why\.jsonl:1: (.|\n)*{complaint}"):
        load_reasons(path)


def test_the_review_says_what_the_second_reading_found() -> None:
    confirmed, disputed = read_again("t001"), read_again("t002", FIRST_SUFFICES)
    reasons = {"t001": confirmed, "t002": disputed}
    accepted = question("t001", review="checked")
    kept_open = question("t002", note="Disputed: the first passage was enough for one reader.")

    assert check_two_passage_set([accepted, kept_open], reasons) == []
    assert reading_of([accepted, kept_open], reasons) == {"confirmed": 1, "disputed": 1}
    assert reading_of([question("t003")], {"t003": reason("t003")}) == {"not read a second time": 1}


@pytest.mark.parametrize(
    ("labels", "changes", "problem"),
    [
        (
            CONFIRMS,
            {},
            "t001: confirmed by the second reading, and still marked pending",
        ),
        (
            FIRST_SUFFICES,
            {"review": "checked"},
            "t001: marked checked, and the second reading did not confirm that it needs both "
            "passages",
        ),
        (FIRST_SUFFICES, {}, "t001: disputed, and no note says what the dispute is"),
        (None, {"review": "checked"}, "t001: marked checked, and no second reading is recorded"),
    ],
)
def test_a_review_that_does_not_fit_the_second_reading_is_a_problem(
    labels: dict[str, str] | None, changes: dict[str, Any], problem: str
) -> None:
    why = reason() if labels is None else read_again(labels=labels)

    assert check_two_passage_set([question(**changes)], {"t001": why}) == [problem]


def line(item: str, question_id: str, shown: str, label: str = "partial") -> dict[str, Any]:
    return {
        "item": item,
        "question_id": question_id,
        "shown": shown,
        "label": label,
        "confidence": "high",
        "answer": "Sixteen weeks." if label == "complete" else "The passage gives half of it.",
        "missing": None if label == "complete" else "The laptop limit.",
    }


READING = [
    line("a-01", "t001", "both", "complete"),
    line("b-01", "t001", "first"),
    line("c-01", "t001", "second"),
    line("a-02", "q001", "control", "complete"),
    line("b-02", "q002", "control-and-neighbour", "complete"),
]


def test_the_lines_of_the_readers_give_each_question_its_three_labels(tmp_path: Path) -> None:
    lines = load_second_reading(write(tmp_path / "second-reading.jsonl", READING))

    assert [reading.item for reading in lines] == ["a-01", "b-01", "c-01", "a-02", "b-02"]
    # The controls are golden questions, and are passed over.
    assert second_reading_of(lines) == {"t001": SecondReading.model_validate(CONFIRMS)}
    assert second_reading_of(lines)["t001"].confirms


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"label": "mostly"}, "label"),
        ({"shown": "third"}, "shown"),
        ({"confidence": "certain"}, "confidence"),
        ({"answer": ""}, "answer"),
        ({"missing": None}, "a complete item has nothing missing"),
        ({"label": "complete"}, "a complete item has nothing missing"),
        ({"reader": "a"}, "reader"),
    ],
)
def test_a_line_of_a_reader_that_is_not_in_form_is_rejected_with_its_place(
    tmp_path: Path, change: dict[str, Any], complaint: str
) -> None:
    rows = [READING[0], {**READING[1], **change}]
    path = write(tmp_path / "second-reading.jsonl", rows)

    with pytest.raises(DatasetError, match=rf"second-reading\.jsonl:2: (.|\n)*{complaint}"):
        load_second_reading(path)


def test_an_item_with_two_lines_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "second-reading.jsonl", [READING[0], READING[0]])

    with pytest.raises(DatasetError, match=r"second-reading\.jsonl:2: item a-01 is used twice"):
        load_second_reading(path)


def test_a_question_must_be_shown_in_each_of_its_three_ways_once() -> None:
    twice = [*READING, line("c-02", "t001", "first")]
    two_ways = READING[:2]

    with pytest.raises(DatasetError, match="t001 was shown with first more than once"):
        second_reading_of([ReadingLine.model_validate(row) for row in twice])
    with pytest.raises(DatasetError, match="t001 was shown with both, first only"):
        second_reading_of([ReadingLine.model_validate(row) for row in two_ways])
