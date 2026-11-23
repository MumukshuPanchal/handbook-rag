"""The labelled answers for the correctness judge, as they are in the repository."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.evals.correctness import (
    CORRECT,
    CORRECT_VERDICTS,
    NOT_CORRECT,
    PLANTED_KINDS,
)
from handbook_rag.evals.judging import UNCLEAR
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE

CASES = Path(__file__).parents[2] / "evals" / "judge" / "correct" / "cases.jsonl"


def cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").split("\n") if line]


def test_the_file_loads_as_the_judge_check_reads_it() -> None:
    labelled = CORRECT.load_labelled(CASES, {})

    assert len(labelled) == len({case.case.case_id for case in labelled}) == 167
    assert sum(bool(case.planted) for case in labelled) == 40


def test_real_answers_and_planted_faults_are_in_the_stated_numbers() -> None:
    kinds = Counter(case["kind"] for case in cases())

    assert kinds.pop("real") == 127
    assert kinds == dict.fromkeys(PLANTED_KINDS, 5)


def test_the_verdicts_are_in_the_numbers_the_dataset_card_gives() -> None:
    expected = Counter(case["expected"] for case in cases())

    assert expected == {"correct": 127, "incorrect": 35, "partial": 5}
    assert set(expected) <= {*CORRECT_VERDICTS, UNCLEAR}


def test_a_copy_with_a_planted_fault_differs_from_its_real_answer_in_the_answer_alone() -> None:
    real = {case["case"]: case for case in cases() if case["kind"] == "real"}
    planted = [case for case in cases() if case["kind"] != "real"]

    # Each fault was planted in another answer.
    assert len({case["question_id"] for case in planted}) == len(planted) == 40
    for case in planted:
        source = real[case["question_id"]]
        assert case["case"] == f"{source['case']}-v1"
        assert (case["question"], case["reference"]) == (source["question"], source["reference"])
        assert case["answer"] != source["answer"], case["case"]
        assert case["change"].strip(), case["case"]
        # It was planted in an answer that every reader called correct.
        assert source["expected"] == "correct", case["case"]
        # An omission leaves an answer that is right as far as it goes. Every other fault
        # makes it wrong.
        wanted = "partial" if case["kind"] == "omission" else "incorrect"
        assert case["expected"] == wanted, case["case"]
        # The second reader, who was not told which answers were altered, faulted it too.
        assert case["readers"]["second"] in NOT_CORRECT, case["case"]


def test_every_label_records_who_decided_it() -> None:
    settled_by_third = 0
    for case in cases():
        readers = case["readers"]
        assert {"first", "second"} <= set(readers) <= {"first", "second", "third"}, case["case"]
        if "third" not in readers:
            # Without a third reader, the label is what the two readers agreed on.
            assert readers["first"] == readers["second"] == case["expected"], case["case"]
            continue
        settled_by_third += 1
        against_two = readers["first"] == readers["second"] != readers["third"]
        assert (case["expected"] == UNCLEAR) == against_two, case["case"]
        if not against_two:
            assert case["expected"] == readers["third"], case["case"]
    assert settled_by_third == 8


def test_no_answer_is_the_sentence_that_code_takes_for_a_decline() -> None:
    # A reply that is exactly that sentence never reaches the judge: code counts it as a
    # decline. The planted replies that give no answer are worded otherwise.
    for case in cases():
        assert case["answer"].strip() != NOT_FOUND_MESSAGE, case["case"]
        assert all(case[field].strip() for field in ("question", "reference", "answer"))
