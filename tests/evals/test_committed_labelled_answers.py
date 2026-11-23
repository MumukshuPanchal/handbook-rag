"""The labelled answers in the repository are well formed.

These tests need no corpus. That every passage is still in the corpus with the words it
was labelled against is checked when the labelled answers are loaded for a judge check.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.evals.judge import VERDICTS
from handbook_rag.evals.judging import UNCLEAR

CASES = Path(__file__).parents[2] / "evals" / "judge" / "support" / "cases.jsonl"
FAULTS = {
    "number",
    "name",
    "added_detail",
    "scope",
    "negation",
    "uncited",
    "wrong_passage",
    "other_question",
}


def cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").split("\n") if line]


def test_every_answer_has_what_a_judge_check_reads() -> None:
    ids = [case["case"] for case in cases()]

    assert len(ids) == len(set(ids)) == 169
    for case in cases():
        assert case["question"].strip(), case["case"]
        assert case["statements"], case["case"]
        numbers = [statement["number"] for statement in case["statements"]]
        assert numbers == list(range(1, len(numbers) + 1)), case["case"]
        given = {passage["number"] for passage in case["passages"]}
        for passage in case["passages"]:
            assert passage["chunk_id"], case["case"]
            assert re.fullmatch(r"[0-9a-f]{16}", passage["text_sha256"]), case["case"]
        for statement in case["statements"]:
            assert statement["text"].strip(), case["case"]
            assert statement["expected"] in {*VERDICTS, UNCLEAR}, case["case"]
            # A statement can only cite a passage the judge is shown.
            assert set(statement["cites"]) <= given, case["case"]


def test_real_answers_and_planted_faults_are_in_the_stated_numbers() -> None:
    kinds = Counter(case["kind"] for case in cases())

    assert kinds.pop("real") == 129
    assert kinds == dict.fromkeys(FAULTS, 5)


def test_an_answer_with_a_planted_fault_has_exactly_one_and_it_is_labelled_unsupported() -> None:
    for case in cases():
        planted = [statement for statement in case["statements"] if "planted" in statement]
        if case["kind"] == "real":
            assert planted == [], case["case"]
            continue
        (statement,) = planted
        assert statement["planted"] == case["kind"], case["case"]
        assert statement["expected"] == "unsupported", case["case"]
        assert statement["change"].strip(), case["case"]
        # The second reader, who was not told which statements were altered, found it too.
        assert statement["readers"]["second"] == "unsupported", case["case"]
        if case["kind"] == "uncited":
            assert statement["cites"] == [], case["case"]


def test_every_label_records_who_decided_it() -> None:
    settled_by_third = 0
    for case in cases():
        for statement in case["statements"]:
            readers = statement["readers"]
            assert {"first", "second"} <= set(readers) <= {"first", "second", "third"}
            if "third" in readers:
                settled_by_third += 1
                continue
            # Without a third reader, the label is what the two readers agreed on.
            assert readers["first"] == readers["second"] == statement["expected"], case["case"]
    assert settled_by_third > 0


def test_a_statement_is_unclear_only_where_the_third_reader_stood_against_two() -> None:
    for case in cases():
        for statement in case["statements"]:
            readers = statement["readers"]
            against_two = (
                "third" in readers and readers["first"] == readers["second"] != readers["third"]
            )
            assert (statement["expected"] == UNCLEAR) == against_two, case["case"]
            if "third" in readers and not against_two:
                assert statement["expected"] == readers["third"], case["case"]


def test_the_verdicts_are_in_the_numbers_the_dataset_card_gives() -> None:
    expected = Counter(
        statement["expected"] for case in cases() for statement in case["statements"]
    )

    assert expected == {"supported": 215, "unsupported": 41, "no_claim": 3, "unclear": 5}
