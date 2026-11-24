"""The labelled answers on dropped limits are what their card says they are.

Only the last two tests need the corpus. They build it from the handbook snapshot, as the
commands do, so they are marked slow and are skipped where there is no snapshot, as in CI.
That a passage still reads as it did is also checked by the command, when it loads the
file for a judge check.
"""

import json
import re
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.agreement import measure_agreement
from handbook_rag.evals.judge import Judgement, StatementVerdict
from handbook_rag.evals.judging import labelled_pairs
from handbook_rag.evals.support import SUPPORT, load_labelled_cases, passage_hash
from handbook_rag.ingest.build import build_corpus
from handbook_rag.ingest.models import Chunk

ROOT = Path(__file__).parents[2]
CASES = ROOT / "evals" / "judge" / "limits" / "cases.jsonl"
CARD = ROOT / "evals" / "judge" / "limits" / "README.md"
FIRST_SET = ROOT / "evals" / "judge" / "support" / "cases.jsonl"
SNAPSHOT = ROOT / "data" / "handbook"

KINDS = {"kept": 40, "dropped": 40, "harmless": 40}
LIMIT_KINDS = {
    "who": 5,
    "where": 7,
    "when": 5,
    "how_much": 6,
    "unless": 6,
    "only_if": 7,
    "what": 4,
}
SECTIONS = {
    "total-rewards": 6,
    "finance": 4,
    "security": 4,
    "engineering": 3,
    "marketing": 3,
    "people-group": 3,
    "sales": 3,
    "company": 2,
    "entity": 2,
    "eta": 2,
    "solutions-architects": 2,
    "support": 2,
    "customer-experience": 1,
    "legal": 1,
    "people-policies": 1,
    "product-development": 1,
}
LEFT_OUT = {"example": 4, "reason": 3, "detail": 3, "other_rule": 30}
# What a judge is given, and so all a second reader may be shown.
SHOWN_OF_AN_ANSWER = ("case", "question", "passages")
SHOWN_OF_A_STATEMENT = ("number", "text", "cites")
# Words that would make a dropped limit plain to see. A statement that leaves a limit out
# may hold one only if its passage does: the limit is to be absent, not denied.
WIDENING = re.compile(
    r"\b(all|any|anyone|every|everyone|always|never|whoever|whatever|regardless|"
    r"under no circumstance)\b",
    re.IGNORECASE,
)


def cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").split("\n") if line]


def of_kind(kind: str) -> list[dict[str, Any]]:
    return [case for case in cases() if case["kind"] == kind]


def by_passage() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases():
        groups.setdefault(case["passages"][0]["chunk_id"], []).append(case)
    return groups


def words_of(case: dict[str, Any]) -> str:
    """Everything of an answer that is written out for a judge, in lower case."""
    statements = " ".join(statement["text"] for statement in case["statements"])
    return f"{case['question']} {statements}".casefold()


def shown_to_a_reader(case: dict[str, Any]) -> dict[str, Any]:
    """An answer without its labels, its kind and the notes on it."""
    shown = {key: case[key] for key in SHOWN_OF_AN_ANSWER}
    shown["statements"] = [
        {key: statement[key] for key in SHOWN_OF_A_STATEMENT} for statement in case["statements"]
    ]
    return shown


def test_every_answer_has_what_a_judge_check_reads() -> None:
    ids = [case["case"] for case in cases()]

    # One real request for each answer: the file is kept to 120.
    assert len(ids) == len(set(ids)) == 120
    for case in cases():
        assert case["question"].strip(), case["case"]
        assert case["statements"], case["case"]
        numbers = [statement["number"] for statement in case["statements"]]
        assert numbers == list(range(1, len(numbers) + 1)), case["case"]
        # Every answer cites one passage, and every statement cites it.
        (passage,) = case["passages"]
        assert passage["chunk_id"], case["case"]
        assert re.fullmatch(r"[0-9a-f]{16}", passage["text_sha256"]), case["case"]
        for statement in case["statements"]:
            assert statement["text"].strip(), case["case"]
            assert statement["expected"] in {"supported", "unsupported"}, case["case"]
            assert statement["cites"] == [passage["number"]], case["case"]


def test_the_answers_are_in_the_numbers_the_card_gives() -> None:
    kept = of_kind("kept")

    assert Counter(case["kind"] for case in cases()) == KINDS
    assert Counter(case["limit_kind"] for case in kept) == LIMIT_KINDS
    assert Counter(case["section"] for case in kept) == SECTIONS
    assert len(SECTIONS) == 16
    assert Counter(case["limit_where"] for case in kept) == {"text": 37, "title": 3}
    assert Counter(case["kept_in"] for case in kept) == {"statement": 36, "question": 4}
    assert Counter(case["left_out_kind"] for case in of_kind("harmless")) == LEFT_OUT
    expected = Counter(
        statement["expected"] for case in cases() for statement in case["statements"]
    )
    assert expected == {"supported": 157, "unsupported": 40}


def test_the_card_gives_the_counts_of_the_file() -> None:
    # A row of a table in the card: a name in code marks first, a count last.
    rows = re.findall(
        r"^\| `([^`]+)` \|(?:.*\|)? (\d+) \|$", CARD.read_text(encoding="utf-8"), re.MULTILINE
    )
    card = {name: int(count) for name, count in rows}

    assert len(card) == len(rows), "a name is counted twice in the card"
    assert card == {**KINDS, **LIMIT_KINDS, **SECTIONS, **LEFT_OUT}


def test_each_passage_has_one_answer_of_each_kind_about_the_same_limit() -> None:
    groups = by_passage()

    assert len(groups) == 40
    for chunk_id, answers in groups.items():
        assert sorted(case["kind"] for case in answers) == sorted(KINDS), chunk_id
        for field in ("section", "limit_kind", "limit", "limit_where", "passages"):
            assert len({json.dumps(case[field]) for case in answers}) == 1, (chunk_id, field)


def test_the_labels_follow_from_the_kinds_and_one_reader_gave_them() -> None:
    for case in cases():
        labels = [statement["expected"] for statement in case["statements"]]
        planted = [statement for statement in case["statements"] if "planted" in statement]
        for statement in case["statements"]:
            # One session labelled the file. A second labelling is planned, not made.
            assert statement["readers"] == {"first": statement["expected"]}, case["case"]
        if case["kind"] != "dropped":
            assert set(labels) == {"supported"}, case["case"]
            assert planted == [], case["case"]
            continue
        # Exactly one statement drops the limit, and nothing else in the answer is wrong.
        assert labels.count("unsupported") == 1, case["case"]
        (statement,) = planted
        assert statement["expected"] == "unsupported", case["case"]
        # Named by the kind of limit, so that a check counts what was caught for each.
        assert statement["planted"] == case["limit_kind"], case["case"]
        assert statement["change"].strip(), case["case"]


def test_every_answer_says_what_it_is_and_why() -> None:
    for case in cases():
        assert case["limit"].strip(), case["case"]
        assert case["limit_where"] in {"text", "title"}, case["case"]
        assert case["limit_kind"] in LIMIT_KINDS, case["case"]
        # One sentence, and long enough to say something.
        why = case["why"]
        assert len(why.split()) >= 8, case["case"]
        assert why.endswith("."), case["case"]
        assert not re.search(r"[.!?] [A-Z]", why), case["case"]
        assert ("kept_in" in case) == (case["kind"] == "kept"), case["case"]
        assert ("left_out" in case) == (case["kind"] == "harmless"), case["case"]
        assert ("left_out_kind" in case) == (case["kind"] == "harmless"), case["case"]


def test_a_dropped_answer_does_not_hold_the_limit_nor_a_harmless_one_what_it_leaves_out() -> None:
    for case in of_kind("dropped"):
        assert case["limit"].casefold() not in words_of(case), case["case"]
    for case in of_kind("harmless"):
        assert case["left_out"].casefold() not in words_of(case), case["case"]


def test_no_two_answers_are_the_same() -> None:
    read = [
        (case["question"], tuple(statement["text"] for statement in case["statements"]))
        for case in cases()
    ]
    assert len(set(read)) == 120

    # One pair is the same statement under two questions, as its card says: the limit is
    # in the question of the kept answer, and in neither part of the dropped one.
    by_statements = Counter(statements for _, statements in read)
    (twice,) = [statements for statements, count in by_statements.items() if count > 1]
    pair = [case for case in cases() if tuple(s["text"] for s in case["statements"]) == twice]
    assert sorted(case["kind"] for case in pair) == ["dropped", "kept"]
    assert len({case["passages"][0]["chunk_id"] for case in pair}) == 1
    (kept,) = [case for case in pair if case["kind"] == "kept"]
    assert kept["kept_in"] == "question"


def test_nothing_a_reader_is_shown_tells_the_kind_of_an_answer() -> None:
    kinds_and_labels = {*KINDS, *LEFT_OUT, "supported", "unsupported", "planted"}
    ordered = cases()

    assert [case["case"] for case in ordered] == [f"lim-{n:03d}" for n in range(1, 121)]
    for case in ordered:
        shown = shown_to_a_reader(case)
        # What is left is what a judge reads, and all of it.
        assert set(shown) == {"case", "question", "passages", "statements"}, case["case"]
        # Apart from the words of the question and the answer, nothing in it carries a
        # kind or a label: not the id, not the passage, not a number.
        shown["question"] = ""
        for statement in shown["statements"]:
            statement["text"] = ""
        rest = json.dumps(shown).casefold()
        assert not any(word in rest for word in kinds_and_labels), case["case"]

    # The order tells nothing either: the three answers of a passage are apart, no kind
    # comes four times running, and the kinds come in every one of their six orders.
    place: dict[str, list[int]] = {}
    for index, case in enumerate(ordered):
        place.setdefault(case["passages"][0]["chunk_id"], []).append(index)
    assert min(b - a for found in place.values() for a, b in pairwise(found)) >= 5
    kinds = [case["kind"] for case in ordered]
    assert all(len(set(kinds[i : i + 4])) > 1 for i in range(len(kinds) - 3))
    orders = {tuple(case["kind"] for case in answers) for answers in by_passage().values()}
    assert len(orders) == 6


def test_no_passage_is_one_the_first_set_already_uses() -> None:
    first = {
        passage["chunk_id"]
        for line in FIRST_SET.read_text(encoding="utf-8").split("\n")
        if line
        for passage in json.loads(line)["passages"]
    }

    assert not first & set(by_passage())


@pytest.fixture(scope="module")
def corpus() -> Mapping[str, Chunk]:
    # Built from the snapshot, as the commands build it. An export of the chunks that
    # was made before the chunking last changed must not be able to stand in for it.
    return {
        chunk.chunk_id: chunk for chunk in build_corpus(Settings(data_dir=ROOT / "data")).chunks
    }


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_every_passage_reads_as_it_did_and_holds_the_words_the_file_quotes(
    corpus: Mapping[str, Chunk],
) -> None:
    for case in cases():
        (passage,) = case["passages"]
        assert passage["chunk_id"] in corpus, f"{case['case']}: the passage is gone"
        chunk = corpus[passage["chunk_id"]]
        assert passage_hash(chunk.text) == passage["text_sha256"], case["case"]
        assert case["section"] == chunk.section, case["case"]
        # The limit is given in the passage's own words. A limit said to be in the title
        # is in the title and nowhere in the text.
        if case["limit_where"] == "title":
            assert case["limit"] in chunk.context_header, case["case"]
            assert case["limit"] not in chunk.text, case["case"]
        else:
            assert case["limit"] in chunk.text, case["case"]
        if case["kind"] == "harmless":
            assert case["left_out"] in chunk.text, case["case"]
        # Every number an answer gives is in its passage or in its question.
        read = f"{chunk.context_header} {chunk.text} {case['question']}"
        for statement in case["statements"]:
            for number in re.findall(r"\d[\d,.]*\d|\d", statement["text"]):
                assert number in read, (case["case"], number)
            if statement["expected"] == "unsupported":
                for word in WIDENING.findall(statement["text"]):
                    in_passage = re.search(rf"\b{re.escape(word)}\b", chunk.text, re.IGNORECASE)
                    assert in_passage, (case["case"], word)


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_the_judge_check_reads_the_file_as_it_is(corpus: Mapping[str, Chunk]) -> None:
    labelled = load_labelled_cases(CASES, corpus)

    assert len(labelled) == 120
    assert len({answer.case.fingerprint for answer in labelled}) == 120
    # A judge that passes everything: what the check would print for it.
    lenient = {
        answer.case.case_id: Judgement(
            answer.case.case_id,
            tuple(StatementVerdict(number, "supported", "") for number in answer.expected),
            "none",
        )
        for answer in labelled
    }
    agreement = measure_agreement(
        labelled_pairs(labelled, lenient), verdicts=SUPPORT.verdicts, faults=SUPPORT.faults
    )
    assert agreement.statements == 197
    assert agreement.same_decision.mean == pytest.approx(157 / 197)
    assert agreement.recall is not None
    assert (agreement.recall.mean, agreement.recall.count) == (0.0, 40)
    assert agreement.caught == {kind: (0, count) for kind, count in sorted(LIMIT_KINDS.items())}
