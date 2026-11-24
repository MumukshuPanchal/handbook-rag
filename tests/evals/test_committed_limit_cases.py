"""The labelled answers on dropped limits are what their card says they are.

Only the last two tests need the corpus. They build it from the handbook snapshot, as the
commands do, so they are marked slow and are skipped where there is no snapshot, as in CI.
That a passage still reads as it did is also checked by the command, when it loads the
file for a judge check.
"""

import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.agreement import Agreement, Pair, measure_agreement
from handbook_rag.evals.judge import VERDICTS, Judgement, StatementVerdict
from handbook_rag.evals.judging import UNCLEAR, labelled_pairs
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
# The faults a check counts. One `when` fault is not among them: the two readers split on
# its statement, which is labelled unclear.
COUNTED_FAULTS = {**LIMIT_KINDS, "when": 4}
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
LABELS = {"supported": 157, "unsupported": 39, UNCLEAR: 1}
# The second labelling: how sure the readers said they were, the one statement on which
# they differ from the first reader, and every label they gave with less than high
# confidence.
CONFIDENCE = {"high": 187, "medium": 9, "low": 1}
SPLIT = ("lim-026", 1)
NOT_SURE = {
    ("lim-026", 1): "medium",
    ("lim-037", 2): "medium",
    ("lim-040", 1): "medium",
    ("lim-066", 1): "low",
    ("lim-083", 1): "medium",
    ("lim-086", 1): "medium",
    ("lim-091", 1): "medium",
    ("lim-102", 1): "medium",
    ("lim-103", 1): "medium",
    ("lim-114", 2): "medium",
}
# What a judge is given, and so all a second reader was shown.
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


def statements() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Every statement, with the answer it belongs to."""
    for case in cases():
        for statement in case["statements"]:
            yield case, statement


def by_passage() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases():
        groups.setdefault(case["passages"][0]["chunk_id"], []).append(case)
    return groups


def words_of(case: dict[str, Any]) -> str:
    """Everything of an answer that is written out for a judge, in lower case."""
    said = " ".join(statement["text"] for statement in case["statements"])
    return f"{case['question']} {said}".casefold()


def shown_to_a_reader(case: dict[str, Any]) -> dict[str, Any]:
    """An answer without its labels, its kind and the notes on it."""
    shown = {key: case[key] for key in SHOWN_OF_AN_ANSWER}
    shown["statements"] = [
        {key: statement[key] for key in SHOWN_OF_A_STATEMENT} for statement in case["statements"]
    ]
    return shown


def between_the_readers() -> Agreement:
    """How far the second labelling agrees with the first, by the code that checks a judge."""
    pairs = [
        Pair(case["case"], statement["number"], *both(statement))
        for case, statement in statements()
    ]
    return measure_agreement(pairs, verdicts=SUPPORT.verdicts, faults=SUPPORT.faults)


def both(statement: dict[str, Any]) -> tuple[str, str]:
    return statement["readers"]["first"], statement["readers"]["second"]


def card() -> str:
    return CARD.read_text(encoding="utf-8")


def card_rows() -> dict[str, list[list[str]]]:
    """The rows of the card's tables: under the text of a first cell, the cells after it."""
    rows: dict[str, list[list[str]]] = {}
    for line in card().splitlines():
        line = line.strip()
        if line.startswith("| ") and line.endswith(" |"):
            first, *rest = (cell.strip() for cell in line[1:-1].split("|"))
            rows.setdefault(first, []).append(rest)
    return rows


def numbers_in(rows: dict[str, list[list[str]]], first: str) -> list[int]:
    """The counts of the one row of the card that begins with this cell."""
    (row,) = rows[first]
    return [int(cell) for cell in row if cell.isdigit()]


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
            assert statement["expected"] in LABELS, case["case"]
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
    assert Counter(statement["expected"] for _, statement in statements()) == LABELS
    # What a check counts as caught, for each kind of limit.
    planted = Counter(s["planted"] for _, s in statements() if "planted" in s)
    assert planted == COUNTED_FAULTS


def test_the_card_gives_the_counts_of_the_file() -> None:
    rows = card_rows()
    kept = of_kind("kept")
    labels = Counter(statement["expected"] for _, statement in statements())

    assert numbers_in(rows, "Passages of the handbook") == [len(by_passage())]
    assert numbers_in(rows, "Answers, three for each passage") == [len(cases())]
    assert numbers_in(rows, "Statements in all") == [sum(labels.values())]
    assert numbers_in(rows, "labelled `supported`") == [labels["supported"]]
    assert numbers_in(rows, "labelled `unsupported`") == [labels["unsupported"]]
    assert numbers_in(rows, "labelled `unclear`, and not counted") == [labels[UNCLEAR]]
    for kind, count in Counter(case["kind"] for case in cases()).items():
        assert numbers_in(rows, f"`{kind}`") == [count]
    # For a kind of limit: its passages, and its faults that a check counts.
    counted = Counter(s["planted"] for _, s in statements() if "planted" in s)
    for kind, count in Counter(case["limit_kind"] for case in kept).items():
        assert numbers_in(rows, f"`{kind}`") == [count, counted[kind]]
    for section, count in Counter(case["section"] for case in kept).items():
        assert numbers_in(rows, f"`{section}`") == [count]
    for left_out, count in Counter(case["left_out_kind"] for case in of_kind("harmless")).items():
        assert numbers_in(rows, f"`{left_out}`") == [count]


def test_each_passage_has_one_answer_of_each_kind_about_the_same_limit() -> None:
    groups = by_passage()

    assert len(groups) == 40
    for chunk_id, answers in groups.items():
        assert sorted(case["kind"] for case in answers) == sorted(KINDS), chunk_id
        for field in ("section", "limit_kind", "limit", "limit_where", "passages"):
            assert len({json.dumps(case[field]) for case in answers}) == 1, (chunk_id, field)


def test_the_first_labels_follow_from_what_each_answer_was_written_to_be() -> None:
    for case in cases():
        first = [statement["readers"]["first"] for statement in case["statements"]]
        left_out = [statement for statement in case["statements"] if "change" in statement]
        if case["kind"] != "dropped":
            assert set(first) == {"supported"}, case["case"]
            assert left_out == [], case["case"]
            continue
        # Exactly one statement leaves the limit out, and nothing else in the answer is
        # wrong. The note on what it leaves out stays whatever the second reader said.
        assert first.count("unsupported") == 1, case["case"]
        (statement,) = left_out
        assert statement["readers"]["first"] == "unsupported", case["case"]
        assert statement["change"].strip(), case["case"]


def test_a_label_is_what_both_readers_gave_and_unclear_where_they_differ() -> None:
    differ = []
    for case, statement in statements():
        assert set(statement["readers"]) == {"first", "second"}, case["case"]
        first, second = both(statement)
        assert {first, second} <= set(VERDICTS), case["case"]
        if first == second:
            assert statement["expected"] == first, case["case"]
        else:
            # No third reader settled it: both labels stay, and it is not counted.
            assert statement["expected"] == UNCLEAR, case["case"]
            differ.append((case["case"], statement["number"], first, second))
        # Only a fault both readers found is one a check counts as caught or missed, and
        # it is counted under its kind of limit. The loader refuses any other planted one.
        counted = statement["expected"] == "unsupported"
        assert ("planted" in statement) == counted, case["case"]
        if counted:
            assert statement["planted"] == case["limit_kind"], case["case"]

    assert differ == [(*SPLIT, "unsupported", "supported")]


def test_the_second_labelling_is_in_the_numbers_the_card_gives() -> None:
    agreement = between_the_readers()

    assert agreement.matrix["supported"] == {"supported": 157, "unsupported": 0, "no_claim": 0}
    assert agreement.matrix["unsupported"] == {"supported": 1, "unsupported": 39, "no_claim": 0}
    assert (agreement.same_verdict.count, agreement.statements) == (197, 197)
    assert agreement.same_verdict.mean == pytest.approx(196 / 197)
    assert agreement.kappa == pytest.approx(0.984, abs=0.0005)
    confidence = Counter(statement["second_confidence"] for _, statement in statements())
    assert confidence == CONFIDENCE
    not_sure = {
        (case["case"], statement["number"]): statement["second_confidence"]
        for case, statement in statements()
        if statement["second_confidence"] != "high"
    }
    assert not_sure == NOT_SURE


def test_no_reader_saw_two_answers_about_one_passage() -> None:
    assert Counter(case["second_session"] for case in cases()) == {1: 40, 2: 40, 3: 40}
    # The answers were dealt by their place in the file: of the three answers of a
    # passage, the first to reader 1, the second to reader 2, the third to reader 3.
    for chunk_id, answers in by_passage().items():
        assert [case["second_session"] for case in answers] == [1, 2, 3], chunk_id
    for session in (1, 2, 3):
        read = {case["kind"] for case in cases() if case["second_session"] == session}
        assert read == set(KINDS), session


def test_the_card_gives_the_second_labelling_as_the_file_records_it() -> None:
    rows, text = card_rows(), card()
    agreement = between_the_readers()
    same = round(agreement.same_verdict.mean * agreement.statements)

    # The headline, and first reader against second.
    assert f"agree on {same} of {agreement.statements} statements" in text
    assert f"({same / agreement.statements:.3f}, Cohen's kappa {agreement.kappa:.3f})" in text
    for first in ("supported", "unsupported"):
        across = agreement.matrix[first]
        assert numbers_in(rows, f"`{first}`") == [across["supported"], across["unsupported"]]
    # By kind of answer: statements, statements alike, answers, answers alike throughout.
    for kind in KINDS:
        answers = of_kind(kind)
        alike = [[len(set(both(s))) == 1 for s in case["statements"]] for case in answers]
        assert numbers_in(rows, f"In `{kind}` answers") == [
            sum(map(len, alike)),
            sum(map(sum, alike)),
            len(answers),
            sum(map(all, alike)),
        ]
    # By reader session: answers, statements, statements given the first reader's label.
    for session in (1, 2, 3):
        read = [s for case, s in statements() if case["second_session"] == session]
        assert numbers_in(rows, f"Reader {session}") == [
            sum(case["second_session"] == session for case in cases()),
            len(read),
            sum(len(set(both(s))) == 1 for s in read),
        ]
    for confidence, count in CONFIDENCE.items():
        assert numbers_in(rows, f"`{confidence}`") == [count]
    # Every label given with less than high confidence is listed, and no other: the case,
    # then the statement, the confidence and the second reader's label.
    listed = [
        (first, *row[:3])
        for first, found in rows.items()
        for row in found
        if len(row) > 2 and row[1] in {"medium", "low"}
    ]
    assert sorted(listed) == sorted(
        (f"`{case['case']}`", str(s["number"]), s["second_confidence"], s["readers"]["second"])
        for case, s in statements()
        if s["second_confidence"] != "high"
    )
    assert f"statement {SPLIT[1]} of `{SPLIT[0]}`" in text


def test_the_card_gives_the_fixed_points_of_a_check() -> None:
    text = card()
    labels = Counter(statement["expected"] for _, statement in statements())
    counted = labels["supported"] + labels["unsupported"]
    faults = labels["unsupported"]
    # An answer counts as faulty by its labels when a counted statement of it is a fault.
    faulty = sum(
        any(statement["expected"] == "unsupported" for statement in case["statements"])
        for case in cases()
    )

    assert f"compares {counted} statements" in text
    assert f"agrees on {labels['supported']} of the {counted}" in text
    assert f"({labels['supported'] / counted:.3f})" in text
    assert f"on {len(cases()) - faulty} of {len(cases())} answers" in text
    assert f"catches none of the {faults}" in text
    assert f"catches all {faults}, and {faults} of its {counted} alarms are right" in text
    assert f"({faults / counted:.3f})" in text


def test_the_dropped_answers_the_first_reader_doubted_are_listed_with_what_came_of_them() -> None:
    # The table of the card with two cells after the cases: the doubt, and the outcome.
    listed = [
        (case_id, row[1])
        for first, found in card_rows().items()
        if first.startswith("`lim-")
        for row in found
        if len(row) == 2
        for case_id in re.findall(r"lim-\d{3}", first)
    ]
    second = {
        case["case"]: statement["readers"]["second"]
        for case, statement in statements()
        if "change" in statement
    }

    assert len(listed) == len({case_id for case_id, _ in listed}) == 11
    for case_id, outcome in listed:
        # The card says of each what its second reader called it.
        assert outcome.casefold().startswith(second[case_id]), case_id
    assert sorted(second[case_id] for case_id, _ in listed) == ["supported", *["unsupported"] * 10]
    assert [case_id for case_id, _ in listed if second[case_id] == "supported"] == [SPLIT[0]]


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
    by_statements = Counter(said for _, said in read)
    (twice,) = [said for said, count in by_statements.items() if count > 1]
    pair = [case for case in cases() if tuple(s["text"] for s in case["statements"]) == twice]
    assert sorted(case["kind"] for case in pair) == ["dropped", "kept"]
    assert len({case["passages"][0]["chunk_id"] for case in pair}) == 1
    (kept,) = [case for case in pair if case["kind"] == "kept"]
    assert kept["kept_in"] == "question"


def test_nothing_a_reader_is_shown_tells_the_kind_of_an_answer() -> None:
    telling = {*KINDS, *LEFT_OUT, *LABELS, *CONFIDENCE, "planted"}
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
        assert not any(word in rest for word in telling), case["case"]

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
            if "change" in statement:
                for word in WIDENING.findall(statement["text"]):
                    in_passage = re.search(rf"\b{re.escape(word)}\b", chunk.text, re.IGNORECASE)
                    assert in_passage, (case["case"], word)


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_the_judge_check_reads_the_file_as_it_is(corpus: Mapping[str, Chunk]) -> None:
    labelled = load_labelled_cases(CASES, corpus)

    assert len(labelled) == 120
    assert len({answer.case.fingerprint for answer in labelled}) == 120
    unclear = [label for answer in labelled for label in answer.expected.values()]
    assert unclear.count(UNCLEAR) == 1
    # A judge that passes everything: what the check would print for it. The statement
    # whose label is unclear is left out, and its answer counts through its other one.
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
    assert agreement.statements == 196
    assert agreement.same_decision.mean == pytest.approx(157 / 196)
    assert agreement.recall is not None
    assert (agreement.recall.mean, agreement.recall.count) == (0.0, 39)
    assert agreement.caught == {kind: (0, count) for kind, count in sorted(COUNTED_FAULTS.items())}
    assert agreement.answers == 120
    assert agreement.same_answer.mean == pytest.approx(81 / 120)


def test_the_second_labels_in_the_cases_are_the_ones_the_readers_gave() -> None:
    # The readers' own lines, with their reasons, are kept beside the cases, so that the
    # second label of a statement can be traced to the reader who gave it.
    lines = (CASES.parent / "second-labelling.jsonl").read_text(encoding="utf-8").splitlines()
    given = {(row["case"], row["statement"]): row for row in map(json.loads, lines)}
    assert len(lines) == len(given) == sum(1 for _ in statements())
    for case, statement in statements():
        row = given[(case["case"], statement["number"])]
        assert row["label"] == statement["readers"]["second"], case["case"]
        assert row["confidence"] == statement["second_confidence"], case["case"]
        assert row["session"] == case["second_session"], case["case"]
        assert row["reason"].strip(), case["case"]
