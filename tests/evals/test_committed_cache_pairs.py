"""The cache question pairs in the repository are what their card says they are.

The loader already refuses a pair that could not be measured: an unknown golden question,
a golden question that is not word for word, labels that do not fit the pair's kind, an
``agreed`` that its second label contradicts. These tests hold the file to the rest of
its card: how the pairs were made, and every count the card gives.

One thing the card says is not checked here. The second labelling was made under neutral
names, in a file of its own, and joined to the pairs through a key. Neither that file nor
the key is in the repository, so that ``second`` is what the second session wrote rests
on the check that was run when the file was built.
"""

import difflib
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from handbook_rag.evals.cache_pairs import (
    DIFFERENT,
    HOWS,
    SAME,
    WHAT_DIFFERS,
    CachePair,
    agreed_pairs,
    load_pairs,
)
from handbook_rag.evals.dataset import GoldenQuestion, load_golden_set
from tests.test_readme_results import table

ROOT = Path(__file__).parents[2]
PAIRS = ROOT / "evals" / "cache" / "pairs.jsonl"
CARD = ROOT / "evals" / "cache" / "README.md"
FIELDS = [
    "pair",
    "kind",
    "question_id",
    "question",
    "other",
    "other_id",
    "how",
    "what_differs",
    "source",
    "note",
    "second",
    "agreed",
]
# Wording that would point at the handbook as a source, which no asker would use.
BANNED = ("handbook", "according to", "as per", "the passage", "this section")


def golden() -> dict[str, GoldenQuestion]:
    questions = load_golden_set(ROOT / "evals" / "golden" / "questions.jsonl")
    return {question.id: question for question in questions}


def pairs() -> list[CachePair]:
    return load_pairs(PAIRS, list(golden().values()))


def of_kind(kind: str) -> list[CachePair]:
    return [pair for pair in pairs() if pair.kind == kind]


def rows() -> list[dict[str, Any]]:
    lines = PAIRS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def card() -> str:
    return CARD.read_text(encoding="utf-8")


def tally(values: Iterable[Any]) -> Counter[Any]:
    """A count of labels. They are literal types in the model, and looked up here by name."""
    return Counter(values)


def counts(cells: list[str]) -> list[int]:
    """The whole numbers among the cells of a table row."""
    return [int(cell) for cell in cells if re.fullmatch(r"\d+", cell)]


def name(cell: str) -> str:
    """The name in the first cell of a row: ``reworded`` from `` `reworded` ``."""
    return cell.strip("`*")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def shingles(text: str, length: int = 6) -> set[str]:
    """Every run of six words in a text, for finding wording copied from the handbook."""
    found = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(found[at : at + length]) for at in range(len(found) - length + 1)}


def evidence_shingles() -> set[str]:
    return {
        run
        for question in golden().values()
        for evidence in question.evidence
        for run in shingles(evidence.quote)
    }


def test_the_file_holds_200_pairs_with_the_fields_of_the_card_in_their_order() -> None:
    assert len(rows()) == 200
    for row in rows():
        assert list(row) == FIELDS, row["pair"]
    assert [pair.pair for pair in of_kind(SAME)] == [f"s{n:03d}" for n in range(1, 101)]
    assert [pair.pair for pair in of_kind(DIFFERENT)] == [f"d{n:03d}" for n in range(1, 101)]


def test_the_kinds_are_in_the_numbers_the_card_gives() -> None:
    agreed = tally(pair.kind for pair in agreed_pairs(pairs()))
    found = {name(row[0]): counts(row[1:]) for row in table(card(), "| Kind |")}

    assert found == {SAME: [100, agreed[SAME]], DIFFERENT: [100, agreed[DIFFERENT]]}
    assert (agreed[SAME], agreed[DIFFERENT]) == (98, 100)
    assert "198 agreed pairs (98 `same`, 100 `different`)" in card()


def test_the_ways_of_rewording_are_in_the_numbers_the_card_gives() -> None:
    made = tally(pair.how for pair in of_kind(SAME))
    agreed = tally(pair.how for pair in agreed_pairs(of_kind(SAME)))
    found = {name(row[0]): counts(row[1:]) for row in table(card(), "| `how` | What changes |")}

    assert found == {how: [made[how], agreed[how]] for how in HOWS}
    assert set(made.values()) == {20}


def test_what_differs_is_in_the_numbers_the_card_gives() -> None:
    different = of_kind(DIFFERENT)
    made = tally((pair.what_differs, pair.source) for pair in different)
    agreed = tally(pair.what_differs for pair in agreed_pairs(different))
    found = {name(row[0]): counts(row[1:]) for row in table(card(), "| `what_differs` | What ")}

    expected = {
        what: [made[(what, "golden")], made[(what, "written")], 0, agreed[what]]
        for what in WHAT_DIFFERS
    }
    for what in WHAT_DIFFERS:
        expected[what][2] = expected[what][0] + expected[what][1]
        # No group is so small that it could not be reported on.
        assert expected[what][2] >= 12
    expected["All"] = [sum(row[column] for row in expected.values()) for column in range(4)]
    assert found == expected
    assert expected["All"] == [13, 87, 100, 100]


def test_the_golden_questions_are_used_as_the_card_says() -> None:
    by_id = golden()
    same, different = of_kind(SAME), of_kind(DIFFERENT)
    in_different = {pair.question_id for pair in different} | {
        pair.other_id for pair in different if pair.other_id
    }
    in_any = in_different | {pair.question_id for pair in same}
    uses = tally(pair.question_id for pair in pairs()) + tally(
        pair.other_id for pair in pairs() if pair.other_id
    )
    answerable = sum(question.type == "answerable" for question in by_id.values())

    # Each same pair rewords another golden question, and one that the handbook answers.
    assert len({pair.question_id for pair in same}) == 100
    assert all(by_id[pair.question_id].type == "answerable" for pair in same)
    assert f"(100 of the {answerable})" in card()
    assert "**131 of the 165 golden questions** are in at least one pair" in card()
    assert (len(in_any), len(by_id), len(in_different)) == (131, 165, 84)
    assert "draw on 84 of them" in card()
    assert "The 200 pairs use 131 golden questions, 59 of them more than once" in card()
    assert sum(count > 1 for count in uses.values()) == 59
    # "four questions (q034, q063, b13 and b08) are in 19 pairs"
    four = {"q034", "q063", "b13", "b08"}
    assert sum(bool({pair.question_id, pair.other_id} & four) for pair in pairs()) == 19
    assert (uses["q034"], uses["q063"], uses["b13"]) == (6, 5, 5)


def test_the_pairs_that_involve_an_unanswerable_question_are_the_26_of_the_card() -> None:
    by_id = golden()

    def unanswerable(question_id: str | None) -> bool:
        return question_id is not None and by_id[question_id].type == "unanswerable"

    different = of_kind(DIFFERENT)
    in_golden = [
        pair
        for pair in different
        if pair.source == "golden"
        and (unanswerable(pair.question_id) or unanswerable(pair.other_id))
    ]
    in_written = [
        pair for pair in different if pair.source == "written" and unanswerable(pair.question_id)
    ]

    assert (len(in_golden), len(in_written)) == (6, 20)
    assert "**26 `different` pairs involve a golden question the handbook does not answer.**" in (
        card()
    )
    # No golden pair is of two questions the handbook does not answer: the assistant
    # declines both with the same sentence, so one reply for the other is not wrong.
    assert not any(
        unanswerable(pair.question_id) and unanswerable(pair.other_id) for pair in different
    )


def test_no_other_question_is_repeated_and_no_golden_pair_is_there_twice() -> None:
    normal = tally(" ".join(pair.other.lower().split()) for pair in pairs())
    golden_pairs = tally(
        frozenset((pair.question_id, pair.other_id)) for pair in pairs() if pair.other_id
    )
    golden_texts = {" ".join(q.question.lower().split()) for q in golden().values()}

    assert [text for text, count in normal.items() if count > 1] == []
    assert [sorted(found) for found, count in golden_pairs.items() if count > 1] == []
    # Only the 13 golden pairs ask a golden question, in whatever case or spacing: no
    # written question is a stored question in disguise.
    assert sum(text in golden_texts for text in normal) == 13
    assert sum(pair.other_id is not None for pair in pairs()) == 13


def test_every_note_is_one_sentence_and_says_which_case_a_different_pair_is() -> None:
    starts: Counter[str] = Counter()
    for pair in pairs():
        note = pair.note.strip()
        assert note.endswith("."), pair.pair
        assert not re.search(r"[.!?] [A-Z]", note[:-1]), pair.pair
        if pair.kind == DIFFERENT:
            starts[note.split(":")[0]] += 1

    assert starts == {"Differs": 90, "Not answered": 10}
    assert "another answer (90 pairs)" in card()
    assert "does not answer it (10 pairs)" in card()


def test_no_written_question_points_at_the_handbook_or_copies_a_passage_of_it() -> None:
    evidence = evidence_shingles()
    for pair in pairs():
        written = pair.kind == SAME or pair.source == "written"
        if written:
            assert not [phrase for phrase in BANNED if phrase in pair.other.lower()], pair.pair
            assert not shingles(pair.other) & evidence, pair.pair
        assert not shingles(pair.note) & evidence, pair.pair


def test_the_second_sessions_reasons_repeat_handbook_wording_as_often_as_the_card_says() -> None:
    evidence = evidence_shingles()
    repeating = [
        pair.pair
        for pair in pairs()
        if (shingles(pair.second.reason) | shingles(pair.second.flag or "")) & evidence
    ]

    assert f"in {len(repeating)} of them a run of six or more words" in card()
    assert len(repeating) == 11


def test_the_second_labelling_is_shaped_as_the_card_says_and_agrees_as_it_says() -> None:
    for row in rows():
        assert list(row["second"]) in (
            ["label", "confidence", "basis", "reason"],
            ["label", "confidence", "basis", "reason", "flag"],
        ), row["pair"]
    labels = ["same", "different", "unsure"]
    matrix = tally((pair.kind, pair.second.label) for pair in pairs())
    found = {name(row[0]): counts(row[1:]) for row in table(card(), "| Maker's label |")}

    assert found == {
        SAME: [*(matrix[(SAME, label)] for label in labels), 100],
        DIFFERENT: [*(matrix[(DIFFERENT, label)] for label in labels), 100],
        "All": [
            *(sum(matrix[(kind, label)] for kind in (SAME, DIFFERENT)) for label in labels),
            200,
        ],
    }
    agreed = len(agreed_pairs(pairs()))
    assert f"agree on {agreed} of 200 pairs" in card()
    # Cohen's kappa over the three labels a second session could give.
    observed = agreed / 200
    by_chance = sum(
        sum(matrix[(label, other)] for other in labels)
        * sum(matrix[(kind, label)] for kind in (SAME, DIFFERENT))
        / 200**2
        for label in labels
    )
    kappa = (observed - by_chance) / (1 - by_chance)
    assert f"Cohen's kappa is {kappa:.3f}" in card()
    assert f"{kappa:.3f}" == "0.980"


def test_the_two_disputed_pairs_and_the_five_flags_are_those_of_the_card() -> None:
    disputed = [pair for pair in pairs() if not pair.agreed]
    flagged = sorted(pair.pair for pair in pairs() if pair.second.flag)
    listed = [name(row[0]) for row in table(card(), "| Pair | Its name | Its label | Flag |")]

    assert [pair.pair for pair in disputed] == ["s078", "s100"]
    for pair in disputed:
        assert (pair.kind, pair.how, pair.second.label) == (SAME, "indirect", "different")
        assert pair.second.confidence == "low"
        assert f"`{pair.pair}`" in card()
    assert sorted(listed) == flagged == ["d034", "d053", "d062", "s078", "s100"]


def test_what_the_second_labels_rest_on_is_in_the_numbers_the_card_gives() -> None:
    confidence = tally(pair.second.confidence for pair in pairs())
    agreed = tally(pair.second.confidence for pair in agreed_pairs(pairs()))
    basis = tally(pair.second.basis for pair in pairs())
    by_kind = tally((pair.kind, pair.second.basis) for pair in pairs())
    found_confidence = {name(row[0]): counts(row[1:]) for row in table(card(), "| `confidence` |")}
    found_basis = {name(row[0]): counts(row[1:]) for row in table(card(), "| `basis` |")}

    assert found_confidence == {
        level: [confidence[level], agreed[level]] for level in ("high", "medium", "low")
    }
    assert found_basis == {based: [basis[based]] for based in ("wording", "handbook", "golden")}
    assert f"{by_kind[(DIFFERENT, 'handbook')]} of the {basis['handbook']} labels" in card()
    assert f"({by_kind[(SAME, 'wording')]} of 100)" in card()
    assert (by_kind[(DIFFERENT, "handbook")], basis["handbook"]) == (73, 75)


def test_the_written_questions_change_as_few_words_as_the_card_says() -> None:
    changed: Counter[str] = Counter()
    for pair in of_kind(DIFFERENT):
        if pair.source != "written":
            continue
        before, after = pair.question.split(), pair.other.split()
        blocks = difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        count = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in blocks if tag != "equal")
        changed["one" if count == 1 else "two or three" if count <= 3 else "more"] += 1

    assert changed == {"one": 22, "two or three": 18, "more": 47}
    assert "**22 of the 87 written questions change one word**" in card()
    assert "18 change two or three, and 47 change more" in card()


def test_the_set_is_as_hard_by_word_overlap_as_the_card_says() -> None:
    def overlap(pair: CachePair) -> float:
        first, second = set(words(pair.question)), set(words(pair.other))
        return len(first & second) / len(first | second)

    different = statistics.median(overlap(pair) for pair in of_kind(DIFFERENT))
    same = statistics.median(overlap(pair) for pair in of_kind(SAME))

    # The near-misses share more words with their golden question than the rewordings do.
    assert f"medians of {different:.2f} against {same:.2f}" in card()
    assert (f"{different:.2f}", f"{same:.2f}") == ("0.63", "0.45")


def test_the_card_no_longer_calls_a_check_that_was_done_planned() -> None:
    assert not re.search(r"\bplanned\b", card())
