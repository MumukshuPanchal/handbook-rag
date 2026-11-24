"""The cache measurement: the pairs file, the split, the threshold rule and the simulated cache.

No model is loaded. An embedding model and a verifier that read from tables stand in for
the real ones, so each test can say exactly how near two questions are.
"""

import itertools
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.cache.verify import Verification
from handbook_rag.evals import cache_pairs
from handbook_rag.evals.cache_pairs import (
    CHOOSING,
    COSINE,
    DIFFERENT,
    ENTAILS_MEAN,
    ENTAILS_MIN,
    MISSED,
    NOT_SERVED,
    POLICIES,
    REPORTING,
    RIGHT,
    SAME,
    WRONG,
    Asked,
    CachePair,
    Candidate,
    Scores,
    SecondLabel,
    area_under_curve,
    choose_policy,
    choose_threshold,
    load_cache_run,
    load_pairs,
    measure,
    outcome,
    served,
    split_pairs,
    summarise,
    tally,
    write_cache_run,
)
from handbook_rag.evals.cache_report import report_lines
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.metrics import share_with_interval
from tests.cache.support import FakeVerifier
from tests.support import FakeEmbedder


def golden_question(question_id: str, text: str) -> GoldenQuestion:
    return GoldenQuestion(
        id=question_id,
        question=text,
        type="unanswerable",
        origin="hand-written",
        unanswerable_kind="near-miss",
    )


GOLDEN = [
    golden_question("q1", "How many weeks of parental leave do I get in the US?"),
    golden_question("q2", "How many weeks of parental leave does GitHub give?"),
    golden_question("q3", "Who approves a laptop purchase?"),
    golden_question("q4", "Where is the incident status published?"),
]
TEXT = {question.id: question.question for question in GOLDEN}


def same_row(pair: str = "s001", question_id: str = "q1", **changes: Any) -> dict[str, Any]:
    """A ``same`` pair as a line of the file holds it."""
    row: dict[str, Any] = {
        "pair": pair,
        "kind": "same",
        "question_id": question_id,
        "question": TEXT[question_id],
        "other": f"a rewording of {question_id} for {pair}",
        "other_id": None,
        "how": "reworded",
        "what_differs": None,
        "source": None,
        "note": "The same answer is right.",
        "second": {"label": "same", "confidence": "high", "basis": "wording", "reason": "Same."},
        "agreed": True,
    }
    return {**row, **changes}


def different_row(pair: str = "d001", question_id: str = "q1", **changes: Any) -> dict[str, Any]:
    """A written ``different`` pair as a line of the file holds it."""
    row = same_row(
        pair,
        question_id,
        kind="different",
        other=f"a near miss of {question_id} for {pair}",
        how=None,
        what_differs="region",
        source="written",
        note="Differs: another country.",
    )
    row["second"] = {**row["second"], "label": "different"}
    return {**row, **changes}


def golden_row(pair: str, question_id: str, other_id: str, **changes: Any) -> dict[str, Any]:
    """A ``different`` pair whose other question is itself a golden question."""
    return different_row(
        pair, question_id, other=TEXT[other_id], other_id=other_id, source="golden", **changes
    )


def as_pair(row: dict[str, Any]) -> CachePair:
    return CachePair.model_validate(row)


def write_pairs(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def scores(cosine: float, forward: float | None = None, backward: float | None = None) -> Scores:
    forward = cosine if forward is None else forward
    return Scores(cosine, Verification(forward, forward if backward is None else backward))


def ask(row: dict[str, Any], *candidates: tuple[str, Scores], own: Scores | None = None) -> Asked:
    """A pair's other question with the candidates a cache found for it, nearest first."""
    return Asked(
        pair=as_pair(row),
        own=own or scores(0.0),
        candidates=tuple(Candidate(question_id, found) for question_id, found in candidates),
        above_floor=len(candidates),
        excluded=row["other_id"],
    )


# --- the pairs file ---------------------------------------------------------------------


def test_a_file_of_valid_pairs_is_loaded_in_its_order(tmp_path: Path) -> None:
    rows = [same_row(), different_row(), golden_row("d002", "q1", "q2")]

    pairs = load_pairs(write_pairs(tmp_path / "pairs.jsonl", rows), GOLDEN)

    assert [pair.pair for pair in pairs] == ["s001", "d001", "d002"]
    assert [pair.group for pair in pairs] == ["reworded", "region", "region"]
    assert pairs[2].other_id == "q2"
    assert pairs[0].second == SecondLabel(
        label="same", confidence="high", basis="wording", reason="Same."
    )


def test_a_flag_of_the_second_labelling_is_kept_and_blank_lines_are_passed_over(
    tmp_path: Path,
) -> None:
    row = same_row()
    row["second"]["flag"] = "Depends on the direction."
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

    (pair,) = load_pairs(path, GOLDEN)

    assert pair.second.flag == "Depends on the direction."


def without(row: dict[str, Any], field: str) -> dict[str, Any]:
    return {name: value for name, value in row.items() if name != field}


def second(row: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**row, "second": {**row["second"], **changes}}


MALFORMED = {
    "a field left out": (without(same_row(), "how"), "how"),
    "a field that does not exist": ({**same_row(), "hwo": "reworded"}, "hwo"),
    "an unknown kind": (same_row(kind="similar"), "kind"),
    "an id of another shape": (same_row(pair="same-1"), "pair"),
    "an id that says another kind": (same_row(pair="d001"), "is labelled same"),
    "a golden question that does not exist": (
        same_row(question_id="q1") | {"question_id": "q9"},
        "q9",
    ),
    "a golden question reworded": (same_row(question="How many weeks of leave?"), "word for word"),
    "a blank other question": (same_row(other="   "), "blank"),
    "a blank note": (same_row(note=" "), "blank"),
    "the golden question asked as the other": (same_row(other=TEXT["q1"]), "itself"),
    "an unknown way of rewording": (same_row(how="shouted"), "how"),
    "a same pair that says what differs": (same_row(what_differs="region"), "is a same pair"),
    "a same pair with a source": (same_row(source="written"), "is a same pair"),
    "a same pair without a way": (same_row(how=None), "is a same pair"),
    "a different pair with a way": (different_row(how="short"), "is a different pair"),
    "a different pair that differs in nothing": (
        different_row(what_differs=None),
        "is a different pair",
    ),
    "a different pair without a source": (different_row(source=None), "is a different pair"),
    "a golden pair without the other id": (different_row(source="golden"), "other_id"),
    "a written pair with an other id": (different_row(other_id="q2"), "other_id"),
    "a written question that is a golden question": (different_row(other=TEXT["q3"]), "q3"),
    "an other id that does not exist": (golden_row("d002", "q1", "q2") | {"other_id": "q9"}, "q9"),
    "an other id that is the question itself": (
        golden_row("d002", "q1", "q2") | {"other_id": "q1"},
        "no other golden question",
    ),
    "an other question that is not its id's": (
        golden_row("d002", "q1", "q2") | {"other_id": "q3"},
        "word for word",
    ),
    "agreed that the second label contradicts": (same_row(agreed=False), "agreed is false"),
    "a disagreement marked as agreed": (
        second(same_row(), label="different"),
        "agreed is true",
    ),
    "agreed that is not true or false": (same_row(agreed="true"), "agreed"),
    "agreed given as a number": (same_row(agreed=1), "agreed"),
    "an unknown second label": (second(same_row(), label="maybe"), "label"),
    "a second labelling without its reason": (
        {**same_row(), "second": without(same_row()["second"], "reason")},
        "reason",
    ),
    "a second labelling that is not an object": ({**same_row(), "second": "same"}, "second"),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_pair_is_refused_with_its_line(name: str, tmp_path: Path) -> None:
    row, said = MALFORMED[name]
    path = write_pairs(tmp_path / "pairs.jsonl", [different_row("d009"), row])

    with pytest.raises(DatasetError) as refused:
        load_pairs(path, GOLDEN)

    assert f"{path}:2:" in str(refused.value)
    assert said in str(refused.value)


def test_a_line_that_is_not_json_or_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    for line in ('{"pair": "s001"', '["s001", "same"]'):
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="not a cache question pair"):
            load_pairs(path, GOLDEN)


def test_an_id_used_twice_is_refused(tmp_path: Path) -> None:
    path = write_pairs(tmp_path / "pairs.jsonl", [same_row(), same_row(other="another wording")])

    with pytest.raises(DatasetError, match="s001 is used twice"):
        load_pairs(path, GOLDEN)


def test_a_missing_or_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_pairs(tmp_path / "none.jsonl", GOLDEN)
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="holds no cache question pair"):
        load_pairs(tmp_path / "empty.jsonl", GOLDEN)


# --- the split --------------------------------------------------------------------------


def golden_ids(pairs: Sequence[CachePair]) -> set[str]:
    return {pair.question_id for pair in pairs} | {p.other_id for p in pairs if p.other_id}


def test_no_golden_question_is_on_both_sides_also_through_a_golden_pair() -> None:
    # q1 and q2 are tied by a golden pair, and q2 and q3 by another: one group of three
    # golden questions, though no pair names q1 and q3 together.
    rows = [
        same_row("s001", "q1"),
        different_row("d001", "q1"),
        golden_row("d002", "q1", "q2"),
        same_row("s002", "q2"),
        golden_row("d003", "q3", "q2"),
        different_row("d004", "q3"),
        same_row("s003", "q4"),
        different_row("d005", "q4"),
    ]
    pairs = [as_pair(row) for row in rows]

    for seed in ("a", "b", "c", "d", "e", "f"):
        split = split_pairs(pairs, seed=seed)
        choosing = [pair for pair in pairs if pair.pair in split.choosing]
        reporting = [pair for pair in pairs if pair.pair in split.reporting]

        assert not golden_ids(choosing) & golden_ids(reporting), seed
        assert split.choosing | split.reporting == {pair.pair for pair in pairs}
        assert not split.choosing & split.reporting
        assert len({split.groups[pair] for pair in ("s001", "d001", "d002", "s002", "d003")}) == 1
        assert split.groups["s003"] != split.groups["s001"]


def test_the_split_is_the_same_each_time_and_another_seed_gives_another() -> None:
    # Forty pairs, each with a golden question of its own, so forty groups to deal.
    pairs = [
        as_pair(same_row(f"s{n:03d}") | {"question_id": f"g{n}", "question": f"question {n}"})
        for n in range(40)
    ]

    assert split_pairs(pairs) == split_pairs(pairs)
    # The order of the file does not matter, only the seed and the ids.
    assert split_pairs(pairs) == split_pairs(list(reversed(pairs)))
    assert len({split_pairs(pairs, seed=str(seed)).choosing for seed in range(5)}) == 5
    assert split_pairs(pairs, seed=cache_pairs.SPLIT_SEED) == split_pairs(pairs)


def test_the_halves_hold_as_many_pairs_of_each_kind_as_the_groups_allow() -> None:
    rows = []
    for n in range(30):
        question = {"question_id": f"g{n}", "question": f"question {n}"}
        rows.append(same_row(f"s{n:03d}") | question)
        rows.append(different_row(f"d{n:03d}") | question)
    pairs = [as_pair(row) for row in rows]

    split = split_pairs(pairs)

    for half in (split.choosing, split.reporting):
        members = [pair for pair in pairs if pair.pair in half]
        assert sum(pair.kind == SAME for pair in members) == 15
        assert sum(pair.kind == DIFFERENT for pair in members) == 15
    assert split.half_of("s000") in (CHOOSING, REPORTING)
    assert split.half_of("s999") is None


def test_a_group_goes_to_the_choosing_half_when_it_makes_no_difference() -> None:
    (pair,) = [as_pair(same_row())]

    assert split_pairs([pair]).choosing == {"s001"}


# --- what is served, and what the labels say of it ----------------------------------------


def test_the_first_candidate_that_reaches_the_threshold_is_served() -> None:
    question = ask(
        same_row(),
        ("q2", scores(0.95, 0.20)),
        ("q1", scores(0.90, 0.80)),
        ("q3", scores(0.85, 0.99)),
    )

    assert served(question, COSINE, 0.93) == "q2"
    assert served(question, COSINE, 0.96) is None
    # The nearest fails the verifier, and the next is tried. The third is never reached.
    assert served(question, ENTAILS_MIN, 0.5) == "q1"
    assert served(question, ENTAILS_MIN, 0.9) == "q3"
    assert served(question, ENTAILS_MIN, 0.1) == "q2"
    assert served(question, ENTAILS_MIN, None) is None
    assert served(ask(same_row()), COSINE, 0.0) is None


def test_the_verifier_is_taken_by_its_weaker_direction_or_by_the_mean() -> None:
    question = ask(same_row(), ("q1", scores(0.9, forward=0.9, backward=0.5)))

    assert served(question, ENTAILS_MIN, 0.6) is None
    assert served(question, ENTAILS_MEAN, 0.6) == "q1"
    assert question.candidates[0].scores.of(ENTAILS_MEAN) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="no score named"):
        question.candidates[0].scores.of("euclid")


def test_the_outcomes_of_a_same_pair_and_of_a_different_pair() -> None:
    same, different = as_pair(same_row()), as_pair(different_row())

    assert outcome(same, "q1") == RIGHT
    assert outcome(same, "q2") == WRONG
    assert outcome(same, None) == MISSED
    # Anything served to a different question is wrong, its own golden question's answer
    # and any other.
    assert outcome(different, "q1") == WRONG
    assert outcome(different, "q2") == WRONG
    assert outcome(different, None) == NOT_SERVED


def test_a_tally_counts_each_outcome_and_tells_the_two_wrong_serves_apart() -> None:
    asked = [
        ask(same_row("s001"), ("q1", scores(0.9))),
        ask(same_row("s002"), ("q2", scores(0.9))),
        ask(same_row("s003"), ("q1", scores(0.5))),
        ask(different_row("d001"), ("q1", scores(0.9))),
        ask(different_row("d002"), ("q3", scores(0.9))),
        ask(different_row("d003"), ("q1", scores(0.5))),
        ask(different_row("d004")),
    ]

    counted = tally(asked, COSINE, 0.8)

    assert (counted.same, counted.right, counted.wrong_same, counted.missed) == (3, 1, 1, 1)
    assert (counted.different, counted.wrong_own, counted.wrong_other) == (4, 1, 1)
    assert (counted.wrong_different, counted.not_served, counted.wrong) == (2, 2, 3)
    nothing = tally(asked, COSINE, None)
    assert (nothing.missed, nothing.not_served, nothing.wrong) == (3, 4, 0)


# --- the threshold rule -------------------------------------------------------------------


def test_the_threshold_sits_halfway_above_the_highest_wrong_serve() -> None:
    asked = [
        ask(same_row("s001"), ("q1", scores(0.90))),
        ask(same_row("s002"), ("q1", scores(0.70))),
        ask(different_row("d001"), ("q1", scores(0.80))),
        ask(different_row("d002"), ("q1", scores(0.60))),
    ]

    chosen = choose_threshold(asked, COSINE)

    assert chosen.value == pytest.approx(0.85)
    assert chosen.decided_by == cache_pairs.WrongServe("d001", "q1", 0.80)
    counted = tally(asked, COSINE, chosen.value)
    # The rewording at 0.70 is the price of the near-miss at 0.80.
    assert (counted.right, counted.missed, counted.wrong) == (1, 1, 0)


def test_no_threshold_is_chosen_when_the_highest_score_of_all_is_a_wrong_serve() -> None:
    asked = [
        ask(same_row("s001"), ("q1", scores(0.90))),
        ask(different_row("d001"), ("q1", scores(0.95))),
    ]

    chosen = choose_threshold(asked, COSINE)

    assert chosen.value is None
    assert chosen.decided_by == cache_pairs.WrongServe("d001", "q1", 0.95)
    assert tally(asked, COSINE, chosen.value).right == 0


def test_with_nothing_wrong_to_serve_every_candidate_passes() -> None:
    asked = [
        ask(same_row("s001"), ("q1", scores(0.90))),
        ask(same_row("s002"), ("q1", scores(0.40))),
        ask(different_row("d001")),
    ]

    chosen = choose_threshold(asked, COSINE)

    assert (chosen.value, chosen.decided_by) == (0.40, None)
    assert tally(asked, COSINE, chosen.value).right == 2
    assert choose_threshold([ask(different_row("d001"))], COSINE).value is None


def test_a_same_question_served_another_stored_question_counts_as_wrong() -> None:
    asked = [
        ask(same_row("s001"), ("q2", scores(0.80)), ("q1", scores(0.70))),
        ask(same_row("s002"), ("q1", scores(0.90))),
    ]

    chosen = choose_threshold(asked, COSINE)

    assert chosen.value == pytest.approx(0.85)
    assert chosen.decided_by == cache_pairs.WrongServe("s001", "q2", 0.80)


def test_a_wrong_candidate_that_a_nearer_one_outscores_does_not_set_the_threshold() -> None:
    # The own question is nearer and scores higher, so the other is never reached.
    asked = [ask(same_row(), ("q1", scores(0.9, 0.95)), ("q2", scores(0.8, 0.90)))]

    chosen = choose_threshold(asked, ENTAILS_MIN)

    assert (chosen.value, chosen.decided_by) == (0.90, None)
    assert served(asked[0], ENTAILS_MIN, 0.90) == "q1"


def test_a_farther_wrong_candidate_that_scores_higher_does_set_it() -> None:
    # At 0.5 the nearer, right candidate is served. Raise the threshold past its score
    # and the farther, wrong one is served instead: stricter is not always safer.
    asked = [
        ask(same_row("s001"), ("q1", scores(0.9, 0.60)), ("q2", scores(0.8, 0.80))),
        ask(same_row("s002"), ("q1", scores(0.9, 0.95))),
    ]
    assert tally(asked, ENTAILS_MIN, 0.5).wrong == 0
    assert tally(asked, ENTAILS_MIN, 0.7).wrong == 1

    chosen = choose_threshold(asked, ENTAILS_MIN)

    assert chosen.value == pytest.approx(0.875)
    assert chosen.decided_by == cache_pairs.WrongServe("s001", "q2", 0.80)


def test_two_scores_too_close_for_a_middle_still_give_a_threshold_above_the_wrong_one() -> None:
    wrong, right = 0.5, math.nextafter(0.5, 1.0)
    assert wrong < right
    assert (wrong + right) / 2 == wrong
    asked = [
        ask(same_row("s001"), ("q1", scores(right))),
        ask(different_row("d001"), ("q1", scores(wrong))),
    ]

    chosen = choose_threshold(asked, COSINE)

    assert chosen.value == right
    assert tally(asked, COSINE, chosen.value).wrong == 0


def by_trying_every_threshold(asked: Sequence[Asked], policy: str) -> float | None:
    """The rule as it is stated, by brute force: the loosest threshold at which, and at
    every stricter one, nothing wrong is served."""
    seen = sorted({c.scores.of(policy) for question in asked for c in question.candidates})
    if not seen:
        return None
    between = [seen[0], *((low + high) / 2 for low, high in itertools.pairwise(seen))]
    everywhere = sorted({*seen, *between})
    safe = [
        threshold
        for threshold in between
        if all(tally(asked, policy, other).wrong == 0 for other in everywhere if other >= threshold)
    ]
    return min(safe) if safe else None


def test_the_rule_is_what_trying_every_threshold_gives_on_random_questions() -> None:
    random_numbers = random.Random(24)
    for _ in range(300):
        asked = []
        for number in range(random_numbers.randint(1, 8)):
            row = same_row(f"s{number:03d}") if random_numbers.random() < 0.5 else different_row()
            found = random_numbers.sample(["q1", "q2", "q3", "q4"], random_numbers.randint(0, 3))
            asked.append(
                ask(
                    row,
                    *(
                        # Few distinct values, so that ties and equal scores happen.
                        (
                            question_id,
                            scores(
                                1 - place / 10,
                                random_numbers.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
                                random_numbers.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
                            ),
                        )
                        for place, question_id in enumerate(found)
                    ),
                )
            )
        for policy in POLICIES:
            assert choose_threshold(asked, policy).value == by_trying_every_threshold(
                asked, policy
            ), (policy, asked)


def test_the_score_that_serves_most_is_taken_and_the_cheaper_one_on_a_tie() -> None:
    assert choose_policy({COSINE: 1, ENTAILS_MIN: 5, ENTAILS_MEAN: 9}) == ENTAILS_MEAN
    assert choose_policy({COSINE: 4, ENTAILS_MIN: 4, ENTAILS_MEAN: 4}) == COSINE
    assert choose_policy({COSINE: 0, ENTAILS_MIN: 3, ENTAILS_MEAN: 3}) == ENTAILS_MIN
    # Nothing served right by any score: there is nothing to take.
    assert choose_policy({COSINE: 0, ENTAILS_MIN: 0, ENTAILS_MEAN: 0}) is None


def test_the_area_under_the_curve() -> None:
    assert area_under_curve([0.9, 0.8], [0.7, 0.1]) == 1.0
    assert area_under_curve([0.1, 0.2], [0.7, 0.3]) == 0.0
    assert area_under_curve([0.5], [0.5]) == 0.5
    # Of the four comparisons: 0.9 beats both, 0.4 beats 0.3 and loses to 0.6.
    assert area_under_curve([0.9, 0.4], [0.6, 0.3]) == 0.75
    with pytest.raises(ValueError, match="both kinds"):
        area_under_curve([], [0.5])


# --- the simulated cache ------------------------------------------------------------------


class TableEmbedder:
    """A stand-in embedding model: each question has the vector a test gives it."""

    name = "table-embedder"
    dimension = 3

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("a question is embedded as a question, never as a passage")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vectors[text]


def towards(cosine: float) -> list[float]:
    """A vector of length 1 with this cosine to [1, 0, 0]."""
    return [cosine, (1 - cosine * cosine) ** 0.5, 0.0]


def test_the_candidates_are_the_nearest_stored_questions_above_the_floor() -> None:
    row = same_row(other="new question")
    embedder = TableEmbedder(
        {
            "new question": [1.0, 0.0, 0.0],
            TEXT["q1"]: towards(0.8),
            TEXT["q2"]: towards(0.9),
            TEXT["q3"]: towards(0.4),
            TEXT["q4"]: towards(0.7),
        }
    )
    verifier = FakeVerifier()

    measured = measure(
        [as_pair(row)], GOLDEN, embedder=embedder, verifier=verifier, candidates=2, floor=0.5
    )

    (question,) = measured.asked
    assert [candidate.question_id for candidate in question.candidates] == ["q2", "q1"]
    assert [c.scores.cosine for c in question.candidates] == pytest.approx([0.9, 0.8])
    # Three stored questions are above the floor, and the two nearest are taken.
    assert (question.above_floor, question.excluded) == (3, None)
    assert question.own.cosine == pytest.approx(0.8)
    assert (measured.stored, measured.candidates, measured.floor) == (4, 2, 0.5)
    # Every question is embedded once, as a question.
    assert sorted(embedder.queries) == sorted(["new question", *TEXT.values()])
    # One call, the stored question first and the new one second, no pair twice.
    assert verifier.calls == [[(TEXT["q1"], "new question"), (TEXT["q2"], "new question")]]
    assert (measured.embedded, measured.verified) == (5, 2)


def test_the_own_golden_question_is_scored_even_when_it_is_no_candidate() -> None:
    row = different_row(other="new question")
    embedder = TableEmbedder(
        {
            "new question": [1.0, 0.0, 0.0],
            TEXT["q1"]: towards(0.3),
            TEXT["q2"]: towards(0.9),
            TEXT["q3"]: towards(0.2),
            TEXT["q4"]: towards(0.1),
        }
    )
    verifier = FakeVerifier()

    (question,) = measure([as_pair(row)], GOLDEN, embedder=embedder, verifier=verifier).asked

    assert [candidate.question_id for candidate in question.candidates] == ["q2"]
    assert question.own.cosine == pytest.approx(0.3)
    assert (TEXT["q1"], "new question") in verifier.calls[0]


def test_a_stored_question_with_the_very_words_asked_is_never_a_candidate() -> None:
    # The other question is the golden question q2. Its own entry is the nearest of all,
    # at a cosine of 1, and is left out: serving it would be an exact match.
    row = golden_row("d001", "q1", "q2")
    embedder = FakeEmbedder()
    verifier = FakeVerifier()

    (question,) = measure(
        [as_pair(row)], GOLDEN, embedder=embedder, verifier=verifier, candidates=4, floor=-1.0
    ).asked

    assert question.excluded == "q2"
    assert sorted(c.question_id for c in question.candidates) == ["q1", "q3", "q4"]
    assert question.above_floor == 3
    assert question.candidates[0].question_id == "q1"
    assert all(first != second for first, second in verifier.calls[0])
    # The vector of a stored question is not computed a second time for asking it.
    assert embedder.queries.count(TEXT["q2"]) == 1


def test_stored_questions_at_the_same_distance_keep_the_order_of_their_ids() -> None:
    row = same_row(other="new question")
    vectors = {text: towards(0.6) for text in TEXT.values()}
    embedder = TableEmbedder({"new question": [1.0, 0.0, 0.0], **vectors})

    (question,) = measure([as_pair(row)], GOLDEN, embedder=embedder, verifier=FakeVerifier()).asked

    assert [candidate.question_id for candidate in question.candidates] == ["q1", "q2", "q3"]


def test_a_pair_whose_golden_question_is_not_stored_cannot_be_asked() -> None:
    row = same_row() | {"question_id": "q9", "question": "A question nobody stored?"}

    with pytest.raises(DatasetError, match="s001 cannot be asked"):
        measure([as_pair(row)], GOLDEN, embedder=FakeEmbedder(), verifier=FakeVerifier())
    with pytest.raises(ValueError, match="at least one candidate"):
        measure([], GOLDEN, embedder=FakeEmbedder(), verifier=FakeVerifier(), candidates=0)


# --- the figures, and a run on disk -------------------------------------------------------


def a_small_world() -> tuple[list[GoldenQuestion], list[CachePair]]:
    """Twelve golden questions, each with a rewording and a near-miss, and two disputed pairs."""
    topics = ["leave", "laptop", "travel", "salary", "badge", "pager"]
    golden = []
    rows = []
    for number, topic in enumerate(topics * 2):
        place = "office" if number < 6 else "home"
        question = {
            "question_id": f"g{number}",
            "question": f"How do I request {topic} approval for the {place} this year?",
        }
        golden.append(golden_question(question["question_id"], question["question"]))
        rows.append(
            same_row(f"s{number:03d}", other=f"How can I request {topic} approval for the {place}?")
            | question
        )
        rows.append(
            different_row(
                f"d{number:03d}",
                other=f"How do I cancel {topic} approval for the {place} this year?",
            )
            | question
        )
    for number in (0, 1):
        disputed = same_row(
            f"s{90 + number:03d}", other=f"Who signs off {topics[number]}?", agreed=False
        )
        disputed = second(disputed, label="different", confidence="low")
        rows.append(disputed | {"question_id": f"g{number}", "question": golden[number].question})
    return golden, [as_pair(row) for row in rows]


def figures_of_the_small_world() -> tuple[cache_pairs.Measured, dict[str, Any]]:
    golden, pairs = a_small_world()
    measured = measure(pairs, golden, embedder=FakeEmbedder(), verifier=FakeVerifier())
    figures = summarise(
        measured.asked,
        stored=measured.stored,
        candidates=measured.candidates,
        floor=measured.floor,
        other_splits=5,
    )
    return measured, figures


def test_disputed_pairs_are_in_no_figure_and_are_listed_with_their_scores() -> None:
    measured, figures = figures_of_the_small_world()
    without_them = summarise(
        [question for question in measured.asked if question.pair.agreed],
        stored=measured.stored,
        candidates=measured.candidates,
        floor=measured.floor,
        other_splits=5,
    )

    assert figures["pairs"] == {
        "in_file": 26,
        "agreed": 24,
        SAME: 12,
        DIFFERENT: 12,
        "disputed": ["s090", "s091"],
        "other_is_a_stored_question": 0,
    }
    # Every figure is the same with the disputed pairs taken out of the file.
    for part in ("pair_view", "cache_view"):
        assert figures[part] == without_them[part]
    assert [pair["pair"] for pair in figures["disputed"]] == ["s090", "s091"]
    for pair in figures["disputed"]:
        assert set(pair["served"]) == set(POLICIES)
        assert 0.0 <= pair[COSINE] <= 1.0
        assert pair["second_label"] == "different"


def test_the_halves_are_counted_apart_and_only_the_reporting_half_has_intervals() -> None:
    _, figures = figures_of_the_small_world()
    cache = figures["cache_view"]
    split = cache["split"]

    assert split[CHOOSING]["pairs"] + split[REPORTING]["pairs"] == 24
    assert split["groups"] == 12
    assert split["largest_group"] == 2
    for policy in POLICIES:
        chose, reported = cache["policies"][policy][CHOOSING], cache["policies"][policy][REPORTING]
        assert chose[SAME]["pairs"] == split[CHOOSING][SAME]
        assert reported[DIFFERENT]["pairs"] == split[REPORTING][DIFFERENT]
        # The rule allows nothing wrong on the half it chooses on.
        assert chose[SAME][WRONG] == chose[DIFFERENT][WRONG] == 0
        assert "right_share" not in chose[SAME]
        same, different = reported[SAME], reported[DIFFERENT]
        assert same[RIGHT] + same[WRONG] + same[MISSED] == same["pairs"]
        assert different[WRONG] + different[NOT_SERVED] == different["pairs"]
        assert different[WRONG] == different["wrong_own"] + different["wrong_other"]
        for name in (RIGHT, WRONG, MISSED):
            wilson = share_with_interval(same[name], same["pairs"])
            assert same[f"{name}_share"] == {
                "mean": wilson.mean,
                "low": wilson.low,
                "high": wilson.high,
                "count": same["pairs"],
            }
        assert different["wrong_share"]["mean"] == different[WRONG] / different["pairs"]
        assert len(cache["policies"][policy]["wrong_on_reporting"]) == (
            same[WRONG] + different[WRONG]
        )
        by_how = cache["policies"][policy]["reporting_by_how"]
        assert sum(row["pairs"] for row in by_how.values()) == same["pairs"]


def test_what_a_clean_half_cannot_show_is_the_top_of_the_interval_of_none_wrong() -> None:
    _, figures = figures_of_the_small_world()
    allows = figures["cache_view"]["zero_wrong_allows"]

    assert allows["near_misses"] == figures["cache_view"]["split"][REPORTING][DIFFERENT]
    assert allows["up_to"] == share_with_interval(0, allows["near_misses"]).high
    assert allows["up_to"] > 0.2  # a handful of near-misses shows next to nothing


def test_the_pair_view_describes_every_agreed_pair_by_kind_and_group() -> None:
    _, figures = figures_of_the_small_world()
    view = figures["pair_view"]["scores"]

    assert list(view) == list(cache_pairs.SCORES)
    for score in cache_pairs.SCORES:
        assert view[score]["by_kind"][SAME]["pairs"] == 12
        assert view[score]["by_kind"][DIFFERENT]["pairs"] == 12
        assert list(view[score]["by_how"]) == ["reworded"]
        assert list(view[score]["by_what_differs"]) == ["region"]
        assert 0.0 <= view[score]["area_under_curve"] <= 1.0
        counts = [(row[SAME], row[DIFFERENT]) for row in view[score]["at_or_above"]]
        # A stricter threshold never lets more pairs through.
        assert counts == sorted(counts, reverse=True)
        assert [row["threshold"] for row in view[score]["at_or_above"]] == list(
            cache_pairs.THRESHOLDS
        )


def test_the_rules_are_run_again_on_other_splits_of_the_same_pairs() -> None:
    _, figures = figures_of_the_small_world()
    others = figures["cache_view"]["other_splits"]

    assert others["splits"] == 5
    assert sum(others["policies"][policy]["chosen"] for policy in POLICIES) == (
        5 - others["chosen_policy"]["splits_without_a_choice"]
    )
    for policy in POLICIES:
        assert 0 <= others["policies"][policy]["splits_with_nothing_wrong"] <= 5
        spread = others["policies"][policy]["right_share"]
        assert 0.0 <= spread["min"] <= spread["median"] <= spread["max"] <= 1.0


def test_when_no_score_is_safe_nothing_is_chosen_and_an_empty_half_has_no_shares() -> None:
    # One golden question, so one group, and it goes to the choosing half whole. Its
    # near-miss scores above its rewording by every score, so no threshold is safe.
    asked = [
        ask(same_row("s001"), ("q1", scores(0.5)), own=scores(0.5)),
        ask(different_row("d001"), ("q1", scores(0.9)), own=scores(0.9)),
    ]

    figures = summarise(asked, stored=4, candidates=3, floor=0.5, other_splits=3)
    cache = figures["cache_view"]

    assert cache["chosen_policy"] is None
    assert cache["split"][CHOOSING]["pairs"] == 2
    assert cache["split"][REPORTING]["pairs"] == 0
    for policy in POLICIES:
        assert cache["policies"][policy]["threshold"] is None
        assert cache["policies"][policy]["decided_by"]["pair"] == "d001"
        assert cache["policies"][policy][REPORTING][SAME]["right_share"] is None
        assert cache["policies"][policy][REPORTING][DIFFERENT]["wrong_share"] is None
    assert cache["zero_wrong_allows"] == {"near_misses": 0, "up_to": None}
    assert cache["other_splits"]["chosen_policy"]["splits_without_a_choice"] == 3
    assert cache["other_splits"]["chosen_policy"]["right_share"] is None
    # With every pair in view it is the same: the near-miss is on top.
    assert cache["whole_set"]["chosen_on_all_pairs"][COSINE]["threshold"] is None
    summary = {
        **figures,
        "config": {
            "embedding_model": "fake-embedder",
            "embedding_device": "cpu",
            "verifier_model": "fake-verifier",
            "verifier_device": "cpu",
            **FakeVerifier().describe(),
        },
        "timing": {
            "questions_embedded": 0,
            "embedding_seconds": 0.0,
            "pairs_verified": 0,
            "verifying_seconds": 0.0,
        },
    }
    report = "\n".join(report_lines(summary))
    assert "the rule takes: none, so a cache would serve nothing" in report
    assert "no pairs" in report


def test_the_measurement_needs_agreed_pairs_of_both_kinds() -> None:
    golden, pairs = a_small_world()
    only_same = [pair for pair in pairs if pair.kind == SAME]
    measured = measure(only_same, golden, embedder=FakeEmbedder(), verifier=FakeVerifier())

    with pytest.raises(DatasetError, match="both kinds"):
        summarise(measured.asked, stored=12, candidates=3, floor=0.5)


def test_a_saved_run_gives_its_figures_again_from_its_rows(tmp_path: Path) -> None:
    _, pairs = a_small_world()
    measured, figures = figures_of_the_small_world()

    summary = write_cache_run(
        measured,
        tmp_path / "run",
        description="A run for a test.",
        config={"embedding_model": "fake-embedder"},
        other_splits=5,
    )

    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "pairs.jsonl",
        "questions.jsonl",
        "summary.json",
    ]
    assert summary == json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["description"] == "A run for a test."
    # What the figures were computed with is recorded by the writer itself.
    assert summary["config"] == {
        "embedding_model": "fake-embedder",
        "candidates": 3,
        "floor": 0.5,
        "split_seed": cache_pairs.SPLIT_SEED,
        "other_splits": 5,
    }
    assert {name: summary[name] for name in figures} == figures
    again = load_cache_run(tmp_path / "run", pairs)
    assert again == measured.asked
    assert summarise(again, stored=12, candidates=3, floor=0.5, other_splits=5) == figures


def test_the_rows_of_a_run_say_what_was_served_at_each_chosen_threshold(tmp_path: Path) -> None:
    measured, figures = figures_of_the_small_world()
    write_cache_run(measured, tmp_path, description="", config={}, other_splits=5)
    lines = (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = {row["pair"]: row for row in map(json.loads, lines)}
    own = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    pair_rows = {row["pair"]: row for row in map(json.loads, own)}

    assert list(rows) == list(pair_rows) == [question.pair.pair for question in measured.asked]
    assert pair_rows["s090"]["half"] is None
    assert pair_rows["s090"]["agreed"] is False
    for question in measured.asked:
        row, pair_row = rows[question.pair.pair], pair_rows[question.pair.pair]
        assert row["half"] == pair_row["half"]
        assert pair_row[COSINE] == question.own.cosine
        assert pair_row[ENTAILS_MIN] == min(
            pair_row["entails_forward"], pair_row["entails_backward"]
        )
        assert [c["question_id"] for c in row["candidates"]] == [
            candidate.question_id for candidate in question.candidates
        ]
        for policy in POLICIES:
            threshold = figures["cache_view"]["policies"][policy]["threshold"]
            assert row["served"][policy] == served(question, policy, threshold)
            assert row["outcome"][policy] == outcome(question.pair, row["served"][policy])
    reported = [row for row in rows.values() if row["half"] == REPORTING]
    for policy in POLICIES:
        counted = figures["cache_view"]["policies"][policy][REPORTING]
        assert sum(row["outcome"][policy] == RIGHT for row in reported) == counted[SAME][RIGHT]


def test_a_run_of_other_pairs_is_refused(tmp_path: Path) -> None:
    _, pairs = a_small_world()
    measured, _ = figures_of_the_small_world()
    write_cache_run(measured, tmp_path, description="", config={}, other_splits=5)

    with pytest.raises(DatasetError, match="was not made from the pairs given"):
        load_cache_run(tmp_path, pairs[:-1])
    with pytest.raises(DatasetError, match="not found"):
        load_cache_run(tmp_path / "none", pairs)


def test_the_report_calls_a_description_a_description(tmp_path: Path) -> None:
    measured, _ = figures_of_the_small_world()
    summary = write_cache_run(
        measured,
        tmp_path,
        description="",
        config={
            "embedding_model": "fake-embedder",
            "embedding_device": "cpu",
            "verifier_model": "fake-verifier",
            "verifier_device": "cpu",
            **FakeVerifier().describe(),
        },
        other_splits=5,
    )

    report = "\n".join(report_lines(summary))

    assert "24 agreed pairs (12 same, 12 different)" in report
    assert "Disputed and listed apart: s090, s091" in report
    assert "A description of the scores, not a held-out result." in report
    assert "A description: these thresholds are scored on the pairs they are read from." in report
    assert "On the reporting half, which chose nothing" in report
    assert "none to any model API" in report
    assert "entailment is output 1 of contradiction, entailment, neutral" in report
    assert "Disputed s090 (maker: same, second: different), in no figure" in report
