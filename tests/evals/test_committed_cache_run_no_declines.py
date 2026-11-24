"""The second committed cache measurement: stored declines left out by the label.

``evals/runs/cache/no-decline-candidates`` simulates a cache in which a stored question
the handbook does not answer is no candidate for a match by meaning. That such a
question holds a decline is the golden set's label, so this run does not measure the
cache as it is built: ``test_committed_cache_run_as_built.py`` holds the run that does,
and the cache's defaults with it. The first part here holds this run to the pairs and
to itself, as ``test_committed_cache_run.py`` holds the first run. The second holds the
documents to the run, number by number where a sentence carries a finding.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.evals import cache_pairs
from handbook_rag.evals.cache_pairs import (
    CHOOSING,
    COSINE,
    DIFFERENT,
    ENTAILS_MEAN,
    ENTAILS_MIN,
    MISSED,
    POLICIES,
    REPORTING,
    RIGHT,
    SAME,
    WRONG,
    Asked,
    load_cache_run,
    summarise,
    tally,
    unanswered_questions,
)
from handbook_rag.evals.cache_report import report_lines
from tests.evals.test_committed_cache_run import (
    DECIDED_BY as LABELS,
)
from tests.evals.test_committed_cache_run import (
    cell,
    day,
    golden,
    named,
    numbers_of_the_run,
    pairs,
    section,
)
from tests.evals.test_committed_cache_run import (
    saved as first_run,
)
from tests.test_readme_results import line_starting, numbers, table, three

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "cache" / "no-decline-candidates"
FIRST = ROOT / "evals" / "runs" / "cache" / "baseline"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "cache" / "README.md"
HEADING = "The measurement again, with stored declines left out"


def saved() -> Any:
    return json.loads((RUN / "summary.json").read_text(encoding="utf-8"))


def rows(run: Path, name: str) -> dict[str, Any]:
    lines = (run / name).read_text(encoding="utf-8").splitlines()
    return {row["pair"]: row for row in map(json.loads, lines)}


def asked() -> list[Asked]:
    return load_cache_run(RUN, pairs())


def chosen(summary: Any) -> Any:
    """The figures of the score the rule took, which is the one the cache decides by."""
    return summary["cache_view"]["policies"][summary["cache_view"]["chosen_policy"]]


# --- the run -----------------------------------------------------------------------------


def test_the_second_run_was_made_as_the_first_but_for_the_stored_declines() -> None:
    summary, first = saved(), first_run()

    assert summary["config"] == {**first["config"], "decline_candidates": False}
    assert "decline_candidates" not in first["config"]
    assert summary["description"] == (
        f"{first['description']} A stored decline is no candidate, as in the answer cache."
    )
    assert sorted(path.name for path in RUN.iterdir()) == [
        "pairs.jsonl",
        "questions.jsonl",
        "summary.json",
    ]
    assert summary["rows"] == cache_pairs.rows_fingerprints(RUN)
    # The first run is as it was: nothing in it speaks of declines as candidates.
    assert "stored_declines" not in first["cache_view"]


def test_the_figures_of_the_second_run_are_what_its_rows_give() -> None:
    summary = saved()
    config = summary["config"]

    again = summarise(
        asked(),
        stored=len(golden()),
        candidates=config["candidates"],
        floor=config["floor"],
        unanswered=unanswered_questions(golden()),
        seed=config["split_seed"],
        other_splits=config["other_splits"],
        decline_candidates=config["decline_candidates"],
    )

    assert set(summary) == {"description", "measured_on", "config", "rows", "timing", *again}
    for part, figures in again.items():
        assert summary[part] == json.loads(json.dumps(figures)), part
    declines = summary["cache_view"]["stored_declines"]
    assert (declines["left_out"], declines["candidates_from"]) == (33, 132)
    assert declines["pairs_whose_golden_question_is_one"] == {SAME: 0, DIFFERENT: 20}
    assert (
        "  a stored decline is no candidate: the 33 are left out, and each question meets the "
        "nearest of the other 132"
    ) in report_lines(summary)


def test_no_candidate_of_the_second_run_is_a_stored_decline() -> None:
    declines = unanswered_questions(golden())
    now, before = rows(RUN, "questions.jsonl"), rows(FIRST, "questions.jsonl")

    met = [c["question_id"] for row in now.values() for c in row["candidates"]]
    met_before = [c["question_id"] for row in before.values() for c in row["candidates"]]

    assert len(declines) == 33
    assert not set(met) & declines
    # The first run did meet them, so leaving them out changed what was measured.
    assert set(met_before) & declines
    changed = [
        pair
        for pair in now
        if [c["question_id"] for c in now[pair]["candidates"]]
        != [c["question_id"] for c in before[pair]["candidates"]]
    ]
    assert len(changed) > 20
    # A candidate that both runs hold is the same stored question with the same cosine.
    for pair in now:
        earlier = {c["question_id"]: c for c in before[pair]["candidates"]}
        for candidate in now[pair]["candidates"]:
            if candidate["question_id"] in earlier:
                same = earlier[candidate["question_id"]]
                assert math.isclose(candidate[COSINE], same[COSINE], abs_tol=1e-8)


def largest_difference_of_the_pair_scores() -> float:
    now, before = rows(RUN, "pairs.jsonl"), rows(FIRST, "pairs.jsonl")
    assert list(now) == list(before)
    for pair in now:
        assert now[pair]["half"] == before[pair]["half"], "the same split, by the same seed"
    return max(
        abs(float(now[pair][score]) - float(before[pair][score]))
        for pair in now
        for score in (COSINE, "entails_forward", "entails_backward")
    )


def test_each_pair_scores_against_its_own_golden_question_as_in_the_first_run() -> None:
    # The verifier read other batches this time, which moves a score in its last digits.
    assert largest_difference_of_the_pair_scores() <= 1e-8


# --- the documents -----------------------------------------------------------------------


def guide() -> str:
    """The guide's section on the second run."""
    return section(GUIDE.read_text(encoding="utf-8"), HEADING)


def card() -> str:
    return section(CARD.read_text(encoding="utf-8"), "Measured again with stored declines left out")


def said(text: str, start: str) -> str:
    return line_starting(text, f"- **{start}")


def test_the_guides_table_of_the_second_run_is_the_run() -> None:
    policies = saved()["cache_view"]["policies"]
    found = table(guide(), "| Decided by | Threshold | Rewordings served right |")

    assert [row[0] for row in found] == [LABELS[policy] for policy in POLICIES]
    for row, policy in zip(found, POLICIES, strict=True):
        figures = policies[policy]
        same, different = figures[REPORTING][SAME], figures[REPORTING][DIFFERENT]
        threshold = figures["threshold"]
        assert row[1:] == [
            "none" if threshold is None else f"{threshold:.4f}",
            cell(same[RIGHT], same["pairs"], same.get("right_share")),
            cell(same[WRONG], same["pairs"], same.get("wrong_share")),
            cell(same[MISSED], same["pairs"], same.get("missed_share")),
            cell(different[WRONG], different["pairs"], different.get("wrong_share")),
        ], policy
    for policy in (ENTAILS_MIN, ENTAILS_MEAN):
        assert policies[policy][CHOOSING][SAME][RIGHT] == 20
    assert "served 20 of 49 rewordings right, so the rule again takes the weaker direction" in (
        guide()
    )


def declined_in_the_answer_evaluation() -> tuple[int, int]:
    """Of the questions the handbook does not answer: how many the saved answer run
    declined, and how many it answered."""
    declines = unanswered_questions(golden())
    lines = (ROOT / "evals" / "runs" / "answers" / "baseline" / "answers.jsonl").read_text(
        encoding="utf-8"
    )
    replies = [json.loads(line) for line in lines.splitlines() if line.strip()]
    answered = [reply["answered"] for reply in replies if reply["question_id"] in declines]
    assert len(answered) == len(declines)
    return answered.count(False), answered.count(True)


def answered_though_the_handbook_does_not_answer() -> list[str]:
    """The questions the handbook does not answer, by their label, that the saved answer
    run answered all the same."""
    declines = unanswered_questions(golden())
    lines = (ROOT / "evals" / "runs" / "answers" / "baseline" / "answers.jsonl").read_text(
        encoding="utf-8"
    )
    replies = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return sorted(
        r["question_id"] for r in replies if r["question_id"] in declines and r["answered"]
    )


def test_what_the_guide_says_changed_in_the_simulation_is_what_the_run_holds() -> None:
    summary = saved()
    cache = summary["cache_view"]
    declines = cache["stored_declines"]
    text = guide()

    assert numbers(said(text, "What changed in the simulation.")) == [
        str(cache["stored_questions"]),
        str(declines["left_out"]),
        str(cache["candidates"]),
        str(declines["candidates_from"]),
        str(declines["pairs_whose_golden_question_is_one"][DIFFERENT]),
        str(summary["pairs"][DIFFERENT]),
        str(declines["left_out"]),
        str(declines["pairs_whose_golden_question_is_one"][DIFFERENT]),
    ]
    # In a real cache the wrong serve can happen, and the guide names the pair that shows it.
    assert "in this simulation the wrong serve that the pair was written to provoke" in text
    assert named(said(text, "What changed in the simulation.")) == ["d067"]
    assert declines["pairs_whose_golden_question_is_one"][SAME] == 0
    assert numbers(said(text, "What did not change.")) == [str(cache["floor"]), "0.00000001"]
    assert largest_difference_of_the_pair_scores() <= 0.00000001
    declined, answered = declined_in_the_answer_evaluation()
    assert numbers(said(text, "One assumption.")) == [
        str(declined),
        str(declines["left_out"]),
        str(answered),
        str(answered),
    ]
    assert (declined, answered) == (31, 2)
    # One of the two is the stored question of the near-miss the second check scores highest.
    assert named(said(text, "One assumption.")) == ["u004", "d067"]
    (pair,) = [pair for pair in pairs() if pair.pair == "d067"]
    assert pair.question_id == "u004"
    assert "u004" in answered_though_the_handbook_does_not_answer()
    assert "--variant no-decline-candidates --no-declines" in text
    # What the run is not, and that its one change was chosen with the first run in view.
    assert "- **So this run does not measure the cache as it is built.**" in text
    assert "It decides by a label what a cache holds, and the cache decides by the reply." in text
    assert "That was decided after the first run had been read, with every pair in view" in text
    assert "So the run below is no held-out test of that decision." in text


def test_what_the_guide_says_the_second_run_found_is_what_the_run_holds() -> None:
    summary, first = saved(), first_run()
    cache = summary["cache_view"]
    weaker, mean = cache["policies"][ENTAILS_MIN], cache["policies"][ENTAILS_MEAN]
    reported, there = weaker[REPORTING], weaker["reporting_by_how"]
    text = guide()

    start = "Without the stored declines the threshold is lower"
    assert numbers(said(text, start)) == [
        f"{weaker['threshold']:.4f}",
        f"{chosen(first)['threshold']:.4f}",
        f"{weaker['decided_by']['score']:.4f}",
        f"{weaker['halfway_to']:.4f}",
    ]
    assert named(said(text, start)) == [weaker["decided_by"]["pair"]] == ["d004"]
    assert (weaker["decided_by"]["group"], weaker["decided_by"]["own"]) == ("region", True)
    # "The two near-misses that set the first threshold were stored declines."
    assert chosen(first)["decided_by"]["stored_unanswered"] is True
    assert weaker["threshold"] == (weaker["decided_by"]["score"] + weaker["halfway_to"]) / 2

    start = "It served 16 of 49 rewordings on the reporting half"
    assert numbers(said(text, start)) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(chosen(first)[REPORTING][SAME][RIGHT]),
        str(reported[DIFFERENT]["pairs"]),
        *(
            str(there[how][key])
            for how in ("reworded", "synonyms", "short", "long", "indirect")
            for key in (RIGHT, "pairs")
        ),
    ]
    assert (reported[SAME][WRONG], reported[DIFFERENT][WRONG]) == (0, 0), "and nothing wrong"

    start = "The reporting half did not test this threshold hard either."
    hardest = weaker["highest_wrong"][REPORTING]
    assert numbers(said(text, start)) == [
        three(hardest[DIFFERENT]["score"]),
        three(hardest[SAME]["score"]),
        three(weaker["threshold"] - hardest[SAME]["score"]),
    ]
    assert named(said(text, start)) == [hardest[SAME]["pair"], hardest[SAME]["served"]]
    assert named(said(text, start)) == ["s073", "q110"]
    assert hardest[SAME]["own"] is False, "another stored question's answer"
    assert named(said(text, "A label covers a pair's own golden question only.")) == [
        "q110",
        "s073",
    ]

    start = "Where the threshold sits in its gap is a choice, and here it matters."
    assert numbers(said(text, start)) == [
        f"{weaker['decided_by']['score']:.4f}",
        str(mean[REPORTING][SAME][WRONG]),
        str(mean[REPORTING][SAME]["pairs"]),
    ]
    just_above = math.nextafter(weaker["decided_by"]["score"], 1.0)
    halves = {
        half: [
            q
            for q in asked()
            if q.pair.agreed and rows(RUN, "pairs.jsonl")[q.pair.pair]["half"] == half
        ]
        for half in (CHOOSING, REPORTING)
    }
    assert tally(halves[CHOOSING], ENTAILS_MIN, just_above).wrong == 0
    assert tally(halves[REPORTING], ENTAILS_MIN, just_above).wrong_same == 1
    (by_the_mean,) = mean["wrong_on_reporting"]
    assert (by_the_mean["pair"], by_the_mean["served"]) == ("s073", "q110")

    start = "On 200 other splits of the same pairs"
    others = cache["other_splits"]
    over = others["policies"][ENTAILS_MIN]
    fell = over["by_where_the_hardest_pair_fell"]
    assert numbers(said(text, start)) == [
        str(others["splits"]),
        str(over["splits_with_something_wrong"]),
        str(over["splits_with_nothing_wrong"]),
        str(fell[REPORTING]["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits"]),
        str(fell[CHOOSING]["splits"]),
        str(others["splits"]),
        three(over["right_share"]["min"]),
        three(over["right_share"]["max"]),
        three(over["right_share"]["median"]),
        three(over["wrong_share_of_different"]["max"]),
    ]
    assert named(said(text, start)) == [fell["pair"]] == ["s073"]
    assert fell[CHOOSING]["splits_with_something_wrong"] == 0

    whole = cache["whole_set"]["chosen_on_all_pairs"][ENTAILS_MIN]
    assert numbers(said(text, "With every pair in view the rule gives 0.8600")) == [
        f"{whole['threshold']:.4f}",
        str(whole[SAME][RIGHT]),
        str(whole[SAME]["pairs"]),
    ]
    assert (whole[SAME][WRONG], whole[DIFFERENT][WRONG]) == (0, 0)

    timing = summary["timing"]
    assert numbers(said(text, "What it cost.")) == [
        str(timing["pairs_verified"]),
        str(timing["verifying_seconds"]),
        str(timing["verifying_ms_per_pair"]),
        str(timing["questions_embedded"]),
        str(timing["embedding_seconds"]),
    ]


def test_the_guide_gives_the_day_of_the_second_run_and_whose_word_it_takes() -> None:
    summary = saved()
    text = guide()

    assert f"`no-decline-candidates`, on {day(summary)}, with the settings of the first run" in text
    assert summary["measured_on"] == first_run()["measured_on"]
    assert (
        "The cache simulated holds 165 questions, and what it holds for each is taken from a label."
    ) in text
    # The cache's defaults are not given here: they rest on the run that follows.
    assert "HANDBOOK_RAG_CACHE" not in text
    assert str(Settings().cache_threshold) not in numbers(text)


def test_every_number_the_guide_gives_of_the_second_run_is_a_number_of_a_run() -> None:
    """The wider net of the first run's test, over the section on the second."""
    summary = saved()
    allowed = numbers_of_the_run(summary) | numbers_of_the_run(first_run())
    allowed |= {
        "0020",  # the decision record
        "2026",
        "0.00000001",  # held above to the largest difference between the two runs
        "0.013",  # the threshold less the score of s073, held above
        *(str(count) for count in declined_in_the_answer_evaluation()),
    }

    unknown = [number for number in numbers(guide()) if number not in allowed]

    assert unknown == []


def test_what_the_card_says_of_the_second_run_is_what_the_run_holds() -> None:
    summary = saved()
    cache = summary["cache_view"]
    weaker, declines = cache["policies"][ENTAILS_MIN], cache["stored_declines"]
    reported, hardest = weaker[REPORTING], weaker["highest_wrong"][REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    text = card()

    assert f"on {day(summary)}, and saved the run as" in text
    assert "(../runs/cache/no-decline-candidates)" in text
    assert numbers(said(text, "What changed.")) == [
        str(declines["left_out"]),
        str(cache["candidates"]),
        str(declines["candidates_from"]),
    ]
    assert "So this run does not measure the cache as it is built" in said(text, "What changed.")
    assert "default" not in text
    start = "The threshold is lower"
    assert numbers(said(text, start)) == [
        f"{weaker['threshold']:.4f}",
        f"{weaker['decided_by']['score']:.4f}",
    ]
    assert named(said(text, start)) == ["d004"]
    assert "a `region` pair" in said(text, start)
    assert numbers(said(text, "On the reporting half it served")) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
    ]
    assert (reported[SAME][WRONG], reported[DIFFERENT][WRONG]) == (0, 0)
    start = "That half held nothing hard near the threshold."
    assert numbers(said(text, start)) == [
        three(hardest[DIFFERENT]["score"]),
        three(hardest[SAME]["score"]),
    ]
    assert named(said(text, start)) == ["s073"]
    assert numbers(said(text, "On 200 other splits of the same pairs")) == [
        str(cache["other_splits"]["splits"]),
        str(over["splits_with_something_wrong"]),
    ]


def test_the_two_saved_runs_hold_the_same_pairs_under_the_same_ids() -> None:
    first, second = rows(FIRST, "pairs.jsonl"), rows(RUN, "pairs.jsonl")

    assert list(first) == list(second) == [pair.pair for pair in pairs()]
    for pair in first:
        for name in ("kind", "agreed", "question_id", "other_id", "how", "what_differs"):
            assert second[pair][name] == first[pair][name], (pair, name)


# --- the real models ---------------------------------------------------------------------


@pytest.mark.slow
def test_the_real_models_give_the_candidates_and_scores_of_the_second_run_again() -> None:
    settings = Settings()
    config = saved()["config"]
    measured = cache_pairs.measure(
        pairs(),
        golden(),
        embedder=wiring.make_embedder(wiring.settings_for_serving(settings)),
        verifier=wiring.make_verifier(settings),
        candidates=config["candidates"],
        floor=config["floor"],
        decline_candidates=config["decline_candidates"],
    )

    for now, then in zip(measured.asked, asked(), strict=True):
        assert now.pair == then.pair
        assert [c.question_id for c in now.candidates] == [c.question_id for c in then.candidates]
        assert (now.above_floor, now.excluded) == (then.above_floor, then.excluded)
        for new, old in zip(
            (now.own, *(c.scores for c in now.candidates)),
            (then.own, *(c.scores for c in then.candidates)),
            strict=True,
        ):
            # Another machine may differ in the last digits, and in nothing a reader sees.
            assert new.cosine == pytest.approx(old.cosine, abs=1e-4)
            assert new.entails.forward == pytest.approx(old.entails.forward, abs=1e-3)
            assert new.entails.backward == pytest.approx(old.entails.backward, abs=1e-3)
