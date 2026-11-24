"""The second committed cache measurement, and what the answer cache takes from it.

``evals/runs/cache/no-decline-candidates`` simulates the cache as it is built: a stored
decline is no candidate for a match by meaning. The first part holds that run to the
pairs and to itself, as ``test_committed_cache_run.py`` holds the first run. The second
holds the cache to the run: its default threshold, its candidates, its floor and the
score it decides by, and that the cache's own code serves each question of the run what
the simulation serves it. The third holds the documents to the run, number by number
where a sentence carries a finding.

The figures of the one check with the real service, in the guide's last subsection, come
from no saved run, and no test can work them out again.
"""

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from handbook_rag import wiring
from handbook_rag.cache.answers import DECIDED_BY, AnswerCache, comparable, entry_id
from handbook_rag.cache.store import Entry
from handbook_rag.cache.verify import Verification
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
    served,
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
from tests.test_readme_results import anchors, line_starting, numbers, table, three

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "cache" / "no-decline-candidates"
FIRST = ROOT / "evals" / "runs" / "cache" / "baseline"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "cache" / "README.md"
RECORD = ROOT / "docs" / "adr" / "0020-an-answer-cache-that-matches-by-meaning.md"
HEADING = "The answer cache, measured as it is built"
REAL_CHECK = "### Checked once with the real service"


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


# --- the cache ---------------------------------------------------------------------------


def test_the_caches_default_threshold_is_the_one_the_second_run_chose() -> None:
    summary, settings = saved(), Settings()
    cache = summary["cache_view"]
    threshold = chosen(summary)["threshold"]

    # The score the rule took is the one the cache decides by.
    assert cache["chosen_policy"] == DECIDED_BY == ENTAILS_MIN
    # Rounded up to four places: never looser than the threshold the run chose.
    assert settings.cache_threshold == math.ceil(threshold * 10**4) / 10**4 == 0.8669
    assert 0 <= settings.cache_threshold - threshold < 1e-4
    # The other two numbers of the cache are the run's own settings.
    assert settings.cache_candidates == summary["config"]["candidates"] == cache_pairs.CANDIDATES
    assert settings.cache_floor == summary["config"]["floor"] == cache_pairs.FLOOR
    assert settings.cache_verifier_model == summary["config"]["verifier_model"]
    assert settings.embedding_model == summary["config"]["embedding_model"]
    assert settings.embedding_query_prefix == summary["config"]["embedding_query_prefix"]


def test_rounding_the_threshold_changes_nothing_the_run_served() -> None:
    summary, default = saved(), Settings().cache_threshold
    threshold = chosen(summary)["threshold"]
    everyone = asked()

    for question in everyone:
        assert served(question, DECIDED_BY, default) == served(question, DECIDED_BY, threshold)
    agreed = [question for question in everyone if question.pair.agreed]
    assert tally(agreed, DECIDED_BY, default) == tally(agreed, DECIDED_BY, threshold)
    # No score of the run lies between the two, so no other question could tell them apart.
    between = [
        candidate.scores.of(DECIDED_BY)
        for question in everyone
        for candidate in question.candidates
        if threshold <= candidate.scores.of(DECIDED_BY) < default
    ]
    assert between == []


class Replayed:
    """A store that holds, for one question, the candidates the saved run found for it."""

    def __init__(self, question: Asked) -> None:
        self.question = question

    def get(self, entry_id: str) -> Entry | None:
        return None

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        # A stored question is named by its id, which stands in for its text here.
        return [
            (
                Entry(
                    entry_id=entry_id(comparable(candidate.question_id), scope),
                    question=candidate.question_id,
                    comparable=comparable(candidate.question_id),
                    scope=scope,
                    segments=(),
                    answered=True,
                    model="replayed",
                    stop_reason="end_turn",
                    stored_at="2026-11-24T00:00:00+00:00",
                    by_meaning=True,
                ),
                candidate.scores.cosine,
            )
            for candidate in self.question.candidates
        ]

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        raise AssertionError("nothing is stored when a run is replayed")


class ReplayedScores:
    """The verifier's scores of the saved run, by the stored question's id."""

    name = "replayed"

    def __init__(self, question: Asked) -> None:
        self._scores = {c.question_id: c.scores.entails for c in question.candidates}

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        return [self._scores[stored] for stored, _ in pairs]


class AnyVector:
    name = "replayed"
    dimension = 1

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("a question is embedded as a question")

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


def test_the_cache_serves_each_question_of_the_run_what_the_simulation_served_it() -> None:
    """The measurement counts what a simulated cache serves. This is the built one.

    Each question of the run is put to the cache's own code, with the run's candidates
    and scores in place of the store and the two models. With the default settings it
    has to serve the stored question the simulation served, or nothing where that
    served nothing: the same candidates, the same order, the same score, the same edge.
    """
    summary, settings = saved(), Settings()
    threshold = chosen(summary)["threshold"]
    by_meaning = 0

    for question in asked():
        cache = AnswerCache(
            Replayed(question),
            embedder=AnyVector(),
            verifier=ReplayedScores(question),
            threshold=settings.cache_threshold,
            candidates=settings.cache_candidates,
            floor=settings.cache_floor,
        )

        found = cache.lookup(question.pair.other).answer

        expected = served(question, DECIDED_BY, threshold)
        got = None if found is None or found.cached is None else found.cached.question
        assert got == expected, question.pair.pair
        if found is not None and found.cached is not None:
            by_meaning += 1
            assert found.cached.tier == "semantic"
            (candidate,) = [c for c in question.candidates if c.question_id == expected]
            assert found.cached.score == candidate.scores.of(DECIDED_BY)
            assert found.cached.similarity == candidate.scores.cosine
    # Every serve of the run, on both halves: 20 and 16 rewordings, and one disputed pair less.
    counted = summary["cache_view"]["policies"][DECIDED_BY]
    served_right = counted[CHOOSING][SAME][RIGHT] + counted[REPORTING][SAME][RIGHT]
    assert by_meaning == served_right == 36


# --- the documents -----------------------------------------------------------------------


def guide() -> str:
    """The guide's section on the second run, without the check that no run holds."""
    whole = section(GUIDE.read_text(encoding="utf-8"), HEADING)
    assert whole.count(REAL_CHECK) == 1
    return whole.split(REAL_CHECK)[0]


def card() -> str:
    return section(CARD.read_text(encoding="utf-8"), "Measured again for the cache as it is built")


def record() -> str:
    return RECORD.read_text(encoding="utf-8")


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
    ]
    assert declines["pairs_whose_golden_question_is_one"][SAME] == 0
    assert numbers(said(text, "What did not change.")) == [str(cache["floor"]), "0.00000001"]
    assert largest_difference_of_the_pair_scores() <= 0.00000001
    declined, answered = declined_in_the_answer_evaluation()
    assert numbers(said(text, "One assumption.")) == [
        str(declined),
        str(declines["left_out"]),
        str(answered),
    ]
    assert (declined, answered) == (31, 2)
    assert "--variant no-decline-candidates --no-declines" in text


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
        three(Settings().cache_threshold - hardest[SAME]["score"]),
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


def test_the_guide_gives_the_day_of_the_second_run_and_the_default_it_gave() -> None:
    summary, settings = saved(), Settings()
    text = guide()

    assert f"`no-decline-candidates`, on {day(summary)}, with the settings of the first run" in text
    assert summary["measured_on"] == first_run()["measured_on"]
    line = line_starting(text, "`HANDBOOK_RAG_CACHE_THRESHOLD` is")
    assert numbers(line) == [str(settings.cache_threshold)]
    assert "by the weaker direction, rounded up to four places" in line
    line = line_starting(text, "It is a threshold fitted on these pairs.")
    assert numbers(line) == [str(settings.cache_candidates), str(settings.cache_floor)]
    assert "there is no evidence about real questions" in line
    assert "The cache simulated holds 165 questions." in text


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
        str(Settings().cache_threshold),
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


def test_the_roadmap_and_the_changelog_quote_the_second_run() -> None:
    summary = saved()
    cache = summary["cache_view"]
    reported = chosen(summary)[REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    threshold = str(Settings().cache_threshold)
    served_right = [str(reported[SAME][RIGHT]), str(reported[SAME]["pairs"])]
    roadmap = line_starting(
        (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8"), "- **The answer cache**"
    )
    changelog = line_starting(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        "- `handbook-rag eval-cache --no-declines`",
    )

    # The issue, the record, the threshold, the reporting half, the other splits, and
    # the requests of the check by hand, which no run holds.
    assert numbers(roadmap) == [
        "24",
        "0020",
        threshold,
        *served_right,
        str(cache["other_splits"]["splits"]),
        str(over["splits_with_something_wrong"]),
        "7",
    ]
    assert numbers(changelog) == [threshold, *served_right, "24"]
    for line in (roadmap, changelog):
        assert "served 16 of 49 rewordings and nothing wrong" in line
        assert "nothing is measured on real questions" in line
    assert (reported[SAME][WRONG], reported[DIFFERENT][WRONG]) == (0, 0)
    # Issue 24 is done: it is no longer in the list of what is open.
    whole = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    assert "Started since 1.0, and not finished" not in whole
    done = whole.index("Done since 1.0:")
    assert whole.index("- **The answer cache**") > done
    assert "issues/24" not in whole[:done]


def test_the_decision_record_gives_the_figures_of_the_two_runs() -> None:
    summary, first, settings = saved(), first_run(), Settings()
    cache = summary["cache_view"]
    weaker, mean = cache["policies"][ENTAILS_MIN], cache["policies"][ENTAILS_MEAN]
    reported, hardest = weaker[REPORTING], weaker["highest_wrong"][REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    fell = over["by_where_the_hardest_pair_fell"]
    before = chosen(first)[REPORTING]
    text = record()

    # The first run, as the context of the decision.
    assert numbers(said(text, "A second model does far better, and is not always right.")) == [
        str(before[SAME][RIGHT]),
        str(before[SAME]["pairs"]),
        str(before[DIFFERENT]["pairs"]),
        str(first["cache_view"]["other_splits"]["chosen_policy"]["splits_with_something_wrong"]),
        str(first["cache_view"]["other_splits"]["splits"]),
    ]
    assert f"On {first['pairs']['agreed']} labelled pairs" in text
    # The settings, as the table gives them.
    found = table(text, "| Setting | Default | What it is |")
    assert [row[:2] for row in found] == [
        ["`HANDBOOK_RAG_CACHE_MODE`", f"`{settings.cache_mode}`"],
        ["`HANDBOOK_RAG_CACHE_THRESHOLD`", str(settings.cache_threshold)],
        ["`HANDBOOK_RAG_CACHE_CANDIDATES`", str(settings.cache_candidates)],
        ["`HANDBOOK_RAG_CACHE_FLOOR`", str(settings.cache_floor)],
    ]
    assert (
        f"at most {settings.cache_candidates}, none below a cosine of {settings.cache_floor}"
        in (text)
    )
    assert f"That gave {settings.cache_threshold} by the weaker direction" in text
    # The second run, as the only evidence of how the threshold behaves.
    assert numbers(said(text, "On the half that chose nothing")) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
    ]
    start = "That half did not test it hard."
    assert numbers(said(text, start)) == [
        three(hardest[DIFFERENT]["score"]),
        three(hardest[SAME]["score"]),
        three(settings.cache_threshold - hardest[SAME]["score"]),
    ]
    assert named(said(text, start)) == ["s073"]
    start = "Where the threshold sits in its gap is a choice."
    assert numbers(said(text, start)) == [
        f"{weaker['decided_by']['score']:.4f}",
        f"{weaker['halfway_to']:.4f}",
        f"{weaker['decided_by']['score']:.4f}",
    ]
    start = "On 200 other splits of the same pairs"
    assert numbers(said(text, start)) == [
        str(cache["other_splits"]["splits"]),
        str(over["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits"]),
    ]
    start = "The mean of the verifier's two directions."
    assert numbers(said(text, start)) == [
        str(mean[REPORTING][SAME][RIGHT]),
        str(mean[REPORTING][SAME]["pairs"]),
        str(reported[SAME][RIGHT]),
    ]
    assert mean[REPORTING][SAME][WRONG] == 1, "and one more another stored question's answer"
    start = "A larger cache than the measurement's."
    assert numbers(said(text, start)) == [
        str(cache["stored_questions"]),
        str(cache["stored_declines"]["candidates_from"]),
    ]
    # What it never says of the cache.
    for word in ("safe", "guarantee", "proves", "proven"):
        assert word not in text.lower(), word


def test_the_settings_name_the_run_their_default_comes_from() -> None:
    run = "evals/runs/cache/no-decline-candidates"
    described = Settings.model_fields["cache_threshold"].description or ""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert (ROOT / run).is_dir()
    assert run in described
    assert run in " ".join(example.split()).replace("# ", "")
    for text in (described, example):
        assert "real questions" in text, "what the default says nothing about"


def test_the_two_saved_runs_hold_the_same_pairs_under_the_same_ids() -> None:
    first, second = rows(FIRST, "pairs.jsonl"), rows(RUN, "pairs.jsonl")

    assert list(first) == list(second) == [pair.pair for pair in pairs()]
    for pair in first:
        for name in ("kind", "agreed", "question_id", "other_id", "how", "what_differs"):
            assert second[pair][name] == first[pair][name], (pair, name)


def test_every_link_in_what_was_written_about_the_cache_leads_somewhere() -> None:
    observability = ROOT / "docs" / "observability.md"
    architecture = ROOT / "docs" / "architecture.md"
    written = [
        (RECORD, record()),
        (GUIDE, section(GUIDE.read_text(encoding="utf-8"), HEADING)),
        (CARD, card()),
        (
            observability,
            section(observability.read_text(encoding="utf-8"), "The trace of a cached answer"),
        ),
        (architecture, architecture.read_text(encoding="utf-8")),
        (ROOT / "CHANGELOG.md", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")),
        (ROOT / "docs" / "roadmap.md", (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")),
    ]
    checked = 0
    for document, text in written:
        # A link written out, or one defined at the foot of the changelog.
        targets = re.findall(r"\]\(([^)\s]+)\)", text)
        targets += re.findall(r"^\[[^\]]+\]: (\S+)$", text, flags=re.MULTILINE)
        for target in targets:
            if target.startswith(("http://", "https://")):
                continue
            path, _, anchor = target.partition("#")
            linked = document.parent / path if path else document
            assert linked.exists(), f"{document.name}: {target}"
            if anchor:
                assert anchor in anchors(linked.read_text(encoding="utf-8")), (
                    f"{document.name}: {target}"
                )
            checked += 1
    assert checked > 60


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
