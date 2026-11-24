"""The third committed cache measurement: the cache as it is built, and its defaults.

``evals/runs/cache/as-built`` fills the simulated cache with the replies of the saved
answer run, and lets the answer cache's own rules decide which reply is kept and which
may be served to a question worded differently. No label says what a cache holds.

The first part holds the run to the pairs, to the saved replies and to itself. The
second holds the cache to the runs: its default threshold, its floor, its candidates and
the score it decides by, and that the cache's own code serves each question of the last
two runs what the simulation served it. The third holds the documents to the run, number
by number where a sentence carries a finding.

The figures of the one check with the real service, in the guide's last subsection, come
from no saved run, and no test can work them out again.
"""

import dataclasses
import hashlib
import inspect
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from handbook_rag import api, wiring
from handbook_rag.cache import rules
from handbook_rag.cache.answers import (
    DECIDED_BY,
    LOOKUP_SECONDS,
    AnswerCache,
    comparable,
    entry_id,
)
from handbook_rag.cache.store import Entry
from handbook_rag.cache.verify import Verification
from handbook_rag.config import Settings
from handbook_rag.evals import cache_pairs
from handbook_rag.evals.answers import load_records, replies_to_current_labels
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
    choose_threshold,
    highest_wrong_serve,
    load_cache_run,
    served,
    summarise,
    tally,
    unanswered_questions,
)
from handbook_rag.evals.cache_replies import as_the_cache_is_handed_it, stored_replies
from handbook_rag.evals.cache_report import report_lines
from handbook_rag.generate.base import holds_the_decline, plain_words
from handbook_rag.generate.models import Answer, Citation, Segment
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
from tests.support import make_chunk
from tests.test_readme_results import anchors, line_starting, numbers, table, three

ROOT = Path(__file__).parents[2]
RUNS = ROOT / "evals" / "runs" / "cache"
RUN = RUNS / "as-built"
SECOND = RUNS / "no-decline-candidates"
FIRST = RUNS / "baseline"
ANSWERS = ROOT / "evals" / "runs" / "answers" / "baseline"
# The labelled answers of the second judge: text the second guard was not written from.
JUDGED = ROOT / "evals" / "judge" / "correct" / "cases.jsonl"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "cache" / "README.md"
RECORD = ROOT / "docs" / "adr" / "0020-an-answer-cache-that-matches-by-meaning.md"
HEADING = "The answer cache, measured as it is built"
SECOND_HEADING = "The measurement again, with stored declines left out"
REAL_CHECK = "### Checked with the real service"
DEFAULTS = "### The defaults of the cache"
# The two ends of the thresholds that serve a run what the rule's threshold serves it, to
# four places. The tests below work them out from the runs. The documents quote them.
SAME_FROM = {RUN: 0.8540, SECOND: 0.8662}
SAME_UP_TO = 0.9131


def saved(run: Path = RUN) -> Any:
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def rows(run: Path, name: str) -> dict[str, Any]:
    lines = (run / name).read_text(encoding="utf-8").splitlines()
    return {row["pair"]: row for row in map(json.loads, lines)}


def asked(run: Path = RUN) -> list[Asked]:
    return load_cache_run(run, pairs())


def agreed(run: Path = RUN) -> list[Asked]:
    return [question for question in asked(run) if question.pair.agreed]


def weaker(summary: Any) -> Any:
    """The figures of the score the cache decides by."""
    return summary["cache_view"]["policies"][ENTAILS_MIN]


def held() -> Any:
    """What the run says its cache held: the replies, sorted by the cache's rules."""
    return saved()["cache_view"]["stored_replies"]


def kept_out() -> dict[str, list[str]]:
    kept: dict[str, list[str]] = held()["kept_from_the_second_tier"]
    return kept


def serves(questions: Sequence[Asked], threshold: float) -> list[str | None]:
    """What each question is served at a threshold, by the score the cache decides by."""
    return [served(question, DECIDED_BY, threshold) for question in questions]


def at_floor(questions: Sequence[Asked], floor: float) -> list[Asked]:
    """The questions with the candidates a higher floor would have left them.

    A run keeps the nearest stored questions above its own floor. Those of them that are
    above a higher floor are exactly the nearest above that one, so no model is needed.
    """
    return [
        dataclasses.replace(
            question,
            candidates=tuple(c for c in question.candidates if c.scores.cosine >= floor),
        )
        for question in questions
    ]


def read_by_the_second_check(questions: Sequence[Asked]) -> int:
    """How many candidates the questions have between them."""
    return sum(len(question.candidates) for question in questions)


def the_reply_to(question_id: str) -> Answer:
    """A saved reply of the answer run, as the pipeline would hand it to the cache."""
    (question,) = [question for question in golden() if question.id == question_id]
    records = replies_to_current_labels(golden(), load_records(ANSWERS / "answers.jsonl"))
    return as_the_cache_is_handed_it(question, records[question_id])


# --- the run -----------------------------------------------------------------------------


def test_the_third_run_was_made_as_the_first_but_for_the_replies_it_was_filled_with() -> None:
    summary, first = saved(), first_run()
    replies = (ANSWERS / "answers.jsonl").read_bytes()

    assert summary["config"] == {
        **first["config"],
        "replies_from": "evals/runs/answers/baseline",
        "replies_fingerprint": hashlib.sha256(replies).hexdigest()[:16],
    }
    assert summary["description"] == (
        f"{first['description']} The cache holds the replies of the answer run "
        "evals/runs/answers/baseline, and the answer cache's own rule decides which of them "
        "are candidates."
    )
    assert sorted(path.name for path in RUN.iterdir()) == [
        "pairs.jsonl",
        "questions.jsonl",
        "summary.json",
    ]
    assert summary["rows"] == cache_pairs.rows_fingerprints(RUN)
    assert summary["measured_on"] == first["measured_on"]
    # It is no run by the label: nothing in it says that declines were left out.
    assert "stored_declines" not in summary["cache_view"]
    # The two earlier runs are as they were: neither speaks of replies.
    for earlier in (FIRST, SECOND):
        assert "stored_replies" not in saved(earlier)["cache_view"]
        assert "replies_from" not in saved(earlier)["config"]


def test_the_figures_of_the_third_run_are_what_its_rows_give() -> None:
    summary = saved()
    config = summary["config"]

    again = summarise(
        asked(),
        stored=held()["kept"],
        candidates=config["candidates"],
        floor=config["floor"],
        unanswered=unanswered_questions(golden()),
        seed=config["split_seed"],
        other_splits=config["other_splits"],
        stored_replies=held(),
    )

    assert set(summary) == {"description", "measured_on", "config", "rows", "timing", *again}
    for part, figures in again.items():
        assert summary[part] == json.loads(json.dumps(figures)), part
    report = report_lines(summary)
    assert (
        "  filled with the replies of evals/runs/answers/baseline: 165 kept, and 124 of them "
        "candidates by the cache's own rule"
    ) in report
    assert (
        "  kept from the second tier: 36 decline, 0 declines_in_part, "
        "4 says_something_is_missing, 1 cites_nothing"
    ) in report
    assert not [line for line in report if "stored decline" in line]


def test_what_the_run_held_is_what_the_caches_rules_give_on_the_saved_replies_today() -> None:
    """The run was made once. This holds it to the rules as they are now.

    A change to the rules that sorts one saved reply another way, a phrase added to the
    second guard for one, fails here: the run then describes another cache, and has to
    be made again.
    """
    today = stored_replies(golden(), ANSWERS).figures(golden())

    # All but the name of the answer run, which is recorded as it was given to the command.
    assert {**today, "from": held()["from"]} == held()
    assert held()["from"] == "evals/runs/answers/baseline"
    assert (ROOT / held()["from"]).samefile(ANSWERS)
    assert (held()["kept"], held()["candidates"]) == (165, 124)
    assert {reason: len(ids) for reason, ids in kept_out().items()} == {
        rules.DECLINE: 36,
        rules.DECLINES_IN_PART: 0,
        rules.SAYS_SOMETHING_IS_MISSING: 4,
        rules.CITES_NOTHING: 1,
    }
    assert kept_out()[rules.SAYS_SOMETHING_IS_MISSING] == ["q092", "q112", "u004", "u006"]
    assert kept_out()[rules.CITES_NOTHING] == ["q064"]
    assert (held()["not_kept"], held()["without_a_reply"]) == ([], [])
    # Where the replies and the labels part ways: no question that the handbook does not
    # answer is a candidate, and eight that it answers are none.
    unanswered = unanswered_questions(golden())
    assert held()["candidates_the_handbook_does_not_answer"] == []
    assert held()["no_candidates_though_the_handbook_answers"] == [
        "q037",
        "q064",
        "q085",
        "q088",
        "q092",
        "q100",
        "q108",
        "q112",
    ]
    declined = set(kept_out()[rules.DECLINE])
    assert len(declined & unanswered) == 31
    assert sorted(declined - unanswered) == ["q037", "q085", "q088", "q100", "q108"]
    assert sorted(unanswered - declined) == ["u004", "u006"]


def phrases_in(text: str) -> list[str]:
    """The phrases of the second guard that a text holds."""
    said = f" {plain_words(text)} "
    return [phrase for phrase in rules.MISSING_PHRASES if f" {phrase} " in said]


def test_the_relatives_of_the_four_phrases_find_nothing_more_in_the_saved_replies() -> None:
    """The list holds far more phrases than the model used. None of them costs anything here.

    Four phrases were read from the saved replies, and the rest are their relatives. On
    every answer at hand, of the answer run and of the adversarial run, the whole list
    finds what the four find and nothing else.
    """
    the_four = {"does not cover", "do not cover", "do not specify", "not provided"}
    assert the_four <= set(rules.MISSING_PHRASES)
    assert len(rules.MISSING_PHRASES) > 1000

    records = replies_to_current_labels(golden(), load_records(ANSWERS / "answers.jsonl"))
    found = {
        question_id: phrases_in(record.text)
        for question_id, record in records.items()
        if record.answered
    }
    matched = {question_id: phrases for question_id, phrases in found.items() if phrases}

    assert len(records) == 165
    assert sorted(matched) == kept_out()[rules.SAYS_SOMETHING_IS_MISSING]
    assert {phrase for phrases in matched.values() for phrase in phrases} == the_four
    # The replies of the adversarial run are never cached: an evaluation gets no cache.
    # They were read for the words a model uses, and none of its answers holds a phrase.
    lines = (ROOT / "evals" / "runs" / "adversarial" / "baseline" / "replies.jsonl").read_text(
        encoding="utf-8"
    )
    replies = [json.loads(line) for line in lines.splitlines() if line.strip()]
    answers = [reply for reply in replies if reply["answered"]]
    assert (len(replies), len(answers)) == (36, 17)
    for reply in answers:
        text = "".join(statement["text"] for statement in reply["statements"])
        assert phrases_in(text) == [], reply["question_id"]


def written_to_give_no_answer() -> dict[str, str]:
    """The replies a model session wrote, for the second judge's labelled answers, to give
    none of what was asked: "the handbook does not say", "it depends", a referral."""
    cases = [json.loads(line) for line in JUDGED.read_text(encoding="utf-8").splitlines() if line]
    return {case["case"]: case["answer"] for case in cases if case["kind"] == "no_answer"}


def test_the_second_guard_finds_one_of_five_replies_it_was_not_written_from() -> None:
    """The list was written from four replies of one answer run, and finds all four.

    That says nothing of other replies. These five were written by another session for
    another purpose, before the guard existed. Each gives none of what was asked, which
    is more than the gap in part that the guard is for, and it finds one of them.
    """
    replies = written_to_give_no_answer()

    found = {case: phrases_in(answer) for case, answer in replies.items()}

    assert len(replies) == 5
    assert {case: phrases for case, phrases in found.items() if phrases} == {
        "q006-v1": ["does not specify"]
    }
    missed = [answer for case, answer in replies.items() if not found[case]]
    assert len(missed) == 4
    # Each of the four says it in words the list does not hold, as the guide quotes them.
    line = said(found_by_the_run(), "On replies it was not written from, it finds 1 of 5.")
    quoted = re.findall(r'"([^"]+)"', line)
    assert quoted == [
        "does not specify",
        "there is no single answer",
        "I don't know",
        "I'm not sure",
        "I wasn't able to find",
    ]
    assert quoted[0] in rules.MISSING_PHRASES
    for words in quoted[1:]:
        assert sum(words in answer for answer in missed) == 1, words
    assert numbers(line) == ["1", str(len(replies)), str(len(replies))]
    # What the list is fitted to: four phrases of its 1,093 are in a reply the model wrote.
    line = said(found_by_the_run(), "The list is fitted to four replies.")
    assert numbers(line) == [f"{len(rules.MISSING_PHRASES):,}", "4"]
    assert len(kept_out()[rules.SAYS_SOMETHING_IS_MISSING]) == 4
    assert "It reads English only." in line
    # The record and the README give the same figure, and the same warning about d067.
    line = said(record(), "An answer in which one of two guards finds that part of the question")
    assert numbers(line) == [str(len(replies)), "1"]
    assert "**The list is fitted to four replies of one answer run**, and it reads English" in line
    assert (
        "That `d067` is not served in the third run is that fit, and no test of the guard."
        in record()
    )
    assert (
        "This is the reply the list was fitted to, so it shows that the fit holds. It is no "
        "test of the guard."
    ) in said(found_by_the_run(), "`d067` is not served.")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "The list was fitted to that reply and three others, so this is no test of it: of 5 "
        "replies it was not written from, it finds 1"
    ) in readme
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Of 5 replies it was not written from, it finds 1." in changelog


def test_no_candidate_of_the_third_run_holds_a_reply_the_cache_keeps_from_the_second_tier() -> None:
    left_out = {question_id for ids in kept_out().values() for question_id in ids}
    now, before = rows(RUN, "questions.jsonl"), rows(SECOND, "questions.jsonl")

    met = {c["question_id"] for row in now.values() for c in row["candidates"]}
    met_before = {c["question_id"] for row in before.values() for c in row["candidates"]}

    assert len(left_out) == 41
    assert not met & left_out
    # The run by the label did offer some of them: replies a label takes for answers.
    assert met_before & left_out
    assert not met_before & unanswered_questions(golden())
    # A candidate that both runs hold is the same stored question with the same cosine.
    for pair in now:
        earlier = {c["question_id"]: c for c in before[pair]["candidates"]}
        for candidate in now[pair]["candidates"]:
            if candidate["question_id"] in earlier:
                same = earlier[candidate["question_id"]]
                assert math.isclose(candidate[COSINE], same[COSINE], abs_tol=1e-8)


def largest_difference_of_the_pair_scores() -> float:
    now, before = rows(RUN, "pairs.jsonl"), rows(FIRST, "pairs.jsonl")
    assert list(now) == list(before) == [pair.pair for pair in pairs()]
    for pair in now:
        assert now[pair]["half"] == before[pair]["half"], "the same split, by the same seed"
        for name in ("kind", "agreed", "question_id", "other_id", "how", "what_differs"):
            assert now[pair][name] == before[pair][name], (pair, name)
    return max(
        abs(float(now[pair][score]) - float(before[pair][score]))
        for pair in now
        for score in (COSINE, "entails_forward", "entails_backward")
    )


def test_each_pair_scores_against_its_own_golden_question_as_in_the_first_run() -> None:
    # The verifier read other batches this time, which moves a score in its last digits.
    assert largest_difference_of_the_pair_scores() <= 1e-8


def test_the_last_two_runs_differ_in_two_rewordings_of_questions_the_model_declined() -> None:
    rule = weaker(saved())["threshold"]
    assert rule == weaker(saved(SECOND))["threshold"]

    differ = {
        now.pair.pair: (served(before, DECIDED_BY, rule), served(now, DECIDED_BY, rule))
        for now, before in zip(asked(), asked(SECOND), strict=True)
        if served(before, DECIDED_BY, rule) != served(now, DECIDED_BY, rule)
    }

    # By the label the handbook answers q037 and q088, and the model declined both.
    assert differ == {"s027": ("q037", None), "s061": ("q088", None)}
    assert {"q037", "q088"} <= set(kept_out()[rules.DECLINE])
    counted, counted_before = weaker(saved()), weaker(saved(SECOND))
    assert (counted[CHOOSING][SAME][RIGHT], counted_before[CHOOSING][SAME][RIGHT]) == (18, 20)
    assert counted[REPORTING] == counted_before[REPORTING]
    # For 7 of the 98 rewordings the reply to their own question is kept from the second
    # tier, and the other 5 were not served before either.
    without = [
        question
        for question in agreed()
        if question.pair.kind == SAME
        and question.pair.question_id not in [c.question_id for c in question.candidates]
        and question.pair.question_id in {q for ids in kept_out().values() for q in ids}
    ]
    own = saved()["cache_view"]["own_question"][SAME]
    assert len(without) == own["pairs"] - own["among_candidates"] == 7
    assert len(without) - len(differ) == 5
    # The second guard cost no rewording: the two it could have cost score next to nothing.
    by_the_guard = {
        question.pair.pair: question
        for question in without
        if question.pair.question_id in kept_out()[rules.SAYS_SOMETHING_IS_MISSING]
    }
    assert {name: q.pair.question_id for name, q in by_the_guard.items()} == {
        "s064": "q092",
        "s077": "q112",
    }
    for question in by_the_guard.values():
        assert three(question.own.of(DECIDED_BY)) == "0.001"


# --- the cache ---------------------------------------------------------------------------


def thresholds_that_serve_the_same(run: Path) -> list[float]:
    """Every threshold of four places, in the region of the rule's, that serves each
    question of a run exactly what the rule's own threshold serves it."""
    everyone = asked(run)
    at_the_rule = serves(everyone, weaker(saved(run))["threshold"])
    return [n / 10**4 for n in range(8300, 9300) if serves(everyone, n / 10**4) == at_the_rule]


@pytest.mark.parametrize("run", [RUN, SECOND], ids=["as built", "by the label"])
def test_the_default_threshold_is_the_top_of_the_range_that_serves_a_run_the_same(
    run: Path,
) -> None:
    summary, default = saved(run), Settings().cache_threshold
    rule = weaker(summary)["threshold"]

    # The score the rule took is the one the cache decides by, and its threshold is the
    # same in both runs.
    assert summary["cache_view"]["chosen_policy"] == DECIDED_BY == ENTAILS_MIN
    assert f"{rule:.4f}" == "0.8669"
    # One unbroken range of thresholds serves the run what the rule's threshold serves it.
    same = thresholds_that_serve_the_same(run)
    assert same == [n / 10**4 for n in range(round(SAME_FROM[run] * 10**4), 9132)]
    assert (same[0], same[-1]) == (SAME_FROM[run], SAME_UP_TO)
    # The default is its top, to two places.
    assert default == math.floor(SAME_UP_TO * 100) / 100 == 0.91
    assert default in same
    everyone = agreed(run)
    assert tally(everyone, DECIDED_BY, default) == tally(everyone, DECIDED_BY, rule)
    # It stands further above the highest score at which the run holds a wrong serve.
    worst = highest_wrong_serve(everyone, DECIDED_BY)
    assert worst is not None
    assert (worst.pair, worst.served, three(worst.score)) == ("s073", "q110", "0.854")
    assert (three(default - worst.score), three(rule - worst.score)) == ("0.056", "0.013")
    # With every pair in view, which is how it was chosen: nothing wrong at the default.
    assert tally(everyone, DECIDED_BY, default).wrong == 0


def test_the_gap_the_default_stands_in_is_between_a_wrong_serve_and_a_right_one() -> None:
    near = sorted(
        (candidate.scores.of(DECIDED_BY), question.pair.pair, candidate.question_id)
        for question in agreed()
        for candidate in question.candidates
        if 0.84 <= candidate.scores.of(DECIDED_BY) <= 0.93
    )

    # No score of the run lies between the two.
    (below, above) = near
    assert (f"{below[0]:.4f}", below[1], below[2]) == ("0.8540", "s073", "q110")
    assert (f"{above[0]:.4f}", above[1], above[2]) == ("0.9132", "s071", "q105")
    (reworded,) = [question for question in agreed() if question.pair.pair == "s071"]
    assert (reworded.pair.kind, reworded.pair.question_id) == (SAME, "q105")
    # The second run holds one more score there, of a rewording whose question the model
    # declined in the answer run: s074 is why its range starts higher.
    more = sorted(
        (f"{candidate.scores.of(DECIDED_BY):.4f}", question.pair.pair, candidate.question_id)
        for question in agreed(SECOND)
        for candidate in question.candidates
        if 0.84 <= candidate.scores.of(DECIDED_BY) <= 0.93
    )
    assert more == [
        ("0.8540", "s073", "q110"),
        ("0.8661", "s074", "q108"),
        ("0.9132", "s071", "q105"),
    ]
    assert "q108" in kept_out()[rules.DECLINE]


def test_the_default_floor_changes_nothing_that_any_of_the_three_runs_served() -> None:
    settings = Settings()
    floor = settings.cache_floor
    assert floor == 0.7
    read = {}

    for run in (FIRST, SECOND, RUN):
        summary = saved(run)
        assert summary["config"]["floor"] == cache_pairs.FLOOR == 0.5
        everyone = asked(run)
        narrowed = at_floor(everyone, floor)
        half = {pair: row["half"] for pair, row in rows(run, "pairs.jsonl").items()}
        choosing = [q for q in narrowed if q.pair.agreed and half[q.pair.pair] == CHOOSING]
        assert len(choosing) == 99
        # The rule gives the same threshold for each of the three scores.
        for policy in POLICIES:
            chosen = summary["cache_view"]["policies"][policy]["threshold"]
            assert choose_threshold(choosing, policy).value == chosen, (run.name, policy)
        # And no question is served anything else, at the rule's threshold or the default.
        for threshold in (weaker(summary)["threshold"], settings.cache_threshold):
            assert serves(narrowed, threshold) == serves(everyone, threshold), run.name
        read[run.name] = (
            read_by_the_second_check([q for q in everyone if q.pair.agreed]),
            read_by_the_second_check([q for q in narrowed if q.pair.agreed]),
        )
        # What the floor was read from, with every pair in view.
        under = [
            candidate.scores.of(DECIDED_BY)
            for question in everyone
            for candidate in question.candidates
            if candidate.scores.cosine < 0.7612
        ]
        assert max(under) < 0.13
        own = [q.own.cosine for q in everyone if q.pair.agreed and q.pair.kind == SAME]
        assert three(min(own)) == "0.737"
        assert min(own) > floor, "every rewording still meets its own stored question"

    assert read == {
        "baseline": (594, 298),
        "no-decline-candidates": (592, 247),
        "as-built": (592, 234),
    }
    # It does nothing for correctness: the wrong serve that scores highest is above it.
    for run in (SECOND, RUN):
        worst = highest_wrong_serve(agreed(run), DECIDED_BY)
        assert worst is not None
        (candidate,) = [
            candidate
            for question in agreed(run)
            if question.pair.pair == worst.pair
            for candidate in question.candidates
            if candidate.question_id == worst.served
        ]
        assert three(candidate.scores.cosine) == "0.789"
        assert candidate.scores.cosine >= floor
    # In the first run it is another serve, of a stored decline, and that one is nearer.
    first = highest_wrong_serve(agreed(FIRST), DECIDED_BY)
    assert first is not None
    assert (first.pair, first.served) == ("d067", "u004")
    (stored_decline,) = [
        candidate
        for question in agreed(FIRST)
        if question.pair.pair == first.pair
        for candidate in question.candidates
        if candidate.question_id == first.served
    ]
    assert three(stored_decline.scores.cosine) == "0.962"
    assert stored_decline.scores.cosine >= floor
    described = Settings.model_fields["cache_floor"].description or ""
    assert "of 0.789 in the last two runs and of 0.962 in the first" in described


def test_the_caches_other_settings_are_those_of_the_run() -> None:
    summary, settings = saved(), Settings()

    assert settings.cache_candidates == summary["config"]["candidates"] == cache_pairs.CANDIDATES
    assert settings.cache_candidates == 3
    assert settings.cache_verifier_model == summary["config"]["verifier_model"]
    assert settings.embedding_model == summary["config"]["embedding_model"]
    assert settings.embedding_query_prefix == summary["config"]["embedding_query_prefix"]


class Replayed:
    """A store that holds, for one question, the candidates the saved run found for it."""

    def __init__(self, question: Asked) -> None:
        self.question = question

    def ensure(self, dimension: int) -> None:
        return None

    def get(self, entry_id: str) -> Entry | None:
        return None

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        # A stored question is named by its id, which stands in for its text here. The
        # floor is the cache's to apply: a store that forgot it must change nothing.
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


@pytest.mark.parametrize(
    ("run", "served_right"), [(RUN, 34), (SECOND, 36)], ids=["as built", "by the label"]
)
def test_the_cache_serves_each_question_of_a_run_what_the_simulation_served_it(
    run: Path, served_right: int
) -> None:
    """The measurement counts what a simulated cache serves. This is the built one.

    Each question of the run is put to the cache's own code, with the run's candidates
    and scores in place of the store and the two models. With the default settings, a
    stricter threshold and a higher floor than the run's own, it has to serve the stored
    question the simulation served, or nothing where that served nothing: the same
    candidates, the same order, the same score, the same edge.
    """
    summary, settings = saved(run), Settings()
    by_meaning = 0

    for question in asked(run):
        cache = AnswerCache(
            Replayed(question),
            embedder=AnyVector(),
            verifier=ReplayedScores(question),
            threshold=settings.cache_threshold,
            candidates=settings.cache_candidates,
            floor=settings.cache_floor,
        )

        found = cache.lookup(question.pair.other).answer

        expected = served(question, DECIDED_BY, weaker(summary)["threshold"])
        got = None if found is None or found.cached is None else found.cached.question
        assert got == expected, question.pair.pair
        if found is not None and found.cached is not None:
            by_meaning += 1
            assert found.cached.tier == "semantic"
            (candidate,) = [c for c in question.candidates if c.question_id == expected]
            assert found.cached.score == candidate.scores.of(DECIDED_BY)
            assert found.cached.similarity == candidate.scores.cosine
    # Every serve of the run, on both halves, and none of them to a disputed pair.
    counted = weaker(summary)
    right = counted[CHOOSING][SAME][RIGHT] + counted[REPORTING][SAME][RIGHT]
    assert by_meaning == right == served_right


def test_without_the_second_guard_the_question_of_d067_would_be_served() -> None:
    settings = Settings()
    (question,) = [question for question in asked() if question.pair.pair == "d067"]
    reply = the_reply_to("u004")

    # As the run has it: the reply to its golden question is no candidate.
    assert question.pair.question_id == "u004"
    assert "u004" in kept_out()[rules.SAYS_SOMETHING_IS_MISSING]
    assert "u004" not in [candidate.question_id for candidate in question.candidates]
    assert served(question, DECIDED_BY, settings.cache_threshold) is None
    # Only the second guard keeps it out: it is an answer, it cites a passage, and it does
    # not hold the fixed sentence of a decline.
    assert rules.kept_from_the_second_tier(reply) == rules.SAYS_SOMETHING_IS_MISSING
    assert reply.answered
    assert reply.sources
    assert not holds_the_decline(reply.segments)
    assert "the specific monetary amount allocated is not provided in the handbook" in reply.text
    # A candidate, it would be the nearest, above the floor and above the threshold.
    assert question.own.cosine > max(c.scores.cosine for c in question.candidates)
    assert question.own.cosine >= settings.cache_floor
    assert question.own.of(DECIDED_BY) >= settings.cache_threshold
    assert (three(question.own.cosine), three(question.own.of(DECIDED_BY))) == ("0.962", "0.989")
    # And the rule would have had to stay above it, as it did in the first run.
    assert rows(RUN, "pairs.jsonl")["d067"]["half"] == CHOOSING
    decided = weaker(first_run())["decided_by"]
    assert (decided["pair"], decided["served"]) == ("d067", "u004")
    assert weaker(first_run())["threshold"] > question.own.of(DECIDED_BY)
    # The fact for the other side: the reply names the quarters the question asks about.
    assert "For which quarters" in question.pair.other
    assert "FY26-Q2 to FY26-Q4" in reply.text
    assert question.pair.kind == DIFFERENT


# --- the documents -----------------------------------------------------------------------


def guide() -> str:
    """The guide's section on the third run, without the check that no run holds."""
    whole = section(GUIDE.read_text(encoding="utf-8"), HEADING)
    assert whole.count(REAL_CHECK) == 1
    return whole.split(REAL_CHECK)[0]


def found_by_the_run() -> str:
    """The section up to the defaults: what the run was filled with and what it found."""
    assert guide().count(DEFAULTS) == 1
    return guide().split(DEFAULTS)[0]


def defaults() -> str:
    return guide().split(DEFAULTS)[1]


def card() -> str:
    return section(CARD.read_text(encoding="utf-8"), "Measured as the cache is built")


def record() -> str:
    return RECORD.read_text(encoding="utf-8")


def said(text: str, start: str) -> str:
    return line_starting(text, f"- **{start}")


def test_the_guides_table_of_the_third_run_is_the_run() -> None:
    policies = saved()["cache_view"]["policies"]
    found = table(guide(), "| Filled with replies, decided by | Threshold |")

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
        assert policies[policy][CHOOSING][SAME][RIGHT] == 18
    assert "served 18 of 49 rewordings right, so the rule takes the weaker direction once more" in (
        guide()
    )


def test_what_the_guide_says_the_cache_holds_is_what_the_run_holds() -> None:
    summary = saved()
    text = found_by_the_run()
    unanswered = unanswered_questions(golden())
    declined = kept_out()[rules.DECLINE]
    missing = kept_out()[rules.SAYS_SOMETHING_IS_MISSING]

    # Why the label is not enough, on the one pair that shows it.
    start = "The run above takes a label's word for what a cache holds."
    line = line_starting(text, start)
    answered_in_part = sorted(set(missing) & unanswered)
    (near_miss,) = [question for question in asked() if question.pair.pair == "d067"]
    assert numbers(line) == [
        str(len(set(declined) - unanswered)),
        str(len(answered_in_part)),
        str(len(answered_in_part)),
        three(near_miss.own.of(DECIDED_BY)),
    ]
    assert named(line) == ["u004", "d067", "u004", "d067"]
    assert answered_in_part == ["u004", "u006"]
    assert '"the specific monetary amount allocated is not provided in the handbook"' in line
    assert "--variant as-built --replies evals/runs/answers/baseline" in text

    line = said(text, "What the cache does with the 165 replies.")
    assert numbers(line) == [
        str(held()["kept"]),
        str(held()["candidates"]),
        str(held()["kept"] - held()["candidates"]),
        str(len(declined)),
        str(len(missing)),
        str(len(kept_out()[rules.CITES_NOTHING])),
    ]
    assert kept_out()[rules.DECLINES_IN_PART] == [], "no reply holds the fixed sentence in part"
    assert "No reply holds the fixed sentence of a decline beside an answer." in line

    line = said(text, "The 36 declines")
    assert numbers(line) == [
        str(len(declined)),
        str(len(set(declined) & unanswered)),
        str(len(unanswered)),
        str(len(set(declined) - unanswered)),
    ]
    assert named(line) == sorted(set(declined) - unanswered)

    line = said(text, "The 4 answers that say something is missing")
    assert numbers(line) == [str(len(missing)), str(len(answered_in_part))]
    assert named(line) == [*missing, *answered_in_part, "q092", "q112"]
    # Each is quoted with the phrase the guard found in it.
    quoted = dict(re.findall(r"`([qu]\d{3})` \(\"([a-z ]+)\"\)", line))
    assert sorted(quoted) == missing
    for question_id, phrase in quoted.items():
        assert phrase in rules.MISSING_PHRASES, phrase
        assert f" {phrase} " in f" {plain_words(the_reply_to(question_id).text)} "

    line = said(text, "The 1 answer that cites no passage")
    assert (numbers(line), named(line)) == (["1"], kept_out()[rules.CITES_NOTHING])
    assert numbers(said(text, "What did not change.")) == [
        str(summary["config"]["candidates"]),
        str(summary["config"]["floor"]),
        "0.00000001",
    ]
    assert largest_difference_of_the_pair_scores() <= 0.00000001


def an_answer(text: str) -> Answer:
    passage = make_chunk("Dental care and vision care are optional benefits.")
    return Answer(
        question="a question",
        segments=(Segment(text, (Citation(passage, "optional benefits"),)),),
        answered=True,
        model="a-model",
        stop_reason="end_turn",
    )


def test_what_the_documents_say_of_the_second_guard_is_what_it_does() -> None:
    line = said(found_by_the_run(), "A guard for a gap that is said in the model's own words.")

    # The two things that follow from a list of phrases, said in the guide and the record.
    for text in (line, record()):
        assert "**It errs towards keeping an answer out**" in text
        assert "**And a reply can get past it**" in text
        assert '"dental care is not covered" is a plain answer' in text
    assert "The instructions fix no words for that, so the guard is a list of phrases" in line
    for phrase in re.findall(r'"([a-z ]+)"', line.split("**It errs")[0]):
        assert phrase in rules.MISSING_PHRASES, phrase
    # The example that errs is kept out, and a gap in other words is a candidate.
    assert rules.kept_from_the_second_tier(an_answer("Dental care is not covered.")) == (
        rules.SAYS_SOMETHING_IS_MISSING
    )
    assert rules.kept_from_the_second_tier(an_answer("The handbook is silent on the amount.")) is (
        None
    )
    # The instructions do ask for the gap to be said, in whatever words.
    assert '"say plainly what is not covered"' in record()
    assert "say plainly what is not covered" in " ".join(
        (ROOT / "src" / "handbook_rag" / "generate" / "prompt.py")
        .read_text(encoding="utf-8")
        .replace("\\\n", "")
        .split()
    )


def test_what_the_guide_says_the_third_run_found_is_what_the_run_holds() -> None:
    summary, second = saved(), saved(SECOND)
    cache = summary["cache_view"]
    figures, mean = weaker(summary), cache["policies"][ENTAILS_MEAN]
    reported, there = figures[REPORTING], figures["reporting_by_how"]
    text = found_by_the_run()

    assert f"`as-built`, on {day(summary)}." in text
    start = "The rule gives the threshold of the second run: 0.8669."
    assert numbers(said(text, start)) == [
        f"{figures['threshold']:.4f}",
        f"{figures['decided_by']['score']:.4f}",
        f"{figures['halfway_to']:.4f}",
    ]
    assert figures["threshold"] == weaker(second)["threshold"]
    assert named(said(text, start)) == [figures["decided_by"]["pair"]] == ["d004"]
    assert figures["decided_by"] == weaker(second)["decided_by"], "again"

    start = "`d067` is not served."
    (near_miss,) = [question for question in asked() if question.pair.pair == "d067"]
    assert numbers(said(text, start)) == [
        str(len(kept_out()[rules.SAYS_SOMETHING_IS_MISSING])),
        three(near_miss.own.cosine),
        three(near_miss.own.of(DECIDED_BY)),
    ]
    assert named(said(text, start)) == ["d067", "u004"]
    start = "The reply that is kept from `d067` names the quarters it asks about."
    assert (numbers(said(text, start)), named(said(text, start))) == ([], ["d067", "u004"])
    assert "the rule counts by labels" in said(text, start)

    start = "It served 16 of 49 rewordings on the reporting half, and nothing wrong."
    assert numbers(said(text, start)) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
        *(
            str(there[how][key])
            for how in ("reworded", "synonyms", "short", "long", "indirect")
            for key in (RIGHT, "pairs")
        ),
    ]
    assert (reported[SAME][WRONG], reported[DIFFERENT][WRONG]) == (0, 0), "and nothing wrong"
    assert there == weaker(second)["reporting_by_how"], "each count is the second run's"

    start = "On the choosing half it served 18 rewordings, where the second run served 20."
    own = cache["own_question"][SAME]
    assert numbers(said(text, start)) == [
        str(figures[CHOOSING][SAME][RIGHT]),
        str(weaker(second)[CHOOSING][SAME][RIGHT]),
        str(own["pairs"] - own["among_candidates"]),
        str(own["pairs"]),
        str(own["pairs"] - own["among_candidates"] - 2),
    ]
    assert named(said(text, start)) == ["s027", "s061", "q037", "q088"]
    start = "The second guard cost no rewording here."
    assert (numbers(said(text, start)), named(said(text, start))) == (
        ["0.001"],
        ["s064", "s077", "q092", "q112"],
    )

    start = "The reporting half did not test this threshold hard."
    hardest = figures["highest_wrong"][REPORTING]
    assert numbers(said(text, start)) == [
        three(hardest[DIFFERENT]["score"]),
        three(hardest[SAME]["score"]),
    ]
    assert named(said(text, start)) == [hardest[SAME]["pair"], hardest[SAME]["served"]]
    assert named(said(text, start)) == ["s073", "q110"]
    assert hardest[SAME]["own"] is False, "another stored question's answer"
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
    assert numbers(said(text, "With every pair in view the rule gives 0.8836")) == [
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
    # How far it goes: one answer run, and a list written after its replies were read.
    line = line_starting(text, "How far this goes:")
    assert numbers(line) == [
        str(held()["kept"]),
        "4",
        str(len(written_to_give_no_answer())),
        "1",
    ]
    assert "The list of phrases was written after these replies had been read." in line


def test_what_the_guide_says_of_the_defaults_is_what_the_runs_hold() -> None:
    """The numbers of the subsection, each from what the tests of the cache work out."""
    summary, settings = saved(), Settings()
    figures = weaker(summary)
    reported = figures[REPORTING][SAME]
    text = defaults()
    worst = highest_wrong_serve(agreed(), DECIDED_BY)
    assert worst is not None

    line = line_starting(text, "**The threshold.**")
    assert numbers(line) == [
        str(settings.cache_threshold),
        f"{figures['threshold']:.4f}",
        f"{worst.score:.4f}",
        f"{figures['halfway_to']:.4f}",
        f"{SAME_FROM[RUN]:.4f}",
        f"{SAME_UP_TO:.4f}",
        f"{figures['threshold']:.4f}",
        f"{SAME_FROM[SECOND]:.4f}",
        f"{SAME_UP_TO:.4f}",
    ]
    assert named(line) == [worst.pair] == ["s073"]
    assert "The default is the top of that range, to two places." in line
    assert numbers(said(text, "It costs nothing on these runs.")) == [
        str(reported[RIGHT]),
        str(reported["pairs"]),
        three(reported["right_share"]["mean"]),
        three(reported["right_share"]["low"]),
        three(reported["right_share"]["high"]),
    ]
    start = "It stands further above the highest score at which the two runs hold a wrong serve."
    assert numbers(said(text, start)) == [
        three(settings.cache_threshold - worst.score),
        three(worst.score),
        three(figures["threshold"] - worst.score),
    ]
    start = "Where it stands in the gap was chosen with every pair in view."
    assert numbers(said(text, start)) == [str(settings.cache_threshold), str(len(agreed()))]
    assert "Nothing here tests that choice." in said(text, start)

    line = line_starting(text, "**The floor.**")
    assert numbers(line) == [str(settings.cache_floor), str(summary["config"]["floor"])]
    assert "with every pair in view" in line
    assert numbers(said(text, "What it was read from.")) == ["0.7612", "0.13", "0.737"]
    start = "What it changes in the runs: nothing that is served."
    assert numbers(said(text, start)) == [
        str(settings.cache_floor),
        str(settings.cache_threshold),
    ]
    read = {
        run: (
            read_by_the_second_check(at_floor(agreed(run), settings.cache_floor)),
            read_by_the_second_check(agreed(run)),
        )
        for run in (RUN, SECOND, FIRST)
    }
    assert numbers(said(text, "What it saves.")) == [
        str(len(agreed())),
        *(str(count) for run in (RUN, SECOND, FIRST) for count in read[run]),
    ]
    start = "It does nothing for correctness."
    assert (numbers(said(text, start)), named(said(text, start))) == (["0.789"], ["s073"])
    assert numbers(line_starting(text, "**The candidates.**")) == [str(settings.cache_candidates)]
    line = line_starting(text, "These defaults are fitted on these pairs.")
    assert "there is no evidence about real questions" in line


def test_every_number_the_guide_gives_of_the_third_run_is_a_number_of_a_run() -> None:
    """The wider net of the first run's test, over the section on the third."""
    settings = Settings()
    allowed = numbers_of_the_run(saved()) | numbers_of_the_run(saved(SECOND))
    allowed |= numbers_of_the_run(first_run())
    allowed |= {
        "2026",
        "0.00000001",  # held above to the largest difference between the runs
        f"{len(rules.MISSING_PHRASES):,}",  # the phrases of the second guard
        # What the cache holds, counted from the run's own lists of replies, above.
        str(held()["kept"] - held()["candidates"]),
        *(str(len(ids)) for ids in kept_out().values()),
        "31",
        "5",
        "2",
        "7",
        # The near-miss the second guard keeps out, and the two rewordings it did not cost.
        "0.962",
        "0.989",
        "0.001",
        # The defaults, and what the tests of the cache work out for them, above.
        str(settings.cache_threshold),
        str(settings.cache_floor),
        f"{SAME_FROM[RUN]:.4f}",
        f"{SAME_FROM[SECOND]:.4f}",
        f"{SAME_UP_TO:.4f}",
        "0.056",
        "0.013",
        "0.7612",
        "0.13",
        "0.737",
        "0.789",
        "234",
        "247",
        "298",
        "592",
        "594",
    }

    unknown = [number for number in numbers(guide()) if number not in allowed]

    assert unknown == []


def test_what_the_card_says_of_the_third_run_is_what_the_run_holds() -> None:
    summary, settings = saved(), Settings()
    cache = summary["cache_view"]
    figures = weaker(summary)
    reported, hardest = figures[REPORTING], figures["highest_wrong"][REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    unanswered = unanswered_questions(golden())
    declined = kept_out()[rules.DECLINE]
    text = card()

    assert f"on {day(summary)}, and saved the run as" in text
    assert "(../runs/cache/as-built)" in text
    assert numbers(said(text, "What the cache holds.")) == [
        str(held()["kept"]),
        str(held()["candidates"]),
        str(held()["kept"] - held()["candidates"]),
        str(len(declined)),
        str(len(kept_out()[rules.SAYS_SOMETHING_IS_MISSING])),
        str(len(kept_out()[rules.CITES_NOTHING])),
        str(len(declined)),
        str(len(set(declined) & unanswered)),
        str(len(unanswered)),
        str(len(set(declined) - unanswered)),
    ]
    start = "One pair shows why the label is not enough."
    (near_miss,) = [question for question in asked() if question.pair.pair == "d067"]
    assert numbers(said(text, start)) == [three(near_miss.own.of(DECIDED_BY))]
    assert named(said(text, start)) == ["u004", "d067"]
    assert "It errs towards keeping an answer out, and a reply in other words gets past it." in (
        said(text, start)
    )
    start = "The rule gives the same threshold: 0.8669 by the weaker direction"
    assert numbers(said(text, start)) == [
        f"{figures['threshold']:.4f}",
        f"{figures['decided_by']['score']:.4f}",
        str(settings.cache_threshold),
        f"{SAME_FROM[RUN]:.4f}",
        f"{SAME_UP_TO:.4f}",
    ]
    assert named(said(text, start)) == ["d004"]
    start = "On the reporting half it served"
    assert numbers(said(text, start)) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
        str(figures[CHOOSING][SAME][RIGHT]),
        str(figures[CHOOSING][SAME]["pairs"]),
        str(weaker(saved(SECOND))[CHOOSING][SAME][RIGHT]),
    ]
    assert named(said(text, start)) == ["s027", "s061"]
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


def test_the_roadmap_and_the_changelog_quote_the_third_run() -> None:
    summary, settings = saved(), Settings()
    cache = summary["cache_view"]
    figures = weaker(summary)
    reported = figures[REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    served_right = [str(reported[SAME][RIGHT]), str(reported[SAME]["pairs"])]
    roadmap = line_starting(
        (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8"), "- **The answer cache**"
    )
    changelog = line_starting(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        "- `handbook-rag eval-cache --no-declines`",
    )

    # The issue, the record, the default, the reporting half, the other splits with the
    # rule's own threshold, and the requests of the two checks by hand, which no run holds.
    assert numbers(roadmap) == [
        "24",
        "0020",
        str(settings.cache_threshold),
        *served_right,
        str(cache["other_splits"]["splits"]),
        f"{figures['threshold']:.4f}",
        str(over["splits_with_something_wrong"]),
        "7",
        "2",
    ]
    assert "It was checked with the real service twice" in roadmap
    assert numbers(changelog) == [
        *served_right,
        f"{figures['threshold']:.4f}",
        str(settings.cache_threshold),
        str(settings.cache_floor),
        str(settings.cache_candidates),
        "24",
    ]
    for line in (roadmap, changelog):
        assert "served 16 of 49 rewordings and nothing wrong" in line
        assert "nothing is measured on real questions" in line
    assert (reported[SAME][WRONG], reported[DIFFERENT][WRONG]) == (0, 0)
    # Each names the run by what it is: the second by the label, the third as built.
    assert "by the golden set's label (the saved run `evals/runs/cache/no-decline-candidates`)" in (
        changelog
    )
    assert "measures the cache as it is built" in changelog
    assert "(the saved run `evals/runs/cache/as-built`)" in changelog
    assert "chosen with every pair in view" in changelog
    assert "does nothing for correctness" in changelog
    # Issue 24 is done: it is in the list of what is done, and neither in the list of what
    # is open nor among what was started and is not finished.
    whole = (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8")
    done = whole.index("Done since 1.0:")
    started = whole.find("Started since 1.0, and not finished:")
    finished = whole[done:] if started == -1 else whole[done:started]
    assert "- **The answer cache**" in finished
    assert "issues/24" not in whole[:done]
    assert started == -1 or "issues/24" not in whole[started:]
    # What an exact guard waits for is in the list of what comes next.
    assert 'a fixed form for "this part is not in the handbook"' in whole[:done]


def test_the_decision_record_gives_the_figures_of_the_runs() -> None:
    summary, first, settings = saved(), first_run(), Settings()
    cache = summary["cache_view"]
    figures, mean = weaker(summary), cache["policies"][ENTAILS_MEAN]
    reported, hardest = figures[REPORTING], figures["highest_wrong"][REPORTING]
    over = cache["other_splits"]["policies"][ENTAILS_MIN]
    fell = over["by_where_the_hardest_pair_fell"]
    before = weaker(first)[REPORTING]
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
    # What the cache's own code does with the saved replies.
    line = line_starting(text, "On the 165 replies of the saved")
    assert numbers(line) == [
        str(held()["kept"]),
        str(held()["kept"] - held()["candidates"]),
        str(len(kept_out()[rules.DECLINE])),
        str(len(kept_out()[rules.SAYS_SOMETHING_IS_MISSING])),
        str(len(kept_out()[rules.CITES_NOTHING])),
        str(len(kept_out()[rules.SAYS_SOMETHING_IS_MISSING])),
        "2",
    ]
    answerable = set(kept_out()[rules.SAYS_SOMETHING_IS_MISSING]) - unanswered_questions(golden())
    assert len(answerable) == 2
    assert "The first guard finds none." in line
    # The two later runs, each named for what it decides by.
    assert "**With stored declines left out by the golden set's label**" in text
    assert "**As the cache is built** ([`as-built`](../../evals/runs/cache/as-built))" in text
    assert "The defaults rest on this run." in text
    line = line_starting(text, "The rule is the first run's:")
    assert numbers(line) == [f"{figures['threshold']:.4f}"]
    assert "halfway to the next score" in line
    assert figures["threshold"] == weaker(saved(SECOND))["threshold"], "in both later runs"
    assert numbers(said(text, "On the half that chose nothing")) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
    ]
    start = "That half did not test it hard."
    assert numbers(said(text, start)) == [
        three(hardest[DIFFERENT]["score"]),
        three(hardest[SAME]["score"]),
    ]
    assert named(said(text, start)) == ["s073"]
    start = "On 200 other splits of the same pairs"
    assert numbers(said(text, start)) == [
        str(cache["other_splits"]["splits"]),
        str(over["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits"]),
    ]
    # The two defaults, and what each rests on.
    worst = highest_wrong_serve(agreed(), DECIDED_BY)
    assert worst is not None
    line = line_starting(text, "**The default threshold is 0.91, and not the rule's 0.8669.**")
    assert numbers(line) == [
        str(settings.cache_threshold),
        f"{figures['threshold']:.4f}",
        f"{figures['decided_by']['score']:.4f}",
        f"{figures['halfway_to']:.4f}",
        three(worst.score),
        f"{SAME_FROM[RUN]:.4f}",
        f"{SAME_UP_TO:.4f}",
        f"{figures['threshold']:.4f}",
        f"{SAME_FROM[SECOND]:.4f}",
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        three(settings.cache_threshold - worst.score),
        three(worst.score),
        three(figures["threshold"] - worst.score),
    ]
    assert "That choice was made with every pair in view, so nothing tests it." in line
    line = line_starting(text, "**The default floor is 0.7, where all three runs were made with")
    assert numbers(line) == [
        str(settings.cache_floor),
        str(summary["config"]["floor"]),
        "0.7612",
        "0.13",
        "0.737",
        str(settings.cache_floor),
        str(read_by_the_second_check(at_floor(agreed(), settings.cache_floor))),
        str(len(agreed())),
        str(read_by_the_second_check(agreed())),
        three(worst.score),
        "0.789",
    ]
    assert "**The floor does nothing for correctness**" in line
    start = "The mean of the verifier's two directions."
    assert numbers(said(text, start)) == [
        str(mean[REPORTING][SAME][RIGHT]),
        str(mean[REPORTING][SAME]["pairs"]),
        str(reported[SAME][RIGHT]),
        str(mean[REPORTING][SAME][WRONG]),
    ]
    start = "A larger cache than the measurement's."
    assert numbers(said(text, start)) == [str(held()["kept"]), str(held()["candidates"])]
    # What one rewording in three stood for: the count, the half and the interval.
    line = line_starting(text, "- A repeated question costs no request")
    assert numbers(line) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        three(reported[SAME]["right_share"]["mean"]),
        three(reported[SAME]["right_share"]["low"]),
        three(reported[SAME]["right_share"]["high"]),
    ]
    assert "In the third run, on the half of the labelled pairs that chose nothing" in line
    # What is still missing, in the words the decision asks for.
    assert "It is planned with the next change of the instructions, and is not in this change." in (
        said(text, "An exact guard for a decline in part.")
    )
    line = said(text, "A label for what the second guard keeps from `d067`.")
    assert named(line) == ["d067", "u004", "d067"]
    assert "The pair's label says `different`, and the rule counts by labels." in line
    # What the time limit of a lookup does not count, and what a lookup that was given
    # up leaves alone. The tests of the cache hold both.
    assert "**What is slow only the first time in a process is not counted**" in text
    assert "**A lookup that was given up starts nothing more.**" in text
    # What `index` cannot reach.
    assert (
        "An answer that a running service is still writing when `index` empties the caches is "
        "stored afterwards, and stays."
    ) in text
    # The second check with the real service is given as what it is.
    whose = (
        "It is the account of the session that coordinated this change, of a single run on "
        "one laptop, and no file of it is in the repository."
    )
    assert whose in text
    assert whose in GUIDE.read_text(encoding="utf-8")
    # What it never says of the cache.
    for word in ("safe", "guarantee", "proves", "proven"):
        assert word not in text.lower(), word


def test_the_sentences_that_said_more_than_is_known_are_gone() -> None:
    """Each was true of less than it said. The corrected ones are held above."""
    documents = {
        path.name: path.read_text(encoding="utf-8")
        for path in (RECORD, GUIDE, CARD, ROOT / "README.md", ROOT / "CHANGELOG.md")
    }
    documents["architecture.md"] = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for name, text in documents.items():
        for gone in (
            "What the second tier never serves",
            "An answer that declines part of the question.",
            "an answer that declines in part and an answer that cites nothing are served",
            "the answers stored before are never served again",
            "empties the cache whenever it changes the index",
            "the settings refuse one",
            "about one rewording in three",
            "552 MB",
            "everything that decides what the right answer is",
            "changes with everything that decides an answer",
            "is one of the 33, so the wrong serve",
            "the loosest threshold from which upwards",
            "the threshold is the loosest one",
            # The check with the real service was made a second time.
            "was not made again",
            "Checked once with the real service",
            "checked once with the real service",
        ):
            assert gone not in text, (name, gone)
    # Only the third run is called the cache as it is built.
    assert (
        "measures the cache as it is built, with stored declines left out"
        not in (documents["CHANGELOG.md"])
    )
    assert (
        "was run again as the cache is built, with stored declines left out"
        not in (documents[RECORD.name])
    )
    second = section(documents[GUIDE.name], SECOND_HEADING)
    assert "- **So this run does not measure the cache as it is built.**" in second
    # One unit for the size of the second model, beside the memory it takes.
    for name in (RECORD.name, GUIDE.name, "README.md", "CHANGELOG.md"):
        assert "552 MiB" in documents[name], name
    assert (
        "The settings refuse a model name that ends in `-latest`. They know no other alias."
        in (documents[RECORD.name])
    )


def test_the_readme_gives_the_range_the_default_is_the_top_of() -> None:
    # The rest of the README's section is held in tests/test_readme_results.py.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        f"Every threshold from {SAME_FROM[RUN]:.4f} to {SAME_UP_TO:.4f} serves each question "
        "of the run the same, and the default is the top of that range, to two places."
    ) in readme


def test_the_documents_give_the_time_limit_of_a_lookup_as_the_code_has_it() -> None:
    limit = f"{LOOKUP_SECONDS:.0f} seconds"
    assert limit == "2 seconds"
    documents = {
        RECORD: "**A lookup waits 2 seconds at most.**",
        ROOT / "README.md": "- **A lookup waits 2 seconds at most.**",
        ROOT / "CHANGELOG.md": "A lookup waits 2 seconds at most, and then the model is asked.",
        ROOT / "docs" / "architecture.md": "A lookup waits 2 seconds at most",
        ROOT / "docs" / "observability.md": "given up after its 2 seconds",
    }
    for document, sentence in documents.items():
        assert limit in sentence
        assert sentence in document.read_text(encoding="utf-8"), document.name
    # What stands before an answer without it, as the record says: the store's own limits.
    assert "10 seconds in the service and 60 on the command line" in record()
    assert api.STORE_TIMEOUT_SECONDS == 10.0
    assert inspect.signature(wiring.make_store).parameters["timeout"].default == 60.0
    assert "The limit is a constant in the code, `LOOKUP_SECONDS`, and no setting." in record()
    assert not [name for name in Settings.model_fields if "lookup" in name]


def test_the_settings_name_the_run_their_defaults_come_from() -> None:
    run = "evals/runs/cache/as-built"
    described = Settings.model_fields["cache_threshold"].description or ""
    floor = Settings.model_fields["cache_floor"].description or ""
    example = " ".join((ROOT / ".env.example").read_text(encoding="utf-8").split())
    example = example.replace("# ", "")

    assert (ROOT / run).samefile(RUN)
    assert run in described
    assert run in example
    for text in (described, example):
        assert "real questions" in text, "what the default says nothing about"
        assert "0.8669" in text, "the rule's own threshold, beside the default"
        assert "0.9131" in text
        assert "chosen with every pair in view, so nothing tests that choice" in text
    for text in (floor, example):
        assert "does nothing for correctness" in text
    assert "with every pair in view" in floor
    assert f"HANDBOOK_RAG_CACHE_THRESHOLD={Settings().cache_threshold}" in example
    assert f"HANDBOOK_RAG_CACHE_FLOOR={Settings().cache_floor}" in example
    assert f"HANDBOOK_RAG_CACHE_CANDIDATES={Settings().cache_candidates}" in example


def test_every_link_in_what_was_written_about_the_cache_leads_somewhere() -> None:
    observability = ROOT / "docs" / "observability.md"
    architecture = ROOT / "docs" / "architecture.md"
    guide_text = GUIDE.read_text(encoding="utf-8")
    written = [
        (RECORD, record()),
        (GUIDE, section(guide_text, SECOND_HEADING)),
        (GUIDE, section(guide_text, HEADING)),
        (CARD, CARD.read_text(encoding="utf-8")),
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
    assert checked > 70
    # The files a document names in code, where a reader will look for them.
    for named_file in (
        "src/handbook_rag/cache/rules.py",
        "src/handbook_rag/evals/cache_replies.py",
        "src/handbook_rag/measuring.py",
        "tests/cache/test_real_server.py",
    ):
        assert (ROOT / named_file).is_file(), named_file
        assert f"`{named_file}`" in guide_text + record(), named_file


# --- the real models ---------------------------------------------------------------------


@pytest.mark.slow
def test_the_real_models_give_the_candidates_and_scores_of_the_third_run_again() -> None:
    settings = Settings()
    config = saved()["config"]
    measured = cache_pairs.measure(
        pairs(),
        golden(),
        embedder=wiring.make_embedder(wiring.settings_for_serving(settings)),
        verifier=wiring.make_verifier(settings),
        candidates=config["candidates"],
        floor=config["floor"],
        held=stored_replies(golden(), ANSWERS).may_be_candidates,
    )

    assert measured.stored == held()["kept"]
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
