"""The saved run of the follow-up rewriting is what today's code makes of its replies.

A run saves each reply as it came from the model, and the step's rules read it again at
every run. So the run's results and figures are only as good as the code that reads the
replies now. These tests read the saved replies and readings with today's code, the
verifier's recorded scores standing in for the model, and ask for the results and every
figure the run wrote. A change to the step's rules, to the decisions or to the counting
that would change a figure of the documents fails here first.

The run asked the first 72 cases. The 10 added after the review of 25 November are asked
in a top-up, which also rewrites the run's results and summary with today's code. Until
it is made, the tests that need it are skipped and say so, and the others hold what the
run has: its replies, read by today's rules, and the decisions that today's code makes
differently from the code that wrote the run's files.
"""

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.cache.verify import Verification
from handbook_rag.config import Settings
from handbook_rag.evals import follow_ups
from handbook_rag.evals.follow_ups import (
    DIFFERENT,
    SAME,
    TO_READ,
    VERIFIED_AT,
    FollowUpCase,
    FollowUpReply,
    FollowUpReport,
    check_readings,
    decide_instruction,
    decide_meaning,
    describe_rewriting,
    load_cases,
    load_readings,
    load_replies,
    pairs_to_verify,
    read_case_reply,
    step_phrases,
    summarise_follow_ups,
)
from tests.evals.test_committed_follow_up_cases import ADDED_AFTER_THE_REVIEW, FORMS_TAKEN_OUT

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "follow-ups" / "baseline"
CASES = ROOT / "evals" / "follow-ups" / "cases.jsonl"
READER = "claude-opus-5-5, one session, 25 November 2026"
# The planted instructions that today's code sends to a reader and the run's code held:
# their rewrites wait for a reader, and since the review an order obeyed without a
# canary is not told apart from such a rewrite by code.
READ_SINCE_THE_REVIEW = {"fu-planted-turn-1", "fu-planted-turn-5", "fu-planted-question-5"}
TOP_UP = (
    f"waits for the top-up: the {len(ADDED_AFTER_THE_REVIEW)} cases added after the review "
    "have no saved reply yet. `uv run handbook-rag eval-follow-ups --variant baseline` asks "
    f"them, one request each ({len(ADDED_AFTER_THE_REVIEW)} requests, more only for a "
    "failure tried again), and rewrites the run's results and summary with today's code."
)


def cases() -> list[FollowUpCase]:
    return load_cases(CASES)


def first_run() -> list[FollowUpCase]:
    """The cases the run asked first, before the review added the others."""
    return [case for case in cases() if case.case_id not in ADDED_AFTER_THE_REVIEW]


def replies() -> dict[str, FollowUpReply]:
    return load_replies(RUN / "replies.jsonl")


def top_up_made() -> bool:
    return {case.case_id for case in cases()} <= set(replies())


def results() -> dict[str, dict[str, Any]]:
    lines = (RUN / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {line["case"]: line for line in map(json.loads, lines)}


def summary() -> dict[str, Any]:
    return dict(json.loads((RUN / "summary.json").read_text(encoding="utf-8")))


def recorded_scores() -> dict[tuple[str, str], Verification]:
    """The verifier's readings as the run recorded them, for every pair it read.

    A row keeps the weaker direction against the form the verifier put nearest, which is
    the highest of the case's forms. Every other form is given nothing, which leaves the
    nearest the highest, as it was.
    """
    scores: dict[tuple[str, str], Verification] = {}
    rows = results()
    for case in cases():
        row = rows.get(case.case_id)
        if row is None or row["meaning_how"] not in ("verifier", "undecided"):
            continue
        for form in case.standalone:
            score = row["verifier_weaker"] if form == row["nearest_form"] else 0.0
            scores[(form, row["searched_for"])] = Verification(score, score)
    return scores


def todays_report(of: list[FollowUpCase]) -> FollowUpReport:
    """What today's code makes of the saved replies and readings of these cases."""
    asked = {case.case_id for case in of}
    readings = [r for r in load_readings(RUN / "readings.jsonl") if r.case_id in asked]
    saved = {case_id: reply for case_id, reply in replies().items() if case_id in asked}
    return summarise_follow_ups(of, saved, recorded_scores(), step_phrases(), readings)


def test_the_run_was_made_with_todays_instructions_layout_and_default_model() -> None:
    run = json.loads((RUN / "run.json").read_text(encoding="utf-8"))

    # A run made with other instructions or another layout would be another measurement,
    # and the top-up could not go on with it.
    assert run["config"] == describe_rewriting(Settings().answer_model)
    assert summary()["config"] == run["config"]
    assert run["attempts"] == summary()["attempts"] >= len(replies())


def test_the_run_holds_a_reply_for_every_case() -> None:
    if not top_up_made():
        pytest.skip(TOP_UP)

    assert set(replies()) == {case.case_id for case in cases()}


def test_every_saved_reply_is_read_by_todays_rules_as_the_run_recorded_it() -> None:
    saved, rows = replies(), results()

    assert set(saved) == set(rows)
    missing = {case.case_id for case in cases()} - set(saved)
    assert missing == (set() if top_up_made() else set(ADDED_AFTER_THE_REVIEW))
    for case in cases():
        if case.case_id in missing:
            continue
        reply, row = saved[case.case_id], rows[case.case_id]
        rewrite = read_case_reply(case, reply)
        assert reply.fingerprint == case.fingerprint == row["case_fingerprint"], case.case_id
        assert (rewrite.outcome, rewrite.question, rewrite.problem) == (
            row["outcome"],
            row["searched_for"],
            row["problem"],
        ), case.case_id
        assert row["reply"] == reply.text
        assert row["reply_fingerprint"] == follow_ups.reply_fingerprint(reply)


def test_every_decision_is_the_one_todays_code_makes() -> None:
    saved, rows, scores = replies(), results(), recorded_scores()
    asked = [case for case in cases() if case.case_id in saved]

    # The verifier read exactly the pairs today's code would give it.
    assert set(pairs_to_verify(asked, saved)) == set(scores)
    changed = set()
    for case in asked:
        row, reply = rows[case.case_id], saved[case.case_id]
        rewrite = read_case_reply(case, reply)
        meaning = decide_meaning(case, rewrite, scores)
        assert (meaning.by_code, meaning.how) == (row["meaning_by_code"], row["meaning_how"])
        if row["nearest_form"] in case.standalone or row["nearest_form"] is None:
            assert meaning.form == row["nearest_form"] or meaning.how in ("as_asked", "words")
        else:
            # The verifier put a form taken out after the review nearest, under the
            # threshold: every form left scored lower still, so the decision stands.
            assert row["nearest_form"] == FORMS_TAKEN_OUT[case.case_id]
            assert row["verifier_weaker"] < VERIFIED_AT
        instruction = decide_instruction(case, reply, rewrite, step_phrases(), meaning)
        by_code = None if instruction is None else instruction.by_code
        if by_code != row["instruction_by_code"]:
            assert (row["instruction_by_code"], by_code) == ("held", TO_READ), case.case_id
            changed.add(case.case_id)
    # The run's files were written by the code of the first run; the top-up rewrites them.
    assert changed == (set() if top_up_made() else READ_SINCE_THE_REVIEW)


def test_the_readings_are_of_the_saved_replies_by_one_reader() -> None:
    readings = load_readings(RUN / "readings.jsonl")
    rows = results()
    first = {case.case_id for case in first_run()}
    meanings = [r for r in readings if r.case_id in first and r.label in (SAME, DIFFERENT)]

    check_readings(cases(), replies(), readings)
    assert {reading.reader for reading in meanings} == {READER}
    # Each is of a rewrite of the first run that code set aside, and of its question.
    assert sorted(reading.case_id for reading in meanings) == sorted(
        case for case, row in rows.items() if case in first and row["meaning_by_code"] == TO_READ
    )
    assert Counter(reading.label for reading in meanings) == {SAME: 12, DIFFERENT: 7}


def test_today_the_first_run_waits_only_for_readings_of_the_instructions_set_aside() -> None:
    report = todays_report(first_run())

    assert (len(report.results), report.not_asked, report.failures) == (72, 0, [])
    waiting = {result.case.case_id for result in report.waiting}
    assert waiting <= READ_SINCE_THE_REVIEW
    for result in report.waiting:
        assert (result.meaning_waits, result.instruction_waits) == (False, True)


def test_every_figure_of_the_summary_is_what_todays_code_counts() -> None:
    if not top_up_made():
        pytest.skip(TOP_UP)
    readings = load_readings(RUN / "readings.jsonl")
    report = summarise_follow_ups(cases(), replies(), recorded_scores(), step_phrases(), readings)
    written = summary()

    assert {name: dataclasses.asdict(f) for name, f in report.figures.items()} == written["figures"]
    assert report.cost is not None
    assert dataclasses.asdict(report.cost) == written["cost"]
    assert dataclasses.asdict(report.checks) == written["checks"]
    assert (len(report.results), report.not_asked, written["cases_failed"]) == (
        len(cases()),
        0,
        [],
    )
    assert written["instruction_phrases"] == list(step_phrases())
    assert report.final is written["final"]
    waiting = [result.case.case_id for result in report.waiting]
    assert report.final, f"the run waits for readings of {waiting}"
