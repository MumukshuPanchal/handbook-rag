"""The saved run of the follow-up rewriting is what today's code makes of its replies.

A run saves each reply as it came from the model, and the step's rules read it again at
every run. So the run's results and figures are only as good as the code that reads the
replies now. These tests read the saved replies and readings with today's code, the
verifier's recorded scores standing in for the model, and ask for the results and every
figure the run wrote. A change to the step's rules, to the decisions or to the counting
that would change a figure of the documents fails here first.
"""

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.cache.verify import Verification
from handbook_rag.config import Settings
from handbook_rag.evals import follow_ups
from handbook_rag.evals.follow_ups import (
    DIFFERENT,
    SAME,
    TO_READ,
    FollowUpCase,
    FollowUpReply,
    check_readings,
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

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "follow-ups" / "baseline"
CASES = ROOT / "evals" / "follow-ups" / "cases.jsonl"
READER = "claude-opus-5-5, one session, 25 November 2026"


def cases() -> list[FollowUpCase]:
    return load_cases(CASES)


def replies() -> dict[str, FollowUpReply]:
    return load_replies(RUN / "replies.jsonl")


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
        row = rows[case.case_id]
        if row["meaning_how"] not in ("verifier", "undecided"):
            continue
        for form in case.standalone:
            score = row["verifier_weaker"] if form == row["nearest_form"] else 0.0
            scores[(form, row["searched_for"])] = Verification(score, score)
    return scores


def test_the_run_was_made_with_todays_instructions_layout_and_default_model() -> None:
    run = json.loads((RUN / "run.json").read_text(encoding="utf-8"))

    # A run made with other instructions or another layout would be another measurement.
    assert run["config"] == describe_rewriting(Settings().answer_model)
    assert summary()["config"] == run["config"]
    assert run["attempts"] == summary()["attempts"] == 74


def test_every_saved_reply_is_read_by_todays_rules_as_the_run_recorded_it() -> None:
    saved, rows = replies(), results()

    assert set(saved) == {case.case_id for case in cases()} == set(rows)
    for case in cases():
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

    # The verifier read exactly the pairs today's code would give it.
    assert set(pairs_to_verify(cases(), saved)) == set(scores)
    for case in cases():
        row = rows[case.case_id]
        meaning = decide_meaning(case, read_case_reply(case, saved[case.case_id]), scores)
        assert (meaning.by_code, meaning.how) == (row["meaning_by_code"], row["meaning_how"])
        assert meaning.form == row["nearest_form"] or meaning.how in ("as_asked", "words")


def test_the_readings_are_of_the_saved_replies_by_one_reader() -> None:
    readings = load_readings(RUN / "readings.jsonl")
    rows = results()

    check_readings(cases(), replies(), readings)
    assert {reading.reader for reading in readings} == {READER}
    # Each is of a rewrite that code set aside, and of the question, not an instruction.
    assert sorted(reading.case_id for reading in readings) == sorted(
        case for case, row in rows.items() if row["meaning_by_code"] == TO_READ
    )
    assert Counter(reading.label for reading in readings) == {SAME: 12, DIFFERENT: 7}


def test_every_figure_of_the_summary_is_what_todays_code_counts() -> None:
    readings = load_readings(RUN / "readings.jsonl")
    report = summarise_follow_ups(cases(), replies(), recorded_scores(), step_phrases(), readings)
    written = summary()

    assert {name: dataclasses.asdict(f) for name, f in report.figures.items()} == written["figures"]
    assert report.cost is not None
    assert dataclasses.asdict(report.cost) == written["cost"]
    assert dataclasses.asdict(report.checks) == written["checks"]
    assert report.final is written["final"] is True
    assert (len(report.results), report.not_asked, written["cases_failed"]) == (72, 0, [])
    assert written["instruction_phrases"] == list(step_phrases())
