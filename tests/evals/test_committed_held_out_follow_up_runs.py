"""The two saved runs of the held-out follow-up cases are what the code made of them.

``held-out`` is the run as the command wrote it, under the rule for a reply without a
question mark as it stood before the run: a reply had to begin with a question word, a
verb that opens a question, a negated verb or a preposition followed by a question word.
That rule was changed after its replies were seen, and today's code no longer checks how
such a reply begins. So the first run is held to today's code with that one check put
back, and it must differ from today's reading in the two replies that check refused and
in nothing else.

``held-out-after`` holds the same replies, copied, read again with no request under the
rule as it stands, with two more readings of the two refused replies, read as if the step
had used them. It is held to today's code as the run of the 82 is.

The figures the documents give of both are worked out here, once, in
:func:`held_out_figures`, and the card's section on the held-out cases is held line by
line. No test here needs a model or sends a request: the verifier's scores are the ones
the runs recorded.
"""

import dataclasses
import functools
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import handbook_rag.evals.follow_ups as follow_ups
from handbook_rag.cache.verify import Verification
from handbook_rag.config import Settings
from handbook_rag.evals.follow_ups import (
    DIFFERENT,
    FOLLOW_UP,
    PLANTED,
    SAME,
    STANDALONE,
    TO_READ,
    FollowUpReply,
    FollowUpReport,
    describe_rewriting,
    load_readings,
    load_replies,
    read_case_reply,
    step_phrases,
    summarise_follow_ups,
)
from handbook_rag.rewrite import base as rewrite_base
from handbook_rag.rewrite.models import Rewrite, RewriteReply, Turn
from tests.evals.test_committed_held_out_follow_up_cases import (
    CARD,
    began_as_the_rule_before_asked,
    cases,
    section,
)
from tests.test_readme_results import assert_numbers, numbers, table

ROOT = Path(__file__).parents[2]
RUNS = ROOT / "evals" / "runs" / "follow-ups"
BEFORE = RUNS / "held-out"
AFTER = RUNS / "held-out-after"
FIRST_RUN = RUNS / "baseline"
READER = "claude-opus-5-5, second session, 25 November 2026"
# The hash of rewrite/base.py that the held-out run recorded: the code of the rule as it
# stood before the run. The file has changed since, so it is not today's.
CODE_BEFORE = "ad244d9a5babf645"
# The problem the rule before the change gave a reply that did not begin as it asked.
DID_NOT_BEGIN = "it has no question mark and does not begin as a question does"
# The two replies that rule refused, and the three rewrites read as another question.
REFUSED = ["ho-01", "ho-06"]
READ_DIFFERENT = ["ho-03", "ho-08", "ho-09"]
# By the account of the session that coordinated the work: when the rule took its last two
# openings, when the run was asked, and when the requirement for an opening was taken out.
OPENINGS_ADDED = ["25", "2026", "15", "09"]
ASKED = ["25", "2026", "16", "25", "16", "27"]
CHANGED = ["16", "29", "25", "2026"]
DECISION_RECORD = "0024"


def rows(run: Path) -> dict[str, dict[str, Any]]:
    lines = (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {line["case"]: line for line in map(json.loads, lines)}


def summary(run: Path) -> dict[str, Any]:
    return dict(json.loads((run / "summary.json").read_text(encoding="utf-8")))


def replies(run: Path) -> dict[str, FollowUpReply]:
    return load_replies(run / "replies.jsonl")


def recorded_scores(run: Path) -> dict[tuple[str, str], Verification]:
    """The verifier's scores as the run recorded them: the nearest form's, and nothing
    for the other forms, which leaves the nearest the highest, as it was."""
    scores: dict[tuple[str, str], Verification] = {}
    recorded = rows(run)
    for case in cases():
        row = recorded[case.case_id]
        if row["meaning_how"] not in ("verifier", "undecided"):
            continue
        for form in case.standalone:
            score = row["verifier_weaker"] if form == row["nearest_form"] else 0.0
            scores[(form, row["searched_for"])] = Verification(score, score)
    return scores


def read_as_the_rule_before(question: str, turns: Sequence[Turn], reply: RewriteReply) -> Rewrite:
    """Today's checks, with the check of how a reply begins put back where it stood.

    It came after the other checks of a reply without a question mark, and before the
    word shared with the conversation and the numbers.
    """
    today = rewrite_base.read_reply(question, turns, reply)
    later = ("it shares no word", "it adds a number")
    if today.outcome != "rewritten" and not (today.problem or "").startswith(later):
        return today
    text = rewrite_base._unwrapped(reply.text)
    asked_without_a_mark = not any(mark in question for mark in rewrite_base._QUESTION_MARKS)
    if (
        asked_without_a_mark
        and not text.endswith(rewrite_base._QUESTION_MARKS)
        and not began_as_the_rule_before_asked(text)
    ):
        return dataclasses.replace(
            today, question=question, outcome="unusable", problem=DID_NOT_BEGIN
        )
    return today


def report(run: Path) -> FollowUpReport:
    """What today's code makes of a run's saved replies and readings."""
    return summarise_follow_ups(
        cases(),
        replies(run),
        recorded_scores(run),
        step_phrases(),
        load_readings(run / "readings.jsonl"),
    )


def written(report: FollowUpReport) -> dict[str, Any]:
    """A report's figures, cost and checks as summary.json holds them."""
    assert report.cost is not None
    return {
        "figures": {name: dataclasses.asdict(f) for name, f in report.figures.items()},
        "cost": dataclasses.asdict(report.cost),
        "checks": json.loads(json.dumps(dataclasses.asdict(report.checks))),
        "final": report.final,
    }


def three(value: float) -> str:
    return f"{value:.3f}"


def share(estimate: dict[str, float]) -> list[str]:
    return [three(estimate["mean"]), three(estimate["low"]), three(estimate["high"])]


def shown(estimate: dict[str, float]) -> str:
    mean, low, high = share(estimate)
    return f"{mean} ({low} to {high})"


@functools.cache
def held_out_figures() -> dict[str, Any]:
    """Every figure the documents give of the held-out runs, from their saved files."""
    before, after, first = summary(BEFORE), summary(AFTER), summary(FIRST_RUN)
    figures = before["figures"]
    recorded = rows(BEFORE)
    by_kind = {
        kind: figures[f"{FOLLOW_UP}/{kind}"]
        for kind in ("subject_left_out", "pronoun", "third_turn")
    }

    def right(figure: dict[str, Any]) -> int:
        return int(
            figure["same_as_asked"]
            + figure["same_by_words"]
            + figure["same_by_verifier"]
            + figure["read_same"]
        )

    follow = figures[FOLLOW_UP]
    misses = [case for case, row in recorded.items() if row["meaning"] == DIFFERENT]
    return {
        "model": numbers(f"`{before['config']['rewrite_model']}`"),
        "reader": numbers(f"`{READER.split(',')[0]}`"),
        "cases": before["cases"],
        "attempts": before["attempts"],
        "replies": before["cases_replied"],
        "by_words": sum(figures[group]["same_by_words"] for group in follow_ups.KINDS),
        "by_verifier": sum(figures[group]["same_by_verifier"] for group in follow_ups.KINDS),
        "alone_given_back": figures[STANDALONE]["same_as_asked"],
        "refused": follow["different_as_asked"],
        "decided": sum(
            figures[group][name]
            for group in follow_ups.KINDS
            for name in ("same_by_words", "same_by_verifier", "same_as_asked", "different_as_asked")
        ),
        "set_aside": sum(figures[group]["set_aside"] for group in follow_ups.KINDS),
        "read_same": sum(figures[group]["read_same"] for group in follow_ups.KINDS),
        "read_different": sum(figures[group]["read_different"] for group in follow_ups.KINDS),
        "checks_read": before["checks"]["read"],
        "checks_agree": before["checks"]["agree"],
        "readings": len(load_readings(BEFORE / "readings.jsonl")),
        "follow_right": right(follow),
        "follow_of": follow["cases"],
        "follow": shown(follow["same"]),
        "follow_share": share(follow["same"]),
        "alone_right": right(figures[STANDALONE]),
        "alone_of": figures[STANDALONE]["cases"],
        "alone": shown(figures[STANDALONE]["same"]),
        "planted_right": right(figures[PLANTED]),
        "planted_of": figures[PLANTED]["cases"],
        "planted": shown(figures[PLANTED]["same"]),
        "held": figures[PLANTED]["held_by_code"] + figures[PLANTED]["read_held"],
        "held_share": shown(figures[PLANTED]["held"]),
        "by_kind": {kind: (right(figure), figure["cases"]) for kind, figure in by_kind.items()},
        "misses": len(misses),
        "model_misses": len(misses) - follow["different_as_asked"],
        "median": f"{before['cost']['seconds_median']:.2f}",
        "p95": f"{before['cost']['seconds_p95']:.2f}",
        "longest": f"{before['cost']['seconds_longest']:.2f}",
        "tokens_in": f"{before['cost']['input_tokens_mean']:.0f}",
        "tokens_out": f"{before['cost']['output_tokens_mean']:.0f}",
        "after_right": right(after["figures"][FOLLOW_UP]),
        "after_of": after["figures"][FOLLOW_UP]["cases"],
        "after_share": share(after["figures"][FOLLOW_UP]["same"]),
        "after_readings": len(load_readings(AFTER / "readings.jsonl")),
        "first_right": right(first["figures"][FOLLOW_UP]),
        "first_of": first["figures"][FOLLOW_UP]["cases"],
        "first_share": share(first["figures"][FOLLOW_UP]["same"]),
    }


# The run as the command wrote it, under the rule fixed before it.


def test_the_held_out_run_holds_one_reply_for_each_case_and_no_other_request() -> None:
    run = json.loads((BEFORE / "run.json").read_text(encoding="utf-8"))
    saved = replies(BEFORE)

    assert run["config"] == describe_rewriting(Settings().answer_model) == summary(BEFORE)["config"]
    assert set(saved) == {case.case_id for case in cases()}
    assert run["attempts"] == summary(BEFORE)["attempts"] == len(saved) == 20
    for case in cases():
        assert saved[case.case_id].fingerprint == case.fingerprint, case.case_id
    assert summary(BEFORE)["cases_failed"] == []


def test_the_held_out_run_was_decided_by_the_code_of_the_rule_before_the_change() -> None:
    today = hashlib.sha256(Path(rewrite_base.__file__).read_bytes()).hexdigest()[:16]

    assert summary(BEFORE)["code"]["rewrite/base.py"] == CODE_BEFORE != today
    assert summary(AFTER)["code"]["rewrite/base.py"] == today


def test_today_reads_the_held_out_replies_as_the_run_did_but_for_the_two_refused() -> None:
    saved, recorded = replies(BEFORE), rows(BEFORE)
    differ = []
    for case in cases():
        reply, row = saved[case.case_id], recorded[case.case_id]
        then = read_as_the_rule_before(case.sent_question, case.sent_turns, reply.as_reply())
        assert (then.outcome, then.question, then.problem) == (
            row["outcome"],
            row["searched_for"],
            row["problem"],
        ), case.case_id
        assert row["reply"] == reply.text
        assert row["reply_fingerprint"] == follow_ups.reply_fingerprint(reply)
        now = read_case_reply(case, reply)
        if (now.outcome, now.question) != (then.outcome, then.question):
            differ.append(case.case_id)
            assert (then.outcome, then.problem, now.outcome) == (
                "unusable",
                DID_NOT_BEGIN,
                "rewritten",
            )
            assert now.question == reply.text
            assert now.question.lower().startswith(("and ", "but "))

    assert differ == REFUSED


def test_every_figure_of_the_held_out_run_is_what_the_rule_before_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the check of how a reply begins put back, today's counting gives every figure
    the held-out run wrote, its readings and checks included."""
    monkeypatch.setattr(follow_ups, "read_reply", read_as_the_rule_before)
    then = report(BEFORE)
    recorded = summary(BEFORE)

    assert written(then) == {
        name: recorded[name] for name in ("figures", "cost", "checks", "final")
    }
    assert recorded["final"] is True
    assert [r.case.case_id for r in then.results if r.rewrite.outcome == "unusable"] == REFUSED
    assert (
        sorted(
            r.case.case_id
            for r in then.results
            if r.meaning_outcome == DIFFERENT and r.meaning.by_code == TO_READ
        )
        == READ_DIFFERENT
    )


def test_the_held_out_readings_are_five_decisions_and_five_checks_by_one_session() -> None:
    readings = load_readings(BEFORE / "readings.jsonl")
    recorded = rows(BEFORE)

    assert {reading.reader for reading in readings} == {READER}
    decisions = [r for r in readings if recorded[r.case_id]["meaning_by_code"] == TO_READ]
    checks = [r for r in readings if recorded[r.case_id]["meaning_how"] == "verifier"]
    assert [(r.case_id, r.label) for r in decisions] == [
        ("ho-03", DIFFERENT),
        ("ho-07", SAME),
        ("ho-08", DIFFERENT),
        ("ho-09", DIFFERENT),
        ("ho-12", SAME),
    ]
    assert [(r.case_id, r.label) for r in checks] == [
        (case, SAME) for case in ("ho-04", "ho-14", "ho-18", "ho-19", "ho-20")
    ]
    assert len(readings) == len(decisions) + len(checks) == 10
    # The two refused replies are not read in this run: code decided them, as asked.
    assert not {r.case_id for r in readings} & set(REFUSED)


# The same replies, read again under the rule as it stands.


def test_the_second_variant_holds_the_same_replies_and_made_no_request_of_its_own() -> None:
    for name in ("replies.jsonl", "run.json"):
        assert (AFTER / name).read_bytes() == (BEFORE / name).read_bytes(), name
    before = (BEFORE / "readings.jsonl").read_text(encoding="utf-8")
    after = (AFTER / "readings.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    added = load_readings(AFTER / "readings.jsonl")[len(load_readings(BEFORE / "readings.jsonl")) :]
    # The two refused replies, read as if the step had used them.
    assert [(r.case_id, r.label, r.reader) for r in added] == [
        (case, SAME, READER) for case in REFUSED
    ]


def test_the_second_variant_says_in_its_own_folder_what_it_is() -> None:
    note = (AFTER / "README.md").read_text(encoding="utf-8")

    assert "This is not a run of its own." in note
    assert "copies of the held-out run's, byte for byte" in note
    assert "none was made for this folder" in note
    assert "is not held-out evidence" in note
    assert f"count the {summary(AFTER)['attempts']} requests that run made" in note
    added = len(load_readings(AFTER / "readings.jsonl")) - len(
        load_readings(BEFORE / "readings.jsonl")
    )
    assert f"and {added} more, of the {len(REFUSED)} replies" in note
    assert summary(AFTER)["description"] == summary(BEFORE)["description"]
    assert "--variant held-out-after" in note


def test_today_reads_the_second_variant_as_it_was_written() -> None:
    saved, recorded = replies(AFTER), rows(AFTER)
    for case in cases():
        reply, row = saved[case.case_id], recorded[case.case_id]
        rewrite = read_case_reply(case, reply)
        assert (rewrite.outcome, rewrite.question, rewrite.problem) == (
            row["outcome"],
            row["searched_for"],
            row["problem"],
        ), case.case_id
    assert written(report(AFTER)) == {
        name: summary(AFTER)[name] for name in ("figures", "cost", "checks", "final")
    }
    assert summary(AFTER)["final"] is True
    # The two refused replies are used now, and set aside for the reader: the verifier
    # took neither for an acceptable form.
    for refused in REFUSED:
        assert (recorded[refused]["outcome"], recorded[refused]["meaning_by_code"]) == (
            "rewritten",
            TO_READ,
        )
        assert recorded[refused]["meaning"] == SAME
    # Nothing else moved between the two variants.
    before = rows(BEFORE)
    moved = [name for name in recorded if recorded[name]["meaning"] != before[name]["meaning"]]
    assert moved == REFUSED


# The card's section on the held-out cases.


def card_section() -> str:
    text = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    return re.sub(r"`ho-\d+`", "`a case`", text)


def test_every_number_of_the_cards_held_out_section_is_worked_out() -> None:
    k, every = held_out_figures(), cases()
    kinds = {kind: sum(case.kind == kind for case in every) for kind in follow_ups.GROUP_OF}
    turns = [sum(len(case.turns) == count for case in every) for count in (1, 2)]
    taken = sum(len(case.forms_from_second) for case in every)
    agreed = sum(bool(case.agreed) for case in every)
    by_kind = k["by_kind"]

    assert_numbers(
        card_section(),
        {
            "[`held-out/cases.jsonl`]": [str(len(every)), "82"],
            "- **In what order.**": [*OPENINGS_ADDED, "82", *ASKED],
            "By kind, they are": [
                str(kinds["subject_left_out"]),
                str(kinds["pronoun"]),
                str(kinds["third_turn"]),
                str(kinds["change_of_subject"]),
                str(kinds["planted_in_question"]),
                *map(str, turns),
                "6182",
                "82",
            ],
            "Another session of the same model, `": ["5", "5", str(len(every))],
            "- **Taken as acceptable forms": [str(taken), "82"],
            "- **Not taken.**": [str(agreed)],
            "- **The reader flagged one conversation": ["82"],
            "- **How they are asked.**": [str(len(every)), "82"],
            "`held-out`, with `": [
                *k["model"],
                str(k["attempts"]),
                str(k["replies"]),
                str(k["decided"]),
                str(k["by_words"]),
                str(k["by_verifier"]),
                str(k["alone_given_back"]),
                str(k["refused"]),
                str(k["set_aside"]),
                *k["reader"],
                str(k["read_same"]),
                str(k["read_different"]),
                str(k["checks_read"]),
                str(k["checks_agree"]),
                str(k["readings"]),
            ],
            "- **By kind**, of the follow-ups:": [
                str(value)
                for kind in ("subject_left_out", "pronoun", "third_turn")
                for value in by_kind[kind]
            ],
            "- **Lower than the first run, on few cases.**": [
                str(k["follow_right"]),
                str(k["follow_of"]),
                str(k["first_right"]),
                str(k["first_of"]),
                *k["first_share"],
                str(k["follow_of"]),
            ],
            "- **No planted instruction was followed.**": [
                str(k["held"]),
                str(k["planted_right"]),
                str(k["alone_right"]),
            ],
            "- **What it cost.**": [
                k["median"],
                k["p95"],
                k["longest"],
                k["tokens_in"],
                k["tokens_out"],
            ],
            "At 16:29 on": [*CHANGED, DECISION_RECORD],
            "- **The same 20 replies, read again under the changed rule**": [
                str(k["replies"]),
                str(k["readings"]),
                str(k["after_readings"] - k["readings"]),
                str(k["after_right"]),
                str(k["after_of"]),
                *k["after_share"],
            ],
            "- **The 82 cases, read again under the changed rule**": ["82"],
        },
    )
    assert table(card_section(), "| Held out, under the rule as it stood before the run |") == [
        [
            "Follow-ups rewritten to a question that means an acceptable form",
            str(k["follow_right"]),
            str(k["follow_of"]),
            k["follow"],
        ],
        [
            "Questions that stand alone, left alone",
            str(k["alone_right"]),
            str(k["alone_of"]),
            k["alone"],
        ],
        [
            "Planted instructions not followed",
            str(k["held"]),
            str(k["planted_of"]),
            k["held_share"],
        ],
        [
            "Follow-ups with a planted instruction that still got a question that stands alone",
            str(k["planted_right"]),
            str(k["planted_of"]),
            k["planted"],
        ],
    ]


def test_the_cards_held_out_section_names_the_misses_for_what_they_are() -> None:
    text = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    recorded, saved = rows(BEFORE), replies(BEFORE)

    refused = next(line for line in text.splitlines() if line.startswith("- **Two good rewrites"))
    assert re.findall(r"`(ho-\d+)`", refused) == REFUSED
    for case in REFUSED:
        assert f'"{saved[case].text}"' in refused
        assert f'"{recorded[case]["searched_for"]}"' in refused
    different = next(line for line in text.splitlines() if line.startswith("- **Three rewrites"))
    assert re.findall(r"`(ho-\d+)`", different)[:3] == READ_DIFFERENT
    assert "No person has read them." in text
    assert "is not held-out evidence" in text
    assert "has no held-out test of its own" in text
    # The status says the run is in, and the run is there.
    assert "> **Status: written, read a second time, asked once, and read.**" in text
    assert BEFORE.is_dir()
    assert AFTER.is_dir()
