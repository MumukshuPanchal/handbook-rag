"""What the documents say of the follow-up rewriting is what the saved run holds.

The run itself is held to today's code by ``test_committed_follow_up_run.py``, and the
cases to their card by ``test_committed_follow_up_cases.py``. These tests hold the
documents written from them: the decision record (ADR 0022), the section of the
evaluation guide and what the guide says of the run elsewhere, the card's results, the
lines of the changelog and the roadmap, and the architecture page. The README's section is
held by ``tests/test_readme_results.py``, with the README's other results.

Every figure is worked out once, in :func:`figures`, from the saved run, the cases, the
second reading and the cache's labelled pairs. A table is built again from the run. Every
line of prose that holds a number is compared with the numbers it must hold, in their
order, and a line with a number that no test works out fails
(``test_readme_results.assert_numbers``). So a figure changed in one document and not in
another fails, a number added by hand fails, and so does a run made again without the
documents being brought up to date.

What no test can hold is the reading of the figures, and what was not done. The documents
say that themselves: no answer was written for the rewritten questions, it is one run,
model sessions wrote and read the cases and one reader decided the rewrites code set
aside, and the step is not wired in.
"""

import functools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.evals.follow_ups import (
    DIFFERENT,
    FOLLOW_UP,
    KINDS,
    PLANTED,
    SAME,
    STANDALONE,
    TO_READ,
    VERIFIED_AT,
    FollowUpCase,
    load_cases,
    load_readings,
    load_replies,
)
from handbook_rag.evals.metrics import share_with_interval
from handbook_rag.generate.base import plain_words
from handbook_rag.rewrite.models import MAX_EARLIER_TURNS
from tests.evals.test_committed_follow_up_cases import (
    DETAIL_HIGH,
    DETAIL_LOW,
    FEWEST_WORDS,
    SAME_WITH_A_DETAIL_ADDED,
    flagged_by_a_simple_test,
    makers_forms,
)
from tests.test_readme_results import (
    anchors,
    assert_numbers,
    in_words,
    line_starting,
    numbers,
    table,
)

ROOT = Path(__file__).parents[2]
ADR = ROOT / "docs" / "adr" / "0022-follow-up-questions-rewritten-before-the-search.md"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "follow-ups" / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
ROADMAP = ROOT / "docs" / "roadmap.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
RUN = ROOT / "evals" / "runs" / "follow-ups" / "baseline"
CASES = ROOT / "evals" / "follow-ups" / "cases.jsonl"
CACHE_PAIRS = ROOT / "evals" / "runs" / "cache" / "baseline" / "pairs.jsonl"
ISSUE = "https://github.com/MumukshuPanchal/handbook-rag/issues/"
# Numbers that are names: the decision record, the record of the two-passage questions,
# and the issues of follow-up questions and of query rewriting.
RECORD, TWO_PASSAGE_RECORD, FOLLOW_UPS, REWRITING = "0022", "0021", "79", "22"
# The night of the run, by the account of the session that made it: its files hold no
# date. And the date of the decision record.
NIGHT = ["24", "25", "2026"]
DATE = ["2026", "11", "25"]
# The thresholds of the line on waits: within so many seconds, and more than so many.
QUICK, SLOW = 2, 10

KIND_NAMES = {
    "subject_left_out": "Subject left out",
    "pronoun": "Pronoun",
    "points_into_answer": "Pointing into the previous answer",
    "third_turn": "Third turn of a chain",
    "two_referents": "Two possible referents",
    "standalone_same_subject": "On the subject of the earlier turn",
    "change_of_subject": "On another subject",
    "planted_in_turn": "Planted in an earlier turn",
    "planted_in_question": "Planted in the question",
}
GROUP_NAMES = {
    FOLLOW_UP: "Follow-ups",
    STANDALONE: "Questions that stand alone",
    PLANTED: "Follow-ups with a planted instruction",
}
# The kinds where a follow-up leans on an earlier question, the first three of them as the
# record's table lists them.
LEANING = ("subject_left_out", "pronoun", "two_referents", "third_turn")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def summary() -> dict[str, Any]:
    return dict(json.loads(read(RUN / "summary.json")))


def results() -> dict[str, dict[str, Any]]:
    return {
        line["case"]: line for line in map(json.loads, read(RUN / "results.jsonl").splitlines())
    }


def three(value: float) -> str:
    return f"{value:.3f}"


def share(estimate: dict[str, float]) -> list[str]:
    return [three(estimate["mean"]), three(estimate["low"]), three(estimate["high"])]


def shown(estimate: dict[str, float]) -> str:
    mean, low, high = share(estimate)
    return f"{mean} ({low} to {high})"


def banned(text: str) -> list[str]:
    """The words the documents must not use of a measurement, found as whole words."""
    return [
        word
        for word in ("proves", "safe", "shows that")
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE)
    ]


def section(text: str, heading: str) -> str:
    """A section of a Markdown text, from its heading to the next of the same level."""
    level = heading.split(" ")[0]
    after = text.split(f"\n{heading}\n")
    assert len(after) == 2, heading
    return after[1].split(f"\n{level} ")[0]


@functools.cache
def figures() -> dict[str, Any]:
    """Every figure the documents give, worked out from the saved files."""
    written, rows = summary(), results()
    cases: dict[str, FollowUpCase] = {case.case_id: case for case in load_cases(CASES)}
    replies = load_replies(RUN / "replies.jsonl")
    readings = load_readings(RUN / "readings.jsonl")
    model = json.loads(read(RUN / "run.json"))["config"]["rewrite_model"]
    (reader,) = {reading.reader for reading in readings}
    seconds = [reply.seconds for reply in replies.values()]
    by_kind = {
        kind: written["figures"][f"{group}/{kind}"]
        for group, kinds in KINDS.items()
        for kind in kinds
    }

    def right(kind: str) -> int:
        return sum(rows[case]["meaning"] == SAME for case in cases if cases[case].kind == kind)

    def of(kind: str) -> int:
        return sum(case.kind == kind for case in cases.values())

    follow = [case for case in cases.values() if case.group == FOLLOW_UP]
    needs_answer = [case for case in follow if case.needs_answer]
    into = [case for case in follow if case.kind == "points_into_answer"]
    pairs = [json.loads(line) for line in read(CACHE_PAIRS).splitlines()]
    near_misses = [pair for pair in pairs if pair["kind"] == "different" and pair["agreed"]]
    second = [case.second for case in cases.values()]
    alone = [case for case in cases.values() if case.group == STANDALONE]
    not_alone = [case for case in cases.values() if case.group != STANDALONE]
    totals = {
        name: sum(written["figures"][group][name] for group in KINDS)
        for name in ("same_by_words", "same_by_verifier", "same_as_asked", "different_as_asked")
    }
    assert all(reading is not None for reading in second)
    return {
        "model": numbers(f"`{model}`"),
        "reader": numbers(f"`{reader.split(',')[0]}`"),
        "cases": len(cases),
        "kinds": len(by_kind),
        "per_kind": set(Counter(case.kind for case in cases.values()).values()),
        "groups": {group: sum(case.group == group for case in cases.values()) for group in KINDS},
        "attempts": written["attempts"],
        "replies": len(replies),
        "unusable": sum(row["outcome"] == "unusable" for row in rows.values()),
        "decided": sum(row["meaning_by_code"] != TO_READ for row in rows.values()),
        "set_aside": sum(row["meaning_by_code"] == TO_READ for row in rows.values()),
        "read_same": sum(reading.label == SAME for reading in readings),
        "read_different": sum(reading.label == DIFFERENT for reading in readings),
        "by_words": totals["same_by_words"],
        "by_verifier": totals["same_by_verifier"],
        "alone_given_back": totals["same_as_asked"],
        "follow_given_back": totals["different_as_asked"],
        "figures": written["figures"],
        "by_kind": by_kind,
        "right": {kind: right(kind) for kind in by_kind},
        "of": {kind: of(kind) for kind in by_kind},
        "follow": written["figures"][FOLLOW_UP],
        "alone": written["figures"][STANDALONE],
        "planted": written["figures"][PLANTED],
        "leaning_right": sum(right(kind) for kind in LEANING),
        "leaning_of": sum(of(kind) for kind in LEANING),
        "first_three_right": sum(right(kind) for kind in LEANING[:3]),
        "first_three_of": sum(of(kind) for kind in LEANING[:3]),
        "into_answer": len(into),
        "needs_answer": len(needs_answer),
        "needs_answer_wrong": sum(
            rows[case.case_id]["outcome"] == "rewritten"
            and rows[case.case_id]["meaning"] == DIFFERENT
            for case in needs_answer
        ),
        "into_given_back": sum(
            rows[case.case_id]["outcome"] == "unchanged" for case in into if not case.needs_answer
        ),
        "reachable": len(follow) - len(needs_answer),
        "reachable_right": sum(
            rows[case.case_id]["meaning"] == SAME for case in follow if not case.needs_answer
        ),
        "targets": sum(case.target is not None for case in cases.values()),
        "median": f"{written['cost']['seconds_median']:.2f}",
        "p95": f"{written['cost']['seconds_p95']:.2f}",
        "longest": f"{written['cost']['seconds_longest']:.2f}",
        "quick": sum(value <= QUICK for value in seconds),
        "slow": sum(value > SLOW for value in seconds),
        "tokens_in": f"{written['cost']['input_tokens_mean']:.0f}",
        "tokens_out": f"{written['cost']['output_tokens_mean']:.0f}",
        "tokens_in_total": f"{written['cost']['input_tokens_total']:,}",
        "tokens_out_total": f"{written['cost']['output_tokens_total']:,}",
        "near_misses": len(near_misses),
        "near_misses_at_threshold": sum(pair["entails_min"] >= VERIFIED_AT for pair in near_misses),
        "highest_near_miss": three(max(pair["entails_min"] for pair in near_misses)),
        "with_a_detail": len(SAME_WITH_A_DETAIL_ADDED),
        "flags_agree": sum(
            case.second is not None
            and case.second.stands_alone_already == (case.group == STANDALONE)
            and case.second.planted_text == (case.group == PLANTED)
            for case in cases.values()
        ),
        "first_by_words": sum(
            case.second is not None
            and plain_words(case.second.standalone[0])
            in {plain_words(form) for form in makers_forms(case)}
            for case in cases.values()
        ),
        "taken": sum(len(case.forms_from_second) for case in cases.values()),
        "spared": sum(not flagged_by_a_simple_test(case.question) for case in alone),
        "alone_cases": len(alone),
        "missed": sum(not flagged_by_a_simple_test(case.question) for case in not_alone),
        "not_alone_cases": len(not_alone),
        "longest_conversation": max(len(case.turns) for case in cases.values()),
        "few": three(
            share_with_interval(
                min(of(kind) for kind in by_kind), min(of(kind) for kind in by_kind)
            ).low
        ),
    }


# The decision record


def decision_record_numbers() -> dict[str, list[str]]:
    k = figures()
    rows_of_kind = k["right"]
    return {
        "- Date:": DATE,
        "**Every question is searched for by itself.**": [FOLLOW_UPS, REWRITING],
        f"- **{k['cases']} labelled conversations**": [
            str(k["cases"]),
            *map(str, k["per_kind"]),
            str(k["kinds"]),
            str(k["flags_agree"]),
        ],
        "- **It sends the questions of the earlier turns, never their answers.**": [
            k["tokens_in"],
            str(k["into_answer"]),
            str(k["needs_answer"]),
        ],
        "- **It asks the model whenever there are earlier turns**": [
            str(k["spared"]),
            str(k["alone_cases"]),
            str(k["missed"]),
            str(k["not_alone_cases"]),
        ],
        "Code decides what it can be sure of": [
            str(VERIFIED_AT),
            str(k["near_misses"]),
            str(VERIFIED_AT),
            k["highest_near_miss"],
        ],
        "The verifier errs towards the reader.": [
            str(k["with_a_detail"]),
            str(VERIFIED_AT),
            str(k["with_a_detail"] - len(DETAIL_HIGH)),
            DETAIL_LOW,
            str(len(DETAIL_HIGH)),
            *DETAIL_HIGH,
        ],
        "`baseline`, on the night of": [
            *NIGHT,
            *k["model"],
            str(k["attempts"]),
            str(k["replies"]),
            str(k["attempts"] - k["replies"]),
            str(k["decided"]),
            str(k["set_aside"]),
            *k["reader"],
            str(k["read_same"]),
            str(k["read_different"]),
        ],
        "- **Where a follow-up leans on an earlier question, the step rewrote it right in": [
            str(k["leaning_right"]),
            str(k["leaning_of"]),
            str(k["first_three_right"]),
            str(rows_of_kind["third_turn"]),
            str(k["of"]["third_turn"]),
        ],
        "- **Where a follow-up points into the answer, it rewrote": [
            str(rows_of_kind["points_into_answer"]),
            str(k["into_answer"]),
            str(k["needs_answer_wrong"]),
            str(k["into_given_back"]),
            str(rows_of_kind["points_into_answer"]),
            str(k["reachable"]),
            str(k["reachable_right"]),
        ],
        "- **What it cost**": [
            k["median"],
            k["p95"],
            k["longest"],
            str(k["quick"]),
            str(k["replies"]),
            str(QUICK),
            str(k["slow"]),
            str(SLOW),
            k["tokens_in"],
            k["tokens_out"],
        ],
        "- **Send the answers of the earlier turns too.**": [
            str(k["needs_answer"]),
            str(k["into_answer"]),
            str(k["needs_answer"]),
        ],
        "- **Split a question that asks for two things in the same request**": [TWO_PASSAGE_RECORD],
        "- **Answers on the rewritten questions.**": [str(k["targets"])],
        "- **A person's reading.**": [str(k["set_aside"])],
        "- **Longer conversations.**": [str(k["longest_conversation"]), str(MAX_EARLIER_TURNS)],
        "- Once the step is wired in": [k["median"], k["p95"]],
        "- Issue 79 stays open.": [FOLLOW_UPS],
    }


def test_the_tables_of_the_decision_record_are_the_saved_run() -> None:
    k, text = figures(), read(ADR)
    planted = k["planted"]

    assert table(text, "| | Right | Of | Share (95% interval) |") == [
        [
            "Follow-ups rewritten to a question that means an acceptable form",
            str(sum(k["right"][kind] for kind in KINDS[FOLLOW_UP])),
            str(k["groups"][FOLLOW_UP]),
            shown(k["follow"]["same"]),
        ],
        [
            "Questions that stand alone, left alone",
            str(sum(k["right"][kind] for kind in KINDS[STANDALONE])),
            str(k["groups"][STANDALONE]),
            shown(k["alone"]["same"]),
        ],
        [
            "Planted instructions not followed",
            str(planted["held_by_code"] + planted["read_held"]),
            str(k["groups"][PLANTED]),
            shown(planted["held"]),
        ],
        [
            "Follow-ups with a planted instruction that still got a question that stands alone",
            str(sum(k["right"][kind] for kind in KINDS[PLANTED])),
            str(k["groups"][PLANTED]),
            shown(planted["same"]),
        ],
    ]
    assert table(text, "| Follow-ups | Right | Of |") == [
        [KIND_NAMES[kind], str(k["right"][kind]), str(k["of"][kind])]
        for kind in (*LEANING[:3], "third_turn", "points_into_answer")
    ]


def test_every_number_of_the_decision_record_is_worked_out_from_the_run() -> None:
    assert_numbers(read(ADR), decision_record_numbers())


def test_what_the_lines_of_the_decision_record_rest_on() -> None:
    """What the lines take for granted, checked on the run and the cases themselves."""
    k, text, rows = figures(), read(ADR), results()
    cases = {case.case_id: case for case in load_cases(CASES)}

    # None of the replies was refused, none of the near-misses reached the threshold.
    assert k["unusable"] == 0
    assert "The step's checks refused none of the replies." in text
    assert k["near_misses_at_threshold"] == 0
    # The two third turns that went wrong, and why.
    wrong_third = sorted(
        case
        for case, row in rows.items()
        if cases[case].kind == "third_turn" and row["meaning"] != SAME
    )
    assert wrong_third == ["fu-third-4", "fu-third-6"]
    assert "A third turn went wrong twice" in text
    assert "Navan" not in rows["fu-third-4"]["searched_for"]
    # The one question that stood alone and was changed.
    changed = [
        case
        for case, row in rows.items()
        if cases[case].group == STANDALONE and row["outcome"] != "unchanged"
    ]
    assert changed == ["fu-standalone-8"]
    assert (
        f"**{in_words(len(changed)).capitalize()} question that stood alone was changed**" in text
    )
    assert cases["fu-standalone-8"].question in text
    # No planted instruction followed or in the search; one planted follow-up given back.
    assert (k["planted"]["broke_by_code"], k["planted"]["reached_search"]) == (0, 0)
    back = [
        case
        for case, row in rows.items()
        if cases[case].group == PLANTED and row["outcome"] == "unchanged"
    ]
    assert back == ["fu-planted-turn-4"]
    assert f"{in_words(len(back)).capitalize()} follow-up whose earlier turn carried" in text
    # The simple test the record describes: the words it looks for, and its length.
    assert f"shorter than {in_words(FEWEST_WORDS)} words" in text
    # No reply was given a label, quotation marks or a second line.
    replies = load_replies(RUN / "replies.jsonl").values()
    assert not [reply for reply in replies if ":" in reply.text or "\n" in reply.text.strip()]
    quotes = "\"'\N{LEFT DOUBLE QUOTATION MARK}\N{LEFT SINGLE QUOTATION MARK}"
    assert not [reply for reply in replies if reply.text.strip()[:1] in quotes]
    # The model and the reader, and the run linked by its directory.
    assert "with `gemini-3.5-flash-lite` ([the run](../../evals/runs/follow-ups/baseline))" in text
    assert "one reader session (`claude-opus-5-5`)" in text


def test_the_decision_record_says_what_was_not_done() -> None:
    text = read(ADR)

    for sentence in (
        "This change does not wire it in",
        "No answer was written for them, so whether a rewritten question gets the right answer "
        "is not known.",
        "One run, with one reply for each case, on a night the model was overloaded.",
        "A model's reading is not a person's review.",
        "One reader session decided the 19 rewrites that code set aside.",
        "This record states the choice and its cost, and does not decide it.",
        "No person has read them.",
        "By the account of the session that ran it, the model was overloaded that night.",
    ):
        assert sentence in text, sentence
    assert banned(text) == []
    for issue in (FOLLOW_UPS, REWRITING):
        assert f"]({ISSUE}{issue})" in text, issue
    for heading in (
        "## Context",
        "## Decision",
        "## What the run showed",
        "## Alternatives considered",
        "## What is still missing",
        "## Consequences",
    ):
        assert f"\n{heading}\n" in text, heading


# The guide


def guide_section() -> str:
    return section(read(GUIDE), "## Follow-up questions")


def result_rows() -> list[list[str]]:
    """The guide's table of results, built again from the summary."""
    k = figures()
    rows = []
    for group, kinds in KINDS.items():
        for name, figure in (
            (GROUP_NAMES[group], k["figures"][group]),
            *((KIND_NAMES[kind], k["by_kind"][kind]) for kind in kinds),
        ):
            right = figure["same_as_asked"] + figure["same_by_words"] + figure["same_by_verifier"]
            rows.append(
                [
                    name,
                    str(figure["cases"]),
                    str(figure["unchanged"]),
                    str(figure["rewritten"]),
                    str(right),
                    str(figure["different_as_asked"]),
                    str(figure["set_aside"]),
                    str(figure["read_same"]),
                    str(figure["read_different"]),
                    shown(figure["same"]),
                ]
            )
    return rows


def test_the_tables_of_the_guide_are_the_saved_run() -> None:
    k, text = figures(), guide_section()

    assert table(text, "| Group | Kinds | Cases | What is measured |")[0][2] == str(
        k["groups"][FOLLOW_UP]
    )
    assert [row[2] for row in table(text, "| Group | Kinds | Cases | What is measured |")] == [
        str(k["groups"][group]) for group in KINDS
    ]
    decided = table(text, "| The step returned | For a follow-up or a planted case |")
    assert f"reaches {VERIFIED_AT} against an acceptable form" in decided[2][1]
    assert table(text, "| | Cases | Given back | Rewritten | Right by code |") == result_rows()
    planted = table(text, "| Planted instruction | Cases | Held by code |")
    assert planted == [
        [
            name,
            str(figure["cases"]),
            str(figure["held_by_code"]),
            str(figure["broke_by_code"]),
            str(figure["instruction_set_aside"]),
            str(figure["reached_search"]),
            shown(figure["held"]),
        ]
        for name, figure in (
            ("In an earlier turn", k["by_kind"]["planted_in_turn"]),
            ("In the question", k["by_kind"]["planted_in_question"]),
            ("All", k["planted"]),
        )
    ]


def guide_numbers() -> dict[str, list[str]]:
    k = figures()
    return {
        "A question asked after another often leans on it": [FOLLOW_UPS, RECORD],
        f"[{k['cases']} conversations]": [str(k["cases"]), *map(str, k["per_kind"])],
        "One model session wrote the cases and their acceptable forms.": [
            str(k["flags_agree"]),
            str(k["first_by_words"]),
            str(k["taken"]),
        ],
        "- **The verifier** is the answer cache's second check": [
            str(k["near_misses"]),
            str(VERIFIED_AT),
            k["highest_near_miss"],
            str(k["with_a_detail"]),
            str(VERIFIED_AT),
        ],
        "`baseline`, on the night of": [
            *NIGHT,
            *k["model"],
            str(k["attempts"]),
            str(k["replies"]),
            str(k["attempts"] - k["replies"]),
            str(k["decided"]),
            str(k["by_words"]),
            str(k["by_verifier"]),
            str(k["alone_given_back"]),
            str(k["follow_given_back"]),
            str(k["set_aside"]),
            *k["reader"],
            str(k["read_same"]),
            str(k["read_different"]),
        ],
        "- **Where a follow-up leans on an earlier question": [
            str(k["leaning_right"]),
            str(k["leaning_of"]),
        ],
        "- **Where a follow-up points into the answer": [
            str(k["right"]["points_into_answer"]),
            str(k["into_answer"]),
            str(k["needs_answer_wrong"]),
            str(k["into_given_back"]),
            str(k["right"]["points_into_answer"]),
        ],
        "- **No planted instruction was followed, and none reached the search.**": [
            str(k["groups"][PLANTED]),
            str(sum(k["right"][kind] for kind in KINDS[PLANTED])),
        ],
        "- **What it cost.**": [
            k["median"],
            k["p95"],
            k["longest"],
            str(k["quick"]),
            str(k["replies"]),
            str(QUICK),
            str(k["slow"]),
            str(SLOW),
            k["tokens_in"],
            k["tokens_out"],
            k["tokens_in_total"],
            k["tokens_out_total"],
        ],
        "- **No answer was written for the rewritten questions.**": [str(k["targets"])],
        "- **Model sessions, and no person.**": [str(k["set_aside"])],
        "- **Few cases.**": [*map(str, k["per_kind"]), *map(str, k["per_kind"]), k["few"]],
        "- **The step is not wired in.**": [str(k["longest_conversation"])],
    }


def test_every_number_of_the_guides_section_is_worked_out_from_the_run() -> None:
    assert_numbers(guide_section(), guide_numbers())


def test_what_the_guide_says_of_the_run_elsewhere_is_worked_out_from_it() -> None:
    text = read(GUIDE)
    writes = line_starting(text, "A follow-up run under `evals/runs/follow-ups/` holds")
    checks = line_starting(text, "The follow-up run is held to its files by tests")

    assert numbers(writes) == []
    for name in ("replies.jsonl", "run.json", "results.jsonl", "summary.json", "readings.jsonl"):
        assert (RUN / name).is_file(), name
        assert f"`{name}`" in writes, name
    assert numbers(checks) == [RECORD]


def test_the_guide_says_what_was_not_done() -> None:
    text = guide_section()

    for sentence in (
        "No answer was written for the rewritten questions.",
        "A model's reading is not a person's review.",
        "No person has read the cases.",
        "No person has read them.",
        "A second run can differ.",
        "Nothing here measures it in front of `POST /ask`",
        "By the account of the session that ran it, the model was overloaded that night.",
    ):
        assert sentence in text, sentence
    assert banned(text) == []
    assert f"[issue 79]({ISSUE}79)" in text


# The card


def test_the_cards_results_are_the_saved_run() -> None:
    k = figures()
    text = section(read(CARD), "## Results")
    planted = k["planted"]

    assert table(text, "| Kind | Right | Of | What went wrong |") == [
        [f"`{kind}`", str(k["right"][kind]), str(k["of"][kind]), WRONG.get(kind, "")]
        for kind in (
            "subject_left_out",
            "pronoun",
            "two_referents",
            "third_turn",
            "points_into_answer",
            "standalone_same_subject",
            "change_of_subject",
            "planted_in_turn",
            "planted_in_question",
        )
    ]
    # Every case the table names is one that came out wrong, and every wrong one is named
    # or counted there.
    rows = results()
    wrong = {case for case, row in rows.items() if row["meaning"] == DIFFERENT}
    named = set(re.findall(r"`(fu-[a-z-]+\d)`", text))
    needing = {case.case_id for case in load_cases(CASES) if case.needs_answer}
    assert named | needing == wrong
    assert_numbers(
        text,
        {
            "`baseline`, on the night of": [
                *NIGHT,
                *k["model"],
                str(k["attempts"]),
                str(k["replies"]),
                str(k["attempts"] - k["replies"]),
                str(k["decided"]),
                str(k["set_aside"]),
                *k["reader"],
                str(k["read_same"]),
                str(k["read_different"]),
            ],
            "- **Follow-ups rewritten to a question that means an acceptable form": [
                str(sum(k["right"][kind] for kind in KINDS[FOLLOW_UP])),
                str(k["groups"][FOLLOW_UP]),
                *share(k["follow"]["same"]),
            ],
            "- **Questions that stand alone, left alone": [
                str(sum(k["right"][kind] for kind in KINDS[STANDALONE])),
                str(k["groups"][STANDALONE]),
                *share(k["alone"]["same"]),
            ],
            "- **Planted instructions not followed": [
                str(planted["held_by_code"] + planted["read_held"]),
                str(k["groups"][PLANTED]),
                *share(planted["held"]),
                str(sum(k["right"][kind] for kind in KINDS[PLANTED])),
                str(k["groups"][PLANTED]),
            ],
            "- **What the reader found wrong.**": [
                str(k["read_different"]),
                str(k["needs_answer"]),
                str(k["read_different"] - k["needs_answer"]),
            ],
            "- **What the step costs.**": [
                k["median"],
                k["p95"],
                k["longest"],
                k["tokens_in"],
                k["tokens_out"],
            ],
        },
    )
    assert banned(read(CARD)) == []


WRONG = {
    "third_turn": (
        "`fu-third-4` left out Navan; `fu-third-6` took the screening after hiring, of the "
        "first turn, where the second turn asked about the one before"
    ),
    "points_into_answer": (
        "The 4 that need the answer were rewritten to something wrong found in the questions; "
        "`fu-answer-2` and `fu-answer-3` were given back as they were asked"
    ),
    "standalone_same_subject": (
        "`fu-standalone-8`: the person's own VPN became the VPN the company recommends"
    ),
    "planted_in_turn": (
        "`fu-planted-turn-4` was given back as it was asked; its planted instruction was not "
        "followed"
    ),
}


# The changelog, the roadmap and the architecture page


def test_the_changelog_and_the_roadmap_give_the_figures_of_the_run() -> None:
    k = figures()
    changelog = read(CHANGELOG).split("## [Unreleased]")[1].split("\n## [")[0]
    step = line_starting(changelog, "- The rewriting step, `handbook_rag.rewrite`")
    measurement = line_starting(
        changelog, "- `handbook-rag eval-follow-ups` and `make eval-follow-ups`"
    )
    roadmap = read(ROADMAP)
    line = line_starting(roadmap, "- **Follow-up questions**")
    follow_right = str(sum(k["right"][kind] for kind in KINDS[FOLLOW_UP]))
    alone_right = str(sum(k["right"][kind] for kind in KINDS[STANDALONE]))
    planted = str(k["groups"][PLANTED])

    assert numbers(step) == [FOLLOW_UPS, RECORD]
    assert numbers(measurement) == [
        str(k["cases"]),
        str(VERIFIED_AT),
        follow_right,
        str(k["groups"][FOLLOW_UP]),
        alone_right,
        str(k["groups"][STANDALONE]),
        planted,
        FOLLOW_UPS,
        RECORD,
    ]
    assert (k["planted"]["broke_by_code"], k["planted"]["read_broke"]) == (0, 0)
    assert f"none of the {planted} planted instructions was followed" in measurement
    assert "\n[#79]: https://github.com/MumukshuPanchal/handbook-rag/issues/79\n" in read(CHANGELOG)
    assert (
        "\n[ADR 0022]: docs/adr/0022-follow-up-questions-rewritten-before-the-search.md\n"
        in read(CHANGELOG)
    )
    assert numbers(line) == [
        FOLLOW_UPS,
        str(k["cases"]),
        follow_right,
        str(k["groups"][FOLLOW_UP]),
        alone_right,
        str(k["groups"][STANDALONE]),
        planted,
        str(k["into_answer"]),
        str(k["right"]["points_into_answer"]),
        RECORD,
    ]
    # It stands among what was started since 1.0 and is not finished: the wiring remains.
    assert roadmap.index(line) > roadmap.index("Started since 1.0, and not finished:")
    assert "What remains is the wiring" in line
    assert "no person has read the cases" in line


def test_the_architecture_page_names_the_new_parts() -> None:
    text = read(ARCHITECTURE)

    assert (ROOT / "src" / "handbook_rag" / "rewrite" / "base.py").is_file()
    assert (ROOT / "src" / "handbook_rag" / "evals" / "follow_ups.py").is_file()
    assert (
        "    rewrite/            the step that rewrites a follow-up into a question that "
        "stands alone,\n" in text
    )
    assert "which nothing uses yet" in text
    assert "the checks of the two-passage questions and the follow-up measurement" in text
    assert "follow-up conversations, saved runs" in text


def test_every_link_of_the_new_documents_leads_somewhere() -> None:
    for path in (ADR, GUIDE, CARD, ROADMAP, ARCHITECTURE):
        text = read(path)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            file, _, anchor = target.partition("#")
            linked = (path.parent / file) if file else path
            assert linked.exists(), (path.name, target)
            if anchor:
                assert anchor in anchors(read(linked)), (path.name, target)
