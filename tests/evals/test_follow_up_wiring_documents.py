"""What the record of the wiring (ADR 0024) says is what the code and the saved runs hold.

Every line of prose in the record that holds a number is compared with the numbers it
must hold, worked out from the saved runs of the follow-up cases, the held-out runs, the
settings and the API's limits; a line with a number that nothing works out fails, as in
``test_follow_up_documents.py``. The real check of the service, made once by hand, is
given in the record as it was made, and its numbers are listed here as that check's.
"""

import json
import re
from pathlib import Path

from handbook_rag.api import MAX_BODY_BYTES, MAX_EARLIER_QUESTIONS, MAX_QUERY_CHARS
from handbook_rag.config import Settings
from handbook_rag.evals.follow_ups import FOLLOW_UP, PLANTED
from handbook_rag.rewrite.models import MAX_EARLIER_TURNS
from tests.evals import test_follow_up_documents as documents
from tests.evals.test_committed_held_out_follow_up_cases import cases as held_out_cases
from tests.evals.test_committed_held_out_follow_up_runs import (
    CHANGED,
    held_out_figures,
    replies,
)
from tests.test_readme_results import anchors, assert_numbers, line_starting, numbers, table

ROOT = Path(__file__).parents[2]
RECORD = ROOT / "docs" / "adr" / "0024-follow-up-questions-in-the-service.md"
FIRST_RUN = ROOT / "evals" / "runs" / "follow-ups" / "baseline"
# The free key's allowance when this was written, as the other documents give it.
PER_MINUTE, PER_DAY = "15", "500"
# The record of how /ask answers, which the stream keeps to.
ANSWERS_OVER_HTTP = "0015"
# The real check of the service, made once by hand on 25 November 2026 and saved as its
# replies were sent, outside the repository: the requests the replies counted, step by
# step, the cap the planned steps were held to, and the waits and limits it reported. The
# last step was added afterwards by the session that coordinated the work, and is given
# as that session reported it.
REAL_CHECK_REQUESTS = [1, 2, 2, 1, 2, 0, 1, 2, 2]
REAL_CHECK_PLANNED = REAL_CHECK_REQUESTS[:-1]
REAL_CHECK_CAP = 12
REAL_CHECK_DAY = ["25", "2026"]


def text() -> str:
    return documents.without_case_names(RECORD.read_text(encoding="utf-8"))


def record_numbers() -> dict[str, list[str]]:
    k, h, settings = documents.figures(), held_out_figures(), Settings()
    first = documents.share(k["follow"]["same"])
    limit = settings.follow_up_timeout_seconds
    seconds = [reply.seconds for reply in replies(FIRST_RUN).values()]
    over_the_limit = sum(value > limit for value in seconds)
    typed_without_a_mark = sum(
        case.group == FOLLOW_UP and not case.question.rstrip().endswith("?")
        for case in held_out_cases()
    )
    planted = k["groups"][PLANTED] + h["planted_of"]
    measured = k["cases"] + h["cases"]
    return {
        "- Date:": documents.DATE,
        "[ADR 0022]": [
            documents.RECORD,
            str(k["cases"]),
            str(k["follow_right"]),
            str(k["groups"][FOLLOW_UP]),
            str(k["alone_right"]),
            str(k["alone_cases"]),
            str(k["groups"][PLANTED]),
            documents.FOLLOW_UPS,
        ],
        "**Held-out cases came first.**": [
            str(h["cases"]),
            str(h["follow_right"]),
            str(h["follow_of"]),
            *h["follow_share"],
            *first,
            str(h["follow_of"]),
            str(h["misses"]),
            str(h["refused"]),
            str(h["model_misses"]),
        ],
        "The caller holds the conversation": [
            str(MAX_EARLIER_QUESTIONS),
            str(settings.follow_up_earlier_questions),
            str(MAX_BODY_BYTES // 1024),
            str(MAX_EARLIER_QUESTIONS),
        ],
        "- **Over HTTP an answer is refused, not ignored.**": ["422"],
        "Tests hold all three.": [str(k["right"]["points_into_answer"]), str(k["into_answer"])],
        "Every piece of text that reaches the rewriting model": [
            str(k["planted"]["held_by_code"] + k["planted"]["read_held"]),
            str(k["groups"][PLANTED]),
            str(h["held"]),
            str(h["planted_of"]),
        ],
        "- **In a stream**": [ANSWERS_OVER_HTTP],
        "One attempt, with a time limit of its own": [
            f"{limit:g}",
            PER_MINUTE,
            k["median"],
            k["p95"],
            str(over_the_limit),
            str(k["cases"]),
            f"{limit:g}",
            h["longest"],
        ],
        "`HANDBOOK_RAG_FOLLOW_UPS` (on by default": [
            str(settings.follow_up_earlier_questions),
            "1",
            str(MAX_EARLIER_TURNS),
            f"{limit:g}",
            str(MAX_QUERY_CHARS),
        ],
        "The held-out run tested the rule as it stood": CHANGED,
        "- **Why.**": [
            str(k["mirrored"]),
            str(h["refused"]),
            str(typed_without_a_mark),
            str(measured),
            str(planted),
        ],
        "- **What it gives, after the fact.**": [
            str(h["replies"]),
            str(h["after_right"]),
            str(h["after_of"]),
            str(k["cases"]),
        ],
        "- **A conversation stored by the service.**": [documents.FOLLOW_UPS],
        "- **The answers of the earlier turns sent to the rewriting model.**": [
            str(k["needs_answer"]),
            str(k["into_answer"]),
            str(k["into_answer"]),
        ],
        "- **More than two earlier questions.**": [str(settings.follow_up_earlier_questions)],
        "- **A follow-up costs one more request**": [PER_MINUTE, PER_DAY],
        "- **A follow-up waits for the rewrite before its search**": [k["median"], f"{limit:g}"],
        "- **Issue 79 is done**": [documents.FOLLOW_UPS],
        "On 25 November 2026, with the real service": [
            *REAL_CHECK_DAY,
            str(sum(REAL_CHECK_REQUESTS)),
            str(sum(REAL_CHECK_PLANNED)),
            str(REAL_CHECK_CAP),
            str(REAL_CHECK_REQUESTS[-1]),
        ],
        "- **The rewrite took 3.4 to 6.3 s that day**": ["3.4", "6.3", k["median"]],
        "- **Answers to rewritten questions, scored by the judges.**": [str(k["targets"])],
        "- **The limits of the check.**": ["0.01"],
    }


def test_every_number_of_the_record_is_worked_out() -> None:
    assert_numbers(text(), record_numbers())


def test_the_table_of_the_real_check_counts_the_requests_it_gives() -> None:
    record = RECORD.read_text(encoding="utf-8")
    rows = table(record, "| What was asked | Searched for | Outcome | Requests |")

    assert [int(row[3]) for row in rows] == REAL_CHECK_REQUESTS
    assert sum(REAL_CHECK_PLANNED) <= REAL_CHECK_CAP
    assert "added afterwards" in rows[-1][0]
    # The golden question of Finland's vacation days is one the assistant should decline.
    golden = (ROOT / "evals" / "golden" / "questions.jsonl").read_text(encoding="utf-8")
    (finland,) = [json.loads(line) for line in golden.splitlines() if '"u019"' in line]
    assert finland["type"] == "unanswerable"
    assert "Finland" in finland["question"]
    assert "`u019`" in record
    # Every outcome the table gives is one the reply can say.
    outcomes = {re.findall(r"`(\w+)`", row[2])[0] for row in rows}
    assert outcomes <= {"not_asked", "unchanged", "rewritten", "unusable", "failed"}
    assert outcomes == {"not_asked", "unchanged", "rewritten", "failed"}


def test_what_the_lines_of_the_record_rest_on() -> None:
    k, h = documents.figures(), held_out_figures()
    record = RECORD.read_text(encoding="utf-8")

    # The body holds the question and every earlier question at their longest, in
    # characters of four bytes each, with room for the rest of the JSON.
    assert MAX_BODY_BYTES == 32 * 1024
    assert 4 * MAX_QUERY_CHARS * (MAX_EARLIER_QUESTIONS + 1) < MAX_BODY_BYTES
    # Of the misses on the held-out cases, the two refused and the three the model made.
    assert (h["refused"], h["model_misses"]) == (2, 3)
    assert k["planted"]["broke_by_code"] == k["planted"]["read_broke"] == 0
    for sentence in (
        "Tests hold all three.",
        "That is not held-out evidence, and the changed rule has no held-out test of its own.",
        "the checks are not the safety boundary",
        "Until now the service never sent a question back.",
        "The service stores no conversation",
        "No person has read them.",
    ):
        assert sentence.lower() in record.lower(), sentence
    assert documents.banned(record) == []
    for heading in (
        "## Context",
        "## Decision",
        "## Alternatives considered",
        "## What is still missing",
        "## Consequences",
    ):
        assert f"\n{heading}\n" in record, heading


def test_every_link_of_the_record_leads_somewhere() -> None:
    record = RECORD.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", record):
        if target.startswith(("http://", "https://")):
            continue
        file, _, anchor = target.partition("#")
        linked = (RECORD.parent / file) if file else RECORD
        assert linked.exists(), target
        if anchor:
            assert anchor in anchors(linked.read_text(encoding="utf-8")), target
    assert numbers("[issue 79]") == ["79"]


def test_the_changelogs_lines_of_the_wiring_give_the_same_figures() -> None:
    k, h = documents.figures(), held_out_figures()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [Unreleased]")[1].split("\n## [")[0]
    wiring = "0024"

    assert numbers(line_starting(unreleased, "- Follow-up questions in the service.")) == [
        documents.FOLLOW_UPS,
        wiring,
    ]
    assert numbers(line_starting(unreleased, "- Held-out follow-up cases")) == [
        str(h["cases"]),
        str(h["follow_right"]),
        str(h["follow_of"]),
        str(k["follow_right"]),
        str(k["groups"][FOLLOW_UP]),
        str(h["refused"]),
        str(h["misses"]),
        documents.FOLLOW_UPS,
        wiring,
    ]
    rule = line_starting(unreleased, "- The rewriting step's rule for a rewrite without")
    assert numbers(rule) == [str(k["cases"]), documents.RECORD, wiring]
    body = line_starting(unreleased, "- `POST /ask` reads a body of up to")
    assert numbers(body) == [str(MAX_BODY_BYTES // 1024), "8", wiring]
    assert "\n[ADR 0024]: docs/adr/0024-follow-up-questions-in-the-service.md\n" in changelog
