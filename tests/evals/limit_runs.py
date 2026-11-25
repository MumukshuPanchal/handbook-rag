"""The figures of the judge's checks on the limit answers, worked out once from the saved runs.

Two tests read them: ``tests/test_readme_results.py`` for the README, and
``tests/evals/test_limit_documents.py`` for the decision record (ADR 0023), the guide, the
cards, the changelog, the roadmap and the architecture page. So every document is held to
the same numbers.

Two runs are not in the repository yet: version 2's check on the first labelled answers,
and the answers of the `baseline` run judged by version 2. The names below say where each
will be, why a test that needs it is skipped while it is absent, and the words by which
the documents say that it is missing. Once a run is there, those words have to go, and
the documents have to give its figures.
"""

import functools
import json
import re
from pathlib import Path
from typing import Any

from handbook_rag.evals.answers import load_records, replies_to_current_labels, summarise_answers
from handbook_rag.evals.dataset import load_golden_set
from handbook_rag.evals.metrics import Estimate

ROOT = Path(__file__).parents[2]
RUNS = ROOT / "evals" / "runs"
CASES = ROOT / "evals" / "judge" / "limits" / "cases.jsonl"
FIRST_SET = ROOT / "evals" / "judge" / "support" / "cases.jsonl"
GOLDEN = ROOT / "evals" / "golden" / "questions.jsonl"
BASELINE = RUNS / "answers" / "baseline"

# The checks on the limit answers, by the version of the judge that made each.
CHECKS = {1: "judge/limits", 2: "judge/limits-v2"}
# The checks on the first labelled answers. Version 2's is still to come.
FIRST_SET_CHECKS = {1: "judge/baseline", 2: "judge/v2"}
FIRST_SET_CHECK_V2 = RUNS / FIRST_SET_CHECKS[2]
# The verdicts of each version on the answers of the `baseline` run. Version 2's are to come.
BASELINE_VERDICTS = {1: BASELINE / "support.jsonl", 2: BASELINE / "support-v2.jsonl"}

WAITS_FOR_THE_FIRST_SET_CHECK = (
    "waits for version 2's check on the first labelled answers (evals/runs/judge/v2), "
    "which is not in the repository yet"
)
WAITS_FOR_THE_BASELINE_VERDICTS = (
    "waits for the baseline answers judged by version 2 "
    "(evals/runs/answers/baseline/support-v2.jsonl), which is not in the repository yet"
)
# How the documents say that each of the two runs is missing.
NOT_YET_CHECKED = "not yet checked on the first labelled answers"
NOT_YET_RECORDED = "not yet recorded"
# The first words of the line in which a document gives the figures of each of the two
# runs. Until a run comes, the line with these words says that it is missing.
FIRST_SET_LINE = "- **Version 2 on the first labelled answers.**"
BASELINE_LINE = "- **Version 2's score of the `baseline` answers.**"

# How version 2 writes down the limits it found, at the end of its reason.
NAMED = re.compile(r"\(Limits the judge named: (.*)\)$")
# The misses of version 2 in which it found the limit, and still called the statement
# supported: the words of the limit that its reason holds. A test checks them against the
# run, and that no other miss's reason holds its limit.
FOUND_THE_LIMIT = {
    "lim-069": "3 or more valid reports",
    "lim-101": "as long as the assignee and a SIRT manager agree",
}


def rows(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON Lines file, split at line breaks only."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def summary(run: str) -> dict[str, Any]:
    """The ``summary.json`` of a saved run, such as ``judge/limits``."""
    loaded: dict[str, Any] = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    return loaded


def hits(estimate: dict[str, Any]) -> int:
    """How many a share counted: its mean times its count."""
    return round(float(estimate["mean"]) * int(estimate["count"]))


@functools.cache
def cases() -> dict[str, dict[str, Any]]:
    """The limit answers, by case."""
    return {case["case"]: case for case in rows(CASES)}


def named_limits(reason: str) -> str:
    """What version 2 named as limits, from the end of its reason."""
    match = NAMED.search(reason)
    if match is None:
        raise ValueError(f"no limits named in {reason!r}")
    return match[1]


@functools.cache
def check(version: int) -> dict[str, Any]:
    """Everything the documents quote of one check on the limit answers.

    The agreement figures are those the run saved. The statements behind them are worked
    out again from its verdicts and the labels, so that what is said of single statements
    (which were caught, which were missed) rests on the verdicts themselves.
    """
    run = CHECKS[version]
    about = summary(run)
    agreement = about["agreement"]
    labelled = cases()
    judgements = rows(RUNS / run / "judgements.jsonl")
    verdicts = {
        (row["case"], v["number"]): v["verdict"] for row in judgements for v in row["verdicts"]
    }
    reasons = {
        (row["case"], v["number"]): v["reason"] for row in judgements for v in row["verdicts"]
    }
    expected = {
        (case["case"], statement["number"]): statement["expected"]
        for case in labelled.values()
        for statement in case["statements"]
    }
    faults = {key for key, label in expected.items() if label == "unsupported"}
    supported = {key for key, label in expected.items() if label == "supported"}
    caught = {key for key in faults if verdicts[key] == "unsupported"}
    in_title = {key for key in faults if labelled[key[0]]["limit_where"] == "title"}
    answers = int(about["answers_judged"])
    return {
        "run": run,
        "summary": about,
        "recall": agreement["recall"],
        "precision": agreement["precision"],
        "same_decision": agreement["same_decision"],
        "same_answer": agreement["same_answer"],
        "kappa": agreement["kappa"],
        "by_kind": {kind: tuple(counts) for kind, counts in agreement["caught"].items()},
        "verdicts": verdicts,
        "reasons": reasons,
        "faults": faults,
        "supported": supported,
        "caught": caught,
        "missed": sorted(faults - caught),
        "false_alarms": sorted(key for key in supported if verdicts[key] == "unsupported"),
        "in_title": in_title,
        "caught_in_title": in_title & caught,
        "answers": answers,
        "attempts": int(about["attempts"]),
        "input_per_answer": about["input_tokens"] / answers,
        "output_per_answer": about["output_tokens"] / answers,
        "median_seconds": float(about["median_seconds"]),
    }


def ratio_of_input_tokens() -> float:
    """How many times the input tokens of version 1 version 2 reads, to the nearest half."""
    ratio = float(check(2)["input_per_answer"]) / float(check(1)["input_per_answer"])
    return round(ratio * 2) / 2


def first_set_check(version: int) -> dict[str, Any]:
    """The figures a document gives of a check on the first labelled answers."""
    run = FIRST_SET_CHECKS[version]
    about = summary(run)
    agreement = about["agreement"]
    recall, precision = agreement["recall"], agreement["precision"]
    alarms = 0 if precision is None else int(precision["count"])
    right = 0 if precision is None else hits(precision)
    return {
        "run": run,
        "summary": about,
        "labelled": int(about["answers_labelled"]),
        "judged": int(about["answers_judged"]),
        "recall": recall,
        "caught": hits(recall),
        "faults": int(recall["count"]),
        "false_alarms": alarms - right,
        # The statements a check counts that are not labelled unsupported: supported, or
        # without a claim.
        "not_faults": int(agreement["statements"]) - int(recall["count"]),
    }


@functools.cache
def baseline_score(version: int) -> Estimate:
    """The score "supported by its sources" of the `baseline` answers, by one version.

    It is worked out as an answer run works it out (``summarise_answers``), from the
    verdicts that version saved on the run: an answer is supported when no statement of it
    is unsupported. For version 1 this is the score in the run's summary, and a test holds
    the two together. For version 2 it is the score ``eval-support --judge-version 2``
    prints, which no file of the run holds.
    """
    questions = load_golden_set(GOLDEN)
    records = replies_to_current_labels(questions, load_records(BASELINE / "answers.jsonl"))
    latest = {row["case"]: row for row in rows(BASELINE_VERDICTS[version])}
    passed = {
        case: all(verdict["verdict"] != "unsupported" for verdict in row["verdicts"])
        for case, row in latest.items()
        if case in records
    }
    report = summarise_answers(questions, records, judged={"supported": passed})
    return report.summary["supported"]


def real_unclear_in_the_first_set() -> list[str]:
    """The questions of the real answers in the first labelled set that hold a statement
    labelled unclear: the real answers that drop a limit, on which the readers split."""
    return sorted(
        row["question_id"]
        for row in rows(FIRST_SET)
        if row["kind"] == "real"
        and any(statement["expected"] == "unclear" for statement in row["statements"])
    )
