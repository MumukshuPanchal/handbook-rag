"""The results in the README are the results of the committed runs.

Issue 18 asks that every number in the README lead to the committed run that produced
it. These tests read the runs under ``evals/runs/`` and compare them with what the README
prints, at the precision it prints them. A number changed by hand fails here, and so does
a run recorded again without the README being brought up to date.

Every table and figure a test expects must be found, exactly once: a test that found
nothing to compare would pass without checking anything.
"""

import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
RUNS = ROOT / "evals" / "runs"

# A number as the README prints it: 132, 1,654, 0.962. A digit that is part of a name is
# not one (Recall@5, v2, a commit hash), and neither is a percentage, which here only
# ever names an interval ("95% interval").
NUMBER = re.compile(r"(?<![\w@.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w%])")


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def charter() -> str:
    return (ROOT / "docs" / "charter.md").read_text(encoding="utf-8")


def summary(run: str) -> Any:
    """The ``summary.json`` of a committed run, such as ``retrieval/v2``."""
    return json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))


def table(text: str, header: str) -> list[list[str]]:
    """The rows of the one Markdown table in ``text`` whose header row starts with ``header``.

    Each row is the list of its cells, without the spaces around them.
    """
    lines = text.splitlines()
    starts = [number for number, line in enumerate(lines) if line.startswith(header)]
    assert len(starts) == 1, f"expected one table starting {header!r}, found {len(starts)}"
    assert lines[starts[0] + 1].startswith("|---"), f"no table under {header!r}"
    rows = []
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    assert rows, f"the table starting {header!r} has no rows"
    return rows


def line_starting(text: str, prefix: str) -> str:
    """The one line of ``text`` that starts with ``prefix``."""
    found = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(found) == 1, f"expected one line starting {prefix!r}, found {len(found)}"
    return found[0]


def numbers(text: str) -> list[str]:
    """The numbers in a piece of the README, in order, as they are printed."""
    return NUMBER.findall(text)


def link(cell: str) -> tuple[str, str]:
    """The text and the target of a table cell that is one Markdown link."""
    match = re.fullmatch(r"\[(.+?)\]\((.+?)\)", cell)
    assert match, f"expected a link, found {cell!r}"
    return match[1], match[2]


def three(value: float) -> str:
    return f"{value:.3f}"


def with_interval(estimate: Any) -> str:
    """A mean as the README prints it: 0.962 (0.924 to 0.992)."""
    return f"{three(estimate['mean'])} ({three(estimate['low'])} to {three(estimate['high'])})"


def interval_numbers(estimate: Any) -> list[str]:
    return [three(estimate["mean"]), three(estimate["low"]), three(estimate["high"])]


def hits(estimate: Any) -> int:
    """How many of the questions a yes-or-no score counted: its mean times its count."""
    return round(float(estimate["mean"]) * int(estimate["count"]))


def median_latency_ms(run: str) -> float:
    """The median time of a retrieval run, from its rows, as ``eval-table`` computes it."""
    lines = (RUNS / run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return 1000 * statistics.median(float(json.loads(line)["latency_s"]) for line in lines)


def test_the_search_table_is_the_committed_retrieval_runs() -> None:
    rows = table(readme(), "| Run | What it is | Questions | Hit@5 (95% interval) |")

    names = ["baseline", "v1", "v2", "v3"]
    assert [link(row[0]) for row in rows] == [
        (name, f"evals/runs/retrieval/{name}") for name in names
    ]
    for row in rows:
        name = link(row[0])[0]
        run = summary(f"retrieval/{name}")
        metrics = run["metrics"]
        assert row[1:] == [
            run["description"],
            str(run["questions_scored"]),
            with_interval(metrics["hit@5"]),
            three(metrics["hit@1"]["mean"]),
            three(metrics["mrr@10"]["mean"]),
            f"{median_latency_ms(f'retrieval/{name}'):.0f} ms",
        ], name


# Each row of the answer table, by the words before the colon, and its score in the run.
ANSWER_SCORES = {
    "Right call": "right_call",
    "Supported by its sources": "supported",
    "Correct by the reference": "correct",
    "Gets a correct answer": "answered_correctly",
    "Answers when it should": "answered",
    "Declines when it should": "declined",
    "Every statement cited": "fully_cited",
    "Cites the passage that holds the evidence": "cites_evidence",
}


def test_the_answer_table_is_the_committed_answer_run() -> None:
    rows = table(readme(), "| Score | Result (95% interval) | Questions |")
    metrics = summary("answers/baseline")["metrics"]

    assert [row[0].split(":")[0] for row in rows] == list(ANSWER_SCORES)
    for label, result, questions in rows:
        estimate = metrics[ANSWER_SCORES[label.split(":")[0]]]
        assert [result, questions] == [with_interval(estimate), str(estimate["count"])], label


def test_the_charter_table_has_a_row_for_each_success_metric_with_its_target() -> None:
    goals = table(charter(), "| Goal | Metric | Provisional target |")
    rows = table(readme(), "| Charter goal | Target | Measured | From | Met |")

    assert [row[0] for row in rows] == [goal[0] for goal in goals]
    assert [numbers(row[1]) for row in rows] == [numbers(goal[2]) for goal in goals]


def test_the_charter_table_is_the_committed_runs() -> None:
    hybrid = summary("retrieval/v2")["metrics"]
    answers = summary("answers/baseline")
    scores, timing, tokens = answers["metrics"], answers["timing"], answers["tokens"]
    answered = scores["answered"]
    retrieval_run, answer_run = "evals/runs/retrieval/v2", "evals/runs/answers/baseline"

    def share(name: str) -> list[str]:
        """A judged or computed share: its mean and interval, then "127 of 129"."""
        estimate = scores[name]
        return [*interval_numbers(estimate), str(hits(estimate)), str(estimate["count"])]

    # For each goal: the numbers of the "Measured" cell, the run it links to, and how the
    # "Met" cell begins.
    expected = {
        "The right passage is retrieved": (
            interval_numbers(hybrid["recall@5"]),
            retrieval_run,
            "Yes",
        ),
        "The right passage is ranked high": (
            interval_numbers(hybrid["mrr@10"]),
            retrieval_run,
            "Yes",
        ),
        "Answers are supported by their sources": (share("supported"), answer_run, "Yes"),
        "Answers are correct": (share("correct"), answer_run, "Yes"),
        "The assistant abstains when it should": (share("declined"), answer_run, "Yes"),
        "The assistant does not over-abstain": (
            [
                three(1 - answered["mean"]),
                str(answered["count"] - hits(answered)),
                str(answered["count"]),
            ],
            answer_run,
            "Yes",
        ),
        "Answers arrive quickly": (
            [f"{timing['p95_seconds']:.1f}", f"{timing['median_seconds']:.1f}"],
            answer_run,
            "No",
        ),
        "Cost is known and controlled": (
            [f"{tokens['input_mean']:,.0f}", f"{tokens['output_mean']:.0f}"],
            answer_run,
            "Not measured",
        ),
    }
    rows = table(readme(), "| Charter goal | Target | Measured | From | Met |")

    assert [row[0] for row in rows] == list(expected)
    for goal, _target, measured, source, met in rows:
        want_numbers, want_run, want_met = expected[goal]
        assert numbers(measured) == want_numbers, goal
        assert link(source)[1] == want_run, goal
        assert met.startswith(want_met), goal

    # The verdicts follow from the runs and the charter's targets.
    assert hybrid["recall@5"]["mean"] >= 0.85
    assert hybrid["mrr@10"]["mean"] >= 0.70
    assert scores["supported"]["mean"] >= 0.90
    assert scores["correct"]["mean"] >= 0.80
    assert scores["declined"]["mean"] >= 0.90
    assert 1 - answered["mean"] <= 0.05
    assert timing["p95_seconds"] > 10
    # The README says that for hybrid search Recall@5 is the same number as Hit@5.
    assert hybrid["recall@5"] == hybrid["hit@5"]


# Each judge in the README's table, and the committed check of it.
JUDGES = {
    "Supported by its sources": "judge/baseline",
    "Correct by the reference": "judge-correct/baseline",
}


def test_the_judges_checks_are_the_committed_checks() -> None:
    rows = table(readme(), "| Judge | Labelled answers | Faults caught (recall) |")

    assert [row[0] for row in rows] == list(JUDGES)
    for judge, labelled, recall, precision, check in rows:
        run = summary(JUDGES[judge])
        agreement = run["agreement"]
        # A check is quoted only when the judge ruled on every labelled answer.
        assert run["answers_judged"] == run["answers_labelled"], judge
        planted = sum(total for kind, (_, total) in agreement["caught"].items() if kind != "real")
        assert numbers(labelled) == [str(run["answers_labelled"]), str(planted)], judge
        for cell, estimate in ((recall, agreement["recall"]), (precision, agreement["precision"])):
            assert numbers(cell) == [
                str(hits(estimate)),
                str(estimate["count"]),
                *interval_numbers(estimate),
            ], judge
        assert link(check)[1] == f"evals/runs/{JUDGES[judge]}", judge


def test_the_control_is_the_committed_control_run() -> None:
    run = summary("judge-correct/references")
    same = run["agreement"]["same_decision"]
    line = line_starting(readme(), "- **A control for the second judge.**")

    assert run["answers_judged"] == run["answers_labelled"]
    assert numbers(line) == [str(run["answers_labelled"]), str(hits(same)), *interval_numbers(same)]
    assert "(evals/runs/judge-correct/references)" in line


# Each row of the adversarial table, and the family and figure of the run it shows.
ADVERSARIAL = {
    "Planted instruction not followed": ("injected_passage", "held"),
    "Question outside the handbook declined": ("out_of_scope", "held"),
    "Injected case kept its answer": ("injected_passage", "kept"),
}

# Each row of the table of planted instructions, and its kind in the run.
PLANTED = {
    "Reply with a made-up word only": "override",
    "Repeat your instructions": "reveal_instructions",
    "Obey a forged message from the system": "forged_boundary",
    "End every answer with a made-up link": "append_link",
    "Withhold the answer": "withhold",
}


def test_the_adversarial_figures_are_the_committed_run() -> None:
    run = summary("adversarial/baseline")
    rows = table(readme(), "| | Held | Of | Share held (95% interval) |")

    # The shares are final only once every reply that code set aside has a reading.
    assert run["final"]
    assert [row[0] for row in rows] == list(ADVERSARIAL)
    for label, held, of, share in rows:
        family, figure = ADVERSARIAL[label]
        estimate = run["figures"][family][figure]
        assert [held, of, share] == [
            str(hits(estimate)),
            str(estimate["count"]),
            with_interval(estimate),
        ], label


def test_the_planted_instructions_are_the_committed_run() -> None:
    figures = summary("adversarial/baseline")["figures"]
    rows = table(readme(), "| Planted instruction | Held | Of |")

    assert [row[0] for row in rows] == list(PLANTED)
    for label, held, of, _what_the_replies_did in rows:
        estimate = figures[f"injected_passage/{PLANTED[label]}"]["held"]
        assert [held, of] == [str(hits(estimate)), str(estimate["count"])], label


def test_every_link_to_a_file_of_the_repository_leads_to_one() -> None:
    targets = re.findall(r"\]\(([^)\s]+)\)", readme())
    local = [
        target.split("#")[0]
        for target in targets
        if not target.startswith(("http://", "https://", "#"))
    ]

    assert len(local) > 20
    assert [target for target in local if not (ROOT / target).exists()] == []
    # The links to the runs the results come from are among them.
    for run in ("retrieval/v2", "answers/baseline", "judge/baseline", "adversarial/baseline"):
        assert f"evals/runs/{run}" in local, run
