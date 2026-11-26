"""Compare two evaluation runs question by question.

"Hybrid scored 0.95 and dense scored 0.83" does not yet say that hybrid is better: both
numbers come from a sample of questions. This module reads two runs of the same golden
set and reports the difference per question, with a confidence interval.
"""

import dataclasses
import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from handbook_rag.evals.metrics import Comparison, Estimate, mean_with_interval, paired_difference


class ComparisonError(ValueError):
    """Two runs cannot be compared question by question."""


@dataclass(frozen=True, slots=True)
class RunScores:
    """What a run directory holds, as far as a comparison needs it."""

    name: str
    description: str
    scores: dict[str, dict[str, float]]
    """Question id to metric id to score."""
    fingerprints: dict[str, str]
    failed: int
    read_again: tuple[str, ...] = ()
    """Questions whose saved reply today's code reads otherwise than the run saved it."""
    scores_as_made: dict[str, dict[str, float]] | None = None
    """The scores as the run was made, where it has questions read again."""


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One metric, for both runs and for the difference between them."""

    base: Estimate
    candidate: Estimate
    change: Comparison


@dataclass(frozen=True, slots=True)
class RunComparison:
    """The result of comparing a candidate run against a base run."""

    base: RunScores
    candidate: RunScores
    questions: int
    metrics: dict[str, MetricComparison]
    not_compared: tuple[str, ...] = ()
    """Scores that a run has, but that no question has in both runs."""
    as_made: "RunComparison | None" = None
    """The same comparison of the runs as they were made, when either has questions read
    again by code changed after it was made."""


@dataclass(frozen=True, slots=True)
class RunOverview:
    """One run, summarised for a results table."""

    name: str
    description: str
    questions: int
    metrics: dict[str, Estimate]
    median_latency_s: float | None
    config: dict[str, object]
    read_again: tuple[str, ...] = ()
    metrics_as_made: dict[str, Estimate] | None = None


def metric_labels(flow_dir: Path) -> dict[str, str]:
    """Metric ids and their display labels, in the order the flow declares them."""
    state = flow_dir / "_state.json"
    if not state.is_file():
        return {}
    declared = json.loads(state.read_text(encoding="utf-8")).get("metrics", [])
    return {str(metric["id"]): str(metric.get("label", metric["id"])) for metric in declared}


def _lines(path: Path) -> list[str]:
    """The rows of a JSON Lines file, split at line breaks only.

    A row may hold other characters that some tools treat as the end of a line, and
    splitting there would cut the row in two.
    """
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_run(run_dir: Path) -> RunScores:
    """Read the per-question scores of one run."""
    results = run_dir / "results.jsonl"
    if not results.is_file():
        raise ComparisonError(f"no run found at {run_dir}")
    scores: dict[str, dict[str, float]] = {}
    fingerprints: dict[str, str] = {}
    as_made: dict[str, dict[str, float]] = {}
    for line in _lines(results):
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        if "fingerprint" not in row:
            raise ComparisonError(
                f"{run_dir} was written by an older version and does not record which "
                "labels it scored. Run the evaluation again."
            )
        scores[row["prompt_id"]] = {name: float(value) for name, value in row["grade"].items()}
        fingerprints[row["prompt_id"]] = row["fingerprint"]
        if "grade_as_made" in row:
            as_made[row["prompt_id"]] = {k: float(v) for k, v in row["grade_as_made"].items()}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    read_again = tuple(summary.get("read_again_after_the_run", {}).get("questions", ()))
    return RunScores(
        name=run_dir.name,
        description=summary.get("description", ""),
        scores=scores,
        fingerprints=fingerprints,
        failed=int(summary.get("questions_failed", 0)),
        read_again=read_again,
        # A judged score of a reply read again is left out as it was made: only code's
        # scores are kept for the run as it was made.
        scores_as_made={q: as_made.get(q, s) for q, s in scores.items()} if read_again else None,
    )


def compare_runs(base_dir: Path, candidate_dir: Path) -> RunComparison:
    """Compare two runs that scored the same questions with the same labels.

    The comparison refuses to run otherwise. If one run scored a question the other did
    not, or the label changed in between, the difference would mix a change in the system
    with a change in the test.

    A score that does not apply to every question, such as one about an answer when the
    assistant declined, is compared on the questions that have it in both runs. Its two
    means are then taken over those questions only, so they can differ from the means
    each run reports for itself. A score with no such question is listed as not compared.
    """
    base, candidate = load_run(base_dir), load_run(candidate_dir)
    if not base.scores:
        raise ComparisonError(f"{base_dir} has no scored questions")
    only_base = sorted(set(base.scores) - set(candidate.scores))
    only_candidate = sorted(set(candidate.scores) - set(base.scores))
    if only_base or only_candidate:
        raise ComparisonError(
            "the runs did not score the same questions: "
            f"{len(only_base)} only in {base.name} {only_base[:5]}, "
            f"{len(only_candidate)} only in {candidate.name} {only_candidate[:5]}. "
            "Run both again on the current golden set."
        )
    changed = sorted(q for q in base.scores if base.fingerprints[q] != candidate.fingerprints[q])
    if changed:
        raise ComparisonError(
            f"the label of {len(changed)} questions changed between the runs {changed[:5]}. "
            "Run both again on the current golden set."
        )

    comparison = _compare(base, candidate)
    if base.read_again or candidate.read_again:
        made = _compare(_as_made(base), _as_made(candidate))
        comparison = dataclasses.replace(comparison, as_made=made)
    return comparison


def _as_made(run: RunScores) -> RunScores:
    """A run's scores as it was made."""
    if run.scores_as_made is None:
        return run
    return dataclasses.replace(run, scores=run.scores_as_made, read_again=(), scores_as_made=None)


def _compare(base: RunScores, candidate: RunScores) -> RunComparison:
    """Compare the scores of two runs of the same questions, question by question."""
    order = sorted(base.scores)
    # Not every score applies to every question: a retrieval score needs evidence, and a
    # score about an answer needs an answer. Each metric is compared on the questions for
    # which both runs have it, which keeps the comparison paired.
    metric_ids = list(
        dict.fromkeys(
            name
            for scores in (base.scores, candidate.scores)
            for question in order
            for name in scores[question]
        )
    )
    metrics: dict[str, MetricComparison] = {}
    for metric in metric_ids:
        shared = [
            question
            for question in order
            if metric in base.scores[question] and metric in candidate.scores[question]
        ]
        if not shared:
            continue
        before = [base.scores[question][metric] for question in shared]
        after = [candidate.scores[question][metric] for question in shared]
        metrics[metric] = MetricComparison(
            base=mean_with_interval(before),
            candidate=mean_with_interval(after),
            change=paired_difference(before, after),
        )
    return RunComparison(
        base=base,
        candidate=candidate,
        questions=len(order),
        metrics=metrics,
        not_compared=tuple(metric for metric in metric_ids if metric not in metrics),
    )


def _run_order(name: str) -> tuple[bool, list[int | str]]:
    """Sort key: the baseline first, then v1, v2, ... v10 in numeric order."""
    parts: list[int | str] = [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name) if part
    ]
    return (name != "baseline", parts)


def overview_of_runs(flow_dir: Path) -> list[RunOverview]:
    """Summarise every run under a flow directory, from the files each run wrote."""
    overviews: list[RunOverview] = []
    for summary_path in flow_dir.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        again = summary.get("read_again_after_the_run") or {}
        latencies = [
            float(row["latency_s"])
            for line in _lines(summary_path.parent / "results.jsonl")
            if "latency_s" in (row := json.loads(line))
        ]
        overviews.append(
            RunOverview(
                name=summary_path.parent.name,
                description=summary.get("description", ""),
                questions=int(summary.get("questions_scored", 0)),
                metrics={
                    name: Estimate(**estimate) for name, estimate in summary["metrics"].items()
                },
                median_latency_s=statistics.median(latencies) if latencies else None,
                config=dict(summary.get("config", {})),
                read_again=tuple(again.get("questions", ())),
                metrics_as_made=(
                    {name: Estimate(**e) for name, e in again["metrics_as_made"].items()}
                    if again
                    else None
                ),
            )
        )
    if not overviews:
        raise ComparisonError(f"no runs found under {flow_dir}")
    return sorted(overviews, key=lambda overview: _run_order(overview.name))


def results_table(
    overviews: Sequence[RunOverview], labels: dict[str, str], metric_ids: Sequence[str]
) -> str:
    """A Markdown table of the runs, one row each.

    The first metric is shown with its 95% confidence interval, the rest as means. The
    table is generated so that numbers in a write-up are never copied by hand.
    """
    headline = metric_ids[0]
    header = ["Run", "What it is", "Questions", f"{labels.get(headline, headline)} (95% interval)"]
    header += [labels.get(metric, metric) for metric in metric_ids[1:]]
    header.append("Median time")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for overview in overviews:
        # A run that stopped early may lack a score, and is shown with an empty cell.
        first = overview.metrics.get(headline)
        cells = [
            overview.name,
            overview.description,
            str(overview.questions),
            "" if first is None else f"{first.mean:.3f} ({first.low:.3f} to {first.high:.3f})",
        ]
        cells += [
            f"{overview.metrics[metric].mean:.3f}" if metric in overview.metrics else ""
            for metric in metric_ids[1:]
        ]
        latency = overview.median_latency_s
        cells.append("" if latency is None else f"{latency * 1000:.0f} ms")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
