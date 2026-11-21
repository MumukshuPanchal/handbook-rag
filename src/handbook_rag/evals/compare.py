"""Compare two evaluation runs question by question.

"Hybrid scored 0.95 and dense scored 0.83" does not yet say that hybrid is better: both
numbers come from a sample of questions. This module reads two runs of the same golden
set and reports the difference per question, with a confidence interval.
"""

import json
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


def metric_labels(flow_dir: Path) -> dict[str, str]:
    """Metric ids and their display labels, in the order the flow declares them."""
    state = flow_dir / "_state.json"
    if not state.is_file():
        return {}
    declared = json.loads(state.read_text(encoding="utf-8")).get("metrics", [])
    return {str(metric["id"]): str(metric.get("label", metric["id"])) for metric in declared}


def load_run(run_dir: Path) -> RunScores:
    """Read the per-question scores of one run."""
    results = run_dir / "results.jsonl"
    if not results.is_file():
        raise ComparisonError(f"no run found at {run_dir}")
    scores: dict[str, dict[str, float]] = {}
    fingerprints: dict[str, str] = {}
    for line in results.read_text(encoding="utf-8").splitlines():
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
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return RunScores(
        name=run_dir.name,
        description=summary.get("description", ""),
        scores=scores,
        fingerprints=fingerprints,
        failed=int(summary.get("questions_failed", 0)),
    )


def compare_runs(base_dir: Path, candidate_dir: Path) -> RunComparison:
    """Compare two runs that scored the same questions with the same labels.

    The comparison refuses to run otherwise. If one run scored a question the other did
    not, or the label changed in between, the difference would mix a change in the system
    with a change in the test.
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

    order = sorted(base.scores)
    metric_ids = [name for name in base.scores[order[0]] if name in candidate.scores[order[0]]]
    metrics: dict[str, MetricComparison] = {}
    for metric in metric_ids:
        before = [base.scores[question][metric] for question in order]
        after = [candidate.scores[question][metric] for question in order]
        metrics[metric] = MetricComparison(
            base=mean_with_interval(before),
            candidate=mean_with_interval(after),
            change=paired_difference(before, after),
        )
    return RunComparison(base=base, candidate=candidate, questions=len(order), metrics=metrics)
