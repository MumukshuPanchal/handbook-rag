import json
from pathlib import Path

import pytest

from handbook_rag.evals.compare import ComparisonError, compare_runs, load_run, metric_labels


def write_run(
    flow_dir: Path,
    name: str,
    hits: dict[str, float],
    *,
    fingerprints: dict[str, str] | None = None,
    description: str = "A run.",
) -> Path:
    """Write the two files a comparison reads, with one metric."""
    run_dir = flow_dir / name
    run_dir.mkdir(parents=True)
    rows = []
    for question_id, hit in hits.items():
        row: dict[str, object] = {
            "prompt_id": question_id,
            "status": "ok",
            "grade": {"hit@5": hit, "mrr@10": hit / 2},
        }
        if fingerprints is None:
            row["fingerprint"] = f"label-{question_id}"
        elif question_id in fingerprints:
            row["fingerprint"] = fingerprints[question_id]
        rows.append(row)
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "summary.json").write_text(
        json.dumps({"description": description, "questions_failed": 0})
    )
    return run_dir


def test_comparison_pairs_the_runs_question_by_question(tmp_path: Path) -> None:
    base = write_run(tmp_path, "baseline", {"q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 1.0})
    candidate = write_run(
        tmp_path, "v1", {"q4": 1.0, "q3": 1.0, "q2": 1.0, "q1": 1.0}, description="Hybrid."
    )

    comparison = compare_runs(base, candidate)

    assert comparison.questions == 4
    assert comparison.candidate.description == "Hybrid."
    hit = comparison.metrics["hit@5"]
    assert hit.base.mean == 0.5
    assert hit.candidate.mean == 1.0
    assert hit.change.difference.mean == 0.5
    assert (hit.change.better, hit.change.worse, hit.change.same) == (2, 0, 2)
    assert comparison.metrics["mrr@10"].change.difference.mean == 0.25


def test_a_run_compared_with_itself_shows_no_difference(tmp_path: Path) -> None:
    run = write_run(tmp_path, "baseline", {"q1": 1.0, "q2": 0.0})

    comparison = compare_runs(run, run)

    assert comparison.metrics["hit@5"].change.same == 2
    assert not comparison.metrics["hit@5"].change.is_clear


def test_runs_that_scored_different_questions_cannot_be_compared(tmp_path: Path) -> None:
    base = write_run(tmp_path, "baseline", {"q1": 1.0, "q2": 0.0})
    candidate = write_run(tmp_path, "v1", {"q1": 1.0, "q3": 1.0})

    with pytest.raises(ComparisonError, match=r"1 only in baseline \['q2'\], 1 only in v1"):
        compare_runs(base, candidate)


def test_runs_that_scored_different_labels_cannot_be_compared(tmp_path: Path) -> None:
    base = write_run(tmp_path, "baseline", {"q1": 1.0, "q2": 0.0})
    candidate = write_run(
        tmp_path, "v1", {"q1": 1.0, "q2": 1.0}, fingerprints={"q1": "label-q1", "q2": "reworded"}
    )

    with pytest.raises(ComparisonError, match=r"label of 1 questions changed.*\['q2'\]"):
        compare_runs(base, candidate)


def test_a_run_that_does_not_record_its_labels_must_be_run_again(tmp_path: Path) -> None:
    old = write_run(tmp_path, "baseline", {"q1": 1.0}, fingerprints={})

    with pytest.raises(ComparisonError, match="older version"):
        load_run(old)


def test_a_missing_or_empty_run_is_an_error(tmp_path: Path) -> None:
    empty = write_run(tmp_path, "empty", {})

    with pytest.raises(ComparisonError, match="no run found"):
        load_run(tmp_path / "nowhere")
    with pytest.raises(ComparisonError, match="no scored questions"):
        compare_runs(empty, empty)


def test_questions_that_failed_are_left_out_of_the_scores(tmp_path: Path) -> None:
    run = write_run(tmp_path, "baseline", {"q1": 1.0})
    with (run / "results.jsonl").open("a") as rows:
        rows.write(json.dumps({"prompt_id": "q2", "status": "error", "grade": {}}) + "\n")

    assert set(load_run(run).scores) == {"q1"}


def test_metric_labels_follow_the_order_the_flow_declares(tmp_path: Path) -> None:
    (tmp_path / "_state.json").write_text(
        json.dumps({"metrics": [{"id": "hit@5", "label": "Hit@5"}, {"id": "mrr@10"}]})
    )

    assert metric_labels(tmp_path) == {"hit@5": "Hit@5", "mrr@10": "mrr@10"}
    assert metric_labels(tmp_path / "elsewhere") == {}
