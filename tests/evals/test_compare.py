import json
from pathlib import Path

import pytest

from handbook_rag.evals.compare import (
    ComparisonError,
    compare_runs,
    load_run,
    metric_labels,
    overview_of_runs,
    results_table,
)


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


def write_graded_run(flow_dir: Path, name: str, grades: dict[str, dict[str, float]]) -> Path:
    """Write a run whose questions do not all have the same scores, as an answer run does."""
    run_dir = flow_dir / name
    run_dir.mkdir(parents=True)
    rows = [
        {"prompt_id": q, "status": "ok", "fingerprint": f"label-{q}", "grade": grade}
        for q, grade in grades.items()
    ]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "summary.json").write_text(json.dumps({"description": name}))
    return run_dir


def test_a_score_is_compared_on_the_questions_that_have_it_in_both_runs(tmp_path: Path) -> None:
    # An answer run: the unanswerable question has no score for answering, and a score
    # about citations exists only where the assistant answered.
    base = write_graded_run(
        tmp_path,
        "baseline",
        {
            "q1": {"right_call": 1.0, "answered": 1.0, "fully_cited": 0.0},
            "q2": {"right_call": 0.0, "answered": 0.0},
            "u1": {"right_call": 1.0, "declined": 1.0},
        },
    )
    candidate = write_graded_run(
        tmp_path,
        "v1",
        {
            "q1": {"right_call": 1.0, "answered": 1.0, "fully_cited": 1.0},
            "q2": {"right_call": 1.0, "answered": 1.0, "fully_cited": 1.0},
            "u1": {"right_call": 0.0, "declined": 0.0, "fully_cited": 0.0},
        },
    )

    comparison = compare_runs(base, candidate)

    assert comparison.questions == 3
    assert list(comparison.metrics) == ["right_call", "answered", "fully_cited", "declined"]
    counts = {name: m.change.difference.count for name, m in comparison.metrics.items()}
    assert counts == {"right_call": 3, "answered": 2, "fully_cited": 1, "declined": 1}
    # Only q1 was answered in both runs, so only q1 is compared on citations.
    cited = comparison.metrics["fully_cited"]
    assert (cited.base.mean, cited.candidate.mean, cited.change.better) == (0.0, 1.0, 1)
    assert comparison.metrics["answered"].change.difference.mean == 0.5
    assert comparison.metrics["declined"].change.worse == 1


def test_a_score_that_no_question_has_in_both_runs_is_named_as_not_compared(
    tmp_path: Path,
) -> None:
    base = write_graded_run(tmp_path, "baseline", {"q1": {"right_call": 1.0, "answered": 1.0}})
    candidate = write_graded_run(tmp_path, "v1", {"q1": {"right_call": 1.0, "fully_cited": 1.0}})

    comparison = compare_runs(base, candidate)

    assert list(comparison.metrics) == ["right_call"]
    # Neither is dropped without a word: one is only in the base run, one only in the other.
    assert comparison.not_compared == ("answered", "fully_cited")


def test_every_score_is_compared_when_all_questions_have_all_of_them(tmp_path: Path) -> None:
    base = write_run(tmp_path, "baseline", {"q1": 1.0, "q2": 0.0})
    candidate = write_run(tmp_path, "v1", {"q1": 1.0, "q2": 1.0})

    assert compare_runs(base, candidate).not_compared == ()


def test_a_question_with_a_character_some_tools_take_for_a_line_end_is_one_row(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "baseline"
    run_dir.mkdir()
    row = {
        "prompt_id": "q1",
        "prompt": "What is the policy on\u2028leave?",
        "status": "ok",
        "fingerprint": "label-q1",
        "grade": {"hit@5": 1.0},
        "latency_s": 0.02,
    }
    (run_dir / "results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    estimate = {"mean": 1.0, "low": 1.0, "high": 1.0, "count": 1}
    (run_dir / "summary.json").write_text(
        json.dumps({"description": "One.", "questions_scored": 1, "metrics": {"hit@5": estimate}})
    )

    assert load_run(run_dir).scores == {"q1": {"hit@5": 1.0}}
    assert overview_of_runs(tmp_path)[0].median_latency_s == 0.02


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


def write_summarised_run(
    flow_dir: Path, name: str, hit: float, latencies: list[float], description: str
) -> None:
    """Write a run the way the evaluation does, as far as the overview reads it."""
    run_dir = flow_dir / name
    run_dir.mkdir(parents=True)
    rows = [{"prompt_id": f"q{i}", "status": "ok", "latency_s": s} for i, s in enumerate(latencies)]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    estimate = {"mean": hit, "low": hit - 0.05, "high": hit + 0.04, "count": len(latencies)}
    summary = {
        "description": description,
        "config": {"retrieval_mode": name},
        "questions_scored": len(latencies),
        "metrics": {"hit@5": estimate, "mrr@10": {**estimate, "mean": hit - 0.1}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_overview_lists_the_baseline_first_and_then_runs_in_numeric_order(tmp_path: Path) -> None:
    for name in ("v10", "v2", "baseline", "v1"):
        write_summarised_run(tmp_path, name, 0.8, [0.03], f"Run {name}.")

    overviews = overview_of_runs(tmp_path)

    assert [overview.name for overview in overviews] == ["baseline", "v1", "v2", "v10"]
    assert overviews[0].config == {"retrieval_mode": "baseline"}


def test_overview_reads_scores_and_the_median_latency(tmp_path: Path) -> None:
    write_summarised_run(tmp_path, "baseline", 0.8, [0.010, 0.030, 0.500], "Dense.")

    (overview,) = overview_of_runs(tmp_path)

    assert overview.description == "Dense."
    assert overview.questions == 3
    assert overview.metrics["hit@5"].mean == 0.8
    assert overview.median_latency_s == 0.030


def test_results_table_is_markdown_with_an_interval_on_the_first_metric(tmp_path: Path) -> None:
    write_summarised_run(tmp_path, "baseline", 0.8, [0.030], "Dense only.")
    write_summarised_run(tmp_path, "v1", 0.9, [0.221], "Hybrid.")

    table = results_table(
        overview_of_runs(tmp_path), {"hit@5": "Hit@5", "mrr@10": "MRR@10"}, ["hit@5", "mrr@10"]
    )

    assert table.splitlines() == [
        "| Run | What it is | Questions | Hit@5 (95% interval) | MRR@10 | Median time |",
        "|---|---|---|---|---|---|",
        "| baseline | Dense only. | 1 | 0.800 (0.750 to 0.840) | 0.700 | 30 ms |",
        "| v1 | Hybrid. | 1 | 0.900 (0.850 to 0.940) | 0.800 | 221 ms |",
    ]


def test_a_run_without_one_of_the_scores_gets_an_empty_cell(tmp_path: Path) -> None:
    # A run that stopped early may not have reached a question the score applies to.
    write_summarised_run(tmp_path, "baseline", 0.8, [0.030], "Whole.")
    write_summarised_run(tmp_path, "v1", 0.9, [0.221], "Stopped early.")
    summary_path = tmp_path / "v1" / "summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["metrics"]["mrr@10"]
    summary_path.write_text(json.dumps(summary))

    labels = {"hit@5": "Hit@5", "mrr@10": "MRR@10"}
    overviews = overview_of_runs(tmp_path)

    assert results_table(overviews, labels, ["hit@5", "mrr@10"]).splitlines()[2:] == [
        "| baseline | Whole. | 1 | 0.800 (0.750 to 0.840) | 0.700 | 30 ms |",
        "| v1 | Stopped early. | 1 | 0.900 (0.850 to 0.940) |  | 221 ms |",
    ]
    assert results_table(overviews, labels, ["mrr@10", "hit@5"]).splitlines()[2:] == [
        "| baseline | Whole. | 1 | 0.700 (0.750 to 0.840) | 0.800 | 30 ms |",
        "| v1 | Stopped early. | 1 |  | 0.900 | 221 ms |",
    ]


def test_a_directory_without_runs_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ComparisonError, match="no runs found"):
        overview_of_runs(tmp_path)
