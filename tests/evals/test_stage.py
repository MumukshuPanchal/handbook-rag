"""A stage on disk, used directly. The answer run's tests cover the same code through it."""

import json
from pathlib import Path

import pytest

from handbook_rag.evals.answers import answers_stage
from handbook_rag.evals.stage import SavedRunError, Stage, load_rows
from handbook_rag.evals.support import judge_check_stage, support_stage

CONFIG: dict[str, object] = {"judge_model": "fake-judge", "instructions": "abc"}


def stage_in(run_dir: Path) -> Stage:
    return Stage(
        run_dir,
        "verdicts.jsonl",
        "verdicts.json",
        holds="verdicts",
        advice="Discard the saved verdicts with --restart.",
        outputs=("report.json",),
    )


def test_a_stage_keeps_its_rows_settings_and_count_in_its_own_files(tmp_path: Path) -> None:
    stage = stage_in(tmp_path / "run")

    assert stage.open(CONFIG) == []
    stage.append({"case": "q001", "verdict": "supported"})
    stage.append({"case": "q002", "text": "Zwölf Wochen"})
    assert stage.add_attempts() == 1
    assert stage.add_attempts(4) == 5

    assert stage.open(CONFIG) == [
        (1, {"case": "q001", "verdict": "supported"}),
        (2, {"case": "q002", "text": "Zwölf Wochen"}),
    ]
    assert (stage.attempts(), stage.config()) == (5, CONFIG)
    assert sorted(path.name for path in stage.run_dir.iterdir()) == [
        "verdicts.json",
        "verdicts.jsonl",
    ]
    # Text is saved as it reads, not as escape codes.
    assert "Zwölf" in stage.rows_path.read_text(encoding="utf-8")


def test_two_stages_in_one_directory_do_not_touch_each_other(tmp_path: Path) -> None:
    answers = Stage(tmp_path, "answers.jsonl", "run.json", outputs=("summary.json",), traces=True)
    verdicts = stage_in(tmp_path)
    answers.open({"answer_model": "fake-model"})
    answers.append({"question_id": "q001"})
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "q001_rep0.json").write_text("[]")
    verdicts.open(CONFIG)
    verdicts.append({"case": "q001"})
    (tmp_path / "report.json").write_text("{}")

    verdicts.open(CONFIG, restart=True)

    # Restarting the judging discards the verdicts and their report, and nothing else.
    assert verdicts.open(CONFIG) == []
    assert not (tmp_path / "report.json").exists()
    assert answers.open({"answer_model": "fake-model"}) == [(1, {"question_id": "q001"})]
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "traces" / "q001_rep0.json").exists()

    answers.open({"answer_model": "fake-model"}, restart=True)

    assert not (tmp_path / "summary.json").exists()
    assert list((tmp_path / "traces").iterdir()) == []
    assert verdicts.config() == CONFIG


def test_messages_name_what_the_stage_holds_and_what_to_do(tmp_path: Path) -> None:
    stage = stage_in(tmp_path)
    stage.open(CONFIG)
    stage.append({"case": "q001"})

    with pytest.raises(SavedRunError) as refused:
        stage.check({**CONFIG, "judge_model": "another-judge"})
    assert "was made with other settings (judge_model)" in str(refused.value)
    assert str(refused.value).endswith("Discard the saved verdicts with --restart.")

    stage.state_path.unlink()
    with pytest.raises(SavedRunError, match="holds verdicts, but nothing records the settings"):
        stage.check(CONFIG)


def test_a_stage_needs_a_file_of_its_own_for_each_purpose(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="a different file for each purpose"):
        Stage(tmp_path, "rows.jsonl", "rows.jsonl")
    with pytest.raises(ValueError, match="a different file for each purpose"):
        Stage(tmp_path, "rows.jsonl", "state.json", outputs=("summary.json", "rows.jsonl"))


def test_the_stages_of_a_run_share_no_file(tmp_path: Path) -> None:
    # Restarting one stage removes its files. None of them may belong to another stage.
    answers, support = answers_stage(tmp_path), support_stage(tmp_path)

    def files(stage: Stage) -> set[str]:
        return {stage.rows_file, stage.state_file, *stage.outputs}

    assert files(answers) & files(support) == set()
    # A judge check has a directory of its own, where it uses the same names as a run.
    assert judge_check_stage(tmp_path).rows_file not in files(answers) | files(support)


def test_a_line_that_is_not_a_row_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"case": "q001"}\n[1, 2]\n')

    with pytest.raises(SavedRunError, match=r"rows\.jsonl:2 is not a saved row"):
        load_rows(path)
    assert load_rows(tmp_path / "missing.jsonl") == []


def test_the_settings_are_kept_as_they_read_after_being_saved(tmp_path: Path) -> None:
    stage = stage_in(tmp_path)

    stage.open({"sizes": (320, 60), "rerank": False})

    assert json.loads(stage.state_path.read_text()) == {
        "config": {"sizes": [320, 60], "rerank": False},
        "attempts": 0,
    }
    assert stage.config() == {"sizes": [320, 60], "rerank": False}


def test_settings_are_saved_as_they_read(tmp_path: Path) -> None:
    stage = stage_in(tmp_path)

    stage.open({"description": "Zwölf Wochen"})

    assert "Zwölf Wochen" in stage.state_path.read_text(encoding="utf-8")
    assert stage.config() == {"description": "Zwölf Wochen"}
