"""The three commands of a person's sample: draw a batch, put its labels in place, report.

They run in an empty folder with the small sets of ``tests/evals/human_fixtures.py``, the
judges' saved checks made from them, and a stand-in corpus. Every label given here as a
person's is a stand-in made by the test.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from handbook_rag.cli import app
from handbook_rag.evals.human_labels import LABELLER, LABELS_FORMAT, LABELS_VERSION
from handbook_rag.evals.human_sample import KEY_FILE, key_hash, read_batches, redraw
from tests.evals import human_fixtures as fixtures

runner = CliRunner()
BATCH = Path("evals/judge/human/batch-1")
RUN = Path("evals/runs/judge/human/current")


@pytest.fixture(autouse=True)
def project(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sets, rubrics and saved checks in the working folder, and a corpus of their passages."""
    fixtures.write_sets(Path.cwd())
    fixtures.write_saved_checks(Path.cwd())
    monkeypatch.setattr("handbook_rag.cli.build_corpus", lambda _settings: fixtures.corpus())


def key() -> dict[str, Any]:
    """Batch 1's key, drawn again as `eval-human` draws it. Only a test reads it this early."""
    (drawn,) = redraw(read_batches(BATCH.parent)[:1], fixtures.load_sets(Path.cwd()))
    return drawn.key


def flipped() -> str:
    """The first item of part A that is not disputed: the stand-in labels give it another label."""
    return next(
        str(item)
        for item, entry in key()["items"].items()
        if entry["part"] == "support" and entry["stratum"] != "disputed"
    )


def stand_in_labels(
    folder: Path, *, saved_at: str = "2026-11-29T10:00:00.000Z", **changes: Any
) -> Path:
    """A labels file as the page writes it: the set's label for every item but two, with notes.

    The flipped item gets the other verdict, and B01 "cannot tell". A label ``unclear`` of the
    set gets "supported".
    """
    entries = key()["items"]
    items = [json.loads(line) for line in (BATCH / "items.jsonl").read_text().splitlines()]
    other = {"supported": "unsupported", "unsupported": "supported"}
    flip = flipped()
    labels = []
    for item in items:
        label = entries[item["item"]]["expected"]
        label = "supported" if label == "unclear" else label
        if item["item"] == flip:
            label = other[label]
        if item["item"] == "B01":
            label = "cannot_tell"
        labels.append(
            {"item": item["item"], "item_sha256": item["sha256"], "label": label, "note": ""}
        )
    labels[0]["note"] = "A stand-in note."
    labels[-1]["note"] = "Another stand-in note."
    meta = json.loads((BATCH / "batch.json").read_text())
    data = {
        "format": LABELS_FORMAT,
        "version": LABELS_VERSION,
        "batch": 1,
        "labeller": LABELLER,
        "pass": 1,
        "items_sha256": meta["items_sha256"],
        "saved_at": saved_at,
        "counted": {"labels": len(labels), "notes": 2},
        "labels": labels,
        **changes,
    }
    folder.mkdir(parents=True, exist_ok=True)
    stamp = saved_at[:19].replace("T", "-").replace(":", "")
    path = folder / f"labels-batch-1-{stamp}.json"
    path.write_text(json.dumps(data))
    return path


def test_human_sample_draws_a_batch_and_writes_it_without_its_key() -> None:
    result = runner.invoke(app, ["human-sample", "--seed", "3"])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in BATCH.iterdir()) == [
        "HOW-TO-LABEL.md",
        "batch.json",
        "items.jsonl",
        "label.html",
    ]
    assert "Batch 1: 24 items for a person to label, blind" in result.output
    assert "part A       16 statements" in result.output
    assert "disputed" not in result.output, "the make-up of the batch is not printed"
    meta = json.loads((BATCH / "batch.json").read_text())
    assert meta["seed"] == 3
    assert meta["sources"]["handbook_commit"] == "a" * 40
    assert meta["sources"]["checks"]["support_v1"]["sha256"]
    assert not any(str(Path.cwd()) in path.read_text() for path in BATCH.iterdir())


def test_human_sample_draws_the_next_batch_and_never_a_question_again() -> None:
    assert runner.invoke(app, ["human-sample"]).exit_code == 0
    result = runner.invoke(app, ["human-sample", "--statements", "6", "--answers", "3"])

    assert result.exit_code == 0, result.output
    first, second = redraw(read_batches(BATCH.parent), fixtures.load_sets(Path.cwd()))
    assert not {c.question for c in first.picked} & {c.question for c in second.picked}
    assert len(second.items) == 9


def test_human_sample_refuses_to_draw_after_a_batch_that_cannot_be_drawn_again() -> None:
    runner.invoke(app, ["human-sample"])
    fixtures.write_saved_checks(Path.cwd(), without=("support_v1",))
    for path in Path("evals/runs/judge/baseline").iterdir():
        path.unlink()

    result = runner.invoke(app, ["human-sample"])

    assert result.exit_code == 1
    assert "is not the one drawn first" in result.output
    assert not Path("evals/judge/human/batch-2").exists()


def test_human_sample_refuses_a_batch_it_cannot_fill_and_writes_nothing() -> None:
    result = runner.invoke(app, ["human-sample", "--statements", "200"])

    assert result.exit_code == 1
    assert "lower --statements" in result.output
    assert not BATCH.exists()


def test_human_labels_checks_the_file_says_what_it_holds_and_puts_it_in_place(
    tmp_path: Path,
) -> None:
    runner.invoke(app, ["human-sample"])
    saved = stand_in_labels(tmp_path / "downloads")

    result = runner.invoke(app, ["human-labels", str(saved)])

    assert result.exit_code == 0, result.output
    assert "all 24 items, each labelled once" in result.output
    assert "saved at     2026-11-29T10:00:00.000Z" in result.output
    assert "notes        2" in result.output
    assert "cannot_tell 1" in result.output
    in_place = json.loads((BATCH / "labels.json").read_text())
    assert in_place == json.loads(saved.read_text()), "notes and all, as the page saved it"


def test_human_labels_puts_the_file_in_place_in_its_own_layout(tmp_path: Path) -> None:
    runner.invoke(app, ["human-sample"])
    saved = stand_in_labels(tmp_path / "downloads")
    data = json.loads(saved.read_text())
    saved.write_text(json.dumps({key: data[key] for key in reversed(list(data))}))

    runner.invoke(app, ["human-labels", str(saved)])

    text = (BATCH / "labels.json").read_text()
    assert text == json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def test_human_labels_refuses_an_older_file_when_a_newer_one_lies_beside_it(
    tmp_path: Path,
) -> None:
    runner.invoke(app, ["human-sample"])
    older = stand_in_labels(tmp_path / "downloads")
    newer = stand_in_labels(tmp_path / "downloads", saved_at="2026-11-29T10:30:00.000Z")

    refused = runner.invoke(app, ["human-labels", str(older)])

    assert refused.exit_code == 1
    assert "A NEWER labels file of batch 1" in refused.output
    assert newer.name in refused.output
    assert not (BATCH / "labels.json").exists()
    accepted = runner.invoke(app, ["human-labels", str(newer)])
    assert accepted.exit_code == 0, accepted.output


def test_human_labels_puts_an_older_file_in_place_when_asked_and_says_so_loudly(
    tmp_path: Path,
) -> None:
    runner.invoke(app, ["human-sample"])
    older = stand_in_labels(tmp_path / "downloads")
    stand_in_labels(tmp_path / "downloads", saved_at="2026-11-29T10:30:00.000Z")

    result = runner.invoke(app, ["human-labels", str(older), "--older"])

    assert result.exit_code == 0, result.output
    assert "WARNING      1 newer labels file of batch 1 lie beside it" in result.output


def test_human_labels_does_not_replace_labels_in_place_unless_told(tmp_path: Path) -> None:
    runner.invoke(app, ["human-sample"])
    saved = stand_in_labels(tmp_path / "downloads")
    runner.invoke(app, ["human-labels", str(saved)])

    again = runner.invoke(app, ["human-labels", str(saved)])
    replaced = runner.invoke(app, ["human-labels", str(saved), "--replace"])

    assert again.exit_code == 1
    assert "Give --replace" in again.output
    assert replaced.exit_code == 0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch": 3}, "no batch 3 was drawn in evals/judge/human (batches drawn: 1)"),
        ({"labeller": "A Person"}, "with no name or address"),
        ({"items_sha256": "0" * 16}, "another drawing of batch 1"),
        ({"counted": {"labels": 24, "notes": 0}}, "it was edited"),
    ],
)
def test_human_labels_refuses_a_file_that_does_not_fit(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    runner.invoke(app, ["human-sample"])
    saved = stand_in_labels(tmp_path / "downloads", **changes)

    result = runner.invoke(app, ["human-labels", str(saved)])

    assert result.exit_code == 1
    assert message in " ".join(result.output.split())
    assert not (BATCH / "labels.json").exists()


def test_human_labels_refuses_a_file_that_is_not_there_or_not_json(tmp_path: Path) -> None:
    runner.invoke(app, ["human-sample"])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")

    missing = runner.invoke(app, ["human-labels", str(tmp_path / "none.json")])
    unreadable = runner.invoke(app, ["human-labels", str(broken)])

    assert missing.exit_code == 1
    assert "There is no file at" in missing.output
    assert unreadable.exit_code == 1
    assert "cannot be read as JSON" in unreadable.output


def test_eval_human_waits_for_labels() -> None:
    runner.invoke(app, ["human-sample"])

    result = runner.invoke(app, ["eval-human"])

    assert result.exit_code == 1
    assert "has its labels in place yet" in result.output
    assert not (BATCH / KEY_FILE).exists()


def test_eval_human_reports_the_stand_in_labels_and_writes_the_key_beside_them(
    tmp_path: Path,
) -> None:
    runner.invoke(app, ["human-sample"])
    runner.invoke(app, ["human-labels", str(stand_in_labels(tmp_path / "downloads"))])
    runner.invoke(app, ["human-sample", "--statements", "4", "--answers", "2"])

    result = runner.invoke(app, ["eval-human"])

    assert result.exit_code == 0, result.output
    meta = json.loads((BATCH / "batch.json").read_text())
    written = json.loads((BATCH / KEY_FILE).read_text())
    assert key_hash(written) == meta["joins_sha256"]
    assert not Path("evals/judge/human/batch-2", KEY_FILE).exists(), "batch 2 waits for labels"
    summary = json.loads((RUN / "summary.json").read_text())
    assert summary["waiting_for_labels"] == [2]
    assert "waiting for labels: 2" in result.output
    part_a, part_b = summary["parts"]["support"], summary["parts"]["correct"]
    against = part_a["against_the_set"]
    # The one statement the set calls unclear is counted apart; one other is flipped.
    assert against["unclear_in_the_set"] == 1
    assert against["compared"] == 15
    assert against["same_verdict"]["successes"] == 14
    version_1 = part_a["against_the_judges"][0]
    assert version_1["config"] == {"judge_model": "stand-in support_v1"}
    # Version 1 differs from the set on one statement, where the owner gave the set's label.
    (sides_v1, sides_v2) = part_a["disputed"]["judges"]
    assert sides_v1["the_owner_sides_with"]["the set"] == 1
    assert sides_v2["the_owner_sides_with"]["the set"] == 1
    assert part_b["cannot_tell"] == 1
    (correctness,) = part_b["disputed"]["judges"]
    assert sum(correctness["the_owner_sides_with"].values()) == 2
    assert summary["differences"] == 1 + 1 + 2  # the flip, and the three judges' differences
    (unsure,) = [json.loads(line) for line in (RUN / "cannot-tell.jsonl").read_text().splitlines()]
    assert unsure["item"] == "B01"
    assert len((RUN / "disputed.jsonl").read_text().splitlines()) == 6
    assert "Where a judge and the set differ, the owner sides with" in result.output


def test_eval_human_refuses_labels_that_were_edited_in_place(tmp_path: Path) -> None:
    runner.invoke(app, ["human-sample"])
    runner.invoke(app, ["human-labels", str(stand_in_labels(tmp_path / "downloads"))])
    in_place = BATCH / "labels.json"
    in_place.write_text(in_place.read_text().replace('"cannot_tell"', '"maybe"'))

    result = runner.invoke(app, ["eval-human"])

    assert result.exit_code == 1
    assert "labels.json" in result.output
    assert "not one of its choices" in result.output


def test_eval_human_refuses_when_the_key_cannot_be_drawn_again(tmp_path: Path) -> None:
    runner.invoke(app, ["human-sample"])
    runner.invoke(app, ["human-labels", str(stand_in_labels(tmp_path / "downloads"))])
    fixtures.write_saved_checks(Path.cwd(), leave_out="q013")

    result = runner.invoke(app, ["eval-human"])

    assert result.exit_code == 1
    assert "is not the one drawn first" in result.output
    assert not (BATCH / KEY_FILE).exists()


def test_a_share_with_nothing_to_compare_still_gives_its_cannot_tell_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from handbook_rag.cli import _echo_share

    _echo_share("Labelled supported: the same", None, after="; 8 cannot tell")

    assert "nothing to compare; 8 cannot tell" in capsys.readouterr().out
