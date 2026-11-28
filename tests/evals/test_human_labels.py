"""The labels file a person's page saves: what fits its batch, what is refused, and which is newest.

Every labels file here is a stand-in made by the test, never anybody's labels.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.human_labels import (
    LABELLER,
    LABELS_FORMAT,
    LABELS_VERSION,
    NOTE_LIMIT,
    LabelsError,
    file_name_prefix,
    labels_text,
    load_labels_file,
    newer_files,
    read_labels,
)
from handbook_rag.evals.human_sample import CANNOT_TELL, Batch, Plan, make_batch, read_batch
from handbook_rag.evals.label_page import write_batch
from tests.evals import human_fixtures as fixtures

RUBRICS = {"support": "Support rubric.", "correct": "Correct rubric."}


@pytest.fixture
def batch(tmp_path: Path) -> Batch:
    sets = fixtures.make_project(tmp_path)
    drawn = make_batch(number=1, seed=7, plan=Plan(), sets=sets, earlier=[], sources={})
    write_batch(tmp_path / "batches", drawn, RUBRICS, sets)
    return read_batch(tmp_path / "batches" / "batch-1")


def labels_for(
    batch: Batch, *, saved_at: str = "2026-11-29T10:00:00.000Z", notes: bool = True
) -> dict[str, Any]:
    """A labels file as the page writes it: "cannot tell" throughout, and two notes."""
    labels = [
        {
            "item": item_id,
            "item_sha256": item["sha256"],
            "label": CANNOT_TELL,
            "note": f"A stand-in note on {item_id}." if notes and place < 2 else "",
        }
        for place, (item_id, item) in enumerate(batch.items.items())
    ]
    return {
        "format": LABELS_FORMAT,
        "version": LABELS_VERSION,
        "batch": batch.number,
        "labeller": LABELLER,
        "pass": 1,
        "items_sha256": batch.items_sha256,
        "saved_at": saved_at,
        "counted": {"labels": len(labels), "notes": 2 if notes else 0},
        "labels": labels,
    }


def test_a_labels_file_as_the_page_writes_it_fits(batch: Batch) -> None:
    labels = read_labels(labels_for(batch), batch)

    assert list(labels.labels) == list(batch.items)
    assert labels.labeller == "the project's owner"
    assert labels.notes == 2


def test_the_notes_are_kept_when_the_labels_are_put_in_place(batch: Batch) -> None:
    data = labels_for(batch)
    labels = read_labels(data, batch)

    assert json.loads(labels_text(labels, batch)) == data


def test_the_labels_are_put_in_place_in_one_layout_whatever_the_file_s(batch: Batch) -> None:
    data = labels_for(batch)
    reordered = {key: data[key] for key in reversed(list(data))}
    labels = read_labels(reordered, batch)

    text = labels_text(labels, batch)

    assert text == json.dumps(data, indent=2, ensure_ascii=False) + "\n"


WRONG: dict[str, tuple[Callable[[dict[str, Any]], Any], str]] = {
    "not an object": (lambda f: ["a list"], "holds no JSON object"),
    "no batch": (lambda f: f.pop("batch"), "does not say which batch"),
    "another format": (lambda f: f.update(format="something else"), "this version"),
    "the first version": (lambda f: f.update(version=1), "this version"),
    "true for the version": (lambda f: f.update(version=True), "this version"),
    "a field too many": (lambda f: f.update(name="A Person"), "not expected: name"),
    "a field missing": (lambda f: f.pop("saved_at"), "missing: saved_at"),
    "no counts": (lambda f: f.pop("counted"), "missing: counted"),
    "another batch": (lambda f: f.update(batch=2), "labels batch 2, not batch 1"),
    "a named labeller": (lambda f: f.update(labeller="A Person"), "with no name or address"),
    "a second pass": (lambda f: f.update(**{"pass": 2}), "a first pass only"),
    "true for the pass": (lambda f: f.update(**{"pass": True}), "a first pass only"),
    "another drawing": (lambda f: f.update(items_sha256="0" * 16), "another drawing"),
    "no time": (lambda f: f.update(saved_at="yesterday"), "is not a time"),
    "labels not a list": (lambda f: f.update(labels={}), "not a list of labels"),
    "a label with a field too many": (
        lambda f: f["labels"][0].update(email="a@b.c"),
        "should hold exactly",
    ),
    "an unknown item": (lambda f: f["labels"][0].update(item="A99"), "does not hold"),
    "an item twice": (lambda f: f["labels"].__setitem__(1, dict(f["labels"][0])), "twice"),
    "an item of other words": (
        lambda f: f["labels"][0].update(item_sha256="0" * 16),
        "labelled on other words",
    ),
    "a label of the other part": (
        lambda f: f["labels"][0].update(label="correct"),
        "not one of its choices",
    ),
    "a label that is no choice": (
        lambda f: f["labels"][-1].update(label="maybe"),
        "not one of its choices",
    ),
    "a label that is not text": (lambda f: f["labels"][-1].update(label=["x"]), "not one of"),
    "a note that is not text": (lambda f: f["labels"][0].update(note=3), "is not text"),
    "a note too long": (
        lambda f: f["labels"][0].update(note="x" * (NOTE_LIMIT + 1)),
        "Shorten it",
    ),
    "an item missing": (lambda f: f["labels"].pop(), "have no label: B08"),
    "counts that are not the file's": (
        lambda f: f.update(counted={"labels": 24, "notes": 0}),
        "it was edited",
    ),
}


@pytest.mark.parametrize("wrong", WRONG)
def test_a_labels_file_that_does_not_fit_is_refused_with_what_to_do(
    batch: Batch, wrong: str
) -> None:
    data = labels_for(batch)
    change, message = WRONG[wrong]
    replaced = change(data)
    if isinstance(replaced, list):
        data = replaced  # type: ignore[assignment]

    with pytest.raises(LabelsError) as refused:
        read_labels(data, batch)

    assert message in str(refused.value), str(refused.value)


def test_a_file_with_a_byte_order_mark_is_read(tmp_path: Path, batch: Batch) -> None:
    path = tmp_path / "labels-batch-1-2026-11-29-100000.json"
    path.write_text("﻿" + json.dumps(labels_for(batch)), encoding="utf-8")

    assert read_labels(load_labels_file(path), batch).batch == 1


def test_a_newer_file_of_the_same_batch_beside_the_given_one_is_found(
    tmp_path: Path, batch: Batch
) -> None:
    folder = tmp_path / "downloads"
    folder.mkdir()
    given = folder / "labels-batch-1-2026-11-29-100000.json"
    given.write_text(json.dumps(labels_for(batch)))
    newer = folder / f"{file_name_prefix(1)}2026-11-29-101500.json"
    newer.write_text(json.dumps(labels_for(batch, saved_at="2026-11-29T10:15:00.000Z")))
    renamed = folder / "labels-batch-1 (1).json"
    renamed.write_text(json.dumps(labels_for(batch, saved_at="2026-11-29T10:20:00.000Z")))
    (folder / "labels-batch-2-2026-11-30-090000.json").write_text(
        json.dumps({**labels_for(batch, saved_at="2026-11-30T09:00:00Z"), "batch": 2})
    )
    (folder / "labels-batch-1-broken.json").write_text("{not json")
    older = folder / "labels-batch-1-2026-11-29-090000.json"
    older.write_text(json.dumps(labels_for(batch, saved_at="2026-11-29T09:00:00.000Z")))

    labels = read_labels(load_labels_file(given), batch)
    found = newer_files(given, labels)

    assert [(path.name, when) for path, when in found] == [
        (renamed.name, "2026-11-29T10:20:00.000Z"),
        (newer.name, "2026-11-29T10:15:00.000Z"),
    ]
    newest = read_labels(load_labels_file(renamed), batch)
    assert newer_files(renamed, newest) == []
