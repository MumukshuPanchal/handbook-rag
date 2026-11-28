"""The labelling page's script, run: a set of scenarios through a small fake page in node.

``tests/evals/test_human_sample.py`` reads the script's source. This runs it, with ``node``
and ``tests/label_page_harness.mjs``, on the page data of a batch drawn from the tests' small
sets: every item labelled with keys and with clicks; a note typed with keys that are also
choices; a label changed after going back; the save button pressed too early and in time,
and the file it saves read back; the browser's copy kept after a save and cleared only when
asked; labels found from before, shown with when they were made and taken only when the
person says so, while labels of another draw or batch are never taken; storage that refuses
or is full; and the page open in a second tab. The project adds no JavaScript toolchain:
node runs the script as it is. Without node the tests are skipped, and say why.

A second test plants faults in the script, those of the review that only a snapshot caught
among them, and asks that the first would have seen each.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.human_sample import Plan, make_batch
from handbook_rag.evals.label_page import SCRIPT, page_data
from tests.evals import human_fixtures as fixtures

NODE = shutil.which("node")
HARNESS = Path(__file__).parents[1] / "label_page_harness.mjs"
SAVED_AT = "2026-11-29T10:15:30.000Z"
FILE_KEYS = {
    "format",
    "version",
    "batch",
    "labeller",
    "pass",
    "items_sha256",
    "saved_at",
    "counted",
    "labels",
}

pytestmark = pytest.mark.skipif(
    NODE is None, reason="node is not installed: the page's script cannot be run here"
)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("project")
    sets = fixtures.make_project(root)
    drawn = make_batch(number=1, seed=7, plan=Plan(), sets=sets, earlier=[], sources={})
    return page_data(drawn, {"support": "Support rubric.", "correct": "Correct rubric."})


def run(source: str, data: dict[str, Any], tmp_path: Path) -> dict[str, dict[str, Any]]:
    """What the page showed and kept after each step of the scenarios, by step name."""
    script, page = tmp_path / "label.js", tmp_path / "data.json"
    script.write_text(source, encoding="utf-8")
    page.write_text(json.dumps(data), encoding="utf-8")
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(HARNESS), str(script), str(page)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "TZ": "UTC"},
    )
    assert done.returncode == 0, done.stderr
    return {step["name"]: step for step in json.loads(done.stdout)}


def stored_labels(step: dict[str, Any], own: str) -> dict[str, Any]:
    return dict(json.loads(step["stored"][own])["labels"]) if own in step["stored"] else {}


def faults(steps: dict[str, dict[str, Any]], data: dict[str, Any]) -> list[str]:
    """What the page did wrong in the scenarios, in words."""
    found: list[str] = []
    items = data["items"]
    count = len(items)
    own = f"handbook-rag labels, batch {data['batch']}, {data['items_sha256']}"

    def choice(index: int, place: int) -> str:
        return str(data["choices"][items[index]["part"]][place])

    keys, saved = steps["keys"], steps["saved"]
    marked = [
        [item["item"], data["names"][choice(index, index % 3)]] for index, item in enumerate(items)
    ]
    if keys["shown"] != marked:
        found.append("a label goes to another item than the one shown")
    if keys["progress"] != f"{count} of {count} items labelled" or keys["done"] != count:
        found.append("keys do not label every item")
    if keys["saveDisabled"]:
        found.append("save stays off with every item labelled")
    if set(keys["stored"]) != {own}:
        found.append(
            "the browser's copy is not kept under a key that names the batch and its items"
        )
    if len(saved["downloads"]) != 1:
        found.append("saving gives no file")
    else:
        download = saved["downloads"][0]
        file = json.loads(download["text"])
        if download["name"] != f"{data['file_prefix']}2026-11-29-101530.json":
            found.append("the saved file is not named by its batch and the time of the save")
        labels = {entry["item"]: entry["label"] for entry in file["labels"]}
        if labels != {item["item"]: choice(index, index % 3) for index, item in enumerate(items)}:
            found.append("the saved file does not hold what was labelled")
        if (
            set(file) != FILE_KEYS
            or file["counted"] != {"labels": count, "notes": 0}
            or file["saved_at"] != SAVED_AT
            or file["labeller"] != "the project's owner"
            or file["items_sha256"] != data["items_sha256"]
            or [entry["item_sha256"] for entry in file["labels"]] != [i["sha256"] for i in items]
        ):
            found.append("the saved file does not hold what was labelled")
        if download["name"] not in saved["status"] or "Saved at" not in saved["status"]:
            found.append("the page does not say the name and time of the last save")
    if own not in saved["stored"]:
        found.append("saving clears the browser's copy")
    refused = steps["clear refused"]
    if own not in refused["stored"] or refused["asked"] < 1:
        found.append("the browser's copy is cleared without asking, or against the answer")
    if own in steps["cleared"]["stored"]:
        found.append("the clear button does not clear the browser's copy")

    early = steps["early save"]
    if early["downloads"] or not early["saveDisabled"]:
        found.append("save works before every item is labelled")

    clicks = steps["clicks"]
    expected = {item["item"]: choice(index, index % 2) for index, item in enumerate(items)}
    given = (
        {e["item"]: e["label"] for e in json.loads(clicks["downloads"][0]["text"])["labels"]}
        if clicks["downloads"]
        else {}
    )
    if given != expected:
        found.append("clicks do not label the item shown")

    typing = steps["typing"]
    if typing["item"] != "A01" or typing["pressed"] is not None:
        found.append("a key typed in the note acts on the page")
    if stored_labels(typing, own).get("A01", {}).get("note") != "2 n":
        found.append("a note is not kept")
    if steps["after typing"]["pressed"] != data["names"][choice(0, 1)]:
        found.append("keys do not work again after Escape")
    back = steps["back"]
    if back["item"] != "A01" or back["pressed"] != data["names"][choice(0, 1)]:
        found.append("going back to change a label does not change it")

    first = steps["found"]
    if not first["found"] or "2 labels and 1 notes" not in first["foundText"]:
        found.append("labels found from before are not shown with how many there are")
    if "2026" not in first["foundText"]:
        found.append("labels found from before are not shown with when they were made")
    pressed = steps["found, a key pressed"]
    if first["done"] or not first["choicesDisabled"] or pressed["done"] or pressed["downloads"]:
        found.append("labels found from before are taken without asking")
    kept = steps["kept"]
    if (
        kept["done"] != 2
        or kept["found"]
        or stored_labels(kept, own).get("A01", {}).get("note") != "kept"
    ):
        found.append("labels found are not restored as they were, or labels that do not fit are")
    again = steps["started again"]
    if again["done"] or own in again["stored"] or again["found"]:
        found.append("starting again keeps the labels found")
    other = steps["other draw under own key"]
    if other["found"] or other["done"]:
        found.append("labels of another draw or another batch are restored")

    for name, words in (("refused", "keeps nothing"), ("full", "is full")):
        step = steps[name]
        if not step["warning"] or "not being kept" not in step["storageNote"]:
            found.append(f"a browser that {words} is not said")
        if not step["unloadBlocked"]:
            found.append(f"closing the page loses its labels unwarned when the browser {words}")
    tab = steps["another tab"]
    if not tab["warning"] or "another tab" not in tab["storageNote"]:
        found.append("a second tab is not said")
    if stored_labels(tab, own).get("A02", {}).get("label") is not None:
        found.append("a second tab's copy is written over")
    return found


def test_the_page_does_what_it_should(data: dict[str, Any], tmp_path: Path) -> None:
    assert faults(run(SCRIPT, data, tmp_path), data) == []


# Faults planted in the script. J1 to J5 are those of the review that only the snapshot of
# the batch caught; the others are the new behaviours this run holds.
PLANTED: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "save works before every item is labelled",  # J1
        [
            (
                'byId("save").disabled = done !== items.length || found !== null;',
                'byId("save").disabled = found !== null;',
            ),
            (
                "if (labelled() !== items.length || found !== null) { return; }",
                "if (found !== null) { return; }",
            ),
        ],
    ),
    (
        "a key typed in the note acts on the page",  # J2
        [
            (
                'if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) {',
                "if (false) {",
            )
        ],
    ),
    (
        "labels found are not restored as they were, or labels that do not fit are",  # J3
        [
            (
                "label: allowed(item, kept.label) ? kept.label : null,",
                'label: typeof kept.label === "string" ? kept.label : null,',
            )
        ],
    ),
    (
        "the browser's copy is not kept under a key that names the batch and its items",  # J4
        [
            (
                'var storageKey = "handbook-rag labels, batch " + data.batch + ", " '
                "+ data.items_sha256;",
                'var storageKey = "handbook-rag labels, batch " + data.batch;',
            )
        ],
    ),
    (
        "a label goes to another item than the one shown",  # J5
        [
            (
                "entry(items[current]).label = label;",
                "entry(items[Math.min(current + 1, items.length - 1)]).label = label;",
            )
        ],
    ),
    (
        "the saved file is not named by its batch and the time of the save",  # P6
        [
            (
                'var name = data.file_prefix + stamp(moment) + ".json";',
                'var name = "labels-batch-" + data.batch + ".json";',
            )
        ],
    ),
    (
        "saving clears the browser's copy",
        [
            (
                "    saved = { name: name,",
                "    try { storage.removeItem(storageKey); } catch (error) { problem = null; }\n"
                "    saved = { name: name,",
            )
        ],
    ),
    (
        "the browser's copy is cleared without asking, or against the answer",
        [("if (!storage || !window.confirm(", "if (!storage || !Boolean(")],
    ),
    (
        "a browser that is full is not said",
        [('problem = "the browser refused to keep more (its storage may be full)";', "return;")],
    ),
    (
        "labels found from before are taken without asking",
        [
            (
                "  found = readStored();",
                "  found = null;\n  state = (readStored() || { labels: {} }).labels;",
            )
        ],
    ),
    (
        "labels of another draw or another batch are restored",
        [("stored.items_sha256 !== data.items_sha256 ||", "")],
    ),
    (
        "a second tab is not said",
        [
            (
                'problem = "this page is open in another tab or window as well',
                'var ignored = "this page is open in another tab or window as well',
            )
        ],
    ),
    (
        "going back to change a label does not change it",
        [
            ('event.key === "p" || event.key === "P"', 'event.key === "P"'),
            ('event.key === "ArrowLeft" || ', ""),
        ],
    ),
    (
        "a note is not kept",
        [("entry(items[current]).note = event.target.value.slice(0, data.note_limit);", "")],
    ),
]


@pytest.mark.parametrize(("fault", "changes"), PLANTED)
def test_the_run_finds_each_fault_planted_in_the_script(
    fault: str, changes: list[tuple[str, str]], data: dict[str, Any], tmp_path: Path
) -> None:
    source = SCRIPT
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    assert fault in faults(run(source, data, tmp_path), data)
