"""The committed batches of the person's sample are what the code draws from the committed sets.

Each batch under ``evals/judge/human`` is read back and held to the two labelled sets and the
judges' saved checks: its items show the words of their cases, with passages whose hash is
the one the set records; its key, drawn again from the seed, the sets and the saved checks,
has the hash its ``batch.json`` holds; nothing in its folder names a label, a kind or a group
for an item until its labels are in place; and nothing the person is shown tells a label by
the checks of :func:`handbook_rag.evals.human_sample.leaks`.

The key is drawn again here as the tests draw it, without the handbook: a saved verdict is
matched to its case by id. The one test that needs the handbook draws the batch again in full,
items and all, as the commands do, and is marked slow and skipped where there is no snapshot.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.correctness import CorrectnessCase, load_labelled_correctness
from handbook_rag.evals.human_sample import (
    BATCHES_DIR,
    CORRECT_PART,
    KEY_FILE,
    LABELS_FILE,
    SUPPORT_PART,
    Batch,
    DrawnBatch,
    Sets,
    find_disputes,
    key_entries,
    leaks,
    load_saved_checks,
    read_batches,
    read_set_aside,
    reader_splits,
    redraw,
    unit_hash,
)
from handbook_rag.evals.judge import JudgedStatement, SupportCase
from handbook_rag.evals.judging import LabelledCase
from handbook_rag.evals.label_page import how_to_label, page_leaks, read_rubrics, render_label_page
from handbook_rag.evals.support import passage_hash

ROOT = Path(__file__).parents[2]
SUPPORT_SET = ROOT / "evals" / "judge" / "support" / "cases.jsonl"
CORRECT_SET = ROOT / "evals" / "judge" / "correct" / "cases.jsonl"
SNAPSHOT = ROOT / "data" / "handbook"
WHILE_UNLABELLED = {"HOW-TO-LABEL.md", "batch.json", "items.jsonl", "label.html"}


def batches() -> list[Batch]:
    return read_batches(ROOT / BATCHES_DIR)


def support_rows() -> dict[str, dict[str, Any]]:
    rows = (json.loads(line) for line in SUPPORT_SET.read_text().splitlines() if line)
    return {row["case"]: row for row in rows}


def correct_rows() -> dict[str, dict[str, Any]]:
    rows = (json.loads(line) for line in CORRECT_SET.read_text().splitlines() if line)
    return {row["case"]: row for row in rows}


def support_without_passages() -> list[LabelledCase[SupportCase]]:
    """The first set as the drawing sees it: the drawing does not read the passages."""
    return [
        LabelledCase(
            SupportCase(
                row["case"],
                row["question"],
                (),
                tuple(
                    JudgedStatement(s["number"], s["text"], tuple(s["cites"]))
                    for s in row["statements"]
                ),
            ),
            row["kind"],
            {s["number"]: s["expected"] for s in row["statements"]},
            {s["number"]: s["planted"] for s in row["statements"] if "planted" in s},
        )
        for row in support_rows().values()
    ]


def committed_sets() -> Sets:
    """The sets and their disputes, without the handbook: saved verdicts are matched by case."""
    support = support_without_passages()
    correct: list[LabelledCase[CorrectnessCase]] = load_labelled_correctness(CORRECT_SET, {})
    splits = {SUPPORT_PART: reader_splits(SUPPORT_SET), CORRECT_PART: reader_splits(CORRECT_SET)}
    saved = load_saved_checks(ROOT)
    disputes = find_disputes(support, correct, saved, splits, by_fingerprint=False)
    return Sets(support, correct, disputes, read_set_aside(ROOT / BATCHES_DIR))


def drawn_again() -> list[DrawnBatch]:
    return redraw(batches(), committed_sets(), with_items=False)


def test_batch_1_is_committed() -> None:
    assert [batch.number for batch in batches()][:1] == [1]


def test_each_key_drawn_again_has_the_hash_written_when_its_batch_was_drawn() -> None:
    # redraw refuses a key whose hash is not the one in batch.json.
    assert [drawn.number for drawn in drawn_again()] == [batch.number for batch in batches()]


def test_no_key_or_label_is_in_a_batch_s_folder_until_its_labels_are_in_place() -> None:
    ids = [*support_rows(), *correct_rows()]
    for batch in batches():
        names = {path.name for path in batch.directory.iterdir()}
        if LABELS_FILE not in names:
            assert names == WHILE_UNLABELLED, batch.number
        for path in batch.directory.iterdir():
            if path.name in {KEY_FILE, LABELS_FILE}:
                continue
            text = path.read_text()
            for field in ('"expected"', '"stratum"', '"kind"', '"disputes"', '"case"'):
                assert field not in text, (path.name, field)
            for case_id in ids:
                assert not re.search(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", text), path.name


def test_no_batch_holds_a_statement_or_answer_set_aside() -> None:
    set_aside = read_set_aside(ROOT / BATCHES_DIR)
    assert len(set_aside) == 24
    for drawn in drawn_again():
        units = {unit_hash(c.part, c.case_id, c.statement) for c in drawn.picked}
        assert not units & set_aside, drawn.number


def test_every_item_shows_the_words_of_its_case() -> None:
    support, correct = support_rows(), correct_rows()
    for drawn in drawn_again():
        batch = next(b for b in batches() if b.number == drawn.number)
        for item_id, entry in key_entries(drawn.key).items():
            item = batch.items[item_id]
            if entry.part == "support":
                row = support[entry.case]
                statement = next(s for s in row["statements"] if s["number"] == entry.statement)
                assert item["question"] == row["question"] == entry.question
                assert item["statements"] == [
                    {"number": s["number"], "text": s["text"], "cites": s["cites"]}
                    for s in row["statements"]
                ]
                assert item["marked"] == entry.statement
                assert entry.expected == statement["expected"]
                hashes = {p["number"]: p["text_sha256"] for p in row["passages"]}
                assert [p["number"] for p in item["passages"]] == statement["cites"]
                for passage in item["passages"]:
                    assert passage_hash(passage["text"]) == hashes[passage["number"]], item_id
            else:
                row = correct[entry.case]
                assert (item["question"], item["reference"], item["answer"]) == (
                    row["question"],
                    row["reference"],
                    row["answer"],
                )
                assert entry.expected == row["expected"]


def test_every_item_has_a_saved_verdict_of_each_judge_of_its_part() -> None:
    saved = load_saved_checks(ROOT)
    for drawn in drawn_again():
        for entry in key_entries(drawn.key).values():
            for verdicts in saved:
                if verdicts.check.part == entry.part:
                    given = verdicts.on(entry.case, entry.statement, None)
                    assert given is not None, (verdicts.check.name, entry.item)


def test_what_the_person_sees_tells_no_label_by_the_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    rubrics = read_rubrics()
    sets = committed_sets()
    for drawn in drawn_again():
        batch = next(b for b in batches() if b.number == drawn.number)
        shown = "".join(
            (batch.directory / name).read_text()
            for name in ("items.jsonl", "label.html", "HOW-TO-LABEL.md")
        )
        found = leaks(list(batch.items.values()), key_entries(drawn.key), sets, shown=shown)
        assert found == [], batch.number
        page = (batch.directory / "label.html").read_text()
        assert page_leaks(page, drawn, rubrics) == [], batch.number


def test_the_page_and_the_steps_are_what_the_code_writes_for_the_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    rubrics = read_rubrics()
    for drawn in drawn_again():
        batch = next(b for b in batches() if b.number == drawn.number)
        assert (batch.directory / "label.html").read_text() == render_label_page(drawn, rubrics)
        assert (batch.directory / "HOW-TO-LABEL.md").read_text() == how_to_label(drawn)


def test_an_address_in_a_batch_is_handbook_text_inside_the_page_s_data() -> None:
    for batch in batches():
        page = (batch.directory / "label.html").read_text()
        start = page.index('<script type="application/json" id="batch-data">')
        end = page.index("</script>", start)
        outside = page[:start] + page[end:]
        assert "https:" not in outside
        assert "http:" not in outside
        assert "//" not in outside
        for item in batch.items.values():
            texts = [item["question"], *[s["text"] for s in item.get("statements", [])]]
            texts += [item.get("reference", ""), item.get("answer", "")]
            assert not any("://" in text for text in texts), (
                "an address may come only from a passage"
            )


def test_no_file_of_a_batch_names_a_path_of_this_machine() -> None:
    for batch in batches():
        for path in batch.directory.iterdir():
            text = path.read_text()
            for prefix in ("/Users/", "/private/", "/tmp/", "/home/", "C:\\"):
                assert prefix not in text, (path.name, prefix)


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_each_batch_drawn_again_with_the_handbook_has_its_items_and_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from handbook_rag.evals.correctness import CORRECT
    from handbook_rag.evals.support import SUPPORT
    from handbook_rag.ingest.build import build_corpus

    monkeypatch.chdir(ROOT)
    chunks = {chunk.chunk_id: chunk for chunk in build_corpus(Settings()).chunks}
    support = SUPPORT.load_labelled(SUPPORT.labelled_path, chunks)
    correct = CORRECT.load_labelled(CORRECT.labelled_path, chunks)
    splits = {SUPPORT_PART: reader_splits(SUPPORT_SET), CORRECT_PART: reader_splits(CORRECT_SET)}
    disputes = find_disputes(support, correct, load_saved_checks(), splits)
    sets = Sets(support, correct, disputes, read_set_aside(ROOT / BATCHES_DIR))

    # redraw refuses items or a key that are not those committed.
    assert len(redraw(batches(), sets, with_items=True)) == len(batches())
