"""What the documents say of the person's sample is what its files hold, and no more.

While a batch has no ``labels.json``, every document that speaks of the sample says that it
waits for the owner's labels, and none says that a person has labelled anything. Once a
batch's labels are in place, the test that waits for them fails until every document stops
saying that batch waits and gives the figures of a kept report, as the project's tests have
waited for runs before.

The figures the card and the decision record give before any label exists, the disputes the
sets hold and the make-up of batch 1, are worked out here from the sets, the saved checks
and the key drawn again, which is never committed before the labels.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.human_sample import (
    BATCHES_DIR,
    LABELS_FILE,
    DrawnBatch,
    candidates,
    read_batches,
    read_set_aside,
    unit_hash,
)
from handbook_rag.evals.label_page import READ_NOTHING_FIRST
from handbook_rag.evals.metrics import share_with_interval
from tests.evals.test_committed_human_batch import committed_sets, drawn_again

ROOT = Path(__file__).parents[2]
CARD = ROOT / BATCHES_DIR / "README.md"
ADR = ROOT / "docs" / "adr" / "0026-one-person-labels-a-sample-of-the-labelled-answers.md"
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "docs" / "evaluation.md",
    ROOT / "docs" / "roadmap.md",
    ROOT / "CHANGELOG.md",
    ROOT / "evals" / "judge" / "README.md",
    CARD,
    ADR,
)
RUNS = ROOT / "evals" / "runs" / "judge" / "human"
FIRST_LINE = "> **If you are the one who labels, read this after you have labelled.**"
# Words that would say a person has labelled something. Each document is searched for them
# while no labels are in place.
CLAIMS = (
    "the owner labelled",
    "the owner has labelled",
    "labelled by the owner",
    "the owner's labels show",
    "the owner's labels agree",
    "a person labelled",
    "a person has labelled",
    "labelled by a person",
    "verified by the owner",
    "human-labelled",
)
# Sentences that are true only while no batch is labelled.
ONLY_WHILE_WAITING = (
    "No person has labelled any item yet",
    "no figure of agreement with a person exists",
)


def pending(number: int) -> str:
    return f"batch {number} waits for the owner's labels"


def labelled() -> list[int]:
    return [
        b.number for b in read_batches(ROOT / BATCHES_DIR) if (b.directory / LABELS_FILE).exists()
    ]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_every_document_says_which_batches_wait_for_the_owner_s_labels() -> None:
    done = set(labelled())
    for batch in read_batches(ROOT / BATCHES_DIR):
        if batch.number in done:
            continue
        for path in DOCUMENTS:
            assert pending(batch.number) in read(path).lower(), (path.name, batch.number)


def test_while_no_labels_are_in_place_no_document_says_a_person_has_labelled_anything() -> None:
    if labelled():
        pytest.skip("labels are in place: the test below holds the documents to the report")
    steps = [b.directory / "HOW-TO-LABEL.md" for b in read_batches(ROOT / BATCHES_DIR)]
    for path in (*DOCUMENTS, *steps):
        text = read(path).lower()
        for claim in CLAIMS:
            assert claim not in text, (path.name, claim)


def test_once_labels_are_in_place_the_documents_give_the_figures_of_a_kept_report() -> None:
    present = labelled()
    if not present:
        pytest.skip(
            "the owner's labels are not in place yet: no batch under evals/judge/human holds "
            "labels.json. The documents say batch 1 waits, and give no figure."
        )
    kept = [
        json.loads(read(summary))
        for summary in RUNS.glob("*/summary.json")
        if summary.parent.name != "current"
        and {entry["batch"] for entry in json.loads(read(summary))["batches"]} == set(present)
    ]
    assert kept, "save the report of every labelled batch with `eval-human --variant <name>`"
    figures = [
        "{successes} of {count}".format(**part["against_the_set"]["same_decision"])
        for part in kept[0]["parts"].values()
        if part["against_the_set"]["same_decision"] is not None
    ]
    for path in DOCUMENTS:
        text = read(path)
        for number in present:
            assert pending(number) not in text.lower(), (path.name, number)
        for sentence in ONLY_WHILE_WAITING:
            assert sentence not in text, (path.name, sentence)
        for figure in figures:
            assert figure in text, (
                f"{path.name} does not give the figure {figure} of the kept report: give the "
                "figures, and hold them here"
            )


def test_the_card_and_the_decision_record_open_by_asking_the_labeller_to_read_them_later() -> None:
    for path in (CARD, ADR):
        lines = [line for line in read(path).splitlines() if line.strip()]
        assert lines[1].startswith(FIRST_LINE), path.name


def batch_1() -> tuple[DrawnBatch, dict[str, Any]]:
    (first, *_) = drawn_again()
    return first, first.key["items"]


def test_the_card_counts_the_disputes_the_sets_and_the_saved_checks_hold() -> None:
    sets = committed_sets()
    support = {k: v for k, v in sets.disputes.items() if k[0] == "support"}
    answers = {k: v for k, v in sets.disputes.items() if k[0] == "correct"}
    sorts = Counter(sort for value in support.values() for sort in value)
    questions = {sets.case("support", case).case.question for _, case, _ in support}
    both = sum(1 for value in support.values() if {"support_v1", "support_v2"} <= set(value))
    readers_and_v1 = sum(1 for value in support.values() if {"support_v1", "readers"} <= set(value))
    shared = questions & {sets.case("correct", case).case.question for _, case, _ in answers}
    card = read(CARD)

    assert (
        f"{len(support)} statements are disputed, on {len(questions)} questions: "
        f"{sorts['support_v1']} on which version 1 of the judge of support differs from the set, "
        f"{sorts['support_v2']} on which version 2 does"
    ) in card
    assert both == 1
    assert readers_and_v1 == 1
    assert f"{sorts['readers']} on which the readers differed" in card
    assert f"and {sorts['unclear']} the set calls `unclear`" in card
    assert f"{len(answers)} answers are disputed" in card
    assert len(shared) == 1
    assert (
        f"{len(support)} statements on {len(questions)} questions, and {len(answers)} answers"
        in read(ADR)
    )


def test_the_card_gives_batch_1_as_its_key_drawn_again_holds_it() -> None:
    drawn, entries = batch_1()
    groups = Counter((e["part"], e["stratum"]) for e in entries.values())
    disputed = [e for e in entries.values() if e["stratum"] == "disputed"]
    sorts = Counter(e["kind"] for e in disputed if e["part"] == "support")
    card, adr = read(CARD), read(ADR)
    faulted = groups[("support", "unsupported")]
    kinds = [e["kind"] for e in entries.values() if e["stratum"] == "unsupported"]
    not_correct = Counter(e["expected"] for e in entries.values() if e["stratum"] == "not_correct")
    a = Counter(e["expected"] for e in disputed if e["part"] == "support")
    b = Counter(e["expected"] for e in disputed if e["part"] == "correct")
    rows = {
        "Disputed statements": (
            f"A | {groups[('support', 'disputed')]}: {sorts['support_v1']} on which version 1 of "
            f"the judge of support differs from the set, {sorts['support_v2']} on which version 2 "
            f"differs, {sorts['readers']} on which the readers differed, and {sorts['unclear']} "
            "the set calls `unclear` |"
        ),
        "Statements labelled `unsupported`": (
            f"A | {faulted}: one of each of {len(set(kinds))} kinds of planted fault |"
        ),
        "Statements of real answers labelled `supported`": (
            f"A | {groups[('support', 'supported')]} |"
        ),
        "Disputed answers": f"B | {groups[('correct', 'disputed')]}, on each of which",
        "Answers labelled `partial` or `incorrect`": (
            f"B | {sum(not_correct.values())}: {not_correct['partial']} `partial` and "
            f"{not_correct['incorrect']} `incorrect` |"
        ),
        "Real answers labelled `correct`": f"B | {groups[('correct', 'correct')]} |",
    }

    assert f"with the seed {drawn.seed}" in card
    assert len(set(kinds)) == faulted
    for group, row in rows.items():
        assert f"| {group} | {row}" in card, group
    assert (
        f"Of the {sum(a.values())} disputed statements the set labels {a['supported']} "
        f"`supported`, {a['unsupported']} `unsupported`, {a['no_claim']} `no_claim` and "
        f"{a['unclear']} `unclear`; of the {sum(b.values())} disputed answers it labels "
        f"{b['correct']} `correct` and {b['partial']} `partial`"
    ) in card
    assert (
        f"| A, 16 statements | {groups[('support', 'disputed')]}: version 1 differs from the set "
        f"on {sorts['support_v1']}, version 2 on {sorts['support_v2']}, the readers on "
        f"{sorts['readers']}, and {sorts['unclear']} the set calls `unclear` | {faulted} "
        "`unsupported`"
    ) in adr
    assert (
        f"| B, 8 answers | {groups[('correct', 'disputed')]}: the judge of correctness differs "
        f"from the set on each | {sum(not_correct.values())}: {not_correct['partial']} "
        f"`partial`, {not_correct['incorrect']} `incorrect` | {groups[('correct', 'correct')]} "
        "`correct` |"
    ) in adr


def test_the_card_gives_the_shapes_and_the_judges_differences_in_batch_1() -> None:
    _, entries = batch_1()
    shapes = {(c.part, c.case_id, c.statement): c.shape for c in candidates(committed_sets())}
    shape = {item: shapes[(e["part"], e["case"], e["statement"])] for item, e in entries.items()}
    by_group = {
        group: Counter(
            shape[item]
            for item, e in entries.items()
            if e["stratum"] == group and e["part"] == "support"
        )
        for group in ("unsupported", "supported", "disputed")
    }
    undisputed = [e for e in entries.values() if e["stratum"] != "disputed"]
    differs = Counter(
        sort
        for e in entries.values()
        for sort in e["disputes"]
        if sort not in {"readers", "unclear"}
    )
    card = read(CARD)
    several = by_group["disputed"]["several"]

    assert by_group["unsupported"] == by_group["supported"] == Counter({"one": 5})
    assert "The 5 statements labelled `unsupported` all come from answers of one statement" in card
    assert f"{several} of the 6 disputed statements come from answers of several statements" in card
    assert f"On the {len(undisputed)} items that are not disputed" in card
    assert f"On the {len(undisputed)} items that are not disputed" in read(ADR)
    assert (
        f"on {differs['support_v1']} for version 1, on {differs['support_v2']} for version 2, "
        f"and on {differs['correct']} for the judge of correctness"
    ) in card


def test_the_first_draw_is_set_aside_and_shares_only_questions_with_batch_1() -> None:
    sets = committed_sets()
    set_aside = read_set_aside(ROOT / BATCHES_DIR)
    aside_questions = {
        c.question
        for c in candidates(type(sets)(sets.support, sets.correct, sets.disputes))
        if unit_hash(c.part, c.case_id, c.statement) in set_aside
    }
    _, entries = batch_1()
    shared = aside_questions & {e["question"] for e in entries.values()}

    assert len(set_aside) == 24
    assert f"{len(shared)} questions of batch 1 were in the first draw too" in read(CARD)
    assert f"{len(shared)} questions of the new batch were in the first draw" in read(ADR)


def test_the_widths_the_card_gives_are_wilson_intervals() -> None:
    card = read(CARD)
    five, sixteen = share_with_interval(5, 5).low, share_with_interval(16, 16).low

    assert f"one label would still leave an interval that reaches down to {five:.3f}" in card
    assert "agreement on all 5 statements of one label" in card
    assert f"on all 16 statements of part A, down to {sixteen:.3f}" in card


def test_the_steps_beside_each_page_are_five_with_the_one_thing_not_to_do() -> None:
    for batch in read_batches(ROOT / BATCHES_DIR):
        steps = read(batch.directory / "HOW-TO-LABEL.md")
        assert re.findall(r"^(\d)\. \*\*", steps, re.M) == ["1", "2", "3", "4", "5"]
        assert "**The one thing not to do: do not look for how this batch was made" in steps
        assert READ_NOTHING_FIRST in steps
        assert f"human-labels ~/Downloads/labels-batch-{batch.number}-<date>-<time>.json" in steps
        assert "the newest file holds your labels" in steps
        assert f"eval-human --variant batch-{batch.number}" in steps
