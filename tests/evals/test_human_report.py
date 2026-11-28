"""The report of a person's labels against the sets and the judges, on stand-in labels.

The labels here are made by the test, item by item, so that every figure can be worked out
by hand in the comments. They are nobody's labels. The items are chosen so that figures a
slip could swap or merge come out different: the owner's faults the judge faulted and the
judge's faults the owner agrees with; an item that is both "cannot tell" and without a
saved verdict; and a judge whose verdicts differ from the owner's, so that kappa is not 1.
"""

from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.human_labels import Label, Labels
from handbook_rag.evals.human_report import (
    LabelledBatch,
    Row,
    differs,
    in_full,
    report,
    rows,
    write_report,
)
from handbook_rag.evals.human_sample import Batch, KeyEntry, Plan, SavedCheck, SavedVerdicts
from handbook_rag.evals.judge import Judgement, StatementVerdict

V1 = SavedCheck(
    "support", "support_v1", "version 1", Path("runs/v1"), ("supported", "unsupported", "no_claim")
)
V2 = SavedCheck(
    "support", "support_v2", "version 2", Path("runs/v2"), ("supported", "unsupported", "no_claim")
)
JC = SavedCheck(
    "correct", "correct", "correctness", Path("runs/c"), ("correct", "partial", "incorrect")
)

# item: set's label, the stand-in label, version 1 (or correctness), version 2 (None: no
# saved verdict), and the disputes the key records.
ITEMS: dict[str, tuple[str, str, str, str | None, tuple[str, ...]]] = {
    "A01": ("supported", "supported", "supported", "supported", ()),
    "A02": ("supported", "unsupported", "supported", "unsupported", ("support_v2",)),
    "A03": ("unsupported", "unsupported", "unsupported", "unsupported", ()),
    "A04": ("unsupported", "supported", "unsupported", None, ()),
    "A05": ("unsupported", "cannot_tell", "unsupported", None, ()),
    "A06": ("supported", "no_claim", "no_claim", "supported", ("support_v1",)),
    "A07": ("unsupported", "unsupported", "supported", "unsupported", ("support_v1",)),
    "A08": ("unclear", "supported", "supported", "unsupported", ("unclear",)),
    "B01": ("correct", "correct", "correct", None, ()),
    "B02": ("incorrect", "partial", "incorrect", None, ()),
    "B03": ("partial", "cannot_tell", "partial", None, ()),
    "B04": ("correct", "incorrect", "partial", None, ("correct",)),
    "B05": ("correct", "cannot_tell", "partial", None, ("correct",)),
}


def fingerprint(part: str, case: str) -> str:
    return f"print-{case}"


def stand_in() -> tuple[LabelledBatch, list[SavedVerdicts]]:
    key = {
        item: KeyEntry(
            item=item,
            part="support" if item < "B" else "correct",
            case=f"case-{item}",
            statement=2 if item < "B" else 1,
            question=f"Question {item}?",
            stratum="disputed" if disputes else expected,
            kind="real",
            disputes=disputes,
            expected=expected,
        )
        for item, (expected, _, _, _, disputes) in ITEMS.items()
    }
    items: dict[str, dict[str, Any]] = {
        item: {"item": item, "part": entry.part, "question": entry.question, "sha256": item}
        for item, entry in key.items()
    }
    batch = Batch(1, Path("batch-1"), 7, Plan(), {}, "items-hash", "key-hash", items)
    labels = Labels(
        1,
        "the project's owner",
        1,
        "2026-11-29T10:00:00Z",
        {item: Label(item, owner, f"note {item}") for item, (_, owner, *_) in ITEMS.items()},
    )
    found: dict[str, dict[str, Judgement]] = {"v1": {}, "v2": {}, "c": {}}
    for item, (_, _, first, second, _) in ITEMS.items():
        entry = key[item]

        def judged(verdict: str, entry: KeyEntry = entry) -> Judgement:
            given = [StatementVerdict(entry.statement, verdict, f"why {verdict}")]
            if entry.part == "support":
                given.insert(0, StatementVerdict(1, "supported", "statement 1"))
            return Judgement(entry.case, tuple(given), "stand-in")

        found["v1" if entry.part == "support" else "c"][fingerprint("", entry.case)] = judged(first)
        if second is not None:
            found["v2"][fingerprint("", entry.case)] = judged(second)
    # A verdict on other words than were drawn is no verdict on this item.
    found["v2"]["another-print"] = Judgement(
        "case-A04", (StatementVerdict(2, "supported", "x"),), "m"
    )
    saved = [
        SavedVerdicts(V1, found["v1"]),
        SavedVerdicts(V2, found["v2"]),
        SavedVerdicts(JC, found["c"]),
    ]
    return LabelledBatch(batch, labels, key), saved


def figures() -> dict[str, Any]:
    done, saved = stand_in()
    summary, _ = report([done], [], saved, fingerprint)
    return summary


def counted(share: dict[str, Any] | None) -> tuple[int, int] | None:
    return None if share is None else (share["successes"], share["count"])


def test_part_a_against_the_set_by_hand() -> None:
    part = figures()["parts"]["support"]
    against = part["against_the_set"]

    assert part["labelled"] == 8
    assert part["cannot_tell"] == 1
    # A05 is "cannot tell" and A08 "unclear" in the set: 6 are compared.
    assert against["compared"] == 6
    assert against["unclear_in_the_set"] == 1
    # Same verdict: A01, A03, A07.
    assert counted(against["same_verdict"]) == (3, 6)
    # Fault or not: A01 no-no, A02 no-yes, A03 yes-yes, A04 yes-no, A06 no-no, A07 yes-yes.
    assert counted(against["same_decision"]) == (4, 6)
    # Observed 4/6; each side says "fault" 3 times in 6, so chance gives 0.5.
    assert against["kappa"] == pytest.approx((4 / 6 - 0.5) / 0.5, abs=1e-4)
    assert counted(against["by_class"]["supported"]["same"]) == (1, 3)
    assert counted(against["by_class"]["unsupported"]["same"]) == (2, 3)
    assert against["by_class"]["unsupported"]["cannot_tell"] == 1
    assert against["matrix"]["rows"] == "the set's label"
    assert against["matrix"]["counts"]["supported"] == {
        "supported": 1,
        "unsupported": 1,
        "no_claim": 1,
    }


def test_a_share_carries_its_count_and_a_wilson_interval_worked_by_hand() -> None:
    share = figures()["parts"]["support"]["against_the_set"]["same_verdict"]

    # 3 of 6 with z = 1.96: centre 0.5 + z²/12, spread z * sqrt(0.25/6 + z²/144), scale 1 + z²/6.
    assert share["mean"] == pytest.approx(0.5)
    assert share["low"] == pytest.approx(0.1876, abs=1e-4)
    assert share["high"] == pytest.approx(0.8124, abs=1e-4)


def test_part_a_against_version_1_by_hand() -> None:
    first = figures()["parts"]["support"]["against_the_judges"][0]

    assert first["judge"] == "version 1"
    assert (first["compared"], first["no_saved_verdict"]) == (7, 0)
    # Same verdict: A01, A03, A06, A08. Same decision: A01, A03, A06, A08.
    assert counted(first["same_verdict"]) == (4, 7)
    assert counted(first["same_decision"]) == (4, 7)
    # The owner faults A02, A03 and A07; the judge faults A03 of them.
    assert counted(first["owner_faults_the_judge_faulted"]) == (1, 3)
    # The judge faults A03 and A04; the owner faults A03 of them.
    assert counted(first["judge_faults_the_owner_agrees_with"]) == (1, 2)
    # Judge "fault" 2 in 7, owner 3 in 7: chance 26/49, observed 28/49, kappa 2/23.
    assert first["kappa"] == pytest.approx(2 / 23, abs=1e-4)
    assert first["matrix"]["rows"] == "the judge's verdict"
    assert first["matrix"]["counts"]["supported"]["unsupported"] == 2


def test_part_a_against_version_2_by_hand() -> None:
    second = figures()["parts"]["support"]["against_the_judges"][1]

    # A04 has no verdict on the words drawn; A05 has none either, and is "cannot tell", so
    # it is counted as "cannot tell" only.
    assert (second["compared"], second["no_saved_verdict"]) == (6, 1)
    assert counted(second["same_verdict"]) == (4, 6)
    assert counted(second["same_decision"]) == (5, 6)
    assert counted(second["owner_faults_the_judge_faulted"]) == (3, 3)
    assert counted(second["judge_faults_the_owner_agrees_with"]) == (3, 4)
    # Judge "fault" 4 in 6, owner 3 in 6: chance 0.5, observed 5/6.
    assert second["kappa"] == pytest.approx((5 / 6 - 0.5) / 0.5, abs=1e-4)


def test_where_a_judge_and_the_set_differ_the_report_says_with_whom_the_owner_sides() -> None:
    parts = figures()["parts"]
    first, second = parts["support"]["disputed"]["judges"]
    (correctness,) = parts["correct"]["disputed"]["judges"]

    # Version 1 differs from the set on A06 (the owner sides with it) and A07 (with the set).
    assert first["items_where_it_differs_from_the_set"] == ["A06", "A07"]
    assert first["the_owner_sides_with"] == {
        "the set": 1,
        "the judge": 1,
        "neither": 0,
        "cannot_tell": 0,
    }
    assert second["items_where_it_differs_from_the_set"] == ["A02"]
    assert second["the_owner_sides_with"]["the judge"] == 1
    # The judge of correctness differs on B04 (the owner says neither) and B05 (cannot tell).
    assert correctness["the_owner_sides_with"] == {
        "the set": 0,
        "the judge": 0,
        "neither": 1,
        "cannot_tell": 1,
    }
    unclear = parts["support"]["disputed"]["unclear_in_the_set"]
    assert unclear["items"] == ["A08"]
    assert unclear["the_owner_called_them"]["supported"] == 1


def test_part_b_by_hand() -> None:
    part = figures()["parts"]["correct"]
    against = part["against_the_set"]
    (judge,) = part["against_the_judges"]

    assert (part["labelled"], part["cannot_tell"]) == (5, 2)
    assert counted(against["same_verdict"]) == (1, 3)
    # Partial and incorrect are both faults, so B02 is the same decision; B04 is not.
    assert counted(against["same_decision"]) == (2, 3)
    assert against["by_class"]["partial"] == {"same": None, "cannot_tell": 1}
    assert counted(against["by_class"]["correct"]["same"]) == (1, 2)
    assert counted(judge["same_verdict"]) == (1, 3)
    assert counted(judge["same_decision"]) == (3, 3)
    assert counted(judge["owner_faults_the_judge_faulted"]) == (2, 2)


def test_kappa_is_left_out_where_one_side_makes_one_decision_only() -> None:
    done, saved = stand_in()
    always = Labels(
        1,
        done.labels.labeller,
        1,
        done.labels.saved_at,
        {item: Label(item, "supported" if item < "B" else "correct", "") for item in ITEMS},
    )
    summary, _ = report([LabelledBatch(done.batch, always, done.key)], [], saved, fingerprint)

    assert summary["parts"]["support"]["against_the_set"]["kappa"] is None


def test_each_difference_names_what_it_differs_from_and_cannot_tell_is_none() -> None:
    done, saved = stand_in()
    found = {row.item: differs(row) for row in rows([done], saved, fingerprint)}

    assert found == {
        "A01": [],
        "A02": ["the set's label", "version 1"],
        "A03": [],
        "A04": ["the set's label", "version 1"],
        "A05": [],
        "A06": ["the set's label", "version 2"],
        "A07": ["version 1"],
        "A08": ["version 2"],
        "B01": [],
        "B02": ["the set's label", "correctness"],
        "B03": [],
        "B04": ["the set's label", "correctness"],
        "B05": [],
    }
    summary = figures()
    assert (summary["differences"], summary["cannot_tell"], summary["disputed_items"]) == (7, 3, 6)


def test_the_batches_waiting_for_labels_are_reported() -> None:
    done, saved = stand_in()
    waiting = Batch(2, Path("batch-2"), 7, Plan(), {}, "h", "k", {})

    summary, _ = report([done], [waiting], saved, fingerprint)

    assert summary["waiting_for_labels"] == [2]
    assert [entry["batch"] for entry in summary["batches"]] == [1]


def test_a_verdict_given_to_another_case_with_the_same_fingerprint_is_not_used() -> None:
    done, saved = stand_in()
    wrong = saved[0].by_fingerprint["print-case-A01"]
    saved[0] = SavedVerdicts(
        V1, {**saved[0].by_fingerprint, "print-case-A01": Judgement("other", wrong.verdicts, "m")}
    )

    (row,) = [r for r in rows([done], saved, fingerprint) if r.item == "A01"]

    assert row.judged["version 1"] is None


def test_the_run_writes_its_files_with_the_notes_and_every_disputed_item_in_full(
    tmp_path: Path,
) -> None:
    done, saved = stand_in()
    summary, found = report([done], [], saved, fingerprint)

    write_report(tmp_path / "run", summary, found, [done], saved)

    names = sorted(path.name for path in (tmp_path / "run").iterdir())
    assert names == [
        "cannot-tell.jsonl",
        "differences.jsonl",
        "disputed.jsonl",
        "run.json",
        "summary.json",
    ]
    disputed = (tmp_path / "run" / "disputed.jsonl").read_text().splitlines()
    assert len(disputed) == 6
    row: Row = next(r for r in found if r.item == "A06")
    assert in_full(row, done.batch)["note"] == "note A06"
    assert '"handbook_rag_version"' in (tmp_path / "run" / "run.json").read_text()
