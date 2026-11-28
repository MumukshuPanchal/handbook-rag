"""How far one person's labels agree with the labelled sets and with the judges' saved verdicts.

No request is made to any model. The person's labels are compared with two things:

- **the label the set held when the batch was drawn**, which model sessions made and
  checked (``evals/judge/README.md``), as the batch's key records it. A statement the set
  calls ``unclear`` has no label to compare, and is counted apart;
- **the verdict each judge saved in its check on the labelled sets**, the same verdicts
  its published agreement rests on. A verdict counts only if it was given on exactly the
  words drawn, by the fingerprint of what the judge read. An item with no such verdict is
  counted apart, as having none, and never as agreement or disagreement.

"Cannot tell" is counted apart in the same way. Every share is given with its count and a
95% Wilson interval. Kappa is Cohen's kappa for the decision the scores rest on, a fault or
not, and is left out where either side makes only one decision, since it then says nothing
(see :attr:`handbook_rag.evals.agreement.Agreement.labels_hold_both_decisions`).

The disputed items, on which a judge and the set differ, two readers differed, or the set
says ``unclear``, have a section of their own: with whom the person sides.
"""

import datetime
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handbook_rag import __version__
from handbook_rag.evals.agreement import decision_kappa
from handbook_rag.evals.human_labels import Labels
from handbook_rag.evals.human_sample import (
    CANNOT_TELL,
    CHOICES,
    CORRECT_PART,
    FAULTS,
    PART_LETTERS,
    PARTS,
    SUPPORT_PART,
    Batch,
    KeyEntry,
    SavedVerdicts,
)
from handbook_rag.evals.judging import UNCLEAR
from handbook_rag.evals.metrics import share_with_interval

RUNS_DIR = Path("evals/runs/judge/human")
PART_TITLES = {
    SUPPORT_PART: "statements against the passages they cite",
    CORRECT_PART: "answers against their reference answers",
}
SIDES = ("the set", "the judge", "neither", CANNOT_TELL)


@dataclass(frozen=True, slots=True)
class Verdict:
    verdict: str
    reason: str


@dataclass(frozen=True, slots=True)
class Row:
    """One labelled item, with everything it is compared with."""

    batch: int
    item: str
    part: str
    case: str
    statement: int
    kind: str
    stratum: str
    disputes: tuple[str, ...]
    expected: str
    """The label the set held when the batch was drawn, or ``unclear``."""
    owner: str
    note: str
    judged: Mapping[str, Verdict | None]
    """By the name of each saved check of this part: its verdict, or ``None`` if it has none."""


@dataclass(frozen=True, slots=True)
class LabelledBatch:
    """A batch whose labels are in place, with its key drawn again."""

    batch: Batch
    labels: Labels
    key: Mapping[str, KeyEntry]


def rows(
    labelled: Sequence[LabelledBatch],
    saved: Sequence[SavedVerdicts],
    fingerprint: Callable[[str, str], str],
) -> list[Row]:
    """Every labelled item beside the set's label and each judge's verdict on it.

    ``fingerprint`` gives, for a part and a case, the fingerprint of the words drawn.
    """
    found: list[Row] = []
    for done in labelled:
        for item_id, label in done.labels.labels.items():
            entry = done.key[item_id]
            judged: dict[str, Verdict | None] = {}
            for verdicts in saved:
                if verdicts.check.part != entry.part:
                    continue
                given = verdicts.on(
                    entry.case, entry.statement, fingerprint(entry.part, entry.case)
                )
                judged[verdicts.check.name] = (
                    None if given is None else Verdict(given.verdict, given.reason)
                )
            found.append(
                Row(
                    done.batch.number,
                    item_id,
                    entry.part,
                    entry.case,
                    entry.statement,
                    entry.kind,
                    entry.stratum,
                    entry.disputes,
                    entry.expected,
                    label.label,
                    label.note,
                    judged,
                )
            )
    return found


def share(successes: int, total: int) -> dict[str, float | int] | None:
    """A share as a run saves it: the count beside it, and its 95% Wilson interval."""
    if total == 0:
        return None
    estimate = share_with_interval(successes, total)
    return {
        "successes": successes,
        "count": total,
        "mean": estimate.mean,
        "low": estimate.low,
        "high": estimate.high,
    }


def _kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Kappa for a yes-or-no decision, or ``None`` where either side never changes its mind."""
    if not pairs or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return None
    return round(decision_kappa(pairs), 4)


def _matrix(
    pairs: Sequence[tuple[str, str]], verdicts: Sequence[str], rows_are: str
) -> dict[str, Any]:
    """How often each pair of labels was given, with which side the rows are."""
    counts = {first: dict.fromkeys(verdicts, 0) for first in verdicts}
    for first, second in pairs:
        counts[first][second] += 1
    return {"rows": rows_are, "columns": "the owner's label", "counts": counts}


def _verdicts(part: str) -> list[str]:
    return [choice for choice in CHOICES[part] if choice != CANNOT_TELL]


def against_the_set(part: str, members: Sequence[Row]) -> dict[str, Any]:
    """The person's labels beside the set's, leaving out "cannot tell" and the set's ``unclear``."""
    faults = FAULTS[part]
    compared = [row for row in members if row.owner != CANNOT_TELL and row.expected != UNCLEAR]
    by_class: dict[str, Any] = {}
    for label in sorted({row.expected for row in members} - {UNCLEAR}):
        with_label = [row for row in compared if row.expected == label]
        by_class[label] = {
            "same": share(sum(row.owner == label for row in with_label), len(with_label)),
            "cannot_tell": sum(
                1 for row in members if row.expected == label and row.owner == CANNOT_TELL
            ),
        }
    return {
        "compared": len(compared),
        "unclear_in_the_set": sum(row.expected == UNCLEAR for row in members),
        "same_verdict": share(sum(row.owner == row.expected for row in compared), len(compared)),
        "same_decision": share(
            sum((row.owner in faults) == (row.expected in faults) for row in compared),
            len(compared),
        ),
        "kappa": _kappa([(row.expected in faults, row.owner in faults) for row in compared]),
        "by_class": by_class,
        "matrix": _matrix(
            [(row.expected, row.owner) for row in compared], _verdicts(part), "the set's label"
        ),
    }


def against_a_judge(part: str, members: Sequence[Row], saved: SavedVerdicts) -> dict[str, Any]:
    """The person's labels beside one judge's saved verdicts, leaving out "cannot tell"."""
    faults = FAULTS[part]
    name = saved.check.name
    labelled = [row for row in members if row.owner != CANNOT_TELL]
    pairs = [
        (row.owner, verdict.verdict)
        for row in labelled
        if (verdict := row.judged.get(name)) is not None
    ]
    owner_faults = [judged for owner, judged in pairs if owner in faults]
    judge_faults = [owner for owner, judged in pairs if judged in faults]
    return {
        "judge": name,
        "run": saved.check.run_dir.as_posix(),
        "config": _config(saved.check.run_dir),
        "compared": len(pairs),
        "no_saved_verdict": sum(1 for row in labelled if row.judged.get(name) is None),
        "same_verdict": share(sum(owner == judged for owner, judged in pairs), len(pairs)),
        "same_decision": share(
            sum((owner in faults) == (judged in faults) for owner, judged in pairs), len(pairs)
        ),
        "kappa": _kappa([(judged in faults, owner in faults) for owner, judged in pairs]),
        "owner_faults_the_judge_faulted": share(
            sum(judged in faults for judged in owner_faults), len(owner_faults)
        ),
        "judge_faults_the_owner_agrees_with": share(
            sum(owner in faults for owner in judge_faults), len(judge_faults)
        ),
        "matrix": _matrix(
            [(judged, owner) for owner, judged in pairs], _verdicts(part), "the judge's verdict"
        ),
    }


def side(row: Row, verdict: str) -> str:
    """With whom the person sides on an item where a judge and the set differ."""
    if row.owner == CANNOT_TELL:
        return CANNOT_TELL
    if row.owner == row.expected:
        return "the set"
    if row.owner == verdict:
        return "the judge"
    return "neither"


def disputed(part: str, members: Sequence[Row], saved: Sequence[SavedVerdicts]) -> dict[str, Any]:
    """With whom the person sides where a judge and the set differ; what they call ``unclear``."""
    judges: list[dict[str, Any]] = []
    for verdicts in saved:
        if verdicts.check.part != part:
            continue
        name = verdicts.check.name
        differing = [
            (row, given.verdict)
            for row in members
            if row.expected != UNCLEAR
            and (given := row.judged.get(name)) is not None
            and given.verdict != row.expected
        ]
        sides = [side(row, verdict) for row, verdict in differing]
        judges.append(
            {
                "judge": name,
                "items_where_it_differs_from_the_set": [row.item for row, _ in differing],
                "the_owner_sides_with": {name: sides.count(name) for name in SIDES},
            }
        )
    unclear = [row for row in members if row.expected == UNCLEAR]
    return {
        "judges": judges,
        "unclear_in_the_set": {
            "items": [row.item for row in unclear],
            "the_owner_called_them": {
                choice: sum(row.owner == choice for row in unclear) for choice in CHOICES[part]
            },
        },
        "readers_differed": [row.item for row in members if "readers" in row.disputes],
    }


def _config(run_dir: Path) -> object:
    """The settings a saved check was made with, as its own files record them."""
    state = run_dir / "run.json"
    if not state.is_file():
        return None
    try:
        loaded = json.loads(state.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return loaded.get("config") if isinstance(loaded, dict) else None


def report(
    labelled: Sequence[LabelledBatch],
    waiting: Sequence[Batch],
    saved: Sequence[SavedVerdicts],
    fingerprint: Callable[[str, str], str],
) -> tuple[dict[str, Any], list[Row]]:
    """The summary of the comparison, and the rows it was made from."""
    found = rows(labelled, saved, fingerprint)
    parts: dict[str, Any] = {}
    for part in PARTS:
        members = [row for row in found if row.part == part]
        if not members:
            continue
        parts[part] = {
            "title": f"Part {PART_LETTERS[part]}: {PART_TITLES[part]}",
            "labelled": len(members),
            "cannot_tell": sum(row.owner == CANNOT_TELL for row in members),
            "owner_labels": {
                choice: sum(row.owner == choice for row in members) for choice in CHOICES[part]
            },
            "against_the_set": against_the_set(part, members),
            "against_the_judges": [
                against_a_judge(part, members, verdicts)
                for verdicts in saved
                if verdicts.check.part == part
            ],
            "disputed": disputed(part, members, saved),
        }
    summary = {
        "description": (
            "One person's labels on a sample drawn from the two labelled sets, against the "
            "labels of the sets and the saved verdicts of the judges. No request to any model."
        ),
        "labeller": labelled[0].labels.labeller if labelled else None,
        "batches": [
            {
                "batch": done.batch.number,
                "items": len(done.batch.items),
                "saved_at": done.labels.saved_at,
            }
            for done in labelled
        ],
        "waiting_for_labels": [batch.number for batch in waiting],
        "parts": parts,
        "differences": sum(1 for row in found if differs(row)),
        "cannot_tell": sum(1 for row in found if row.owner == CANNOT_TELL),
        "disputed_items": sum(1 for row in found if row.disputes),
    }
    return summary, found


def differs(row: Row) -> list[str]:
    """What the person's label differs from: the set's label, and each judge's verdict."""
    if row.owner == CANNOT_TELL:
        return []
    found = ["the set's label"] if row.expected != UNCLEAR and row.owner != row.expected else []
    found.extend(
        name
        for name, verdict in row.judged.items()
        if verdict is not None and verdict.verdict != row.owner
    )
    return found


def in_full(row: Row, batch: Batch) -> dict[str, Any]:
    """An item as a later look needs it: what was shown, the person's label, and every verdict."""
    return {
        "batch": row.batch,
        "item": row.item,
        "differs_from": differs(row),
        "owner": row.owner,
        "note": row.note,
        "set_label": row.expected,
        "case": row.case,
        "statement": row.statement,
        "kind": row.kind,
        "group": row.stratum,
        "disputes": list(row.disputes),
        "judges": {
            name: None
            if verdict is None
            else {"verdict": verdict.verdict, "reason": verdict.reason}
            for name, verdict in row.judged.items()
        },
        "shown": {key: value for key, value in batch.items[row.item].items() if key != "sha256"},
    }


def write_report(
    run_dir: Path,
    summary: Mapping[str, Any],
    found: Sequence[Row],
    labelled: Sequence[LabelledBatch],
    saved: Sequence[SavedVerdicts],
) -> None:
    """Save the run: its summary, how it was made, and the items a later look needs in full.

    ``differences.jsonl`` holds every item on which the person differs from the set or a
    judge, ``cannot-tell.jsonl`` every "cannot tell", and ``disputed.jsonl`` every disputed
    item, each with what was shown and every verdict.
    """
    batches = {done.batch.number: done.batch for done in labelled}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def digest(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    made = {
        "made_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "handbook_rag_version": __version__,
        "batches": [
            {
                "batch": done.batch.number,
                "items_sha256": done.batch.items_sha256,
                "joins_sha256": done.batch.joins_sha256,
                "labels_sha256": digest(done.batch.labels_path),
            }
            for done in labelled
        ],
        "checks": [
            {
                "judge": verdicts.check.name,
                "run": verdicts.check.run_dir.as_posix(),
                "judgements_sha256": digest(verdicts.check.run_dir / "judgements.jsonl"),
            }
            for verdicts in saved
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(made, indent=2) + "\n", encoding="utf-8")
    for name, chosen in (
        ("differences.jsonl", [row for row in found if differs(row)]),
        ("cannot-tell.jsonl", [row for row in found if row.owner == CANNOT_TELL]),
        ("disputed.jsonl", [row for row in found if row.disputes]),
    ):
        (run_dir / name).write_text(
            "".join(
                json.dumps(in_full(row, batches[row.batch]), ensure_ascii=False) + "\n"
                for row in chosen
            ),
            encoding="utf-8",
        )
