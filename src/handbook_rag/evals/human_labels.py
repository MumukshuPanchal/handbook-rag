"""The labels file a person's page saves, checked before it is put in place.

The page saves one file each time the person presses its button, named by the time of the
save, and the file says when it was saved and how many labels and notes it holds. A file
that does not fit its batch is refused with what to do. And since a browser never writes
over a file it saved before, the newest file is the one to give: when a newer file of the
same batch lies beside the one given, it is refused, so that a label changed after an
earlier save is not lost.
"""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from handbook_rag.evals.human_sample import CHOICES, Batch, read_batches

LABELLER = "the project's owner"
"""Who labels. A labels file records this and no name or address."""
LABELS_FORMAT = "handbook-rag labels of a person"
LABELS_VERSION = 2
"""Version 2 says how many labels and notes it holds. No file of version 1 was ever put in
place: it was saved only by the page of a draw that was replaced before any label existed."""
FIRST_PASS = 1
NOTE_LIMIT = 1000
"""The longest note a label may carry, in characters. A note is meant to be short."""

_TOP_KEYS = (
    "format",
    "version",
    "batch",
    "labeller",
    "pass",
    "items_sha256",
    "saved_at",
    "counted",
    "labels",
)
_LABEL_KEYS = ("item", "item_sha256", "label", "note")
_SAVE_AGAIN = "Open the batch's page again, and save the labels file from it."


class LabelsError(ValueError):
    """A labels file does not fit its batch. The message says what to do about it."""


@dataclass(frozen=True, slots=True)
class Label:
    """The person's answer for one item."""

    item: str
    label: str
    note: str


@dataclass(frozen=True, slots=True)
class Labels:
    """A labels file that fits its batch."""

    batch: int
    labeller: str
    pass_number: int
    saved_at: str
    labels: dict[str, Label]
    """By item id, in the order of the batch."""

    @property
    def notes(self) -> int:
        return sum(1 for label in self.labels.values() if label.note.strip())

    @property
    def saved(self) -> datetime:
        return saved_time(self.saved_at)


def file_name_prefix(batch: int) -> str:
    """How the name of every labels file of a batch starts; the page adds the time of the save."""
    return f"labels-batch-{batch}-"


def saved_time(text: str) -> datetime:
    """The time a file says it was saved, as a time a later one can be compared with."""
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def labels_batch_number(data: object) -> int:
    """Which batch a labels file says it is for, so that the batch can be read to check it."""
    if not isinstance(data, dict):
        raise LabelsError(
            "This is not a labels file: it holds no JSON object. Give the file that the page's "
            "button 'Save labels file' saved, labels-batch-<number>-<time>.json, usually in "
            "your Downloads folder."
        )
    number = data.get("batch")
    if type(number) is not int or number < 1:
        raise LabelsError(
            f"The file does not say which batch it labels ('batch' is {number!r}). {_SAVE_AGAIN}"
        )
    return number


def _whole(value: object, wanted: int) -> bool:
    """Whether a value is the whole number wanted. JSON's ``true`` is not taken for 1."""
    return type(value) is int and value == wanted


def read_labels(data: object, batch: Batch) -> Labels:
    """Check a labels file against its batch, and return the labels if it fits.

    Refused, each with what to do: a file that is not one the page saves, or of another
    version of it; a field it does not write, or one missing; a file of another batch or of
    another drawing of it; a labeller recorded as anything but "the project's owner"; an
    item missing, given twice, unknown or of other words than were drawn; a label that is
    not one of the item's choices; a note that is not text or is too long; and counts of
    labels and notes that are not those of the file. A note is free text, and the code does
    not read what it says. Nothing is guessed or mended.
    """
    number = labels_batch_number(data)
    assert isinstance(data, dict)
    if data.get("format") != LABELS_FORMAT or not _whole(data.get("version"), LABELS_VERSION):
        raise LabelsError(
            "This file was not saved by this version of the project's labelling page ('format' "
            f"and 'version' should be {LABELS_FORMAT!r} and {LABELS_VERSION}). {_SAVE_AGAIN}"
        )
    unknown = sorted(set(data) - set(_TOP_KEYS))
    missing = [name for name in _TOP_KEYS if name not in data]
    if unknown or missing:
        raise LabelsError(
            "The file holds other fields than the page writes"
            + (f"; not expected: {', '.join(unknown)}" if unknown else "")
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + f". Do not edit the file by hand. {_SAVE_AGAIN}"
        )
    if number != batch.number:
        raise LabelsError(f"The file labels batch {number}, not batch {batch.number}.")
    if data["labeller"] != LABELLER:
        raise LabelsError(
            f"The labeller must be recorded as {LABELLER!r}, with no name or address. This "
            f"version takes the labels of the project's owner only. {_SAVE_AGAIN}"
        )
    if not _whole(data["pass"], FIRST_PASS):
        raise LabelsError(
            f"The file says it is pass {data['pass']!r}. This version takes a first pass only. "
            f"{_SAVE_AGAIN}"
        )
    if data["items_sha256"] != batch.items_sha256:
        raise LabelsError(
            f"The file was saved from a page of another drawing of batch {number}: its items "
            "are not the ones drawn now. Label the batch again on its current page, "
            f"{batch.directory / 'label.html'}."
        )
    saved_at = data["saved_at"]
    if not isinstance(saved_at, str) or len(saved_at) > 40 or not _is_a_time(saved_at):
        raise LabelsError(f"'saved_at' is not a time: {saved_at!r}. {_SAVE_AGAIN}")
    entries = data["labels"]
    if not isinstance(entries, list):
        raise LabelsError(f"'labels' is not a list of labels. {_SAVE_AGAIN}")
    labels: dict[str, Label] = {}
    for place, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or sorted(entry) != sorted(_LABEL_KEYS):
            raise LabelsError(
                f"Label {place} in the file is not as the page writes it: it should hold "
                f"exactly {', '.join(_LABEL_KEYS)}. {_SAVE_AGAIN}"
            )
        item_id = entry["item"]
        if not isinstance(item_id, str) or item_id not in batch.items:
            raise LabelsError(
                f"Label {place} is for item {item_id!r}, which batch {number} does not hold. "
                f"{_SAVE_AGAIN}"
            )
        if item_id in labels:
            raise LabelsError(
                f"Item {item_id} is labelled twice in the file. Give each item one label. "
                f"{_SAVE_AGAIN}"
            )
        if entry["item_sha256"] != batch.items[item_id]["sha256"]:
            raise LabelsError(
                f"Item {item_id} in the file is not the item that was drawn: it was labelled on "
                f"other words. {_SAVE_AGAIN}"
            )
        allowed = CHOICES[batch.items[item_id]["part"]]
        if not isinstance(entry["label"], str) or entry["label"] not in allowed:
            raise LabelsError(
                f"Item {item_id} has the label {entry['label']!r}, which is not one of its "
                f"choices: {', '.join(allowed)}. Choose one on the page and save again."
            )
        note = entry["note"]
        if not isinstance(note, str):
            raise LabelsError(f"The note of item {item_id} is not text. {_SAVE_AGAIN}")
        if len(note) > NOTE_LIMIT:
            raise LabelsError(
                f"The note of item {item_id} is {len(note)} characters long; at most "
                f"{NOTE_LIMIT} are kept. Shorten it on the page and save again."
            )
        labels[item_id] = Label(item_id, entry["label"], note)
    unlabelled = [item_id for item_id in batch.items if item_id not in labels]
    if unlabelled:
        raise LabelsError(
            f"{len(unlabelled)} of the {len(batch.items)} items of batch {number} have no "
            f"label: {', '.join(unlabelled)}. Label them on the page, then save again."
        )
    result = Labels(
        number,
        LABELLER,
        FIRST_PASS,
        saved_at,
        {item_id: labels[item_id] for item_id in batch.items},
    )
    counted = data["counted"]
    if counted != {"labels": len(result.labels), "notes": result.notes}:
        raise LabelsError(
            f"The file says it holds {counted!r}, and it holds {len(result.labels)} labels and "
            f"{result.notes} notes: it was edited. {_SAVE_AGAIN}"
        )
    return result


def _is_a_time(text: str) -> bool:
    try:
        saved_time(text)
    except ValueError:
        return False
    return True


def load_labels_file(path: Path) -> object:
    """The content of a labels file, or a message a beginner can act on."""
    if not path.is_file():
        raise LabelsError(
            f"There is no file at {path}. The page saves labels-batch-<number>-<time>.json, "
            "usually in your Downloads folder; give its path."
        )
    try:
        # An editor may have added a byte-order mark; it is no part of the content.
        loaded: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LabelsError(
            f"{path} is not a labels file: it cannot be read as JSON ({error}). Give the file "
            "the page saved, unchanged."
        ) from error
    return loaded


def newer_files(path: Path, labels: Labels) -> list[tuple[Path, str]]:
    """Labels files of the same batch beside ``path`` that say they were saved later.

    Every file whose name starts as the page names a batch's files is read, with the name
    a browser gives a second file of the same name too. One that cannot be read as a labels
    file of that batch is passed over: the file given is checked in full, and these only
    for their time.
    """
    pattern = re.compile(rf"labels-batch-{labels.batch}(-.*|\s*\(\d+\))?\.json")
    found: list[tuple[Path, str]] = []
    for other in sorted(path.parent.iterdir()) if path.parent.is_dir() else []:
        if other == path or not other.is_file() or not pattern.fullmatch(other.name):
            continue
        try:
            data = json.loads(other.read_text(encoding="utf-8-sig"))
            when = (
                data["saved_at"]
                if isinstance(data, dict) and data.get("batch") == labels.batch
                else None
            )
            if isinstance(when, str) and saved_time(when) > labels.saved:
                found.append((other, when))
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            continue
    return found


def labels_text(labels: Labels, batch: Batch) -> str:
    """The labels as they are put in place: the file the page writes, laid out to be read."""
    data = {
        "format": LABELS_FORMAT,
        "version": LABELS_VERSION,
        "batch": labels.batch,
        "labeller": labels.labeller,
        "pass": labels.pass_number,
        "items_sha256": batch.items_sha256,
        "saved_at": labels.saved_at,
        "counted": {"labels": len(labels.labels), "notes": labels.notes},
        "labels": [
            {
                "item": label.item,
                "item_sha256": batch.items[label.item]["sha256"],
                "label": label.label,
                "note": label.note,
            }
            for label in labels.labels.values()
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def labelled_batches(root: Path) -> tuple[list[tuple[Batch, Labels]], list[Batch]]:
    """The batches whose labels are in place, each with its labels checked again, and the rest.

    A labels file in place is checked as when it was put there, so a file edited since is
    refused, not counted.
    """
    done: list[tuple[Batch, Labels]] = []
    waiting: list[Batch] = []
    for batch in read_batches(root):
        if not batch.labels_path.is_file():
            waiting.append(batch)
            continue
        try:
            labels = read_labels(load_labels_file(batch.labels_path), batch)
        except LabelsError as error:
            raise LabelsError(f"{batch.labels_path}: {error}") from error
        done.append((batch, labels))
    return done, waiting
