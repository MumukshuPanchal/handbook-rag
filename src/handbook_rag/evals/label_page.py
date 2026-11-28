"""The page on which one person labels a batch, and the five steps that go beside it.

The page is one file that works from disk: no server, no network, its style and its script
inside it. A content security policy in the page itself forbids it to load or send
anything, and allows only its own style and script, named by their hashes. Every text from
the labelled sets and the handbook goes into the page as JSON data that is never run, and
the script puts it on screen as text, never as markup: an answer or a passage that holds
``<script>`` shows those characters.

It shows one item at a time. One item on the screen keeps the eye on the passage the
statement cites, and keeps two answers from being read side by side. A strip of the item
ids shows what is done and jumps to any item.

The page's markup, style and script are in ``label_files/`` next to this module, and are
put together into one file for each batch.
"""

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

from handbook_rag.evals.human_labels import (
    FIRST_PASS,
    LABELLER,
    LABELS_FORMAT,
    LABELS_VERSION,
    NOTE_LIMIT,
    file_name_prefix,
)
from handbook_rag.evals.human_sample import (
    BATCH_FILE,
    CHOICES,
    CORRECT_PART,
    ITEMS_FILE,
    SUPPORT_PART,
    DrawnBatch,
    SampleError,
    Sets,
    batch_dir,
    items_text,
    key_entries,
    leaks,
)

_FILES = files("handbook_rag.evals") / "label_files"
TEMPLATE = (_FILES / "label.html").read_text(encoding="utf-8")
STYLE = (_FILES / "label.css").read_text(encoding="utf-8")
SCRIPT = (_FILES / "label.js").read_text(encoding="utf-8")

RUBRIC_PATHS = {
    SUPPORT_PART: Path("evals/judge/rubric.md"),
    CORRECT_PART: Path("evals/judge/correct/rubric.md"),
}
"""The rubrics the model readers labelled by. The page shows each in full under its short form."""

CHOICE_WORDS = {
    "supported": "Supported",
    "unsupported": "Unsupported",
    "no_claim": "No claim",
    "correct": "Correct",
    "partial": "Partial",
    "incorrect": "Incorrect",
    "cannot_tell": "Cannot tell",
}

PAGE_DATA_KEYS = frozenset(
    {
        "batch",
        "items_sha256",
        "format",
        "version",
        "labeller",
        "pass",
        "note_limit",
        "file_prefix",
        "choices",
        "names",
        "rubrics",
        "items",
    }
)
"""Every key of the page's data. Anything else in it is refused before a batch is written."""

TELLING_WORDS = (
    "half",
    "fault",
    "faults",
    "planted",
    "copies",
    "disputed",
    "stratum",
    "group",
    "groups",
    "kind",
    "kinds",
    "split",
)
"""Words about how a batch is made up. The page's own text and the steps hold none of them."""

READ_NOTHING_FIRST = (
    "Read nothing about this sample before your labels are in place: not its card, its "
    "decision record, its sections in the README, the guide or the changelog, nor the "
    "commit or pull request that added it."
)
"""The one line the page and the steps both give about what to leave unread until afterwards."""


def read_rubrics(root: Path = Path()) -> dict[str, str]:
    """The two rubrics as the readers had them.

    The rubric of correctness holds a note for people above a line, and below it the
    words the readers and the judge were given; only those are shown.
    """
    rubrics: dict[str, str] = {}
    for part, path in RUBRIC_PATHS.items():
        if not (root / path).is_file():
            raise SampleError(f"The rubric {path} is missing. Run the command from the project.")
        text = (root / path).read_text(encoding="utf-8")
        if part == CORRECT_PART and "\n---\n" in text:
            text = text.split("\n---\n", 1)[1]
        rubrics[part] = text.strip()
    return rubrics


def page_data(drawn: DrawnBatch, rubrics: Mapping[str, str]) -> dict[str, Any]:
    """Everything the page's script reads: the items, the choices and what a file must hold."""
    return {
        "batch": drawn.number,
        "items_sha256": drawn.meta["items_sha256"],
        "format": LABELS_FORMAT,
        "version": LABELS_VERSION,
        "labeller": LABELLER,
        "pass": FIRST_PASS,
        "note_limit": NOTE_LIMIT,
        "file_prefix": file_name_prefix(drawn.number),
        "choices": {part: list(choices) for part, choices in CHOICES.items()},
        "names": CHOICE_WORDS,
        "rubrics": dict(rubrics),
        "items": drawn.items,
    }


def inert_json(data: object) -> str:
    """JSON that can stand inside a script element and never end it or start markup.

    Every character that could close the element or open a tag or an entity is written
    as an escape, which JSON reads back as the same character. So is the slash, so that
    no text of the sets can make ``</script>`` or ``//`` appear in the file.
    """
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for character, written in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("/", "\\/"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(character, written)
    return text


def _csp_hash(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode()


def content_security_policy() -> str:
    """The policy the page sets for itself: nothing loaded or sent, and no code but its own."""
    return (
        "default-src 'none'; "
        f"script-src '{_csp_hash(SCRIPT)}'; "
        f"style-src '{_csp_hash(STYLE)}'; "
        "base-uri 'none'; form-action 'none'"
    )


def part_counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many items each part holds."""
    return {part: sum(1 for item in items if item["part"] == part) for part in CHOICES}


def render_label_page(drawn: DrawnBatch, rubrics: Mapping[str, str]) -> str:
    """The page for one batch, as one self-contained HTML file."""
    parts = part_counts(drawn.items)
    statements, answers = parts[SUPPORT_PART], parts[CORRECT_PART]
    counted = " and ".join(
        text
        for text in (
            f"{statements} statement{'s' * (statements != 1)} (part A)" if statements else "",
            f"{answers} answer{'s' * (answers != 1)} (part B)" if answers else "",
        )
        if text
    )
    return (
        TEMPLATE.replace("{{policy}}", escape(content_security_policy(), quote=True))
        .replace("{{style}}", STYLE)
        .replace("{{script}}", SCRIPT)
        .replace("{{batch}}", str(drawn.number))
        .replace("{{total}}", str(len(drawn.items)))
        .replace("{{counted}}", escape(counted))
        .replace("{{first}}", escape(READ_NOTHING_FIRST))
        .replace("{{data}}", inert_json(page_data(drawn, rubrics)))
    )


_DATA = re.compile(r'<script type="application/json" id="batch-data">(.*?)</script>', re.S)


def visible_text(markup: str) -> str:
    """The words of markup a reader sees: without its script, style, tags and comments."""
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", markup, flags=re.S | re.I)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return re.sub(r"<[^>]+>", " ", text)


def telling_words_in(text: str) -> list[str]:
    """The words of :data:`TELLING_WORDS` that a text holds."""
    return [word for word in TELLING_WORDS if re.search(rf"\b{word}\b", text, re.IGNORECASE)]


def page_leaks(page: str, drawn: DrawnBatch, rubrics: Mapping[str, str]) -> list[str]:
    """What, in a page apart from its items, could tell how its batch is made up.

    The code checks that the page's data holds the keys of :data:`PAGE_DATA_KEYS` and no
    other, with the choices, the words for them and the rubrics as they are and the items
    as drawn; that the page holds no comment; and that neither its text nor its script holds
    a word of :data:`TELLING_WORDS`.
    """
    found: list[str] = []
    match = _DATA.search(page)
    if match is None:
        return ["the page holds no data"]
    data = json.loads(match.group(1))
    if set(data) != PAGE_DATA_KEYS:
        found.append(f"the page's data holds other keys: {sorted(set(data) ^ PAGE_DATA_KEYS)}")
    expected = page_data(drawn, rubrics)
    for name in ("choices", "names", "rubrics", "items", "file_prefix", "items_sha256"):
        if data.get(name) != expected[name]:
            found.append(f"the page's {name} are not what they should be")
    outside = page[: match.start(1)] + page[match.end(1) :]
    if "<!--" in outside:
        found.append("the page holds a comment")
    for word in telling_words_in(visible_text(outside)) + telling_words_in(SCRIPT):
        found.append(f"the page says {word!r}")
    return found


def how_to_label(drawn: DrawnBatch) -> str:
    """The five steps for the person who labels, in plain words, for one batch."""
    parts = part_counts(drawn.items)
    number = drawn.number
    total = len(drawn.items)
    prefix = file_name_prefix(number)
    return f"""# How to label batch {number}

{total} judgements: {parts[SUPPORT_PART]} statements to hold against the passages they cite \
(part A), and {parts[CORRECT_PART]} answers to hold against a reference answer (part B). You \
label from what the page shows and the rubric at its top, as the model readers did. Nobody \
expects you to know GitLab.

**The one thing not to do: do not look for how this batch was made before your labels are \
in place.** {READ_NOTHING_FIRST} The labels the sets give these statements and answers are \
in the project's files, so do not search them for an item's words either.

1. **Open the page.** Open `label.html` in this folder with your browser: double-click it, or \
drag it onto a browser window. It needs no internet and no server, and it sends nothing \
anywhere.
2. **Read the rubric at the top, then label one item at a time.** Press 1, 2, 3 or 4, or click \
a choice. N or → goes to the next item, P or ← back. Add a short note where you hesitate. \
"Cannot tell" is a fair answer; it is counted apart. Decide each item on its own: do not count \
how many of each label you have given.
3. **Stop and come back when you like.** The page keeps your labels in this browser as you go, \
and says so; if it cannot, it says that too. Come back in the same browser. If it finds \
labels from before, it says when they were made and asks whether to keep them.
4. **Save the file.** When every item has a label, press "Save labels file". Each save makes a \
new file, `{prefix}<date>-<time>.json`, usually in your Downloads folder, and the page shows \
its name and time. To change a label, change it on the page and save again: the newest file \
holds your labels. The page keeps its copy until you press "Clear this browser's copy".
5. **Put the newest file in place.** In a terminal, in the project folder, run, with the name \
the page showed at your last save:

   ```bash
   uv run handbook-rag human-labels ~/Downloads/{prefix}<date>-<time>.json
   ```

   It checks the file, says when it was saved and how many labels and notes it holds, and \
puts it here as `labels.json`. It refuses an older file when a newer one lies beside it. \
If it refuses, its message says what to do. Then \
`uv run handbook-rag eval-human --variant batch-{number}` gives the report.
"""


def write_batch(root: Path, drawn: DrawnBatch, rubrics: Mapping[str, str], sets: Sets) -> Path:
    """Write a drawn batch to its folder: items, how it was drawn, the page and the steps.

    The key is not written: it waits for the labels. Before anything is written, the items,
    the page and the steps are checked (:func:`handbook_rag.evals.human_sample.leaks` and
    :func:`page_leaks`), and a batch that fails is not written.
    """
    directory = batch_dir(root, drawn.number)
    if directory.exists():
        raise SampleError(f"{directory} exists already. A batch is never drawn twice.")
    items = items_text(drawn.items)
    page = render_label_page(drawn, rubrics)
    steps = how_to_label(drawn)
    found = leaks(drawn.items, key_entries(drawn.key), sets, shown=items + page + steps)
    found += page_leaks(page, drawn, rubrics)
    found += [f"the steps say {word!r}" for word in telling_words_in(steps)]
    if found:
        raise SampleError(
            "The batch would tell a label, so it was not written: " + "; ".join(found)
        )
    directory.mkdir(parents=True)
    (directory / ITEMS_FILE).write_text(items, encoding="utf-8")
    (directory / BATCH_FILE).write_text(
        json.dumps(drawn.meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "label.html").write_text(page, encoding="utf-8")
    (directory / "HOW-TO-LABEL.md").write_text(steps, encoding="utf-8")
    return directory
