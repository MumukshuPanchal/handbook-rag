"""A sample of the labelled answers for one person to label blind, drawn in batches.

Every figure of agreement in this project compares a model with labels that model sessions
made and checked. This module draws a small sample of those labelled answers for one
person to label without seeing anyone else's label. The labels file that comes back is
checked by :mod:`handbook_rag.evals.human_labels`, the report that compares the labels
with the sets and the judges is :mod:`handbook_rag.evals.human_report`, and the page the
person labels on is :mod:`handbook_rag.evals.label_page`.

**What a batch's folder holds before its labels are in place.** ``items.jsonl``, what the
person is shown; ``batch.json``, how the batch was drawn (the seed, the plan, the files it
was drawn from, with their hashes) and the hashes of the items and of the key; the page and
its steps. The key, which says which case and statement each item is and the label the set
gives it, is not written then. It can be drawn again from the seed, the sets and the
judges' saved checks, and ``eval-human`` draws it again, checks it against its hash and
writes it beside the labels.

**Where a person's label tells most.** A batch first takes the items on which the
instruments disagree: a judge's saved verdict that differs from the set's label, two
readers of the set who differed, or a statement the set calls ``unclear``. The rest is
drawn in groups by the set's label.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from handbook_rag.evals.agreement import REAL
from handbook_rag.evals.correctness import CORRECT, NOT_CORRECT, CorrectnessCase
from handbook_rag.evals.judge import Judgement, StatementVerdict, SupportCase
from handbook_rag.evals.judging import UNCLEAR, LabelledCase, load_judgements
from handbook_rag.evals.support import SUPPORT

SUPPORT_PART = "support"
CORRECT_PART = "correct"
PARTS = (SUPPORT_PART, CORRECT_PART)
"""Part A of a batch is statements against their passages, part B answers against their
reference answers: one part for each judge."""
PART_LETTERS = {SUPPORT_PART: "A", CORRECT_PART: "B"}

CANNOT_TELL = "cannot_tell"
"""The person's answer when what is shown does not let them decide. It is counted apart,
never as agreement or as disagreement."""
CHOICES: Mapping[str, tuple[str, ...]] = {
    SUPPORT_PART: ("supported", "unsupported", "no_claim", CANNOT_TELL),
    CORRECT_PART: ("correct", "partial", "incorrect", CANNOT_TELL),
}
"""What the person may answer in each part: the judge's verdicts, and "cannot tell"."""
FAULTS: Mapping[str, frozenset[str]] = {
    SUPPORT_PART: frozenset({"unsupported"}),
    CORRECT_PART: NOT_CORRECT,
}
"""The answers that count against an answer, as the judges' scores count them."""

BATCHES_DIR = Path("evals/judge/human")
BATCH_FILE = "batch.json"
ITEMS_FILE = "items.jsonl"
KEY_FILE = "key.json"
LABELS_FILE = "labels.json"
SET_ASIDE_FILE = "set-aside.json"
"""Beside the batches: statements and answers no batch may draw, by hash, with the reason."""
DEFAULT_SEED = 20261128
SET_PATHS = {SUPPORT_PART: SUPPORT.labelled_path, CORRECT_PART: CORRECT.labelled_path}

_BATCH_DIR = re.compile(r"batch-([1-9]\d*)")

DISPUTED = "disputed"
READERS = "readers"
"""A dispute: the set's first two readers gave the statement or answer different labels."""


class SampleError(ValueError):
    """A batch cannot be drawn as asked."""


class BatchError(ValueError):
    """A drawn batch on disk is not as the drawing wrote it, or cannot be drawn again."""


# The judges' saved checks, and the disputes they and the sets hold.


@dataclass(frozen=True, slots=True)
class SavedCheck:
    """A judge's saved check on a labelled set: its verdicts are those the person's labels meet."""

    part: str
    short: str
    """Its name in files: also the name of the dispute in which it differs from the set."""
    name: str
    run_dir: Path
    verdicts: tuple[str, ...]


SAVED_CHECKS: tuple[SavedCheck, ...] = (
    SavedCheck(
        SUPPORT_PART,
        "support_v1",
        "the judge of support, version 1",
        SUPPORT.checks_dir / "baseline",
        SUPPORT.verdicts,
    ),
    SavedCheck(
        SUPPORT_PART,
        "support_v2",
        "the judge of support, version 2",
        SUPPORT.checks_dir / "v2",
        SUPPORT.verdicts,
    ),
    SavedCheck(
        CORRECT_PART,
        "correct",
        "the judge of correctness",
        CORRECT.checks_dir / "baseline",
        CORRECT.verdicts,
    ),
)
"""Version 1 of the judge of support gives every published score; version 2 is kept beside it
for one fault (ADR 0023). Both have saved checks on the first set, so both are used."""

DISPUTE_ORDER = (*(check.short for check in SAVED_CHECKS), READERS, UNCLEAR)
"""The sorts of dispute, in the order in which an item that has several is named by one."""


@dataclass(frozen=True, slots=True)
class SavedVerdicts:
    """The verdicts of one saved check, by the fingerprint of what the judge read."""

    check: SavedCheck
    by_fingerprint: Mapping[str, Judgement]

    def on(self, case_id: str, statement: int, fingerprint: str | None) -> StatementVerdict | None:
        """The verdict on one statement, or ``None`` if the check has none on these words.

        With a fingerprint, only a verdict on exactly those words counts. Without one,
        which only a test that has no handbook asks for, the verdict on the case counts.
        """
        if fingerprint is not None:
            judgement = self.by_fingerprint.get(fingerprint)
        else:
            judgement = next(
                (j for j in self.by_fingerprint.values() if j.case_id == case_id), None
            )
        if judgement is None or judgement.case_id != case_id:
            return None
        return next((v for v in judgement.verdicts if v.number == statement), None)


def load_saved_checks(root: Path = Path()) -> list[SavedVerdicts]:
    """The verdicts of every saved check. A check that is not on disk has no verdict."""
    found: list[SavedVerdicts] = []
    for check in SAVED_CHECKS:
        rows = root / check.run_dir / "judgements.jsonl"
        found.append(
            SavedVerdicts(check, load_judgements(rows, check.verdicts) if rows.is_file() else {})
        )
    return found


def reader_splits(path: Path) -> set[tuple[str, int]]:
    """The statements and answers on which the set's first two readers gave different labels."""
    splits: set[tuple[str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        units = row.get("statements") or [{"number": 1, "readers": row.get("readers", {})}]
        for unit in units:
            readers = unit.get("readers") or {}
            first, second = readers.get("first"), readers.get("second")
            if first is not None and second is not None and first != second:
                splits.add((str(row["case"]), int(unit["number"])))
    return splits


@dataclass(frozen=True, slots=True)
class Sets:
    """The two labelled sets, and what the drawing needs to know about them."""

    support: Sequence[LabelledCase[SupportCase]]
    correct: Sequence[LabelledCase[CorrectnessCase]]
    disputes: Mapping[tuple[str, str, int], tuple[str, ...]]
    """By part, case and statement: the sorts of dispute it is in, in :data:`DISPUTE_ORDER`."""
    set_aside: frozenset[str] = frozenset()
    """Hashes of statements and answers that no batch may draw (:func:`read_set_aside`)."""

    def case(self, part: str, case_id: str) -> LabelledCase[Any]:
        cases: Sequence[LabelledCase[Any]] = self.support if part == SUPPORT_PART else self.correct
        return next(labelled for labelled in cases if labelled.case.case_id == case_id)

    def case_ids(self) -> list[str]:
        return [c.case.case_id for c in self.support] + [c.case.case_id for c in self.correct]

    def real_questions(self) -> set[str]:
        found = {c.case.question for c in self.support if c.kind == REAL}
        return found | {c.case.question for c in self.correct if c.kind == REAL}

    def telling_words(self) -> set[str]:
        """Names of kinds, groups and sorts of dispute that no real answer of the sets uses.

        Such a word in what the person is shown says what an item is. A name that real
        answers use too, such as "name" or "scope", says nothing by itself.
        """
        real_text = " ".join(
            [c.case.question for c in self.support]
            + [c.case.question for c in self.correct]
            + [s.text for c in self.support if c.kind == REAL for s in c.case.statements]
            + [c.case.answer for c in self.correct if c.kind == REAL]
            + [c.case.reference for c in self.correct]
        ).lower()
        names = {c.kind for c in self.support} | {c.kind for c in self.correct}
        names |= {kind for c in self.support for kind in c.planted.values()}
        names |= {"supported", "unsupported", "no_claim", "unclear", "correct", "partial"}
        names |= {"incorrect", "not_correct", DISPUTED, "planted", "stratum", *DISPUTE_ORDER}
        return {
            name
            for name in names
            if not re.search(rf"\b{re.escape(name.replace('_', ' '))}\b", real_text)
            and not re.search(rf"\b{re.escape(name)}\b", real_text)
        }


def find_disputes(
    support: Sequence[LabelledCase[SupportCase]],
    correct: Sequence[LabelledCase[CorrectnessCase]],
    saved: Sequence[SavedVerdicts],
    splits: Mapping[str, Collection[tuple[str, int]]],
    *,
    by_fingerprint: bool = True,
) -> dict[tuple[str, str, int], tuple[str, ...]]:
    """Every statement and answer on which the instruments disagree, with how they disagree.

    A judge's saved verdict that differs from the set's label (a label ``unclear`` has no
    verdict to differ from); the set's first two readers who gave different labels; and a
    statement the set calls ``unclear``. ``splits`` gives the readers' splits by part.
    """
    found: dict[tuple[str, str, int], tuple[str, ...]] = {}
    cases: list[tuple[str, LabelledCase[Any]]] = [(SUPPORT_PART, c) for c in support]
    cases += [(CORRECT_PART, c) for c in correct]
    for part, labelled in cases:
        fingerprint = labelled.case.fingerprint if by_fingerprint else None
        for number, expected in sorted(labelled.expected.items()):
            sorts: list[str] = []
            for verdicts in saved:
                if verdicts.check.part != part or expected == UNCLEAR:
                    continue
                given = verdicts.on(labelled.case.case_id, number, fingerprint)
                if given is not None and given.verdict != expected:
                    sorts.append(verdicts.check.short)
            if (labelled.case.case_id, number) in splits.get(part, ()):
                sorts.append(READERS)
            if expected == UNCLEAR:
                sorts.append(UNCLEAR)
            if sorts:
                found[(part, labelled.case.case_id, number)] = tuple(sorts)
    return found


# What can be drawn.


@dataclass(frozen=True, slots=True)
class Candidate:
    """One statement or answer that can be drawn, with what the drawing needs to know."""

    part: str
    case_id: str
    statement: int
    """The number of the statement; 1 for an answer judged as a whole."""
    question: str
    kind: str
    """The kind of statement or answer: ``real``, the kind of fault planted in it, or, for a
    disputed one, the first sort of its dispute. The places of a group spread over them."""
    expected: str
    stratum: str
    shape: str
    """Whether its answer has ``one`` statement or ``several``."""
    disputes: tuple[str, ...] = ()


def stratum_of(part: str, case_kind: str, expected: str, *, disputed: bool) -> str | None:
    """The group a labelled statement or answer belongs to, or ``None`` if it is not drawn.

    A disputed one is in the group of disputed items, whatever its label. Of the others,
    statements labelled ``unclear`` or ``no_claim`` are not drawn, and nor is a supported
    statement in a copy with a planted fault: it repeats a statement of the real answer word
    for word, beside the planted fault. So "supported" and "correct" come from real answers.
    """
    if disputed:
        return DISPUTED
    if part == SUPPORT_PART:
        if expected == "unsupported":
            return "unsupported"
        if expected == "supported" and case_kind == REAL:
            return "supported"
        return None
    if expected in NOT_CORRECT:
        return "not_correct"
    if expected == "correct" and case_kind == REAL:
        return "correct"
    return None


def unit_hash(part: str, case_id: str, statement: int) -> str:
    """The short hash by which a statement or answer is set aside without being named."""
    return hashlib.sha256(f"{part}\n{case_id}\n{statement}".encode()).hexdigest()[:16]


def read_set_aside(root: Path) -> frozenset[str]:
    """The hashes of the statements and answers set aside beside the batches under ``root``.

    One is set aside when its label has been made public where the person could meet it,
    such as in the key of a draw that was replaced. No batch draws it.
    """
    path = root / SET_ASIDE_FILE
    if not path.is_file():
        return frozenset()
    try:
        hashes = json.loads(path.read_text(encoding="utf-8"))["units_sha256"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BatchError(
            f"{path} is not a list of statements and answers set aside: {error}"
        ) from error
    return frozenset(str(value) for value in hashes)


def candidates(sets: Sets) -> list[Candidate]:
    """Every statement and answer of the two labelled sets that can be drawn.

    One that is set aside (:func:`read_set_aside`) is not.
    """
    found: list[Candidate] = []
    for labelled in sets.support:
        shape = "one" if len(labelled.case.statements) == 1 else "several"
        for number, expected in sorted(labelled.expected.items()):
            disputes = sets.disputes.get((SUPPORT_PART, labelled.case.case_id, number), ())
            stratum = stratum_of(SUPPORT_PART, labelled.kind, expected, disputed=bool(disputes))
            if stratum is not None:
                kind = disputes[0] if disputes else labelled.planted.get(number, REAL)
                found.append(
                    Candidate(
                        SUPPORT_PART,
                        labelled.case.case_id,
                        number,
                        labelled.case.question,
                        kind,
                        expected,
                        stratum,
                        shape,
                        disputes,
                    )
                )
    for answer in sets.correct:
        expected = answer.expected[1]
        disputes = sets.disputes.get((CORRECT_PART, answer.case.case_id, 1), ())
        stratum = stratum_of(CORRECT_PART, answer.kind, expected, disputed=bool(disputes))
        if stratum is not None:
            found.append(
                Candidate(
                    CORRECT_PART,
                    answer.case.case_id,
                    1,
                    answer.case.question,
                    disputes[0] if disputes else answer.kind,
                    expected,
                    stratum,
                    "one",
                    disputes,
                )
            )
    return [c for c in found if unit_hash(c.part, c.case_id, c.statement) not in sets.set_aside]


def plan_as_kept(plan: "Plan") -> dict[str, int]:
    """A plan as ``batch.json`` keeps it.

    The file lies beside the page the labeller opens, so its names say how many
    judgements a batch holds and nothing of how they were chosen. The card of the sample
    says what the two "first" counts are.
    """
    return {
        "statements": plan.statements,
        "answers": plan.answers,
        "first_statements": plan.disputed_statements,
        "first_answers": plan.disputed_answers,
    }


def plan_from_kept(kept: Mapping[str, Any]) -> "Plan":
    """The plan a ``batch.json`` holds, read back from :func:`plan_as_kept`."""
    return Plan(
        statements=kept["statements"],
        answers=kept["answers"],
        disputed_statements=kept["first_statements"],
        disputed_answers=kept["first_answers"],
    )


@dataclass(frozen=True, slots=True)
class Plan:
    """How many of each part a batch takes, and how many of them may be disputed items."""

    statements: int = 16
    answers: int = 8
    disputed_statements: int = 6
    disputed_answers: int = 3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 0:
                raise SampleError(f"A batch cannot take {value!r} for {name}.")
        if self.statements + self.answers == 0:
            raise SampleError("A batch needs at least one statement or answer.")


DEFAULT_PLAN = Plan()


def split(rest: int, disputed: Sequence[Candidate], faults: Collection[str]) -> tuple[int, int]:
    """How the places left after the disputed items are shared: (labelled faults, not).

    Half each. An odd place goes to the side of which the disputed items hold fewer, and to
    the faults when they hold as many of each; a label ``unclear`` is on neither side.
    """
    faulted = sum(c.expected in faults for c in disputed)
    clean = sum(c.expected not in faults and c.expected != UNCLEAR for c in disputed)
    half = rest // 2
    if rest % 2 == 0:
        return half, half
    return (half + 1, half) if faulted <= clean else (half, half + 1)


@dataclass(slots=True)
class _Drawing:
    """The state of one drawing: the seed's generator, the questions taken, what was drawn."""

    rng: random.Random
    used: set[str]
    drawn_so_far: Counter[tuple[str, str, str]]
    picked: list[Candidate] = field(default_factory=list)

    def take(
        self,
        pool: Sequence[Candidate],
        part: str,
        stratum: str,
        count: int,
        *,
        phrase: str,
        option: str,
        required: Sequence[str] = (),
        shapes: Sequence[str] = (),
    ) -> list[Candidate]:
        """Draw ``count`` members of one group, each place by the rules of :func:`draw`."""
        members = sorted(
            (c for c in pool if c.part == part and c.stratum == stratum),
            key=lambda c: (c.case_id, c.statement),
        )
        left = len({member.question for member in members} - self.used)
        if left < count:
            raise SampleError(
                f"This batch needs {count} {phrase}, and only {left} are left whose question is "
                f"in no earlier batch and not already in this one. Draw a smaller batch: "
                f"lower {option}."
            )
        kinds = sorted({member.kind for member in members})
        self.rng.shuffle(kinds)
        order = {kind: place for place, kind in enumerate(kinds)}
        taken: list[Candidate] = []
        for place in range(count):
            # Never empty: each place takes one question, and enough were counted above.
            open_members = [m for m in members if m.question not in self.used]
            unmet = [label for label in required if not any(t.expected == label for t in taken)]
            for label in unmet:
                wanted = [m for m in open_members if m.expected == label]
                if wanted:
                    open_members = wanted
                    break
            if place < len(shapes):
                matched = [m for m in open_members if m.shape == shapes[place]]
                open_members = matched or open_members
            by_kind: dict[str, list[Candidate]] = defaultdict(list)
            for member in open_members:
                by_kind[member.kind].append(member)
            kind = min(
                by_kind,
                key=lambda name: (
                    self.drawn_so_far[(part, stratum, name)],
                    -len(by_kind[name]),
                    order[name],
                ),
            )
            # By answer: the question first, then one of its statements. A long answer is
            # not drawn more often for having more statements.
            question = self.rng.choice(sorted({m.question for m in by_kind[kind]}))
            choice = self.rng.choice([m for m in by_kind[kind] if m.question == question])
            taken.append(choice)
            self.used.add(choice.question)
            self.drawn_so_far[(part, stratum, kind)] += 1
        self.picked.extend(taken)
        return taken


def draw(
    pool: Sequence[Candidate],
    *,
    earlier: Sequence[Candidate],
    batch: int,
    seed: int,
    plan: Plan,
) -> list[Candidate]:
    """Draw one batch: the same pool, earlier batches, batch number, seed and plan give the same.

    **Disputed items first**, as many as there are, up to the plan's number for each part:
    answers first, since they are fewer, then statements. Their places spread over the
    sorts of dispute as the rest spread over kinds.

    **Then the rest, in groups by the set's label**: half labelled with a fault and half
    not (see :func:`split`). The faulted statements are drawn first and spread over the
    kinds of planted fault; the answers not correct hold at least one ``partial`` and one
    ``incorrect`` where the set has them; and each supported statement is drawn from an
    answer of the same shape, one statement or several, as a faulted statement drawn
    before it, as far as the set allows, so that the shape of an answer does not tell.

    **Within a group**, each place goes to the kind drawn least so far, counting earlier
    batches; among those, to the kind with the most left; among those, by the seed. A kind
    with nothing left is passed over. A question, then one of its statements, is chosen by
    the seed.

    **No question twice**, in either part or in any later batch. Seeing a question twice
    would show an answer beside its copy with a planted fault, and the reference answer of
    part B would tell the person more than the passages of part A.
    """
    drawing = _Drawing(
        random.Random(f"handbook-rag human sample {seed} {batch}"),
        {c.question for c in earlier},
        Counter((c.part, c.stratum, c.kind) for c in earlier),
    )

    def disputed_left(part: str) -> int:
        return len(
            {c.question for c in pool if c.part == part and c.stratum == DISPUTED} - drawing.used
        )

    answers_disputed = drawing.take(
        pool,
        CORRECT_PART,
        DISPUTED,
        min(plan.disputed_answers, plan.answers, disputed_left(CORRECT_PART)),
        phrase="disputed answers",
        option="--disputed-answers",
    )
    statements_disputed = drawing.take(
        pool,
        SUPPORT_PART,
        DISPUTED,
        min(plan.disputed_statements, plan.statements, disputed_left(SUPPORT_PART)),
        phrase="disputed statements",
        option="--disputed-statements",
    )
    unsupported, supported = split(
        plan.statements - len(statements_disputed), statements_disputed, FAULTS[SUPPORT_PART]
    )
    not_correct, correct = split(
        plan.answers - len(answers_disputed), answers_disputed, FAULTS[CORRECT_PART]
    )
    faulted = drawing.take(
        pool,
        SUPPORT_PART,
        "unsupported",
        unsupported,
        phrase="statements labelled unsupported",
        option="--statements",
    )
    drawing.take(
        pool,
        CORRECT_PART,
        "not_correct",
        not_correct,
        phrase="answers labelled partial or incorrect",
        option="--answers",
        required=("partial", "incorrect"),
    )
    drawing.take(
        pool,
        SUPPORT_PART,
        "supported",
        supported,
        phrase="statements of real answers labelled supported",
        option="--statements",
        shapes=[c.shape for c in faulted],
    )
    drawing.take(
        pool,
        CORRECT_PART,
        "correct",
        correct,
        phrase="real answers labelled correct",
        option="--answers",
    )
    return drawing.picked


SMALLEST_GROUP_FOR_ORDER = 3
"""Below this many members in a group, the order of a part is not checked or reshuffled.

With a group of one or two, most orders have it at an end or alone, and rejecting some of
them would give its position away. Their order is then left as the shuffle gives it."""
MAX_SHUFFLES = 1000


def runs(order: Sequence[str]) -> int:
    """How many runs of one group an order holds: ``a a b a`` holds three."""
    if not order:
        return 0
    return 1 + sum(1 for before, after in pairwise(order) if before != after)


def telling(order: Sequence[str]) -> bool:
    """Whether an order shows its groups: each group in one run, or no two of a group together.

    Both are checked only when every group has at least :data:`SMALLEST_GROUP_FOR_ORDER`
    members; with fewer, an order says too little to check, and rejecting some orders would
    itself give a position away.
    """
    sizes = Counter(order)
    if len(sizes) < 2 or min(sizes.values()) < SMALLEST_GROUP_FOR_ORDER:
        return False
    largest = max(sizes.values())
    most = len(order) if largest <= len(order) - largest + 1 else 2 * (len(order) - largest) + 1
    return runs(order) in {len(sizes), most}


def ordered(picked: Sequence[Candidate], *, batch: int, seed: int) -> list[tuple[str, Candidate]]:
    """The drawn items in the order the person sees them, with neutral ids.

    Part A comes first and part B after it. Within a part the order is shuffled by the
    seed, and shuffled again while it shows its groups (:func:`telling`), at most
    :data:`MAX_SHUFFLES` times, so that it always ends. The ids are the part's letter and a
    place, given after the shuffle: ``A01`` is the first item of part A, whatever it is.
    """
    rng = random.Random(f"handbook-rag human order {seed} {batch}")
    result: list[tuple[str, Candidate]] = []
    for part in PARTS:
        members = [candidate for candidate in picked if candidate.part == part]
        rng.shuffle(members)
        for _ in range(MAX_SHUFFLES):
            if not telling([member.stratum for member in members]):
                break
            rng.shuffle(members)
        else:
            raise SampleError(f"No order of part {PART_LETTERS[part]} hides its groups.")
        letter = PART_LETTERS[part]
        result.extend(
            (f"{letter}{place:02d}", member) for place, member in enumerate(members, start=1)
        )
    return result


# The items, the key and the batch's files.


def support_item(labelled: LabelledCase[SupportCase], number: int) -> dict[str, Any]:
    """What the person is shown for one statement: the whole answer, and its passages.

    The answer is shown as its statements, each with the passages it cites, and the one to
    label is marked. The passages shown are those the marked statement cites, in full and
    with their title path. They are what the rubric lets a reader judge it by.
    """
    case = labelled.case
    marked = next(statement for statement in case.statements if statement.number == number)
    return {
        "part": SUPPORT_PART,
        "question": case.question,
        "statements": [
            {"number": statement.number, "text": statement.text, "cites": list(statement.cites)}
            for statement in case.statements
        ],
        "marked": number,
        "passages": [
            {"number": passage.number, "title": passage.title, "text": passage.text}
            for passage in case.passages
            if passage.number in marked.cites
        ],
    }


def correct_item(labelled: LabelledCase[CorrectnessCase]) -> dict[str, Any]:
    """What the person is shown for one answer: the question, the reference and the answer."""
    case = labelled.case
    return {
        "part": CORRECT_PART,
        "question": case.question,
        "reference": case.reference,
        "answer": case.answer,
    }


def short_hash(data: bytes) -> str:
    """The short hash by which the files of a batch name each other's exact content."""
    return hashlib.sha256(data).hexdigest()[:16]


def item_hash(item: Mapping[str, Any]) -> str:
    """The hash of what an item shows, its id included and its own hash left out."""
    content = {key: value for key, value in item.items() if key != "sha256"}
    return short_hash(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def items_text(items: Sequence[Mapping[str, Any]]) -> str:
    """The items file: one item on each line, in the order the person sees them."""
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def key_text(key: Mapping[str, Any]) -> str:
    """The key as it is written, once the labels are in place, and as its hash is taken."""
    return json.dumps(key, indent=2, ensure_ascii=False) + "\n"


def key_hash(key: Mapping[str, Any]) -> str:
    """The whole SHA-256 of the key: a commitment to it, made before any label exists."""
    return hashlib.sha256(key_text(key).encode("utf-8")).hexdigest()


def draw_key(
    pool: Sequence[Candidate],
    *,
    number: int,
    seed: int,
    plan: Plan,
    earlier: Sequence[Candidate],
) -> tuple[list[tuple[str, Candidate]], dict[str, Any]]:
    """Draw a batch's items and order them, and make its key. No passage is needed for this."""
    chosen = ordered(
        draw(pool, earlier=earlier, batch=number, seed=seed, plan=plan), batch=number, seed=seed
    )
    key = {
        "batch": number,
        "seed": seed,
        "plan": asdict(plan),
        "items": {
            item_id: {
                "part": c.part,
                "case": c.case_id,
                "statement": c.statement,
                "question": c.question,
                "stratum": c.stratum,
                "kind": c.kind,
                "disputes": list(c.disputes),
                "expected": c.expected,
            }
            for item_id, c in chosen
        },
    }
    return chosen, key


def build_items(chosen: Sequence[tuple[str, Candidate]], sets: Sets) -> list[dict[str, Any]]:
    """What the person is shown for each drawn item, in order, each with its hash."""
    items: list[dict[str, Any]] = []
    for item_id, c in chosen:
        labelled = sets.case(c.part, c.case_id)
        content = (
            support_item(labelled, c.statement)
            if c.part == SUPPORT_PART
            else correct_item(labelled)
        )
        item = {"item": item_id, **content}
        item["sha256"] = item_hash(item)
        items.append(item)
    return items


@dataclass(frozen=True, slots=True)
class DrawnBatch:
    """A batch as the drawing makes it."""

    number: int
    seed: int
    plan: Plan
    items: list[dict[str, Any]]
    """In the order the person sees them, each with its id and its hash."""
    key: dict[str, Any]
    picked: list[Candidate]
    sources: dict[str, Any]

    @property
    def meta(self) -> dict[str, Any]:
        """What ``batch.json`` holds: how it was drawn, and the hashes of its items and key."""
        return {
            "batch": self.number,
            "seed": self.seed,
            "plan": plan_as_kept(self.plan),
            "sources": self.sources,
            "items_sha256": short_hash(items_text(self.items).encode("utf-8")),
            "joins_sha256": key_hash(self.key),
        }


def make_batch(
    *,
    number: int,
    seed: int,
    plan: Plan,
    sets: Sets,
    earlier: Sequence[Candidate],
    sources: Mapping[str, Any],
) -> DrawnBatch:
    """Draw a batch from the two labelled sets, and build its items and its key.

    ``sources`` says what the batch was drawn from, in words that hold no path of this
    machine. It goes into ``batch.json`` as it is.
    """
    chosen, key = draw_key(candidates(sets), number=number, seed=seed, plan=plan, earlier=earlier)
    return DrawnBatch(
        number, seed, plan, build_items(chosen, sets), key, [c for _, c in chosen], dict(sources)
    )


def sources_of(root: Path, *, handbook_commit: str, chunking: str) -> dict[str, Any]:
    """What a batch is drawn from: the sets and saved checks, by path and hash, and the handbook.

    A key is drawn again from exactly these. If one of them changes, the key drawn again
    can differ, and the hashes say which file did.
    """

    def digest(path: Path) -> str | None:
        target = root / path
        return hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None

    return {
        "sets": {
            part: {"path": path.as_posix(), "sha256": digest(path)}
            for part, path in SET_PATHS.items()
        },
        "checks": {
            check.short: {
                "path": (check.run_dir / "judgements.jsonl").as_posix(),
                "sha256": digest(check.run_dir / "judgements.jsonl"),
            }
            for check in SAVED_CHECKS
        },
        "set_aside": {
            "path": (BATCHES_DIR / SET_ASIDE_FILE).as_posix(),
            "sha256": digest(BATCHES_DIR / SET_ASIDE_FILE),
        },
        "handbook_commit": handbook_commit,
        "chunking": chunking,
    }


@dataclass(frozen=True, slots=True)
class KeyEntry:
    """What the key says of one item."""

    item: str
    part: str
    case: str
    statement: int
    question: str
    stratum: str
    kind: str
    disputes: tuple[str, ...]
    expected: str
    """The label the set held when the batch was drawn."""


def key_entries(key: Mapping[str, Any]) -> dict[str, KeyEntry]:
    """The entries of a key, by item id. A key that is not as the drawing writes it fails."""
    return {
        str(item_id): KeyEntry(
            item=str(item_id),
            part=str(entry["part"]),
            case=str(entry["case"]),
            statement=int(entry["statement"]),
            question=str(entry["question"]),
            stratum=str(entry["stratum"]),
            kind=str(entry["kind"]),
            disputes=tuple(str(sort) for sort in entry["disputes"]),
            expected=str(entry["expected"]),
        )
        for item_id, entry in key["items"].items()
    }


@dataclass(frozen=True, slots=True)
class Batch:
    """A drawn batch, read back from disk and checked against itself."""

    number: int
    directory: Path
    seed: int
    plan: Plan
    sources: dict[str, Any]
    items_sha256: str
    joins_sha256: str
    items: dict[str, dict[str, Any]]
    """By id, in the order of the file."""

    @property
    def labels_path(self) -> Path:
        return self.directory / LABELS_FILE

    @property
    def key_path(self) -> Path:
        return self.directory / KEY_FILE


def batch_dir(root: Path, number: int) -> Path:
    return root / f"batch-{number}"


def read_batches(root: Path) -> list[Batch]:
    """Every batch drawn under ``root``, by number. A folder that is not a whole batch fails."""
    if not root.is_dir():
        return []
    numbers = sorted(
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir() and (match := _BATCH_DIR.fullmatch(path.name))
    )
    return [read_batch(batch_dir(root, number)) for number in numbers]


def read_batch(directory: Path) -> Batch:
    """Read a batch, and check that its items are those its ``batch.json`` was written for."""
    match = _BATCH_DIR.fullmatch(directory.name)
    if match is None:
        raise BatchError(f"{directory} is not named as a batch is: batch-1, batch-2 and so on.")
    number = int(match.group(1))
    items_path, meta_path = directory / ITEMS_FILE, directory / BATCH_FILE
    for path in (items_path, meta_path):
        if not path.is_file():
            raise BatchError(
                f"{path} is missing. A batch is drawn with `handbook-rag human-sample`."
            )
    raw_items = items_path.read_bytes()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        items = [json.loads(line) for line in raw_items.decode("utf-8").splitlines() if line]
        plan = plan_from_kept(meta["plan"])
        seed, items_sha256, joins_sha256 = (
            int(meta["seed"]),
            str(meta["items_sha256"]),
            str(meta["joins_sha256"]),
        )
        meta_number, sources = meta["batch"], dict(meta["sources"])
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
        raise BatchError(f"{directory} is not a batch as the drawing writes it: {error}") from error
    if meta_number != number:
        raise BatchError(f"{meta_path} is of batch {meta_number}, not of batch {number}.")
    if short_hash(raw_items) != items_sha256:
        raise BatchError(
            f"{items_path} is not the file {meta_path} was written for: it was changed."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("item") if isinstance(item, dict) else None
        if not isinstance(item_id, str) or item_id in by_id:
            raise BatchError(f"{items_path} holds an item without an id of its own: {item_id!r}")
        if item_hash(item) != item.get("sha256") or item.get("part") not in PARTS:
            raise BatchError(f"{items_path}: item {item_id} is not what was drawn.")
        by_id[item_id] = item
    return Batch(number, directory, seed, plan, sources, items_sha256, joins_sha256, by_id)


def redraw(batches: Sequence[Batch], sets: Sets, *, with_items: bool = True) -> list[DrawnBatch]:
    """Draw each batch again, in order, from its seed and plan, and check it against its hashes.

    The key drawn again must have the hash its ``batch.json`` holds. With ``with_items`` the
    items are built again too and must be those of the folder, which needs the handbook's
    passages; without, only the key is drawn, and the items are taken from the folder.
    A batch that is not drawn again exactly is refused, with the files that may have changed.
    """
    pool = candidates(sets)
    earlier: list[Candidate] = []
    drawn: list[DrawnBatch] = []
    for batch in batches:
        chosen, key = draw_key(
            pool, number=batch.number, seed=batch.seed, plan=batch.plan, earlier=earlier
        )
        if key_hash(key) != batch.joins_sha256:
            raise BatchError(
                f"The key of batch {batch.number} drawn again is not the one drawn first: its hash "
                f"differs from {batch.directory / BATCH_FILE}. A labelled set or a saved check has "
                f"changed since; compare their hashes with those in {BATCH_FILE}."
            )
        items = build_items(chosen, sets) if with_items else list(batch.items.values())
        if with_items and short_hash(items_text(items).encode("utf-8")) != batch.items_sha256:
            raise BatchError(
                f"The items of batch {batch.number} drawn again are not those in its folder. The "
                "handbook's passages or the sets have changed since it was drawn."
            )
        earlier += [c for _, c in chosen]
        drawn.append(
            DrawnBatch(
                batch.number,
                batch.seed,
                batch.plan,
                items,
                key,
                [c for _, c in chosen],
                batch.sources,
            )
        )
    return drawn


def write_key(batch: Batch, key: Mapping[str, Any]) -> None:
    """Write a batch's key beside its labels, once they are in place, after checking its hash."""
    if key_hash(key) != batch.joins_sha256:
        raise BatchError(f"The key of batch {batch.number} does not have the hash drawn first.")
    if not batch.labels_path.is_file():
        raise BatchError(f"Batch {batch.number} has no labels in place: its key waits for them.")
    text = key_text(key)
    if batch.key_path.is_file() and batch.key_path.read_text(encoding="utf-8") != text:
        raise BatchError(f"{batch.key_path} is not the key drawn again: it was changed.")
    batch.key_path.write_text(text, encoding="utf-8")


# What could tell a label.


_SUPPORT_FIELDS = frozenset(
    {"item", "part", "question", "statements", "marked", "passages", "sha256"}
)
_STATEMENT_FIELDS = frozenset({"number", "text", "cites"})
_PASSAGE_FIELDS = frozenset({"number", "title", "text"})
_CORRECT_FIELDS = frozenset({"item", "part", "question", "reference", "answer", "sha256"})


def item_texts(item: Mapping[str, Any]) -> list[str]:
    """The texts of an item that come from the sets, not from the handbook's passages."""
    texts = [str(item.get("question", ""))]
    texts += [str(s.get("text", "")) for s in item.get("statements", [])]
    texts += [str(item.get(name, "")) for name in ("reference", "answer") if name in item]
    return texts


def leaks(
    items: Sequence[Mapping[str, Any]],
    key: Mapping[str, KeyEntry],
    sets: Sets,
    *,
    shown: str,
) -> list[str]:
    """What, in the items of a batch and in what the person can open, could tell a label.

    ``shown`` is every text the person is given: the items file, the page and its steps.
    The code checks:

    - **fields**: each item holds only the fields of what is shown, so that no label,
      verdict, reader, kind or group can ride along in another field;
    - **ids**: each item's id is its part's letter and its place, ``A01``, ``A02`` and so on;
    - **the words of the sets**: each item shows exactly the words of its case: its question
      is the question of the real answer, down to a space, and no text of it starts or ends
      with white space;
    - **names**: no id of the sets (an id such as ``b04-v1`` names a copy) is in ``shown``,
      and no item's text holds a name of a kind, a group or a sort of dispute that no real
      answer of the sets uses (:meth:`Sets.telling_words`);
    - **order**: within each part, the order does not put each group in one run, nor keep
      every two of a group apart, where the groups are large enough to be checked
      (:func:`telling`).

    It does not check what the shape of an answer, or the kind of fault in it, could hint
    at; the card says what is left.
    """
    found: list[str] = []
    real_questions = sets.real_questions()
    telling_words = sets.telling_words()
    for place, item in enumerate(items):
        item_id = item.get("item")
        part = item.get("part")
        if set(item) != (_SUPPORT_FIELDS if part == SUPPORT_PART else _CORRECT_FIELDS):
            found.append(f"item {item_id} holds fields beyond what is shown: {sorted(item)}")
        for statement in item.get("statements", []):
            if set(statement) != _STATEMENT_FIELDS:
                found.append(
                    f"item {item_id} has a statement with other fields: {sorted(statement)}"
                )
        for passage in item.get("passages", []):
            if set(passage) != _PASSAGE_FIELDS:
                found.append(f"item {item_id} has a passage with other fields: {sorted(passage)}")
        letter = PART_LETTERS.get(str(part), "?")
        same_part = [other for other in items[:place] if other.get("part") == part]
        if item_id != f"{letter}{len(same_part) + 1:02d}":
            found.append(f"item {item_id!r} does not have the neutral id of its place")
        if item.get("question") not in real_questions:
            found.append(f"item {item_id}: its question is not, word for word, a real answer's")
        texts = item_texts(item)
        if any(text != text.strip() for text in texts):
            found.append(f"item {item_id}: a text starts or ends with white space")
        for word in sorted(telling_words):
            pattern = rf"\b({re.escape(word)}|{re.escape(word.replace('_', ' '))})\b"
            if any(re.search(pattern, text, re.IGNORECASE) for text in texts):
                found.append(f"item {item_id}: its text names {word!r}")
        entry = key.get(str(item_id))
        if entry is not None:
            source = sets.case(entry.part, entry.case)
            expected_item = (
                support_item(source, entry.statement)
                if entry.part == SUPPORT_PART
                else correct_item(source)
            )
            shown_item = {k: v for k, v in item.items() if k not in {"item", "sha256"}}
            if entry.part == SUPPORT_PART and not source.case.passages:
                # Sets read without the handbook have no passages to compare; a test holds
                # the passages to the sets by their hashes instead.
                shown_item.pop("passages", None)
                expected_item.pop("passages", None)
            if shown_item != expected_item:
                found.append(f"item {item_id} does not show exactly the words of its case")
    for case_id in sets.case_ids():
        if re.search(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", shown):
            found.append(f"the id {case_id} of a labelled set is in what is shown")
    for part in PARTS:
        order = [
            entry.stratum if (entry := key.get(str(item.get("item")))) else "not in the key"
            for item in items
            if item.get("part") == part
        ]
        if telling(order):
            found.append(f"the order of part {PART_LETTERS[part]} shows its groups")
    return found
