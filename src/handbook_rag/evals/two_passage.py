"""Questions whose answer needs two passages on two different pages.

Almost every golden question rests on one passage, so a search that brings one of two
needed passages looks like a hit. These questions are kept in a file of their own, in
the golden set's format, so that every command that reads a golden set reads them too:
``eval-retrieval`` scores them, and ``golden-check`` verifies their quotes.

What the golden format has no place for is in a second file, one line for each question:
which kind of question it is, and why it needs each of its two passages. "Needs two" is
a claim about the question, and a claim that is not written down cannot be checked. The
reason is given from both sides: what a reader who has only the first passage gets wrong
or leaves out, and the same for the second.

The claim was then put to a second reading. A reader who had not seen the reasons was
shown a question with both passages, or with the first only, or with the second only,
and said whether the question could be answered in full from what it was shown. The
second file records what the readers said of each question, and a third file holds the
readers' own lines. A question is confirmed when it was complete with both passages and
with neither one alone. One that was not is kept, marked, and left out of every figure.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from handbook_rag.evals.dataset import DatasetError, GoldenQuestion

Kind = Literal["comparison", "rule-and-exception", "process", "term-and-rule"]
KINDS: tuple[str, ...] = get_args(Kind)
"""``comparison``: the same thing in two countries, teams or programmes, each on its own
page. ``rule-and-exception``: a rule on one page, and its exception, its condition or a
local version of it on another. ``process``: two steps of one process, on two pages.
``term-and-rule``: one page says what a thing is or which class it falls in, and a rule
on another page is written for that term or class."""

Label = Literal["complete", "partial", "none"]
"""What a second reader said of a question and the passages it was shown: the question
can be answered in full from them, in part, or not at all."""

Shown = Literal["both", "first", "second", "control", "control-and-neighbour"]
"""What an item of the second reading showed. ``both``, ``first``, ``second``: the two
passages of a two-passage question, or one of them. ``control``: a golden question that
rests on one passage, with that passage. ``control-and-neighbour``: the same, with a
neighbouring passage of the page beside it. Without the controls, an item with one passage would
always have been half of the evidence, and the layout would have given the answer away."""
OF_A_QUESTION: tuple[str, ...] = ("both", "first", "second")


class SecondReading(BaseModel):
    """The labels one question got in the second reading, one for each way it was shown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    both: Label
    first: Label
    second: Label

    @property
    def confirms(self) -> bool:
        """Complete with both passages, and with neither one alone."""
        return self.both == "complete" and "complete" not in (self.first, self.second)


class WhyTwo(BaseModel):
    """Why one question needs both of its passages, and whether a second reading agreed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: Kind
    only_first: str = Field(min_length=1)
    """What a reader who has only the first passage gets wrong or leaves out."""
    only_second: str = Field(min_length=1)
    """The same for a reader who has only the second passage."""
    second_reading: SecondReading | None = None
    """What the second readers said. ``None``: the question has not been read again."""
    confirmed: bool | None = None
    """Whether the second reading found that the answer needs both passages. Written
    out, so that a disputed question can be seen in the file, and held to the labels."""

    @model_validator(mode="after")
    def _confirmed_is_what_the_labels_say(self) -> Self:
        if self.second_reading is None:
            if self.confirmed is not None:
                raise ValueError("confirmed is given, and no second reading that it rests on")
        elif self.confirmed is None:
            raise ValueError("a second reading is given, and confirmed is not")
        elif self.confirmed != self.second_reading.confirms:
            raise ValueError(
                f"confirmed is {str(self.confirmed).lower()}, and the labels of the second "
                f"reading say {str(self.second_reading.confirms).lower()}"
            )
        return self


class ReadingLine(BaseModel):
    """One line of a second reader: what it said of one item it was shown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item: str = Field(min_length=1)
    """The neutral name the reader knew the item by. Its first letter is the reader's form."""
    question_id: str = Field(min_length=1)
    shown: Shown
    label: Label
    confidence: Literal["high", "medium", "low"]
    answer: str = Field(min_length=1)
    """The answer the reader gave from what it was shown."""
    missing: str | None
    """What the reader said was missing. ``None`` where it called the item complete."""

    @model_validator(mode="after")
    def _something_is_missing_unless_complete(self) -> Self:
        if (self.missing is None) != (self.label == "complete"):
            raise ValueError("a complete item has nothing missing, and any other says what is")
        return self


def _load[M: BaseModel](path: Path, model: type[M]) -> list[tuple[int, M]]:
    """The lines of a JSON Lines file, each as a model beside its line number.

    A line that is not what the model asks for is reported with its place in the file.
    """
    loaded: list[tuple[int, M]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            loaded.append((number, model.model_validate(json.loads(line))))
        except (json.JSONDecodeError, ValidationError) as error:
            raise DatasetError(f"{path}:{number}: {error}") from error
    return loaded


def load_reasons(path: Path) -> dict[str, WhyTwo]:
    """Read the reasons from a JSON Lines file, one question per line, by question id."""
    reasons: dict[str, WhyTwo] = {}
    for number, reason in _load(path, WhyTwo):
        if reason.id in reasons:
            raise DatasetError(f"{path}:{number}: id {reason.id} is used twice")
        reasons[reason.id] = reason
    return reasons


def load_second_reading(path: Path) -> list[ReadingLine]:
    """Read the second readers' lines from a JSON Lines file, one item per line."""
    lines: list[ReadingLine] = []
    seen: set[str] = set()
    for number, line in _load(path, ReadingLine):
        if line.item in seen:
            raise DatasetError(f"{path}:{number}: item {line.item} is used twice")
        seen.add(line.item)
        lines.append(line)
    return lines


def second_reading_of(lines: Sequence[ReadingLine]) -> dict[str, SecondReading]:
    """The three labels of each two-passage question, from the readers' lines.

    Raises if a question was not shown in each of its three ways exactly once. The lines
    of the controls are passed over: they are not questions of the set.
    """
    labels: dict[str, dict[str, Label]] = {}
    for line in lines:
        if line.shown not in OF_A_QUESTION:
            continue
        of_question = labels.setdefault(line.question_id, {})
        if line.shown in of_question:
            raise DatasetError(f"{line.question_id} was shown with {line.shown} more than once")
        of_question[line.shown] = line.label
    readings: dict[str, SecondReading] = {}
    for question_id, of_question in labels.items():
        if set(of_question) != set(OF_A_QUESTION):
            shown = ", ".join(sorted(of_question))
            raise DatasetError(f"{question_id} was shown with {shown} only")
        readings[question_id] = SecondReading.model_validate(of_question)
    return readings


def check_two_passage_set(
    questions: Sequence[GoldenQuestion], reasons: Mapping[str, WhyTwo]
) -> list[str]:
    """Everything that keeps a set from being a set of two-passage questions.

    This is about the shape of the labels, and about the review status saying what the
    second reading found: ``checked`` for a question it confirmed, ``pending`` with a
    note for one it did not. Whether a question truly needs both passages is a matter of
    reading, and no code can decide it.
    """
    problems: list[str] = []
    for question in questions:
        pages = [evidence.doc_id for evidence in question.evidence]
        if question.type != "answerable":
            problems.append(f"{question.id}: a two-passage question has an answer in the handbook")
        elif len(pages) != 2:
            problems.append(f"{question.id}: {len(pages)} pieces of evidence, where two are needed")
        elif pages[0] == pages[1]:
            problems.append(f"{question.id}: both passages are on {pages[0]}")
        if question.evidence_mode != "all":
            problems.append(f"{question.id}: evidence_mode any says that one passage is enough")
        reason = reasons.get(question.id)
        if reason is None:
            problems.append(f"{question.id}: no reason says why it needs both passages")
        elif reason.confirmed is None:
            if question.review == "checked":
                problems.append(f"{question.id}: marked checked, and no second reading is recorded")
        elif reason.confirmed:
            if question.review == "pending":
                problems.append(
                    f"{question.id}: confirmed by the second reading, and still marked pending"
                )
        elif question.review != "pending":
            problems.append(
                f"{question.id}: marked {question.review}, and the second reading did not "
                "confirm that it needs both passages"
            )
        elif not question.note.strip():
            problems.append(f"{question.id}: disputed, and no note says what the dispute is")
    known = {question.id for question in questions}
    for question_id in reasons:
        if question_id not in known:
            problems.append(f"{question_id}: a reason for a question that is not in the set")
    return problems


def kinds_of(questions: Sequence[GoldenQuestion], reasons: Mapping[str, WhyTwo]) -> Counter[str]:
    """How many of the questions are of each kind."""
    return Counter(reasons[q.id].kind for q in questions if q.id in reasons)


def reading_of(questions: Sequence[GoldenQuestion], reasons: Mapping[str, WhyTwo]) -> Counter[str]:
    """How many of the questions the second reading confirmed, disputed or has not seen."""
    names = {None: "not read a second time", True: "confirmed", False: "disputed"}
    return Counter(names[reasons[q.id].confirmed] for q in questions if q.id in reasons)
