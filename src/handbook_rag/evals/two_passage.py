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
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from handbook_rag.evals.dataset import DatasetError, GoldenQuestion

Kind = Literal["comparison", "rule-and-exception", "process", "term-and-rule"]
KINDS: tuple[str, ...] = get_args(Kind)
"""``comparison``: the same thing in two countries, teams or programmes, each on its own
page. ``rule-and-exception``: a rule on one page, and its exception, its condition or a
local version of it on another. ``process``: two steps of one process, on two pages.
``term-and-rule``: one page says what a thing is or which class it falls in, and a rule
on another page is written for that term or class."""


class WhyTwo(BaseModel):
    """Why one question needs both of its passages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    kind: Kind
    only_first: str = Field(min_length=1)
    """What a reader who has only the first passage gets wrong or leaves out."""
    only_second: str = Field(min_length=1)
    """The same for a reader who has only the second passage."""


def load_reasons(path: Path) -> dict[str, WhyTwo]:
    """Read the reasons from a JSON Lines file, one question per line, by question id."""
    reasons: dict[str, WhyTwo] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            reason = WhyTwo.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise DatasetError(f"{path}:{number}: {error}") from error
        if reason.id in reasons:
            raise DatasetError(f"{path}:{number}: id {reason.id} is used twice")
        reasons[reason.id] = reason
    return reasons


def check_two_passage_set(
    questions: Sequence[GoldenQuestion], reasons: Mapping[str, WhyTwo]
) -> list[str]:
    """Everything that keeps a set from being a set of two-passage questions.

    This is about the shape of the labels. Whether a question truly needs both passages
    is a matter of reading, and no code can decide it.
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
        if question.id not in reasons:
            problems.append(f"{question.id}: no reason says why it needs both passages")
    known = {question.id for question in questions}
    for question_id in reasons:
        if question_id not in known:
            problems.append(f"{question_id}: a reason for a question that is not in the set")
    return problems


def kinds_of(questions: Sequence[GoldenQuestion], reasons: Mapping[str, WhyTwo]) -> Counter[str]:
    """How many of the questions are of each kind."""
    return Counter(reasons[q.id].kind for q in questions if q.id in reasons)
