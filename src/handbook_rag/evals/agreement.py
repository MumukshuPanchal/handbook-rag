"""How far a judge agrees with labelled examples.

A judge's score means nothing until the judge itself has been checked. The check is
simple: give the judge statements whose verdict is known, and count how often it reaches
the same one. Two numbers matter most, and both are about the *faults*: the verdicts
that count against an answer, such as ``unsupported``.

- **Recall**: of the statements labelled with a fault, how many did the judge fault? A
  judge with low recall lets faults through, so every answer it passes is worth less.
- **Precision**: of the statements the judge faulted, how many are labelled with a fault?
  A judge with low precision raises false alarms, so the score it reports is too low.

The labelled examples hold real answers and copies of them with one planted fault. The
planted faults are what makes recall measurable at all: real answers hold too few.

Every share is reported with a Wilson interval, which stays honest when the judge agrees
every time: 41 of 41 is not certainty.
"""

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from handbook_rag.evals.metrics import Estimate, share_with_interval

REAL = "real"
"""The kind of a statement that is as the assistant wrote it, with no planted fault."""


@dataclass(frozen=True, slots=True)
class Pair:
    """One statement: the verdict its label gives it, and the verdict the judge gave it."""

    case_id: str
    statement: int
    expected: str
    judged: str
    kind: str = REAL
    """``real``, or the kind of fault that was planted in this statement."""


@dataclass(frozen=True, slots=True)
class Agreement:
    """The judge's agreement with the labels, statement by statement and answer by answer."""

    statements: int
    matrix: dict[str, dict[str, int]]
    """How often each labelled verdict (outer key) was judged as each verdict (inner key)."""
    same_verdict: Estimate
    """Share of statements judged exactly as labelled, of the three verdicts."""
    same_decision: Estimate
    """Share of statements on which judge and label agree whether it is a fault.
    This is the decision an answer's score depends on."""
    kappa: float
    """Cohen's kappa for that decision: agreement beyond what chance alone would give.
    1 is complete agreement, 0 is no better than chance, and a judge that does worse than
    chance scores below 0, down to -1."""
    recall: Estimate | None
    """Of the statements labelled with a fault, the share the judge faulted."""
    precision: Estimate | None
    """Of the statements the judge faulted, the share labelled with a fault."""
    caught: dict[str, tuple[int, int]]
    """For each kind of statement labelled with a fault: how many the judge caught, of how many."""
    answers: int
    same_answer: Estimate
    """Share of answers on which judge and labels agree whether the answer is supported."""
    answer_recall: Estimate | None
    """Of the answers with a statement labelled with a fault, the share the judge failed."""
    answer_precision: Estimate | None
    """Of the answers the judge failed, the share with a statement labelled with a fault."""


def measure_agreement(
    pairs: Sequence[Pair], *, verdicts: Sequence[str], faults: Collection[str]
) -> Agreement:
    """Compare the judge's verdicts with the labels.

    ``verdicts`` are the verdicts the criterion allows, and ``faults`` those among them
    that count against an answer. Both have to be given: with the wrong faults every
    figure here would be about another decision.
    """
    if not pairs:
        raise ValueError("Agreement needs at least one labelled statement.")
    matrix = {expected: dict.fromkeys(verdicts, 0) for expected in verdicts}
    for pair in pairs:
        matrix[pair.expected][pair.judged] += 1

    def is_fault(verdict: str) -> bool:
        return verdict in faults

    labelled_bad = [pair for pair in pairs if is_fault(pair.expected)]
    judged_bad = [pair for pair in pairs if is_fault(pair.judged)]
    by_kind: dict[str, list[Pair]] = defaultdict(list)
    for pair in labelled_bad:
        by_kind[pair.kind].append(pair)

    by_case: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        by_case[pair.case_id].append(pair)
    # An answer is faulted when any of its statements is.
    answers = [
        (
            any(is_fault(pair.expected) for pair in members),
            any(is_fault(pair.judged) for pair in members),
        )
        for members in by_case.values()
    ]
    return Agreement(
        statements=len(pairs),
        matrix=matrix,
        same_verdict=_share(pair.expected == pair.judged for pair in pairs),
        same_decision=_share(is_fault(pair.expected) == is_fault(pair.judged) for pair in pairs),
        kappa=_kappa([(is_fault(pair.expected), is_fault(pair.judged)) for pair in pairs]),
        recall=_share_or_none(is_fault(pair.judged) for pair in labelled_bad),
        precision=_share_or_none(is_fault(pair.expected) for pair in judged_bad),
        caught={
            kind: (sum(is_fault(pair.judged) for pair in members), len(members))
            for kind, members in sorted(by_kind.items())
        },
        answers=len(answers),
        same_answer=_share(labelled == judged for labelled, judged in answers),
        answer_recall=_share_or_none(judged for labelled, judged in answers if labelled),
        answer_precision=_share_or_none(labelled for labelled, judged in answers if judged),
    )


def _share(flags: Iterable[bool]) -> Estimate:
    values = [bool(flag) for flag in flags]
    return share_with_interval(sum(values), len(values))


def _share_or_none(flags: Iterable[bool]) -> Estimate | None:
    values = [bool(flag) for flag in flags]
    return share_with_interval(sum(values), len(values)) if values else None


def _kappa(decisions: Sequence[tuple[bool, bool]]) -> float:
    """Cohen's kappa for two raters and a yes-or-no decision.

    Two raters who both say "no" almost every time agree almost every time by chance
    alone. Kappa takes that out: it is the agreement observed beyond the agreement
    expected from how often each rater says "yes", as a share of the most that could be.
    When chance alone already gives full agreement, there is nothing left to measure, and
    kappa is given as 1 if the raters agree throughout and 0 otherwise.
    """
    total = len(decisions)
    counts = Counter(decisions)
    observed = (counts[(True, True)] + counts[(False, False)]) / total
    first_yes = (counts[(True, True)] + counts[(True, False)]) / total
    second_yes = (counts[(True, True)] + counts[(False, True)]) / total
    by_chance = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)
    if by_chance == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - by_chance) / (1 - by_chance)
