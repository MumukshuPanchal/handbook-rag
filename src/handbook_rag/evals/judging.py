"""What every judged criterion shares: the run of the judge, and the verdicts it saves.

A *criterion* is one thing a judge decides about an answer. There are two:

- whether the statements of an answer are backed by the passages they cite
  (:mod:`handbook_rag.evals.support`);
- whether an answer agrees with the reference answer
  (:mod:`handbook_rag.evals.correctness`).

They differ in what the judge reads and in the verdicts it may give. Everything else is
the same for both, and is here: the judge runs over the saved answers of a run to score
them, and over labelled answers to be checked itself (see
:mod:`handbook_rag.evals.agreement`), with the same code.

A verdict is saved with a fingerprint of everything the judge read. A saved verdict is
used again only for exactly the same words, so checking the judge and scoring a run never
pay twice for the same answer.
"""

import hashlib
import json
import statistics
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from handbook_rag.evals.agreement import REAL, Agreement, Pair
from handbook_rag.evals.answers import AnswerRecord
from handbook_rag.evals.careful import run_carefully
from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.judge import JudgeError, Judgement, StatementVerdict
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.stage import SavedRunError, Stage, load_rows
from handbook_rag.ingest.models import Chunk

UNCLEAR = "unclear"
"""The label of a statement on which careful readers did not agree. It is not counted."""


class JudgedCase(Protocol):
    """What the judge is given for one answer, whatever the criterion."""

    @property
    def case_id(self) -> str: ...

    @property
    def fingerprint(self) -> str:
        """A short hash of everything the judge reads for this answer."""
        ...


class Judge[C](Protocol):
    """Anything that can rule on the cases of one criterion."""

    @property
    def model(self) -> str: ...

    def judge(self, case: C) -> Judgement: ...


@dataclass(frozen=True, slots=True)
class LabelledCase[C]:
    """An answer whose verdicts are known."""

    case: C
    kind: str
    """``real``, or the kind of fault planted in it."""
    expected: dict[int, str]
    """Number of a statement to its labelled verdict, or to ``unclear``. A criterion that
    rules on the answer as a whole has one entry, under the number 1."""
    planted: dict[int, str]
    """Number of a statement to the kind of fault planted in it."""


@dataclass(frozen=True, slots=True)
class Criterion[C: JudgedCase]:
    """One thing a judge decides about an answer, and all that sets it apart from another."""

    name: str
    """Its short name in commands and files: ``support`` or ``correct``."""
    metric: str
    """The id of the score it adds to an answer run."""
    title: str
    """What judging a run by it is called, as a heading."""
    unit: str
    """What one verdict is given on: ``statements`` or ``answers``."""
    verdicts: tuple[str, ...]
    faults: frozenset[str]
    """The verdicts that count against an answer."""
    verdict_words: Mapping[str, str]
    """Each verdict as it reads after a count, such as "statements supported"."""
    faults_phrase: str
    """What a labelled fault is called, such as "statements labelled unsupported"."""
    passed_word: str
    failed_word: str
    instructions: str
    """What the judge is told. Verdicts are recorded with a hash of it."""
    request_fingerprint: Callable[[], str]
    """A short hash of how a request is laid out, apart from the instructions."""
    cases: Callable[
        [Sequence[GoldenQuestion], Mapping[str, AnswerRecord], Mapping[str, Chunk]], list[C]
    ]
    """The answers of a run, as the judge reads them."""
    load_labelled: Callable[[Path, Mapping[str, Chunk]], list[LabelledCase[C]]]
    judge: Callable[..., Judge[C]]
    """Builds the judge from a client, a model name and a limit on its reply."""
    reads_passages: bool
    """Whether the judge is shown handbook passages, which have to be fetched from the corpus."""
    labelled_path: Path
    checks_dir: Path
    """Where checks of this judge against the labelled answers are kept."""
    rows_file: str
    state_file: str
    check_file: str
    """Beside a run's verdicts: how far the judge that made them agreed with the labels."""

    def run_stage(self, run_dir: Path) -> Stage:
        """The files in which an answer run keeps this judge's verdicts on its replies."""
        return Stage(
            run_dir,
            self.rows_file,
            self.state_file,
            holds="verdicts",
            advice="Discard the saved verdicts with --restart, or use the judge that made them.",
            outputs=(self.check_file,),
        )

    def passed(self, judgement: Judgement) -> bool:
        """Whether an answer passes: none of its verdicts counts against it."""
        return all(verdict.verdict not in self.faults for verdict in judgement.verdicts)


def judge_check_stage(run_dir: Path) -> Stage:
    """The files of a run that checks a judge against labelled answers."""
    return Stage(
        run_dir,
        "judgements.jsonl",
        "run.json",
        holds="verdicts",
        advice="Give this run another name, or discard its saved verdicts with --restart.",
        outputs=("summary.json", "disagreements.jsonl"),
    )


def judgement_row(case: JudgedCase, judgement: Judgement) -> dict[str, Any]:
    """A judgement as it is saved, with the fingerprint of what was judged."""
    return {
        "case": case.case_id,
        "fingerprint": case.fingerprint,
        "verdicts": [
            {"number": v.number, "verdict": v.verdict, "reason": v.reason}
            for v in judgement.verdicts
        ],
        "model": judgement.model,
        "input_tokens": judgement.input_tokens,
        "output_tokens": judgement.output_tokens,
        "seconds": round(judgement.seconds, 4),
    }


def load_judgements(path: Path, verdicts: Collection[str]) -> dict[str, Judgement]:
    """Read saved judgements, by the fingerprint of what was judged. The latest one wins.

    ``verdicts`` are the verdicts the criterion allows. A row with any other is reported:
    it was edited, or it belongs to another criterion.
    """
    judgements: dict[str, Judgement] = {}
    for number, row in load_rows(path):
        try:
            judgements[str(row["fingerprint"])] = Judgement(
                case_id=str(row["case"]),
                verdicts=tuple(
                    StatementVerdict(
                        int(v["number"]), _verdict(v["verdict"], verdicts), str(v["reason"])
                    )
                    for v in row["verdicts"]
                ),
                model=str(row["model"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                seconds=float(row["seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SavedRunError(f"{path}:{number} is not a saved judgement: {error}") from error
    return judgements


def _verdict(value: object, verdicts: Collection[str]) -> str:
    if not isinstance(value, str) or value not in verdicts:
        raise ValueError(f"unknown verdict {value!r}")
    return value


@dataclass(slots=True)
class JudgeRun:
    """What one invocation of the judge did."""

    judgements: dict[str, Judgement]
    """Every judgement the run holds, by case id: the saved ones and the new ones."""
    reused: int = 0
    """Judgements taken from an earlier run because the judge had read exactly this."""
    failures: list[FailedQuestion] = field(default_factory=list)
    attempts: int = 0
    stopped: str | None = None


def run_judge[C: JudgedCase](
    cases: Sequence[C],
    judge: Judge[C],
    *,
    saved: Mapping[str, Judgement] | None = None,
    wait: Callable[[], None] | None = None,
    on_judgement: Callable[[C, Judgement, bool], None] | None = None,
    on_failure: Callable[[FailedQuestion], None] | None = None,
    rest: Callable[[], None] | None = None,
    max_attempts: int | None = None,
    failures_before_stopping: int = 3,
) -> JudgeRun:
    """Judge every answer that has no saved judgement of exactly what it holds.

    ``saved`` maps the fingerprint of a case to a judgement made earlier. ``on_judgement``
    receives each judgement the run will hold, with a flag saying whether it is new, so
    the caller can save it.

    Requests are spent as in :func:`handbook_rag.evals.careful.run_carefully`: spaced
    out, each failure tried once more, and the run stopped when several fail in a row. A
    reply from the judge that is not a complete set of verdicts counts as a failure. It
    is never read as a verdict.
    """
    saved = saved or {}
    run = JudgeRun(judgements={})
    pending: list[C] = []
    for case in cases:
        earlier = saved.get(case.fingerprint)
        if earlier is None:
            pending.append(case)
            continue
        run.judgements[case.case_id] = earlier
        run.reused += 1
        if on_judgement is not None:
            on_judgement(case, earlier, False)

    # A fixed order that mixes the kinds of answer, so that a run that stops early has
    # still judged a fair sample, with and without planted faults.
    pending.sort(key=lambda case: hashlib.sha256(case.case_id.encode("utf-8")).hexdigest())

    def keep(case: C, judgement: Judgement) -> None:
        run.judgements[case.case_id] = judgement
        if on_judgement is not None:
            on_judgement(case, judgement, True)

    outcome = run_carefully(
        pending,
        judge.judge,
        keep,
        name=lambda case: case.case_id,
        errors=(JudgeError,),
        things="answers",
        wait=wait,
        on_failure=on_failure,
        rest=rest,
        max_attempts=max_attempts,
        failures_before_stopping=failures_before_stopping,
    )
    run.attempts, run.failures, run.stopped = outcome.attempts, outcome.failures, outcome.stopped
    return run


def labelled_pairs[C: JudgedCase](
    cases: Sequence[LabelledCase[C]], judgements: Mapping[str, Judgement]
) -> list[Pair]:
    """Each labelled verdict beside the verdict the judge gave.

    A statement labelled ``unclear`` is left out, and so is every statement of an answer
    the judge has not judged.
    """
    pairs: list[Pair] = []
    for labelled in cases:
        judgement = judgements.get(labelled.case.case_id)
        if judgement is None:
            continue
        judged = {verdict.number: verdict.verdict for verdict in judgement.verdicts}
        for number, expected in labelled.expected.items():
            if expected == UNCLEAR:
                continue
            pairs.append(
                Pair(
                    case_id=labelled.case.case_id,
                    statement=number,
                    expected=expected,
                    judged=judged[number],
                    kind=labelled.planted.get(number, REAL),
                )
            )
    return pairs


def saved_verdicts[C: JudgedCase](
    criterion: Criterion[C], run_dir: Path, cases: Sequence[C]
) -> dict[str, Judgement]:
    """The verdicts a run has saved that are of exactly these answers, by case id.

    A verdict on an answer that has since been asked again, and so reads differently, is
    left out: it is a verdict on other words.
    """
    saved = load_judgements(criterion.run_stage(run_dir).rows_path, criterion.verdicts)
    return {case.case_id: saved[case.fingerprint] for case in cases if case.fingerprint in saved}


def judge_record[C: JudgedCase](
    criterion: Criterion[C], run_dir: Path, judged: Mapping[str, Judgement]
) -> dict[str, object] | None:
    """Who judged a run's answers, what it cost, and how far that judge can be trusted.

    This goes into the run's summary beside the judged score. The agreement figures are
    included only if they were measured with the same judge: the same model, the same
    instructions and the same layout of a request.
    """
    if not judged:
        return None
    stage = criterion.run_stage(run_dir)
    config = stage.config()
    verdicts = [verdict.verdict for judgement in judged.values() for verdict in judgement.verdicts]
    record: dict[str, object] = {
        "config": config,
        "attempts": stage.attempts(),
        "answers_judged": len(judged),
        "verdicts": {name: verdicts.count(name) for name in criterion.verdicts},
        "input_tokens": sum(judgement.input_tokens for judgement in judged.values()),
        "output_tokens": sum(judgement.output_tokens for judgement in judged.values()),
        "median_seconds": statistics.median(j.seconds for j in judged.values()),
        "check": None,
    }
    check_file = run_dir / criterion.check_file
    if check_file.is_file():
        check = json.loads(check_file.read_text(encoding="utf-8"))
        if isinstance(check, dict) and check.get("config") == config:
            record["check"] = {key: check.get(key) for key in ("run", "answers", "agreement")}
    return record


def agreement_headline(agreement: Agreement) -> dict[str, object]:
    """The figures of a judge check that are quoted beside a judged score."""

    def figures(estimate: Estimate | None) -> dict[str, float | int] | None:
        return None if estimate is None else asdict(estimate)

    return {
        "statements": agreement.statements,
        "same_decision": figures(agreement.same_decision),
        "recall": figures(agreement.recall),
        "precision": figures(agreement.precision),
        "kappa": round(agreement.kappa, 4),
        "caught": {kind: list(counts) for kind, counts in agreement.caught.items()},
        "matrix": agreement.matrix,
        "answers": agreement.answers,
        "same_answer": figures(agreement.same_answer),
        "answer_recall": figures(agreement.answer_recall),
        "answer_precision": figures(agreement.answer_precision),
        "same_verdict": figures(agreement.same_verdict),
    }
