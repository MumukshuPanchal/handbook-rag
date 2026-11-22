"""Judge saved replies: is each statement backed by the passages it cites?

This is the second stage of answer evaluation. It reads the replies a run has saved, sends
each answer to a judge, and saves the verdicts. It runs the judge over two kinds of
material with the same code:

- the answers of a run, to score them;
- labelled answers, to check the judge itself (see :mod:`handbook_rag.evals.agreement`).

A verdict is saved with a fingerprint of everything the judge read. A saved verdict is
used again only for exactly the same question, passages and statements, so judging the
labelled answers and scoring a run never pays twice for the same answer.
"""

import hashlib
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from handbook_rag.evals.agreement import REAL, Agreement, Pair
from handbook_rag.evals.answers import AnswerRecord
from handbook_rag.evals.careful import run_carefully
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.judge import (
    VERDICTS,
    JudgedPassage,
    JudgedStatement,
    JudgeError,
    Judgement,
    StatementVerdict,
    SupportCase,
    SupportJudge,
    Verdict,
)
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.stage import SavedRunError, Stage, load_rows
from handbook_rag.ingest.models import Chunk

UNCLEAR = "unclear"
"""The label of a statement on which careful readers did not agree. It is not counted."""


CHECK_FILE = "support_check.json"
"""Beside a run's verdicts: how far the judge that made them agreed with labelled answers."""


def support_stage(run_dir: Path) -> Stage:
    """The files in which an answer run keeps the judge's verdicts on its replies."""
    return Stage(
        run_dir,
        "support.jsonl",
        "support.json",
        holds="verdicts",
        advice="Discard the saved verdicts with --restart, or use the judge that made them.",
        outputs=(CHECK_FILE,),
    )


def judge_check_stage(run_dir: Path) -> Stage:
    """The files of a run that checks the judge against labelled answers."""
    return Stage(
        run_dir,
        "judgements.jsonl",
        "run.json",
        holds="verdicts",
        advice="Give this run another name, or discard its saved verdicts with --restart.",
        outputs=("summary.json", "disagreements.jsonl"),
    )


def support_cases(
    questions: Sequence[GoldenQuestion],
    records: Mapping[str, AnswerRecord],
    chunks: Mapping[str, Chunk],
) -> list[SupportCase]:
    """The answers of a run, as the judge reads them. A reply that declined is left out.

    A statement keeps the passage numbers the model used, and the judge is given the
    passages the answer cites and no others.
    """
    cases: list[SupportCase] = []
    for question in questions:
        record = records.get(question.id)
        if record is None or not record.answered:
            continue
        passages = []
        for passage in record.passages:
            if passage.number not in record.cited:
                continue
            chunk = chunks.get(passage.chunk_id)
            if chunk is None:
                raise DatasetError(
                    f"The reply to {question.id} cites passage {passage.chunk_id}, which is "
                    "not in the corpus. The run was made with another snapshot or chunking."
                )
            passages.append(JudgedPassage(passage.number, chunk.context_header, chunk.text))
        spoken = [s for s in record.statements if any(c.isalnum() for c in s.text)]
        if not spoken:
            # Nothing in the reply says anything, so there is nothing to judge.
            continue
        cases.append(
            SupportCase(
                case_id=question.id,
                question=question.question,
                passages=tuple(passages),
                statements=tuple(
                    JudgedStatement(number, statement.text.strip(), statement.cites)
                    for number, statement in enumerate(spoken, start=1)
                ),
            )
        )
    return cases


@dataclass(frozen=True, slots=True)
class LabelledCase:
    """An answer whose statements have known verdicts."""

    case: SupportCase
    kind: str
    """``real``, or the kind of fault planted in one of its statements."""
    expected: dict[int, str]
    """Statement number to its labelled verdict, or to ``unclear``."""
    planted: dict[int, str]
    """Statement number to the kind of fault planted in it."""


def load_labelled_cases(path: Path, chunks: Mapping[str, Chunk]) -> list[LabelledCase]:
    """Read labelled answers, and fetch the text of their passages from the corpus.

    The file names each passage by its chunk id and keeps a hash of its text. A label is
    about exact words: if the corpus no longer holds a passage, or holds other words under
    its id, the label can no longer be trusted, and loading fails.
    """
    if not path.is_file():
        raise DatasetError(f"labelled answers not found: {path}")
    cases: list[LabelledCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            case_id = str(row["case"])
            passages = []
            for passage in row["passages"]:
                chunk = chunks.get(passage["chunk_id"])
                if chunk is None:
                    raise DatasetError(
                        f"{path}:{number}: {case_id} names passage {passage['chunk_id']}, which "
                        "is not in the corpus. The labels were made for another snapshot or "
                        "chunking."
                    )
                if _text_hash(chunk.text) != passage["text_sha256"]:
                    raise DatasetError(
                        f"{path}:{number}: the text of passage {passage['chunk_id']} has "
                        f"changed since {case_id} was labelled."
                    )
                passages.append(
                    JudgedPassage(int(passage["number"]), chunk.context_header, chunk.text)
                )
            statements = tuple(
                JudgedStatement(int(s["number"]), str(s["text"]), _numbers(s["cites"]))
                for s in row["statements"]
            )
            expected = {int(s["number"]): str(s["expected"]) for s in row["statements"]}
            planted = {
                int(s["number"]): str(s["planted"]) for s in row["statements"] if "planted" in s
            }
            kind = str(row["kind"])
            question = str(row["question"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise DatasetError(f"{path}:{number}: not a labelled answer: {error}") from error
        unknown = sorted(set(expected.values()) - {*VERDICTS, UNCLEAR})
        if unknown:
            raise DatasetError(f"{path}:{number}: {case_id} has an unknown verdict: {unknown}")
        problem = _fault_in_case(passages, statements, expected, planted)
        if problem:
            raise DatasetError(f"{path}:{number}: {case_id} {problem}")
        if case_id in seen:
            raise DatasetError(f"{path}:{number}: the id {case_id} is used twice")
        seen.add(case_id)
        cases.append(
            LabelledCase(
                SupportCase(case_id, question, tuple(passages), statements),
                kind,
                expected,
                planted,
            )
        )
    if not cases:
        raise DatasetError(f"{path} holds no labelled answer")
    return cases


def _numbers(value: object) -> tuple[int, ...]:
    """Passage numbers as they must be written: a list of whole numbers."""
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise TypeError(f"cites must be a list of passage numbers, not {value!r}")
    return tuple(value)


def _fault_in_case(
    passages: Sequence[JudgedPassage],
    statements: Sequence[JudgedStatement],
    expected: Mapping[int, str],
    planted: Mapping[int, str],
) -> str | None:
    """What is wrong with a labelled answer that the judge could not be fairly given."""
    if not statements:
        return "has no statement"
    numbers = [statement.number for statement in statements]
    if len(set(numbers)) != len(numbers):
        return "numbers two statements alike"
    given = [passage.number for passage in passages]
    if len(set(given)) != len(given):
        return "numbers two passages alike"
    for statement in statements:
        if not set(statement.cites) <= set(given):
            return f"has statement {statement.number} citing a passage it does not list"
    for statement_number in planted:
        if expected[statement_number] != "unsupported":
            return f"has a planted fault in statement {statement_number} that is not unsupported"
    return None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def case_fingerprint(case: SupportCase) -> str:
    """A short hash of everything the judge reads for one answer."""
    content = [
        case.question,
        [[p.number, p.title, p.text] for p in case.passages],
        [[s.number, s.text, list(s.cites)] for s in case.statements],
    ]
    encoded = json.dumps(content, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def judgement_row(case: SupportCase, judgement: Judgement) -> dict[str, Any]:
    """A judgement as it is saved, with the fingerprint of what was judged."""
    return {
        "case": case.case_id,
        "fingerprint": case_fingerprint(case),
        "verdicts": [
            {"number": v.number, "verdict": v.verdict, "reason": v.reason}
            for v in judgement.verdicts
        ],
        "model": judgement.model,
        "input_tokens": judgement.input_tokens,
        "output_tokens": judgement.output_tokens,
        "seconds": round(judgement.seconds, 4),
    }


def load_judgements(path: Path) -> dict[str, Judgement]:
    """Read saved judgements, by the fingerprint of what was judged. The latest one wins."""
    judgements: dict[str, Judgement] = {}
    for number, row in load_rows(path):
        try:
            judgements[str(row["fingerprint"])] = Judgement(
                case_id=str(row["case"]),
                verdicts=tuple(
                    StatementVerdict(int(v["number"]), _verdict(v["verdict"]), str(v["reason"]))
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


def _verdict(value: object) -> Verdict:
    for verdict in VERDICTS:
        if value == verdict:
            return verdict
    raise ValueError(f"unknown verdict {value!r}")


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


def run_judge(
    cases: Sequence[SupportCase],
    judge: SupportJudge,
    *,
    saved: Mapping[str, Judgement] | None = None,
    wait: Callable[[], None] | None = None,
    on_judgement: Callable[[SupportCase, Judgement, bool], None] | None = None,
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
    is never read as "supported".
    """
    saved = saved or {}
    run = JudgeRun(judgements={})
    pending: list[SupportCase] = []
    for case in cases:
        earlier = saved.get(case_fingerprint(case))
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

    def keep(case: SupportCase, judgement: Judgement) -> None:
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


def labelled_pairs(
    cases: Sequence[LabelledCase], judgements: Mapping[str, Judgement]
) -> list[Pair]:
    """Each labelled statement beside the verdict the judge gave it.

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
                    expected=_verdict(expected),
                    judged=judged[number],
                    kind=labelled.planted.get(number, REAL),
                )
            )
    return pairs


def saved_support(run_dir: Path, cases: Sequence[SupportCase]) -> dict[str, Judgement]:
    """The verdicts a run has saved that are of exactly these answers, by case id.

    A verdict on an answer that has since been asked again, and so reads differently, is
    left out: it is a verdict on other words.
    """
    saved = load_judgements(support_stage(run_dir).rows_path)
    return {
        case.case_id: saved[fingerprint]
        for case in cases
        if (fingerprint := case_fingerprint(case)) in saved
    }


def judge_record(run_dir: Path, judged: Mapping[str, Judgement]) -> dict[str, object] | None:
    """Who judged a run's answers, what it cost, and how far that judge can be trusted.

    This goes into the run's summary beside the judged score. The agreement figures are
    included only if they were measured with the same judge: the same model and the same
    instructions.
    """
    if not judged:
        return None
    stage = support_stage(run_dir)
    config = stage.config()
    verdicts = [verdict.verdict for judgement in judged.values() for verdict in judgement.verdicts]
    record: dict[str, object] = {
        "config": config,
        "attempts": stage.attempts(),
        "answers_judged": len(judged),
        "statements": {name: verdicts.count(name) for name in VERDICTS},
        "input_tokens": sum(judgement.input_tokens for judgement in judged.values()),
        "output_tokens": sum(judgement.output_tokens for judgement in judged.values()),
        "median_seconds": statistics.median(j.seconds for j in judged.values()),
        "check": None,
    }
    check_file = run_dir / CHECK_FILE
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
