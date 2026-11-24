"""The first judged criterion: is each statement backed by the passages it cites?

The judge is given an answer cut into numbered statements, with the passages those
statements cite, and rules on each statement (:mod:`handbook_rag.evals.judge`). This
module builds what it reads, from the saved replies of a run and from labelled answers,
and describes the criterion to the code that all judged criteria share
(:mod:`handbook_rag.evals.judging`).
"""

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from handbook_rag.evals.answers import AnswerRecord
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.judge import (
    SUPPORT_INSTRUCTIONS,
    VERDICTS,
    GeminiSupportJudge,
    JudgedPassage,
    JudgedStatement,
    SupportCase,
    request_fingerprint,
)
from handbook_rag.evals.judge_v2 import (
    SUPPORT_INSTRUCTIONS_V2,
    GeminiSupportJudgeV2,
    request_fingerprint_v2,
)
from handbook_rag.evals.judging import UNCLEAR, Criterion, LabelledCase
from handbook_rag.ingest.models import Chunk


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


def load_labelled_cases(path: Path, chunks: Mapping[str, Chunk]) -> list[LabelledCase[SupportCase]]:
    """Read labelled answers, and fetch the text of their passages from the corpus.

    The file names each passage by its chunk id and keeps a hash of its text. A label is
    about exact words: if the corpus no longer holds a passage, or holds other words under
    its id, the label can no longer be trusted, and loading fails.
    """
    if not path.is_file():
        raise DatasetError(f"labelled answers not found: {path}")
    cases: list[LabelledCase[SupportCase]] = []
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
                if passage_hash(chunk.text) != passage["text_sha256"]:
                    raise DatasetError(
                        f"{path}:{number}: the text of passage {passage['chunk_id']} has "
                        f"changed since {case_id} was labelled."
                    )
                passages.append(
                    JudgedPassage(_whole(passage["number"]), chunk.context_header, chunk.text)
                )
            statements = tuple(
                JudgedStatement(_whole(s["number"]), str(s["text"]), _numbers(s["cites"]))
                for s in row["statements"]
            )
            expected = {_whole(s["number"]): str(s["expected"]) for s in row["statements"]}
            planted = {
                _whole(s["number"]): str(s["planted"]) for s in row["statements"] if "planted" in s
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


def _whole(value: object) -> int:
    """A number of a statement or a passage as it must be written: a whole number.

    ``int()`` would turn 1.9 and ``true`` into 1 without a word.
    """
    if type(value) is not int:
        raise TypeError(f"a number must be a whole number, not {value!r}")
    return value


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


def passage_hash(text: str) -> str:
    """The short hash by which a committed file names the exact words of a passage.

    Committed files that need a whole passage do not copy it. They name it by its chunk
    id and keep this hash, and the text is read from the corpus when the file is loaded.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


SUPPORT: Criterion[SupportCase] = Criterion(
    name="support",
    metric="supported",
    title="Support of answers",
    unit="statements",
    verdicts=VERDICTS,
    faults=frozenset({"unsupported"}),
    verdict_words={
        "supported": "statements supported",
        "unsupported": "unsupported",
        "no_claim": "without a claim",
    },
    faults_phrase="statements labelled unsupported",
    passed_word="supported",
    failed_word="UNSUPPORTED",
    instructions=SUPPORT_INSTRUCTIONS,
    request_fingerprint=request_fingerprint,
    cases=support_cases,
    load_labelled=load_labelled_cases,
    judge=GeminiSupportJudge,
    reads_passages=True,
    labelled_path=Path("evals/judge/support/cases.jsonl"),
    checks_dir=Path("evals/runs/judge"),
    rows_file="support.jsonl",
    state_file="support.json",
    check_file="support_check.json",
)

SUPPORT_V2: Criterion[SupportCase] = dataclasses.replace(
    SUPPORT,
    version=2,
    instructions=SUPPORT_INSTRUCTIONS_V2,
    request_fingerprint=request_fingerprint_v2,
    judge=GeminiSupportJudgeV2,
)
"""The same criterion with version 2 of its judge (:mod:`handbook_rag.evals.judge_v2`).

Only the judge differs: what it is told, the shape of its reply, and so the settings that
are recorded with its verdicts. It rules on the same answers, gives the same verdicts and
saves into the same files. The settings keep the two apart: a run that holds the verdicts
of one version is not continued by the other.

Version 1 stays the judge a command means when it names no version. Every saved run was
made by it.
"""
