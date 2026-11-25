"""How often the assistant's answers leave out a limit: the 40 questions of the limit set.

The labelled answers in ``evals/judge/limits/`` were written to check a judge (ADR 0023).
Each of their 40 passages states a rule with a limit on it: who the rule is for, where or
when it holds, how much, on what condition, with what exception. For each passage, the
answer of the kind ``dropped`` was written to a question that does not name the limit,
so an answer to that question keeps the limit only if it states it.

Here the assistant itself is asked those 40 questions, each with its one passage, as the
adversarial command gives saved passages: there is no search, so only the model and its
instructions decide the reply. Each answer is then judged by version 2 of the support
judge, the version written to look for a limit left out. An answer counts as *faulted*
when the judge calls one of its statements unsupported, for whatever reason it gives;
each reason is kept with the answer, so a reader can see whether it is a limit.

What the share can carry: the questions were chosen because their passages state a limit,
so the share is a share on this set and not on the questions people ask. The judge is
wrong sometimes, by its check on the labelled answers. So the figure is for comparing two
runs on the same 40 questions, such as the old and the new answering instructions, and
for little else.
"""

import dataclasses
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from handbook_rag.evals.answers import AnswerRecord, AnswerRun, load_records, record_reply
from handbook_rag.evals.careful import run_carefully
from handbook_rag.evals.dataset import DatasetError
from handbook_rag.evals.judge import Judgement, SupportCase
from handbook_rag.evals.metrics import Estimate, share_with_interval, sign_test
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.stage import Stage, write_json
from handbook_rag.evals.support import SUPPORT_V2, passage_hash, support_case
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk

CASES_PATH = Path("evals/judge/limits/cases.jsonl")
"""The labelled answers whose questions are asked."""
JUDGE = SUPPORT_V2
"""The judge of these answers: version 2 of the support judge, which keeps its verdicts in
files of its own (``support-v2.jsonl``) beside the replies."""
ANSWERS_FILE = "answers.jsonl"
RUN_FILE = "run.json"
RESULTS_FILE = "results.jsonl"
SUMMARY_FILE = "summary.json"
FAULT = "unsupported"
"""The verdict that makes an answer faulted."""
ENDINGS = frozenset({"end_turn", "max_tokens"})
"""The stop reasons of a reply that came to a proper end."""

GOOD, FAULTED = "good", "faulted"
DECLINED_BY_THE_MODEL = "declined_by_the_model"
DECLINED_IN_THE_GAP_FORM = "declined_in_the_gap_form"
DECLINED_BY_A_CHECK = "declined_by_a_check"
UNFINISHED = "unfinished"
NOTHING_TO_JUDGE, WAITING = "nothing_to_judge", "waiting_for_the_judge"
OUTCOMES = (
    GOOD,
    FAULTED,
    DECLINED_BY_THE_MODEL,
    DECLINED_IN_THE_GAP_FORM,
    DECLINED_BY_A_CHECK,
    UNFINISHED,
    NOTHING_TO_JUDGE,
    WAITING,
)
"""What can become of a question with a reply, in the order they are reported."""


@dataclass(frozen=True, slots=True)
class LimitQuestion:
    """The question of one passage of the limit set, with that passage and its limit."""

    case_id: str
    """The id of the labelled ``dropped`` answer whose question this is."""
    question: str
    chunk_id: str
    text_sha256: str
    """The first 16 hexadecimal digits of the SHA-256 hash of the passage's text."""
    section: str
    limit_kind: str
    limit: str
    """The limit in the passage's own words."""
    limit_where: str
    """``text`` or ``title``: where the passage states the limit."""

    @property
    def fingerprint(self) -> str:
        """A short hash of what the model is sent: the question and its passage.

        The limit and its kind decide only how a reply is reported, so a change to them
        does not ask the question again.
        """
        content = [self.question, self.chunk_id, self.text_sha256]
        encoded = json.dumps(content, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def load_limit_questions(path: Path) -> list[LimitQuestion]:
    """The question of each passage's ``dropped`` answer, in the order of the file.

    That answer's question was written not to name the limit (the set's card), so an
    answer keeps the limit only by stating it. A passage must have exactly one such
    answer, and the answer one passage: anything else is refused, since one passage asked
    twice, or not at all, would not be the set the figure is said to be of.
    """
    if not path.is_file():
        raise DatasetError(f"limit answers not found: {path}")
    questions: list[LimitQuestion] = []
    dropped: dict[str, list[str]] = defaultdict(list)
    passages: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("a labelled answer is a JSON object")
            given = row["passages"]
            if not isinstance(given, list) or len(given) != 1:
                raise ValueError("an answer of the limit set cites one passage")
            (passage,) = given
            chunk_id = _text(passage["chunk_id"])
            passages.add(chunk_id)
            if _text(row["kind"]) != "dropped":
                continue
            question = LimitQuestion(
                case_id=_text(row["case"]),
                question=_text(row["question"]),
                chunk_id=chunk_id,
                text_sha256=_text(passage["text_sha256"]),
                section=_text(row["section"]),
                limit_kind=_text(row["limit_kind"]),
                limit=_text(row["limit"]),
                limit_where=_text(row["limit_where"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise DatasetError(f"{path}:{number}: not a labelled answer: {error}") from error
        dropped[chunk_id].append(question.case_id)
        questions.append(question)
    for chunk_id in sorted(passages):
        if len(dropped[chunk_id]) != 1:
            raise DatasetError(
                f"{path}: passage {chunk_id} has {len(dropped[chunk_id])} answers of the kind "
                "dropped, not one"
            )
    if not questions:
        raise DatasetError(f"{path} holds no answer of the kind dropped")
    return questions


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"expected text, not {value!r}")
    return value


def passages_of(
    questions: Sequence[LimitQuestion], chunks: Mapping[str, Chunk]
) -> dict[str, Chunk]:
    """The passage of each question, by the question's id, checked against its hash.

    The labels were made for exact words. A passage the corpus no longer holds, or holds
    in other words, is refused before a request is spent on it.
    """
    found: dict[str, Chunk] = {}
    for question in questions:
        chunk = chunks.get(question.chunk_id)
        if chunk is None:
            raise DatasetError(
                f"{question.case_id} names passage {question.chunk_id}, which is not in the "
                "corpus. The set was made for another snapshot or chunking."
            )
        if passage_hash(chunk.text) != question.text_sha256:
            raise DatasetError(
                f"the text of passage {question.chunk_id} has changed since {question.case_id} "
                "was written."
            )
        found[question.case_id] = chunk
    return found


def asker(generator: Generator, passages: Mapping[str, Chunk]) -> Callable[[LimitQuestion], Answer]:
    """How one question is put to the assistant: with its passage, and no search."""

    def ask(question: LimitQuestion) -> Answer:
        chunk = passages[question.case_id]
        answer = generator.generate(question.question, [chunk])
        # The pipeline records the passages it gave the model; here we do.
        return _given(answer, chunk)

    return ask


def _given(answer: Answer, chunk: Chunk) -> Answer:
    return dataclasses.replace(answer, retrieved=(ScoredChunk(chunk, 0.0),))


def record_limit_reply(question: LimitQuestion, answer: Answer) -> AnswerRecord:
    """Keep a reply in the form the answer evaluation keeps its own. Its one passage holds
    the limit, which stands for the evidence of a golden question."""
    return record_reply(question.case_id, question.fingerprint, answer, covers=lambda _chunk: (0,))


def replies_to_current_questions(
    questions: Sequence[LimitQuestion], records: Mapping[str, AnswerRecord]
) -> dict[str, AnswerRecord]:
    """The saved replies to the questions as they are written now, by the question's id."""
    return {
        question.case_id: records[question.case_id]
        for question in questions
        if question.case_id in records
        and records[question.case_id].fingerprint == question.fingerprint
    }


def in_run_order(questions: Sequence[LimitQuestion]) -> list[LimitQuestion]:
    """A fixed order that mixes the kinds of limit, so a run that stops early is fair."""
    return sorted(questions, key=lambda q: hashlib.sha256(q.case_id.encode("utf-8")).hexdigest())


def run_questions(
    questions: Sequence[LimitQuestion],
    ask: Callable[[LimitQuestion], Answer],
    *,
    saved: Mapping[str, AnswerRecord] | None = None,
    wait: Callable[[], None] | None = None,
    on_record: Callable[[LimitQuestion, AnswerRecord], None] | None = None,
    on_failure: Callable[[FailedQuestion], None] | None = None,
    rest: Callable[[], None] | None = None,
    max_attempts: int | None = None,
    failures_before_stopping: int = 3,
) -> AnswerRun:
    """Ask every question that has no saved reply to it as it is written now.

    Requests are spent as in :func:`handbook_rag.evals.careful.run_carefully`, as every
    other stage spends them.
    """
    run = AnswerRun(records=replies_to_current_questions(questions, saved or {}))
    pending = [q for q in in_run_order(questions) if q.case_id not in run.records]

    def keep(question: LimitQuestion, answer: Answer) -> None:
        record = record_limit_reply(question, answer)
        run.records[question.case_id] = record
        if on_record is not None:
            on_record(question, record)

    outcome = run_carefully(
        pending,
        ask,
        keep,
        name=lambda question: question.case_id,
        errors=(GenerationError,),
        wait=wait,
        on_failure=on_failure,
        rest=rest,
        max_attempts=max_attempts,
        failures_before_stopping=failures_before_stopping,
    )
    run.attempts, run.failures, run.stopped = outcome.attempts, outcome.failures, outcome.stopped
    return run


def answers_stage(run_dir: Path) -> Stage:
    """The files in which a run keeps its replies. The judge's are another stage."""
    return Stage(
        run_dir,
        ANSWERS_FILE,
        RUN_FILE,
        holds="replies",
        outputs=(RESULTS_FILE, SUMMARY_FILE),
    )


def load_replies(run_dir: Path) -> dict[str, AnswerRecord]:
    """The replies a run has saved, by the question's id."""
    return load_records(answers_stage(run_dir).rows_path)


def judged_cases(
    questions: Sequence[LimitQuestion],
    records: Mapping[str, AnswerRecord],
    chunks: Mapping[str, Chunk],
) -> list[SupportCase]:
    """The answers as the judge reads them. A reply that declined has nothing to judge."""
    current = replies_to_current_questions(questions, records)
    cases: list[SupportCase] = []
    for question in questions:
        record = current.get(question.case_id)
        if record is None:
            continue
        case = support_case(question.case_id, question.question, record, chunks)
        if case is not None:
            cases.append(case)
    return cases


@dataclass(frozen=True, slots=True)
class LimitResult:
    """One question, the reply to it, and the judge's verdicts if it has given them."""

    question: LimitQuestion
    record: AnswerRecord
    case: SupportCase | None
    """What the judge reads, or ``None`` for a reply with nothing to judge."""
    judgement: Judgement | None

    @property
    def faulted(self) -> bool | None:
        """Whether the judge called a statement unsupported; ``None`` until it has judged."""
        if self.judgement is None:
            return None
        return any(verdict.verdict == FAULT for verdict in self.judgement.verdicts)

    @property
    def outcome(self) -> str:
        """What became of the question, one of :data:`OUTCOMES`.

        ``good`` is an answer in which the judge faulted no statement: what the figure that
        counts counts. A question that got no answer is not good, whatever the cause, and
        the cause is named: the model declined, its reply was the fixed form for a missing
        part alone, a check declined its reply, or the reply did not come to a proper end.
        """
        record = self.record
        if not record.answered:
            actions = {check.action for check in record.checks}
            names = {check.name for check in record.checks}
            if record.model_text:
                return DECLINED_IN_THE_GAP_FORM
            if "declined" in actions:
                return DECLINED_BY_A_CHECK
            if record.stop_reason not in ENDINGS or "unfinished_reply" in names:
                return UNFINISHED
            return DECLINED_BY_THE_MODEL
        if self.case is None:
            return NOTHING_TO_JUDGE
        if self.judgement is None:
            return WAITING
        return FAULTED if self.faulted else GOOD


@dataclass(frozen=True, slots=True)
class LimitFigures:
    """The counts of all questions, or of one kind of limit."""

    questions: int
    replied: int
    answered: int
    judged: int
    faulted: int
    share_faulted: Estimate | None
    """Faulted answers among the answers judged, once every answer has a verdict. Not the
    headline: a question that got no answer drops out of it, so the instructions that
    decline more would look better by it."""
    good: int = 0
    """Answers with no statement faulted: the figure that counts."""
    share_good: Estimate | None = None
    """Answers with no statement faulted, out of all the questions, once final."""
    outcomes: dict[str, int] = field(default_factory=dict)
    """How many questions came to each of :data:`OUTCOMES`."""


@dataclass(frozen=True, slots=True)
class LimitReport:
    """Everything a run has found so far."""

    results: list[LimitResult]
    figures: dict[str, LimitFigures]
    """``all``, and each kind of limit by its name."""
    failures: list[FailedQuestion]
    not_asked: int
    waiting_for_the_judge: list[str] = field(default_factory=list)
    """The answers that have no verdict yet, by the question's id."""
    tokens: dict[str, float] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        """Every question has a reply, and every answer a verdict."""
        return not self.not_asked and not self.failures and not self.waiting_for_the_judge


def summarise_limits(
    questions: Sequence[LimitQuestion],
    records: Mapping[str, AnswerRecord],
    cases: Sequence[SupportCase],
    judgements: Mapping[str, Judgement],
    failures: Sequence[FailedQuestion] = (),
) -> LimitReport:
    """Put each reply beside its verdicts, and count, in all and by kind of limit.

    ``judgements`` are by the question's id, and only of exactly the answers in ``cases``.
    The share of faulted answers is given only once every question has a reply and every
    answer a verdict: before that it would be of the answers that happened to come first.
    """
    current = replies_to_current_questions(questions, records)
    by_id = {case.case_id: case for case in cases}
    results = [
        LimitResult(
            question,
            current[question.case_id],
            by_id.get(question.case_id),
            judgements.get(question.case_id),
        )
        for question in questions
        if question.case_id in current
    ]
    failed = [failure for failure in failures if failure.question_id not in current]
    waiting = [r.question.case_id for r in results if r.case is not None and r.judgement is None]
    complete = len(results) == len(questions) and not waiting

    def figures(members: Sequence[LimitResult], total: int) -> LimitFigures:
        answered = [r for r in members if r.record.answered]
        judged = [r for r in members if r.judgement is not None]
        faulted = sum(bool(r.faulted) for r in judged)
        good = sum(r.outcome == GOOD for r in members)
        return LimitFigures(
            questions=total,
            replied=len(members),
            answered=len(answered),
            judged=len(judged),
            faulted=faulted,
            share_faulted=(
                share_with_interval(faulted, len(judged)) if complete and judged else None
            ),
            good=good,
            share_good=share_with_interval(good, total) if complete and total else None,
            outcomes={o: sum(r.outcome == o for r in members) for o in OUTCOMES},
        )

    grouped = {"all": figures(results, len(questions))}
    for kind in sorted({question.limit_kind for question in questions}):
        grouped[kind] = figures(
            [r for r in results if r.question.limit_kind == kind],
            sum(question.limit_kind == kind for question in questions),
        )
    asked = [r.record for r in results if r.record.stop_reason != "no_context"]
    tokens: dict[str, float] = {}
    if asked:
        tokens = {
            "input_total": float(sum(record.input_tokens for record in asked)),
            "output_total": float(sum(record.output_tokens for record in asked)),
            "input_mean": statistics.fmean(record.input_tokens for record in asked),
            "output_mean": statistics.fmean(record.output_tokens for record in asked),
        }
    return LimitReport(
        results=results,
        figures=grouped,
        failures=failed,
        not_asked=len(questions) - len(results) - len(failed),
        waiting_for_the_judge=waiting,
        tokens=tokens,
    )


def write_limit_run(
    report: LimitReport,
    run_dir: Path,
    *,
    description: str,
    config: Mapping[str, object],
    attempts: int,
    judge_config: Mapping[str, object],
    judge_attempts: int,
) -> None:
    """Write one line for each question with a reply, and the counts with their settings."""
    with (run_dir / RESULTS_FILE).open("w", encoding="utf-8") as output:
        for result in report.results:
            question, record = result.question, result.record
            texts = (
                {} if result.case is None else {s.number: s.text for s in result.case.statements}
            )
            row = {
                "case": question.case_id,
                "section": question.section,
                "limit_kind": question.limit_kind,
                "limit": question.limit,
                "limit_where": question.limit_where,
                "question": question.question,
                "question_fingerprint": question.fingerprint,
                "answered": record.answered,
                "stop_reason": record.stop_reason,
                "cites": sorted(record.cited),
                "reply": record.text_with_markers,
                "checks": [asdict(check) for check in record.checks],
                "model_text": record.model_text,
                "judged": result.judgement is not None,
                "faulted": result.faulted,
                "outcome": result.outcome,
                "verdicts": (
                    []
                    if result.judgement is None
                    else [
                        {
                            "number": verdict.number,
                            "statement": texts.get(verdict.number, ""),
                            "verdict": verdict.verdict,
                            "reason": verdict.reason,
                        }
                        for verdict in result.judgement.verdicts
                    ]
                ),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "latency_s": round(record.seconds, 4),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        run_dir / SUMMARY_FILE,
        {
            "description": description,
            "config": dict(config),
            "judge": {"config": dict(judge_config), "attempts": judge_attempts},
            "questions": report.figures["all"].questions,
            "questions_replied": len(report.results),
            "questions_failed": [asdict(failure) for failure in report.failures],
            "questions_not_asked": report.not_asked,
            "attempts": attempts,
            "final": report.final,
            "waiting_for_the_judge": report.waiting_for_the_judge,
            "figures": {name: asdict(figures) for name, figures in report.figures.items()},
            "tokens": report.tokens,
        },
    )


@dataclass(frozen=True, slots=True)
class LimitComparison:
    """Two final runs of the limit questions, question by question."""

    better: list[str]
    """Questions good in the later run and not in the earlier one."""
    worse: list[str]
    same: int

    @property
    def sign_test(self) -> float:
        return sign_test(len(self.better), len(self.worse))


def compare_limit_runs(earlier: Path, later: Path) -> LimitComparison:
    """Compare two final runs by the figure that counts: a good answer or not, per question.

    Both runs must be final and of the same questions, or there is nothing to compare.
    """

    def outcomes(run_dir: Path) -> dict[str, str]:
        if not (run_dir / SUMMARY_FILE).is_file():
            raise DatasetError(f"there is no run of the limit questions in {run_dir}")
        summary = json.loads((run_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
        if not summary.get("final"):
            raise DatasetError(f"the run in {run_dir} is not final, so it cannot be compared")
        rows = (run_dir / RESULTS_FILE).read_text(encoding="utf-8").splitlines()
        return {row["case"]: row["outcome"] for row in map(json.loads, rows) if row}

    before, after = outcomes(earlier), outcomes(later)
    if set(before) != set(after):
        raise DatasetError(f"{earlier} and {later} hold replies to other questions")
    better = sorted(c for c in before if after[c] == GOOD and before[c] != GOOD)
    worse = sorted(c for c in before if before[c] == GOOD and after[c] != GOOD)
    return LimitComparison(better, worse, len(before) - len(better) - len(worse))
