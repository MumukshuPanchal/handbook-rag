"""Run the answer evaluation: ask every question, keep each reply, score what code can check.

Retrieval metrics say whether the right passage was found. They say nothing about what
the assistant then wrote. This module puts every golden question through the same entry
point the command line uses, and scores the replies on what a program can decide:

- Did the assistant answer the questions the handbook answers, and decline the others?
- Does every statement in an answer name a passage?
- Does an answer name the passage that holds the known evidence?

Whether a statement is *true to* the passage it names needs a reader, not a program. A
judge measures that separately, from the replies this module saves.

Each question costs one real request to the answer model, and a free key allows a fixed
number a day. So a run is careful with requests: they are spaced out, each reply is saved
as soon as it arrives, a run that was stopped continues where it left off, and a run that
keeps failing stops itself.
"""

import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from handbook_rag.evals.dataset import EvidenceSpan, GoldenQuestion, locate_evidence
from handbook_rag.evals.metrics import Estimate, covered_evidence, mean_with_interval
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer
from handbook_rag.index.store import VectorStoreError
from handbook_rag.ingest.models import Document

# The headline comes first. Each score is 1 or 0 for one question, and a score that does
# not apply to a question is left out, so every mean says how many questions it rests on.
METRICS: list[dict[str, str]] = [
    {"id": "right_call", "label": "Right call", "kind": "binary"},
    {"id": "answered", "label": "Answers when it should", "kind": "binary"},
    {"id": "declined", "label": "Declines when it should", "kind": "binary"},
    {"id": "evidence_in_context", "label": "Evidence among the passages", "kind": "binary"},
    {"id": "fully_cited", "label": "Every statement cited", "kind": "binary"},
    {"id": "cites_evidence", "label": "Cites the evidence", "kind": "binary"},
]
METRIC_IDS = tuple(metric["id"] for metric in METRICS)

ANSWERS_FILE = "answers.jsonl"
RUN_FILE = "run.json"


class SavedRunError(ValueError):
    """A saved run cannot be continued: other settings made it, or its file cannot be read."""


@dataclass(frozen=True, slots=True)
class Passage:
    """One passage the model was given."""

    number: int
    """The number the model cites it by: its rank among the passages."""
    chunk_id: str
    doc_id: str
    context_header: str
    covers: tuple[int, ...]
    """Which pieces of the question's evidence this passage covers."""


@dataclass(frozen=True, slots=True)
class Statement:
    """A run of answer text, with the numbers of the passages it cites."""

    text: str
    cites: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """The assistant's reply to one golden question, kept so it can be scored again later.

    It holds what the model was given and what it wrote, by passage number. The text of
    the passages is not copied: a chunk id finds it again in the corpus.
    """

    question_id: str
    fingerprint: str
    """Of the question's label when it was asked. A reply to an older label is not reused."""
    passages: tuple[Passage, ...]
    statements: tuple[Statement, ...]
    answered: bool
    stop_reason: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    seconds_to_first_text: float | None = None

    @property
    def text(self) -> str:
        return "".join(statement.text for statement in self.statements)

    @property
    def text_with_markers(self) -> str:
        """The reply with the cited passage numbers after each statement, like [2]."""
        return "".join(
            statement.text + "".join(f"[{number}]" for number in statement.cites)
            for statement in self.statements
        )

    @property
    def cited(self) -> frozenset[int]:
        """The numbers of every passage the reply cites."""
        return frozenset(number for statement in self.statements for number in statement.cites)

    @property
    def uncited_statements(self) -> tuple[str, ...]:
        """Statements that say something and name no passage."""
        return tuple(
            statement.text.strip()
            for statement in self.statements
            if not statement.cites and any(character.isalnum() for character in statement.text)
        )


def record_answer(
    question: GoldenQuestion, answer: Answer, spans: Sequence[EvidenceSpan]
) -> AnswerRecord:
    """Reduce a reply to what scoring needs, with citations as passage numbers."""
    numbers = {
        scored.chunk.chunk_id: number for number, scored in enumerate(answer.retrieved, start=1)
    }
    return AnswerRecord(
        question_id=question.id,
        fingerprint=question.fingerprint,
        passages=tuple(
            Passage(
                number=number,
                chunk_id=scored.chunk.chunk_id,
                doc_id=scored.chunk.doc_id,
                context_header=scored.chunk.context_header,
                covers=tuple(sorted(covered_evidence(scored.chunk, spans))),
            )
            for number, scored in enumerate(answer.retrieved, start=1)
        ),
        statements=tuple(
            Statement(
                text=segment.text,
                cites=tuple(sorted({numbers[c.chunk.chunk_id] for c in segment.citations})),
            )
            for segment in answer.segments
        ),
        answered=answer.answered,
        stop_reason=answer.stop_reason,
        model=answer.model,
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        seconds=answer.seconds,
        seconds_to_first_text=answer.seconds_to_first_text,
    )


def score_answer(question: GoldenQuestion, record: AnswerRecord) -> dict[str, float]:
    """The scores that apply to one reply. A score that does not apply is left out.

    - ``right_call``: the assistant answered a question the handbook answers, or declined
      one it does not. It is judged from outside: declining an answerable question is a
      wrong call even when the search is to blame for it.
    - ``answered`` and ``declined`` are the same score for each kind of question.
    - ``evidence_in_context``: a passage the model was given covers the known evidence.
      It is the search's Hit@k for this run, kept here to tell the two causes apart.
    - ``fully_cited``, for an answer: every statement names at least one passage.
    - ``cites_evidence``, for an answer that was given the evidence: a passage it names
      covers that evidence.
    """
    scores: dict[str, float] = {}
    if question.type == "unanswerable":
        scores["declined"] = float(not record.answered)
        scores["right_call"] = scores["declined"]
    else:
        scores["answered"] = float(record.answered)
        scores["right_call"] = scores["answered"]
        with_evidence = {passage.number for passage in record.passages if passage.covers}
        scores["evidence_in_context"] = float(bool(with_evidence))
        if record.answered and with_evidence:
            scores["cites_evidence"] = float(bool(record.cited & with_evidence))
    if record.answered:
        scores["fully_cited"] = float(not record.uncited_statements)
    return scores


class Pacer:
    """Keeps requests a fixed time apart, so a run stays under a limit per minute."""

    def __init__(
        self,
        per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("A pace needs a positive number of requests per minute.")
        self._gap = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Return when the next request may start. The first one starts at once."""
        if self._last is not None:
            remaining = self._last + self._gap - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


@dataclass(slots=True)
class AnswerRun:
    """What one invocation did."""

    records: dict[str, AnswerRecord]
    """Every reply the run holds, by question id: the saved ones and the new ones."""
    failures: list[FailedQuestion] = field(default_factory=list)
    """Questions that were asked and got no reply, after one more try."""
    attempts: int = 0
    """Questions put to the pipeline this time. Each sends at most one request."""
    stopped: str | None = None
    """Why the run ended before every question was asked, if it did."""


def in_run_order(questions: Sequence[GoldenQuestion]) -> list[GoldenQuestion]:
    """A fixed order that mixes the kinds of question.

    The file lists the answerable questions first. Asked in that order, a run that stops
    early would hold no unanswerable question at all. Sorting by a hash of the id gives
    the same order every time, and any first part of it is a fair sample.
    """
    return sorted(questions, key=lambda q: hashlib.sha256(q.id.encode("utf-8")).hexdigest())


def run_answers(
    questions: Sequence[GoldenQuestion],
    documents: Sequence[Document],
    ask: Callable[[str], Answer],
    *,
    saved: Mapping[str, AnswerRecord] | None = None,
    wait: Callable[[], None] | None = None,
    on_record: Callable[[GoldenQuestion, AnswerRecord], None] | None = None,
    max_attempts: int | None = None,
    failures_before_stopping: int = 3,
) -> AnswerRun:
    """Ask every question that has no saved reply yet.

    ``ask`` answers one question through the whole pipeline. ``wait`` is called before
    each question and returns when the next request may start. ``on_record`` receives each
    new reply as soon as it arrives, which is where it is saved.

    A question that fails, because the model or the vector store could not be reached, is
    asked once more after all the others. A failure is never scored as a wrong answer.
    When several questions fail one after another, the run stops: the cause is then
    almost certainly the day's allowance, the network or the key, and each further
    question would only spend a request to learn the same thing.

    A broken label, such as a quote that is not in its page, stops the run before the
    first request.
    """
    pages = {document.doc_id: document for document in documents}
    spans = {question.id: locate_evidence(question, pages) for question in questions}
    saved = saved or {}
    run = AnswerRun(
        records={
            question.id: saved[question.id]
            for question in questions
            if question.id in saved and saved[question.id].fingerprint == question.fingerprint
        }
    )
    pending = [question for question in in_run_order(questions) if question.id not in run.records]
    failed: dict[str, FailedQuestion] = {}
    failures_in_a_row = 0
    # The second round gives each question that failed one more try.
    for _ in range(2):
        again: list[GoldenQuestion] = []
        for question in pending:
            if max_attempts is not None and run.attempts >= max_attempts:
                run.stopped = f"the limit of {max_attempts} requests for this run was reached"
                break
            if wait is not None:
                wait()
            run.attempts += 1
            try:
                answer = ask(question.question)
            except (GenerationError, VectorStoreError) as error:
                failed[question.id] = FailedQuestion(question.id, type(error).__name__, str(error))
                again.append(question)
                failures_in_a_row += 1
                if failures_in_a_row >= failures_before_stopping:
                    run.stopped = (
                        f"{failures_in_a_row} questions in a row failed. The last error: {error}"
                    )
                    break
                continue
            failures_in_a_row = 0
            failed.pop(question.id, None)
            record = record_answer(question, answer, spans[question.id])
            run.records[question.id] = record
            if on_record is not None:
                on_record(question, record)
        if run.stopped is not None or not again:
            break
        pending = again
    run.failures = list(failed.values())
    return run


def open_run(
    run_dir: Path, config: Mapping[str, object], *, restart: bool = False
) -> dict[str, AnswerRecord]:
    """Start a run, or pick up one that was stopped. Returns the replies already saved.

    A run is one configuration: one model, one prompt, one way of searching. Replies made
    with another configuration are never mixed in. The caller chooses: a new name for the
    new configuration, or ``restart`` to discard what the run holds.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file, answers_file = run_dir / RUN_FILE, run_dir / ANSWERS_FILE
    wanted = json.loads(json.dumps(dict(config)))
    if restart:
        answers_file.unlink(missing_ok=True)
        run_file.unlink(missing_ok=True)
    if run_file.is_file():
        stored = json.loads(run_file.read_text(encoding="utf-8")).get("config", {})
        if stored != wanted:
            changed = sorted(
                key for key in {*stored, *wanted} if stored.get(key) != wanted.get(key)
            )
            raise SavedRunError(
                f"the run in {run_dir} was made with other settings ({', '.join(changed)}). "
                "Give this run another name, or discard its saved replies with --restart."
            )
    elif answers_file.is_file():
        raise SavedRunError(
            f"{answers_file} holds replies, but nothing records the settings that made them. "
            "Give this run another name, or discard them with --restart."
        )
    else:
        _write_json(run_file, {"config": wanted, "attempts": 0})
    return load_records(answers_file)


def add_attempts(run_dir: Path, attempts: int) -> int:
    """Add this invocation's attempts to the run's count, and return the new total."""
    run_file = run_dir / RUN_FILE
    state = json.loads(run_file.read_text(encoding="utf-8"))
    state["attempts"] = int(state.get("attempts", 0)) + attempts
    _write_json(run_file, state)
    return int(state["attempts"])


def append_record(run_dir: Path, record: AnswerRecord) -> None:
    """Save one reply at once, so that a run that stops loses nothing it paid for."""
    with (run_dir / ANSWERS_FILE).open("a", encoding="utf-8") as output:
        output.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_records(path: Path) -> dict[str, AnswerRecord]:
    """Read saved replies. The latest reply to a question wins.

    A run can be killed while it writes. Its last line may then be cut short, and is
    dropped: that question is simply asked again. A broken line anywhere else means the
    file was edited or damaged, which is reported.
    """
    if not path.is_file():
        return {}
    records: dict[str, AnswerRecord] = {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines):
                break
            raise SavedRunError(f"{path}:{number} cannot be read: {error}") from error
        try:
            record = AnswerRecord(
                **{
                    **row,
                    "passages": tuple(
                        Passage(**{**passage, "covers": tuple(passage["covers"])})
                        for passage in row["passages"]
                    ),
                    "statements": tuple(
                        Statement(statement["text"], tuple(statement["cites"]))
                        for statement in row["statements"]
                    ),
                }
            )
        except (KeyError, TypeError) as error:
            raise SavedRunError(f"{path}:{number} is not a saved reply: {error}") from error
        records[record.question_id] = record
    return records


@dataclass(frozen=True, slots=True)
class ScoredAnswer:
    """One question, the reply to it, and the scores that apply."""

    question: GoldenQuestion
    record: AnswerRecord
    scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class AnswerReport:
    """Everything a run has measured so far."""

    scored: list[ScoredAnswer]
    failures: list[FailedQuestion]
    not_asked: int
    summary: dict[str, Estimate]
    by_group: dict[str, dict[str, dict[str, Estimate]]]
    timing: dict[str, float]
    tokens: dict[str, float]


def summarise_answers(
    questions: Sequence[GoldenQuestion],
    records: Mapping[str, AnswerRecord],
    failures: Sequence[FailedQuestion] = (),
) -> AnswerReport:
    """Score every reply and average the scores, overall and for the groups worth reading."""
    scored = [
        ScoredAnswer(question, records[question.id], score_answer(question, records[question.id]))
        for question in questions
        if question.id in records
    ]
    failed = [failure for failure in failures if failure.question_id not in records]
    # A reply made without any passage never reached the model, so it has no timing.
    asked = [item.record for item in scored if item.record.stop_reason != "no_context"]
    timing: dict[str, float] = {}
    tokens: dict[str, float] = {}
    if asked:
        seconds = sorted(record.seconds for record in asked)
        timing["median_seconds"] = statistics.median(seconds)
        # The value that 95% of the replies were at or under: 95% of the count, rounded up.
        timing["p95_seconds"] = seconds[-(-95 * len(seconds) // 100) - 1]
        first = [r.seconds_to_first_text for r in asked if r.seconds_to_first_text is not None]
        if first:
            timing["median_seconds_to_first_text"] = statistics.median(first)
        tokens = {
            "input_total": float(sum(record.input_tokens for record in asked)),
            "output_total": float(sum(record.output_tokens for record in asked)),
            "input_mean": statistics.fmean(record.input_tokens for record in asked),
            "output_mean": statistics.fmean(record.output_tokens for record in asked),
        }
    return AnswerReport(
        scored=scored,
        failures=failed,
        not_asked=len(questions) - len(scored) - len(failed),
        summary=_summarise(scored),
        by_group={
            "origin": _summarise_groups(scored, lambda item: item.question.origin),
            "unanswerable_kind": _summarise_groups(
                scored, lambda item: item.question.unanswerable_kind
            ),
            "evidence": _summarise_groups(scored, _evidence_group),
        },
        timing=timing,
        tokens=tokens,
    )


def _evidence_group(item: ScoredAnswer) -> str | None:
    """Whether an answerable question's evidence was among the passages."""
    if "evidence_in_context" not in item.scores:
        return None
    return "given" if item.scores["evidence_in_context"] else "not given"


def _summarise(scored: Sequence[ScoredAnswer]) -> dict[str, Estimate]:
    summary: dict[str, Estimate] = {}
    for metric in METRIC_IDS:
        values = [item.scores[metric] for item in scored if metric in item.scores]
        if values:
            summary[metric] = mean_with_interval(values)
    return summary


def _summarise_groups(
    scored: Sequence[ScoredAnswer], key: Callable[[ScoredAnswer], str | None]
) -> dict[str, dict[str, Estimate]]:
    groups: dict[str, list[ScoredAnswer]] = defaultdict(list)
    for item in scored:
        name = key(item)
        if name is not None:
            groups[name].append(item)
    return {name: _summarise(members) for name, members in sorted(groups.items())}


def write_answer_run(
    report: AnswerReport,
    flow_dir: Path,
    variant: str,
    description: str,
    config: Mapping[str, object],
    *,
    attempts: int,
) -> Path:
    """Write the scores of one run next to its saved replies, and return its directory.

    The layout matches a retrieval run: ``results.jsonl`` has one row per question and is
    small enough to commit, ``summary.json`` has the means with what produced them, and
    ``traces/`` has one readable file per question for the machine that ran it.
    """
    run_dir = flow_dir / variant
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    _write_json(
        flow_dir / "_state.json",
        {"metrics": METRICS, "perf_fields": [{"id": "latency_s", "unit": "s"}]},
    )
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for item in report.scored:
            question, record = item.question, item.record
            row = {
                "prompt_id": question.id,
                "prompt": question.question,
                "fingerprint": question.fingerprint,
                "tags": [question.type, question.origin, f"review:{question.review}"],
                "status": "ok",
                "grade": item.scores,
                "latency_s": round(record.seconds, 4),
                "answered": record.answered,
                "stop_reason": record.stop_reason,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            _write_json(run_dir / "traces" / f"{question.id}_rep0.json", _trace(item))

    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as output:
        for failure in report.failures:
            output.write(json.dumps(asdict(failure), ensure_ascii=False) + "\n")

    labels = sorted(f"{item.question.id}:{item.question.fingerprint}" for item in report.scored)
    summary = {
        "description": description,
        "config": dict(config),
        "labels_fingerprint": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()[:16],
        "questions_scored": len(report.scored),
        "questions_failed": len(report.failures),
        "questions_not_asked": report.not_asked,
        "attempts": attempts,
        "metrics": {name: asdict(estimate) for name, estimate in report.summary.items()},
        "by_group": {
            group: {
                name: {metric: asdict(estimate) for metric, estimate in estimates.items()}
                for name, estimates in members.items()
            }
            for group, members in report.by_group.items()
        },
        "timing": report.timing,
        "tokens": report.tokens,
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "change.md").write_text(f"{description}\n", encoding="utf-8")
    return run_dir


def _trace(item: ScoredAnswer) -> list[dict[str, str]]:
    """One question as a short conversation, for reading by eye."""
    question, record = item.question, item.record
    expected = (
        f"Reference answer: {question.reference_answer}"
        if question.type == "answerable"
        else f"The handbook does not answer this ({question.unanswerable_kind}). {question.note}"
    )
    return [
        {"role": "user", "content": question.question},
        {
            "role": "tool_result",
            "content": "\n".join(
                f"[{passage.number}] {'EVIDENCE' if passage.covers else 'other   '}  "
                f"{passage.context_header}  ({passage.doc_id})"
                for passage in record.passages
            ),
        },
        {"role": "assistant", "content": record.text_with_markers},
        {"role": "user", "content": f"{expected}\nScores: {json.dumps(item.scores)}"},
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
