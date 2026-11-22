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
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from handbook_rag.evals.careful import run_carefully
from handbook_rag.evals.dataset import EvidenceSpan, GoldenQuestion, locate_evidence
from handbook_rag.evals.metrics import Estimate, covered_evidence, mean_with_interval
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.stage import (
    SavedRunError,
    Stage,
    clear_traces,
    load_rows,
    write_json,
)
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer
from handbook_rag.index.store import VectorStoreError
from handbook_rag.ingest.models import Document

# The headline comes first. Each score is 1 or 0 for one question, and a score that does
# not apply to a question is left out, so every mean says how many questions it rests on.
METRICS: list[dict[str, str]] = [
    {"id": "right_call", "label": "Right call", "kind": "binary"},
    {"id": "supported", "label": "Supported by its sources", "kind": "binary"},
    {"id": "answered", "label": "Answers when it should", "kind": "binary"},
    {"id": "declined", "label": "Declines when it should", "kind": "binary"},
    {"id": "evidence_in_context", "label": "Evidence among the passages", "kind": "binary"},
    {"id": "fully_cited", "label": "Every statement cited", "kind": "binary"},
    {"id": "cites_evidence", "label": "Cites the evidence", "kind": "binary"},
]
METRIC_IDS = tuple(metric["id"] for metric in METRICS)

ANSWERS_FILE = "answers.jsonl"
RUN_FILE = "run.json"


class ReplyError(ValueError):
    """A reply breaks a rule the generator must keep, so it cannot be recorded."""


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
    the passages is not copied: a chunk id finds it again in the corpus. The statements
    are the model's own words, which may repeat a few of a passage's.
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
    for segment in answer.segments:
        for citation in segment.citations:
            if citation.chunk.chunk_id not in numbers:
                raise ReplyError(
                    f"The reply to {question.id} cites a passage the model was not given "
                    f"({citation.chunk.chunk_id}). That is a fault in the generator."
                )
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


@dataclass(slots=True)
class AnswerRun:
    """What one invocation did."""

    records: dict[str, AnswerRecord]
    """Every reply the run holds, by question id: the saved ones and the new ones."""
    failures: list[FailedQuestion] = field(default_factory=list)
    """Questions that were asked and got no reply, after one more try."""
    attempts: int = 0
    """Questions put to the pipeline this time. Each sends at most one request: none
    when the search fails or finds nothing, because the model is then never called."""
    stopped: str | None = None
    """Why the run ended before every question was asked, if it did."""


def replies_to_current_labels(
    questions: Sequence[GoldenQuestion], records: Mapping[str, AnswerRecord]
) -> dict[str, AnswerRecord]:
    """The saved replies that were made for the questions as they are labelled now.

    A reply to a question whose wording, reference answer or evidence has since changed
    is a reply to another test. It is left out, here and wherever replies are scored.
    """
    return {
        question.id: records[question.id]
        for question in questions
        if question.id in records and records[question.id].fingerprint == question.fingerprint
    }


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
    on_failure: Callable[[FailedQuestion], None] | None = None,
    rest: Callable[[], None] | None = None,
    max_attempts: int | None = None,
    failures_before_stopping: int = 3,
) -> AnswerRun:
    """Ask every question that has no saved reply yet.

    ``ask`` answers one question through the whole pipeline. ``wait`` is called before
    each question and returns when the next request may start: it is where the pace is
    kept and the attempt is counted. ``on_record`` receives each new reply as soon as it
    arrives, which is where it is saved. ``on_failure`` receives each failure as it
    happens, including one that a second try then makes good.

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
    run = AnswerRun(records=replies_to_current_labels(questions, saved or {}))
    pending = [question for question in in_run_order(questions) if question.id not in run.records]

    def keep(question: GoldenQuestion, answer: Answer) -> None:
        record = record_answer(question, answer, spans[question.id])
        run.records[question.id] = record
        if on_record is not None:
            on_record(question, record)

    outcome = run_carefully(
        pending,
        lambda question: ask(question.question),
        keep,
        name=lambda question: question.id,
        errors=(GenerationError, VectorStoreError),
        wait=wait,
        on_failure=on_failure,
        rest=rest,
        max_attempts=max_attempts,
        failures_before_stopping=failures_before_stopping,
    )
    run.attempts, run.failures, run.stopped = outcome.attempts, outcome.failures, outcome.stopped
    return run


def answers_stage(run_dir: Path) -> Stage:
    """The files in which a run keeps its replies."""
    return Stage(
        run_dir,
        ANSWERS_FILE,
        RUN_FILE,
        holds="replies",
        outputs=("results.jsonl", "errors.jsonl", "summary.json", "change.md"),
        traces=True,
    )


def open_run(
    run_dir: Path, config: Mapping[str, object], *, restart: bool = False
) -> dict[str, AnswerRecord]:
    """Start a run, or pick up one that was stopped. Returns the replies already saved.

    A run is one configuration: one model, one prompt, one way of searching. Replies made
    with another configuration are never mixed in (see :meth:`Stage.open`).
    """
    stage = answers_stage(run_dir)
    return _records(stage.open(config, restart=restart), stage.rows_path)


def check_run(run_dir: Path, config: Mapping[str, object]) -> None:
    """Refuse a run that these settings cannot continue. Reads only, and writes nothing."""
    answers_stage(run_dir).check(config)


def add_attempts(run_dir: Path, attempts: int = 1) -> int:
    """Add to the run's count of attempts, and return the new total."""
    return answers_stage(run_dir).add_attempts(attempts)


def attempts_so_far(run_dir: Path) -> int:
    """How many questions this run has put to the pipeline since it was started or restarted."""
    return answers_stage(run_dir).attempts()


def append_record(run_dir: Path, record: AnswerRecord) -> None:
    """Save one reply at once, so that a run that stops loses nothing it paid for."""
    answers_stage(run_dir).append(asdict(record))


def load_records(path: Path) -> dict[str, AnswerRecord]:
    """Read saved replies. The latest reply to a question wins."""
    return _records(load_rows(path), path)


def load_failures(run_dir: Path) -> list[FailedQuestion]:
    """The questions a run last reported as failed, from its ``errors.jsonl``."""
    path = run_dir / "errors.jsonl"
    try:
        return [FailedQuestion(**row) for _, row in load_rows(path)]
    except TypeError as error:
        raise SavedRunError(f"{path} does not hold failed questions: {error}") from error


def saved_description(run_dir: Path) -> str | None:
    """The one-line description a run was last written with, if it has been written."""
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    description = json.loads(path.read_text(encoding="utf-8")).get("description")
    return description if isinstance(description, str) and description else None


def _records(rows: Sequence[tuple[int, dict[str, Any]]], path: Path) -> dict[str, AnswerRecord]:
    records: dict[str, AnswerRecord] = {}
    for number, row in rows:
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
    supported: Mapping[str, bool] | None = None,
) -> AnswerReport:
    """Score every reply and average the scores, overall and for the groups worth reading.

    ``supported`` holds the judge's finding for the answers it has judged, by question
    id: whether every statement of the answer is backed by the passages it cites. It adds
    the one score that code alone cannot compute. An answer the judge has not judged gets
    no such score, and is not counted as supported or as unsupported.
    """
    scored = []
    for question in questions:
        if question.id not in records:
            continue
        record = records[question.id]
        scores = score_answer(question, record)
        if supported is not None and record.answered and question.id in supported:
            scores["supported"] = float(supported[question.id])
        scored.append(ScoredAnswer(question, record, scores))
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
    judge: Mapping[str, object] | None = None,
) -> Path:
    """Write the scores of one run next to its saved replies, and return its directory.

    The layout matches a retrieval run: ``results.jsonl`` has one row per question and is
    small enough to commit, ``summary.json`` has the means with what produced them, and
    ``traces/`` has one readable file per question for the machine that ran it.
    """
    run_dir = flow_dir / variant
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    # A trace from an earlier invocation may be of a question that is no longer scored.
    clear_traces(run_dir)
    write_json(
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
            write_json(run_dir / "traces" / f"{question.id}_rep0.json", _trace(item))

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
    if judge is not None:
        # Who judged, what it cost, and how far it agreed with labelled answers.
        summary["judge"] = dict(judge)
    write_json(run_dir / "summary.json", summary)
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
