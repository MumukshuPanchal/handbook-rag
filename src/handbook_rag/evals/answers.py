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

import dataclasses
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
from handbook_rag.generate.base import GenerationError, only_the_gap_form, plain_words
from handbook_rag.generate.checks import CITES_NOTHING, DECLINED, WORDS_BESIDE_A_DECLINE
from handbook_rag.generate.models import Answer, ReplyCheck
from handbook_rag.generate.prompt import GAP_FORM_VERSIONS, NOT_FOUND_MESSAGE
from handbook_rag.index.store import VectorStoreError
from handbook_rag.ingest.models import Chunk, Document

# The headline comes first. Each score is 1 or 0 for one question, and a score that does
# not apply to a question is left out, so every mean says how many questions it rests on.
METRICS: list[dict[str, str]] = [
    {"id": "right_call", "label": "Right call", "kind": "binary"},
    {"id": "supported", "label": "Supported by its sources", "kind": "binary"},
    {"id": "correct", "label": "Correct by the reference", "kind": "binary"},
    {"id": "answered_correctly", "label": "Gets a correct answer", "kind": "binary"},
    {"id": "answered", "label": "Answers when it should", "kind": "binary"},
    {"id": "declined", "label": "Declines when it should", "kind": "binary"},
    {"id": "evidence_in_context", "label": "Evidence among the passages", "kind": "binary"},
    {"id": "fully_cited", "label": "Every statement cited", "kind": "binary"},
    {"id": "cites_evidence", "label": "Cites the evidence", "kind": "binary"},
]
METRIC_IDS = tuple(metric["id"] for metric in METRICS)

ANSWERS_FILE = "answers.jsonl"
CHANGED_REPLIES_FILE = "changed-replies.jsonl"
"""The replies to questions the handbook answers that a check changed or that were read as
the fixed form, for a reader (:func:`changed_replies`)."""
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
    checks: tuple[ReplyCheck, ...] = ()
    """The checks of the reply in code that fired (``generate/checks.py``). The
    statements are the reply as they left it, and each check keeps the words it took
    out. A reply saved before there were checks has none."""
    model_text: str = ""
    """The model's own words, when they were nothing but the fixed form for a missing part
    and the reply is a decline (``Answer.model_text``): shown as the fixed declining
    sentence when they cited nothing, and as these words when they carried markers
    (:func:`shown_in_the_models_words`). A decline by the model, not a check."""
    seconds_to_first_shown: float | None = None
    """When the first words were passed on to the reader (``Answer.seconds_to_first_shown``).
    A reply saved before this was recorded has none."""

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
    return record_reply(
        question.id,
        question.fingerprint,
        answer,
        covers=lambda chunk: tuple(sorted(covered_evidence(chunk, spans))),
    )


def record_reply(
    reply_to: str,
    fingerprint: str,
    answer: Answer,
    *,
    covers: Callable[[Chunk], tuple[int, ...]],
) -> AnswerRecord:
    """Reduce any reply to the passages it was given and what it said, by passage number.

    ``reply_to`` and ``fingerprint`` name what was asked, as it was when it was asked.
    ``covers`` says which pieces of the known evidence a passage holds. The answer
    evaluation and the adversarial tests both keep their replies in this one form, so
    that they are saved, read back and scored by the same code.
    """
    numbers = {
        scored.chunk.chunk_id: number for number, scored in enumerate(answer.retrieved, start=1)
    }
    for segment in answer.segments:
        for citation in segment.citations:
            if citation.chunk.chunk_id not in numbers:
                raise ReplyError(
                    f"The reply to {reply_to} cites a passage the model was not given "
                    f"({citation.chunk.chunk_id}). That is a fault in the generator."
                )
    return AnswerRecord(
        question_id=reply_to,
        fingerprint=fingerprint,
        passages=tuple(
            Passage(
                number=number,
                chunk_id=scored.chunk.chunk_id,
                doc_id=scored.chunk.doc_id,
                context_header=scored.chunk.context_header,
                covers=covers(scored.chunk),
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
        checks=answer.checks,
        model_text=answer.model_text,
        seconds_to_first_shown=answer.seconds_to_first_shown,
    )


def as_read_today(record: AnswerRecord, *, form_read: bool = False) -> AnswerRecord:
    """A saved reply as today's generator would have given it.

    Two readings of a reply that is nothing but the fixed form for a missing part
    (``base.read_the_gap_form_as_a_decline``) came after replies were saved:

    - **One that cites nothing.** Before, it went to the check for a reply that cites
      nothing, which declined it and said so. The reply shown was the same fixed declining
      sentence; what differs is whose decline it is, the model's, with no note. So a saved
      reply whose one check is that one, and whose words are nothing but the form, is read
      with no check and with those words as the model's.
    - **One that carries citation markers.** Before, it was an answer. It is now a decline
      shown in the model's words with no source, and those words are the model's. This
      applies only to a run made under instructions that give the form (``form_read``):
      under the others no reply was ever read so.

    Every other reply is read as it was saved.
    """
    if (
        len(record.checks) == 1
        and record.checks[0].name == CITES_NOTHING
        and only_the_gap_form(record.checks[0].removed)
    ):
        return dataclasses.replace(record, checks=(), model_text=record.checks[0].removed)
    if form_read and record.answered and record.cited and only_the_gap_form(record.text):
        return dataclasses.replace(
            record,
            answered=False,
            statements=tuple(Statement(statement.text, ()) for statement in record.statements),
            model_text=record.text.strip(),
        )
    return record


def shown_in_the_models_words(record: AnswerRecord) -> bool:
    """Whether a reply read as the fixed form alone carried citation markers.

    Such a reply is shown in the model's own words, with no source; one that cites nothing
    is shown as the fixed declining sentence. So a saved run tells the two apart by the
    words shown, with no request.
    """
    return bool(record.model_text) and plain_words(record.text) != plain_words(NOT_FOUND_MESSAGE)


def gives_the_form(run_dir: Path) -> bool:
    """Whether a saved run was made under instructions that give the fixed form for a
    missing part, as the settings it recorded say. A run with no record of its settings,
    or one of version 1, was not."""
    settings = run_dir / RUN_FILE
    if not settings.is_file():
        return False
    try:
        config = json.loads(settings.read_text(encoding="utf-8")).get("config", {})
    except json.JSONDecodeError:
        return False
    return config.get("instructions_version", 1) in GAP_FORM_VERSIONS


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


def judged_scores(
    question: GoldenQuestion,
    record: AnswerRecord,
    judged: Mapping[str, Mapping[str, bool]],
    *,
    every_answer_ruled_on: bool,
) -> dict[str, float]:
    """The scores the judges add for one reply. A score that does not apply is left out.

    - ``supported``, for an answer: no statement in it goes beyond the passage it cites.
    - ``correct``, for an answer to a question the handbook answers: it agrees with the
      reference answer.
    - ``answered_correctly``, for every question the handbook answers: the asker got a
      correct answer. A reply that declined counts as 0 here, so this is the one score
      that joins the search, the decision to answer and the answer itself.

    An answer a judge has not ruled on gets no score from that judge. It is counted
    neither way.

    ``every_answer_ruled_on`` says whether the judge of correctness has ruled on every
    answer of the run that has a reference answer. Until it has, ``answered_correctly``
    is not given at all: the replies that declined would all be counted, and only some
    of the answers, which would make the score too low.
    """
    scores: dict[str, float] = {}
    if record.answered:
        for metric, passed in judged.items():
            if question.id in passed:
                scores[metric] = float(passed[question.id])
    if question.type == "answerable" and every_answer_ruled_on:
        scores["answered_correctly"] = scores.get("correct", 0.0) if record.answered else 0.0
    return scores


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
        outputs=(
            "results.jsonl",
            "errors.jsonl",
            "summary.json",
            "change.md",
            CHANGED_REPLIES_FILE,
        ),
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
    # The settings are kept beside the replies, whichever kind of run holds them.
    form_read = gives_the_form(path.parent)
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
                    "checks": tuple(ReplyCheck(**check) for check in row.get("checks", ())),
                }
            )
        except (KeyError, TypeError) as error:
            raise SavedRunError(f"{path}:{number} is not a saved reply: {error}") from error
        records[record.question_id] = as_read_today(record, form_read=form_read)
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
    reply_checks: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    """For each check of the reply in code that fired, the questions it fired on, split
    by whether the handbook answers them: ``answerable`` and ``unanswerable``. A reply
    declined on a question the handbook answers costs a right call, whatever its words;
    each kept its words in the saved reply, for a reader to say whether it was a right
    answer. On a question the handbook does not answer, a decline is the right call."""
    declined_in_the_gap_form: dict[str, list[str]] = field(default_factory=dict)
    """The questions whose reply was nothing but the fixed form for a missing part and so
    a decline by the model, split as above."""
    shown_in_the_models_words: dict[str, list[str]] = field(default_factory=dict)
    """Of those, the replies that carried citation markers, shown in the model's words with
    no source (:func:`shown_in_the_models_words`), split as above. The others were shown
    as the fixed declining sentence."""


def summarise_answers(
    questions: Sequence[GoldenQuestion],
    records: Mapping[str, AnswerRecord],
    failures: Sequence[FailedQuestion] = (),
    judged: Mapping[str, Mapping[str, bool]] | None = None,
) -> AnswerReport:
    """Score every reply and average the scores, overall and for the groups worth reading.

    ``judged`` holds what the judges found, by the id of the score and then by question
    id: whether the answer passed. These are the scores that code alone cannot compute
    (see :func:`judged_scores`).
    """
    judged = judged or {}
    # Whether every answer that has a reference answer has been ruled on for correctness.
    whole = "correct" in judged and all(
        question.id in judged["correct"]
        for question in questions
        if question.type == "answerable"
        and question.id in records
        and records[question.id].answered
    )
    scored = []
    for question in questions:
        if question.id not in records:
            continue
        record = records[question.id]
        scores = score_answer(question, record)
        scores.update(judged_scores(question, record, judged, every_answer_ruled_on=whole))
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
        # Only replies that recorded it: a run saved before it was recorded has none.
        shown = [r.seconds_to_first_shown for r in asked if r.seconds_to_first_shown is not None]
        if shown:
            timing["median_seconds_to_first_shown"] = statistics.median(shown)
            timing["p95_seconds_to_first_shown"] = sorted(shown)[-(-95 * len(shown) // 100) - 1]
        tokens = {
            "input_total": float(sum(record.input_tokens for record in asked)),
            "output_total": float(sum(record.output_tokens for record in asked)),
            "input_mean": statistics.fmean(record.input_tokens for record in asked),
            "output_mean": statistics.fmean(record.output_tokens for record in asked),
        }
    reply_checks: dict[str, dict[str, list[str]]] = {}
    gap_form: dict[str, list[str]] = {}
    as_written: dict[str, list[str]] = {}
    for item in scored:
        for check in item.record.checks:
            reply_checks.setdefault(check.name, {}).setdefault(item.question.type, []).append(
                item.question.id
            )
        if item.record.model_text:
            gap_form.setdefault(item.question.type, []).append(item.question.id)
        if shown_in_the_models_words(item.record):
            as_written.setdefault(item.question.type, []).append(item.question.id)
    return AnswerReport(
        reply_checks=reply_checks,
        declined_in_the_gap_form=gap_form,
        shown_in_the_models_words=as_written,
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


def model_words(record: AnswerRecord) -> str:
    """The words the model wrote, as far as a saved reply keeps them.

    Exact for a reply read as the fixed form, for one a check declined and for one shown as
    the fixed declining sentence in place of its words: the saved run keeps them whole.
    Otherwise the reply as shown, with the words each check took out after it: lines a
    check dropped stood somewhere after the reply's last citation, not always at its end.
    """
    if record.model_text:
        return record.model_text
    for check in record.checks:
        if check.action == DECLINED or check.name == WORDS_BESIDE_A_DECLINE:
            return check.removed
    taken = [check.removed for check in record.checks if check.removed]
    return "\n".join([record.text.rstrip(), *taken]) if taken else record.text


def changed_replies(report: AnswerReport) -> list[dict[str, object]]:
    """Every reply to a question the handbook answers that a check changed or marked, or
    that was read as the fixed form for a missing part, as a reader is to be given it.

    Each holds the question, its reference answer, the model's own words
    (:func:`model_words`), the reply as shown, and what was done to it, and nothing of the
    run's totals: so that a reader can say, reply by reply, whether the person lost an
    answer the model wrote.
    """
    rows: list[dict[str, object]] = []
    for item in report.scored:
        question, record = item.question, item.record
        if question.type != "answerable" or not (record.checks or record.model_text):
            continue
        done = [f"{check.name}: {check.action}" for check in record.checks]
        if shown_in_the_models_words(record):
            done.append(
                "read as the fixed form for a missing part, with citation markers: "
                "a decline, shown in the model's words with no source"
            )
        elif record.model_text:
            done.append("read as the fixed form for a missing part: shown as the decline")
        rows.append(
            {
                "question_id": question.id,
                "question": question.question,
                "reference_answer": question.reference_answer,
                "model_words": model_words(record),
                "reply_as_shown": record.text,
                "done": done,
            }
        )
    return rows


def write_answer_run(
    report: AnswerReport,
    flow_dir: Path,
    variant: str,
    description: str,
    config: Mapping[str, object],
    *,
    attempts: int,
    judges: Mapping[str, Mapping[str, object]] | None = None,
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
    # What the checks of the reply did, and which replies the model wrote in the fixed form
    # alone. Written only when there is any, so that a run of instructions without checks
    # is summarised as every run before there were checks.
    if report.reply_checks:
        summary["reply_checks"] = report.reply_checks
    if report.declined_in_the_gap_form:
        summary["declined_in_the_gap_form"] = report.declined_in_the_gap_form
    if report.shown_in_the_models_words:
        summary["declined_in_the_gap_form_with_markers"] = report.shown_in_the_models_words
    # For a reader of the replies a check changed or that were read as the fixed form, on
    # questions the handbook answers: a file of its own, without the run's totals.
    changed = changed_replies(report)
    if changed:
        with (run_dir / CHANGED_REPLIES_FILE).open("w", encoding="utf-8") as output:
            for row in changed:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        (run_dir / CHANGED_REPLIES_FILE).unlink(missing_ok=True)
    if judges:
        # For each judged score: who judged, what it cost, and how far that judge agreed
        # with labelled answers.
        summary["judges"] = {metric: dict(record) for metric, record in judges.items()}
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
