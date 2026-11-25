"""The answer run: what it asks, what it saves, how it scores, and when it stops."""

import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from handbook_rag.evals.answers import (
    CHANGED_REPLIES_FILE,
    METRIC_IDS,
    AnswerRecord,
    Passage,
    ReplyError,
    Statement,
    add_attempts,
    append_record,
    as_read_today,
    attempts_so_far,
    changed_replies,
    check_run,
    in_run_order,
    load_records,
    model_words,
    open_run,
    record_answer,
    replies_to_current_labels,
    run_answers,
    score_answer,
    summarise_answers,
    write_answer_run,
)
from handbook_rag.evals.careful import Pacer
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion, locate_evidence
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.stage import SavedRunError
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.checks import ADDRESS_IN_CLOSING_LINE, CITES_NOTHING, NOTES
from handbook_rag.generate.models import Answer, Citation, ReplyCheck, Segment, TokenUsage
from handbook_rag.generate.prompt import GAP_FORM, NOT_FOUND_MESSAGE
from handbook_rag.index.store import ScoredChunk, VectorStoreError
from handbook_rag.ingest.models import Chunk, Document

LEAVE_TEXT = "## Parental leave\n\nNew parents get sixteen weeks of paid leave."
EXPENSE_TEXT = "## Laptops\n\nExpense a laptop up to one thousand dollars with approval."


def page(doc_id: str, text: str) -> Document:
    return Document(doc_id, "Title", "https://example.com/", doc_id.split("/")[0], "a" * 40, text)


def whole_page(document: Document) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{document.doc_id}",
        doc_id=document.doc_id,
        title=document.title,
        url=document.url,
        section=document.section,
        source_commit=document.source_commit,
        heading_path=(),
        start_char=0,
        end_char=len(document.text),
        text=document.text,
        token_count=10,
    )


LEAVE = page("people/leave.md", LEAVE_TEXT)
EXPENSES = page("finance/expenses.md", EXPENSE_TEXT)
DOCUMENTS = [LEAVE, EXPENSES]
LEAVE_CHUNK, EXPENSE_CHUNK = whole_page(LEAVE), whole_page(EXPENSES)

LEAVE_QUESTION = GoldenQuestion.model_validate(
    {
        "id": "q001",
        "question": "How long is parental leave?",
        "reference_answer": "Sixteen weeks.",
        "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
        "type": "answerable",
        "origin": "model-drafted",
    }
)
LAPTOP_QUESTION = GoldenQuestion.model_validate(
    {
        "id": "q002",
        "question": "What is the laptop expense limit?",
        "reference_answer": "One thousand dollars.",
        "evidence": [{"doc_id": "finance/expenses.md", "quote": "one thousand dollars"}],
        "type": "answerable",
        "origin": "model-blind",
    }
)
PARKING_QUESTION = GoldenQuestion.model_validate(
    {
        "id": "u001",
        "question": "How much is the parking allowance?",
        "type": "unanswerable",
        "origin": "model-drafted",
        "unanswerable_kind": "near-miss",
        "note": "No page mentions parking.",
    }
)
QUESTIONS = [LEAVE_QUESTION, LAPTOP_QUESTION, PARKING_QUESTION]
BY_TEXT = {question.question: question for question in QUESTIONS}


def reply(
    question: GoldenQuestion,
    *segments: Segment,
    answered: bool = True,
    given: tuple[Chunk, ...] = (EXPENSE_CHUNK, LEAVE_CHUNK),
    stop_reason: str = "end_turn",
    seconds: float = 2.0,
) -> Answer:
    return Answer(
        question=question.question,
        segments=segments,
        answered=answered,
        model="fake-model",
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=1800, output_tokens=90),
        seconds=seconds,
        seconds_to_first_text=1.0,
        retrieved=tuple(ScoredChunk(chunk, 0.5) for chunk in given),
    )


def declines(
    question: GoldenQuestion, *, given: tuple[Chunk, ...] = (EXPENSE_CHUNK, LEAVE_CHUNK)
) -> Answer:
    return reply(question, Segment(NOT_FOUND_MESSAGE), answered=False, given=given)


def good_reply(question: GoldenQuestion) -> Answer:
    """The reply a careful assistant gives: the right passage cited, or a decline."""
    if question is LEAVE_QUESTION:
        return reply(question, Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)))
    if question is LAPTOP_QUESTION:
        return reply(question, Segment("One thousand dollars.", (Citation(EXPENSE_CHUNK),)))
    return declines(question)


def careful(question_text: str) -> Answer:
    return good_reply(BY_TEXT[question_text])


def scored(question: GoldenQuestion, answer: Answer) -> dict[str, float]:
    spans = locate_evidence(question, {document.doc_id: document for document in DOCUMENTS})
    return score_answer(question, record_answer(question, answer, spans))


# What a reply is reduced to


def test_a_record_names_citations_by_passage_number_and_marks_the_evidence() -> None:
    answer = reply(
        LEAVE_QUESTION,
        Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK), Citation(EXPENSE_CHUNK))),
        Segment(" Ask the People team."),
    )
    spans = locate_evidence(LEAVE_QUESTION, {d.doc_id: d for d in DOCUMENTS})

    record = record_answer(LEAVE_QUESTION, answer, spans)

    assert [(p.number, p.doc_id, p.covers) for p in record.passages] == [
        (1, "finance/expenses.md", ()),
        (2, "people/leave.md", (0,)),
    ]
    assert record.statements == (
        Statement("Sixteen weeks.", (1, 2)),
        Statement(" Ask the People team.", ()),
    )
    assert record.text == "Sixteen weeks. Ask the People team."
    assert record.text_with_markers == "Sixteen weeks.[1][2] Ask the People team."
    assert record.cited == {1, 2}
    assert record.uncited_statements == ("Ask the People team.",)
    assert (record.question_id, record.fingerprint) == ("q001", LEAVE_QUESTION.fingerprint)
    assert (record.input_tokens, record.output_tokens, record.seconds) == (1800, 90, 2.0)
    assert (record.answered, record.stop_reason, record.model) == (True, "end_turn", "fake-model")


def test_a_citation_of_a_passage_the_model_was_not_given_is_a_fault_and_says_so() -> None:
    answer = reply(
        LEAVE_QUESTION, Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)), given=(EXPENSE_CHUNK,)
    )

    with pytest.raises(ReplyError, match="q001 cites a passage the model was not given"):
        record_answer(LEAVE_QUESTION, answer, [])


# Scores


def test_a_cited_answer_to_an_answerable_question_scores_on_every_count() -> None:
    assert scored(LEAVE_QUESTION, good_reply(LEAVE_QUESTION)) == {
        "right_call": 1.0,
        "answered": 1.0,
        "evidence_in_context": 1.0,
        "cites_evidence": 1.0,
        "fully_cited": 1.0,
    }


def test_declining_an_answerable_question_is_a_wrong_call_whoever_is_to_blame() -> None:
    # The evidence was among the passages, and the assistant still declined.
    assert scored(LEAVE_QUESTION, declines(LEAVE_QUESTION)) == {
        "right_call": 0.0,
        "answered": 0.0,
        "evidence_in_context": 1.0,
    }
    # The search did not find the evidence. From outside it is still a question that
    # went unanswered, and the score for the evidence shows where it went wrong.
    assert scored(LEAVE_QUESTION, declines(LEAVE_QUESTION, given=(EXPENSE_CHUNK,))) == {
        "right_call": 0.0,
        "answered": 0.0,
        "evidence_in_context": 0.0,
    }


def test_an_answer_that_cites_only_another_passage_does_not_cite_the_evidence() -> None:
    answer = reply(LEAVE_QUESTION, Segment("One thousand dollars.", (Citation(EXPENSE_CHUNK),)))

    scores = scored(LEAVE_QUESTION, answer)

    assert scores["cites_evidence"] == 0.0
    assert scores["fully_cited"] == 1.0


def test_an_answer_given_without_the_evidence_has_no_score_for_citing_it() -> None:
    answer = reply(
        LEAVE_QUESTION,
        Segment("Sixteen weeks.", (Citation(EXPENSE_CHUNK),)),
        given=(EXPENSE_CHUNK,),
    )

    scores = scored(LEAVE_QUESTION, answer)

    assert scores["evidence_in_context"] == 0.0
    assert scores["answered"] == 1.0
    assert "cites_evidence" not in scores


def test_declining_an_unanswerable_question_is_the_right_call() -> None:
    assert scored(PARKING_QUESTION, declines(PARKING_QUESTION)) == {
        "right_call": 1.0,
        "declined": 1.0,
    }


def test_answering_an_unanswerable_question_is_a_wrong_call() -> None:
    answer = reply(PARKING_QUESTION, Segment("Fifty dollars.", (Citation(LEAVE_CHUNK),)))

    assert scored(PARKING_QUESTION, answer) == {
        "right_call": 0.0,
        "declined": 0.0,
        "fully_cited": 1.0,
    }


@pytest.mark.parametrize(
    ("segments", "fully_cited"),
    [
        ((Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)), Segment(" \n\n")), 1.0),
        ((Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)), Segment(" - ")), 1.0),
        ((Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)), Segment(" Paid.")), 0.0),
        # A number is a statement too.
        ((Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)), Segment(" 16")), 0.0),
        ((Segment("Sixteen weeks."),), 0.0),
    ],
)
def test_text_without_words_needs_no_citation(
    segments: tuple[Segment, ...], fully_cited: float
) -> None:
    assert scored(LEAVE_QUESTION, reply(LEAVE_QUESTION, *segments))["fully_cited"] == fully_cited


# The run


def test_every_question_is_asked_once_and_each_reply_is_handed_over_at_once() -> None:
    asked: list[str] = []
    kept: list[str] = []

    def ask(question_text: str) -> Answer:
        # A reply is handed over before the next question is asked.
        assert len(kept) == len(asked)
        asked.append(question_text)
        return careful(question_text)

    run = run_answers(
        QUESTIONS, DOCUMENTS, ask, on_record=lambda question, record: kept.append(question.id)
    )

    assert sorted(asked) == sorted(question.question for question in QUESTIONS)
    assert kept == [question.id for question in in_run_order(QUESTIONS)]
    assert set(run.records) == {"q001", "q002", "u001"}
    assert (run.attempts, run.failures, run.stopped) == (3, [], None)


def test_the_order_is_fixed_and_does_not_follow_the_file() -> None:
    many = [
        LEAVE_QUESTION.model_copy(update={"id": f"q{n:03d}", "question": f"Question {n}?"})
        for n in range(40)
    ]

    order = [question.id for question in in_run_order(many)]

    assert order == [question.id for question in in_run_order(list(reversed(many)))]
    assert sorted(order) == [question.id for question in many]
    assert order != sorted(order)


def test_a_saved_reply_is_not_asked_again() -> None:
    first = run_answers(QUESTIONS, DOCUMENTS, careful)
    asked: list[str] = []

    def ask(question_text: str) -> Answer:
        asked.append(question_text)
        return careful(question_text)

    saved = {"q001": first.records["q001"], "u001": first.records["u001"]}
    second = run_answers(QUESTIONS, DOCUMENTS, ask, saved=saved)

    assert asked == [LAPTOP_QUESTION.question]
    assert second.attempts == 1
    assert set(second.records) == {"q001", "q002", "u001"}
    assert second.records["q001"] is saved["q001"]


def test_a_reply_to_an_older_wording_of_the_label_is_asked_again() -> None:
    first = run_answers(QUESTIONS, DOCUMENTS, careful)
    reworded = LEAVE_QUESTION.model_copy(update={"reference_answer": "16 weeks, fully paid."})
    asked: list[str] = []

    def ask(question_text: str) -> Answer:
        asked.append(question_text)
        return careful(question_text)

    run_answers([reworded, LAPTOP_QUESTION, PARKING_QUESTION], DOCUMENTS, ask, saved=first.records)

    assert asked == [LEAVE_QUESTION.question]


def test_only_replies_to_the_questions_as_they_are_labelled_now_are_kept() -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records
    reworded = LEAVE_QUESTION.model_copy(update={"reference_answer": "16 weeks, fully paid."})

    current = replies_to_current_labels([reworded, PARKING_QUESTION], records)

    # The reply to the reworded question was made for another label, and the laptop
    # question is no longer in the set.
    assert set(current) == {"u001"}
    assert replies_to_current_labels(QUESTIONS, records) == records
    assert replies_to_current_labels(QUESTIONS, {}) == {}


def test_a_saved_reply_to_a_question_no_longer_in_the_set_is_left_out() -> None:
    first = run_answers(QUESTIONS, DOCUMENTS, careful)

    second = run_answers([LEAVE_QUESTION], DOCUMENTS, careful, saved=first.records)

    assert set(second.records) == {"q001"}
    assert second.attempts == 0


def failing_first(times: int, error: Exception) -> Callable[[str], Answer]:
    """An assistant whose first ``times`` calls fail."""
    calls = 0

    def ask(question_text: str) -> Answer:
        nonlocal calls
        calls += 1
        if calls <= times:
            raise error
        return careful(question_text)

    return ask


@pytest.mark.parametrize(
    "error", [GenerationError("The Gemini API returned an error (503)."), VectorStoreError("down")]
)
def test_a_question_that_fails_is_asked_once_more_after_the_others(error: Exception) -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, failing_first(1, error))

    assert set(run.records) == {"q001", "q002", "u001"}
    assert run.failures == []
    # Three questions, and one more try for the one that failed.
    assert run.attempts == 4
    assert run.stopped is None


def test_each_failure_is_passed_on_as_it_happens_even_one_that_is_made_good() -> None:
    seen: list[FailedQuestion] = []

    run = run_answers(
        QUESTIONS,
        DOCUMENTS,
        failing_first(1, GenerationError("The Gemini API returned an error (503).")),
        on_failure=seen.append,
    )

    # The second try succeeded, so the run reports no failure. The first one was still seen.
    assert run.failures == []
    assert [(failure.question_id, failure.error_class) for failure in seen] == [
        (in_run_order(QUESTIONS)[0].id, "GenerationError")
    ]
    assert "503" in seen[0].message


def test_a_question_that_fails_twice_is_reported_and_never_scored() -> None:
    first_in_order = in_run_order(QUESTIONS)[0]

    def ask(question_text: str) -> Answer:
        if question_text == first_in_order.question:
            raise GenerationError("The Gemini API sent a reply that could not be read.")
        return careful(question_text)

    run = run_answers(QUESTIONS, DOCUMENTS, ask)
    report = summarise_answers(QUESTIONS, run.records, run.failures)

    assert [failure.question_id for failure in run.failures] == [first_in_order.id]
    assert run.failures[0].error_class == "GenerationError"
    assert "could not be read" in run.failures[0].message
    assert first_in_order.id not in run.records
    assert run.attempts == 4
    assert len(report.scored) == 2
    assert len(report.failures) == 1
    assert report.not_asked == 0
    assert report.summary["right_call"].count == 2


def test_the_run_stops_itself_when_questions_fail_one_after_another() -> None:
    many = [
        LEAVE_QUESTION.model_copy(update={"id": f"q{n:03d}", "question": f"Question {n}?"})
        for n in range(10)
    ]
    calls = 0

    def ask(question_text: str) -> Answer:
        nonlocal calls
        calls += 1
        raise GenerationError("Today's limit is used up.")

    run = run_answers(many, DOCUMENTS, ask)

    # Three requests were spent to learn it, not ten, and no second round was started.
    assert calls == 3
    assert run.attempts == 3
    assert run.stopped is not None
    assert "3 questions in a row failed" in run.stopped
    assert "Today's limit is used up." in run.stopped
    assert len(run.failures) == 3
    assert summarise_answers(many, run.records, run.failures).not_asked == 7


def test_failures_with_a_success_between_them_do_not_stop_the_run() -> None:
    many = [
        LEAVE_QUESTION.model_copy(update={"id": f"q{n:03d}", "question": f"Question {n}?"})
        for n in range(9)
    ]
    answers = {question.question: question for question in many}
    calls = 0

    def ask(question_text: str) -> Answer:
        nonlocal calls
        calls += 1
        # In the first round every third call succeeds. The second round succeeds.
        if calls <= 9 and calls % 3 != 0:
            raise GenerationError("busy")
        question = answers[question_text]
        return reply(question, Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)))

    run = run_answers(many, DOCUMENTS, ask)

    assert run.stopped is None
    assert len(run.records) == 9
    assert run.failures == []
    assert run.attempts == 9 + 6


def test_an_error_that_is_a_bug_is_not_swallowed() -> None:
    def ask(question_text: str) -> Answer:
        raise TypeError("a mistake in our own code")

    with pytest.raises(TypeError, match="our own code"):
        run_answers(QUESTIONS, DOCUMENTS, ask)


def test_a_limit_on_requests_ends_the_run_early_and_says_so() -> None:
    kept: list[str] = []

    run = run_answers(
        QUESTIONS,
        DOCUMENTS,
        careful,
        max_attempts=2,
        on_record=lambda question, record: kept.append(question.id),
    )

    assert run.attempts == 2
    assert kept == [question.id for question in in_run_order(QUESTIONS)[:2]]
    assert run.stopped == "the limit of 2 requests for this run was reached"
    assert summarise_answers(QUESTIONS, run.records, run.failures).not_asked == 1


def test_a_limit_that_is_never_reached_is_not_reported_as_a_stop() -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful, max_attempts=3)

    assert run.attempts == 3
    assert run.stopped is None


def test_a_broken_label_stops_the_run_before_any_request() -> None:
    broken = GoldenQuestion.model_validate(
        {
            **LEAVE_QUESTION.model_dump(),
            "evidence": [{"doc_id": "people/leave.md", "quote": "twenty weeks"}],
        }
    )
    asked: list[str] = []

    def ask(question_text: str) -> Answer:
        asked.append(question_text)
        return careful(question_text)

    with pytest.raises(DatasetError, match="quote not found"):
        run_answers([broken, PARKING_QUESTION], DOCUMENTS, ask)

    assert asked == []


def test_the_run_waits_before_every_request() -> None:
    events: list[str] = []

    def ask(question_text: str) -> Answer:
        events.append("ask")
        return careful(question_text)

    run_answers(QUESTIONS, DOCUMENTS, ask, wait=lambda: events.append("wait"))

    assert events == ["wait", "ask"] * 3


# Pacing


class Clock:
    """A clock that moves only when someone sleeps or the test moves it."""

    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_requests_are_spaced_to_the_rate_and_the_first_one_starts_at_once() -> None:
    clock = Clock()
    pacer = Pacer(10, clock=clock.time, sleep=clock.sleep)

    pacer.wait()
    clock.now += 2.0  # the request took two seconds
    pacer.wait()
    clock.now += 2.5
    pacer.wait()

    # Ten a minute is one start every six seconds, whatever each request takes.
    assert clock.slept == [pytest.approx(4.0), pytest.approx(3.5)]


def test_a_request_slower_than_the_gap_is_followed_without_a_pause() -> None:
    clock = Clock()
    pacer = Pacer(10, clock=clock.time, sleep=clock.sleep)

    pacer.wait()
    clock.now += 9.0
    pacer.wait()
    clock.now += 1.0
    pacer.wait()

    # No pause after the slow request, and the gap then counts from when the next began.
    assert clock.slept == [pytest.approx(5.0)]


@pytest.mark.parametrize("rate", [0, -5])
def test_a_pace_must_be_positive(rate: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        Pacer(rate)


# Saving a run and picking it up again

CONFIG: dict[str, object] = {
    "answer_model": "fake-model",
    "instructions": "abc123",
    "rerank": False,
}


def test_saved_replies_come_back_exactly_as_they_were(tmp_path: Path) -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful)
    open_run(tmp_path, CONFIG)

    for record in run.records.values():
        append_record(tmp_path, record)

    assert open_run(tmp_path, CONFIG) == run.records


def test_a_new_run_starts_empty_and_records_its_settings(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "baseline"

    assert open_run(run_dir, CONFIG) == {}

    assert json.loads((run_dir / "run.json").read_text()) == {"config": CONFIG, "attempts": 0}


def test_a_run_made_with_other_settings_is_not_continued(tmp_path: Path) -> None:
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])

    with pytest.raises(SavedRunError, match=r"other settings \(answer_model, rerank\)") as error:
        open_run(tmp_path, {**CONFIG, "answer_model": "another-model", "rerank": True})

    assert "--restart" in str(error.value)
    # Nothing was discarded by the refusal.
    assert set(open_run(tmp_path, CONFIG)) == {"q001"}


def test_restarting_discards_the_saved_replies_and_takes_the_new_settings(tmp_path: Path) -> None:
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])
    add_attempts(tmp_path, 5)
    changed = {**CONFIG, "answer_model": "another-model"}

    assert open_run(tmp_path, changed, restart=True) == {}

    assert json.loads((tmp_path / "run.json").read_text()) == {"config": changed, "attempts": 0}
    assert open_run(tmp_path, changed) == {}


def test_replies_without_a_record_of_their_settings_are_not_adopted(tmp_path: Path) -> None:
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])
    (tmp_path / "run.json").unlink()

    with pytest.raises(SavedRunError, match="nothing records the settings"):
        open_run(tmp_path, CONFIG)


def test_a_run_can_be_checked_before_anything_is_loaded_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "baseline"

    # A run that does not exist yet can be started, and checking does not create it.
    check_run(run_dir, CONFIG)
    assert not run_dir.exists()

    open_run(run_dir, CONFIG)
    append_record(run_dir, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])
    before = (run_dir / "run.json").read_text()

    check_run(run_dir, CONFIG)
    with pytest.raises(SavedRunError, match=r"other settings \(answer_model\)"):
        check_run(run_dir, {**CONFIG, "answer_model": "another-model"})

    assert (run_dir / "run.json").read_text() == before
    assert sorted(path.name for path in run_dir.iterdir()) == ["answers.jsonl", "run.json"]


def test_blank_lines_and_other_line_ends_between_replies_are_passed_over(tmp_path: Path) -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, records["q001"])
    append_record(tmp_path, records["q002"])
    answers = tmp_path / "answers.jsonl"
    first, second = answers.read_text().splitlines()

    answers.write_bytes(f"{first}\r\n\r\n\n  \n{second}\n".encode())

    assert load_records(answers) == {"q001": records["q001"], "q002": records["q002"]}


def test_a_run_that_holds_no_reply_takes_new_settings(tmp_path: Path) -> None:
    # For example a run whose every request failed, or one that never got to ask.
    open_run(tmp_path, CONFIG)
    add_attempts(tmp_path, 3)
    changed = {**CONFIG, "answer_model": "another-model"}

    assert open_run(tmp_path, changed) == {}

    # The requests it spent stay counted.
    assert json.loads((tmp_path / "run.json").read_text()) == {"config": changed, "attempts": 3}


@pytest.mark.parametrize(
    "damaged",
    [
        "{",
        "[]",
        '{"attempts": 2}',
        '{"config": {}, "attempts": "x"}',
        '{"config": {}, "attempts": -3}',
        '{"config": {}, "attempts": 2.5}',
        '{"config": {}, "attempts": true}',
    ],
)
def test_a_damaged_record_of_the_settings_is_reported_and_can_be_discarded(
    tmp_path: Path, damaged: str
) -> None:
    open_run(tmp_path, CONFIG)
    (tmp_path / "run.json").write_text(damaged)

    with pytest.raises(SavedRunError, match=r"run\.json cannot be read .*--restart"):
        open_run(tmp_path, CONFIG)
    with pytest.raises(SavedRunError, match=r"run\.json cannot be read"):
        add_attempts(tmp_path)

    assert open_run(tmp_path, CONFIG, restart=True) == {}
    assert attempts_so_far(tmp_path) == 0


def test_settings_match_whatever_shape_they_were_given_in(tmp_path: Path) -> None:
    # A tuple is saved as a list, and must still be the same settings when read back.
    config: dict[str, object] = {"answer_model": "fake-model", "sizes": (320, 60)}
    open_run(tmp_path, config)
    append_record(tmp_path, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])

    assert set(open_run(tmp_path, config)) == {"q001"}


def test_restarting_removes_everything_the_run_wrote(tmp_path: Path) -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful)
    open_run(tmp_path / "baseline", CONFIG)
    for record in run.records.values():
        append_record(tmp_path / "baseline", record)
    report = summarise_answers(QUESTIONS, run.records)
    run_dir = write_answer_run(report, tmp_path, "baseline", "First.", CONFIG, attempts=3)
    assert (run_dir / "summary.json").is_file()

    open_run(run_dir, CONFIG, restart=True)

    # Nothing is left that describes the discarded replies.
    assert sorted(path.name for path in run_dir.iterdir()) == ["run.json", "traces"]
    assert list((run_dir / "traces").iterdir()) == []


def test_the_count_of_attempts_survives_an_interruption(tmp_path: Path) -> None:
    open_run(tmp_path, CONFIG)
    calls = 0

    def ask(question_text: str) -> Answer:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt  # someone pressed Ctrl-C during the second request
        return careful(question_text)

    def count_the_request() -> None:
        # What the command does before each request: count it, then send it.
        add_attempts(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        run_answers(
            QUESTIONS,
            DOCUMENTS,
            ask,
            wait=count_the_request,
            on_record=lambda question, record: append_record(tmp_path, record),
        )

    assert attempts_so_far(tmp_path) == 2
    assert len(open_run(tmp_path, CONFIG)) == 1


def test_attempts_add_up_over_invocations(tmp_path: Path) -> None:
    open_run(tmp_path, CONFIG)

    assert add_attempts(tmp_path, 40) == 40
    assert add_attempts(tmp_path, 3) == 43
    assert json.loads((tmp_path / "run.json").read_text())["config"] == CONFIG


def test_a_line_cut_short_at_the_end_is_dropped_and_a_damaged_line_is_reported(
    tmp_path: Path,
) -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, records["q001"])
    append_record(tmp_path, records["q002"])
    answers = tmp_path / "answers.jsonl"
    whole = answers.read_text()
    first, second = whole.splitlines()

    # The process was killed while it wrote the last line: no line break ends it.
    answers.write_text(whole[:-40])
    assert set(load_records(answers)) == {"q001"}

    # A finished line that cannot be read is damage, wherever it is.
    answers.write_text(f"{first[:-40]}\n{second}\n")
    with pytest.raises(SavedRunError, match=r"answers\.jsonl:1 cannot be read"):
        load_records(answers)
    answers.write_text(f"{first}\n{second[:-40]}\n")
    with pytest.raises(SavedRunError, match=r"answers\.jsonl:2 cannot be read"):
        load_records(answers)


def test_a_reply_saved_after_a_cut_line_is_not_lost(tmp_path: Path) -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, records["q001"])
    answers = tmp_path / "answers.jsonl"
    with answers.open("a") as output:
        output.write('{"question_id": "q002", "fingerprint": "abc", "pass')  # killed here

    # The next invocation removes the cut line before it saves anything.
    assert set(open_run(tmp_path, CONFIG)) == {"q001"}
    append_record(tmp_path, records["q002"])
    append_record(tmp_path, records["u001"])

    assert open_run(tmp_path, CONFIG) == records
    assert answers.read_text().count("\n") == 3


@pytest.mark.parametrize("character", ["\u2028", "\u2029", "\x85", "\x0b", "\x0c", "\r"])
def test_a_reply_may_hold_characters_that_some_tools_take_for_the_end_of_a_line(
    tmp_path: Path, character: str
) -> None:
    record = run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"]
    odd = dataclasses.replace(record, statements=(Statement(f"Sixteen{character}weeks.", (2,)),))
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, odd)
    append_record(tmp_path, dataclasses.replace(record, question_id="q002"))

    assert open_run(tmp_path, CONFIG) == {
        "q001": odd,
        "q002": dataclasses.replace(record, question_id="q002"),
    }


def test_a_line_that_is_not_a_reply_is_reported(tmp_path: Path) -> None:
    answers = tmp_path / "answers.jsonl"
    answers.write_text('{"question_id": "q001"}\n')

    with pytest.raises(SavedRunError, match=r"answers\.jsonl:1 is not a saved reply"):
        load_records(answers)


def test_the_latest_reply_to_a_question_wins(tmp_path: Path) -> None:
    record = run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"]
    later = dataclasses.replace(record, seconds=9.0)
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, record)
    append_record(tmp_path, later)

    assert load_records(tmp_path / "answers.jsonl")["q001"].seconds == 9.0


# The summary


def test_an_assistant_that_does_everything_right_scores_one_on_every_metric() -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful)

    # The judges found both answers backed by the passages they cite, and correct.
    both = {"q001": True, "q002": True}
    report = summarise_answers(QUESTIONS, run.records, judged={"supported": both, "correct": both})

    assert set(report.summary) == set(METRIC_IDS)
    assert {name: estimate.mean for name, estimate in report.summary.items()} == dict.fromkeys(
        METRIC_IDS, 1.0
    )
    # Each mean says how many questions it rests on.
    assert {name: estimate.count for name, estimate in report.summary.items()} == {
        "right_call": 3,
        "supported": 2,
        "correct": 2,
        "answered_correctly": 2,
        "answered": 2,
        "declined": 1,
        "evidence_in_context": 2,
        "fully_cited": 2,
        "cites_evidence": 2,
    }
    assert (report.not_asked, report.failures) == (0, [])


def test_the_judged_score_is_given_only_to_answers_the_judge_has_ruled_on() -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful)

    unjudged = summarise_answers(QUESTIONS, run.records)
    partly = summarise_answers(
        QUESTIONS,
        run.records,
        judged={"supported": {"q001": False, "u001": True, "q999": True}},
    )

    # Without a judge there is no such score: code cannot compute it.
    assert "supported" not in unjudged.summary
    # One answer was judged and found wanting. The other answer was not judged, and is
    # counted neither way. A reply that declined has nothing to judge, whatever is passed.
    assert (partly.summary["supported"].mean, partly.summary["supported"].count) == (0.0, 1)
    grades = {item.question.id: item.scores for item in partly.scored}
    assert grades["q001"]["supported"] == 0.0
    assert "supported" not in grades["q002"]
    assert "supported" not in grades["u001"]
    # No judge has ruled on correctness, so nothing is said about it either.
    assert "correct" not in partly.summary
    assert "answered_correctly" not in partly.summary


def test_how_many_got_a_correct_answer_waits_for_the_last_verdict() -> None:
    # Both answerable questions are answered. The judge has ruled on one of them.
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records

    half = summarise_answers(QUESTIONS, records, judged={"correct": {"q001": True}})
    whole = summarise_answers(QUESTIONS, records, judged={"correct": {"q001": True, "q002": False}})

    # One verdict is not the whole story, however good it is.
    assert (half.summary["correct"].mean, half.summary["correct"].count) == (1.0, 1)
    assert "answered_correctly" not in half.summary
    assert (
        whole.summary["answered_correctly"].mean,
        whole.summary["answered_correctly"].count,
    ) == (
        0.5,
        2,
    )


def test_a_correct_answer_counts_for_the_answer_and_for_the_question() -> None:
    def ask(question_text: str) -> Answer:
        question = BY_TEXT[question_text]
        if question is LAPTOP_QUESTION:
            return declines(question)
        # The parking question has no answer in the handbook, and gets one all the same.
        return reply(question, Segment("Sixteen weeks.", (Citation(LEAVE_CHUNK),)))

    records = run_answers(QUESTIONS, DOCUMENTS, ask).records

    right = summarise_answers(QUESTIONS, records, judged={"correct": {"q001": True}})
    wrong = summarise_answers(QUESTIONS, records, judged={"correct": {"q001": False}})
    # The judge has ruled on correctness in this run, but not yet on this answer.
    waiting = summarise_answers(QUESTIONS, records, judged={"correct": {"q999": True}})
    # Only the other judge has run.
    other = summarise_answers(QUESTIONS, records, judged={"supported": {"q001": True}})

    grades = {item.question.id: item.scores for item in right.scored}
    # The answer that was given is correct: one of one answers.
    assert (right.summary["correct"].mean, right.summary["correct"].count) == (1.0, 1)
    # Of the two questions the handbook answers, one got a correct answer and one was
    # declined, which counts as no correct answer.
    assert (grades["q001"]["answered_correctly"], grades["q002"]["answered_correctly"]) == (
        1.0,
        0.0,
    )
    assert right.summary["answered_correctly"].mean == 0.5
    assert right.summary["answered_correctly"].count == 2
    # A question the handbook does not answer has no correct answer to give.
    assert "correct" not in grades["u001"]
    assert "answered_correctly" not in grades["u001"]
    assert (wrong.summary["correct"].mean, wrong.summary["answered_correctly"].mean) == (0.0, 0.0)
    # An answer that is still to be judged is counted neither way. And while one is
    # waiting, nothing is said about how many questions got a correct answer: the reply
    # that declined would be counted and the answer would not, which is half the story.
    assert "correct" not in waiting.summary
    assert "answered_correctly" not in waiting.summary
    assert "answered_correctly" not in other.summary


def test_an_assistant_that_does_everything_wrong_scores_zero() -> None:
    def careless(question_text: str) -> Answer:
        question = BY_TEXT[question_text]
        if question.type == "unanswerable":
            return reply(question, Segment("Fifty dollars."))
        return declines(question)

    report = summarise_answers(QUESTIONS, run_answers(QUESTIONS, DOCUMENTS, careless).records)

    means = {name: estimate.mean for name, estimate in report.summary.items()}
    assert means == {
        "right_call": 0.0,
        "answered": 0.0,
        "declined": 0.0,
        "evidence_in_context": 1.0,
        "fully_cited": 0.0,
    }


def test_groups_show_where_a_score_comes_from() -> None:
    def ask(question_text: str) -> Answer:
        question = BY_TEXT[question_text]
        if question is LAPTOP_QUESTION:
            # The search missed the evidence, and the assistant declined.
            return declines(question, given=(LEAVE_CHUNK,))
        return good_reply(question)

    report = summarise_answers(QUESTIONS, run_answers(QUESTIONS, DOCUMENTS, ask).records)

    evidence = report.by_group["evidence"]
    assert (evidence["given"]["answered"].mean, evidence["given"]["answered"].count) == (1.0, 1)
    assert evidence["not given"]["answered"].mean == 0.0
    assert set(report.by_group["unanswerable_kind"]) == {"near-miss"}
    assert report.by_group["unanswerable_kind"]["near-miss"]["declined"].mean == 1.0
    assert set(report.by_group["origin"]) == {"model-blind", "model-drafted"}
    assert report.by_group["origin"]["model-blind"]["answered"].mean == 0.0


def test_timing_and_tokens_leave_out_replies_that_never_reached_the_model() -> None:
    def ask(question_text: str) -> Answer:
        question = BY_TEXT[question_text]
        if question is PARKING_QUESTION:
            # Nothing was retrieved, so no request was sent.
            return Answer(
                question=question.question,
                segments=(Segment(NOT_FOUND_MESSAGE),),
                answered=False,
                model="fake-model",
                stop_reason="no_context",
            )
        seconds = 2.0 if question is LEAVE_QUESTION else 4.0
        return reply(question, Segment("x", (Citation(LEAVE_CHUNK),)), seconds=seconds)

    report = summarise_answers(QUESTIONS, run_answers(QUESTIONS, DOCUMENTS, ask).records)

    assert report.timing == {
        "median_seconds": 3.0,
        "p95_seconds": 4.0,
        "median_seconds_to_first_text": 1.0,
    }
    assert report.tokens == {
        "input_total": 3600.0,
        "output_total": 180.0,
        "input_mean": 1800.0,
        "output_mean": 90.0,
    }


def test_the_slowest_reply_in_twenty_is_the_95_percent_mark() -> None:
    many = [
        LEAVE_QUESTION.model_copy(update={"id": f"q{n:03d}", "question": f"Question {n}?"})
        for n in range(1, 21)
    ]
    by_text = {question.question: n for n, question in enumerate(many, start=1)}

    def ask(question_text: str) -> Answer:
        return reply(
            LEAVE_QUESTION,
            Segment("x", (Citation(LEAVE_CHUNK),)),
            seconds=float(by_text[question_text]),
        )

    report = summarise_answers(many, run_answers(many, DOCUMENTS, ask).records)

    # Nineteen of the twenty replies took 19 seconds or less.
    assert report.timing["p95_seconds"] == 19.0
    assert report.timing["median_seconds"] == 10.5


def test_a_failure_of_a_question_that_was_later_answered_is_not_reported() -> None:
    run = run_answers(QUESTIONS, DOCUMENTS, careful)
    earlier = [FailedQuestion("q001", "GenerationError", "busy")]

    report = summarise_answers(QUESTIONS, run.records, earlier)

    assert report.failures == []
    assert report.not_asked == 0
    assert len(report.scored) == 3


def test_a_reply_without_a_time_to_first_text_is_left_out_of_that_median() -> None:
    records = dict(run_answers(QUESTIONS, DOCUMENTS, careful).records)
    records["q001"] = dataclasses.replace(records["q001"], seconds_to_first_text=None)
    records["q002"] = dataclasses.replace(records["q002"], seconds_to_first_text=3.0)

    report = summarise_answers(QUESTIONS, records)

    # The median of 3.0 and 1.0. Counting the missing one as zero would give 1.0.
    assert report.timing["median_seconds_to_first_text"] == 2.0


def test_a_summary_of_nothing_is_empty() -> None:
    report = summarise_answers(QUESTIONS, {})

    assert (report.summary, report.timing, report.tokens) == ({}, {}, {})
    assert report.not_asked == 3


# What a run writes


def test_a_run_is_written_in_the_layout_the_other_tools_read(tmp_path: Path) -> None:
    def ask(question_text: str) -> Answer:
        question = BY_TEXT[question_text]
        if question is LAPTOP_QUESTION:
            raise GenerationError("busy")
        return good_reply(question)

    run = run_answers(QUESTIONS, DOCUMENTS, ask)
    report = summarise_answers(QUESTIONS, run.records, run.failures)

    run_dir = write_answer_run(
        report, tmp_path, "baseline", "Answers by a fake model.", CONFIG, attempts=4
    )

    assert run_dir == tmp_path / "baseline"
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == ["q001", "u001"]
    assert rows[0] == {
        "prompt_id": "q001",
        "prompt": "How long is parental leave?",
        "fingerprint": LEAVE_QUESTION.fingerprint,
        "tags": ["answerable", "model-drafted", "review:pending"],
        "status": "ok",
        "grade": {
            "answered": 1.0,
            "right_call": 1.0,
            "evidence_in_context": 1.0,
            "cites_evidence": 1.0,
            "fully_cited": 1.0,
        },
        "latency_s": 2.0,
        "answered": True,
        "stop_reason": "end_turn",
        "input_tokens": 1800,
        "output_tokens": 90,
    }
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert errors == [{"question_id": "q002", "error_class": "GenerationError", "message": "busy"}]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["description"] == "Answers by a fake model."
    assert summary["config"] == CONFIG
    assert (summary["questions_scored"], summary["questions_failed"]) == (2, 1)
    assert (summary["questions_not_asked"], summary["attempts"]) == (0, 4)
    assert summary["metrics"]["right_call"] == {"mean": 1.0, "low": 1.0, "high": 1.0, "count": 2}
    assert summary["by_group"]["unanswerable_kind"]["near-miss"]["declined"]["count"] == 1
    assert summary["timing"]["median_seconds"] == 2.0
    assert len(summary["labels_fingerprint"]) == 16
    assert (run_dir / "change.md").read_text() == "Answers by a fake model.\n"
    declared = json.loads((tmp_path / "_state.json").read_text())["metrics"]
    assert [metric["id"] for metric in declared] == list(METRIC_IDS)

    trace = json.loads((run_dir / "traces" / "q001_rep0.json").read_text())
    assert [turn["role"] for turn in trace] == ["user", "tool_result", "assistant", "user"]
    assert "[2] EVIDENCE  Title  (people/leave.md)" in trace[1]["content"]
    assert trace[2]["content"] == "Sixteen weeks.[2]"
    assert "Reference answer: Sixteen weeks." in trace[3]["content"]
    refused = json.loads((run_dir / "traces" / "u001_rep0.json").read_text())
    assert "does not answer this (near-miss). No page mentions parking." in refused[3]["content"]


def test_the_fingerprint_of_the_labels_does_not_depend_on_the_order_of_the_questions(
    tmp_path: Path,
) -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records

    for name, questions in (("a", QUESTIONS), ("b", list(reversed(QUESTIONS)))):
        report = summarise_answers(questions, records)
        write_answer_run(report, tmp_path, name, "Same replies.", CONFIG, attempts=3)

    first, second = (
        json.loads((tmp_path / name / "summary.json").read_text())["labels_fingerprint"]
        for name in ("a", "b")
    )
    assert first == second


def test_traces_of_questions_that_are_no_longer_scored_are_removed(tmp_path: Path) -> None:
    records = run_answers(QUESTIONS, DOCUMENTS, careful).records
    write_answer_run(
        summarise_answers(QUESTIONS, records), tmp_path, "baseline", "All.", CONFIG, attempts=3
    )
    assert len(list((tmp_path / "baseline" / "traces").iterdir())) == 3

    write_answer_run(
        summarise_answers([LEAVE_QUESTION], records),
        tmp_path,
        "baseline",
        "One.",
        CONFIG,
        attempts=3,
    )

    assert [p.name for p in (tmp_path / "baseline" / "traces").iterdir()] == ["q001_rep0.json"]


def test_a_killed_write_never_leaves_half_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    open_run(run_dir, CONFIG)
    replaced: list[tuple[str, str, str]] = []
    real_replace = os.replace

    def watching_replace(source: Path, target: Path) -> None:
        # At the moment of the swap the old file is still whole, and the new one is ready.
        replaced.append((source.name, target.name, target.read_text()))
        real_replace(source, target)

    monkeypatch.setattr("handbook_rag.evals.stage.os.replace", watching_replace)

    add_attempts(run_dir, 7)

    # The new content is written beside the file and swapped in with one step.
    assert [(source, target) for source, target, _ in replaced] == [
        ("run.json.partial", "run.json")
    ]
    assert json.loads(replaced[0][2])["attempts"] == 0
    assert sorted(path.name for path in run_dir.iterdir()) == ["run.json"]
    assert attempts_so_far(run_dir) == 7
    assert (run_dir / "run.json").read_text().endswith("}\n")


def test_passages_are_kept_by_id_and_not_by_their_text(tmp_path: Path) -> None:
    # The corpus is not redistributed with the repository, and a saved run is committed.
    open_run(tmp_path, CONFIG)
    append_record(tmp_path, run_answers(QUESTIONS, DOCUMENTS, careful).records["q001"])

    saved = (tmp_path / "answers.jsonl").read_text()

    assert "chunk-people/leave.md" in saved
    assert "New parents get" not in saved
    assert Passage(1, "chunk-finance/expenses.md", "finance/expenses.md", "Title", ()) in (
        load_records(tmp_path / "answers.jsonl")["q001"].passages
    )


# Replies that a check of the reply declined, and replies of the fixed form alone

GAP_ONLY = f"{GAP_FORM} where to park."


def declined_by_the_check(question: GoldenQuestion, words: str) -> Answer:
    check = ReplyCheck(CITES_NOTHING, "declined", NOTES[CITES_NOTHING], words)
    return dataclasses.replace(declines(question), checks=(check,))


def saved(tmp_path: Path, *answers: tuple[GoldenQuestion, Answer]) -> dict[str, AnswerRecord]:
    path = tmp_path / "answers.jsonl"
    rows = []
    for question, answer in answers:
        spans = locate_evidence(question, {document.doc_id: document for document in DOCUMENTS})
        rows.append(json.dumps(dataclasses.asdict(record_answer(question, answer, spans))))
    path.write_text("".join(row + "\n" for row in rows))
    return load_records(path)


def test_a_saved_reply_of_the_gap_form_alone_that_a_check_declined_is_read_as_the_models_decline(
    tmp_path: Path,
) -> None:
    records = saved(
        tmp_path,
        (PARKING_QUESTION, declined_by_the_check(PARKING_QUESTION, GAP_ONLY)),
        (LEAVE_QUESTION, declined_by_the_check(LEAVE_QUESTION, "Sixteen weeks.")),
    )

    parking, leave = records[PARKING_QUESTION.id], records[LEAVE_QUESTION.id]
    assert (parking.checks, parking.model_text, parking.answered) == ((), GAP_ONLY, False)
    # A real answer that cited nothing stays a check's decline: it is the checks' cost.
    assert [check.name for check in leave.checks] == [CITES_NOTHING]
    assert leave.model_text == ""
    assert as_read_today(leave) == leave


def test_a_run_counts_what_the_checks_declined_apart_from_the_models_own_declines() -> None:
    gap = dataclasses.replace(declines(PARKING_QUESTION), model_text=GAP_ONLY)
    records = {
        LEAVE_QUESTION.id: record_answer(
            LEAVE_QUESTION, declined_by_the_check(LEAVE_QUESTION, "Sixteen weeks."), []
        ),
        LAPTOP_QUESTION.id: record_answer(LAPTOP_QUESTION, good_reply(LAPTOP_QUESTION), []),
        PARKING_QUESTION.id: record_answer(PARKING_QUESTION, gap, []),
    }

    report = summarise_answers(QUESTIONS, records)

    assert report.reply_checks == {CITES_NOTHING: {"answerable": [LEAVE_QUESTION.id]}}
    assert report.declined_in_the_gap_form == {"unanswerable": [PARKING_QUESTION.id]}
    # The decline of the right call is counted where it always was: the checks' cost.
    leave = next(item for item in report.scored if item.question is LEAVE_QUESTION)
    assert leave.scores["right_call"] == 0.0


def test_a_run_writes_the_split_only_when_there_is_one(tmp_path: Path) -> None:
    gap = dataclasses.replace(declines(PARKING_QUESTION), model_text=GAP_ONLY)
    with_gap = summarise_answers(
        QUESTIONS, {PARKING_QUESTION.id: record_answer(PARKING_QUESTION, gap, [])}
    )
    without = summarise_answers(
        QUESTIONS,
        {PARKING_QUESTION.id: record_answer(PARKING_QUESTION, declines(PARKING_QUESTION), [])},
    )

    write_answer_run(with_gap, tmp_path, "a", "d", {}, attempts=1)
    write_answer_run(without, tmp_path, "b", "d", {}, attempts=1)

    first = json.loads((tmp_path / "a" / "summary.json").read_text())
    second = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert first["declined_in_the_gap_form"] == {"unanswerable": [PARKING_QUESTION.id]}
    assert "reply_checks" not in first
    assert "declined_in_the_gap_form" not in second
    assert "reply_checks" not in second


def test_a_saved_reply_whose_dropped_closing_line_is_in_the_gap_form_keeps_its_check(
    tmp_path: Path,
) -> None:
    check = ReplyCheck(
        ADDRESS_IN_CLOSING_LINE,
        "dropped_line",
        NOTES[ADDRESS_IN_CLOSING_LINE],
        # Nothing but the form, read as one piece: were the check not looked at, this
        # would be taken for the model's own decline.
        f"{GAP_FORM} where the form is, see https://x.org/a",
    )
    answer = dataclasses.replace(good_reply(LEAVE_QUESTION), checks=(check,))

    (record,) = saved(tmp_path, (LEAVE_QUESTION, answer)).values()

    assert record.checks == (check,)
    assert record.model_text == ""


def test_the_time_until_the_first_words_were_passed_on_is_given_beside_the_first_token() -> None:
    records = dict(run_answers(QUESTIONS, DOCUMENTS, careful).records)
    records["q001"] = dataclasses.replace(records["q001"], seconds_to_first_shown=None)
    records["q002"] = dataclasses.replace(records["q002"], seconds_to_first_shown=4.0)
    records["u001"] = dataclasses.replace(records["u001"], seconds_to_first_shown=2.0)

    report = summarise_answers(QUESTIONS, records)

    # Only the replies that recorded it: a run saved before it was recorded has none.
    assert report.timing["median_seconds_to_first_shown"] == 3.0
    assert report.timing["p95_seconds_to_first_shown"] == 4.0


def test_the_median_and_the_95_percent_mark_of_the_first_words_passed_on() -> None:
    many = [
        LEAVE_QUESTION.model_copy(update={"id": f"q{n:03d}", "question": f"Question {n}?"})
        for n in range(1, 22)
    ]
    by_text = {question.question: n for n, question in enumerate(many, start=1)}

    def ask(question_text: str) -> Answer:
        n = by_text[question_text]
        answer = reply(LEAVE_QUESTION, Segment("x", (Citation(LEAVE_CHUNK),)))
        # One slow reply pulls the mean up and leaves the median where it is.
        return dataclasses.replace(answer, seconds_to_first_shown=100.0 if n == 21 else float(n))

    report = summarise_answers(many, run_answers(many, DOCUMENTS, ask).records)

    assert report.timing["median_seconds_to_first_shown"] == 11.0
    # 95% of 21 is 19.95, so the 20th of the replies in order, not the slowest.
    assert report.timing["p95_seconds_to_first_shown"] == 20.0


def test_the_replies_for_a_reader_are_those_a_check_changed_or_read_as_the_fixed_form(
    tmp_path: Path,
) -> None:
    dropped = ReplyCheck(
        ADDRESS_IN_CLOSING_LINE, "dropped_line", NOTES[ADDRESS_IN_CLOSING_LINE], "See x.org/a"
    )
    records = {
        LEAVE_QUESTION.id: record_answer(
            LEAVE_QUESTION, declined_by_the_check(LEAVE_QUESTION, "Sixteen weeks."), []
        ),
        LAPTOP_QUESTION.id: record_answer(
            LAPTOP_QUESTION,
            dataclasses.replace(good_reply(LAPTOP_QUESTION), checks=(dropped,)),
            [],
        ),
        # A question the handbook does not answer: not for the reader of line c.
        PARKING_QUESTION.id: record_answer(
            PARKING_QUESTION,
            dataclasses.replace(declines(PARKING_QUESTION), model_text=GAP_ONLY),
            [],
        ),
    }
    report = summarise_answers(QUESTIONS, records)

    rows = changed_replies(report)

    assert [row["question_id"] for row in rows] == [LEAVE_QUESTION.id, LAPTOP_QUESTION.id]
    leave, laptop = rows
    assert leave == {
        "question_id": LEAVE_QUESTION.id,
        "question": LEAVE_QUESTION.question,
        "reference_answer": LEAVE_QUESTION.reference_answer,
        "model_words": "Sixteen weeks.",
        "reply_as_shown": NOT_FOUND_MESSAGE,
        "done": ["cites_nothing: declined"],
    }
    assert laptop["model_words"] == "One thousand dollars.\nSee x.org/a"
    assert laptop["reply_as_shown"] == "One thousand dollars."
    # Written beside the run, with nothing of its totals; and gone when there is none.
    run_dir = write_answer_run(report, tmp_path, "v", "d", CONFIG, attempts=3)
    written = [
        json.loads(line) for line in (run_dir / CHANGED_REPLIES_FILE).read_text().splitlines()
    ]
    assert written == rows
    assert "right_call" not in (run_dir / CHANGED_REPLIES_FILE).read_text()
    empty = summarise_answers(
        QUESTIONS,
        {LEAVE_QUESTION.id: record_answer(LEAVE_QUESTION, good_reply(LEAVE_QUESTION), [])},
    )
    write_answer_run(empty, tmp_path, "v", "d", CONFIG, attempts=3)
    assert not (run_dir / CHANGED_REPLIES_FILE).exists()


def test_a_reply_read_as_the_fixed_form_is_given_to_the_reader_with_the_models_words() -> None:
    gap = dataclasses.replace(declines(LEAVE_QUESTION), model_text=f"{GAP_FORM} how long.")
    record = record_answer(LEAVE_QUESTION, gap, [])

    assert model_words(record) == f"{GAP_FORM} how long."
    (row,) = changed_replies(summarise_answers(QUESTIONS, {LEAVE_QUESTION.id: record}))
    assert row["done"] == ["read as the fixed form for a missing part: shown as the decline"]
    assert row["reply_as_shown"] == NOT_FOUND_MESSAGE


def test_the_models_words_beside_a_decline_are_given_whole() -> None:
    words = f"{NOT_FOUND_MESSAGE}\nSee https://x.org/a"
    check = ReplyCheck("words_beside_a_decline", "dropped_words", "A note.", words)
    record = record_answer(
        LEAVE_QUESTION, dataclasses.replace(declines(LEAVE_QUESTION), checks=(check,)), []
    )

    assert model_words(record) == words
