"""The 40 questions of the limit set, asked of the assistant: loading, asking, counting.

Every model here is a stand-in. No request is sent.
"""

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals import limit_questions
from handbook_rag.evals.answers import AnswerRecord, Statement
from handbook_rag.evals.dataset import DatasetError
from handbook_rag.evals.judge import Judgement, StatementVerdict
from handbook_rag.evals.limit_questions import (
    LimitQuestion,
    judged_cases,
    load_limit_questions,
    passages_of,
    run_questions,
    summarise_limits,
    write_limit_run,
)
from handbook_rag.evals.retrieval import FailedQuestion
from handbook_rag.evals.support import passage_hash
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer, Citation, ReplyCheck, Segment, TokenUsage
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk
from tests.support import StandInModel, make_chunk

ROOT = Path(__file__).parents[2]
LEAVE = make_chunk("Staff in Canada get sixteen weeks of paid leave.")
LAPTOP = make_chunk("Managers may expense a laptop up to one thousand dollars.", doc_id="f/l.md")
CHUNKS = {chunk.chunk_id: chunk for chunk in (LEAVE, LAPTOP)}


def row(case: str, kind: str, chunk: Chunk, question: str, **changes: Any) -> dict[str, Any]:
    line: dict[str, Any] = {
        "case": case,
        "kind": kind,
        "section": chunk.section,
        "limit_kind": "where" if chunk is LEAVE else "who",
        "limit": "in Canada" if chunk is LEAVE else "Managers",
        "limit_where": "text",
        "question": question,
        "passages": [
            {"number": 3, "chunk_id": chunk.chunk_id, "text_sha256": passage_hash(chunk.text)}
        ],
        "statements": [{"number": 1, "text": "A statement.", "cites": [3]}],
    }
    return {**line, **changes}


ROWS = [
    row("lim-001", "kept", LEAVE, "How long is parental leave in Canada?"),
    row("lim-002", "dropped", LAPTOP, "Can I expense a laptop?"),
    row("lim-003", "dropped", LEAVE, "How long is parental leave?"),
    row("lim-004", "harmless", LAPTOP, "Can I expense a laptop?"),
]


def write_rows(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in rows))
    return path


def questions(tmp_path: Path) -> list[LimitQuestion]:
    return load_limit_questions(write_rows(tmp_path, *ROWS))


def test_the_question_of_each_passage_s_dropped_answer_is_asked(tmp_path: Path) -> None:
    loaded = questions(tmp_path)

    assert [(q.case_id, q.question, q.chunk_id) for q in loaded] == [
        ("lim-002", "Can I expense a laptop?", LAPTOP.chunk_id),
        ("lim-003", "How long is parental leave?", LEAVE.chunk_id),
    ]
    first = loaded[0]
    assert (first.limit_kind, first.limit, first.limit_where, first.section) == (
        "who",
        "Managers",
        "text",
        LAPTOP.section,
    )
    assert first.text_sha256 == passage_hash(LAPTOP.text)


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        (ROWS[:1], f"passage {LEAVE.chunk_id} has 0 answers of the kind dropped, not one"),
        (
            [*ROWS, row("lim-005", "dropped", LEAVE, "Leave?")],
            f"passage {LEAVE.chunk_id} has 2 answers of the kind dropped, not one",
        ),
    ],
)
def test_a_passage_asked_twice_or_not_at_all_is_refused(
    tmp_path: Path, rows: list[dict[str, Any]], problem: str
) -> None:
    with pytest.raises(DatasetError, match=problem):
        load_limit_questions(write_rows(tmp_path, *rows))


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"passages": []}, "cites one passage"),
        ({"question": "  "}, "expected text"),
        ({"limit_kind": None}, "expected text"),
        ({"section": 3}, "expected text"),
    ],
)
def test_a_line_that_is_not_a_labelled_answer_is_reported(
    tmp_path: Path, changes: dict[str, Any], problem: str
) -> None:
    broken = {**ROWS[1], **changes}

    with pytest.raises(DatasetError, match=rf"cases\.jsonl:2: not a labelled answer: .*{problem}"):
        load_limit_questions(write_rows(tmp_path, ROWS[0], broken, *ROWS[2:]))


def test_a_missing_or_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="limit answers not found"):
        load_limit_questions(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(DatasetError, match="holds no answer of the kind dropped"):
        load_limit_questions(empty)


def test_the_committed_set_gives_forty_questions_on_forty_passages() -> None:
    loaded = load_limit_questions(ROOT / limit_questions.CASES_PATH)
    labelled = [
        json.loads(line)
        for line in (ROOT / limit_questions.CASES_PATH).read_text().splitlines()
        if line.strip()
    ]
    dropped = {line["case"]: line for line in labelled if line["kind"] == "dropped"}

    assert len(loaded) == 40
    assert len({q.chunk_id for q in loaded}) == 40
    assert {q.case_id for q in loaded} == set(dropped)
    for question in loaded:
        assert question.question == dropped[question.case_id]["question"]
        assert question.limit == dropped[question.case_id]["limit"]


def test_the_fingerprint_changes_with_what_the_model_is_sent_and_nothing_else(
    tmp_path: Path,
) -> None:
    question = questions(tmp_path)[0]
    replaced = dataclasses.replace

    sent = [
        replaced(question, question="Another?"),
        replaced(question, chunk_id="other"),
        replaced(question, text_sha256="0" * 16),
    ]
    read = [replaced(question, limit="x", limit_kind="when", limit_where="title", section="s")]

    assert len({q.fingerprint for q in sent} | {question.fingerprint}) == 4
    assert {q.fingerprint for q in read} == {question.fingerprint}


def test_a_passage_that_has_changed_or_gone_is_refused(tmp_path: Path) -> None:
    loaded = questions(tmp_path)

    assert passages_of(loaded, CHUNKS) == {"lim-002": LAPTOP, "lim-003": LEAVE}
    with pytest.raises(DatasetError, match="which is not in the corpus"):
        passages_of(loaded, {LEAVE.chunk_id: LEAVE})
    changed = {**CHUNKS, LEAVE.chunk_id: dataclasses.replace(LEAVE, text="Other words.")}
    with pytest.raises(DatasetError, match="has changed since lim-003 was written"):
        passages_of(loaded, changed)


class Model(StandInModel):
    """Answers from the passage it is given, faults on purpose where told, and records."""

    def __init__(self, *, fails: Sequence[str] = ()) -> None:
        self.sent: list[tuple[str, list[str]]] = []
        self.fails = list(fails)

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.sent.append((question, [chunk.text for chunk in chunks]))
        if question in self.fails:
            self.fails.remove(question)
            raise GenerationError("The model is overloaded.")
        (chunk,) = chunks
        return Answer(
            question=question,
            segments=(Segment("Sixteen weeks.", (Citation(chunk),)),),
            answered=True,
            model="stand-in",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=500, output_tokens=20),
            seconds=1.0,
        )


def test_each_question_goes_to_the_generator_with_its_one_passage_and_no_search(
    tmp_path: Path,
) -> None:
    loaded = questions(tmp_path)
    model = Model()

    answer = limit_questions.asker(model, passages_of(loaded, CHUNKS))(loaded[1])

    assert model.sent == [("How long is parental leave?", [LEAVE.text])]
    assert [scored.chunk for scored in answer.retrieved] == [LEAVE]
    record = limit_questions.record_limit_reply(loaded[1], answer)
    assert (record.question_id, record.fingerprint) == ("lim-003", loaded[1].fingerprint)
    assert [(p.number, p.chunk_id, p.covers) for p in record.passages] == [
        (1, LEAVE.chunk_id, (0,))
    ]
    assert record.statements == (Statement("Sixteen weeks.", (1,)),)


def test_a_run_asks_what_it_lacks_and_a_failed_question_once_more(tmp_path: Path) -> None:
    loaded = questions(tmp_path)
    model = Model(fails=["Can I expense a laptop?"])
    ask = limit_questions.asker(model, passages_of(loaded, CHUNKS))

    run = run_questions(loaded, ask)

    assert sorted(run.records) == ["lim-002", "lim-003"]
    assert (run.attempts, run.failures, run.stopped) == (3, [], None)

    again = run_questions(loaded, ask, saved=run.records)
    assert again.attempts == 0
    # A reply to a question written otherwise since is asked again.
    reworded = [dataclasses.replace(loaded[0], question="May I buy a laptop?"), loaded[1]]
    assert run_questions(reworded, ask, saved=run.records).attempts == 1


def reply(question: LimitQuestion, text: str, *, answered: bool = True) -> AnswerRecord:
    """A reply as a run keeps it: an answer cites its one passage, a decline nothing."""
    chunk = CHUNKS[question.chunk_id]
    answer = Answer(
        question=question.question,
        segments=(Segment(text, (Citation(chunk),) if answered else ()),),
        answered=answered,
        model="stand-in",
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=500, output_tokens=20),
        seconds=1.0,
        retrieved=(ScoredChunk(chunk, 0.0),),
    )
    return limit_questions.record_limit_reply(question, answer)


def verdicts(case_id: str, *labels: str) -> Judgement:
    return Judgement(
        case_id=case_id,
        verdicts=tuple(
            StatementVerdict(number, label, f"Reason {number}.")
            for number, label in enumerate(labels, start=1)
        ),
        model="judge",
        input_tokens=900,
        output_tokens=60,
        seconds=1.5,
    )


def test_the_share_faulted_is_given_once_every_answer_has_a_verdict(tmp_path: Path) -> None:
    laptop_q, leave_q = questions(tmp_path)
    records = {
        laptop_q.case_id: reply(laptop_q, "You may expense a laptop up to one thousand dollars."),
        leave_q.case_id: reply(leave_q, "Sixteen weeks."),
    }
    cases = judged_cases([laptop_q, leave_q], records, CHUNKS)

    partial = summarise_limits(
        [laptop_q, leave_q], records, cases, {laptop_q.case_id: verdicts("lim-002", "unsupported")}
    )
    assert partial.waiting_for_the_judge == ["lim-003"]
    assert not partial.final
    assert partial.figures["all"].share_faulted is None
    assert (partial.figures["all"].judged, partial.figures["all"].faulted) == (1, 1)

    whole = summarise_limits(
        [laptop_q, leave_q],
        records,
        cases,
        {
            laptop_q.case_id: verdicts("lim-002", "unsupported"),
            leave_q.case_id: verdicts("lim-003", "supported"),
        },
    )
    assert whole.final
    figures = whole.figures["all"]
    assert (figures.questions, figures.replied, figures.answered, figures.judged) == (2, 2, 2, 2)
    assert figures.faulted == 1
    assert figures.share_faulted is not None
    assert figures.share_faulted.mean == 0.5
    assert whole.figures["who"].faulted == 1
    assert whole.figures["where"].faulted == 0
    assert [result.faulted for result in whole.results] == [True, False]


def test_a_decline_is_counted_and_not_judged_and_a_missing_reply_is_not_final(
    tmp_path: Path,
) -> None:
    laptop_q, leave_q = questions(tmp_path)
    records = {laptop_q.case_id: reply(laptop_q, NOT_FOUND_MESSAGE, answered=False)}
    cases = judged_cases([laptop_q, leave_q], records, CHUNKS)

    report = summarise_limits(
        [laptop_q, leave_q],
        records,
        cases,
        {},
        [FailedQuestion("lim-003", "GenerationError", "overloaded")],
    )

    assert cases == []
    assert (report.figures["all"].answered, report.figures["all"].judged) == (0, 0)
    assert report.waiting_for_the_judge == []
    assert [failure.question_id for failure in report.failures] == ["lim-003"]
    assert report.not_asked == 0
    assert not report.final


def test_a_run_writes_a_line_for_each_reply_with_its_verdicts_and_the_counts(
    tmp_path: Path,
) -> None:
    laptop_q, leave_q = questions(tmp_path)
    records = {
        laptop_q.case_id: reply(laptop_q, "You may expense a laptop."),
        leave_q.case_id: reply(leave_q, "Sixteen weeks."),
    }
    cases = judged_cases([laptop_q, leave_q], records, CHUNKS)
    report = summarise_limits(
        [laptop_q, leave_q],
        records,
        cases,
        {
            laptop_q.case_id: verdicts("lim-002", "unsupported"),
            leave_q.case_id: verdicts("lim-003", "supported"),
        },
    )

    write_limit_run(
        report,
        tmp_path,
        description="d",
        config={"answer_model": "stand-in"},
        attempts=2,
        judge_config={"judge_model": "judge"},
        judge_attempts=2,
    )

    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    first = rows[0]
    assert (first["case"], first["limit_kind"], first["limit"]) == ("lim-002", "who", "Managers")
    assert (first["answered"], first["judged"], first["faulted"]) == (True, True, True)
    assert first["reply"] == "You may expense a laptop.[1]"
    assert first["verdicts"] == [
        {
            "number": 1,
            "statement": "You may expense a laptop.",
            "verdict": "unsupported",
            "reason": "Reason 1.",
        }
    ]
    assert first["question_fingerprint"] == laptop_q.fingerprint
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["judge"] == {"config": {"judge_model": "judge"}, "attempts": 2}
    assert (summary["questions"], summary["questions_replied"], summary["final"]) == (2, 2, True)
    assert summary["figures"]["all"]["faulted"] == 1
    assert summary["figures"]["all"]["share_faulted"]["mean"] == 0.5
    assert summary["config"] == {"answer_model": "stand-in"}


def test_an_answer_that_cites_two_passages_is_refused(tmp_path: Path) -> None:
    two = {**ROWS[1], "passages": [*ROWS[1]["passages"], *ROWS[2]["passages"]]}

    with pytest.raises(DatasetError, match="cites one passage"):
        load_limit_questions(write_rows(tmp_path, ROWS[0], two, *ROWS[2:]))


def test_no_share_is_given_before_every_question_has_a_reply(tmp_path: Path) -> None:
    laptop_q, leave_q = questions(tmp_path)
    records = {laptop_q.case_id: reply(laptop_q, "You may expense a laptop.")}
    cases = judged_cases([laptop_q, leave_q], records, CHUNKS)

    report = summarise_limits(
        [laptop_q, leave_q], records, cases, {laptop_q.case_id: verdicts("lim-002", "supported")}
    )

    assert report.waiting_for_the_judge == []
    assert report.not_asked == 1
    assert report.figures["all"].share_faulted is None
    assert report.figures["all"].share_good is None
    assert report.figures["all"].good == 1


def test_an_answer_with_one_statement_faulted_among_others_is_faulted(tmp_path: Path) -> None:
    laptop_q, leave_q = questions(tmp_path)
    records = {
        laptop_q.case_id: reply(laptop_q, "You may expense a laptop."),
        leave_q.case_id: reply(leave_q, "Sixteen weeks."),
    }
    cases = judged_cases([laptop_q, leave_q], records, CHUNKS)

    report = summarise_limits(
        [laptop_q, leave_q],
        records,
        cases,
        {
            laptop_q.case_id: verdicts("lim-002", "supported", "unsupported"),
            leave_q.case_id: verdicts("lim-003", "supported", "supported"),
        },
    )

    assert [result.outcome for result in report.results] == ["faulted", "good"]
    assert report.figures["all"].faulted == 1


def test_tokens_leave_out_replies_that_needed_no_request(tmp_path: Path) -> None:
    laptop_q, leave_q = questions(tmp_path)
    nothing_sent = dataclasses.replace(
        reply(leave_q, NOT_FOUND_MESSAGE, answered=False),
        stop_reason="no_context",
        input_tokens=0,
        output_tokens=0,
    )
    records = {
        laptop_q.case_id: reply(laptop_q, "You may expense a laptop."),
        leave_q.case_id: nothing_sent,
    }

    report = summarise_limits([laptop_q, leave_q], records, [], {})

    assert report.tokens["input_mean"] == 500.0
    assert report.tokens["input_total"] == 500.0


def with_checks(record: AnswerRecord, *checks: tuple[str, str], **changes: Any) -> AnswerRecord:
    return dataclasses.replace(
        record,
        checks=tuple(ReplyCheck(name, action, "A note.", "words") for name, action in checks),
        **changes,
    )


def test_the_figure_that_counts_is_good_answers_out_of_every_question_with_each_cause(
    tmp_path: Path,
) -> None:
    rows = [
        row(f"lim-{n:03}", "dropped", chunk, f"Question {n}?")
        for n, chunk in enumerate([LEAVE, LAPTOP], start=1)
    ]
    loaded = load_limit_questions(write_rows(tmp_path, *rows))
    leave_q, laptop_q = loaded
    # One answer the judge faults nothing in; and one question that got no answer.
    records = {
        leave_q.case_id: reply(leave_q, "Sixteen weeks in Canada."),
        laptop_q.case_id: reply(laptop_q, NOT_FOUND_MESSAGE, answered=False),
    }
    cases = judged_cases(loaded, records, CHUNKS)

    report = summarise_limits(
        loaded, records, cases, {leave_q.case_id: verdicts("lim-001", "supported")}
    )

    figures = report.figures["all"]
    assert report.final
    assert (figures.good, figures.judged, figures.faulted) == (1, 1, 0)
    assert figures.share_faulted is not None
    assert figures.share_faulted.mean == 0.0
    assert figures.share_good is not None
    assert figures.share_good.mean == 0.5
    assert figures.outcomes["good"] == 1
    assert figures.outcomes["declined_by_the_model"] == 1
    assert list(figures.outcomes) == list(limit_questions.OUTCOMES)


@pytest.mark.parametrize(
    ("changes", "outcome"),
    [
        ({}, "declined_by_the_model"),
        ({"model_text": "The passages found do not say who."}, "declined_in_the_gap_form"),
        ({"checks": (("cites_nothing", "declined"),)}, "declined_by_a_check"),
        ({"checks": (("words_beside_a_decline", "dropped_words"),)}, "declined_by_the_model"),
        ({"stop_reason": "refusal"}, "unfinished"),
        (
            {"checks": (("unfinished_reply", "dropped_words"),), "stop_reason": "SAFETY"},
            "unfinished",
        ),
    ],
)
def test_each_cause_of_no_answer_is_named(
    tmp_path: Path, changes: dict[str, Any], outcome: str
) -> None:
    laptop_q, _ = questions(tmp_path)
    checks = changes.get("checks", ())
    rest = {name: value for name, value in changes.items() if name != "checks"}
    record = with_checks(reply(laptop_q, NOT_FOUND_MESSAGE, answered=False), *checks, **rest)
    result = limit_questions.LimitResult(laptop_q, record, None, None)

    assert result.outcome == outcome


def write_outcomes(run_dir: Path, outcomes: dict[str, str], *, final: bool = True) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"final": final}))
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps({"case": case, "outcome": o}) + "\n" for case, o in outcomes.items())
    )
    return run_dir


def test_two_final_runs_are_compared_question_by_question(tmp_path: Path) -> None:
    earlier = write_outcomes(
        tmp_path / "a", {"q1": "good", "q2": "declined_by_the_model", "q3": "faulted", "q4": "good"}
    )
    later = write_outcomes(
        tmp_path / "b", {"q1": "faulted", "q2": "good", "q3": "good", "q4": "good"}
    )

    comparison = limit_questions.compare_limit_runs(earlier, later)

    assert (comparison.better, comparison.worse, comparison.same) == (["q2", "q3"], ["q1"], 1)
    assert comparison.sign_test == 1.0
    with pytest.raises(DatasetError, match="not final"):
        limit_questions.compare_limit_runs(
            earlier, write_outcomes(tmp_path / "c", {"q1": "good"}, final=False)
        )
    with pytest.raises(DatasetError, match="other questions"):
        limit_questions.compare_limit_runs(earlier, write_outcomes(tmp_path / "d", {"q1": "good"}))
