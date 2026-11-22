"""Judging saved answers: which answers are read, what is saved, and what is used again."""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.agreement import measure_agreement
from handbook_rag.evals.answers import AnswerRecord, Passage, Statement
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.judge import (
    JudgedPassage,
    JudgedStatement,
    JudgeError,
    Judgement,
    StatementVerdict,
    SupportCase,
    Verdict,
)
from handbook_rag.evals.stage import SavedRunError
from handbook_rag.evals.support import (
    CHECK_FILE,
    agreement_headline,
    case_key,
    judge_check_stage,
    judge_record,
    judgement_row,
    labelled_pairs,
    load_judgements,
    load_labelled_cases,
    run_judge,
    saved_support,
    support_cases,
    support_stage,
)
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
CANADA = make_chunk("In Canada, statutory leave of up to 61 weeks applies.", doc_id="ca.md")
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
CHUNKS = {chunk.chunk_id: chunk for chunk in (LEAVE, CANADA, EXPENSES)}
CONFIG: dict[str, object] = {"judge_model": "fake-judge", "judge_instructions": "abc"}


def question(question_id: str, text: str, kind: str = "answerable") -> GoldenQuestion:
    row: dict[str, Any] = {
        "id": question_id,
        "question": text,
        "type": kind,
        "origin": "model-drafted",
    }
    if kind == "answerable":
        row["reference_answer"] = "An answer."
        row["evidence"] = [{"doc_id": "people/leave.md", "quote": "sixteen weeks"}]
    else:
        row["unanswerable_kind"] = "near-miss"
    return GoldenQuestion.model_validate(row)


LEAVE_QUESTION = question("q001", "How long is parental leave?")
LAPTOP_QUESTION = question("q002", "What is the laptop limit?")
PARKING_QUESTION = question("u001", "How much is the parking allowance?", "unanswerable")
QUESTIONS = [LEAVE_QUESTION, LAPTOP_QUESTION, PARKING_QUESTION]


def record(question_id: str, *statements: Statement, answered: bool = True) -> AnswerRecord:
    return AnswerRecord(
        question_id=question_id,
        fingerprint="f" * 16,
        passages=(
            Passage(1, EXPENSES.chunk_id, EXPENSES.doc_id, "Expenses", ()),
            Passage(2, LEAVE.chunk_id, LEAVE.doc_id, "Leave", (0,)),
            Passage(3, CANADA.chunk_id, CANADA.doc_id, "Canada", ()),
        ),
        statements=statements,
        answered=answered,
        stop_reason="end_turn",
        model="fake-model",
        input_tokens=1800,
        output_tokens=40,
        seconds=2.0,
    )


RECORDS = {
    "q001": record(
        "q001",
        Statement("Sixteen weeks.", (2,)),
        Statement(" \n\n", ()),
        Statement("  In Canada, up to 61 weeks also applies.  ", (2, 3)),
        Statement(" The handbook does not say more.", ()),
    ),
    "q002": record("q002", Statement("One thousand dollars.", (1,))),
    "u001": record("u001", Statement("I could not find this in the handbook.", ()), answered=False),
}


class ScriptedJudge:
    """Gives every statement the verdict it was told to, and notes what it was asked."""

    model = "fake-judge"

    def __init__(
        self,
        verdicts: dict[tuple[str, int], str] | None = None,
        *,
        failing: dict[str, int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._verdicts = verdicts or {}
        self._failing = dict(failing or {})
        self._error = error
        self.asked: list[str] = []

    def judge(self, case: SupportCase) -> Judgement:
        self.asked.append(case.case_id)
        if self._error is not None:
            raise self._error
        if self._failing.get(case.case_id, 0) > 0:
            self._failing[case.case_id] -= 1
            raise JudgeError("The judge's reply was not in the agreed form: bad json")
        return Judgement(
            case_id=case.case_id,
            verdicts=tuple(
                StatementVerdict(
                    s.number,
                    verdict(self._verdicts.get((case.case_id, s.number), "supported")),
                    f"Reason {s.number}.",
                )
                for s in case.statements
            ),
            model="fake-judge",
            input_tokens=900,
            output_tokens=60,
            seconds=1.5,
        )


def verdict(name: str) -> Verdict:
    assert name in ("supported", "unsupported", "no_claim")
    return name  # type: ignore[return-value]


# Which answers the judge reads


def test_an_answer_is_read_with_the_passages_it_cites_and_its_statements_numbered() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)

    assert [case.case_id for case in cases] == ["q001", "q002"]
    assert cases[0] == SupportCase(
        case_id="q001",
        question="How long is parental leave?",
        # Passage 1 was given to the model but is not cited, so the judge is not shown it.
        passages=(
            JudgedPassage(2, LEAVE.context_header, LEAVE.text),
            JudgedPassage(3, CANADA.context_header, CANADA.text),
        ),
        # Text without words is not a statement, and the others are numbered from 1.
        statements=(
            JudgedStatement(1, "Sixteen weeks.", (2,)),
            JudgedStatement(2, "In Canada, up to 61 weeks also applies.", (2, 3)),
            JudgedStatement(3, "The handbook does not say more.", ()),
        ),
    )
    assert cases[1].passages == (JudgedPassage(1, EXPENSES.context_header, EXPENSES.text),)


def test_a_reply_that_declined_and_a_question_without_a_reply_are_not_read() -> None:
    cases = support_cases(QUESTIONS, {"u001": RECORDS["u001"]}, CHUNKS)

    assert cases == []


def test_a_cited_passage_that_is_not_in_the_corpus_stops_the_stage() -> None:
    without_leave = {EXPENSES.chunk_id: EXPENSES, CANADA.chunk_id: CANADA}

    with pytest.raises(DatasetError, match=r"q001 cites passage .* not in the corpus"):
        support_cases(QUESTIONS, RECORDS, without_leave)


def test_the_key_of_a_case_changes_with_anything_the_judge_reads() -> None:
    (case, _) = support_cases(QUESTIONS, RECORDS, CHUNKS)
    replace = dataclasses.replace
    variants = [
        replace(case, question="How long is leave?"),
        replace(case, passages=(case.passages[0],)),
        replace(case, passages=(replace(case.passages[0], text="Other words."), case.passages[1])),
        replace(case, passages=(replace(case.passages[0], title="Other"), case.passages[1])),
        replace(case, statements=case.statements[:2]),
        replace(
            case, statements=(replace(case.statements[0], text="Six weeks."), *case.statements[1:])
        ),
        replace(case, statements=(replace(case.statements[0], cites=(3,)), *case.statements[1:])),
    ]

    keys = {case_key(variant) for variant in variants}

    assert len(keys) == len(variants)
    assert case_key(case) not in keys
    # The id is a name, not something the judge reads.
    assert case_key(replace(case, case_id="another-name")) == case_key(case)
    assert len(case_key(case)) == 16


# Labelled answers


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def labelled_row(case_id: str = "q001", **changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": case_id,
        "question_id": "q001",
        "kind": "real",
        "question": "How long is parental leave?",
        "passages": [
            {"number": 2, "chunk_id": LEAVE.chunk_id, "text_sha256": text_hash(LEAVE.text)},
        ],
        "statements": [
            {"number": 1, "text": "Sixteen weeks.", "cites": [2], "expected": "supported"},
            {"number": 2, "text": "It can be split.", "cites": [2], "expected": "unclear"},
        ],
    }
    return {**row, **changes}


PLANTED_ROW = labelled_row(
    "q001-v1",
    kind="number",
    statements=[
        {
            "number": 1,
            "text": "Eighteen weeks.",
            "cites": [2],
            "expected": "unsupported",
            "planted": "number",
            "change": "sixteen became eighteen",
        },
        {"number": 2, "text": "It can be split.", "cites": [2], "expected": "unclear"},
    ],
)


def write_cases(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "\n")
    return path


def test_labelled_answers_are_read_with_their_passages_fetched_from_the_corpus(
    tmp_path: Path,
) -> None:
    real, planted = load_labelled_cases(write_cases(tmp_path, labelled_row(), PLANTED_ROW), CHUNKS)

    assert real.case == SupportCase(
        "q001",
        "How long is parental leave?",
        (JudgedPassage(2, LEAVE.context_header, LEAVE.text),),
        (JudgedStatement(1, "Sixteen weeks.", (2,)), JudgedStatement(2, "It can be split.", (2,))),
    )
    assert (real.kind, real.expected, real.planted) == (
        "real",
        {1: "supported", 2: "unclear"},
        {},
    )
    assert (planted.kind, planted.expected, planted.planted) == (
        "number",
        {1: "unsupported", 2: "unclear"},
        {1: "number"},
    )


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        (
            labelled_row(passages=[{"number": 2, "chunk_id": "gone", "text_sha256": "0" * 16}]),
            "names passage gone, which is not in the corpus",
        ),
        (
            labelled_row(
                passages=[{"number": 2, "chunk_id": LEAVE.chunk_id, "text_sha256": "0" * 16}]
            ),
            "has changed since q001 was labelled",
        ),
        (
            labelled_row(statements=[{"number": 1, "text": "x", "cites": [2], "expected": "true"}]),
            r"unknown verdict: \['true'\]",
        ),
        (labelled_row(statements=[{"number": 1, "text": "x"}]), "not a labelled answer"),
        ({"case": "q001"}, "not a labelled answer"),
    ],
)
def test_a_labelled_answer_that_cannot_be_trusted_stops_the_check(
    tmp_path: Path, row: dict[str, Any], problem: str
) -> None:
    with pytest.raises(DatasetError, match=rf"cases\.jsonl:1: .*{problem}"):
        load_labelled_cases(write_cases(tmp_path, row), CHUNKS)


def test_other_faults_in_the_file_of_labelled_answers_are_reported(tmp_path: Path) -> None:
    twice = write_cases(tmp_path, labelled_row(), labelled_row())
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2: the id q001 is used twice"):
        load_labelled_cases(twice, CHUNKS)

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n")
    with pytest.raises(DatasetError, match=r"broken\.jsonl:1: not a labelled answer"):
        load_labelled_cases(broken, CHUNKS)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")
    with pytest.raises(DatasetError, match="holds no labelled answer"):
        load_labelled_cases(empty, CHUNKS)

    with pytest.raises(DatasetError, match="labelled answers not found"):
        load_labelled_cases(tmp_path / "missing.jsonl", CHUNKS)


# Saving judgements


def test_a_saved_judgement_comes_back_as_it_was_under_the_key_of_what_was_judged(
    tmp_path: Path,
) -> None:
    (case, other) = support_cases(QUESTIONS, RECORDS, CHUNKS)
    judgement = ScriptedJudge({("q001", 2): "unsupported", ("q001", 3): "no_claim"}).judge(case)
    stage = support_stage(tmp_path)
    stage.open(CONFIG)

    stage.append(judgement_row(case, judgement))
    stage.append(judgement_row(other, ScriptedJudge().judge(other)))

    saved = load_judgements(stage.rows_path)
    assert set(saved) == {case_key(case), case_key(other)}
    assert saved[case_key(case)] == judgement
    assert [v.verdict for v in saved[case_key(case)].verdicts] == [
        "supported",
        "unsupported",
        "no_claim",
    ]
    assert not saved[case_key(case)].supported


@pytest.mark.parametrize(
    "row",
    [
        {"case": "q001"},
        {
            "case": "q001",
            "key": "k",
            "verdicts": [{"number": 1, "verdict": "perhaps", "reason": ""}],
            "model": "m",
            "input_tokens": 1,
            "output_tokens": 1,
            "seconds": 1.0,
        },
    ],
)
def test_a_saved_line_that_is_not_a_judgement_is_reported(tmp_path: Path, row: object) -> None:
    path = tmp_path / "support.jsonl"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(SavedRunError, match=r"support\.jsonl:1 is not a saved judgement"):
        load_judgements(path)


def test_the_two_judge_stages_keep_their_files_apart_and_say_what_to_do(tmp_path: Path) -> None:
    tmp_path = tmp_path / "run"
    support, check = support_stage(tmp_path), judge_check_stage(tmp_path)

    assert (support.rows_file, support.state_file) == ("support.jsonl", "support.json")
    assert (check.rows_file, check.state_file) == ("judgements.jsonl", "run.json")
    (case, _) = support_cases(QUESTIONS, RECORDS, CHUNKS)
    support.open(CONFIG)
    support.append(judgement_row(case, ScriptedJudge().judge(case)))
    (tmp_path / CHECK_FILE).write_text("{}")

    with pytest.raises(SavedRunError, match="use the judge that made them") as refused:
        support.check({**CONFIG, "judge_model": "another-judge"})
    assert "other settings (judge_model)" in str(refused.value)

    # Restarting the judging of a run removes its verdicts and the note of their check.
    support.open({**CONFIG, "judge_model": "another-judge"}, restart=True)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["support.json"]


# The run


def test_every_answer_is_judged_once_and_handed_over_as_new() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    judge = ScriptedJudge({("q001", 2): "unsupported"})
    kept: list[tuple[str, bool]] = []

    run = run_judge(
        cases, judge, on_judgement=lambda case, _j, new: kept.append((case.case_id, new))
    )

    assert sorted(judge.asked) == ["q001", "q002"]
    assert sorted(kept) == [("q001", True), ("q002", True)]
    assert {name: j.supported for name, j in run.judgements.items()} == {
        "q001": False,
        "q002": True,
    }
    assert (run.attempts, run.reused, run.failures, run.stopped) == (2, 0, [], None)


def test_a_saved_verdict_on_exactly_the_same_answer_is_used_again() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    first = run_judge(cases, ScriptedJudge({("q001", 2): "unsupported"}))
    saved = {case_key(case): first.judgements[case.case_id] for case in cases}
    judge = ScriptedJudge()
    kept: list[tuple[str, bool]] = []

    second = run_judge(
        cases,
        judge,
        saved=saved,
        on_judgement=lambda case, _j, new: kept.append((case.case_id, new)),
    )

    assert judge.asked == []
    assert kept == [("q001", False), ("q002", False)]
    assert (second.attempts, second.reused) == (0, 2)
    assert second.judgements == first.judgements


def test_an_answer_that_reads_differently_is_judged_again() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    first = run_judge(cases, ScriptedJudge())
    saved = {case_key(case): first.judgements[case.case_id] for case in cases}
    # The question was asked again and the reply now says something else.
    changed = {**RECORDS, "q002": record("q002", Statement("Two thousand dollars.", (1,)))}
    judge = ScriptedJudge({("q002", 1): "unsupported"})

    second = run_judge(support_cases(QUESTIONS, changed, CHUNKS), judge, saved=saved)

    assert judge.asked == ["q002"]
    assert (second.attempts, second.reused) == (1, 1)
    assert not second.judgements["q002"].supported


def test_answers_are_judged_in_a_fixed_order_that_does_not_follow_the_file() -> None:
    many = [
        SupportCase(f"q{n:03d}", "A question?", (), (JudgedStatement(1, "A statement.", ()),))
        for n in range(30)
    ]
    forwards, backwards = ScriptedJudge(), ScriptedJudge()

    run_judge(many, forwards)
    run_judge(list(reversed(many)), backwards)

    assert forwards.asked == backwards.asked
    assert sorted(forwards.asked) == [case.case_id for case in many]
    assert forwards.asked != sorted(forwards.asked)


def test_a_reply_that_is_not_a_set_of_verdicts_is_tried_once_more_and_never_read_as_supported() -> (
    None
):
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    seen: list[str] = []

    once = run_judge(
        cases, ScriptedJudge(failing={"q001": 1}), on_failure=lambda f: seen.append(f.question_id)
    )
    twice = run_judge(cases, ScriptedJudge(failing={"q001": 2}))

    assert (once.attempts, once.failures, seen) == (3, [], ["q001"])
    assert set(once.judgements) == {"q001", "q002"}
    # Two answers, and one more try for the one whose reply could not be read.
    assert twice.attempts == 3
    assert [failure.question_id for failure in twice.failures] == ["q001"]
    assert twice.failures[0].error_class == "JudgeError"
    # The answer the judge could not rule on has no verdict at all.
    assert set(twice.judgements) == {"q002"}


def test_the_judging_stops_itself_when_answers_fail_one_after_another() -> None:
    many = [
        SupportCase(f"q{n:03d}", "A question?", (), (JudgedStatement(1, "A statement.", ()),))
        for n in range(10)
    ]
    judge = ScriptedJudge(error=JudgeError("The Gemini API returned an error (429): Quota."))

    run = run_judge(many, judge)

    assert len(judge.asked) == 3
    assert run.stopped is not None
    assert "3 answers in a row failed" in run.stopped
    assert "429" in run.stopped
    assert run.judgements == {}


def test_a_limit_on_requests_and_a_wait_before_each_apply_to_the_judge_too() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    waits: list[str] = []

    run = run_judge(cases, ScriptedJudge(), max_attempts=1, wait=lambda: waits.append("wait"))

    assert (run.attempts, waits) == (1, ["wait"])
    assert run.stopped == "the limit of 1 requests for this run was reached"
    assert len(run.judgements) == 1


def test_a_fault_in_our_own_code_is_not_counted_as_a_failed_request() -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)

    with pytest.raises(TypeError, match="our own mistake"):
        run_judge(cases, ScriptedJudge(error=TypeError("our own mistake")))


# From verdicts to scores


def test_each_labelled_statement_is_paired_with_the_judges_verdict(tmp_path: Path) -> None:
    labelled = load_labelled_cases(write_cases(tmp_path, labelled_row(), PLANTED_ROW), CHUNKS)
    judge = ScriptedJudge({("q001-v1", 1): "unsupported", ("q001", 2): "unsupported"})
    run = run_judge([case.case for case in labelled], judge)

    pairs = labelled_pairs(labelled, run.judgements)

    # The two statements labelled unclear are left out, whatever the judge said of them.
    assert [(p.case_id, p.statement, p.expected, p.judged, p.kind) for p in pairs] == [
        ("q001", 1, "supported", "supported", "real"),
        ("q001-v1", 1, "unsupported", "unsupported", "number"),
    ]
    agreement = measure_agreement(pairs)
    assert agreement.caught == {"number": (1, 1)}


def test_an_answer_the_judge_has_not_ruled_on_is_left_out_of_the_comparison(
    tmp_path: Path,
) -> None:
    labelled = load_labelled_cases(write_cases(tmp_path, labelled_row(), PLANTED_ROW), CHUNKS)
    run = run_judge([labelled[0].case], ScriptedJudge())

    assert [pair.case_id for pair in labelled_pairs(labelled, run.judgements)] == ["q001"]
    assert labelled_pairs(labelled, {}) == []


def test_only_verdicts_on_the_answers_as_they_now_read_count_for_a_run(tmp_path: Path) -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    stage = support_stage(tmp_path)
    stage.open(CONFIG)
    for case in cases:
        stage.append(judgement_row(case, ScriptedJudge().judge(case)))
    changed = {**RECORDS, "q002": record("q002", Statement("Two thousand dollars.", (1,)))}

    assert set(saved_support(tmp_path, cases)) == {"q001", "q002"}
    assert set(saved_support(tmp_path, support_cases(QUESTIONS, changed, CHUNKS))) == {"q001"}
    assert saved_support(tmp_path / "nowhere", cases) == {}


def test_the_record_of_the_judge_says_what_it_read_and_cost_and_how_it_was_checked(
    tmp_path: Path,
) -> None:
    cases = support_cases(QUESTIONS, RECORDS, CHUNKS)
    stage = support_stage(tmp_path)
    stage.open(CONFIG)
    stage.add_attempts(3)
    run = run_judge(cases, ScriptedJudge({("q001", 2): "unsupported", ("q001", 3): "no_claim"}))
    for case in cases:
        stage.append(judgement_row(case, run.judgements[case.case_id]))
    judged = saved_support(tmp_path, cases)

    assert judge_record(tmp_path, {}) is None
    unchecked = judge_record(tmp_path, judged)
    assert unchecked == {
        "config": CONFIG,
        "attempts": 3,
        "answers_judged": 2,
        "statements": {"supported": 2, "unsupported": 1, "no_claim": 1},
        "input_tokens": 1800,
        "output_tokens": 120,
        "median_seconds": 1.5,
        "check": None,
    }

    agreement = {"recall": {"mean": 0.9, "low": 0.8, "high": 1.0, "count": 41}}
    check = {"run": "evals/runs/judge/baseline", "config": CONFIG, "agreement": agreement}
    (tmp_path / CHECK_FILE).write_text(json.dumps(check))
    checked = judge_record(tmp_path, judged)
    assert checked is not None
    assert checked["check"] == {"run": "evals/runs/judge/baseline", "agreement": agreement}

    # A check of another judge says nothing about this one.
    (tmp_path / CHECK_FILE).write_text(
        json.dumps({**check, "config": {**CONFIG, "judge_instructions": "reworded"}})
    )
    other = judge_record(tmp_path, judged)
    assert other is not None
    assert other["check"] is None


def test_the_figures_quoted_beside_a_score_can_be_saved_as_json(tmp_path: Path) -> None:
    labelled = load_labelled_cases(write_cases(tmp_path, labelled_row(), PLANTED_ROW), CHUNKS)
    run = run_judge(
        [case.case for case in labelled], ScriptedJudge({("q001-v1", 1): "unsupported"})
    )

    headline = agreement_headline(measure_agreement(labelled_pairs(labelled, run.judgements)))

    assert json.loads(json.dumps(headline)) == headline
    assert headline["statements"] == 2
    assert headline["recall"] == {"mean": 1.0, "low": 1.0, "high": 1.0, "count": 1}
    assert headline["caught"] == {"number": [1, 1]}
    assert headline["kappa"] == 1.0
    assert headline["answers"] == 2
