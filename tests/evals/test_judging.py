"""What every judged criterion shares, tested where the two criteria differ.

The run of the judge, saved verdicts and their reuse are tested through the first
criterion in ``test_support.py`` and through the second in ``test_correctness.py``.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from handbook_rag.evals.agreement import Pair, measure_agreement
from handbook_rag.evals.correctness import CORRECT, CorrectnessCase
from handbook_rag.evals.judge import (
    JudgedPassage,
    JudgedStatement,
    Judgement,
    StatementVerdict,
    SupportCase,
)
from handbook_rag.evals.judging import (
    agreement_headline,
    judge_check_stage,
    judge_record,
    judgement_row,
    load_judgements,
    saved_verdicts,
)
from handbook_rag.evals.stage import SavedRunError
from handbook_rag.evals.support import SUPPORT

ANSWER = CorrectnessCase("q001", "How long is leave?", "Sixteen weeks.", "Sixteen weeks, paid.")
STATEMENTS = SupportCase(
    "q001",
    "How long is leave?",
    (JudgedPassage(2, "Leave", "New parents get sixteen weeks of paid leave."),),
    (JudgedStatement(1, "Sixteen weeks.", (2,)), JudgedStatement(2, "It is paid.", (2,))),
)


def judgement(*verdicts: str) -> Judgement:
    return Judgement(
        "q001",
        tuple(StatementVerdict(n, verdict, "A reason.") for n, verdict in enumerate(verdicts, 1)),
        "fake-judge",
        300,
        20,
        1.5,
    )


def test_an_answer_passes_only_when_no_verdict_on_it_is_a_fault() -> None:
    # One statement that is not backed is enough to fail an answer, whatever the others are.
    assert SUPPORT.passed(judgement("supported", "supported"))
    assert SUPPORT.passed(judgement("supported", "no_claim"))
    assert not SUPPORT.passed(judgement("supported", "unsupported"))
    assert not SUPPORT.passed(judgement("unsupported", "supported", "supported"))
    assert not CORRECT.passed(judgement("partial"))
    # What counts against an answer is the criterion's own business.
    assert SUPPORT.faults == {"unsupported"}
    assert CORRECT.faults == {"partial", "incorrect"}


def test_a_saved_row_without_a_verdict_is_reported(tmp_path: Path) -> None:
    # It would otherwise pass as an answer that nothing counts against.
    path = tmp_path / "correct.jsonl"
    row = {**judgement_row(ANSWER, judgement("correct")), "verdicts": []}
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(SavedRunError, match=r"correct\.jsonl:1 is not a saved judgement: it holds"):
        load_judgements(path, CORRECT.verdicts)


def test_the_saved_verdicts_of_one_criterion_are_never_read_as_the_others(tmp_path: Path) -> None:
    # Someone copied the first judge's verdicts into the second judge's file.
    support = SUPPORT.run_stage(tmp_path)
    support.open({"judge_model": "fake-judge"})
    support.append(judgement_row(STATEMENTS, judgement("supported", "unsupported")))
    (tmp_path / CORRECT.rows_file).write_text(support.rows_path.read_text())

    assert set(saved_verdicts(SUPPORT, tmp_path, [STATEMENTS])) == {"q001"}
    with pytest.raises(SavedRunError, match="is not a saved judgement: unknown verdict"):
        saved_verdicts(CORRECT, tmp_path, [ANSWER])


def test_each_judge_is_described_with_the_check_of_its_own_criterion(tmp_path: Path) -> None:
    config = {"judge_model": "fake-judge"}
    for criterion, case, verdicts in (
        (SUPPORT, STATEMENTS, ("supported", "supported")),
        (CORRECT, ANSWER, ("correct",)),
    ):
        stage = criterion.run_stage(tmp_path)
        stage.open(config)
        stage.append(judgement_row(case, judgement(*verdicts)))
    # Only the second judge has a check on record.
    check = {"run": "evals/runs/judge-correct/baseline", "config": config, "agreement": {"x": 1}}
    (tmp_path / "correct_check.json").write_text(json.dumps(check))

    correct = judge_record(CORRECT, tmp_path, saved_verdicts(CORRECT, tmp_path, [ANSWER]))
    support = judge_record(SUPPORT, tmp_path, saved_verdicts(SUPPORT, tmp_path, [STATEMENTS]))

    assert correct is not None
    assert support is not None
    assert correct["check"] == {
        "run": "evals/runs/judge-correct/baseline",
        "answers": None,
        "agreement": {"x": 1},
    }
    assert support["check"] is None
    assert correct["verdicts"] == {"correct": 1, "partial": 0, "incorrect": 0}
    assert support["verdicts"] == {"supported": 2, "unsupported": 0, "no_claim": 0}


def test_restarting_a_check_discards_everything_it_wrote(tmp_path: Path) -> None:
    run_dir = tmp_path / "baseline"
    stage = judge_check_stage(run_dir)
    stage.open({"judge_model": "fake-judge"})
    stage.append(judgement_row(ANSWER, judgement("correct")))
    for name in ("summary.json", "disagreements.jsonl"):
        (run_dir / name).write_text("{}\n")

    stage.open({"judge_model": "another-judge"}, restart=True)

    # Nothing is left that describes the discarded verdicts.
    assert sorted(path.name for path in run_dir.iterdir()) == ["run.json"]


def test_kappa_is_quoted_to_four_places() -> None:
    pairs = [
        Pair("a", 1, "unsupported", "unsupported", "number"),
        Pair("b", 1, "unsupported", "supported", "number"),
        Pair("c", 1, "unsupported", "supported", "number"),
        Pair("d", 1, "supported", "supported"),
        Pair("e", 1, "supported", "supported"),
        Pair("f", 1, "supported", "supported"),
        Pair("g", 1, "supported", "supported"),
    ]
    agreement = measure_agreement(pairs, verdicts=SUPPORT.verdicts, faults=SUPPORT.faults)

    quoted = agreement_headline(agreement)["kappa"]

    # Observed 5 of 7; by chance (3/7)(1/7) + (4/7)(6/7) = 27/49. Kappa is 4/11.
    assert agreement.kappa == pytest.approx(4 / 11)
    assert quoted == 0.3636


def test_a_fingerprint_is_of_the_text_as_it_reads_in_any_script() -> None:
    # Pinned values. Saved verdicts are found by them, so they must not move.
    german = CorrectnessCase(
        "x",
        "Wie lange dauert der Elternurlaub?",
        "Sechzehn Wochen, voll bezahlt.",
        "Sechzehn Wochen – bezahlt, für Eltern.",  # noqa: RUF001
    )
    japanese = SupportCase(
        "x",
        "育児休暇は何週間ですか？",  # noqa: RUF001
        (JudgedPassage(2, "休暇 > 育児", "新しい親は16週間の有給休暇を取得できます。"),),
        (JudgedStatement(1, "16週間です。", (2,)),),
    )

    assert german.fingerprint == "c01632345cd884be"
    assert japanese.fingerprint == "38fa44a31629b89c"
    plain = dataclasses.replace(german, answer="Sechzehn Wochen - bezahlt, fur Eltern.")
    assert plain.fingerprint != german.fingerprint
