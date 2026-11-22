"""Measuring how far a judge agrees with labelled statements."""

from typing import cast

import pytest

from handbook_rag.evals.agreement import Pair, measure_agreement
from handbook_rag.evals.judge import Verdict


def pairs(*rows: tuple[str, int, str, str, str]) -> list[Pair]:
    return [
        Pair(case, number, cast(Verdict, expected), cast(Verdict, judged), kind)
        for case, number, expected, judged, kind in rows
    ]


def test_a_judge_that_matches_every_label_agrees_completely() -> None:
    agreement = measure_agreement(
        pairs(
            ("a", 1, "supported", "supported", "real"),
            ("a", 2, "no_claim", "no_claim", "real"),
            ("b", 1, "unsupported", "unsupported", "number"),
            ("c", 1, "supported", "supported", "real"),
        )
    )

    assert agreement.statements == 4
    assert agreement.same_verdict.mean == 1.0
    assert agreement.same_decision.mean == 1.0
    assert agreement.kappa == 1.0
    assert agreement.recall is not None
    assert (agreement.recall.mean, agreement.recall.count) == (1.0, 1)
    assert agreement.precision is not None
    assert (agreement.precision.mean, agreement.precision.count) == (1.0, 1)
    assert agreement.caught == {"number": (1, 1)}
    assert agreement.answers == 3
    assert agreement.same_answer.mean == 1.0


def test_recall_counts_the_faults_caught_and_precision_the_alarms_that_were_right() -> None:
    agreement = measure_agreement(
        pairs(
            # Four planted faults: three caught, one let through.
            ("f1", 1, "unsupported", "unsupported", "number"),
            ("f2", 1, "unsupported", "unsupported", "number"),
            ("f3", 1, "unsupported", "unsupported", "scope"),
            ("f4", 1, "unsupported", "supported", "scope"),
            # Six sound statements: one false alarm.
            ("r1", 1, "supported", "unsupported", "real"),
            ("r2", 1, "supported", "supported", "real"),
            ("r3", 1, "supported", "supported", "real"),
            ("r4", 1, "supported", "supported", "real"),
            ("r5", 1, "supported", "supported", "real"),
            ("r6", 1, "supported", "supported", "real"),
        )
    )

    assert agreement.recall is not None
    assert agreement.precision is not None
    assert (agreement.recall.mean, agreement.recall.count) == (0.75, 4)
    assert (agreement.precision.mean, agreement.precision.count) == (0.75, 4)
    assert agreement.same_decision.mean == 0.8
    assert agreement.caught == {"number": (2, 2), "scope": (1, 2)}
    assert agreement.matrix == {
        "supported": {"supported": 5, "unsupported": 1, "no_claim": 0},
        "unsupported": {"supported": 1, "unsupported": 3, "no_claim": 0},
        "no_claim": {"supported": 0, "unsupported": 0, "no_claim": 0},
    }
    # Observed agreement 0.8; by chance 0.4 * 0.4 + 0.6 * 0.6 = 0.52.
    assert agreement.kappa == pytest.approx((0.8 - 0.52) / (1 - 0.52))


def test_mixing_up_supported_and_no_claim_is_a_different_verdict_but_the_same_decision() -> None:
    agreement = measure_agreement(
        pairs(
            ("a", 1, "supported", "no_claim", "real"),
            ("a", 2, "no_claim", "supported", "real"),
            ("b", 1, "unsupported", "unsupported", "name"),
        )
    )

    # Neither mix-up changes whether an answer counts as supported.
    assert agreement.same_verdict.mean == pytest.approx(1 / 3)
    assert agreement.same_decision.mean == 1.0
    assert agreement.kappa == 1.0
    assert agreement.same_answer.mean == 1.0


def test_an_answer_is_judged_unsupported_when_any_of_its_statements_is() -> None:
    agreement = measure_agreement(
        pairs(
            # The fault is in statement 2, and the judge blamed statement 1: the answer
            # is failed either way, but neither statement is judged as labelled.
            ("a", 1, "supported", "unsupported", "real"),
            ("a", 2, "unsupported", "supported", "number"),
            # A sound answer the judge failed.
            ("b", 1, "supported", "supported", "real"),
            ("b", 2, "supported", "unsupported", "real"),
            # A sound answer the judge passed.
            ("c", 1, "supported", "supported", "real"),
        )
    )

    assert agreement.answers == 3
    assert agreement.same_answer.mean == pytest.approx(2 / 3)
    assert agreement.answer_recall is not None
    assert agreement.answer_precision is not None
    assert (agreement.answer_recall.mean, agreement.answer_recall.count) == (1.0, 1)
    assert (agreement.answer_precision.mean, agreement.answer_precision.count) == (0.5, 2)
    # Statement by statement the judge did much worse than answer by answer.
    assert agreement.recall is not None
    assert agreement.recall.mean == 0.0
    assert agreement.same_decision.mean == pytest.approx(2 / 5)


def test_without_any_fault_or_any_alarm_there_is_no_recall_or_precision_to_report() -> None:
    agreement = measure_agreement(
        pairs(("a", 1, "supported", "supported", "real"), ("b", 1, "no_claim", "supported", "real"))
    )

    assert agreement.recall is None
    assert agreement.precision is None
    assert agreement.answer_recall is None
    assert agreement.answer_precision is None
    assert agreement.caught == {}
    # Full agreement that chance alone would also give: nothing was tested.
    assert agreement.kappa == 1.0


def test_a_judge_that_never_raises_an_alarm_is_no_better_than_chance() -> None:
    agreement = measure_agreement(
        pairs(
            ("a", 1, "unsupported", "supported", "number"),
            ("b", 1, "supported", "supported", "real"),
            ("c", 1, "supported", "supported", "real"),
        )
    )

    assert agreement.same_decision.mean == pytest.approx(2 / 3)
    assert agreement.kappa == pytest.approx(0.0, abs=1e-9)
    assert agreement.recall is not None
    assert agreement.recall.mean == 0.0
    assert agreement.precision is None


def test_a_judge_that_always_raises_an_alarm_on_sound_statements_only_is_worse_than_chance() -> (
    None
):
    agreement = measure_agreement(
        pairs(
            ("a", 1, "unsupported", "supported", "number"),
            ("b", 1, "supported", "unsupported", "real"),
        )
    )

    assert agreement.kappa == -1.0


def test_agreement_needs_something_to_compare() -> None:
    with pytest.raises(ValueError, match="at least one labelled statement"):
        measure_agreement([])
