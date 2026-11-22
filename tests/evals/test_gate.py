import json
from pathlib import Path

import pytest

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.gate import (
    GateError,
    evidence_pages,
    gate_problems,
    load_gate_config,
    select_pages,
    shown_with_minimum,
)
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.retrieval import FailedQuestion, QuestionResult, RetrievalReport
from handbook_rag.ingest.models import Document

REPOSITORY = Path(__file__).resolve().parents[2]


def page(doc_id: str) -> Document:
    return Document(
        doc_id, "Title", "https://example.com/", doc_id.split("/")[0], "a" * 40, "Text."
    )


PAGES = [page(f"section/page-{number:02d}.md") for number in range(20)]
QUESTION = GoldenQuestion.model_validate(
    {
        "id": "q001",
        "question": "What does page seven say?",
        "reference_answer": "Text.",
        "evidence": [
            {"doc_id": "section/page-07.md", "quote": "Text."},
            {"doc_id": "section/page-12.md", "quote": "Text."},
        ],
        "type": "answerable",
        "origin": "model-blind",
        "evidence_mode": "any",
    }
)
UNANSWERABLE = GoldenQuestion.model_validate(
    {
        "id": "q002",
        "question": "What is the dress code on Mars?",
        "type": "unanswerable",
        "origin": "model-drafted",
        "unanswerable_kind": "out-of-scope",
    }
)


def report(scores: dict[str, float], *, scored: int = 1, failures: int = 0) -> RetrievalReport:
    """A report with the given mean scores, without running a search."""
    return RetrievalReport(
        results=[QuestionResult(QUESTION, scores, (), 0.01)] * scored,
        failures=[
            FailedQuestion(f"q{number:03d}", "VectorStoreError", "connection refused")
            for number in range(1, failures + 1)
        ],
        not_scored=0,
        summary={metric: Estimate(mean, mean, mean, scored) for metric, mean in scores.items()}
        if scored
        else {},
    )


def test_the_slice_holds_every_page_with_evidence_and_a_fixed_number_of_others() -> None:
    chosen = select_pages(PAGES, [QUESTION, UNANSWERABLE], distractors=5, seed=3)

    names = [document.doc_id for document in chosen]
    assert evidence_pages([QUESTION, UNANSWERABLE]) == {"section/page-07.md", "section/page-12.md"}
    assert "section/page-07.md" in names
    assert "section/page-12.md" in names
    assert len(names) == 7
    assert names == sorted(names), "pages keep the order of the corpus"


def test_the_same_seed_gives_the_same_slice_and_another_seed_a_different_one() -> None:
    def names(seed: int) -> list[str]:
        return [d.doc_id for d in select_pages(PAGES, [QUESTION], distractors=5, seed=seed)]

    assert names(3) == names(3)
    assert names(3) != names(4)


def test_the_slice_does_not_depend_on_the_order_pages_were_loaded_in() -> None:
    shuffled = list(reversed(PAGES))

    forwards = {d.doc_id for d in select_pages(PAGES, [QUESTION], distractors=5, seed=3)}
    backwards = {d.doc_id for d in select_pages(shuffled, [QUESTION], distractors=5, seed=3)}

    assert forwards == backwards


def test_a_slice_can_have_no_extra_pages_or_every_page() -> None:
    only_evidence = select_pages(PAGES, [QUESTION], distractors=0, seed=1)

    assert [d.doc_id for d in only_evidence] == ["section/page-07.md", "section/page-12.md"]
    assert len(select_pages(PAGES, [QUESTION], distractors=500, seed=1)) == len(PAGES)


def test_the_gate_passes_at_its_minimums_and_fails_below_them() -> None:
    measured = report({"hit@5": 0.90, "mrr@10": 0.70})

    assert gate_problems(measured, {"hit@5": 0.90, "mrr@10": 0.70}) == []
    assert gate_problems(measured, {"hit@5": 0.95, "mrr@10": 0.70}) == [
        "hit@5 is 0.900, below the minimum of 0.950"
    ]
    assert gate_problems(measured, {"ndcg@10": 0.5}) == ["ndcg@10 was not measured"]


def test_a_score_just_below_its_minimum_is_never_printed_as_equal_to_it() -> None:
    one_question_short = report({"hit@5": 128 / 132})

    assert gate_problems(one_question_short, {"hit@5": 0.97}) == [
        "hit@5 is 0.9697, below the minimum of 0.9700"
    ]
    assert gate_problems(report({"hit@5": 129 / 132}), {"hit@5": 0.97}) == []
    assert shown_with_minimum(128 / 132, 0.97) == ("0.9697", "0.9700")
    assert shown_with_minimum(0.5, 0.9) == ("0.500", "0.900")
    assert shown_with_minimum(0.97, 0.97) == ("0.970", "0.970")
    assert shown_with_minimum(0.9924, 0.97) == ("0.992", "0.970")


def test_a_question_that_could_not_be_searched_fails_the_gate_whatever_the_scores() -> None:
    measured = report({"hit@5": 1.0}, scored=3, failures=2)

    assert gate_problems(measured, {"hit@5": 0.9}) == [
        "2 of 5 questions could not be searched; the first was q001: "
        "VectorStoreError: connection refused"
    ]


def test_a_gate_that_measured_nothing_fails() -> None:
    assert gate_problems(report({}, scored=0), {"hit@5": 0.9}) == ["no question could be scored"]
    (problem,) = gate_problems(report({}, scored=0, failures=4), {"hit@5": 0.9})
    assert problem.startswith("4 of 4 questions could not be searched")


def test_gate_settings_are_read_from_json(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "distractor_pages": 150,
                "seed": 11,
                "minimums": {"hit@5": 0.9},
                "note": "Measured 0.93 on 2026-11-22.",
            }
        )
    )

    config = load_gate_config(path)

    assert (config.distractor_pages, config.seed, config.minimums) == (150, 11, {"hit@5": 0.9})


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("{not json", "Expecting property name"),
        ('{"seed": 1, "minimums": {"hit@5": 0.9}}', "distractor_pages"),
        (
            '{"distractor_pages": -1, "seed": 1, "minimums": {"hit@5": 0.9}}',
            "greater than or equal",
        ),
        ('{"distractor_pages": 1, "seed": 1, "minimums": {}}', "checks nothing"),
        ('{"distractor_pages": 1, "seed": 1, "minimums": {"bleu": 0.5}}', "bleu is not a metric"),
        ('{"distractor_pages": 1, "seed": 1, "minimums": {"hit@5": 1.5}}', "at most 1"),
        ('{"distractor_pages": 1, "seed": 1, "minimums": {"hit@5": 0}}', "above 0"),
        ('{"distractor_pages": 1, "seed": 1, "minimum": {"hit@5": 0.9}}', "minimums"),
    ],
)
def test_broken_gate_settings_are_rejected_with_the_reason(
    tmp_path: Path, content: str, reason: str
) -> None:
    path = tmp_path / "gate.json"
    path.write_text(content)

    with pytest.raises(GateError, match="not a valid gate settings file") as raised:
        load_gate_config(path)

    assert reason in str(raised.value)


def test_missing_gate_settings_are_reported() -> None:
    with pytest.raises(GateError, match="gate settings not found"):
        load_gate_config(Path("nowhere.json"))


def test_the_committed_gate_settings_are_valid_and_guard_what_the_model_reads() -> None:
    config = load_gate_config(REPOSITORY / "evals" / "gate.json")

    assert "hit@5" in config.minimums, "the answer model reads five chunks"
    assert config.note, "say where the minimums come from"
