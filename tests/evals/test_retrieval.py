import json
from pathlib import Path

import pytest

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.retrieval import METRIC_IDS, evaluate_retrieval, write_run
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk, Document

LEAVE_TEXT = "## Parental leave\n\nNew parents get sixteen weeks of paid leave."
EXPENSE_TEXT = "## Laptops\n\nExpense a laptop up to one thousand dollars with approval."


def page(doc_id: str, text: str) -> Document:
    return Document(doc_id, "Title", "https://example.com/", doc_id.split("/")[0], "a" * 40, text)


def whole_page_chunk(document: Document) -> Chunk:
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
QUESTIONS = [
    GoldenQuestion.model_validate(
        {
            "id": "q001",
            "question": "How long is parental leave?",
            "reference_answer": "Sixteen weeks.",
            "evidence": [{"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"}],
            "type": "answerable",
            "origin": "model-drafted",
        }
    ),
    GoldenQuestion.model_validate(
        {
            "id": "q002",
            "question": "What is the laptop expense limit?",
            "reference_answer": "One thousand dollars.",
            "evidence": [{"doc_id": "finance/expenses.md", "quote": "one thousand dollars"}],
            "type": "answerable",
            "origin": "hand-written",
        }
    ),
    GoldenQuestion.model_validate(
        {
            "id": "q003",
            "question": "What is the dress code on Mars?",
            "type": "unanswerable",
            "origin": "hand-written",
            "unanswerable_kind": "out-of-scope",
        }
    ),
]


class FixedRetriever:
    """Returns a prepared ranking for each question, whatever the limit."""

    def __init__(self, rankings: dict[str, list[Document]]) -> None:
        self._rankings = rankings

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        ranked = self._rankings.get(query, [])
        return [
            ScoredChunk(whole_page_chunk(document), 1.0 - 0.1 * position)
            for position, document in enumerate(ranked[:limit])
        ]


PERFECT = FixedRetriever(
    {
        "How long is parental leave?": [LEAVE, EXPENSES],
        "What is the laptop expense limit?": [EXPENSES, LEAVE],
    }
)
USELESS = FixedRetriever(
    {"How long is parental leave?": [EXPENSES], "What is the laptop expense limit?": [LEAVE]}
)
SECOND_PLACE = FixedRetriever(
    {
        "How long is parental leave?": [EXPENSES, LEAVE],
        "What is the laptop expense limit?": [EXPENSES],
    }
)


def test_a_perfect_retriever_scores_one_on_every_metric() -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, PERFECT)

    assert {name: estimate.mean for name, estimate in report.summary.items()} == dict.fromkeys(
        METRIC_IDS, 1.0
    )


def test_a_retriever_that_never_finds_the_evidence_scores_zero() -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, USELESS)

    assert {name: estimate.mean for name, estimate in report.summary.items()} == dict.fromkeys(
        METRIC_IDS, 0.0
    )


def test_rank_of_the_evidence_shows_up_in_the_scores() -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, SECOND_PLACE)

    by_id = {result.question.id: result for result in report.results}
    assert by_id["q001"].scores["hit@1"] == 0.0
    assert by_id["q001"].scores["hit@3"] == 1.0
    assert by_id["q001"].scores["mrr@10"] == 0.5
    assert by_id["q002"].scores["mrr@10"] == 1.0
    assert report.summary["hit@1"].mean == 0.5
    assert report.summary["mrr@10"].mean == 0.75


def test_unanswerable_questions_are_counted_but_not_scored() -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, PERFECT)

    assert [result.question.id for result in report.results] == ["q001", "q002"]
    assert report.not_scored == 1


def test_results_are_broken_down_by_origin_and_section() -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, SECOND_PLACE)

    assert report.by_group["origin"]["model-drafted"]["hit@1"].mean == 0.0
    assert report.by_group["origin"]["hand-written"]["hit@1"].mean == 1.0
    assert set(report.by_group["section"]) == {"people", "finance"}


def test_a_harness_failure_is_recorded_separately_and_never_scored_as_a_miss() -> None:
    class BrokenRetriever:
        def retrieve(
            self, query: str, *, limit: int, section: str | None = None
        ) -> list[ScoredChunk]:
            if "laptop" in query:
                raise ConnectionError("vector store is down")
            return PERFECT.retrieve(query, limit=limit)

    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, BrokenRetriever())

    assert [result.question.id for result in report.results] == ["q001"]
    assert report.summary["hit@5"].mean == 1.0
    (failure,) = report.failures
    assert (failure.question_id, failure.error_class) == ("q002", "ConnectionError")
    assert "vector store is down" in failure.message


def test_a_broken_label_stops_the_run() -> None:
    from handbook_rag.evals.dataset import DatasetError

    rewritten = [page("people/leave.md", "The policy was rewritten."), EXPENSES]

    with pytest.raises(DatasetError, match="quote not found"):
        evaluate_retrieval(QUESTIONS, rewritten, PERFECT)


def test_run_is_written_as_one_row_per_question_with_a_summary(tmp_path: Path) -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, SECOND_PLACE)

    run_dir = write_run(
        report,
        tmp_path / "retrieval",
        "baseline",
        "Dense retrieval only.",
        {"retrieval_mode": "dense", "rerank": False},
    )

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == ["q001", "q002"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["tags"] == ["people", "model-drafted", "review:pending"]
    assert rows[0]["grade"]["hit@5"] == 1.0
    assert "retrieved" not in rows[0]
    details = [json.loads(line) for line in (run_dir / "retrieved.jsonl").read_text().splitlines()]
    assert [detail["prompt_id"] for detail in details] == ["q001", "q002"]
    assert details[0]["retrieved"][1]["covers"] == [0]
    assert details[0]["retrieved"][1]["doc_id"] == "people/leave.md"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["questions_scored"] == 2
    assert summary["config"] == {"retrieval_mode": "dense", "rerank": False}
    assert len(summary["labels_fingerprint"]) == 16
    assert rows[0]["fingerprint"] == QUESTIONS[0].fingerprint
    assert summary["questions_not_scored"] == 1
    assert summary["metrics"]["mrr@10"]["mean"] == 0.75
    assert (run_dir / "traces" / "q001_rep0.json").is_file()
    assert (run_dir / "errors.jsonl").read_text() == ""
    state = json.loads((tmp_path / "retrieval" / "_state.json").read_text())
    assert state["metrics"][0]["id"] == "hit@5"
