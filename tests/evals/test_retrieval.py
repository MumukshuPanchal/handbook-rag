import json
from pathlib import Path

import pytest

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.retrieval import (
    ALL_CUTOFFS,
    METRIC_IDS,
    RETRIEVAL_DEPTH,
    SeveralPassages,
    evaluate_retrieval,
    several_passages_lines,
    write_run,
)
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


def test_alternative_evidence_is_found_when_any_piece_is_returned() -> None:
    two_homes = {
        "id": "q010",
        "question": "Where is a limit or an entitlement stated?",
        "reference_answer": "In the leave policy and in the expense policy.",
        "evidence": [
            {"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"},
            {"doc_id": "finance/expenses.md", "quote": "one thousand dollars"},
        ],
        "type": "answerable",
        "origin": "model-blind",
        "review": "checked",
    }
    only_one_page = FixedRetriever({"Where is a limit or an entitlement stated?": [LEAVE]})

    needs_all = evaluate_retrieval(
        [GoldenQuestion.model_validate(two_homes)], DOCUMENTS, only_one_page
    )
    needs_any = evaluate_retrieval(
        [GoldenQuestion.model_validate({**two_homes, "evidence_mode": "any"})],
        DOCUMENTS,
        only_one_page,
    )

    assert needs_all.summary["recall@5"].mean == 0.5
    assert needs_all.summary["ndcg@10"].mean < 1.0
    assert needs_any.summary["recall@5"].mean == 1.0
    assert needs_any.summary["ndcg@10"].mean == 1.0
    assert needs_any.summary["hit@1"].mean == 1.0
    assert list(needs_any.by_group["origin"]) == ["model-blind"]


def test_two_alternatives_found_together_are_not_counted_twice() -> None:
    two_homes = {
        "id": "q011",
        "question": "Where is a limit or an entitlement stated?",
        "reference_answer": "In the leave policy and in the expense policy.",
        "evidence": [
            {"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"},
            {"doc_id": "finance/expenses.md", "quote": "one thousand dollars"},
        ],
        "type": "answerable",
        "origin": "model-blind",
        "evidence_mode": "any",
    }
    both_pages = FixedRetriever({"Where is a limit or an entitlement stated?": [LEAVE, EXPENSES]})

    report = evaluate_retrieval([GoldenQuestion.model_validate(two_homes)], DOCUMENTS, both_pages)

    assert {name: estimate.mean for name, estimate in report.summary.items()} == dict.fromkeys(
        METRIC_IDS, 1.0
    )
    assert report.results[0].retrieved[1].covers == (1,)


# Pages that hold no evidence, to put between the two passages a question needs.
FILLERS = [page(f"other/page-{n}.md", f"## Page {n}\n\nNothing to see here.") for n in range(1, 10)]
WITH_FILLERS = [*DOCUMENTS, *FILLERS]


def needs_both(question_id: str, question: str) -> GoldenQuestion:
    """A question whose answer needs the leave page and the expenses page, in that order."""
    return GoldenQuestion.model_validate(
        {
            "id": question_id,
            "question": question,
            "reference_answer": "Sixteen weeks of leave, and one thousand dollars for a laptop.",
            "evidence": [
                {"doc_id": "people/leave.md", "quote": "sixteen weeks of paid leave"},
                {"doc_id": "finance/expenses.md", "quote": "one thousand dollars"},
            ],
            "type": "answerable",
            "origin": "model-drafted",
        }
    )


TWO_PAGES = needs_both("t001", "How do the leave and the laptop limit compare?")


def test_one_of_two_needed_passages_is_a_hit_and_is_not_all() -> None:
    only_one = FixedRetriever({TWO_PAGES.question: [LEAVE, *FILLERS]})

    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, only_one)

    (result,) = report.results
    assert result.scores["hit@1"] == result.scores["hit@5"] == 1.0
    assert result.scores["recall@5"] == 0.5
    assert [result.scores[f"all@{k}"] for k in ALL_CUTOFFS] == [0.0, 0.0, 0.0]
    assert result.evidence_ranks == result.page_ranks == (1, None)
    one_of_two = {"all": 0, "some": 1, "none": 0}
    assert report.several == SeveralPassages(
        questions=1,
        found_in_top={3: one_of_two, 5: one_of_two, 10: one_of_two},
        all_found_by_rank={},
        not_all_retrieved=1,
        pages_in_top={3: one_of_two, 5: one_of_two, 10: one_of_two},
    )


def test_a_passage_that_comes_late_counts_from_its_rank_on() -> None:
    # The expenses page first, the leave page seventh.
    late = FixedRetriever({TWO_PAGES.question: [EXPENSES, *FILLERS[:5], LEAVE, *FILLERS[5:]]})

    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, late)

    (result,) = report.results
    # In the order the question lists its evidence: the leave page, then the expenses page.
    assert result.evidence_ranks == (7, 1)
    assert [result.scores[f"all@{k}"] for k in ALL_CUTOFFS] == [0.0, 0.0, 1.0]
    assert result.scores["hit@1"] == 1.0
    assert report.several is not None
    assert report.several.found_in_top[5] == {"all": 0, "some": 1, "none": 0}
    assert report.several.found_in_top[10] == {"all": 1, "some": 0, "none": 0}
    assert report.several.all_found_by_rank == {7: 1}
    assert report.several.not_all_retrieved == 0


def test_a_passage_at_the_cutoff_counts_as_among_the_first_results() -> None:
    # The leave page third, the expenses page fifth: exactly at two of the cutoffs.
    at_the_cutoffs = FixedRetriever(
        {TWO_PAGES.question: [FILLERS[0], FILLERS[1], LEAVE, FILLERS[2], EXPENSES]}
    )

    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, at_the_cutoffs)

    (result,) = report.results
    assert result.evidence_ranks == result.page_ranks == (3, 5)
    assert [result.scores[f"all@{k}"] for k in ALL_CUTOFFS] == [0.0, 1.0, 1.0]
    assert report.several is not None
    for counts in (report.several.found_in_top, report.several.pages_in_top):
        assert counts == {
            3: {"all": 0, "some": 1, "none": 0},
            5: {"all": 1, "some": 0, "none": 0},
            10: {"all": 1, "some": 0, "none": 0},
        }
    assert report.several.all_found_by_rank == {5: 1}


def test_a_retriever_that_brings_both_passages_first_scores_one_on_every_metric() -> None:
    both_first = FixedRetriever({TWO_PAGES.question: [LEAVE, EXPENSES, *FILLERS]})

    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, both_first)

    assert {name: estimate.mean for name, estimate in report.summary.items()} == dict.fromkeys(
        METRIC_IDS, 1.0
    )
    assert report.several is not None
    assert report.several.all_found_by_rank == {2: 1}


def test_the_counts_for_several_passages_are_the_counts_behind_the_means() -> None:
    questions = [
        needs_both("t001", "both are found early"),
        needs_both("t002", "one is found"),
        needs_both("t003", "none is found"),
        needs_both("t004", "the second comes fourth"),
    ]
    retriever = FixedRetriever(
        {
            "both are found early": [LEAVE, EXPENSES],
            "one is found": [FILLERS[0], EXPENSES],
            "none is found": FILLERS,
            "the second comes fourth": [LEAVE, FILLERS[0], FILLERS[1], EXPENSES],
        }
    )

    report = evaluate_retrieval(questions, WITH_FILLERS, retriever)

    assert report.several == SeveralPassages(
        questions=4,
        found_in_top={
            3: {"all": 1, "some": 2, "none": 1},
            5: {"all": 2, "some": 1, "none": 1},
            10: {"all": 2, "some": 1, "none": 1},
        },
        all_found_by_rank={2: 1, 4: 1},
        not_all_retrieved=2,
        pages_in_top={
            3: {"all": 1, "some": 2, "none": 1},
            5: {"all": 2, "some": 1, "none": 1},
            10: {"all": 2, "some": 1, "none": 1},
        },
    )
    for k, counts in report.several.found_in_top.items():
        assert report.summary[f"all@{k}"].mean == counts["all"] / 4
        assert report.summary[f"hit@{k}"].mean == (counts["all"] + counts["some"]) / 4
        assert sum(counts.values()) == 4


def test_a_chunk_of_the_right_page_that_misses_the_quote_counts_for_the_page_only() -> None:
    class HeadingOfTheLeavePage:
        """Returns the expenses page, then a chunk of the leave page that ends before the quote."""

        def retrieve(
            self, query: str, *, limit: int, section: str | None = None
        ) -> list[ScoredChunk]:
            whole = whole_page_chunk(LEAVE)
            heading = Chunk(
                chunk_id="chunk-heading",
                doc_id=whole.doc_id,
                title=whole.title,
                url=whole.url,
                section=whole.section,
                source_commit=whole.source_commit,
                heading_path=(),
                start_char=0,
                end_char=LEAVE_TEXT.index("New parents"),
                text=LEAVE_TEXT[: LEAVE_TEXT.index("New parents")],
                token_count=3,
            )
            return [ScoredChunk(whole_page_chunk(EXPENSES), 1.0), ScoredChunk(heading, 0.9)]

    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, HeadingOfTheLeavePage())

    (result,) = report.results
    assert result.evidence_ranks == (None, 1)
    assert result.page_ranks == (2, 1)
    assert result.scores["all@5"] == 0.0
    assert report.several is not None
    assert report.several.found_in_top[5] == {"all": 0, "some": 1, "none": 0}
    assert report.several.pages_in_top[5] == {"all": 1, "some": 0, "none": 0}


def test_a_question_with_one_passage_or_with_alternatives_does_not_need_several() -> None:
    alternatives = GoldenQuestion.model_validate(
        {**needs_both("q010", "either page will do").model_dump(), "evidence_mode": "any"}
    )
    retriever = FixedRetriever(
        {
            "How long is parental leave?": [EXPENSES, LEAVE],
            "What is the laptop expense limit?": [EXPENSES],
            "either page will do": [FILLERS[0], EXPENSES],
        }
    )

    report = evaluate_retrieval([*QUESTIONS, alternatives], WITH_FILLERS, retriever)

    assert report.several is None
    for result in report.results:
        for k in ALL_CUTOFFS:
            assert result.scores[f"all@{k}"] == result.scores[f"hit@{k}"]
    by_id = {result.question.id: result for result in report.results}
    assert by_id["q001"].evidence_ranks == (2,)
    # Each alternative keeps its own rank, although finding one of them is enough.
    assert by_id["q010"].evidence_ranks == (None, 2)
    assert by_id["q010"].scores["all@3"] == 1.0


def test_a_run_records_where_each_passage_came_and_the_counts_for_several(tmp_path: Path) -> None:
    late = FixedRetriever({TWO_PAGES.question: [EXPENSES, *FILLERS[:5], LEAVE, *FILLERS[5:]]})
    report = evaluate_retrieval([TWO_PAGES], WITH_FILLERS, late)

    run_dir = write_run(report, tmp_path / "two-passage", "hybrid", "Hybrid.")

    (row,) = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert row["evidence_ranks"] == row["page_ranks"] == [7, 1]
    assert row["grade"]["all@5"] == 0.0
    assert row["grade"]["all@10"] == 1.0
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["needs_several_passages"] == {
        "questions": 1,
        "found_in_top": {
            "3": {"all": 0, "some": 1, "none": 0},
            "5": {"all": 0, "some": 1, "none": 0},
            "10": {"all": 1, "some": 0, "none": 0},
        },
        "all_found_by_rank": {"7": 1},
        "not_all_retrieved": 0,
        "pages_in_top": {
            "3": {"all": 0, "some": 1, "none": 0},
            "5": {"all": 0, "some": 1, "none": 0},
            "10": {"all": 1, "some": 0, "none": 0},
        },
        "retrieved": RETRIEVAL_DEPTH,
    }
    declared = json.loads((tmp_path / "two-passage" / "_state.json").read_text())["metrics"]
    assert [metric["id"] for metric in declared] == list(METRIC_IDS)
    assert {f"all@{k}" for k in ALL_CUTOFFS} < set(METRIC_IDS)


def test_a_run_of_questions_with_one_passage_has_no_counts_for_several(tmp_path: Path) -> None:
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, SECOND_PLACE)

    run_dir = write_run(report, tmp_path / "retrieval", "baseline", "Dense retrieval only.")

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [row["evidence_ranks"] for row in rows] == [[2], [1]]
    assert [row["page_ranks"] for row in rows] == [[2], [1]]
    assert "needs_several_passages" not in json.loads((run_dir / "summary.json").read_text())


def test_the_lines_about_several_passages_give_the_counts_and_the_ranks() -> None:
    several = SeveralPassages(
        questions=35,
        found_in_top={
            3: {"all": 10, "some": 20, "none": 5},
            5: {"all": 14, "some": 18, "none": 3},
            10: {"all": 20, "some": 14, "none": 1},
        },
        all_found_by_rank={2: 6, 3: 4, 5: 4, 9: 6},
        not_all_retrieved=15,
        pages_in_top={
            3: {"all": 14, "some": 17, "none": 4},
            5: {"all": 19, "some": 14, "none": 2},
            10: {"all": 26, "some": 9, "none": 0},
        },
    )

    assert several_passages_lines(several) == [
        "  questions that need several passages: 35",
        "    in the top 3   all found 10, some 20, none 5",
        "    in the top 5   all found 14, some 18, none 3",
        "    in the top 10  all found 20, some 14, none 1",
        "    all found by rank  2: 6, 3: 4, 5: 4, 9: 6; not all in the top 10: 15",
        "    a chunk of every passage's page, whatever it says, in the top 3: 14, top 5: 19, "
        "top 10: 26",
    ]
    none = {3: {"all": 0, "some": 0, "none": 1}}
    nothing = SeveralPassages(1, none, {}, 1, none)
    assert several_passages_lines(nothing)[-2] == (
        "    all found by rank  never; not all in the top 10: 1"
    )


def test_reviewed_only_leaves_out_the_questions_whose_review_is_pending(tmp_path: Path) -> None:
    accepted = GoldenQuestion.model_validate({**QUESTIONS[0].model_dump(), "review": "checked"})
    by_a_person = GoldenQuestion.model_validate(
        {**QUESTIONS[1].model_dump(), "id": "q004", "question": "verified", "review": "verified"}
    )
    questions = [accepted, QUESTIONS[1], QUESTIONS[2], by_a_person]
    retriever = FixedRetriever({accepted.question: [LEAVE], "verified": [EXPENSES]})

    report = evaluate_retrieval(questions, DOCUMENTS, retriever, reviewed_only=True)

    # The pending question with an answer is left out. The unanswerable one is not
    # scored in any case, and is counted as before.
    assert [result.question.id for result in report.results] == ["q001", "q004"]
    assert report.left_out == ("q002",)
    assert report.not_scored == 1
    assert report.summary["hit@1"].mean == 1.0
    assert report.summary["hit@1"].count == 2
    run_dir = write_run(report, tmp_path / "retrieval", "reviewed", "Reviewed questions only.")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["questions_left_out"] == ["q002"]
    assert (summary["questions_scored"], summary["questions_not_scored"]) == (2, 1)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == ["q001", "q004"]


def test_every_answerable_question_is_scored_unless_reviewed_only_is_asked_for(
    tmp_path: Path,
) -> None:
    # The first two questions are pending, as a question is until someone has checked it.
    report = evaluate_retrieval(QUESTIONS, DOCUMENTS, PERFECT)
    nothing_pending = [
        GoldenQuestion.model_validate({**question.model_dump(), "review": "checked"})
        for question in QUESTIONS
    ]
    all_reviewed = evaluate_retrieval(nothing_pending, DOCUMENTS, PERFECT, reviewed_only=True)

    assert [result.question.id for result in report.results] == ["q001", "q002"]
    assert report.left_out is None
    run_dir = write_run(report, tmp_path / "retrieval", "baseline", "Dense retrieval only.")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "questions_left_out" not in summary
    assert list(summary) == [
        "description",
        "config",
        "labels_fingerprint",
        "questions_scored",
        "questions_failed",
        "questions_not_scored",
        "metrics",
        "by_group",
    ]
    # Asked for, and nothing to leave out: the run says so with an empty list.
    assert all_reviewed.left_out == ()
    assert [result.question.id for result in all_reviewed.results] == ["q001", "q002"]
    run_dir = write_run(all_reviewed, tmp_path / "retrieval", "reviewed", "Reviewed only.")
    assert json.loads((run_dir / "summary.json").read_text())["questions_left_out"] == []


def test_a_run_writes_where_the_passage_came_apart_from_where_its_page_came(
    tmp_path: Path,
) -> None:
    class AnotherChunkOfThePageFirst:
        """Returns a chunk of the leave page that ends before the quote, then the whole page."""

        def retrieve(
            self, query: str, *, limit: int, section: str | None = None
        ) -> list[ScoredChunk]:
            whole = whole_page_chunk(LEAVE)
            heading = Chunk(
                chunk_id="chunk-heading",
                doc_id=whole.doc_id,
                title=whole.title,
                url=whole.url,
                section=whole.section,
                source_commit=whole.source_commit,
                heading_path=(),
                start_char=0,
                end_char=LEAVE_TEXT.index("New parents"),
                text=LEAVE_TEXT[: LEAVE_TEXT.index("New parents")],
                token_count=3,
            )
            return [ScoredChunk(heading, 1.0), ScoredChunk(whole, 0.9)]

    report = evaluate_retrieval(QUESTIONS[:1], DOCUMENTS, AnotherChunkOfThePageFirst())

    run_dir = write_run(report, tmp_path / "retrieval", "baseline", "Dense retrieval only.")

    (row,) = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert (row["evidence_ranks"], row["page_ranks"]) == ([2], [1])
    assert row["grade"]["hit@1"] == 0.0
    assert row["grade"]["hit@3"] == 1.0
