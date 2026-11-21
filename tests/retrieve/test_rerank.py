import pytest

from handbook_rag.index.store import ScoredChunk
from handbook_rag.retrieve.rerank import CrossEncoderReranker, RerankingRetriever
from tests.support import FakeReranker, make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid parental leave.")
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
INCIDENTS = make_chunk("Report a security incident immediately.", doc_id="security/incidents.md")


class FixedRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.limits: list[int] = []
        self.sections: list[str | None] = []

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        self.limits.append(limit)
        self.sections.append(section)
        return self._results[:limit]


FIRST_STAGE = [ScoredChunk(EXPENSES, 0.9), ScoredChunk(INCIDENTS, 0.8), ScoredChunk(LEAVE, 0.7)]


def test_reranking_moves_the_best_match_to_the_top() -> None:
    retriever = RerankingRetriever(FixedRetriever(FIRST_STAGE), FakeReranker())

    results = retriever.retrieve("how many weeks of parental leave", limit=2)

    assert results[0].chunk == LEAVE
    assert len(results) == 2
    assert results[0].score > results[1].score


def test_scores_are_the_rerankers_not_the_first_stage_scores() -> None:
    retriever = RerankingRetriever(FixedRetriever(FIRST_STAGE), FakeReranker())

    (top,) = retriever.retrieve("parental leave", limit=1)

    assert top.score == 2.0


def test_first_stage_is_asked_for_more_candidates_than_are_returned() -> None:
    base = FixedRetriever(FIRST_STAGE)

    RerankingRetriever(base, FakeReranker(), candidates=30).retrieve("q", limit=5, section="people")

    assert base.limits == [30]
    assert base.sections == ["people"]


def test_never_asks_for_fewer_candidates_than_the_limit() -> None:
    base = FixedRetriever(FIRST_STAGE)

    RerankingRetriever(base, FakeReranker(), candidates=10).retrieve("q", limit=40)

    assert base.limits == [40]


def test_reranker_reads_each_chunk_with_its_context_header() -> None:
    reranker = FakeReranker()
    titled = make_chunk("Sixteen weeks.", heading_path=("Parental leave",))

    RerankingRetriever(FixedRetriever([ScoredChunk(titled, 0.5)]), reranker).retrieve("q", limit=1)

    assert reranker.calls == [("q", ["Leave > Parental leave\n\nSixteen weeks."])]


def test_equal_scores_keep_the_first_stage_order() -> None:
    retriever = RerankingRetriever(FixedRetriever(FIRST_STAGE), FakeReranker())

    results = retriever.retrieve("zebra", limit=3)

    assert [r.chunk for r in results] == [EXPENSES, INCIDENTS, LEAVE]


def test_nothing_retrieved_means_nothing_reranked() -> None:
    reranker = FakeReranker()

    assert RerankingRetriever(FixedRetriever([]), reranker).retrieve("q", limit=3) == []
    assert reranker.calls == []


@pytest.mark.slow
def test_real_cross_encoder_prefers_the_passage_that_answers_the_question() -> None:
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2")

    scores = reranker.score(
        "How many weeks of parental leave do I get?",
        [EXPENSES.text, LEAVE.text, INCIDENTS.text],
    )

    assert scores.index(max(scores)) == 1
    assert reranker.score("anything", []) == []
    assert reranker.name == "cross-encoder/ms-marco-MiniLM-L6-v2"
