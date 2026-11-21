from collections.abc import Callable, Sequence

from handbook_rag.generate.models import Answer, Segment
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import answer_question
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.")
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")


class FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        self.calls.append((query, limit, section))
        return self._results[:limit]


class FakeGenerator:
    def __init__(self) -> None:
        self.received: list[Chunk] = []

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.received = list(chunks)
        if on_text is not None:
            on_text("An answer.")
        return Answer(
            question=question,
            segments=(Segment("An answer."),),
            answered=True,
            model="fake-model",
            stop_reason="end_turn",
        )


def test_question_is_retrieved_then_answered_from_the_chunks_in_ranked_order() -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])
    generator = FakeGenerator()

    answer = answer_question(
        "How long is parental leave?", retriever=retriever, generator=generator, limit=2
    )

    assert retriever.calls == [("How long is parental leave?", 2, None)]
    assert generator.received == [LEAVE, EXPENSES]
    assert answer.text == "An answer."
    assert [r.chunk for r in answer.retrieved] == [LEAVE, EXPENSES]
    assert [r.score for r in answer.retrieved] == [0.9, 0.4]


def test_limit_and_section_are_passed_to_the_retriever() -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])
    generator = FakeGenerator()

    answer_question("q", retriever=retriever, generator=generator, limit=1, section="people")

    assert retriever.calls == [("q", 1, "people")]
    assert generator.received == [LEAVE]


def test_streamed_text_reaches_the_caller() -> None:
    received: list[str] = []

    answer_question(
        "q",
        retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]),
        generator=FakeGenerator(),
        limit=1,
        on_text=received.append,
    )

    assert received == ["An answer."]
