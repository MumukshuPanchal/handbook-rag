from collections.abc import Callable, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from handbook_rag import tracing, wiring
from handbook_rag.config import Settings
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import answer_question
from handbook_rag.retrieve.dense import Retriever
from tests.support import (
    FakeEmbedder,
    FakeSparseEncoder,
    StandInModel,
    everything_recorded,
    make_chunk,
)

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.")
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
QUESTION = "How long is parental leave?"


class FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(self, query: str, *, limit: int, section: str | None = None) -> list[ScoredChunk]:
        self.calls.append((query, limit, section))
        return self._results[:limit]


def reply(question: str, chunks: Sequence[Chunk]) -> Answer:
    """What the stand-in model says by default: one statement, citing the first passage."""
    return Answer(
        question=question,
        segments=(Segment("An answer.", (Citation(chunks[0]),) if chunks else ()),),
        answered=bool(chunks),
        model="fake-model",
        stop_reason="end_turn" if chunks else "no_context",
    )


class FakeGenerator(StandInModel):
    def __init__(
        self,
        replies: Callable[[str, Sequence[Chunk]], Answer] = reply,
        *,
        error: Exception | None = None,
    ) -> None:
        self.received: list[Chunk] = []
        self._replies = replies
        self._error = error

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.received = list(chunks)
        if self._error is not None:
            raise self._error
        if on_text is not None:
            on_text("An answer.")
        return self._replies(question, chunks)


def test_question_is_retrieved_then_answered_from_the_chunks_in_ranked_order() -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])
    generator = FakeGenerator()

    answer = answer_question(QUESTION, retriever=retriever, generator=generator, limit=2)

    assert retriever.calls == [(QUESTION, 2, None)]
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


# The trace of an answer. Every caller gets it, because the pipeline opens the spans.


def indexed_retriever() -> Retriever:
    """The default hybrid retriever over two chunks in memory, with stand-in models."""
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("answers")
    index_chunks([LEAVE, EXPENSES], embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    return wiring.make_retriever(
        Settings(), store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )


def paths(exporter: InMemorySpanExporter) -> list[str]:
    """Every recorded span as the names from the root of its trace down to it, in start order."""
    finished = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in finished}

    def path(span: ReadableSpan) -> str:
        names = [span.name]
        while span.parent is not None and span.parent.span_id in by_id:
            span = by_id[span.parent.span_id]
            names.append(span.name)
        return " > ".join(reversed(names))

    return [path(span) for span in sorted(finished, key=lambda span: span.start_time or 0)]


def named(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = [span for span in exporter.get_finished_spans() if span.name == name]
    return span


def attributes(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def test_an_answer_is_one_trace_with_the_search_and_the_model_call_inside_it(
    spans: InMemorySpanExporter,
) -> None:
    answer_question(QUESTION, retriever=indexed_retriever(), generator=FakeGenerator(), limit=2)

    assert paths(spans) == [
        "answer",
        "answer > search",
        "answer > search > retrieve.hybrid",
        "answer > search > retrieve.hybrid > retrieve.dense",
        "answer > search > retrieve.hybrid > retrieve.dense > embed.query",
        "answer > search > retrieve.hybrid > retrieve.dense > vector_store.query",
        "answer > search > retrieve.hybrid > retrieve.sparse",
        "answer > search > retrieve.hybrid > retrieve.sparse > vector_store.query",
        "answer > search > retrieve.hybrid > fuse",
        # Named by the conventions for generative AI: the operation, then the model.
        "answer > chat stand-in-model",
    ]
    assert len({span.context.trace_id for span in spans.get_finished_spans()}) == 1
    assert named(spans, "chat stand-in-model").kind is SpanKind.CLIENT, "a call to another system"
    assert named(spans, "answer").kind is SpanKind.INTERNAL
    assert named(spans, "search").kind is SpanKind.INTERNAL


def test_the_search_span_records_the_question_by_its_length_and_what_was_found(
    spans: InMemorySpanExporter,
) -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])

    answer_question(QUESTION, retriever=retriever, generator=FakeGenerator(), limit=2)

    assert attributes(named(spans, "search")) == {
        "handbook_rag.limit": 2,
        "handbook_rag.query.chars": len(QUESTION),
        "handbook_rag.results.count": 2,
        "handbook_rag.results.chunk_ids": (LEAVE.chunk_id, EXPENSES.chunk_id),
        "handbook_rag.results.doc_ids": ("people/leave.md", "finance/expenses.md"),
        "handbook_rag.results.scores": (0.9, 0.4),
    }
    assert attributes(named(spans, "answer")) == {
        "handbook_rag.query.chars": len(QUESTION),
        "handbook_rag.answered": True,
        "handbook_rag.requests": 1,
        "handbook_rag.fallback": False,
    }


def test_the_model_call_records_the_reply_under_the_names_of_the_conventions(
    spans: InMemorySpanExporter,
) -> None:
    def cut_off(question: str, chunks: Sequence[Chunk]) -> Answer:
        return Answer(
            question=question,
            segments=(
                Segment("Laptops are expensed.", (Citation(chunks[1]),)),
                Segment(" Sixteen weeks.", (Citation(chunks[0]), Citation(chunks[1]))),
                Segment(" And"),
            ),
            answered=True,
            model="stand-in-model-002",
            stop_reason="max_tokens",
            usage=TokenUsage(input_tokens=1200, output_tokens=80, cache_read_input_tokens=300),
            seconds=2.5,
            seconds_to_first_text=0.75,
        )

    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])
    # Asked for more than the search finds, so the passages given are not the limit.
    answer_question(QUESTION, retriever=retriever, generator=FakeGenerator(cut_off), limit=5)

    assert attributes(named(spans, "chat stand-in-model")) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "stand-in",
        "gen_ai.request.model": "stand-in-model",
        "gen_ai.response.model": "stand-in-model-002",
        "gen_ai.response.finish_reasons": ("max_tokens",),
        # Every token read, those from the provider's cache included.
        "gen_ai.usage.input_tokens": 1500,
        "gen_ai.usage.output_tokens": 80,
        "gen_ai.usage.cache_read.input_tokens": 300,
        "gen_ai.response.time_to_first_chunk": 0.75,
        "handbook_rag.answered": True,
        "handbook_rag.passages.given": 2,
        "handbook_rag.passages.cited": 2,
        # In the order the answer numbers them, which is not the order they were found.
        "handbook_rag.passages.cited_chunk_ids": (EXPENSES.chunk_id, LEAVE.chunk_id),
        # A generator that is not wrapped sends one request for each answer.
        "handbook_rag.attempt": 1,
        "handbook_rag.fallback": False,
    }


def test_a_reply_with_no_text_and_tokens_written_to_the_cache_is_recorded_as_such(
    spans: InMemorySpanExporter,
) -> None:
    def empty(question: str, chunks: Sequence[Chunk]) -> Answer:
        return Answer(
            question=question,
            segments=(),
            answered=False,
            model="stand-in-model",
            stop_reason="refusal",
            usage=TokenUsage(input_tokens=10, output_tokens=0, cache_creation_input_tokens=7),
        )

    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9)])
    answer_question(QUESTION, retriever=retriever, generator=FakeGenerator(empty), limit=1)

    recorded = attributes(named(spans, "chat stand-in-model"))
    assert recorded["gen_ai.usage.input_tokens"] == 17
    assert recorded["gen_ai.usage.cache_creation.input_tokens"] == 7
    assert "gen_ai.usage.cache_read.input_tokens" not in recorded, "none were read"
    assert "gen_ai.response.time_to_first_chunk" not in recorded, "no text came"
    assert recorded["handbook_rag.answered"] is False
    assert recorded["handbook_rag.passages.cited"] == 0
    assert recorded["handbook_rag.passages.cited_chunk_ids"] == ()
    assert attributes(named(spans, "answer"))["handbook_rag.answered"] is False


def test_the_question_and_the_answer_are_left_out_of_a_trace_unless_content_is_switched_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9)])

    answer_question(QUESTION, retriever=retriever, generator=FakeGenerator(), limit=1)
    recorded = everything_recorded(spans.get_finished_spans())

    assert QUESTION not in recorded
    assert "An answer." not in recorded
    assert "handbook_rag.query.text" not in recorded
    assert "handbook_rag.answer.text" not in recorded

    spans.clear()
    monkeypatch.setattr(tracing, "_record_content", True)
    answer_question(QUESTION, retriever=retriever, generator=FakeGenerator(), limit=1)

    whole = attributes(named(spans, "answer"))
    assert whole["handbook_rag.query.text"] == QUESTION
    assert whole["handbook_rag.answer.text"] == "An answer.[1]", "with its citation markers"
    assert attributes(named(spans, "search"))["handbook_rag.query.text"] == QUESTION
    assert "handbook_rag.answer.text" not in attributes(named(spans, "chat stand-in-model"))


def test_a_failed_model_call_is_marked_failed_and_so_is_the_answer_around_it(
    spans: InMemorySpanExporter,
) -> None:
    failing = FakeGenerator(error=GenerationError("No Gemini API key was found."))

    with pytest.raises(GenerationError, match="No Gemini API key"):
        answer_question(
            QUESTION, retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]), generator=failing, limit=1
        )

    call = named(spans, "chat stand-in-model")
    assert call.status.status_code is StatusCode.ERROR
    assert call.status.description == "handbook_rag.generate.base.GenerationError"
    assert "gen_ai.response.model" not in attributes(call), "there was no reply"
    assert attributes(call)["gen_ai.request.model"] == "stand-in-model", "the request is known"
    assert attributes(call)["handbook_rag.failure"] == "unavailable", "the reason, in fixed words"
    whole = named(spans, "answer")
    assert whole.status.status_code is StatusCode.ERROR
    assert (whole.attributes or {})["handbook_rag.requests"] == 1
    assert (whole.attributes or {})["handbook_rag.fallback"] is False
    assert named(spans, "search").status.status_code is not StatusCode.ERROR


def test_an_errors_own_message_reaches_the_trace_of_an_answer_only_with_content_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "Could not read /Users/someone/handbook/.env: check GEMINI_API_KEY."
    failing = FakeGenerator(error=GenerationError(secret, reason="configuration"))
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9)])

    def fail() -> str:
        with pytest.raises(GenerationError):
            answer_question(QUESTION, retriever=retriever, generator=failing, limit=1)
        return everything_recorded(spans.get_finished_spans())

    quiet = fail()
    spans.clear()
    monkeypatch.setattr(tracing, "_record_content", True)
    full = fail()

    for part in (secret, "/Users/someone", "GEMINI_API_KEY", QUESTION):
        assert part not in quiet
        assert part in full, "the search above can find it"
    assert "handbook_rag.generate.base.GenerationError" in quiet, "the kind of error only"


def test_with_nothing_found_there_is_no_span_for_a_model_call_that_was_not_made(
    spans: InMemorySpanExporter,
) -> None:
    answer = answer_question(
        QUESTION, retriever=FakeRetriever([]), generator=FakeGenerator(), limit=2, section="legal"
    )

    assert answer.stop_reason == "no_context"
    assert paths(spans) == ["answer", "answer > search"]
    assert attributes(named(spans, "answer"))["handbook_rag.answered"] is False
    assert attributes(named(spans, "answer"))["handbook_rag.section"] == "legal"
    assert attributes(named(spans, "search"))["handbook_rag.section"] == "legal"
    # The stand-in does not say so itself, and still no request is counted.
    assert (answer.requests, answer.fallback) == (0, False)
    assert attributes(named(spans, "answer"))["handbook_rag.requests"] == 0
    assert attributes(named(spans, "answer"))["handbook_rag.fallback"] is False


def test_the_caller_learns_the_trace_id_before_anything_else_happens(
    spans: InMemorySpanExporter,
) -> None:
    happened: list[str] = []

    class Noting(FakeRetriever):
        def retrieve(
            self, query: str, *, limit: int, section: str | None = None
        ) -> list[ScoredChunk]:
            happened.append("search")
            return []

    answer_question(
        QUESTION, retriever=Noting([]), generator=FakeGenerator(), limit=1, on_trace=happened.append
    )

    root = named(spans, "answer")
    assert happened == [format(root.context.trace_id, "032x"), "search"]


def test_the_trace_id_is_not_passed_on_when_nothing_is_recorded() -> None:
    told: list[str] = []

    answer_question(
        QUESTION,
        retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]),
        generator=FakeGenerator(),
        limit=1,
        on_trace=told.append,
    )

    assert told == []


# An answer that takes more than one request, through the wrapper that sends them.


class Flaky(StandInModel):
    """A stand-in model that fails with the given errors first, then answers."""

    def __init__(self, *failures: GenerationError, model: str = "stand-in-model") -> None:
        self.failures = list(failures)
        self.model = model

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        if self.failures:
            raise self.failures.pop(0)
        if on_text is not None:
            on_text("An answer.")
        return reply(question, chunks)


def retrying(model: Generator, fallback: Generator | None = None) -> RetryingGenerator:
    return RetryingGenerator(
        model,
        attempts=3,
        deadline_seconds=30.0,
        fallback=None if fallback is None else lambda: fallback,
        sleep=lambda _seconds: None,
    )


def test_an_answer_that_took_two_requests_has_a_span_for_each_and_counts_them(
    spans: InMemorySpanExporter,
) -> None:
    generator = retrying(Flaky(GenerationError("The model is overloaded.")))

    answer = answer_question(
        QUESTION, retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]), generator=generator, limit=1
    )

    assert (answer.requests, answer.fallback) == (2, False)
    # One span for each request, the failed one included, and no span around both.
    assert paths(spans) == [
        "answer",
        "answer > search",
        "answer > chat stand-in-model",
        "answer > chat stand-in-model",
    ]
    first, second = sorted(
        (span for span in spans.get_finished_spans() if span.name == "chat stand-in-model"),
        key=lambda span: span.start_time or 0,
    )
    assert (first.status.status_code, attributes(first)["handbook_rag.attempt"]) == (
        StatusCode.ERROR,
        1,
    )
    assert (second.status.status_code, attributes(second)["handbook_rag.attempt"]) == (
        StatusCode.UNSET,
        2,
    )
    whole = attributes(named(spans, "answer"))
    assert (whole["handbook_rag.requests"], whole["handbook_rag.fallback"]) == (2, False)


def test_an_answer_written_by_the_fallback_says_so_on_the_answer_and_in_the_trace(
    spans: InMemorySpanExporter,
) -> None:
    used_up = GenerationError("Today's limit is used up.", reason="quota")
    generator = retrying(Flaky(used_up), fallback=Flaky(model="fallback-model"))

    answer = answer_question(
        QUESTION, retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]), generator=generator, limit=1
    )

    assert (answer.requests, answer.fallback) == (2, True)
    assert paths(spans)[2:] == ["answer > chat stand-in-model", "answer > chat fallback-model"]
    assert attributes(named(spans, "chat fallback-model"))["handbook_rag.fallback"] is True
    whole = attributes(named(spans, "answer"))
    assert (whole["handbook_rag.requests"], whole["handbook_rag.fallback"]) == (2, True)


@pytest.mark.parametrize("wrapped", [False, True])
def test_a_failed_answer_hands_on_the_passages_it_found_without_searching_again(
    wrapped: bool, spans: InMemorySpanExporter
) -> None:
    retriever = FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])
    busy = [GenerationError("The model is overloaded.") for _ in range(3)]
    generator: Generator = (
        retrying(Flaky(*busy), fallback=Flaky(GenerationError("Down too."), model="fallback"))
        if wrapped
        else FakeGenerator(error=busy[0])
    )

    with pytest.raises(GenerationError) as raised:
        answer_question(QUESTION, retriever=retriever, generator=generator, limit=2)

    assert [(found.chunk, found.score) for found in raised.value.retrieved] == [
        (LEAVE, 0.9),
        (EXPENSES, 0.4),
    ]
    assert len(retriever.calls) == 1
    whole = attributes(named(spans, "answer"))
    assert (whole["handbook_rag.requests"], whole["handbook_rag.fallback"]) == (
        (4, True) if wrapped else (1, False)
    )
