"""The one entry point of an answer, with an answer cache: what is served, kept and traced.

``tests/test_pipeline.py`` holds the entry point without a cache, which is how an
evaluation calls it. The cache here is the real one over a collection in memory, with
stand-ins for the embedding model and the verifier.
"""

from collections.abc import Callable, Sequence

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from handbook_rag import tracing
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer, Citation, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk
from handbook_rag.pipeline import answer_question
from tests.cache.support import TableVerifier, memory_answers
from tests.cache.test_answers import Down, Vectors
from tests.evals.test_cache_pairs import towards
from tests.support import StandInModel, everything_recorded
from tests.test_pipeline import (
    EXPENSES,
    LEAVE,
    FakeGenerator,
    FakeRetriever,
    attributes,
    named,
    paths,
)

STORED = "How many weeks of parental leave do I get?"
REWORDED = "How long is my parental leave?"
NEAR_MISS = "How many weeks of parental leave does GitHub give?"
VECTORS = {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0], NEAR_MISS: towards(0.98)}


class Counting(FakeGenerator):
    """A stand-in answer model that says how often it was asked."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.asked: list[str] = []

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.asked.append(question)
        return super().generate(question, chunks, on_text=on_text)


def both_tiers(verifier: TableVerifier | None = None) -> AnswerCache:
    scores = {(STORED, REWORDED): (0.97, 0.93), (STORED, NEAR_MISS): (0.99, 0.02)}
    return AnswerCache(
        memory_answers(),
        embedder=Vectors(VECTORS),
        verifier=verifier or TableVerifier(scores),
        threshold=0.8,
    )


def retriever() -> FakeRetriever:
    return FakeRetriever([ScoredChunk(LEAVE, 0.9), ScoredChunk(EXPENSES, 0.4)])


def ask(
    question: str,
    cache: AnswerCache | None,
    model: FakeGenerator,
    found: FakeRetriever | None = None,
    **options: object,
) -> Answer:
    return answer_question(
        question,
        retriever=found or retriever(),
        generator=model,
        limit=2,
        cache=cache,
        **options,  # type: ignore[arg-type]
    )


def test_a_question_asked_again_is_served_without_a_search_or_a_request() -> None:
    cache, model, found = both_tiers(), Counting(), retriever()
    received: list[str] = []

    first = ask(STORED, cache, model, found)
    second = ask(f"  {STORED.lower()}  ", cache, model, found, on_text=received.append)

    assert (first.cached, first.requests) == (None, 1)
    assert len(found.calls) == 1, "no second search"
    assert model.asked == [STORED], "no second request"
    assert second.cached is not None
    assert (second.cached.tier, second.cached.question) == ("exact", STORED)
    assert (second.requests, second.fallback, second.retrieved) == (0, False, ())
    assert second.segments == first.segments, "word for word, with its citation"
    assert second.sources == (LEAVE,)
    assert received == ["An answer."], "the whole text, as one piece"


def test_a_rewording_is_served_by_meaning_and_a_near_miss_gets_an_answer_of_its_own() -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)

    reworded = ask(REWORDED, cache, model)
    near_miss = ask(NEAR_MISS, cache, model)

    assert reworded.cached is not None
    assert (reworded.cached.tier, reworded.cached.question) == ("semantic", STORED)
    assert reworded.cached.similarity == pytest.approx(1.0)
    assert reworded.cached.score == 0.93
    assert reworded.requests == 0
    # The near-miss is nearer than the floor and is turned down by the verifier.
    assert (near_miss.cached, near_miss.requests) == (None, 1)
    assert model.asked == [STORED, NEAR_MISS]


def test_a_stored_decline_is_served_to_the_same_question_and_never_to_a_rewording() -> None:
    def decline(question: str, chunks: Sequence[Chunk]) -> Answer:
        return Answer(question, (Segment(NOT_FOUND_MESSAGE),), False, "fake-model", "end_turn")

    cache, model = both_tiers(TableVerifier({(STORED, REWORDED): (1.0, 1.0)})), Counting(decline)
    ask(STORED, cache, model)

    same = ask(STORED, cache, model)
    reworded = ask(REWORDED, cache, model)

    assert same.cached is not None
    assert (same.cached.tier, same.text, same.answered) == ("exact", NOT_FOUND_MESSAGE, False)
    assert reworded.cached is None, "the verifier would have passed it, and is not asked"
    assert model.asked == [STORED, REWORDED]


def test_a_fresh_answer_is_written_by_the_model_and_takes_the_stored_ones_place() -> None:
    replies = iter(["The first answer.", "The second answer."])

    def next_reply(question: str, chunks: Sequence[Chunk]) -> Answer:
        return Answer(
            question, (Segment(next(replies), (Citation(chunks[0]),)),), True, "fake", "end_turn"
        )

    cache, model = both_tiers(), Counting(next_reply)
    ask(STORED, cache, model)

    fresh = ask(STORED, cache, model, fresh=True)
    after = ask(STORED, cache, model)

    assert (fresh.cached, fresh.requests, fresh.text) == (None, 1, "The second answer.")
    assert model.asked == [STORED, STORED]
    assert after.cached is not None
    assert after.text == "The second answer."


def test_a_question_asked_of_one_section_does_not_get_the_answer_written_from_all() -> None:
    cache, model, found = both_tiers(), Counting(), retriever()

    ask(STORED, cache, model, found)
    of_people = ask(STORED, cache, model, found, section="people")
    again = ask(STORED, cache, model, found, section="people")

    assert (of_people.cached, of_people.requests) == (None, 1)
    assert again.cached is not None
    assert [section for _, _, section in found.calls] == [None, "people"]


def test_a_failed_answer_is_not_kept_and_still_leaves_with_what_the_search_found() -> None:
    cache = both_tiers()
    failing = Counting(error=GenerationError("The model is overloaded."))

    with pytest.raises(GenerationError) as raised:
        ask(STORED, cache, failing)

    assert [found.chunk for found in raised.value.retrieved] == [LEAVE, EXPENSES]
    assert cache.lookup(STORED).answer is None
    # The next time the model is asked again, and its answer is kept.
    model = Counting()
    assert ask(STORED, cache, model).cached is None
    assert ask(STORED, cache, model).cached is not None


def test_a_reply_that_broke_off_after_its_first_words_is_not_kept() -> None:
    class BreaksOff(StandInModel):
        def generate(
            self,
            question: str,
            chunks: Sequence[Chunk],
            *,
            on_text: Callable[[str], None] | None = None,
        ) -> Answer:
            if on_text is not None:
                on_text("Sixteen ")
            raise GenerationError("The Gemini API closed the stream.")

    cache = both_tiers()

    with pytest.raises(GenerationError):
        answer_question(
            STORED,
            retriever=retriever(),
            generator=BreaksOff(),
            limit=2,
            cache=cache,
            on_text=lambda _text: None,
        )

    assert cache.lookup(STORED).answer is None


def test_nothing_found_is_declined_without_a_model_and_that_decline_is_not_kept() -> None:
    # An index that is empty or half built finds nothing. Its declines must not outlive it.
    cache, model = both_tiers(), Counting()

    empty = ask(STORED, cache, model, FakeRetriever([]))
    filled = ask(STORED, cache, model)

    assert (empty.stop_reason, empty.requests) == ("no_context", 0)
    assert (filled.cached, filled.requests) == (None, 1), "asked properly once there are passages"


def test_a_cache_that_cannot_be_reached_changes_nothing_about_the_answer(
    spans: InMemorySpanExporter,
) -> None:
    cache = AnswerCache(Down(), embedder=Vectors(VECTORS), verifier=TableVerifier(), threshold=0.8)
    model = Counting()

    answer = ask(STORED, cache, model)

    assert (answer.text, answer.cached, answer.requests) == ("An answer.", None, 1)
    looked = attributes(named(spans, "cache.lookup"))
    assert looked["handbook_rag.cache.hit"] is False
    assert looked["handbook_rag.cache.failure"] == "builtins.ConnectionError"
    # The lookup did not fail the answer, and is not marked as a failed stage.
    assert named(spans, "cache.lookup").status.status_code is not StatusCode.ERROR
    assert named(spans, "answer").status.status_code is not StatusCode.ERROR


# The trace. Every caller with a cache gets it, because the pipeline opens the spans.


def test_a_hit_is_a_trace_that_ends_with_the_lookup(spans: InMemorySpanExporter) -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)
    spans.clear()

    ask(STORED, cache, model)

    assert paths(spans) == [
        "answer",
        "answer > cache.lookup",
        "answer > cache.lookup > vector_store.retrieve",
    ]
    assert attributes(named(spans, "cache.lookup")) == {
        "handbook_rag.mode": "semantic",
        "handbook_rag.cache.read": True,
        "handbook_rag.cache.hit": True,
        "handbook_rag.cache.tier": "exact",
        "handbook_rag.cache.candidates": 0,
    }
    assert attributes(named(spans, "answer")) == {
        "handbook_rag.query.chars": len(STORED),
        "handbook_rag.answered": True,
        "handbook_rag.requests": 0,
        "handbook_rag.fallback": False,
    }


def test_a_miss_records_each_candidate_with_its_scores_and_then_the_answer_is_written(
    spans: InMemorySpanExporter,
) -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)
    spans.clear()

    ask(NEAR_MISS, cache, model)

    assert paths(spans) == [
        "answer",
        "answer > cache.lookup",
        "answer > cache.lookup > vector_store.retrieve",
        "answer > cache.lookup > embed.query",
        "answer > cache.lookup > vector_store.query",
        "answer > cache.lookup > cache.verify",
        "answer > search",
        "answer > chat stand-in-model",
    ]
    assert attributes(named(spans, "cache.lookup")) == {
        "handbook_rag.mode": "semantic",
        "handbook_rag.cache.read": True,
        "handbook_rag.cache.hit": False,
        "handbook_rag.cache.candidates": 1,
        "handbook_rag.cache.similarities": (0.98,),
        "handbook_rag.cache.entails_forward": (0.99,),
        "handbook_rag.cache.entails_backward": (0.02,),
    }
    assert attributes(named(spans, "cache.verify")) == {
        "handbook_rag.model": "table-verifier",
        "handbook_rag.pairs": 1,
    }
    assert attributes(named(spans, "embed.query")) == {"handbook_rag.model": "table-embedder"}
    assert attributes(named(spans, "answer"))["handbook_rag.requests"] == 1
    # Keeping the new answer is no part of the trace: nothing is recorded after it.
    assert len({span.context.trace_id for span in spans.get_finished_spans()}) == 1


def test_a_hit_by_meaning_records_its_tier_and_scores(spans: InMemorySpanExporter) -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)
    spans.clear()

    ask(REWORDED, cache, model)

    looked = attributes(named(spans, "cache.lookup"))
    assert (looked["handbook_rag.cache.hit"], looked["handbook_rag.cache.tier"]) == (
        True,
        "semantic",
    )
    assert looked["handbook_rag.cache.similarities"] == (1.0,)
    assert looked["handbook_rag.cache.entails_forward"] == (0.97,)
    assert looked["handbook_rag.cache.entails_backward"] == (0.93,)
    assert "search" not in {span.name for span in spans.get_finished_spans()}


def test_a_fresh_answer_is_traced_as_a_lookup_that_read_nothing(
    spans: InMemorySpanExporter,
) -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)
    spans.clear()

    ask(STORED, cache, model, fresh=True)

    assert paths(spans) == [
        "answer",
        "answer > cache.lookup",
        "answer > search",
        "answer > chat stand-in-model",
    ]
    assert attributes(named(spans, "cache.lookup")) == {
        "handbook_rag.mode": "semantic",
        "handbook_rag.cache.read": False,
        "handbook_rag.cache.hit": False,
        "handbook_rag.cache.candidates": 0,
    }


def test_the_stored_questions_are_left_out_of_a_trace_unless_content_is_switched_on(
    spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, model = both_tiers(), Counting()
    ask(STORED, cache, model)
    spans.clear()

    ask(NEAR_MISS, cache, model)
    ask(REWORDED, cache, model)
    quiet = everything_recorded(spans.get_finished_spans())

    for text in (STORED, NEAR_MISS, REWORDED, "An answer."):
        assert text not in quiet
    assert "handbook_rag.cache.questions" not in quiet

    spans.clear()
    monkeypatch.setattr(tracing, "_record_content", True)
    ask("How long is parental leave, in weeks?", cache, model)
    ask(REWORDED, cache, model)

    (miss, hit) = [span for span in spans.get_finished_spans() if span.name == "cache.lookup"]
    assert "handbook_rag.cache.questions" not in attributes(miss), "it had no candidate"
    # Every candidate's stored question, nearest first: the near-miss's answer is stored too.
    assert attributes(hit)["handbook_rag.cache.questions"] == (STORED, NEAR_MISS)


def test_a_verifier_that_failed_leaves_the_similarities_and_the_kind_of_error(
    spans: InMemorySpanExporter,
) -> None:
    cache = both_tiers(TableVerifier(error=RuntimeError("out of memory at /Users/someone")))
    model = Counting()
    ask(STORED, cache, model)
    spans.clear()

    answer = ask(REWORDED, cache, model)

    assert (answer.cached, answer.requests) == (None, 1), "the question is answered all the same"
    looked = attributes(named(spans, "cache.lookup"))
    assert looked["handbook_rag.cache.failure"] == "builtins.RuntimeError"
    assert looked["handbook_rag.cache.similarities"] == (1.0,)
    assert "handbook_rag.cache.entails_forward" not in looked
    assert named(spans, "cache.verify").status.status_code is StatusCode.ERROR
    assert "/Users/someone" not in everything_recorded(spans.get_finished_spans())


def test_without_a_cache_no_lookup_is_traced_and_nothing_is_kept(
    spans: InMemorySpanExporter,
) -> None:
    model = Counting()

    first = ask(STORED, None, model)
    second = ask(STORED, None, model, fresh=True)

    assert (first.cached, second.cached) == (None, None)
    assert model.asked == [STORED, STORED]
    assert "cache.lookup" not in {span.name for span in spans.get_finished_spans()}
