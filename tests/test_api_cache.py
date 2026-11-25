"""POST /ask with the answer cache: what a caller is told, and what the service builds.

The cache is the real one over a collection in memory. The answer model, the embedding
model and the verifier are stand-ins, so nothing is downloaded or loaded and no request
leaves the machine.
"""

import json
import logging
import re
import threading
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from handbook_rag import wiring
from handbook_rag.api import CacheStartError, create_app
from handbook_rag.cache import answers as cache_answers
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.verify import Verification
from handbook_rag.config import Settings
from handbook_rag.generate.base import Generator
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from tests.cache.support import TableVerifier, memory_answers
from tests.cache.test_answers import Down, Slow, Vectors
from tests.evals.test_cache_pairs import towards
from tests.support import FakeEmbedder, FakeSparseEncoder
from tests.test_api import (
    LEAVE_SOURCE,
    QUESTION,
    Writing,
    answer_event,
    answers_ended,
    indexed_store,
)

REWORDED = "how long is paid leave, in weeks"
NEAR_MISS = "how many weeks of unpaid leave"
VECTORS = {QUESTION: towards(1.0), REWORDED: [1.0, 0.0, 0.0], NEAR_MISS: towards(0.97)}
SCORES = {(QUESTION, REWORDED): (0.96, 0.91), (QUESTION, NEAR_MISS): (0.98, 0.03)}


def both_tiers(**options: Any) -> AnswerCache:
    return AnswerCache(
        options.pop("store", None) or memory_answers(),
        embedder=Vectors(VECTORS),
        verifier=TableVerifier(SCORES),
        threshold=0.8,
        **options,
    )


def cached(
    model: Generator, cache: AnswerCache | None, settings: Settings | None = None
) -> FastAPI:
    """The API over the two-chunk index, answering with a stand-in model and this cache."""
    settings = settings or Settings()
    store, embedder, sparse_encoder = indexed_store()
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )
    return create_app(settings, retriever=retriever, store=store, generator=model, cache=cache)


def test_an_answer_asked_for_again_says_that_it_came_from_the_cache() -> None:
    model = Writing(("Sixteen weeks.", (0,)), (" Ask the People team.", ()))
    with TestClient(cached(model, both_tiers())) as client:
        first = client.post("/ask", json={"query": QUESTION}).json()
        second = client.post("/ask", json={"query": f"{QUESTION.upper()}?"}).json()

    assert (first["cached"], first["requests"]) == (None, 1)
    assert len(model.asked) == 1, "the second answer cost no request to the model"
    hit = second["cached"]
    assert set(hit) == {"tier", "question", "similarity", "score", "stored_at"}
    assert (hit["tier"], hit["question"]) == ("exact", QUESTION)
    assert (hit["similarity"], hit["score"]) == (None, None)
    assert hit["stored_at"].endswith("+00:00"), "a time in UTC"
    # The stored answer word for word, with its markers and its sources.
    for field in ("answer", "answered", "segments", "sources", "model", "stop_reason"):
        assert second[field] == first[field], field
    assert second["sources"] == [{"number": 1, **LEAVE_SOURCE}]
    # No model was asked: no request, no tokens.
    assert (second["requests"], second["fallback"]) == (0, False)
    assert second["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert second["first_text_ms"] is not None


def test_a_rewording_is_served_with_the_question_the_answer_was_written_for() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(cached(model, both_tiers())) as client:
        client.post("/ask", json={"query": QUESTION})
        reworded = client.post("/ask", json={"query": REWORDED}).json()
        near_miss = client.post("/ask", json={"query": NEAR_MISS}).json()

    hit = reworded["cached"]
    assert (hit["tier"], hit["question"]) == ("semantic", QUESTION)
    assert hit["similarity"] == pytest.approx(1.0)
    assert hit["score"] == 0.91, "the weaker direction"
    assert reworded["requests"] == 0
    # The near-miss is not served the first question's answer: the model is asked.
    assert (near_miss["cached"], near_miss["requests"]) == (None, 1)
    assert [question for question, _ in model.asked] == [QUESTION, NEAR_MISS]


def test_a_hit_is_streamed_as_one_piece_of_text_then_the_answer() -> None:
    model = Writing(("Sixteen weeks.", (0,)), ("\nAsk the People team.", ()))
    with TestClient(cached(model, both_tiers())) as client:
        client.post("/ask", json={"query": QUESTION})
        response = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    searched, text, answer = re.split(b"(?=event: (?:text|answer)\n)", response.content)
    # The question searched for first, then the whole text in one event, without citation
    # markers, as a model's pieces are.
    assert searched.startswith(b"event: searched\n")
    assert text == (b'event: text\ndata: {"text":"Sixteen weeks.\\nAsk the People team."}\n\n')
    body = json.loads(answer_event(response.content))
    assert body["answer"] == "Sixteen weeks.[1]\nAsk the People team."
    assert body["cached"]["tier"] == "exact"
    assert (body["requests"], len(model.asked)) == (0, 1)
    assert answer.endswith(b"\n\n")


def test_a_caller_can_ask_for_a_fresh_answer_which_takes_the_stored_ones_place() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(cached(model, both_tiers())) as client:
        client.post("/ask", json={"query": QUESTION})
        model.pieces = (("Sixteen weeks, fully paid.", (0,)),)
        fresh = client.post("/ask", json={"query": QUESTION, "cache": False}).json()
        after = client.post("/ask", json={"query": QUESTION}).json()

    assert (fresh["cached"], fresh["requests"]) == (None, 1)
    assert fresh["answer"] == "Sixteen weeks, fully paid.[1]"
    assert len(model.asked) == 2
    assert after["cached"]["tier"] == "exact"
    assert after["answer"] == "Sixteen weeks, fully paid.[1]"


@pytest.mark.parametrize("value", ["false", 0, None, "no"])
def test_the_field_that_asks_for_a_fresh_answer_is_true_or_false_and_nothing_else(
    value: object,
) -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(cached(model, both_tiers())) as client:
        response = client.post("/ask", json={"query": QUESTION, "cache": value})

    assert response.status_code == 422
    assert [error["loc"][-1] for error in response.json()["detail"]] == ["cache"]
    assert model.asked == []


def test_a_stored_decline_is_served_to_the_same_question_as_a_reply_that_is_not_an_answer() -> None:
    model = Writing((NOT_FOUND_MESSAGE, ()), answered=False)
    cache = both_tiers()
    cache._verifier = TableVerifier({(QUESTION, REWORDED): (1.0, 1.0)})
    with TestClient(cached(model, cache)) as client:
        client.post("/ask", json={"query": QUESTION})
        same = client.post("/ask", json={"query": QUESTION}).json()
        reworded = client.post("/ask", json={"query": REWORDED}).json()

    assert (same["answer"], same["answered"], same["sources"]) == (NOT_FOUND_MESSAGE, False, [])
    assert same["cached"]["tier"] == "exact"
    # By meaning it is never served, whatever the verifier would say of the two questions.
    assert (reworded["cached"], reworded["requests"]) == (None, 1)


def test_a_service_without_a_cache_takes_the_field_and_asks_the_model_each_time() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(cached(model, None)) as client:
        first = client.post("/ask", json={"query": QUESTION, "cache": False}).json()
        second = client.post("/ask", json={"query": QUESTION}).json()

    assert (first["cached"], second["cached"]) == (None, None)
    assert len(model.asked) == 2


def test_a_cache_whose_store_is_down_fails_no_answer() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    cache = both_tiers(store=Down())
    with TestClient(cached(model, cache)) as client:
        plain = client.post("/ask", json={"query": QUESTION})
        streamed = client.post("/ask", json={"query": QUESTION, "stream": True})

    assert (plain.status_code, plain.json()["answer"]) == (200, "Sixteen weeks.[1]")
    assert plain.json()["cached"] is None
    assert json.loads(answer_event(streamed.content))["answer"] == "Sixteen weeks.[1]"
    assert len(model.asked) == 2


def test_an_answer_does_not_wait_for_a_slow_store_and_is_kept_when_the_service_ends() -> None:
    store = Slow()
    cache = both_tiers(store=store, background=True)
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(cached(model, cache)) as client:
        response = client.post("/ask", json={"query": QUESTION})

        # The whole answer is back while the store has not let the write through.
        assert response.status_code == 200
        assert store.writing.wait(5.0)
        assert store.puts == 0
        threading.Timer(0.05, store.release.set).start()
    # Leaving the service waited for the answer still being stored.
    assert store.puts == 1
    assert store.held().answers == 1
    assert answers_ended()


# What the service builds when it starts, by the settings.


class Models:
    """Stand-ins for what the service would load, each noting when it was built or asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, verifier_error: Exception | None = None):
        self.events: list[str] = []
        self.verifier = TableVerifier(error=None)
        self.embedder = embedder = FakeEmbedder()

        def make_embedder(_settings: Settings) -> FakeEmbedder:
            self.events.append("embedder built")
            return embedder

        def make_verifier(_settings: Settings) -> TableVerifier:
            self.events.append("verifier built")
            if verifier_error is not None:
                raise verifier_error
            return self.verifier

        monkeypatch.setattr(wiring, "make_embedder", make_embedder)
        monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
        monkeypatch.setattr(wiring, "make_verifier", make_verifier)


def built_by_the_service(settings: Settings, model: Generator | None) -> FastAPI:
    """The service as `handbook-rag serve` builds it, but for its store and its answer model."""
    store, _, _ = indexed_store()
    return create_app(settings, store=store, generator=model)


def test_by_default_the_service_has_both_tiers_and_loads_the_verifier_before_any_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))
    assert Settings().cache_mode == "semantic"

    with TestClient(built_by_the_service(Settings(), model)) as client:
        cache = client.app.state.cache  # type: ignore[attr-defined]
        assert isinstance(cache, AnswerCache)
        assert cache.mode == "semantic"
        # Loaded and asked once when the service started: no question waits for it.
        assert models.events == ["embedder built", "verifier built"]
        assert models.verifier.calls == [[cache_answers._WARM_UP]]
        # The embedding model was run for the cache too. The first run of a model in a
        # process is slow, and after this a lookup counts every embedding in its time.
        assert cache_answers._WARM_UP[0] in models.embedder.queries
        first = client.post("/ask", json={"query": QUESTION}).json()
        assert cache.close() is True, "the answer is stored by a thread of its own"
        second = client.post("/ask", json={"query": f"{QUESTION}?"}).json()

    assert (first["cached"], second["cached"]["tier"]) == (None, "exact")
    assert len(model.asked) == 1
    assert models.events == ["embedder built", "verifier built"], "each model once"


def test_the_threshold_the_floor_and_the_candidates_of_the_service_are_those_of_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))
    settings = Settings(cache_threshold=0.9, cache_floor=0.1, cache_candidates=1)
    under, at = "how many weeks of paid leave do parents get", "how many weeks of paid leave"
    models.verifier.scores[(QUESTION, under)] = (0.95, 0.8999)
    models.verifier.scores[(QUESTION, f"{at} is there")] = (0.95, 0.9)

    with TestClient(built_by_the_service(settings, model)) as client:
        client.post("/ask", json={"query": QUESTION})
        client.app.state.cache.close()  # type: ignore[attr-defined]
        below = client.post("/ask", json={"query": under, "cache": True}).json()
        reached = client.post("/ask", json={"query": f"{at} is there"}).json()

    assert below["cached"] is None, "just under the threshold of the settings"
    assert (reached["cached"]["question"], reached["cached"]["score"]) == (QUESTION, 0.9)
    # One candidate for each question, as the settings say: the nearest stored question.
    assert [len(pairs) for pairs in models.verifier.calls[1:]] == [1, 1]
    # With the default floor of 0.5 the stand-in's vectors would still be near enough.
    assert reached["cached"]["similarity"] > 0.1


def test_in_the_exact_mode_the_service_never_builds_a_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))

    with TestClient(built_by_the_service(Settings(cache_mode="exact"), model)) as client:
        cache = client.app.state.cache  # type: ignore[attr-defined]
        assert cache.mode == "exact"
        client.post("/ask", json={"query": QUESTION})
        cache.close()
        same = client.post("/ask", json={"query": QUESTION}).json()
        other = client.post("/ask", json={"query": "how many weeks of paid leave are there"}).json()

    assert (same["cached"]["tier"], other["cached"]) == ("exact", None)
    assert models.events == ["embedder built"]


def test_switched_off_the_service_has_no_cache_and_builds_no_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))

    with TestClient(built_by_the_service(Settings(cache_mode="off"), model)) as client:
        assert client.app.state.cache is None  # type: ignore[attr-defined]
        answers = [client.post("/ask", json={"query": QUESTION}).json() for _ in range(2)]

    assert [answer["cached"] for answer in answers] == [None, None]
    assert len(model.asked) == 2
    assert models.events == ["embedder built"]


def test_a_service_without_an_answer_model_builds_no_cache_and_loads_no_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No key in the settings, so no answer model: nothing could ever be kept or served.
    models = Models(monkeypatch)

    with TestClient(built_by_the_service(Settings(), None)) as client:
        assert client.app.state.cache is None  # type: ignore[attr-defined]
        assert client.app.state.generator is None  # type: ignore[attr-defined]
        assert client.post("/search", json={"query": QUESTION}).status_code == 200

    assert models.events == ["embedder built"]


def test_a_caller_that_brings_its_own_retriever_gets_no_cache_built_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # How most tests build the service. With the default mode, which is `semantic`, a
    # cache built for them would load the embedding model and the verifier.
    models = Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))

    with TestClient(cached(model, None, Settings())) as client:
        assert client.app.state.cache is None  # type: ignore[attr-defined]
        client.post("/ask", json={"query": QUESTION})
        client.post("/ask", json={"query": QUESTION})

    assert models.events == []
    assert len(model.asked) == 2


def test_a_verifier_that_cannot_be_loaded_stops_the_service_from_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Models(monkeypatch, verifier_error=OSError("the model is not on disk and the Hub is down"))
    model = Writing(("Sixteen weeks.", (0,)))

    with (
        pytest.raises(CacheStartError) as raised,
        TestClient(built_by_the_service(Settings(), model)),
    ):
        pass

    # The model's own error names no setting. This one says which to change.
    message = str(raised.value)
    assert "cross-encoder/nli-deberta-v3-small" in message
    assert "OSError: the model is not on disk and the Hub is down" in message
    assert "HANDBOOK_RAG_CACHE_MODE=semantic needs it" in message
    assert "HANDBOOK_RAG_CACHE_VERIFIER_MODEL" in message
    assert "set HANDBOOK_RAG_CACHE_MODE to exact or to off" in message
    assert isinstance(raised.value.__cause__, OSError)


def test_the_service_has_made_the_caches_collection_before_it_takes_a_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Made by the first answer that is stored, it would be made while questions are
    already being looked up, and for those seconds a lookup finds nothing or fails."""
    Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))

    with TestClient(built_by_the_service(Settings(), model)) as client:
        store: Any = client.app.state.cache._store  # type: ignore[attr-defined]
        assert store._client.collection_exists(store.collection)
        made = store._client.get_collection(store.collection).config.params.vectors
        assert made.size == FakeEmbedder.dimension, "in the size of the embedding model"
        assert model.asked == [], "before any question"


def test_a_service_whose_store_is_down_starts_all_the_same_and_logs_the_cache(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    Models(monkeypatch)
    model = Writing(("Sixteen weeks.", (0,)))
    settings = Settings(qdrant_url="http://127.0.0.1:1")

    with (
        caplog.at_level(logging.WARNING),
        TestClient(create_app(settings, generator=model)) as client,
    ):
        cache = client.app.state.cache  # type: ignore[attr-defined]
        assert isinstance(cache, AnswerCache), "the cache is there, for when the store is back"
        refused = client.post("/ask", json={"query": QUESTION})

    assert "the answer cache could not be prepared" in caplog.text
    # The search has no store either, so there is no answer, and the cache did not hide that.
    assert refused.status_code == 503
    assert model.asked == []


def test_a_score_from_the_verifier_reaches_the_caller_as_the_weaker_direction() -> None:
    class Lopsided(TableVerifier):
        def score(self, pairs: Any) -> list[Verification]:
            return [Verification(forward=0.999, backward=0.85) for _ in pairs]

    cache = both_tiers()
    cache._verifier = Lopsided()
    with TestClient(cached(Writing(("Sixteen weeks.", (0,))), cache)) as client:
        client.post("/ask", json={"query": QUESTION})
        hit = client.post("/ask", json={"query": REWORDED}).json()["cached"]

    assert hit["score"] == 0.85
