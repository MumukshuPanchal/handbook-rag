"""Where cached answers are kept: a Qdrant collection, here in this process's memory."""

import dataclasses
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from handbook_rag.cache.store import (
    COLLECTION_PREFIX,
    Entry,
    Held,
    QdrantAnswerStore,
    answer_collections,
)
from handbook_rag.generate.models import Citation, Segment
from tests.cache.support import memory_answers
from tests.evals.test_cache_pairs import towards
from tests.support import everything_recorded, make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental", "US"))
LAPTOP = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
NEW = [1.0, 0.0, 0.0]


def entry(name: str = "first", **changes: Any) -> Entry:
    made = Entry(
        # A point's id has to be a UUID, as the cache's own ids are.
        entry_id=f"00000000-0000-0000-0000-{abs(hash(name)) % 10**12:012d}",
        question=f"The {name} question?",
        comparable=f"the {name} question",
        scope="all",
        segments=(
            Segment("Sixteen weeks", (Citation(LEAVE, "sixteen weeks"), Citation(LAPTOP))),
            Segment(", and a laptop.", (Citation(LAPTOP),)),
            Segment(" Ask the People team."),
        ),
        answered=True,
        model="stand-in-model-002",
        stop_reason="end_turn",
        stored_at="2031-02-03T04:05:06+00:00",
        by_meaning=True,
    )
    return dataclasses.replace(made, **changes)


def test_an_entry_comes_back_as_it_was_stored_with_its_passages_whole() -> None:
    store = memory_answers()
    stored = entry()

    store.put(stored, towards(1.0))
    found = store.get(stored.entry_id)

    assert found == stored
    assert found is not None
    # The passages are whole, so serving the answer needs nothing from the handbook's index.
    (first, second, _) = found.segments
    assert first.citations[0].chunk == LEAVE
    assert first.citations[0].chunk.heading_path == ("Parental", "US"), "a tuple again"
    assert (first.citations[0].cited_text, first.citations[1].cited_text) == ("sixteen weeks", None)
    assert second.citations[0].chunk == LAPTOP
    # Each cited passage is written once, however often it is cited.
    (point,) = store._client.retrieve(store.collection, ids=[stored.entry_id], with_payload=True)
    assert point.payload is not None
    assert sorted(point.payload["chunks"]) == sorted([LEAVE.chunk_id, LAPTOP.chunk_id])


def test_a_cache_nothing_was_stored_in_is_empty_and_reading_it_is_no_failure() -> None:
    store = memory_answers()

    assert store.get(entry().entry_id) is None
    assert store.nearest(NEW, scope="all", limit=3, floor=0.5) == []
    assert store.held() == Held(answers=0, declines=0, by_meaning=0)
    assert store.clear() == 0
    assert not store._client.collection_exists(store.collection), "reading makes nothing"


def test_the_collection_is_made_with_the_first_answer_in_the_size_of_its_vector() -> None:
    store = memory_answers()

    store.put(entry(), [0.0, 1.0, 0.0, 0.0, 0.0])

    made = store._client.get_collection(store.collection).config.params.vectors
    assert isinstance(made, models.VectorParams)
    assert (made.size, made.distance) == (5, models.Distance.COSINE)


def test_the_nearest_are_of_one_scope_may_be_served_by_meaning_and_are_above_the_floor() -> None:
    store = memory_answers()
    store.put(entry("far"), towards(0.3))
    store.put(entry("near"), towards(0.8))
    store.put(entry("nearest"), towards(0.95))
    store.put(entry("a decline", answered=False, by_meaning=False), towards(0.99))
    store.put(entry("of a section", scope="section:people"), towards(0.97))

    found = store.nearest(NEW, scope="all", limit=5, floor=0.5)
    of_people = store.nearest(NEW, scope="section:people", limit=5, floor=0.5)
    one = store.nearest(NEW, scope="all", limit=1, floor=0.5)

    assert [(e.comparable, round(score, 2)) for e, score in found] == [
        ("the nearest question", 0.95),
        ("the near question", 0.8),
    ]
    assert [e.comparable for e, _ in of_people] == ["the of a section question"]
    # The filter is inside the search: what may not be served takes no nearer one's place.
    assert [e.comparable for e, _ in one] == ["the nearest question"]


def test_a_store_says_how_much_it_holds_and_emptying_it_removes_the_collection() -> None:
    store = memory_answers()
    store.put(entry("one"), towards(1.0))
    store.put(entry("two", by_meaning=False), towards(0.9))
    store.put(entry("three", answered=False, by_meaning=False), towards(0.8))

    assert store.held() == Held(answers=3, declines=1, by_meaning=1)
    assert store.clear() == 3
    assert store.held() == Held(0, 0, 0)
    assert not store._client.collection_exists(store.collection)


def test_an_answer_stored_after_the_cache_was_emptied_makes_the_collection_again() -> None:
    # The service keeps its store while someone runs `handbook-rag cache-clear`.
    client = QdrantClient(":memory:")
    service = QdrantAnswerStore(client, "answers-shared")
    service.put(entry("before"), towards(1.0))

    assert QdrantAnswerStore(client, "answers-shared").clear() == 1
    assert service.get(entry("before").entry_id) is None, "a miss, not an error"
    assert service.nearest(NEW, scope="all", limit=3, floor=0.5) == []
    service.put(entry("after"), towards(1.0))

    assert service.held().answers == 1
    assert service.get(entry("after").entry_id) == entry("after")


def test_an_entry_with_the_same_id_takes_the_place_of_the_one_before() -> None:
    store = memory_answers()
    store.put(entry(model="first-model"), towards(1.0))
    store.put(entry(model="second-model"), towards(0.2))

    found = store.get(entry().entry_id)

    assert found is not None
    assert found.model == "second-model"
    assert store.held().answers == 1
    assert store.nearest(NEW, scope="all", limit=3, floor=0.5) == [], "its vector went too"


class Failing(QdrantClient):
    """A client whose reads fail with the error a test gives it."""

    def __init__(self, error: Exception) -> None:
        super().__init__(":memory:")
        self.error = error
        self.asked_whether_it_exists = 0

    def retrieve(self, *args: object, **kwargs: object) -> list[models.Record]:
        raise self.error

    def query_points(self, *args: object, **kwargs: object) -> models.QueryResponse:
        raise self.error

    def collection_exists(self, collection_name: str, **kwargs: object) -> bool:
        self.asked_whether_it_exists += 1
        return super().collection_exists(collection_name, **kwargs)


def response(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "reason", b"", None)  # type: ignore[arg-type]


def test_a_missing_collection_is_an_empty_cache_and_any_other_failure_is_a_failure() -> None:
    not_found = QdrantAnswerStore(Failing(response(404)), "answers-x")
    assert not_found.get("id") is None
    assert not_found.nearest(NEW, scope="all", limit=3, floor=0.5) == []

    broken = QdrantAnswerStore(Failing(response(500)), "answers-x")
    with pytest.raises(UnexpectedResponse):
        broken.get("id")
    with pytest.raises(UnexpectedResponse):
        broken.nearest(NEW, scope="all", limit=3, floor=0.5)


def test_a_store_that_did_not_answer_is_not_asked_anything_more() -> None:
    client = Failing(ResponseHandlingException(TimeoutError("timed out")))
    store = QdrantAnswerStore(client, "answers-x")

    with pytest.raises(ResponseHandlingException):
        store.get("id")

    assert client.asked_whether_it_exists == 0, "that could take as long again"


def test_a_plain_error_from_a_collection_that_exists_is_a_failure() -> None:
    # Qdrant's in-memory mode raises plain errors, so the store asks whether it is there.
    client = Failing(ValueError("something else went wrong"))
    store = QdrantAnswerStore(client, "answers-x")
    store.put(entry(), towards(1.0))

    with pytest.raises(ValueError, match="something else"):
        store.get(entry().entry_id)


def test_every_collection_of_cached_answers_is_found_by_how_its_name_starts() -> None:
    client = QdrantClient(":memory:")
    for name in ("answers-bbbbbbbb-2", "handbook-aaaaaaaa-model", "answers-aaaaaaaa-1"):
        QdrantAnswerStore(client, name).put(entry(), towards(1.0))

    assert COLLECTION_PREFIX == "answers-"
    assert answer_collections(client) == ["answers-aaaaaaaa-1", "answers-bbbbbbbb-2"]


def test_reading_the_cache_is_traced_as_a_call_to_the_store_and_holds_no_question(
    spans: InMemorySpanExporter,
) -> None:
    store = memory_answers("answers-traced")
    store.put(entry(), towards(1.0))
    spans.clear()

    store.get(entry().entry_id)
    store.nearest(NEW, scope="all", limit=3, floor=0.5)

    retrieve, query = spans.get_finished_spans()
    assert (retrieve.name, query.name) == ("vector_store.retrieve", "vector_store.query")
    assert retrieve.kind is query.kind is SpanKind.CLIENT
    assert dict(query.attributes or {}) == {
        "db.system.name": "qdrant",
        "db.collection.name": "answers-traced",
        "db.operation.name": "query",
        "handbook_rag.limit": 3,
        "handbook_rag.results.count": 1,
    }
    assert (retrieve.attributes or {})["db.operation.name"] == "retrieve"
    assert "question" not in everything_recorded([retrieve, query])
