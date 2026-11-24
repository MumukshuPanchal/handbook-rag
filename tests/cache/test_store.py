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
    collection_name,
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


def test_a_store_says_how_much_it_holds_and_emptying_it_keeps_the_collection() -> None:
    store = memory_answers()
    store.put(entry("one"), towards(1.0))
    store.put(entry("two", by_meaning=False), towards(0.9))
    store.put(entry("three", answered=False, by_meaning=False), towards(0.8))

    assert store.held() == Held(answers=3, declines=1, by_meaning=1)
    assert store.clear() == 3
    assert store.held() == Held(0, 0, 0)
    # The collection is still there: a running service never meets one being made again.
    assert store._client.collection_exists(store.collection)
    assert store.clear() == 0, "emptying an empty cache changes nothing"


def test_a_service_goes_on_reading_and_storing_while_its_cache_is_emptied() -> None:
    # The service keeps its store while someone runs `handbook-rag cache-clear`, or
    # `handbook-rag index` empties the cache.
    client = Counting()
    service = QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef")
    service.put(entry("before"), towards(1.0))
    made = client.calls.count("create_collection")

    assert QdrantAnswerStore(client, service.collection).clear() == 1
    assert service.get(entry("before").entry_id) is None, "a miss, not an error"
    assert service.nearest(NEW, scope="all", limit=3, floor=0.5) == []
    service.put(entry("after"), towards(1.0))

    assert service.held().answers == 1
    assert service.get(entry("after").entry_id) == entry("after")
    assert client.calls.count("create_collection") == made == 1, "nothing was made again"


def test_a_collection_left_behind_is_removed_whole_and_made_again_if_it_is_still_used() -> None:
    client = QdrantClient(":memory:")
    service = QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef")
    service.put(entry("before"), towards(1.0))

    assert QdrantAnswerStore(client, service.collection).remove() == 1
    assert not client.collection_exists(service.collection)
    assert QdrantAnswerStore(client, service.collection).remove() == 0
    # A service that still ran under those settings reads nothing, and stores again.
    assert service.get(entry("before").entry_id) is None
    assert service.nearest(NEW, scope="all", limit=3, floor=0.5) == []
    service.put(entry("after"), towards(1.0))
    assert service.held().answers == 1


class Counting(QdrantClient):
    """An in-memory client that notes each call that makes or describes a collection."""

    def __init__(self, indexed: tuple[str, ...] | None = None) -> None:
        super().__init__(":memory:")
        self.calls: list[str] = []
        self.indexed = indexed

    def get_collection(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("get_collection")
        described = super().get_collection(*args, **kwargs)
        if self.indexed is not None:
            # The in-memory store keeps no payload index. A server says which it has.
            described.payload_schema = dict.fromkeys(self.indexed)  # type: ignore[assignment]
        return described

    def create_collection(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("create_collection")
        return super().create_collection(*args, **kwargs)

    def create_payload_index(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(f"create_payload_index {kwargs['field_name']}")
        return super().create_payload_index(*args, **kwargs)

    def collection_exists(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("collection_exists")
        return super().collection_exists(*args, **kwargs)


def test_ensuring_makes_the_collection_and_both_indexes_and_is_asked_once() -> None:
    client = Counting()
    store = QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef")

    store.ensure(3)
    store.ensure(3)

    assert client.calls == [
        "get_collection",
        # The in-memory store fails with a plain error, so it is asked what the error
        # meant. A server answers 404, which says it at once.
        "collection_exists",
        "create_collection",
        "create_payload_index scope",
        "create_payload_index by_meaning",
    ]
    made = client.get_collection(store.collection).config.params.vectors
    assert isinstance(made, models.VectorParams)
    assert (made.size, made.distance) == (3, models.Distance.COSINE)
    # Nothing is in it, and reading it finds nothing without a failure.
    assert store.get(entry().entry_id) is None
    assert store.nearest(NEW, scope="all", limit=3, floor=0.5) == []


def test_a_collection_that_is_all_there_costs_one_question_to_the_store() -> None:
    client = Counting(indexed=("scope", "by_meaning"))
    QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef").ensure(3)
    client.calls.clear()

    # Another process, such as the command after the service.
    QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef").ensure(3)

    assert client.calls == ["get_collection"]


@pytest.mark.parametrize(
    ("indexed", "added"),
    [
        ((), ["create_payload_index scope", "create_payload_index by_meaning"]),
        (("scope",), ["create_payload_index by_meaning"]),
        (("by_meaning",), ["create_payload_index scope"]),
    ],
)
def test_a_collection_left_half_made_gets_the_indexes_it_lacks(
    indexed: tuple[str, ...], added: list[str]
) -> None:
    # The process that was making it ended between the collection and its indexes.
    client = Counting(indexed=indexed)
    client.create_collection(
        "answers-aaaaaaaa-0123456789abcdef",
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    client.calls.clear()

    QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef").ensure(3)

    assert client.calls == ["get_collection", *added]


def test_storing_ensures_the_collection_when_nobody_did_before() -> None:
    # A service that started without its store makes the collection with its first answer.
    client = Counting()
    store = QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef")

    store.put(entry("first"), towards(1.0))
    store.put(entry("second"), towards(0.9))

    assert client.calls.count("create_collection") == 1
    assert client.calls.count("get_collection") == 1, "asked once, then remembered"
    assert store.held().answers == 2


def answered_with(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "reason", b"", None)  # type: ignore[arg-type]


class Refusing(QdrantClient):
    """A client whose writes fail with the error a test gives it, and are counted."""

    def __init__(self, error: Exception) -> None:
        super().__init__(":memory:")
        self.error = error
        self.writes = 0

    def upsert(self, *args: Any, **kwargs: Any) -> Any:
        self.writes += 1
        raise self.error


@pytest.mark.parametrize(
    "error",
    [answered_with(400), answered_with(500), ValueError("a point the store will not take")],
    ids=["refused", "the store failed", "a plain error"],
)
def test_a_write_that_failed_is_not_tried_again_unless_the_collection_has_gone(
    error: Exception,
) -> None:
    client = Refusing(error)
    store = QdrantAnswerStore(client, "answers-aaaaaaaa-0123456789abcdef")

    with pytest.raises(type(error)):
        store.put(entry(), towards(1.0))

    # One write: the same point would be refused the same way again.
    assert client.writes == 1
    assert client.collection_exists(store.collection), "the collection is there"


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


def test_only_a_collection_named_as_this_project_names_them_is_one_of_cached_answers() -> None:
    client = QdrantClient(":memory:")
    ours = [
        collection_name("aaaaaaaa", "0123456789abcdef"),
        collection_name("aaaaaaaa", "fedcba9876543210"),
        collection_name("bbbbbbbb", "0123456789abcdef"),
        # A commit named by a word: its first eight characters, as a collection name holds them.
        collection_name("feature", "0123456789abcdef"),
        collection_name("", "0123456789abcdef"),
    ]
    others = [
        "handbook-aaaaaaaa-model",
        # Something else that keeps answers in the same store, under a name that starts alike.
        "answers-of-the-survey",
        "answers-aaaaaaaa-1",
        "answers-aaaaaaaa-0123456789ABCDEF",
        "answers-aaaaaaaa-0123456789abcdef-copy",
        "my-answers-aaaaaaaa-0123456789abcdef",
    ]
    for name in (*ours, *others):
        QdrantAnswerStore(client, name).put(entry(), towards(1.0))

    assert COLLECTION_PREFIX == "answers-"
    assert ours[0] == "answers-aaaaaaaa-0123456789abcdef"
    assert answer_collections(client) == sorted(ours)
    assert answer_collections(client, commit="aaaaaaaa") == sorted(ours[:2])
    assert answer_collections(client, commit="feature") == [ours[3]]
    assert answer_collections(client, commit="cccccccc") == []


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
