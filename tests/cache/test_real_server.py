"""The answer cache against a real Qdrant server.

Every other test of the cache uses Qdrant's embedded mode. A server differs from it in
ways that matter here. It takes its time to make a collection. It uses the payload
indexes, which embedded mode accepts and ignores. And it answers a question about a
collection that is not there with another error. The first version of the cache went
wrong exactly there, where no fast test could see it: the first write made the
collection, in the background, while questions were already being looked up.

These tests need the server, so they are ``slow``, and they are skipped when none
answers at the address in the settings (``make qdrant`` starts one). Each works in
collections of its own, under a prefix that no collection of the application has, and
removes them at the end. No model is loaded and nothing goes to a model API.
"""

import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from handbook_rag.cache import rules
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.store import Held, QdrantAnswerStore, answer_collections
from handbook_rag.config import Settings
from tests.cache.support import TableVerifier
from tests.cache.test_answers import (
    DECLINE,
    LEAVE,
    NEAR_MISS,
    REWORDED,
    STORED,
    Vectors,
    keep,
    written,
)
from tests.evals.test_cache_pairs import towards

pytestmark = pytest.mark.slow

# No collection of the application is named like this: the handbook's start with
# "handbook-" and the cache's with "answers-".
PREFIX = "test-of-the-answer-cache-"
# A lookup may take this long here. The tests are about the store, and the machine they
# run on may be busy: the time limit of a lookup has tests of its own.
PATIENT = 30.0
VECTORS = {STORED: [1.0, 0.0, 0.0], REWORDED: towards(0.9), NEAR_MISS: towards(0.8)}
MEANS_THE_SAME = {(STORED, REWORDED): (0.95, 0.9)}


@pytest.fixture
def server() -> Iterator[QdrantClient]:
    """The Qdrant server of the settings, or a skipped test when none answers."""
    address = Settings().qdrant_url
    client = QdrantClient(url=address, timeout=60)
    try:
        client.get_collections()
    except Exception as error:
        client.close()
        pytest.skip(f"no Qdrant server answers at {address}: {type(error).__name__}")
    yield client
    client.close()


@pytest.fixture
def collection(server: QdrantClient) -> Iterator[Callable[[], str]]:
    """Names for collections of this test's own, each removed when the test has ended."""
    made: list[str] = []

    def name() -> str:
        made.append(f"{PREFIX}{uuid.uuid4().hex[:12]}")
        return made[-1]

    yield name
    for each in made:
        if server.collection_exists(each):
            server.delete_collection(each)
    left = [item.name for item in server.get_collections().collections if item.name in made]
    assert not left, "the test leaves the server as it found it"


def cache_on(store: QdrantAnswerStore, **options: Any) -> AnswerCache:
    """A cache as a service builds it: both tiers, and answers stored in the background."""
    return AnswerCache(
        store,
        embedder=Vectors(VECTORS),
        verifier=TableVerifier(MEANS_THE_SAME),
        threshold=0.8,
        background=True,
        lookup_seconds=PATIENT,
        **options,
    )


def indexes(server: QdrantClient, name: str) -> dict[str, str]:
    """The payload fields a collection has an index for, with the kind of each."""
    schema = server.get_collection(name).payload_schema or {}
    return {field: str(index.data_type.value) for field, index in schema.items()}


def timed[T](what: str, call: Callable[[], T]) -> T:
    """Run one call and print how long it took, for whoever runs the test with ``-s``."""
    started = time.perf_counter()
    result = call()
    print(f"  {what}: {(time.perf_counter() - started) * 1000:.0f} ms")
    return result


def test_a_cache_made_ready_at_the_start_serves_the_first_answer_it_stored(
    server: QdrantClient, collection: Callable[[], str]
) -> None:
    name = collection()
    store = QdrantAnswerStore(server, name)
    cache = cache_on(store)

    assert timed("making the collection and its indexes", cache.prepare)

    # What a service has before its first question: the collection, made for the
    # embedding model's vectors, and both indexes the tier by meaning filters on.
    described = server.get_collection(name)
    vectors = described.config.params.vectors
    assert isinstance(vectors, models.VectorParams)
    assert (vectors.size, vectors.distance) == (3, models.Distance.COSINE)
    assert indexes(server, name) == {"scope": "keyword", "by_meaning": "bool"}
    # The first question finds nothing, and nothing fails.
    first = timed("the first lookup, of an empty cache", lambda: cache.lookup(STORED))
    assert (first.answer, first.checked, first.failure) == (None, (), None)
    cache.store(first, written())
    # Asked again at once, while the answer may still be on its way to the store.
    meanwhile = cache.lookup(STORED)
    assert meanwhile.failure is None
    assert cache.close(PATIENT), "the answer was stored in time"

    same = timed("a lookup served by the exact tier", lambda: cache.lookup(STORED))
    assert (same.tier, same.failure) == ("exact", None)
    assert same.answer is not None
    assert (same.answer.text, same.answer.sources) == ("Sixteen weeks.", (LEAVE,))
    reworded = timed("a lookup served by meaning", lambda: cache.lookup(REWORDED))
    assert (reworded.tier, reworded.failure) == ("semantic", None)
    assert reworded.answer is not None
    assert reworded.answer.cached is not None
    assert reworded.answer.cached.question == STORED
    assert reworded.answer.cached.similarity == pytest.approx(0.9, abs=1e-4)
    assert reworded.answer.cached.score == pytest.approx(0.9)
    assert reworded.answer.sources == (LEAVE,)
    # The server filters by the part of the handbook: asked of one section, the answer
    # written from all of it is no candidate.
    of_a_section = cache.lookup(REWORDED, section="people")
    assert (of_a_section.answer, of_a_section.checked, of_a_section.failure) == (None, (), None)
    # And by what may be served by meaning: a stored decline is nearer and is passed over.
    keep(cache, NEAR_MISS, DECLINE)
    assert store.held() == Held(answers=2, declines=1, by_meaning=1)
    near = cache.lookup("a question as near to the decline as to the answer")
    assert near.failure is None
    asked_again = cache.lookup(NEAR_MISS)
    assert (asked_again.tier, asked_again.answer and asked_again.answer.answered) == (
        "exact",
        False,
    )


def test_without_that_the_first_write_makes_the_collection_and_the_answer_is_still_kept(
    server: QdrantClient, collection: Callable[[], str]
) -> None:
    # A service that started while its store was down has made nothing.
    name = collection()
    cache = cache_on(QdrantAnswerStore(server, name))
    assert not server.collection_exists(name)

    # A collection that is not there holds nothing, which is no failure.
    first = cache.lookup(STORED)
    assert (first.answer, first.failure) == (None, None)
    started = time.perf_counter()
    cache.store(first, written())
    # Questions go on being asked while the first write makes the collection. None of
    # them raises. What each finds in those moments is not promised.
    seen = []
    while cache.lookup(STORED).answer is None:
        seen.append(cache.lookup(REWORDED).failure)
        assert time.perf_counter() - started < PATIENT, "the answer never arrived"
        time.sleep(0.05)
    print(
        f"  the first write made the collection: the answer was there after "
        f"{(time.perf_counter() - started) * 1000:.0f} ms, {len(seen)} lookups missed "
        f"meanwhile, failures among them: {sorted({kind for kind in seen if kind})}"
    )

    assert cache.close(PATIENT)
    assert indexes(server, name) == {"scope": "keyword", "by_meaning": "bool"}
    assert cache.lookup(STORED).tier == "exact"
    assert cache.lookup(REWORDED).tier == "semantic"


def test_emptying_keeps_the_collection_and_removing_takes_it_away(
    server: QdrantClient, collection: Callable[[], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    name = collection()
    store = QdrantAnswerStore(server, name)
    cache = cache_on(store)
    assert cache.prepare()
    keep(cache, STORED, written())
    keep(cache, NEAR_MISS, DECLINE)
    assert cache.close(PATIENT)
    assert store.held() == Held(answers=2, declines=1, by_meaning=1)

    assert timed("emptying a collection of two answers", store.clear) == 2

    # The collection is still there with both indexes, and holds nothing.
    assert indexes(server, name) == {"scope": "keyword", "by_meaning": "bool"}
    assert store.held() == Held(0, 0, 0)
    assert store.clear() == 0
    # Another process on the same collection, as a service beside the command that
    # emptied it. It asks once whether the collection is there and makes nothing.
    asked: list[str] = []
    get_collection = server.get_collection

    def counted(*arguments: Any, **options: Any) -> Any:
        asked.append("get_collection")
        return get_collection(*arguments, **options)

    def never(*_arguments: Any, **_options: Any) -> None:
        raise AssertionError("everything is there already: nothing has to be made")

    monkeypatch.setattr(server, "get_collection", counted)
    monkeypatch.setattr(server, "create_collection", never)
    monkeypatch.setattr(server, "create_payload_index", never)
    beside = AnswerCache(
        QdrantAnswerStore(server, name),
        embedder=Vectors(VECTORS),
        threshold=0.8,
        lookup_seconds=PATIENT,
    )
    assert beside.prepare()
    assert asked == ["get_collection"]
    keep(beside, STORED, written())
    assert asked == ["get_collection"], "storing asks nothing more"
    assert beside.lookup(STORED).tier == "exact"
    monkeypatch.undo()

    assert timed("removing the collection", store.remove) == 1

    assert not server.collection_exists(name)
    # Read after that, it holds nothing, and no call fails.
    key = rules.entry_id(rules.comparable(STORED), rules.scope_of(None))
    assert store.get(key) is None
    assert store.nearest([1.0, 0.0, 0.0], scope=rules.scope_of(None), limit=3, floor=0.5) == []
    assert store.held() == Held(0, 0, 0)
    assert (store.clear(), store.remove()) == (0, 0)
    assert not server.collection_exists(name), "reading makes nothing"


def test_a_collection_left_half_made_gets_the_index_it_lacks(
    server: QdrantClient, collection: Callable[[], str]
) -> None:
    # The process that was making it ended after the first of its two indexes.
    name = collection()
    server.create_collection(
        name, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    server.create_payload_index(
        name, field_name="scope", field_schema=models.PayloadSchemaType.KEYWORD
    )
    assert indexes(server, name) == {"scope": "keyword"}

    QdrantAnswerStore(server, name).ensure(3)

    assert indexes(server, name) == {"scope": "keyword", "by_meaning": "bool"}


def test_the_collections_of_these_tests_are_not_taken_for_the_applications(
    server: QdrantClient, collection: Callable[[], str]
) -> None:
    # `cache-clear --all` and `index` remove or empty what this lists, and nothing else.
    name = collection()
    QdrantAnswerStore(server, name).ensure(3)
    every = {item.name for item in server.get_collections().collections}
    assert name in every

    listed = answer_collections(server)

    assert name not in listed
    assert not [each for each in listed if not each.startswith("answers-")]
    assert set(listed) <= every
