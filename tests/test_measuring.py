"""The switch that keeps the answer cache out of an evaluation.

While an evaluation runs, a cache cannot be built and cannot be used, wherever in the
code it is tried and however the call is written. ``tests/test_cli.py`` checks that each
command that measures runs inside the switch, and this file checks the switch.
"""

import ast
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from handbook_rag import pipeline, wiring
from handbook_rag.api import create_app
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.config import Settings
from handbook_rag.measuring import CacheNotAllowed, measuring, measuring_the_model
from handbook_rag.pipeline import answer_question
from tests.cache.support import TableVerifier, memory_answers
from tests.cache.test_answers import Vectors
from tests.support import FakeEmbedder, FakeSparseEncoder
from tests.test_api import Writing, indexed_store
from tests.test_pipeline import LEAVE, FakeGenerator, FakeRetriever

LIVE_TESTS = Path(__file__).parent / "live"


def a_cache() -> AnswerCache:
    return AnswerCache(memory_answers(), embedder=Vectors({}), threshold=0.9)


def test_the_switch_is_off_until_an_evaluation_runs_and_off_again_afterwards() -> None:
    assert not measuring()
    with measuring_the_model():
        assert measuring()
        # One evaluation inside another leaves it on until both have ended.
        with measuring_the_model():
            assert measuring()
        assert measuring()
    assert not measuring()


def test_an_evaluation_that_fails_switches_it_off_too() -> None:
    with pytest.raises(RuntimeError, match="the run broke"), measuring_the_model():
        raise RuntimeError("the run broke")

    assert not measuring()
    assert a_cache()


def test_the_switch_is_for_the_whole_process_and_not_for_one_thread() -> None:
    seen: list[bool] = []
    refused: list[str] = []

    def in_a_thread_of_its_own() -> None:
        seen.append(measuring())
        try:
            a_cache()
        except CacheNotAllowed as error:
            refused.append(str(error))

    with measuring_the_model():
        thread = threading.Thread(target=in_a_thread_of_its_own)
        thread.start()
        thread.join(5.0)

    assert seen == [True]
    assert len(refused) == 1


@pytest.mark.parametrize(
    "build",
    [
        a_cache,
        lambda: AnswerCache(
            memory_answers(), embedder=Vectors({}), verifier=TableVerifier(), threshold=0.9
        ),
        lambda: wiring.make_answer_cache(
            Settings(), store=wiring.make_memory_store("handbook"), embedder=FakeEmbedder()
        ),
        lambda: wiring.make_answer_cache(
            Settings(cache_mode="exact"),
            store=wiring.make_memory_store("handbook"),
            embedder=FakeEmbedder(),
        ),
    ],
    ids=["the exact tier", "both tiers", "from the settings", "from the settings, exact"],
)
def test_no_cache_can_be_built_while_an_evaluation_runs(build: Any) -> None:
    with measuring_the_model(), pytest.raises(CacheNotAllowed, match="cannot be built"):
        build()

    assert build() is not None, "and it can as soon as the evaluation has ended"


def test_what_builds_the_retriever_of_the_command_cannot_build_its_cache_then(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wiring, "make_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
    store = wiring.make_memory_store("handbook")

    with measuring_the_model(), pytest.raises(CacheNotAllowed):
        wiring.make_retriever_and_cache(Settings(), store)

    # With the cache switched off there is nothing to refuse.
    with measuring_the_model():
        _, cache = wiring.make_retriever_and_cache(Settings(cache_mode="off"), store)
    assert cache is None


def ways_to_hand_over(cache: AnswerCache) -> dict[str, Any]:
    """The entry point called with a cache, written in every way a caller might."""
    parts: dict[str, Any] = {
        "retriever": FakeRetriever([]),
        "generator": FakeGenerator(),
        "limit": 1,
    }
    renamed = answer_question
    return {
        "by name": lambda: answer_question("q", cache=cache, **parts),
        "as an attribute of its module": lambda: pipeline.answer_question(
            "q", cache=cache, **parts
        ),
        "under another name": lambda: renamed("q", cache=cache, **parts),
        "among arguments not written out": lambda: answer_question(
            "q", **{**parts, "cache": cache}
        ),
        "with a fresh answer asked for": lambda: answer_question(
            "q", cache=cache, fresh=True, **parts
        ),
    }


@pytest.mark.parametrize("how", sorted(ways_to_hand_over(a_cache())))
def test_a_cache_built_before_cannot_be_handed_to_the_entry_point_while_an_evaluation_runs(
    how: str,
) -> None:
    # Built before the evaluation began, so that only handing it over is left to refuse.
    cache = a_cache()
    call = ways_to_hand_over(cache)[how]

    with measuring_the_model(), pytest.raises(CacheNotAllowed, match="cannot be used"):
        call()

    assert call() is not None, "outside an evaluation the same call answers"


def test_without_a_cache_the_entry_point_answers_while_an_evaluation_runs() -> None:
    from handbook_rag.index.store import ScoredChunk

    with measuring_the_model():
        answer = answer_question(
            "q",
            retriever=FakeRetriever([ScoredChunk(LEAVE, 0.9)]),
            generator=FakeGenerator(),
            limit=1,
        )

    assert (answer.text, answer.cached) == ("An answer.", None)


def test_a_service_built_from_the_settings_cannot_start_with_a_cache_while_one_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A live test starts the real service. It brings its own retriever, so it gets no
    # cache. Built as `serve` builds it, the service would try to, and is stopped.
    monkeypatch.setattr(wiring, "make_embedder", lambda _settings: FakeEmbedder())
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
    store, embedder, sparse_encoder = indexed_store()
    model = Writing(("Sixteen weeks.", (0,)))
    retriever = wiring.make_retriever(
        Settings(), store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )

    with measuring_the_model():
        with (
            pytest.raises(CacheNotAllowed),
            TestClient(create_app(Settings(), store=store, generator=model)),
        ):
            pass
        with TestClient(
            create_app(Settings(), retriever=retriever, store=store, generator=model)
        ) as client:
            assert client.app.state.cache is None  # type: ignore[attr-defined]
            body = client.post("/ask", json={"query": "how many weeks of paid leave"}).json()

    assert (body["cached"], body["requests"]) == (None, 1)


def test_every_live_test_runs_with_the_switch_on() -> None:
    """`make test-live` spends real requests to see what the model does today.

    The live tests are not run here. What is checked is that their directory has a
    fixture that every test in it gets without asking, and that it holds the switch.
    """
    tree = ast.parse((LIVE_TESTS / "conftest.py").read_text(encoding="utf-8"))
    fixtures = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any("autouse=True" in ast.unparse(decorator) for decorator in node.decorator_list)
    ]

    (fixture,) = fixtures
    held = [
        ast.unparse(item.context_expr)
        for node in ast.walk(fixture)
        if isinstance(node, ast.With)
        for item in node.items
    ]
    assert held == ["measuring_the_model()"]
    assert "yield" in ast.unparse(fixture)
    # And there are live tests for it to cover.
    assert sorted(path.name for path in LIVE_TESTS.glob("test_*.py")) == [
        "test_api_live.py",
        "test_gemini_live.py",
    ]
