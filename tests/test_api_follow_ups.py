"""POST /ask with the earlier questions of a conversation (issue 79, ADR 0024).

The service keeps no conversation: the caller sends the earlier questions, questions only.
Every reply says which question was searched for, in its JSON and in a stream's first
event. The rewriting model and the answer model are stand-ins.
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from handbook_rag import wiring
from handbook_rag.api import MAX_BODY_BYTES, MAX_EARLIER_QUESTIONS, MAX_QUERY_CHARS, create_app
from handbook_rag.config import Settings
from handbook_rag.generate.base import GenerationError
from handbook_rag.pipeline import FollowUps
from tests.rewrite.support import StandInRewriter, failing, replying
from tests.test_api import Writing, answer_event, answers_ended, indexed_store

FIRST = "How many weeks of parental leave do new parents get?"
FOLLOW_UP = "and how many for adoption"
STANDALONE = "How many weeks of parental leave do new parents get for adoption?"


def conversing(
    model: Writing, rewriter: StandInRewriter | None, settings: Settings | None = None
) -> FastAPI:
    """The API over the two-chunk index, with a stand-in answer model and rewriting model."""
    settings = settings or Settings()
    store, embedder, sparse_encoder = indexed_store()
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )
    follow_ups = None if rewriter is None else FollowUps(rewriter, earlier_questions=3)
    return create_app(
        settings, retriever=retriever, store=store, generator=model, follow_ups=follow_ups
    )


def events(stream: bytes) -> list[tuple[str, Any]]:
    """The name and data of each event of a stream, in order."""
    found = []
    for block in stream.decode().removesuffix("\n\n").split("\n\n"):
        name, data = block.split("\n", 1)
        found.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return found


def test_a_follow_up_is_answered_for_the_question_it_was_rewritten_to() -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": [FIRST]})

    assert response.status_code == 200
    searched = response.json()["searched_for"]
    assert searched == {
        "asked": FOLLOW_UP,
        "question": STANDALONE,
        "outcome": "rewritten",
        "failure": None,
        "earlier": 1,
        "requests": 1,
        "usage": {"input_tokens": 420, "output_tokens": 18},
        "took_ms": 750.0,
    }
    ((asked, _chunks),) = model.asked
    assert asked == STANDALONE
    # The answer's count is the answer model's; the step's request is counted apart.
    assert response.json()["requests"] == 1
    assert [turn.question for turn in rewriter.calls[0][0]] == [FIRST]


def test_a_stream_names_the_question_searched_for_before_its_first_text() -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post(
            "/ask", json={"query": FOLLOW_UP, "earlier": [FIRST], "stream": True}
        )

    found = events(response.content)
    assert [name for name, _ in found] == ["searched", "text", "answer"]
    assert found[0][1]["question"] == STANDALONE
    assert found[0][1] == json.loads(answer_event(response.content))["searched_for"]


def test_a_failed_step_answers_the_question_as_typed_and_the_reply_says_so() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    rewriter = failing(GenerationError("the model did not answer in time", reason="timeout"))
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": [FIRST]})

    assert response.status_code == 200
    searched = response.json()["searched_for"]
    assert (searched["question"], searched["outcome"], searched["failure"]) == (
        FOLLOW_UP,
        "failed",
        "timeout",
    )
    assert searched["requests"] == 1
    assert "in time" not in response.text, "the error's own words stay in the log"
    ((asked, _chunks),) = model.asked
    assert asked == FOLLOW_UP
    assert len(rewriter.calls) == 1, "one attempt, and no other"


def test_an_unusable_reply_answers_the_question_as_typed_without_quoting_the_reply() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    rewriter = replying("Rewritten question: adoption leave in 2031.")
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": [FIRST]})

    searched = response.json()["searched_for"]
    assert (searched["question"], searched["outcome"]) == (FOLLOW_UP, "unusable")
    assert "2031" not in response.text


def test_without_earlier_questions_the_rewriting_model_is_not_asked() -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        plain = client.post("/ask", json={"query": FIRST})
        empty = client.post("/ask", json={"query": FIRST, "earlier": []})

    assert rewriter.calls == []
    for response in (plain, empty):
        assert response.json()["searched_for"]["outcome"] == "not_asked"


def test_a_service_without_the_step_ignores_earlier_questions_and_says_it_did_not_run() -> None:
    model = Writing(("Sixteen weeks.", (0,)))
    with TestClient(conversing(model, None)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": [FIRST]})

    searched = response.json()["searched_for"]
    assert (searched["question"], searched["outcome"], searched["earlier"]) == (
        FOLLOW_UP,
        "not_asked",
        0,
    )


def test_more_earlier_questions_may_be_sent_and_the_latest_are_read() -> None:
    rewriter = replying(STANDALONE)
    earlier = [f"Question number {number}?" for number in range(1, MAX_EARLIER_QUESTIONS + 1)]
    with TestClient(conversing(Writing(("Sixteen weeks.", (0,))), rewriter)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": earlier})

    assert response.json()["searched_for"]["earlier"] == 3
    assert [turn.question for turn in rewriter.calls[0][0]] == earlier[-3:]


@pytest.mark.parametrize(
    ("earlier", "where"),
    [
        # An answer is refused in every shape: an object with one, or anything not a string.
        ([{"question": FIRST, "answer": "SECRET-ANSWER"}], "earlier"),
        ([[FIRST, "SECRET-ANSWER"]], "earlier"),
        ([7], "earlier"),
        ([""], "earlier"),
        (["   "], "earlier"),
        (["x" * (MAX_QUERY_CHARS + 1)], "earlier"),
        ([FIRST] * (MAX_EARLIER_QUESTIONS + 1), "earlier"),
        (FIRST, "earlier"),
    ],
)
def test_earlier_questions_that_are_not_questions_are_refused_before_anything_is_asked(
    earlier: object, where: str
) -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post("/ask", json={"query": FOLLOW_UP, "earlier": earlier})

    assert response.status_code == 422
    assert where in [str(part) for error in response.json()["detail"] for part in error["loc"]]
    assert "SECRET-ANSWER" not in response.text, "the refusal does not echo what was sent"
    assert (rewriter.calls, model.asked) == ([], [])


@pytest.mark.parametrize("field", ["answer", "answers", "turns", "history"])
def test_a_request_that_carries_answers_beside_the_question_is_refused(field: str) -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        response = client.post(
            "/ask", json={"query": FOLLOW_UP, "earlier": [FIRST], field: "SECRET-ANSWER"}
        )

    assert response.status_code == 422
    assert [error["loc"][-1] for error in response.json()["detail"]] == [field]
    assert (rewriter.calls, model.asked) == ([], [])


def test_every_earlier_question_at_its_longest_fits_in_a_body_the_service_reads() -> None:
    # A character of four bytes, as some scripts have, in every question at its longest.
    longest = "\N{GRINNING FACE}" * MAX_QUERY_CHARS
    body = {"query": longest, "earlier": [longest] * MAX_EARLIER_QUESTIONS}

    assert len(json.dumps(body, ensure_ascii=False).encode()) < MAX_BODY_BYTES
    rewriter = replying(longest)
    with TestClient(conversing(Writing(("Sixteen weeks.", (0,))), rewriter)) as client:
        response = client.post(
            "/ask",
            content=json.dumps(body, ensure_ascii=False).encode(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200


def test_the_settings_build_the_step_with_its_own_time_limit_or_none_when_it_is_off() -> None:
    on = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"), cache_mode="off")
    off = on.model_copy(update={"follow_ups": False})

    built_on = wiring.make_follow_ups(on)
    built_off = wiring.make_follow_ups(off)

    assert built_on is not None
    assert built_on.earlier_questions == on.follow_up_earlier_questions
    client = built_on.rewriter._client._api_client  # type: ignore[attr-defined]
    assert client._http_options.timeout == on.follow_up_timeout_seconds * 1000
    assert client._http_options.retry_options.attempts == 1
    assert built_off is None


def test_the_streams_and_the_answers_threads_end_after_follow_ups() -> None:
    model, rewriter = Writing(("Sixteen weeks.", (0,))), replying(STANDALONE)
    with TestClient(conversing(model, rewriter)) as client:
        client.post("/ask", json={"query": FOLLOW_UP, "earlier": [FIRST], "stream": True})
        assert answers_ended()
