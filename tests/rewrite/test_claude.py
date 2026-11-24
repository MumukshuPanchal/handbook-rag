"""The Claude rewriter, tested against a stand-in client that returns real SDK objects.

This path has never been run for real. The rewriter writes out the answer generator's
reading of failures again, so every failure is put to both, and both must say the same.
"""

from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.rewrite.claude import ClaudeRewriter
from handbook_rag.rewrite.models import MAX_REPLY_TOKENS, Turn
from handbook_rag.rewrite.prompt import REWRITE_INSTRUCTIONS, build_rewrite_prompt
from tests.generate.test_claude import LEAVE, REQUEST, message, rate_limited, status_error
from tests.generate.test_claude import FakeClient as StreamingClient

UK = Turn("How many days of paid annual leave do UK team members get?", answer="28 days.")
MODEL = "claude-haiku-5-5"


class FakeMessages:
    def __init__(self, outcome: Message | BaseException) -> None:
        self._outcome = outcome
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Message:
        self.requests.append(request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class FakeClient:
    def __init__(self, outcome: Message | BaseException) -> None:
        self.messages = FakeMessages(outcome)


def rewriter_for(outcome: Message | BaseException) -> tuple[ClaudeRewriter, FakeMessages]:
    client = FakeClient(outcome)
    rewriter = ClaudeRewriter(cast(anthropic.Anthropic, client), model=MODEL, effort="low")
    return rewriter, client.messages


REWRITTEN = message(
    [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "How many days of paid annual leave do team members in"},
        {"type": "text", "text": " Finland get?"},
    ],
    usage={
        "input_tokens": 400,
        "output_tokens": 20,
        "cache_read_input_tokens": 12,
        "cache_creation_input_tokens": 3,
    },
)


def test_one_request_that_is_not_streamed_carries_the_instructions_and_the_layout() -> None:
    rewriter, messages = rewriter_for(REWRITTEN)

    rewriter.rewrite([UK], "And in Finland?")

    (request,) = messages.requests
    assert request["model"] == MODEL
    assert request["max_tokens"] == MAX_REPLY_TOKENS
    assert request["system"] == REWRITE_INSTRUCTIONS
    assert request["messages"] == [
        {"role": "user", "content": build_rewrite_prompt([UK], "And in Finland?")}
    ]
    assert request["output_config"] == {"effort": "low"}
    assert "28 days" not in request["messages"][0]["content"]


def test_the_reply_is_its_text_blocks_without_the_models_reasoning() -> None:
    rewriter, _ = rewriter_for(REWRITTEN)

    reply = rewriter.rewrite([UK], "And in Finland?")

    assert reply.text == "How many days of paid annual leave do team members in Finland get?"
    assert reply.model == "claude-haiku-5-5"
    assert reply.stop_reason == "end_turn"
    assert reply.usage.input_tokens == 400
    assert reply.usage.output_tokens == 20
    assert reply.usage.cache_read_input_tokens == 12
    assert reply.usage.cache_creation_input_tokens == 3
    assert reply.seconds >= 0


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "pause_turn"])
def test_a_reply_that_did_not_come_to_its_end_says_why(stop_reason: str) -> None:
    rewriter, _ = rewriter_for(message([{"type": "text", "text": "How"}], stop_reason=stop_reason))

    assert rewriter.rewrite([UK], "And in Finland?").stop_reason == stop_reason


FAILURES: list[BaseException] = [
    anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=REQUEST), body=None
    ),
    anthropic.PermissionDeniedError(
        "not allowed", response=httpx2.Response(403, request=REQUEST), body=None
    ),
    anthropic.NotFoundError(
        "model not found", response=httpx2.Response(404, request=REQUEST), body=None
    ),
    rate_limited(),
    rate_limited({"retry-after": "20"}),
    rate_limited({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
    anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx2.Response(400, request=REQUEST),
        body=None,
    ),
    status_error(413, "request_too_large"),
    status_error(499, "client_closed_request"),
    status_error(408, "request_timeout"),
    status_error(409, "conflict"),
    status_error(529, "overloaded_error"),
    anthropic.APITimeoutError(request=REQUEST),
    anthropic.APIConnectionError(request=REQUEST),
    httpx2.ReadTimeout("timed out"),
    httpx2.RemoteProtocolError("peer closed connection"),
    TypeError('"Could not resolve authentication method. Expected one of api_key..."'),
]


@pytest.mark.parametrize("error", FAILURES, ids=lambda error: type(error).__name__)
def test_every_failure_is_read_as_the_answer_generator_reads_it(error: BaseException) -> None:
    generator = ClaudeGenerator(
        cast(anthropic.Anthropic, StreamingClient(error)),
        model=MODEL,
        max_tokens=4000,
        effort="low",
    )
    rewriter, _ = rewriter_for(error)

    with pytest.raises(GenerationError) as answering:
        generator.generate("q", [LEAVE])
    with pytest.raises(GenerationError) as rewriting:
        rewriter.rewrite([UK], "And in Finland?")

    assert str(rewriting.value) == str(answering.value)
    assert rewriting.value.reason == answering.value.reason
    assert rewriting.value.retry_after == answering.value.retry_after
    assert rewriting.value.__cause__ is error


def test_an_unrelated_type_error_is_not_swallowed() -> None:
    rewriter, _ = rewriter_for(TypeError("a real bug"))

    with pytest.raises(TypeError, match="a real bug"):
        rewriter.rewrite([UK], "Why?")


def test_the_rewriter_goes_by_the_answer_generators_names() -> None:
    rewriter, _ = rewriter_for(REWRITTEN)

    assert rewriter.provider == ClaudeGenerator.provider == "anthropic"
    assert rewriter.operation == ClaudeGenerator.operation
    assert rewriter.model == MODEL
