"""The Gemini rewriter, tested against a stand-in client that returns real SDK objects.

No request is sent. Every failure is also put to the answer generator, whose reading of
errors the rewriter shares, and both must say the same.
"""

from typing import Any, cast

import httpx
import httpx2
import pydantic
import pytest
from google import genai
from google.genai import errors, types

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.rewrite.base import standalone_question
from handbook_rag.rewrite.gemini import GeminiRewriter
from handbook_rag.rewrite.models import MAX_REPLY_TOKENS, Turn
from handbook_rag.rewrite.prompt import REWRITE_INSTRUCTIONS, build_rewrite_prompt
from tests.generate.test_gemini import (
    KEY_INVALID,
    LEAVE,
    MODEL,
    PER_DAY,
    PER_MINUTE,
    RETRY_IN_36S,
    api_error,
    quota,
    wrong_shape,
)
from tests.generate.test_gemini import FakeClient as StreamingClient

UK = Turn("How many days of paid annual leave do UK team members get?", answer="28 days.")


def response(
    text: str | None = "How many days of paid annual leave do team members in Finland get?",
    *,
    finish: str | None = "STOP",
    thought: str | None = None,
    blocked: bool = False,
    model_version: str | None = "gemini-3.5-flash-lite-001",
) -> types.GenerateContentResponse:
    """A whole response, as the SDK hands over one that was not streamed."""
    parts: list[dict[str, Any]] = []
    if thought is not None:
        parts.append({"text": thought, "thought": True})
    if text is not None:
        parts.append({"text": text})
    body: dict[str, Any] = {
        "usage_metadata": {
            "prompt_token_count": 430,
            "candidates_token_count": 17,
            "thoughts_token_count": 5,
            "cached_content_token_count": 30,
        },
        "model_version": model_version,
    }
    if blocked:
        body["prompt_feedback"] = {"block_reason": "SAFETY"}
    else:
        body["candidates"] = [
            {"content": {"role": "model", "parts": parts}, "finish_reason": finish}
        ]
    return types.GenerateContentResponse.model_validate(body)


class FakeModels:
    def __init__(
        self, outcome: types.GenerateContentResponse | BaseException | None = None
    ) -> None:
        self._outcome = outcome if outcome is not None else response()
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class FakeClient:
    def __init__(
        self, outcome: types.GenerateContentResponse | BaseException | None = None
    ) -> None:
        self.models = FakeModels(outcome)


def rewriter_for(client: FakeClient, **options: Any) -> GeminiRewriter:
    return GeminiRewriter(cast(genai.Client, client), model=MODEL, **options)


def test_one_request_carries_the_instructions_the_layout_and_the_limit() -> None:
    client = FakeClient()

    rewriter_for(client).rewrite([UK], "And in Finland?")

    (call,) = client.models.calls
    assert call["model"] == MODEL
    assert call["contents"] == build_rewrite_prompt([UK], "And in Finland?")
    config = call["config"]
    assert config.system_instruction == REWRITE_INSTRUCTIONS
    assert config.max_output_tokens == MAX_REPLY_TOKENS
    # No request of the SDK's own, which nobody would count.
    assert config.automatic_continuation is False
    assert "28 days" not in call["contents"]


def test_the_reply_is_read_as_an_answers_reply_is() -> None:
    client = FakeClient(response("  How many days in Finland? ", thought="Let me think."))

    reply = rewriter_for(client).rewrite([UK], "And in Finland?")

    assert reply.text == "  How many days in Finland? "
    assert reply.model == "gemini-3.5-flash-lite-001"
    assert reply.stop_reason == "end_turn"
    # Cached tokens are kept apart, and reasoning counts as output.
    assert reply.usage.input_tokens == 400
    assert reply.usage.cache_read_input_tokens == 30
    assert reply.usage.output_tokens == 22
    assert reply.seconds >= 0


def test_the_reply_carries_the_wait_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([100.0, 102.5])
    monkeypatch.setattr("handbook_rag.rewrite.gemini.time.perf_counter", lambda: next(clock))

    reply = rewriter_for(FakeClient()).rewrite([UK], "And in Finland?")

    assert reply.seconds == 2.5


@pytest.mark.parametrize(
    ("outcome", "stop_reason"),
    [
        (response(finish="MAX_TOKENS"), "max_tokens"),
        (response(finish="SAFETY"), "refusal"),
        (response(finish="RECITATION"), "refusal"),
        (response(blocked=True), "refusal"),
        (response(finish=None), "unknown"),
    ],
)
def test_a_reply_cut_off_or_withheld_says_so(
    outcome: types.GenerateContentResponse, stop_reason: str
) -> None:
    reply = rewriter_for(FakeClient(outcome)).rewrite([UK], "And in Finland?")

    assert reply.stop_reason == stop_reason
    # The step does not use it, and searches for the question as it was asked.
    rewrite = standalone_question(
        "And in Finland?", [UK], rewriter=rewriter_for(FakeClient(outcome))
    )
    assert rewrite.outcome == "unusable"
    assert rewrite.question == "And in Finland?"


def test_a_reply_without_a_model_name_names_the_model_asked() -> None:
    reply = rewriter_for(FakeClient(response(model_version=None))).rewrite([UK], "Why?")

    assert reply.model == MODEL


FAILURES: list[BaseException] = [
    api_error(400, "INVALID_ARGUMENT", KEY_INVALID),
    api_error(401, "UNAUTHENTICATED"),
    api_error(403, "PERMISSION_DENIED", message="Not enabled for this project."),
    api_error(404, "NOT_FOUND", message="No such model."),
    api_error(429, "RESOURCE_EXHAUSTED", [quota(PER_MINUTE, "15"), RETRY_IN_36S]),
    api_error(429, "RESOURCE_EXHAUSTED", [quota(PER_DAY, "500"), RETRY_IN_36S]),
    api_error(429, "RESOURCE_EXHAUSTED"),
    api_error(400, "INVALID_ARGUMENT", message="Request contains an invalid argument."),
    api_error(409, "ABORTED", message="The operation was aborted."),
    api_error(408, "DEADLINE_EXCEEDED", message="Request timeout."),
    api_error(503, "UNAVAILABLE", message="The model is overloaded."),
    api_error(504, "DEADLINE_EXCEEDED", message="Deadline expired before operation."),
    httpx2.ConnectError("Connection refused"),
    httpx2.ReadTimeout("timed out"),
    httpx.ReadTimeout("timed out"),
    httpx2.RemoteProtocolError("peer closed"),
    httpx.DecodingError("bad gzip data"),
    errors.UnknownApiResponseError("Failed to parse response as JSON."),
    wrong_shape(),
]


@pytest.mark.parametrize("error", FAILURES, ids=lambda error: type(error).__name__)
def test_every_failure_is_read_as_the_answer_generator_reads_it(error: BaseException) -> None:
    generator = GeminiGenerator(
        cast(genai.Client, StreamingClient(error=error)), model=MODEL, max_tokens=4000
    )

    with pytest.raises(GenerationError) as answering:
        generator.generate("q", [LEAVE])
    with pytest.raises(GenerationError) as rewriting:
        rewriter_for(FakeClient(error)).rewrite([UK], "And in Finland?")

    assert str(rewriting.value) == str(answering.value)
    assert rewriting.value.reason == answering.value.reason
    assert rewriting.value.retry_after == answering.value.retry_after
    assert rewriting.value.__cause__ is error


def test_a_mistake_in_our_own_settings_is_not_reported_as_an_unreadable_reply() -> None:
    client = FakeClient()

    with pytest.raises(pydantic.ValidationError):
        rewriter_for(client, max_tokens=cast(int, "many")).rewrite([UK], "Why?")
    assert client.models.calls == []


def test_an_unrelated_error_is_not_swallowed() -> None:
    with pytest.raises(TypeError, match="unsupported operand"):
        rewriter_for(FakeClient(TypeError("unsupported operand"))).rewrite([UK], "Why?")


def test_the_rewriter_goes_by_the_answer_generators_names() -> None:
    rewriter = rewriter_for(FakeClient())

    assert rewriter.provider == GeminiGenerator.provider == "gcp.gemini"
    assert rewriter.operation == GeminiGenerator.operation
    assert rewriter.model == MODEL
