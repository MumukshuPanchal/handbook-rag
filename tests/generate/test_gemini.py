"""The Gemini generator is tested against a stand-in client that returns real SDK objects.

No request is sent. Responses are built with the SDK's own response type, so these tests
fail if our code and the SDK disagree about the shape of a response. The one test that
does send a request is in ``tests/live``.
"""

from collections.abc import Iterator
from typing import Any, cast

import httpx
import httpx2
import pytest
from google import genai
from google.genai import errors, types

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
MODEL = "gemini-3.5-flash-lite"
SERVICE = "generativelanguage.googleapis.com"


def piece(
    text: str | None = None,
    *,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
    thought: str | None = None,
    model_version: str | None = MODEL,
) -> types.GenerateContentResponse:
    """One piece of a streamed response, as the SDK hands it over."""
    parts: list[dict[str, Any]] = []
    if thought is not None:
        parts.append({"text": thought, "thought": True})
    if text is not None:
        parts.append({"text": text})
    return types.GenerateContentResponse.model_validate(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
            "usage_metadata": usage,
            "model_version": model_version,
        }
    )


def answer_pieces(*texts: str, finish: str = "STOP") -> list[types.GenerateContentResponse]:
    """A stream that writes the texts, then ends the way a real one does."""
    last = piece(
        "", finish=finish, usage={"prompt_token_count": 1200, "candidates_token_count": 80}
    )
    return [*(piece(text) for text in texts), last]


class FakeModels:
    def __init__(
        self,
        pieces: list[types.GenerateContentResponse] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._pieces = pieces or []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any) -> Iterator[types.GenerateContentResponse]:
        # Like the SDK's method, this sends nothing until the first piece is asked for,
        # and an error ends the stream after the pieces that did arrive.
        self.calls.append(kwargs)
        yield from self._pieces
        if self._error is not None:
            raise self._error


class FakeClient:
    def __init__(
        self,
        pieces: list[types.GenerateContentResponse] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.models = FakeModels(pieces, error=error)


def generator_for(client: FakeClient) -> GeminiGenerator:
    return GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000)


def api_error(code: int, status: str, details: object = None, message: str = "It failed.") -> Any:
    """An error as the SDK raises it, from the JSON body the API sent."""
    body: dict[str, Any] = {"error": {"code": code, "message": message, "status": status}}
    if details is not None:
        body["error"]["details"] = details
    kind = errors.ServerError if code >= 500 else errors.ClientError
    return kind(code, body)


def quota(quota_id: str, value: str) -> dict[str, Any]:
    return {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
            {
                "quotaMetric": f"{SERVICE}/generate_content_free_tier_requests",
                "quotaId": quota_id,
                "quotaDimensions": {"location": "global", "model": MODEL},
                "quotaValue": value,
            }
        ],
    }


RETRY_IN_36S = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"}
# The body the API sent for a wrong key on 2026-11-22.
KEY_INVALID = [
    {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID",
        "domain": "googleapis.com",
        "metadata": {"service": SERVICE},
    }
]


def test_answer_text_and_citations_map_back_to_the_right_chunks() -> None:
    # The second marker is cut in two, as a real stream may cut it.
    client = FakeClient(
        answer_pieces(
            "New parents get sixteen weeks of paid leave [1]. A laptop can be expensed up to ",
            "one thousand dollars [",
            "2].",
        )
    )

    answer = generator_for(client).generate("leave and laptops?", [LEAVE, EXPENSES])

    assert answer.text == (
        "New parents get sixteen weeks of paid leave. "
        "A laptop can be expensed up to one thousand dollars."
    )
    assert answer.answered
    assert answer.sources == (LEAVE, EXPENSES)
    assert [c.chunk for c in answer.segments[0].citations] == [LEAVE]
    assert [c.chunk for c in answer.segments[1].citations] == [EXPENSES]
    assert answer.text_with_markers == (
        "New parents get sixteen weeks of paid leave.[1] "
        "A laptop can be expensed up to one thousand dollars.[2]"
    )


def test_request_carries_the_instructions_model_limit_and_numbered_passages() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))

    generator_for(client).generate("How long is parental leave?", [LEAVE, EXPENSES])

    (call,) = client.models.calls
    assert call["model"] == MODEL
    assert call["config"].system_instruction == MARKER_SYSTEM_PROMPT
    assert call["config"].max_output_tokens == 4000
    prompt = call["contents"]
    assert '<search_result number="1">' in prompt
    assert '<search_result number="2">' in prompt
    assert prompt.index(LEAVE.text) < prompt.index(EXPENSES.text)
    assert prompt.endswith("Question: How long is parental leave?")


def test_text_is_passed_on_as_it_arrives_without_the_markers() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1", "]. One thousand", " dollars [2]."))
    received: list[str] = []

    answer = generator_for(client).generate("q", [LEAVE, EXPENSES], on_text=received.append)

    assert len(received) > 1  # in pieces, not all at the end
    assert "".join(received) == answer.text == "Sixteen weeks. One thousand dollars."
    assert all(text for text in received)


def test_usage_model_and_stop_reason_come_from_the_response() -> None:
    usage = {
        "prompt_token_count": 1200,
        "cached_content_token_count": 200,
        "candidates_token_count": 80,
        "thoughts_token_count": 40,
    }
    client = FakeClient(
        [
            piece("Sixteen weeks [1].", model_version="gemini-3.5-flash-lite-001"),
            piece("", finish="STOP", usage=usage, model_version="gemini-3.5-flash-lite-001"),
        ]
    )

    answer = generator_for(client).generate("q", [LEAVE])

    assert answer.model == "gemini-3.5-flash-lite-001"
    assert answer.stop_reason == "end_turn"
    # Cached tokens are counted apart from the rest of the prompt, and the tokens the
    # model spent reasoning count as output: both are how the request is billed.
    assert answer.usage.input_tokens == 1000
    assert answer.usage.cache_read_input_tokens == 200
    assert answer.usage.output_tokens == 120
    assert answer.seconds > 0
    assert answer.seconds_to_first_text is not None
    assert 0 <= answer.seconds_to_first_text <= answer.seconds


def test_a_response_without_token_counts_or_model_name_still_gives_an_answer() -> None:
    client = FakeClient([piece("Sixteen weeks [1].", finish="STOP", model_version=None)])

    answer = generator_for(client).generate("q", [LEAVE])

    assert answer.model == MODEL
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (0, 0)


def test_an_answer_cut_off_at_the_token_limit_says_so() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]. And then", finish="MAX_TOKENS"))

    answer = generator_for(client).generate("q", [LEAVE])

    assert answer.stop_reason == "max_tokens"  # the command line looks for this word
    assert answer.answered


def test_a_stream_that_ends_without_a_stop_reason_is_reported_as_unknown() -> None:
    client = FakeClient([piece("Sixteen weeks [1].")])

    assert generator_for(client).generate("q", [LEAVE]).stop_reason == "unknown"


def test_the_models_own_reasoning_is_not_part_of_the_answer() -> None:
    client = FakeClient(
        [
            piece("Sixteen ", thought="The user asks about leave. Passage 1 covers it."),
            piece("weeks [1].", finish="STOP"),
        ]
    )
    received: list[str] = []

    answer = generator_for(client).generate("q", [LEAVE], on_text=received.append)

    assert answer.text == "".join(received) == "Sixteen weeks."


def test_declining_is_recognised_as_not_answered() -> None:
    client = FakeClient(answer_pieces(NOT_FOUND_MESSAGE))

    answer = generator_for(client).generate("what is the wifi password?", [LEAVE])

    assert not answer.answered
    assert answer.sources == ()
    assert answer.stop_reason == "end_turn"


def test_partial_answer_that_mentions_a_gap_still_counts_as_answered() -> None:
    client = FakeClient(
        answer_pieces(f"Parents get sixteen weeks [1]. For adoption: {NOT_FOUND_MESSAGE}")
    )

    assert generator_for(client).generate("leave and adoption?", [LEAVE]).answered


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"])
def test_an_answer_the_api_withheld_is_not_an_answer(finish: str) -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]. And", finish=finish))

    answer = generator_for(client).generate("q", [LEAVE])

    assert not answer.answered
    assert answer.stop_reason == "refusal"


def test_a_question_the_api_blocked_is_not_an_answer() -> None:
    blocked = types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}}
    )
    client = FakeClient([blocked])
    received: list[str] = []

    answer = generator_for(client).generate("q", [LEAVE], on_text=received.append)

    assert not answer.answered
    assert answer.stop_reason == "refusal"
    assert answer.text == ""
    assert received == []


def test_a_marker_that_names_no_passage_is_not_a_citation() -> None:
    client = FakeClient(answer_pieces("Everyone gets a company car [7]."))

    answer = generator_for(client).generate("q", [LEAVE, EXPENSES])

    assert answer.sources == ()
    assert answer.text == "Everyone gets a company car [7]."  # left visible, not hidden


def test_a_reply_with_no_text_is_not_an_answer() -> None:
    client = FakeClient([piece("", finish="STOP")])

    answer = generator_for(client).generate("q", [LEAVE])

    assert not answer.answered
    assert answer.segments == ()
    assert answer.stop_reason == "end_turn"


def test_declining_without_chunks_works_when_nobody_listens_for_text() -> None:
    answer = generator_for(FakeClient()).generate("anything", [])

    assert answer.text == NOT_FOUND_MESSAGE
    assert not answer.answered


def test_no_chunks_means_no_model_call_and_no_answer() -> None:
    client = FakeClient(answer_pieces("should not be used"))
    received: list[str] = []

    answer = generator_for(client).generate("anything", [], on_text=received.append)

    assert client.models.calls == []
    assert not answer.answered
    assert answer.text == NOT_FOUND_MESSAGE
    assert received == [NOT_FOUND_MESSAGE]
    assert answer.stop_reason == "no_context"
    assert answer.model == MODEL


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (api_error(400, "INVALID_ARGUMENT", KEY_INVALID), "rejected the credential"),
        (api_error(401, "UNAUTHENTICATED"), "rejected the credential"),
        (
            api_error(403, "PERMISSION_DENIED", message="Not enabled for this project."),
            f"not allowed to use {MODEL}: Not enabled for this project.",
        ),
        (api_error(404, "NOT_FOUND"), f"The model {MODEL} was not found"),
        (
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [quota("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "15"), RETRY_IN_36S],
            ),
            "rate limit was reached. Try again in 36s.",
        ),
        (
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [quota("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "20"), RETRY_IN_36S],
            ),
            f"Today's request limit for {MODEL}",
        ),
        (api_error(429, "RESOURCE_EXHAUSTED"), "rate limit was reached. Try again shortly."),
        (
            api_error(429, "RESOURCE_EXHAUSTED", "not a list"),
            "rate limit was reached. Try again shortly.",
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", [{"violations": "not a list"}, "not a dict"]),
            "rate limit was reached. Try again shortly.",
        ),
        (
            api_error(400, "INVALID_ARGUMENT", message="Request contains an invalid argument."),
            r"returned an error \(400\): Request contains an invalid argument.",
        ),
        (
            api_error(503, "UNAVAILABLE", message="The model is overloaded."),
            r"returned an error \(503\): The model is overloaded.",
        ),
        (httpx2.ConnectError("Connection refused"), "Could not reach the Gemini API"),
        (httpx2.ReadTimeout("timed out"), "Could not reach the Gemini API"),
        (httpx.ConnectError("Connection refused"), "Could not reach the Gemini API"),
    ],
)
def test_api_failures_become_clear_errors(error: BaseException, expected: str) -> None:
    client = FakeClient(error=error)

    with pytest.raises(GenerationError, match=expected) as raised:
        generator_for(client).generate("q", [LEAVE])
    assert raised.value.__cause__ is error


def test_an_error_body_that_is_not_json_still_becomes_a_clear_error() -> None:
    client = FakeClient(error=errors.ServerError(502, "<html>Bad Gateway</html>"))

    with pytest.raises(GenerationError, match=r"returned an error \(502\)"):
        generator_for(client).generate("q", [LEAVE])


def test_a_failure_after_part_of_the_answer_arrived_is_still_an_error() -> None:
    client = FakeClient(
        [piece("Sixteen weeks [1]. And then")], error=httpx2.ReadError("connection lost")
    )
    received: list[str] = []

    with pytest.raises(GenerationError, match="Could not reach the Gemini API"):
        generator_for(client).generate("q", [LEAVE], on_text=received.append)
    assert "".join(received).startswith("Sixteen weeks.")


def test_an_unrelated_error_is_not_swallowed() -> None:
    client = FakeClient(error=TypeError("unsupported operand"))

    with pytest.raises(TypeError, match="unsupported operand"):
        generator_for(client).generate("q", [LEAVE])
