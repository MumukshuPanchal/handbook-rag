"""The Gemini generator is tested against a stand-in client that returns real SDK objects.

No request is sent. Responses are built with the SDK's own response type, so these tests
fail if our code and the SDK disagree about the shape of a response. The tests that do
send requests are in ``tests/live``.
"""

import re
from collections.abc import Iterator
from typing import Any, cast

import httpx
import httpx2
import pydantic
import pytest
from google import genai
from google.genai import errors, types

from handbook_rag.generate.base import FailureReason, GenerationError
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT, MARKER_SYSTEM_PROMPTS
from handbook_rag.generate.prompt import CURRENT_INSTRUCTIONS, NOT_FOUND_MESSAGE
from handbook_rag.measuring import measuring_the_model
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


def generator_for(
    client: FakeClient, *, instructions: int = CURRENT_INSTRUCTIONS
) -> GeminiGenerator:
    """A generator on a stand-in client. Instructions other than the current ones are built
    inside the measuring switch, as only an evaluation may build them."""
    with measuring_the_model():
        return GeminiGenerator(
            cast(genai.Client, client), model=MODEL, max_tokens=4000, instructions=instructions
        )


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


PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
RETRY_IN_36S = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"}
# The two bodies below are what the API really sent on 2026-11-22, for a wrong key and
# for a model name that does not exist.
KEY_INVALID = [
    {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID",
        "domain": "googleapis.com",
        "metadata": {"service": SERVICE},
    }
]
MODEL_NOT_FOUND = (
    "models/gemini-0-does-not-exist is not found for API version v1, or is not supported "
    "for generateContent. Call ModelService.ListModels to see the list of available "
    "models and their supported methods."
)


def wrong_shape() -> pydantic.ValidationError:
    """The error the SDK raises for a reply that is JSON of a shape its types reject."""
    try:
        types.GenerateContentResponse.model_validate(
            {"usage_metadata": {"prompt_token_count": "not a number"}}
        )
    except pydantic.ValidationError as error:
        return error
    raise AssertionError("the SDK accepted a reply of the wrong shape")


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
    assert call["config"].system_instruction == MARKER_SYSTEM_PROMPTS[CURRENT_INSTRUCTIONS]
    assert call["config"].max_output_tokens == 4000
    # One question, one request: the SDK must not send follow-up requests by itself.
    assert call["config"].automatic_continuation is False
    prompt = call["contents"]
    # Each passage between two lines that hold the request's own boundary code.
    boundary = re.fullmatch(r"Boundary code: ([0-9a-f]{32})", prompt.split("\n")[0])
    assert boundary is not None
    code = boundary.group(1)
    assert f"BEGIN SEARCH RESULT 1 {code}\nTitle: {LEAVE.context_header}\n{LEAVE.text}\n" in prompt
    assert f"END SEARCH RESULT 2 {code}" in prompt
    assert prompt.index(LEAVE.text) < prompt.index(EXPENSES.text)
    assert prompt.endswith(f"QUESTION {code}\nHow long is parental leave?")


def test_each_request_gets_a_boundary_code_of_its_own() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))
    generator = generator_for(client)

    generator.generate("q", [LEAVE])
    client.models._pieces = answer_pieces("Sixteen weeks [1].")
    generator.generate("q", [LEAVE])

    first, second = (call["contents"].split("\n")[0] for call in client.models.calls)
    assert first != second


def test_a_generator_of_version_1_sends_the_first_instructions_and_layout() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))

    generator_for(client, instructions=1).generate("How long is parental leave?", [LEAVE])

    (call,) = client.models.calls
    assert call["config"].system_instruction == MARKER_SYSTEM_PROMPT
    assert '<search_result number="1">' in call["contents"]
    assert call["contents"].endswith("Question: How long is parental leave?")


def test_text_is_passed_on_as_it_arrives_without_the_markers() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1", "]. One thousand", " dollars [2]."))
    received: list[str] = []

    answer = generator_for(client).generate("q", [LEAVE, EXPENSES], on_text=received.append)

    assert len(received) > 1  # in pieces, not all at the end
    assert "".join(received) == answer.text == "Sixteen weeks. One thousand dollars."
    assert all(text for text in received)


def test_text_still_held_back_when_the_stream_ends_is_passed_on_too() -> None:
    # The reply ends in what could have become a marker, so the parser was holding it.
    client = FakeClient(answer_pieces("Sixteen weeks [1]. See section [1"))
    received: list[str] = []

    answer = generator_for(client).generate("q", [LEAVE], on_text=received.append)

    assert "".join(received) == answer.text == "Sixteen weeks. See section [1"


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


def test_token_counts_are_a_running_total_so_the_last_ones_are_kept() -> None:
    # Every piece of a real stream carries the counts so far.
    client = FakeClient(
        [
            piece("Sixteen ", usage={"prompt_token_count": 1200, "candidates_token_count": 2}),
            piece("weeks [1].", usage={"prompt_token_count": 1200, "candidates_token_count": 6}),
            piece(
                "", finish="STOP", usage={"prompt_token_count": 1200, "candidates_token_count": 7}
            ),
        ]
    )

    answer = generator_for(client).generate("q", [LEAVE])

    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (1200, 7)


def test_a_response_without_token_counts_or_model_name_still_gives_an_answer() -> None:
    client = FakeClient([piece("Sixteen weeks [1].", finish="STOP", model_version=None)])

    answer = generator_for(client).generate("q", [LEAVE])

    assert answer.answered
    assert answer.model == MODEL
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (0, 0)


def test_an_answer_cut_off_at_the_token_limit_says_so() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]. And then", finish="MAX_TOKENS"))

    answer = generator_for(client).generate("q", [LEAVE])

    assert answer.stop_reason == "max_tokens"  # the command line looks for this word
    assert answer.answered


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


def test_declining_with_a_marker_attached_is_still_not_answered() -> None:
    client = FakeClient(answer_pieces(f"{NOT_FOUND_MESSAGE} [1]"))

    answer = generator_for(client).generate("what is the wifi password?", [LEAVE])

    assert answer.text == NOT_FOUND_MESSAGE
    assert not answer.answered


def test_partial_answer_that_mentions_a_gap_still_counts_as_answered() -> None:
    client = FakeClient(
        answer_pieces(f"Parents get sixteen weeks [1]. For adoption: {NOT_FOUND_MESSAGE}")
    )

    assert generator_for(client).generate("leave and adoption?", [LEAVE]).answered


@pytest.mark.parametrize(
    "finish",
    [
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MODEL_ARMOR",
        "ESCALATION",
        "PUP_LIMITED_DISABLED",
    ],
)
def test_an_answer_the_api_withheld_is_not_an_answer(finish: str) -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]. And", finish=finish))

    answer = generator_for(client).generate("q", [LEAVE])

    assert not answer.answered
    assert answer.stop_reason == "refusal"


@pytest.mark.parametrize(
    ("finish", "stop_reason"),
    [("OTHER", "other"), ("MALFORMED_RESPONSE", "malformed_response"), (None, "unknown")],
)
def test_a_reply_that_did_not_come_to_a_proper_end_is_not_an_answer(
    finish: str | None, stop_reason: str
) -> None:
    client = FakeClient([piece("Sixteen weeks [1]. And", finish=finish)])

    answer = generator_for(client).generate("q", [LEAVE])

    assert not answer.answered
    assert answer.stop_reason == stop_reason
    assert answer.text == "Sixteen weeks. And"  # what did arrive is kept


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

    # Under instructions without checks: a checked reply that cites nothing is declined.
    answer = generator_for(client, instructions=1).generate("q", [LEAVE, EXPENSES])

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
    assert answer.requests == 0


@pytest.mark.parametrize(
    ("error", "expected", "reason", "retry_after"),
    [
        (
            api_error(400, "INVALID_ARGUMENT", KEY_INVALID),
            "rejected the credential",
            "configuration",
            None,
        ),
        (api_error(401, "UNAUTHENTICATED"), "rejected the credential", "configuration", None),
        (
            api_error(403, "PERMISSION_DENIED", message="Not enabled for this project."),
            f"not allowed to use {MODEL}: Not enabled for this project.",
            "configuration",
            None,
        ),
        (
            api_error(404, "NOT_FOUND", message=MODEL_NOT_FOUND),
            f"The model {MODEL} was not found",
            "configuration",
            None,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", [quota(PER_MINUTE, "15"), RETRY_IN_36S]),
            f"The per-minute limit for {MODEL} was reached for this key's Google project. "
            "Try again in 36s.",
            "rate_limit",
            36.0,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", [quota(PER_MINUTE, "15")]),
            f"The per-minute limit for {MODEL} was reached for this key's Google project. "
            "Try again shortly.",
            "rate_limit",
            None,
        ),
        (
            # The wait the API names is left out: it does not bring back a day's allowance.
            api_error(429, "RESOURCE_EXHAUSTED", [quota(PER_DAY, "20"), RETRY_IN_36S]),
            f"Today's limit for {MODEL} is used up for this key's Google project. "
            "Try again tomorrow",
            "quota",
            None,
        ),
        (
            # Both limits reported at once: the day's limit is the one that decides.
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [quota(PER_MINUTE, "15"), quota(PER_DAY, "20"), RETRY_IN_36S],
            ),
            f"Today's limit for {MODEL} is used up",
            "quota",
            None,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", [RETRY_IN_36S]),
            "The Gemini API rate limit was reached. Try again in 36s.",
            "rate_limit",
            36.0,
        ),
        (
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0.5s"}],
            ),
            "The Gemini API rate limit was reached. Try again in 0.5s.",
            "rate_limit",
            0.5,
        ),
        (
            # A wait in a form the API does not use is not guessed at.
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36"}],
            ),
            "The Gemini API rate limit was reached. Try again in 36.",
            "rate_limit",
            None,
        ),
        (
            api_error(
                429,
                "RESOURCE_EXHAUSTED",
                [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": 36}],
            ),
            "The Gemini API rate limit was reached. Try again shortly.",
            "rate_limit",
            None,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED"),
            "The Gemini API rate limit was reached. Try again shortly.",
            "rate_limit",
            None,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", "not a list"),
            "The Gemini API rate limit was reached. Try again shortly.",
            "rate_limit",
            None,
        ),
        (
            api_error(429, "RESOURCE_EXHAUSTED", [{"violations": "not a list"}, "not a dict"]),
            "The Gemini API rate limit was reached. Try again shortly.",
            "rate_limit",
            None,
        ),
        (
            # A request the API refuses as it was sent is refused again: not worth repeating.
            api_error(400, "INVALID_ARGUMENT", message="Request contains an invalid argument."),
            r"returned an error \(400\): Request contains an invalid argument.",
            "configuration",
            None,
        ),
        (
            api_error(400, "FAILED_PRECONDITION", message="User location is not supported."),
            r"returned an error \(400\): User location is not supported.",
            "configuration",
            None,
        ),
        (
            # Neither 409 nor 499 is on the SDK's own list of codes worth repeating.
            api_error(409, "ABORTED", message="The operation was aborted."),
            r"returned an error \(409\)",
            "configuration",
            None,
        ),
        (
            api_error(499, "CANCELLED", message="The operation was cancelled."),
            r"returned an error \(499\)",
            "configuration",
            None,
        ),
        (
            # The server gave up waiting for the request, which a new one may fix.
            api_error(408, "DEADLINE_EXCEEDED", message="Request timeout."),
            r"returned an error \(408\)",
            "unavailable",
            None,
        ),
        (
            api_error(503, "UNAVAILABLE", message="The model is overloaded."),
            r"returned an error \(503\): The model is overloaded.",
            "unavailable",
            None,
        ),
        (
            api_error(500, "INTERNAL", message="An internal error has occurred."),
            r"returned an error \(500\)",
            "unavailable",
            None,
        ),
        (
            api_error(504, "DEADLINE_EXCEEDED", message="Deadline expired before operation."),
            r"returned an error \(504\): Deadline expired before operation.",
            "timeout",
            None,
        ),
        (
            httpx2.ConnectError("Connection refused"),
            "Could not reach the Gemini API",
            "unavailable",
            None,
        ),
        (httpx2.ReadTimeout("timed out"), "did not answer in time", "timeout", None),
        (httpx2.ConnectTimeout("timed out"), "did not answer in time", "timeout", None),
        (httpx.ReadTimeout("timed out"), "did not answer in time", "timeout", None),
        (httpx.PoolTimeout("timed out"), "did not answer in time", "timeout", None),
        (
            httpx2.RemoteProtocolError("peer closed"),
            "Could not reach the Gemini API",
            "unavailable",
            None,
        ),
        (
            httpx.ConnectError("Connection refused"),
            "Could not reach the Gemini API",
            "unavailable",
            None,
        ),
        (
            httpx2.DecodingError("bad gzip data"),
            "sent a reply that could not be read",
            "unavailable",
            None,
        ),
        (
            httpx.DecodingError("bad gzip data"),
            "sent a reply that could not be read",
            "unavailable",
            None,
        ),
        (
            errors.UnknownApiResponseError("Failed to parse response as JSON. Raw response: {no"),
            "sent a reply that could not be read",
            "unavailable",
            None,
        ),
        (wrong_shape(), "sent a reply that could not be read", "unavailable", None),
    ],
)
def test_api_failures_become_clear_errors(
    error: BaseException, expected: str, reason: FailureReason, retry_after: float | None
) -> None:
    client = FakeClient(error=error)

    with pytest.raises(GenerationError, match=expected) as raised:
        generator_for(client).generate("q", [LEAVE])
    assert raised.value.__cause__ is error
    assert raised.value.reason == reason
    assert raised.value.retry_after == retry_after


@pytest.mark.parametrize("wait", ["36sx", "36s extra", "about 36s", "1e3s", "36 s", "-36s"])
def test_a_wait_written_in_any_other_form_gives_no_wait(wait: str) -> None:
    """Only the whole of a written wait counts, so a caller is never told a wrong one."""
    retry = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": wait}
    client = FakeClient(error=api_error(429, "RESOURCE_EXHAUSTED", [retry]))

    with pytest.raises(GenerationError) as raised:
        generator_for(client).generate("q", [LEAVE])
    assert raised.value.reason == "rate_limit"
    assert raised.value.retry_after is None


def test_the_generator_names_the_model_it_asks_for() -> None:
    # The span of a failed request names the model too, and there is no reply to read it from.
    assert generator_for(FakeClient()).model == MODEL


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


def test_a_silence_after_part_of_the_answer_arrived_is_a_timeout() -> None:
    client = FakeClient(
        [piece("Sixteen weeks [1]. And then")], error=httpx2.ReadTimeout("timed out")
    )
    received: list[str] = []

    with pytest.raises(GenerationError, match="did not answer in time") as raised:
        generator_for(client).generate("q", [LEAVE], on_text=received.append)
    assert raised.value.reason == "timeout"
    assert "".join(received).startswith("Sixteen weeks.")


def test_a_mistake_in_our_own_settings_is_not_reported_as_an_unreadable_reply() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))
    generator = GeminiGenerator(
        cast(genai.Client, client), model=MODEL, max_tokens=cast(int, "many")
    )

    # The SDK's own types reject the setting. That is our mistake, so it must surface
    # as what it is, and nothing is sent.
    with pytest.raises(pydantic.ValidationError):
        generator.generate("q", [LEAVE])
    assert client.models.calls == []


def test_an_unrelated_error_is_not_swallowed() -> None:
    client = FakeClient(error=TypeError("unsupported operand"))

    with pytest.raises(TypeError, match="unsupported operand"):
        generator_for(client).generate("q", [LEAVE])
