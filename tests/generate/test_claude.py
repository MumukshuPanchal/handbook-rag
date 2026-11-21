"""The generator is tested against a stand-in client that returns real SDK objects.

No request is sent. Responses are built with the SDK's own ``Message`` type, so these
tests fail if our code and the SDK disagree about the shape of a response.
"""

from collections.abc import Iterator
from types import TracebackType
from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from handbook_rag.generate.claude import ClaudeGenerator, GenerationError
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def citation(index: int, cited_text: str) -> dict[str, Any]:
    return {
        "type": "search_result_location",
        "cited_text": cited_text,
        "source": "https://handbook.example.com/",
        "title": "Leave",
        "search_result_index": index,
        "start_block_index": 0,
        "end_block_index": 1,
    }


def message(
    blocks: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 1200, "output_tokens": 80},
        }
    )


class FakeStream:
    def __init__(self, final: Message) -> None:
        self._final = final

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    @property
    def text_stream(self) -> Iterator[str]:
        for block in self._final.content:
            if block.type == "text":
                middle = len(block.text) // 2
                yield block.text[:middle]
                yield block.text[middle:]

    def get_final_message(self) -> Message:
        return self._final


class FakeMessages:
    def __init__(self, outcome: Message | BaseException) -> None:
        self._outcome = outcome
        self.requests: list[dict[str, Any]] = []

    def stream(self, **request: Any) -> FakeStream:
        self.requests.append(request)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return FakeStream(self._outcome)


class FakeClient:
    def __init__(self, outcome: Message | BaseException) -> None:
        self.messages = FakeMessages(outcome)


def generator_for(outcome: Message | BaseException) -> tuple[ClaudeGenerator, FakeMessages]:
    client = FakeClient(outcome)
    generator = ClaudeGenerator(
        cast(anthropic.Anthropic, client), model="claude-haiku-5-5", max_tokens=4000, effort="low"
    )
    return generator, client.messages


CITED_ANSWER = message(
    [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {
            "type": "text",
            "text": "New parents get sixteen weeks of paid leave.",
            "citations": [citation(0, "New parents get sixteen weeks of paid leave.")],
        },
        {"type": "text", "text": " A laptop can be expensed."},
        {
            "type": "text",
            "text": " The limit is one thousand dollars.",
            "citations": [citation(1, "Expense a laptop up to one thousand dollars.")],
        },
    ]
)


def test_answer_text_and_citations_map_back_to_the_right_chunks() -> None:
    generator, _ = generator_for(CITED_ANSWER)

    answer = generator.generate("What do parents get?", [LEAVE, EXPENSES])

    assert answer.text == (
        "New parents get sixteen weeks of paid leave. A laptop can be expensed."
        " The limit is one thousand dollars."
    )
    assert answer.answered
    assert answer.sources == (LEAVE, EXPENSES)
    assert answer.segments[0].citations[0].chunk == LEAVE
    assert (
        answer.segments[0].citations[0].cited_text == "New parents get sixteen weeks of paid leave."
    )
    assert answer.segments[1].citations == ()
    assert answer.text_with_markers == (
        "New parents get sixteen weeks of paid leave.[1] A laptop can be expensed."
        " The limit is one thousand dollars.[2]"
    )


def test_request_carries_the_instructions_model_and_search_results() -> None:
    generator, messages = generator_for(CITED_ANSWER)

    generator.generate("What do parents get?", [LEAVE, EXPENSES])

    (request,) = messages.requests
    assert request["model"] == "claude-haiku-5-5"
    assert request["max_tokens"] == 4000
    assert request["system"] == SYSTEM_PROMPT
    assert request["output_config"] == {"effort": "low"}
    (user_message,) = request["messages"]
    assert user_message["role"] == "user"
    assert [block["type"] for block in user_message["content"]] == [
        "search_result",
        "search_result",
        "text",
    ]
    assert user_message["content"][-1]["text"] == "What do parents get?"


def test_text_is_passed_on_as_it_arrives() -> None:
    generator, _ = generator_for(CITED_ANSWER)
    received: list[str] = []

    answer = generator.generate("q", [LEAVE, EXPENSES], on_text=received.append)

    assert len(received) > 3
    assert "".join(received) == answer.text
    assert answer.seconds_to_first_text is not None
    assert answer.seconds >= answer.seconds_to_first_text


def test_usage_model_and_stop_reason_come_from_the_response() -> None:
    response = message(
        [{"type": "text", "text": "Partial", "citations": [citation(0, "x")]}],
        stop_reason="max_tokens",
        usage={
            "input_tokens": 900,
            "output_tokens": 4000,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 50,
        },
    )
    generator, _ = generator_for(response)

    answer = generator.generate("q", [LEAVE])

    assert answer.model == "claude-haiku-5-5"
    assert answer.stop_reason == "max_tokens"
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (900, 4000)
    assert (answer.usage.cache_read_input_tokens, answer.usage.cache_creation_input_tokens) == (
        300,
        50,
    )


def test_declining_is_recognised_as_not_answered() -> None:
    generator, _ = generator_for(message([{"type": "text", "text": NOT_FOUND_MESSAGE}]))

    answer = generator.generate("What is the meaning of life?", [LEAVE])

    assert not answer.answered
    assert answer.sources == ()
    assert answer.text == NOT_FOUND_MESSAGE


def test_partial_answer_that_mentions_a_gap_still_counts_as_answered() -> None:
    response = message(
        [
            {
                "type": "text",
                "text": "Parents get sixteen weeks.",
                "citations": [citation(0, "sixteen weeks")],
            },
            {"type": "text", "text": " I could not find this in the handbook: the pay rate."},
        ]
    )
    generator, _ = generator_for(response)

    assert generator.generate("q", [LEAVE]).answered


def test_safety_refusal_is_not_an_answer() -> None:
    generator, _ = generator_for(message([{"type": "text", "text": ""}], stop_reason="refusal"))

    answer = generator.generate("q", [LEAVE])

    assert not answer.answered
    assert answer.stop_reason == "refusal"


def test_citation_pointing_outside_our_chunks_is_ignored() -> None:
    response = message([{"type": "text", "text": "Claim.", "citations": [citation(7, "x")]}])
    generator, _ = generator_for(response)

    answer = generator.generate("q", [LEAVE])

    assert answer.sources == ()


def test_no_chunks_means_no_model_call_and_no_answer() -> None:
    generator, messages = generator_for(CITED_ANSWER)
    received: list[str] = []

    answer = generator.generate("q", [], on_text=received.append)

    assert messages.requests == []
    assert not answer.answered
    assert answer.text == NOT_FOUND_MESSAGE
    assert received == [NOT_FOUND_MESSAGE]
    assert answer.usage.input_tokens == 0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            anthropic.AuthenticationError(
                "invalid x-api-key", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            "ANTHROPIC_API_KEY",
        ),
        (
            anthropic.RateLimitError(
                "slow down", response=httpx2.Response(429, request=REQUEST), body=None
            ),
            "rate limit",
        ),
        (
            anthropic.BadRequestError(
                "Your credit balance is too low to access the Anthropic API.",
                response=httpx2.Response(400, request=REQUEST),
                body=None,
            ),
            "credit balance is too low",
        ),
        (
            anthropic.NotFoundError(
                "model not found", response=httpx2.Response(404, request=REQUEST), body=None
            ),
            "was not found",
        ),
        (anthropic.APIConnectionError(request=REQUEST), "Could not reach"),
        (
            TypeError('"Could not resolve authentication method. Expected one of api_key..."'),
            "No Claude API credential",
        ),
    ],
)
def test_api_failures_become_clear_errors(error: BaseException, expected: str) -> None:
    generator, _ = generator_for(error)

    with pytest.raises(GenerationError, match=expected):
        generator.generate("q", [LEAVE])


def test_an_unrelated_type_error_is_not_swallowed() -> None:
    generator, _ = generator_for(TypeError("a real bug"))

    with pytest.raises(TypeError, match="a real bug"):
        generator.generate("q", [LEAVE])
