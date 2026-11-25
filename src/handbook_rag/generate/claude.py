"""Generate answers with Claude, using the API's built-in citations."""

import math
import time
from collections.abc import Callable, Sequence
from typing import Literal

import anthropic
import httpx2
from anthropic.types import Message

from handbook_rag.generate.base import (
    GenerationError,
    decline_without_context,
    is_answered,
    read_the_gap_form_as_a_decline,
)
from handbook_rag.generate.checks import HeldBack, check_reply
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import (
    CHECKED_VERSIONS,
    CURRENT_INSTRUCTIONS,
    GAP_FORM_VERSIONS,
    SYSTEM_PROMPTS,
    build_user_content,
    known_instructions,
)
from handbook_rag.ingest.models import Chunk
from handbook_rag.measuring import refuse_old_instructions

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# The refusals in the 400s that the SDK itself would send again by default: the server
# gave up waiting for the request (408), or for a lock (409). Every other one, 499
# included, means that the request, as it was sent, will not be accepted. The same rule
# holds for Gemini, by its own SDK's default list (gemini.py): a refusal in the 400s is
# worth another attempt only if the provider's SDK would make one.
_WORTH_REPEATING = frozenset({408, 409})
_NOT_IN_TIME = "The Claude API did not answer in time."


class ClaudeGenerator:
    """Answers a question from retrieved chunks with one streamed Claude request."""

    # The names OpenTelemetry's conventions for generative AI give Anthropic and a
    # request to its Messages API. tests/test_tracing.py checks them against the conventions.
    provider = "anthropic"
    operation = "chat"

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        max_tokens: int,
        effort: Effort,
        instructions: int = CURRENT_INSTRUCTIONS,
    ) -> None:
        # Another version than the current one is for an evaluation only (measuring.py).
        refuse_old_instructions(known_instructions(instructions), CURRENT_INSTRUCTIONS)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._instructions = instructions

    @property
    def model(self) -> str:
        return self._model

    @property
    def instructions(self) -> int:
        """The version of the answering instructions this generator sends."""
        return self._instructions

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Write an answer grounded in ``chunks``. ``on_text`` receives text as it arrives.

        Under instructions whose replies are checked, the text is passed on only when the
        reply is whole and checked (``checks.py``): the API says which passages a text
        cites only in its final message, so nothing could be passed on earlier without
        the risk of taking it back. The passages are given as the API's own blocks under
        every version, so there is no boundary of ours to make.
        """
        if not chunks:
            return decline_without_context(question, self._model, on_text)

        checked = self._instructions in CHECKED_VERSIONS
        started = time.perf_counter()
        first_text_at: float | None = None
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPTS[self._instructions],
                messages=[{"role": "user", "content": build_user_content(question, chunks)}],
                output_config={"effort": self._effort},
            ) as stream:
                for text in stream.text_stream:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    if on_text is not None and not checked:
                        on_text(text)
                message = stream.get_final_message()
        except anthropic.AuthenticationError as error:
            raise GenerationError(
                "The Claude API rejected the credential. Check ANTHROPIC_API_KEY in .env.",
                reason="configuration",
            ) from error
        except anthropic.PermissionDeniedError as error:
            raise GenerationError(
                f"The credential is not allowed to use {self._model}: {error.message}",
                reason="configuration",
            ) from error
        except anthropic.NotFoundError as error:
            raise GenerationError(
                f"The model {self._model} was not found.", reason="configuration"
            ) from error
        except anthropic.RateLimitError as error:
            raise GenerationError(
                "The Claude API rate limit was reached. Try again shortly.",
                reason="rate_limit",
                retry_after=_retry_after(error),
            ) from error
        except anthropic.APIStatusError as error:
            said = f"The Claude API returned an error ({error.status_code}): {error.message}"
            if 400 <= error.status_code < 500 and error.status_code not in _WORTH_REPEATING:
                # A rejected request, such as an organization with no credit left. The
                # same request would be rejected again.
                raise GenerationError(said, reason="configuration") from error
            raise GenerationError(said) from error
        except anthropic.APITimeoutError as error:
            raise GenerationError(_NOT_IN_TIME, reason="timeout") from error
        except anthropic.APIConnectionError as error:
            raise GenerationError("Could not reach the Claude API. Check the network.") from error
        except httpx2.TimeoutException as error:
            # The SDK turns a failure of the connection into its own error only while the
            # request is sent. In the middle of the stream it passes the HTTP library's
            # error on unchanged.
            raise GenerationError(_NOT_IN_TIME, reason="timeout") from error
        except httpx2.TransportError as error:
            raise GenerationError(
                "The connection to the Claude API broke. Check the network."
            ) from error
        except TypeError as error:
            # With no credential configured at all, the SDK raises a plain TypeError
            # before sending anything. Any other TypeError is a bug and must surface.
            if "authentication method" not in str(error):
                raise
            raise GenerationError(
                "No Claude API credential was found. Put ANTHROPIC_API_KEY in a .env file "
                "in the project folder.",
                reason="configuration",
            ) from error
        finished = time.perf_counter()

        segments = _segments(message, chunks)
        # An answer is a reply the model finished, or one that ran out of room.
        answered = is_answered(
            segments, refused=message.stop_reason not in ("end_turn", "max_tokens")
        )
        model_text = ""
        if self._instructions in GAP_FORM_VERSIONS:
            # Before the checks: a decline in other words is the model's, not a check's.
            segments, answered, model_text = read_the_gap_form_as_a_decline(
                segments, answered=answered
            )
        reply = check_reply(segments, chunks, answered=answered) if checked else None
        if reply is not None:
            segments, answered = reply.segments, reply.answered
            HeldBack(on_text).finished(segments)
        return Answer(
            question=question,
            segments=segments,
            answered=answered,
            model=message.model,
            stop_reason=message.stop_reason or "unknown",
            usage=TokenUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
            ),
            seconds=finished - started,
            seconds_to_first_text=None if first_text_at is None else first_text_at - started,
            checks=() if reply is None else reply.checks,
            model_text=model_text,
        )


def _retry_after(error: anthropic.RateLimitError) -> float | None:
    """The wait the API asked for in its retry-after header, in seconds, if it sent one.

    The header may also hold a date. That form is not read: a wait that cannot be read
    is reported as unknown, which is true.
    """
    value = error.response.headers.get("retry-after", "")
    try:
        seconds = float(value)
    except ValueError:
        return None
    # float() also reads "nan", "inf" and negative numbers, none of which is a wait.
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _segments(message: Message, chunks: Sequence[Chunk]) -> tuple[Segment, ...]:
    """Turn the response's text blocks into segments, mapping each citation to its chunk."""
    segments: list[Segment] = []
    for block in message.content:
        if block.type != "text":
            continue  # thinking blocks carry no answer text
        citations: list[Citation] = []
        for citation in block.citations or []:
            # A search result citation gives the position of the cited result among the
            # search results we sent, which is the position of the chunk in our list.
            if (
                citation.type == "search_result_location"
                and 0 <= citation.search_result_index < len(chunks)
            ):
                citations.append(
                    Citation(
                        chunk=chunks[citation.search_result_index], cited_text=citation.cited_text
                    )
                )
        segments.append(Segment(text=block.text, citations=tuple(citations)))
    return tuple(segments)
