"""Generate answers with Claude, using the API's built-in citations."""

import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import anthropic
from anthropic.types import Message

from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_content
from handbook_rag.ingest.models import Chunk

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class GenerationError(RuntimeError):
    """The model could not be reached or refused the request."""


class Generator(Protocol):
    """Anything that can write an answer to a question from a set of chunks."""

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer: ...


class ClaudeGenerator:
    """Answers a question from retrieved chunks with one streamed Claude request."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        max_tokens: int,
        effort: Effort,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Write an answer grounded in ``chunks``. ``on_text`` receives text as it arrives."""
        if not chunks:
            # Nothing was retrieved, so there is nothing to ground an answer in. Say so
            # without spending a model call.
            if on_text is not None:
                on_text(NOT_FOUND_MESSAGE)
            return Answer(
                question=question,
                segments=(Segment(NOT_FOUND_MESSAGE),),
                answered=False,
                model=self._model,
                stop_reason="no_context",
            )

        started = time.perf_counter()
        first_text_at: float | None = None
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_content(question, chunks)}],
                output_config={"effort": self._effort},
            ) as stream:
                for text in stream.text_stream:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    if on_text is not None:
                        on_text(text)
                message = stream.get_final_message()
        except anthropic.AuthenticationError as error:
            raise GenerationError(
                "The Claude API rejected the credential. Check ANTHROPIC_API_KEY in .env."
            ) from error
        except anthropic.PermissionDeniedError as error:
            raise GenerationError(
                f"The credential is not allowed to use {self._model}: {error.message}"
            ) from error
        except anthropic.NotFoundError as error:
            raise GenerationError(f"The model {self._model} was not found.") from error
        except anthropic.RateLimitError as error:
            raise GenerationError(
                "The Claude API rate limit was reached. Try again shortly."
            ) from error
        except anthropic.APIStatusError as error:
            # Includes a rejected request, such as an organization with no credit left.
            raise GenerationError(
                f"The Claude API returned an error ({error.status_code}): {error.message}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise GenerationError("Could not reach the Claude API. Check the network.") from error
        except TypeError as error:
            # With no credential configured at all, the SDK raises a plain TypeError
            # before sending anything. Any other TypeError is a bug and must surface.
            if "authentication method" not in str(error):
                raise
            raise GenerationError(
                "No Claude API credential was found. Put ANTHROPIC_API_KEY in a .env file "
                "in the project folder."
            ) from error
        finished = time.perf_counter()

        segments = _segments(message, chunks)
        return Answer(
            question=question,
            segments=segments,
            answered=_is_answered(message, segments),
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
        )


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


def _is_answered(message: Message, segments: Sequence[Segment]) -> bool:
    """Whether the assistant gave an answer, as opposed to declining or being cut off early."""
    if message.stop_reason == "refusal":
        return False
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        return False
    has_citations = any(segment.citations for segment in segments)
    return has_citations or NOT_FOUND_MESSAGE.rstrip(".").lower() not in text.lower()
