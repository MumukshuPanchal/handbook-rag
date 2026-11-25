"""Generate answers with Gemini, citing passages by numbered markers.

The Gemini API has no feature that returns which of *our* passages a statement rests
on, so this generator uses the portable method in :mod:`handbook_rag.generate.markers`.
"""

import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx
import httpx2
import pydantic
from google import genai
from google.genai import errors, types

from handbook_rag.generate.base import (
    GenerationError,
    decline_without_context,
    is_answered,
    read_the_gap_form_as_a_decline,
)
from handbook_rag.generate.checks import HeldBack, check_reply
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPTS, MarkerParser, numbered_prompt
from handbook_rag.generate.models import Answer, TokenUsage
from handbook_rag.generate.prompt import (
    CHECKED_VERSIONS,
    CURRENT_INSTRUCTIONS,
    GAP_FORM_VERSIONS,
    known_instructions,
)
from handbook_rag.ingest.models import Chunk
from handbook_rag.measuring import refuse_old_instructions

# The SDK passes failures of the connection on unchanged, from whichever HTTP library it
# runs on. The first group means the API went silent for longer than the client's time
# limit, while connecting or between two pieces of the reply. The second means the API
# was not reached or the connection broke. The third means something arrived that could
# not be read: it could not be decoded, it was not JSON, or it was JSON of a shape the
# SDK's own types reject.
_TIMEOUT_ERRORS = (httpx.TimeoutException, httpx2.TimeoutException)
_NETWORK_ERRORS = (httpx.TransportError, httpx2.TransportError)
_UNREADABLE_REPLY_ERRORS = (
    httpx.RequestError,
    httpx2.RequestError,
    errors.UnknownApiResponseError,
    pydantic.ValidationError,
)

# The two ways of stopping that leave an answer: the model finished, or it ran out of
# room. After any other stop the reply is not counted as an answer, whatever it holds.
_ANSWER_ENDINGS = frozenset({types.FinishReason.STOP, types.FinishReason.MAX_TOKENS})
# Reasons for stopping that mean the API withheld the answer.
_WITHHELD = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
        types.FinishReason.LANGUAGE,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.SPII,
        types.FinishReason.MODEL_ARMOR,
        types.FinishReason.ESCALATION,
        types.FinishReason.PUP_LIMITED_DISABLED,
    }
)
# How the API writes a wait, such as "36s" or "0.5s".
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)s")


class GeminiGenerator:
    """Answers a question from retrieved chunks with one streamed Gemini request."""

    # The names OpenTelemetry's conventions for generative AI give the Gemini API and
    # its request for content. tests/test_tracing.py checks them against the conventions.
    provider = "gcp.gemini"
    operation = "generate_content"

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        max_tokens: int,
        instructions: int = CURRENT_INSTRUCTIONS,
    ) -> None:
        # Another version than the current one is for an evaluation only (measuring.py).
        refuse_old_instructions(known_instructions(instructions), CURRENT_INSTRUCTIONS)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
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

        Under instructions whose replies are checked, text is passed on once a citation
        follows it, and the rest after the checks (``checks.py``), so that nothing passed
        on is ever taken back.
        """
        if not chunks:
            return decline_without_context(question, self._model, on_text)

        checked = self._instructions in CHECKED_VERSIONS
        held = HeldBack(on_text)
        parser = MarkerParser(chunks)
        started = time.perf_counter()
        first_text_at: float | None = None
        # The stop reason comes with the last piece of the stream. The token counts come
        # with every piece as a running total, so the last ones are the request's.
        finish: types.FinishReason | None = None
        usage: types.GenerateContentResponseUsageMetadata | None = None
        served_by: str | None = None
        blocked = False
        # Built before the request, so that a mistake in our own settings is never
        # reported as a reply that could not be read.
        prompt = numbered_prompt(question, chunks, instructions=self._instructions)
        config = types.GenerateContentConfig(
            system_instruction=MARKER_SYSTEM_PROMPTS[self._instructions],
            max_output_tokens=self._max_tokens,
            # No follow-up requests. Left on, the SDK sends the request again by itself
            # when a reply stops at the API's size limit or the stream drops, which would
            # spend a small daily allowance unseen. The only repeats left are the ones our
            # own code makes and counts: generate/retrying.py for a question asked on the
            # command line or over HTTP, and the evaluation's own loop for a run.
            automatic_continuation=False,
        )
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model, contents=prompt, config=config
            )
            for piece in stream:
                text = _answer_text(piece)
                if text:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    shown = parser.feed(text)
                    if checked:
                        held.closed_so_far(parser.segments)
                    else:
                        _pass_on(shown, on_text)
                candidate = piece.candidates[0] if piece.candidates else None
                if candidate is not None and candidate.finish_reason is not None:
                    finish = candidate.finish_reason
                usage = piece.usage_metadata or usage
                served_by = piece.model_version or served_by
                if piece.prompt_feedback is not None and piece.prompt_feedback.block_reason:
                    blocked = True
        except errors.APIError as error:
            raise self._explain(error) from error
        except _TIMEOUT_ERRORS as error:
            raise GenerationError(
                "The Gemini API did not answer in time.", reason="timeout"
            ) from error
        except _NETWORK_ERRORS as error:
            raise GenerationError("Could not reach the Gemini API. Check the network.") from error
        except _UNREADABLE_REPLY_ERRORS as error:
            raise GenerationError(
                "The Gemini API sent a reply that could not be read. Try again."
            ) from error
        rest = parser.finish()
        finished = time.perf_counter()

        withheld = blocked or finish in _WITHHELD
        segments = parser.segments
        answered = is_answered(segments, refused=withheld or finish not in _ANSWER_ENDINGS)
        model_text = ""
        if self._instructions in GAP_FORM_VERSIONS:
            # Before the checks: a decline in other words is the model's, not a check's.
            segments, answered, model_text = read_the_gap_form_as_a_decline(
                segments, answered=answered
            )
        reply = check_reply(segments, chunks, answered=answered) if checked else None
        if reply is None:
            _pass_on(rest, on_text)
        else:
            segments, answered = reply.segments, reply.answered
            held.finished(segments)
        return Answer(
            question=question,
            segments=segments,
            answered=answered,
            model=served_by or self._model,
            stop_reason=_stop_reason(finish, withheld=withheld),
            usage=_usage(usage),
            seconds=finished - started,
            seconds_to_first_text=None if first_text_at is None else first_text_at - started,
            checks=() if reply is None else reply.checks,
            model_text=model_text,
        )

    def _explain(self, error: errors.APIError) -> GenerationError:
        """Turn an error from the API into one that says what to do about it."""
        details = _details(error)
        reasons = {detail.get("reason") for detail in details}
        if error.code == 401 or "API_KEY_INVALID" in reasons:
            return GenerationError(
                "The Gemini API rejected the credential. Check GEMINI_API_KEY in .env.",
                reason="configuration",
            )
        if error.code == 403:
            return GenerationError(
                f"The credential is not allowed to use {self._model}: {error.message}",
                reason="configuration",
            )
        if error.code == 404:
            return GenerationError(
                f"The model {self._model} was not found.", reason="configuration"
            )
        if error.code == 429:
            return self._explain_rate_limit(details)
        message = f"The Gemini API returned an error ({error.code}): {error.message}"
        if error.code == 504:
            # The API, or a gateway in front of it, gave up waiting for the model. The
            # SDK also sends the client's time limit with every request, as the
            # X-Server-Timeout header, so this can be the same limit, kept at the other end.
            return GenerationError(message, reason="timeout")
        if 400 <= error.code < 500 and error.code != 408:
            # The API refused the request as it was sent. The same request would be
            # refused again, so trying again only spends another one. Of the refusals in
            # the 400s, the SDK's own default list of codes worth repeating holds 408 and
            # 429 only, and a 429 is read above. So 409 and 499 are not repeated here,
            # where the Claude SDK repeats 409 (claude.py).
            return GenerationError(message, reason="configuration")
        return GenerationError(message)

    def _explain_rate_limit(self, details: Sequence[Mapping[str, Any]]) -> GenerationError:
        """Say which limit was reached, when the API tells us.

        The free tier allows a small amount of use a minute and a fixed amount a day,
        for each model. Which of the two was reached decides what helps: waiting a few
        seconds, or waiting until the next day. When the response does not say, the
        message stays general.

        The limits belong to the Google project a key was made in, not to the key, so
        the message does not suggest that another key would help.
        """
        limits = [
            str(violation.get("quotaId", ""))
            for detail in details
            for violation in _dictionaries(detail.get("violations"))
        ]
        if any("PerDay" in limit for limit in limits):
            # The API may name a wait here too. It is not passed on: waiting that long
            # does not bring back an allowance that is renewed once a day.
            return GenerationError(
                f"Today's limit for {self._model} is used up for this key's Google project. "
                "Try again tomorrow, or choose another model with HANDBOOK_RAG_ANSWER_MODEL.",
                reason="quota",
            )
        wait = next((detail["retryDelay"] for detail in details if "retryDelay" in detail), None)
        when = f"in {wait}" if isinstance(wait, str) and wait else "shortly"
        if any("PerMinute" in limit for limit in limits):
            message = (
                f"The per-minute limit for {self._model} was reached for this key's Google "
                f"project. Try again {when}."
            )
        else:
            message = f"The Gemini API rate limit was reached. Try again {when}."
        return GenerationError(message, reason="rate_limit", retry_after=_seconds(wait))


def _pass_on(text: str, on_text: Callable[[str], None] | None) -> None:
    if text and on_text is not None:
        on_text(text)


def _answer_text(piece: types.GenerateContentResponse) -> str:
    """The answer text in one piece of the stream, leaving out the model's own reasoning."""
    if not piece.candidates:
        return ""
    content = piece.candidates[0].content
    if content is None or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text and not part.thought)


def _stop_reason(finish: types.FinishReason | None, *, withheld: bool) -> str:
    """Why the model stopped, in the words the rest of the application uses."""
    if withheld:
        return "refusal"
    if finish is None:
        return "unknown"
    if finish is types.FinishReason.STOP:
        return "end_turn"
    return finish.name.lower()  # MAX_TOKENS becomes "max_tokens"


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> TokenUsage:
    if metadata is None:
        return TokenUsage()
    # Gemini counts tokens read from its cache inside the prompt count. We keep them
    # apart, because they are billed at a different price.
    cached = metadata.cached_content_token_count or 0
    return TokenUsage(
        input_tokens=(metadata.prompt_token_count or 0) - cached,
        # Tokens the model spends reasoning before it answers are billed as output.
        output_tokens=(metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0),
        cache_read_input_tokens=cached,
    )


def _seconds(wait: object) -> float | None:
    """A wait the API wrote, such as "36s", in seconds, or ``None`` if it is not one."""
    if not isinstance(wait, str):
        return None
    written = _SECONDS.fullmatch(wait)
    return float(written.group(1)) if written else None


def _details(error: errors.APIError) -> list[Mapping[str, Any]]:
    """The structured details the API attaches to an error, or nothing if it sent none."""
    body = error.details
    inner = body.get("error") if isinstance(body, Mapping) else None
    return _dictionaries(inner.get("details") if isinstance(inner, Mapping) else None)


def _dictionaries(value: object) -> list[Mapping[str, Any]]:
    """The dictionaries in a value that should be a list of them.

    The API's error details are not covered by the SDK's types, so their shape is checked
    instead of trusted.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]
