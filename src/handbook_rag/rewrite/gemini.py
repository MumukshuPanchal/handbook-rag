"""Ask Gemini to rewrite a question: one small request, read as an answer's request is.

The request is not streamed. The reply is one line that nobody reads as it arrives, and
without a stream the client's time limit bounds the whole wait for the reply, where for
a stream it bounds each silence.

Everything else is the answer generator's own (``generate/gemini.py``), used from here
and not written a second time: which failures of the connection mean what, how an error
from the API becomes a reason (which limit was reached, what is a timeout, what is not
worth another try: ADR 0018), and how a reply's text, stop reason and tokens are read.
A second copy of those rules would drift from the first.

Those names are private to that module, and they are imported as they are. Making them
public means changing a module whose code is pinned together with the answer cache's
version (``tests/test_answer_cache_version.py``), and this step changes nothing that
answers a question. Whoever renames one of them hears of it from the type checker and
from this package's tests.
"""

import time
from collections.abc import Sequence

from google import genai
from google.genai import errors, types

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.gemini import (
    _NETWORK_ERRORS,
    _TIMEOUT_ERRORS,
    _UNREADABLE_REPLY_ERRORS,
    _WITHHELD,
    GeminiGenerator,
    _answer_text,
    _stop_reason,
    _usage,
)
from handbook_rag.rewrite.models import MAX_REPLY_TOKENS, RewriteReply, Turn
from handbook_rag.rewrite.prompt import REWRITE_INSTRUCTIONS, build_rewrite_prompt


class GeminiRewriter:
    """Asks a Gemini model for a question that stands alone, with one request."""

    provider = GeminiGenerator.provider
    operation = GeminiGenerator.operation

    def __init__(
        self, client: genai.Client, *, model: str, max_tokens: int = MAX_REPLY_TOKENS
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        # The answer generator of the same model, kept for one thing: its reading of an
        # error from the API. It never sends a request from here.
        self._answers = GeminiGenerator(client, model=model, max_tokens=max_tokens)

    @property
    def model(self) -> str:
        return self._model

    def rewrite(self, turns: Sequence[Turn], question: str) -> RewriteReply:
        """Send the earlier questions and the new one. Returns the reply as it came."""
        # Built before the request, so that a mistake in our own settings is never
        # reported as a reply that could not be read.
        prompt = build_rewrite_prompt(turns, question)
        config = types.GenerateContentConfig(
            system_instruction=REWRITE_INSTRUCTIONS,
            max_output_tokens=self._max_tokens,
            # No follow-up requests by the SDK: every request is one the caller counts.
            automatic_continuation=False,
        )
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=config
            )
        except errors.APIError as error:
            raise self._answers._explain(error) from error
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
        seconds = time.perf_counter() - started

        candidate = response.candidates[0] if response.candidates else None
        finish = candidate.finish_reason if candidate is not None else None
        feedback = response.prompt_feedback
        blocked = feedback is not None and bool(feedback.block_reason)
        return RewriteReply(
            text=_answer_text(response),
            model=response.model_version or self._model,
            stop_reason=_stop_reason(finish, withheld=blocked or finish in _WITHHELD),
            usage=_usage(response.usage_metadata),
            seconds=seconds,
        )
