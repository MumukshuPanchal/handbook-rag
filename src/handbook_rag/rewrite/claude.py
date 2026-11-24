"""Ask Claude to rewrite a question: one small request that is not streamed.

**This path has never been run.** Like the Claude answer generator, it is tested against
the SDK's own types only (ADR 0018).

The answer generator reads the API's failures inside its one method
(``generate/claude.py``), so that reading cannot be called from here, and it is written
out again below, clause for clause and in the same order. A test puts every failure to
both and asks for the same message, the same reason and the same wait, so the two cannot
drift apart unseen. What can be shared is used as it is: which refusals are worth another
try, the sentence for a request that did not answer in time, and how the wait of a rate
limit is read. Those names are private to that module, which is pinned together with the
answer cache's version (``rewrite/gemini.py`` says why they are imported and not made
public).

The generator also catches the HTTP library's own errors, which a stream lets through
once it is open. A request that is not streamed should not meet them, and they are
caught here all the same, in the same words, so that the two read every failure alike.
"""

import time
from collections.abc import Sequence

import anthropic
import httpx2

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.claude import (
    _NOT_IN_TIME,
    _WORTH_REPEATING,
    ClaudeGenerator,
    Effort,
    _retry_after,
)
from handbook_rag.generate.models import TokenUsage
from handbook_rag.rewrite.models import MAX_REPLY_TOKENS, RewriteReply, Turn
from handbook_rag.rewrite.prompt import REWRITE_INSTRUCTIONS, build_rewrite_prompt


class ClaudeRewriter:
    """Asks a Claude model for a question that stands alone, with one request."""

    provider = ClaudeGenerator.provider
    operation = ClaudeGenerator.operation

    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str,
        effort: Effort,
        max_tokens: int = MAX_REPLY_TOKENS,
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def rewrite(self, turns: Sequence[Turn], question: str) -> RewriteReply:
        """Send the earlier questions and the new one. Returns the reply as it came."""
        prompt = build_rewrite_prompt(turns, question)
        started = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=REWRITE_INSTRUCTIONS,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
            )
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
        seconds = time.perf_counter() - started

        return RewriteReply(
            # Thinking blocks carry no part of the reply.
            text="".join(block.text for block in message.content if block.type == "text"),
            model=message.model,
            stop_reason=message.stop_reason or "unknown",
            usage=TokenUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
            ),
            seconds=seconds,
        )
