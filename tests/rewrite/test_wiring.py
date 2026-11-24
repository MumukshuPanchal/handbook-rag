"""How the rewriting step's model is built from the settings (``wiring.make_rewriter``).

No request is sent: building a client sends nothing.
"""

import anthropic
import pytest
from pydantic import SecretStr

from handbook_rag.config import Settings
from handbook_rag.generate.base import GenerationError
from handbook_rag.rewrite.claude import ClaudeRewriter
from handbook_rag.rewrite.gemini import GeminiRewriter
from handbook_rag.wiring import EVALUATION_TIMEOUT_SECONDS, make_rewriter


def test_a_gemini_answer_model_rewrites_with_one_attempt_and_the_evaluations_time_limit(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key-not-a-real-one"), answer_timeout_seconds=5.0
    )

    rewriter = make_rewriter(settings)

    assert isinstance(rewriter, GeminiRewriter)
    assert rewriter.model == settings.answer_model
    client = rewriter._client._api_client
    assert client.api_key == "test-key-not-a-real-one"
    options = client._http_options
    # The time limit of an evaluation, not the one of an answer over HTTP.
    assert options.timeout == EVALUATION_TIMEOUT_SECONDS * 1000
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1
    # The key reaches the client directly, never through the process environment.
    assert "GEMINI_API_KEY" not in monkeypatch_environment


def test_a_gemini_answer_model_without_its_key_is_refused_before_anything_is_sent() -> None:
    with pytest.raises(GenerationError, match="No Gemini API key was found") as refused:
        make_rewriter(Settings())

    assert refused.value.reason == "configuration"


def test_a_claude_answer_model_rewrites_with_no_repeat_and_no_span_of_the_sdks_own() -> None:
    settings = Settings(
        answer_model="claude-haiku-5-5",
        anthropic_api_key=SecretStr("test-key-not-a-real-one"),
        answer_effort="medium",
    )

    rewriter = make_rewriter(settings)

    assert isinstance(rewriter, ClaudeRewriter)
    assert rewriter.model == "claude-haiku-5-5"
    assert rewriter._effort == "medium"
    client = rewriter._client
    assert isinstance(client, anthropic.Anthropic)
    assert (client.timeout, client.max_retries) == (EVALUATION_TIMEOUT_SECONDS, 0)
    assert client.api_key == "test-key-not-a-real-one"
    # The step opens a span for its request itself, so the SDK's would count it twice.
    assert client._open_telemetry["traces"]["enabled"] is False


def test_a_claude_answer_model_may_find_its_credential_in_the_sdks_own_places() -> None:
    # As for answers, a missing Claude key is found by the first request, not here.
    rewriter = make_rewriter(Settings(answer_model="claude-haiku-5-5"))

    assert isinstance(rewriter, ClaudeRewriter)
