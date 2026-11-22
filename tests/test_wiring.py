import pytest
from pydantic import SecretStr

from handbook_rag.config import Settings
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.wiring import make_generator

CLAUDE = "claude-haiku-5-5"


def test_answers_come_from_gemini_unless_configured_otherwise(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"))

    generator = make_generator(settings)

    assert isinstance(generator, GeminiGenerator)
    assert generator._model == "gemini-3.5-flash-lite"
    assert generator._client._api_client.api_key == "test-key-not-a-real-one"
    # The key reaches the client directly, never through the process environment.
    assert "GEMINI_API_KEY" not in monkeypatch_environment
    assert "GOOGLE_API_KEY" not in monkeypatch_environment


def test_gemini_requests_have_a_time_limit_and_retry_only_what_a_retry_can_fix() -> None:
    generator = make_generator(Settings(gemini_api_key=SecretStr("test-key-not-a-real-one")))

    assert isinstance(generator, GeminiGenerator)
    options = generator._client._api_client._http_options
    assert options.timeout == 60_000  # milliseconds
    assert options.retry_options is not None
    assert options.retry_options.attempts == 3
    codes = options.retry_options.http_status_codes
    assert codes is not None
    assert 503 in codes
    # A rate limit on a free key lasts a minute or a day. Trying again at once only
    # spends another request, so it is reported instead.
    assert 429 not in codes


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_a_gemini_model_without_a_key_is_a_clear_error(key: SecretStr | None) -> None:
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        make_generator(Settings(gemini_api_key=key))


def test_an_empty_key_copied_from_the_example_file_is_a_clear_error(
    monkeypatch_environment: dict[str, str],
) -> None:
    monkeypatch_environment["GEMINI_API_KEY"] = ""

    with pytest.raises(GenerationError, match="No Gemini API key was found"):
        make_generator(Settings())


def test_a_claude_model_name_selects_the_claude_generator(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(
        answer_model=CLAUDE, anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key")
    )

    generator = make_generator(settings)

    assert isinstance(generator, ClaudeGenerator)
    assert generator._model == CLAUDE
    assert generator._client.api_key == "sk-ant-test-not-a-real-key"
    assert "ANTHROPIC_API_KEY" not in monkeypatch_environment


def test_claude_generator_can_be_built_without_a_key_in_the_settings() -> None:
    # The Claude SDK then looks in its own standard places, and a Gemini key is not needed.
    assert isinstance(make_generator(Settings(answer_model=CLAUDE)), ClaudeGenerator)


@pytest.mark.parametrize(
    ("variable", "field"),
    [("GEMINI_API_KEY", "gemini_api_key"), ("ANTHROPIC_API_KEY", "anthropic_api_key")],
)
def test_keys_are_read_from_their_standard_variables_and_never_shown(
    monkeypatch_environment: dict[str, str], variable: str, field: str
) -> None:
    monkeypatch_environment[variable] = "test-key-not-a-real-one"

    settings = Settings()

    key = getattr(settings, field)
    assert key is not None
    assert key.get_secret_value() == "test-key-not-a-real-one"
    assert "test-key-not-a-real-one" not in repr(settings)
    assert "test-key-not-a-real-one" not in str(settings.model_dump())
