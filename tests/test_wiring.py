from pydantic import SecretStr

from handbook_rag.config import Settings
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.wiring import make_generator


def test_generator_uses_the_key_from_settings_without_exporting_it(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key"))

    generator = make_generator(settings)

    assert isinstance(generator, ClaudeGenerator)
    assert generator._client.api_key == "sk-ant-test-not-a-real-key"
    assert "ANTHROPIC_API_KEY" not in monkeypatch_environment


def test_generator_can_be_built_without_a_key() -> None:
    assert isinstance(make_generator(Settings()), ClaudeGenerator)


def test_key_is_read_from_the_standard_variable_and_never_shown(
    monkeypatch_environment: dict[str, str],
) -> None:
    monkeypatch_environment["ANTHROPIC_API_KEY"] = "sk-ant-test-not-a-real-key"

    settings = Settings()

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-not-a-real-key"
    assert "sk-ant-test" not in repr(settings)
