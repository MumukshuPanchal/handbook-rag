import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from handbook_rag.config import Settings, get_settings


def test_defaults_work_without_any_configuration() -> None:
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.log_level == "INFO"


def test_environment_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", "/tmp/handbook")
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.data_dir == Path("/tmp/handbook")
    assert settings.log_level == "DEBUG"


def test_dotenv_file_is_read() -> None:
    Path(".env").write_text("HANDBOOK_RAG_LOG_LEVEL=WARNING\n")

    assert Settings().log_level == "WARNING"


def test_environment_variable_wins_over_dotenv_file(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text("HANDBOOK_RAG_LOG_LEVEL=WARNING\n")
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "ERROR")

    assert Settings().log_level == "ERROR"


def test_unrelated_keys_in_dotenv_file_are_ignored() -> None:
    Path(".env").write_text("SOME_OTHER_TOOL_KEY=value\nHANDBOOK_RAG_LOG_LEVEL=DEBUG\n")

    assert Settings().log_level == "DEBUG"


def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_returns_the_same_object_each_time() -> None:
    assert get_settings() is get_settings()


def test_handbook_is_pinned_to_a_full_commit_hash_by_default() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", Settings().handbook_commit)


def test_collection_name_identifies_corpus_model_and_chunking() -> None:
    assert Settings().collection_name == (
        "handbook-e645aac6-baai-bge-small-en-v1-5-qdrant-bm25-v1-max320-min60-overlap40"
    )


def test_collection_name_changes_with_any_of_its_parts() -> None:
    default = Settings().collection_name

    assert Settings(handbook_commit="b" * 40).collection_name != default
    assert Settings(embedding_model="BAAI/bge-base-en-v1.5").collection_name != default
    assert Settings(chunk_max_tokens=200).collection_name != default
    assert Settings(sparse_model="Qdrant/bm42").collection_name != default


def test_search_is_hybrid_and_not_reranked_unless_configured() -> None:
    from handbook_rag.config import Settings

    settings = Settings()

    assert settings.retrieval_mode == "hybrid"
    assert settings.rerank is False


def test_answers_are_written_by_an_exact_gemini_model_by_default() -> None:
    model = Settings().answer_model

    assert model.startswith("gemini-")
    # An alias such as gemini-flash-latest would change models without notice.
    assert "latest" not in model


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "claude-haiku-5-5"])
def test_answer_model_may_be_a_gemini_or_a_claude_model(model: str) -> None:
    assert Settings(answer_model=model).answer_model == model


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "",
        "Gemini-3.8-flash",
        "models/gemini-3.8-flash",
        "geminiX",  # the dash is part of the name
        "gemini-",  # a provider and no model
        "claude-",
    ],
)
def test_answer_model_from_an_unknown_provider_is_rejected(model: str) -> None:
    with pytest.raises(ValidationError, match=r"must be a Gemini model .* or a Claude model"):
        Settings(answer_model=model)


@pytest.mark.parametrize("model", ["gemini-flash-latest", "gemini-flash-lite-latest"])
def test_answer_model_must_be_an_exact_name_not_an_alias(model: str) -> None:
    with pytest.raises(ValidationError, match="must be an exact model name"):
        Settings(answer_model=model)


def test_the_model_is_given_five_chunks_by_default() -> None:
    assert Settings().answer_context_chunks == 5


def test_an_evaluation_sends_ten_requests_a_minute_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().eval_requests_per_minute == 10

    monkeypatch.setenv("HANDBOOK_RAG_EVAL_REQUESTS_PER_MINUTE", "4.5")

    assert Settings().eval_requests_per_minute == 4.5


@pytest.mark.parametrize("rate", [0, -1, 601])
def test_the_evaluation_pace_must_be_a_usable_rate(rate: float) -> None:
    with pytest.raises(ValidationError):
        Settings(eval_requests_per_minute=rate)
