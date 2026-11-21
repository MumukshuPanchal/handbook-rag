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
