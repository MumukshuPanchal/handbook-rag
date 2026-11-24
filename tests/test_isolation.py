"""Every test stays isolated from the machine when it undoes patches of its own."""

import os
from pathlib import Path

import pytest

from handbook_rag.config import get_settings


def test_undoing_a_tests_own_patches_leaves_it_in_its_empty_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A test may take back a patch of its own with monkeypatch.undo(). That must not
    # take back the isolation: in the project folder settings would read the developer's
    # .env, which holds a real API key.
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_MODEL", "gemini-for-this-test-only")
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    monkeypatch.undo()

    assert Path.cwd() == tmp_path / "workdir"
    assert not [
        name
        for name in os.environ
        if name.startswith(("HANDBOOK_RAG_", "ANTHROPIC_", "GEMINI_", "GOOGLE_"))
    ]
    assert os.environ["GIT_CONFIG_NOSYSTEM"] == "1"
    get_settings.cache_clear()
    assert get_settings().gemini_api_key is None
