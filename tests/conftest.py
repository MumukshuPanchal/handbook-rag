"""Fixtures shared by the test suite."""

import os
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from handbook_rag.config import get_settings
from tests.support import Upstream, git


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make every test independent of the machine it runs on.

    Tests run in an empty directory, see none of the developer's application settings,
    and ignore the developer's git configuration. Without this, a local ``.env`` file or
    a global git option could make a test pass on one machine and fail on another.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    for name in os.environ:
        if name.startswith(("HANDBOOK_RAG_", "ANTHROPIC_", "GEMINI_", "GOOGLE_")):
            monkeypatch.delenv(name)
    get_settings.cache_clear()

    empty_git_config = tmp_path / "gitconfig"
    empty_git_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_git_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.com")


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Record the spans the code under test opens, without a trace backend.

    Without this fixture tracing is off, as it is by default: spans record nothing.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("handbook_rag.tracing._provider", provider)
    return exporter


@pytest.fixture
def upstream(tmp_path: Path) -> Upstream:
    repo = tmp_path / "upstream"
    pages = repo / "content" / "handbook"
    (pages / "people").mkdir(parents=True)
    (repo / "content" / "other").mkdir()

    git(repo, "init", "--quiet", "--initial-branch=main")
    # Let clients fetch one exact commit without file contents, as GitLab does.
    git(repo, "config", "uploadpack.allowFilter", "true")
    git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")

    (repo / "README.md").write_text("# Not part of the handbook\n")
    (pages / "_index.md").write_text("# Handbook\n\nWelcome.\n")
    (pages / "people" / "leave.md").write_text("# Leave\n\nTake 25 days.\n")
    (pages / "people" / "photo.png").write_bytes(b"\x89PNG not really an image")
    (repo / "content" / "other" / "skip.md").write_text("# Outside the handbook\n")
    (pages / "link.md").symlink_to("../../README.md")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message", "first")
    first_commit = git(repo, "rev-parse", "HEAD")

    (pages / "people" / "leave.md").write_text("# Leave\n\nTake 30 days.\n")
    (pages / "expenses.md").write_text("# Expenses\n\nKeep receipts.\n")
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message", "second")
    second_commit = git(repo, "rev-parse", "HEAD")

    return Upstream(url=repo.as_uri(), first_commit=first_commit, second_commit=second_commit)


@pytest.fixture
def monkeypatch_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The process environment as a dictionary whose changes are undone after the test."""
    environment = dict(os.environ)
    monkeypatch.setattr(os, "environ", environment)
    return environment
