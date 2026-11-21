"""Helpers shared by tests. Fixtures live in ``conftest.py``; plain helpers live here."""

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return its output."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class Upstream:
    """A small stand-in for the handbook's git repository, with two commits."""

    url: str
    first_commit: str
    second_commit: str


def count_words(texts: Sequence[str]) -> list[int]:
    """A stand-in token counter for tests: one whitespace-separated word is one token."""
    return [len(text.split()) for text in texts]


def write_snapshot(data_dir: Path, commit: str, pages: dict[str, str]) -> None:
    """Create what ``handbook-rag snapshot`` leaves on disk, without git or the network."""
    content_dir = data_dir / "handbook" / "repo" / "content" / "handbook"
    for relative, text in pages.items():
        page = content_dir / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text)
    (data_dir / "handbook" / "manifest.json").write_text(json.dumps({"commit": commit}))
