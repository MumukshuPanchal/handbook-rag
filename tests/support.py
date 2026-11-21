"""Helpers shared by tests. Fixtures live in ``conftest.py``; plain helpers live here."""

import subprocess
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
