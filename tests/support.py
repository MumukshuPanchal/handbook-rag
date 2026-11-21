"""Helpers shared by tests. Fixtures live in ``conftest.py``; plain helpers live here."""

import json
import math
import re
import subprocess
import uuid
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from handbook_rag.index.sparse import SparseVector
from handbook_rag.ingest.models import Chunk


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


class FakeEmbedder:
    """A stand-in embedding model for tests: fast, deterministic, and needs no download.

    Each word is hashed to one of a few dimensions, so two texts that share words get
    similar vectors. That is enough to test indexing and retrieval logic.
    """

    name = "fake-embedder"
    dimension = 64
    device = "cpu"
    max_tokens = 512

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimension] += 1.0
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]


class FakeSparseEncoder:
    """A stand-in keyword encoder for tests: one position per distinct word, weight 1."""

    name = "fake-sparse"

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self.encode_query(text) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        words = sorted(set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "a", "of", "is"})
        return SparseVector([zlib.crc32(word.encode()) for word in words], [1.0] * len(words))


def make_chunk(
    text: str,
    *,
    doc_id: str = "people/leave.md",
    section: str = "people",
    title: str = "Leave",
    heading_path: tuple[str, ...] = (),
) -> Chunk:
    """A chunk with a valid id derived from its content, for tests that need one."""
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}|{text}")),
        doc_id=doc_id,
        title=title,
        url=f"https://handbook.example.com/handbook/{doc_id.removesuffix('.md')}/",
        section=section,
        source_commit="a" * 40,
        heading_path=heading_path,
        start_char=0,
        end_char=len(text),
        text=text,
        token_count=len(text.split()),
    )
