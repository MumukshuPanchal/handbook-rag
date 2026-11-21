"""Turn the downloaded snapshot into documents and chunks, using the application settings."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from handbook_rag.config import Settings
from handbook_rag.ingest.chunker import ChunkingConfig, chunk_document
from handbook_rag.ingest.loader import load_documents
from handbook_rag.ingest.models import Chunk, Document
from handbook_rag.ingest.tokens import token_counter


class CorpusError(RuntimeError):
    """The snapshot is missing or does not match the settings."""


@dataclass(frozen=True, slots=True)
class Corpus:
    """Everything ingestion produces from one snapshot."""

    documents: list[Document]
    chunks: list[Chunk]
    skipped: Counter[str]
    chunking_version: str
    source_commit: str


def snapshot_dir(settings: Settings) -> Path:
    return settings.data_dir / "handbook"


def build_corpus(settings: Settings) -> Corpus:
    """Load, clean and chunk every page of the snapshot on disk."""
    manifest_path = snapshot_dir(settings) / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusError(
            f"No handbook snapshot found in {snapshot_dir(settings)}. "
            "Run `handbook-rag snapshot` first."
        )
    source_commit = json.loads(manifest_path.read_text(encoding="utf-8"))["commit"]
    if source_commit != settings.handbook_commit:
        raise CorpusError(
            f"The snapshot on disk is commit {source_commit[:12]}, but the settings pin "
            f"{settings.handbook_commit[:12]}. Run `handbook-rag snapshot` to update it."
        )

    loaded = load_documents(
        snapshot_dir(settings) / "repo" / settings.handbook_content_path.strip("/"),
        source_commit=source_commit,
        base_url=settings.handbook_base_url,
        min_chars=settings.min_page_chars,
    )
    config = ChunkingConfig(
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    count_tokens = token_counter(settings.embedding_model)
    chunks = [
        chunk
        for document in loaded.documents
        for chunk in chunk_document(document, count_tokens=count_tokens, config=config)
    ]
    return Corpus(
        documents=loaded.documents,
        chunks=chunks,
        skipped=loaded.skipped,
        chunking_version=config.version,
        source_commit=source_commit,
    )
