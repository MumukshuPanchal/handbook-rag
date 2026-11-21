from pathlib import Path

import pytest

from handbook_rag.config import Settings
from handbook_rag.ingest import build
from handbook_rag.ingest.build import CorpusError, build_corpus
from tests.support import count_words, write_snapshot

COMMIT = "a" * 40
PAGE = "## Limits\n\nYou can expense a laptop up to one thousand dollars with approval."


@pytest.fixture(autouse=True)
def word_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count words instead of downloading the embedding model's tokenizer."""
    monkeypatch.setattr(build, "token_counter", lambda _model: count_words)


def settings_for(data_dir: Path, commit: str = COMMIT) -> Settings:
    return Settings(data_dir=data_dir, handbook_commit=commit, min_page_chars=20)


def test_builds_documents_and_chunks_from_the_snapshot(tmp_path: Path) -> None:
    write_snapshot(
        tmp_path,
        COMMIT,
        {
            "finance/expenses.md": f'---\ntitle: "Expenses"\n---\n\n{PAGE}',
            "people/readmes/jane.md": PAGE,
            "stub.md": "Moved.",
        },
    )

    corpus = build_corpus(settings_for(tmp_path))

    assert [d.doc_id for d in corpus.documents] == ["finance/expenses.md"]
    assert corpus.skipped == {"personal page": 1, "too little text": 1}
    assert corpus.source_commit == COMMIT
    assert corpus.chunking_version == "v1-max320-min60-overlap40"
    (chunk,) = corpus.chunks
    assert chunk.doc_id == "finance/expenses.md"
    assert chunk.source_commit == COMMIT
    assert chunk.context_header == "Expenses > Limits"


def test_missing_snapshot_is_reported_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="handbook-rag snapshot"):
        build_corpus(settings_for(tmp_path))


def test_snapshot_of_a_different_commit_is_rejected(tmp_path: Path) -> None:
    write_snapshot(tmp_path, "b" * 40, {"page.md": PAGE})

    with pytest.raises(CorpusError, match="settings pin"):
        build_corpus(settings_for(tmp_path))
