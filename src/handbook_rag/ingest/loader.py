"""Read handbook pages from the snapshot and turn each into a clean ``Document``."""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from handbook_rag.ingest.cleaner import clean_markdown, split_front_matter
from handbook_rag.ingest.models import Document

_INDEX_FILES = {"_index.md", "index.md"}
# Personal pages written by and about individual team members.
_PERSONAL_PAGES_DIRECTORY = "readmes"
_FIRST_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class LoadResult:
    """The pages that were loaded, and a count of the ones left out by reason."""

    documents: list[Document]
    skipped: Counter[str] = field(default_factory=Counter)


def load_documents(
    content_dir: Path, *, source_commit: str, base_url: str, min_chars: int
) -> LoadResult:
    """Load every Markdown page under ``content_dir``, in a stable order.

    Two kinds of page are left out: personal pages about individual team members, and
    pages with almost no text once the markup is removed (redirects and empty indexes).
    """
    documents: list[Document] = []
    skipped: Counter[str] = Counter()
    for path in sorted(content_dir.rglob("*.md")):
        if path.is_symlink() or not path.is_file():
            continue
        doc_id = path.relative_to(content_dir).as_posix()
        if _PERSONAL_PAGES_DIRECTORY in PurePosixPath(doc_id).parts[:-1]:
            skipped["personal page"] += 1
            continue

        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, body = split_front_matter(raw)
        text = clean_markdown(body)
        if len(text) < min_chars:
            skipped["too little text"] += 1
            continue

        documents.append(
            Document(
                doc_id=doc_id,
                title=_title(front_matter.get("title"), text, doc_id),
                url=page_url(base_url, doc_id),
                section=_section(doc_id),
                source_commit=source_commit,
                text=text,
            )
        )
    return LoadResult(documents=documents, skipped=skipped)


def page_url(base_url: str, doc_id: str) -> str:
    """The public address of a page, following the website generator's rules.

    ``finance/expenses.md`` is served at ``finance/expenses/``, and an index file is
    served at its directory: ``finance/_index.md`` at ``finance/``.
    """
    path = PurePosixPath(doc_id)
    page = path.parent if path.name in _INDEX_FILES else path.with_suffix("")
    suffix = "" if page.as_posix() == "." else f"{page.as_posix()}/"
    return f"{base_url.rstrip('/')}/{suffix}"


def _section(doc_id: str) -> str:
    parts = PurePosixPath(doc_id).parts
    return parts[0] if len(parts) > 1 else "handbook"


def _title(front_matter_title: object, text: str, doc_id: str) -> str:
    """Prefer the title in the front matter, then the first heading, then the file name."""
    if isinstance(front_matter_title, str) and front_matter_title.strip():
        return front_matter_title.strip()
    heading = _FIRST_HEADING.search(text)
    if heading:
        return heading.group(1).strip()
    path = PurePosixPath(doc_id)
    name = path.parent.name if path.name in _INDEX_FILES else path.stem
    return name.replace("-", " ").replace("_", " ").strip().capitalize() or "Handbook"
