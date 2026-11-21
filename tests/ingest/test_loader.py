from pathlib import Path

import pytest

from handbook_rag.ingest.loader import LoadResult, load_documents, page_url

BASE_URL = "https://handbook.example.com/handbook/"
COMMIT = "a" * 40
LONG_ENOUGH = "This page has enough words in it to count as real content for a reader. " * 3


def write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def load(root: Path, min_chars: int = 50) -> LoadResult:
    return load_documents(root, source_commit=COMMIT, base_url=BASE_URL, min_chars=min_chars)


def test_loads_a_page_with_its_metadata(tmp_path: Path) -> None:
    write(
        tmp_path,
        "finance/expenses.md",
        f'---\ntitle: "Expenses"\n---\n\n## Limits\n\n{LONG_ENOUGH}',
    )

    result = load(tmp_path)

    (document,) = result.documents
    assert document.doc_id == "finance/expenses.md"
    assert document.title == "Expenses"
    assert document.section == "finance"
    assert document.url == "https://handbook.example.com/handbook/finance/expenses/"
    assert document.source_commit == COMMIT
    assert document.text.startswith("## Limits")
    assert "title:" not in document.text


@pytest.mark.parametrize(
    ("doc_id", "expected"),
    [
        ("finance/expenses.md", "https://handbook.example.com/handbook/finance/expenses/"),
        ("finance/_index.md", "https://handbook.example.com/handbook/finance/"),
        ("finance/tax/index.md", "https://handbook.example.com/handbook/finance/tax/"),
        ("_index.md", "https://handbook.example.com/handbook/"),
    ],
)
def test_page_url_follows_the_site_layout(doc_id: str, expected: str) -> None:
    assert page_url(BASE_URL, doc_id) == expected


def test_documents_come_back_in_a_stable_order(tmp_path: Path) -> None:
    for name in ("zebra.md", "alpha.md", "middle/_index.md"):
        write(tmp_path, name, LONG_ENOUGH)

    assert [d.doc_id for d in load(tmp_path).documents] == [
        "alpha.md",
        "middle/_index.md",
        "zebra.md",
    ]


def test_personal_pages_are_left_out(tmp_path: Path) -> None:
    write(tmp_path, "engineering/readmes/jane-doe.md", LONG_ENOUGH)
    write(tmp_path, "engineering/readme-guidelines.md", LONG_ENOUGH)

    result = load(tmp_path)

    assert [d.doc_id for d in result.documents] == ["engineering/readme-guidelines.md"]
    assert result.skipped["personal page"] == 1


def test_pages_with_almost_no_text_are_left_out(tmp_path: Path) -> None:
    write(tmp_path, "stub.md", '---\ntitle: "Stub"\n---\n\n{{< redirect-notice >}}\n')
    write(tmp_path, "real.md", LONG_ENOUGH)

    result = load(tmp_path)

    assert [d.doc_id for d in result.documents] == ["real.md"]
    assert result.skipped["too little text"] == 1


def test_title_falls_back_to_the_first_heading_then_the_file_name(tmp_path: Path) -> None:
    write(tmp_path, "a/with-heading.md", f"## Travel policy\n\n{LONG_ENOUGH}")
    write(tmp_path, "a/no-heading-at-all.md", LONG_ENOUGH)
    write(tmp_path, "team-onboarding/_index.md", LONG_ENOUGH)

    titles = {d.doc_id: d.title for d in load(tmp_path).documents}

    assert titles == {
        "a/with-heading.md": "Travel policy",
        "a/no-heading-at-all.md": "No heading at all",
        "team-onboarding/_index.md": "Team onboarding",
    }


def test_top_level_page_belongs_to_the_handbook_section(tmp_path: Path) -> None:
    write(tmp_path, "_index.md", LONG_ENOUGH)

    assert load(tmp_path).documents[0].section == "handbook"


def test_files_that_are_not_markdown_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "notes.txt", LONG_ENOUGH)
    write(tmp_path, "page.md", LONG_ENOUGH)

    assert [d.doc_id for d in load(tmp_path).documents] == ["page.md"]
