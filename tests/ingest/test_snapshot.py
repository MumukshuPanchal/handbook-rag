import json
from pathlib import Path

import pytest

from handbook_rag.ingest.snapshot import (
    Snapshot,
    SnapshotError,
    download_snapshot,
    snapshot_commit,
)
from tests.support import Upstream, git


def download(upstream: Upstream, dest: Path, commit: str | None = None) -> Snapshot:
    return download_snapshot(
        repo_url=upstream.url,
        commit=commit or upstream.first_commit,
        content_path="content/handbook",
        licence="CC-BY-SA-4.0",
        dest=dest,
    )


def markdown_files(snapshot: Snapshot) -> list[str]:
    return sorted(
        p.relative_to(snapshot.content_dir).as_posix() for p in snapshot.content_dir.rglob("*")
    )


def test_downloads_only_markdown_under_the_content_path(upstream: Upstream, tmp_path: Path) -> None:
    snapshot = download(upstream, tmp_path / "data")

    assert markdown_files(snapshot) == ["_index.md", "link.md", "people", "people/leave.md"]
    repo_dir = tmp_path / "data" / "repo"
    assert not (repo_dir / "README.md").exists()
    assert not (repo_dir / "content" / "other").exists()


def test_snapshot_matches_the_pinned_commit_not_the_latest(
    upstream: Upstream, tmp_path: Path
) -> None:
    snapshot = download(upstream, tmp_path / "data", commit=upstream.first_commit)

    assert "25 days" in (snapshot.content_dir / "people" / "leave.md").read_text()
    assert not (snapshot.content_dir / "expenses.md").exists()


def test_manifest_describes_the_snapshot(upstream: Upstream, tmp_path: Path) -> None:
    snapshot = download(upstream, tmp_path / "data")

    manifest = json.loads(snapshot.manifest_path.read_text())
    sizes = [p.stat().st_size for p in snapshot.content_dir.rglob("*.md")]
    assert manifest["source_url"] == upstream.url
    assert manifest["commit"] == upstream.first_commit
    assert manifest["content_path"] == "content/handbook"
    assert manifest["licence"] == "CC-BY-SA-4.0"
    assert manifest["file_count"] == 3
    assert manifest["total_bytes"] == sum(sizes)
    assert len(manifest["content_sha256"]) == 64
    assert manifest["commit_date"].startswith("20")


def test_running_twice_changes_nothing(upstream: Upstream, tmp_path: Path) -> None:
    first = download(upstream, tmp_path / "data")
    manifest_text = first.manifest_path.read_text()
    modified_times = {p: p.stat().st_mtime_ns for p in first.content_dir.rglob("*.md")}

    second = download(upstream, tmp_path / "data")

    assert second.manifest == first.manifest
    assert second.manifest_path.read_text() == manifest_text
    assert {p: p.stat().st_mtime_ns for p in second.content_dir.rglob("*.md")} == modified_times


def test_local_edits_and_stray_files_are_discarded(upstream: Upstream, tmp_path: Path) -> None:
    first = download(upstream, tmp_path / "data")
    (first.content_dir / "people" / "leave.md").write_text("tampered")
    (first.content_dir / "stray.md").write_text("should not be here")

    second = download(upstream, tmp_path / "data")

    assert "25 days" in (second.content_dir / "people" / "leave.md").read_text()
    assert not (second.content_dir / "stray.md").exists()
    assert second.manifest == first.manifest


def test_changing_the_pinned_commit_updates_the_snapshot(
    upstream: Upstream, tmp_path: Path
) -> None:
    first = download(upstream, tmp_path / "data", commit=upstream.first_commit)

    second = download(upstream, tmp_path / "data", commit=upstream.second_commit)

    assert "30 days" in (second.content_dir / "people" / "leave.md").read_text()
    assert (second.content_dir / "expenses.md").exists()
    assert second.manifest.commit == upstream.second_commit
    assert second.manifest.file_count == first.manifest.file_count + 1
    assert second.manifest.content_sha256 != first.manifest.content_sha256


def test_symbolic_links_are_checked_out_as_plain_files(upstream: Upstream, tmp_path: Path) -> None:
    snapshot = download(upstream, tmp_path / "data")

    link = snapshot.content_dir / "link.md"
    assert not link.is_symlink()
    assert link.read_text() == "../../README.md"


@pytest.mark.parametrize("pin", ["main", "e645aac", "HEAD"])
def test_rejects_anything_but_a_full_commit_hash(
    upstream: Upstream, tmp_path: Path, pin: str
) -> None:
    with pytest.raises(SnapshotError, match="full commit hash"):
        download(upstream, tmp_path / "data", commit=pin)


def test_unknown_commit_is_reported(upstream: Upstream, tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="git fetch failed"):
        download(upstream, tmp_path / "data", commit="0" * 40)


def test_missing_content_path_is_reported(upstream: Upstream, tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="does not exist"):
        download_snapshot(
            repo_url=upstream.url,
            commit=upstream.first_commit,
            content_path="content/missing",
            licence="CC-BY-SA-4.0",
            dest=tmp_path / "data",
        )


def test_refuses_to_modify_a_repository_it_did_not_create(
    upstream: Upstream, tmp_path: Path
) -> None:
    someone_elses_repo = tmp_path / "data" / "repo"
    someone_elses_repo.mkdir(parents=True)
    git(someone_elses_repo, "init", "--quiet")
    precious = someone_elses_repo / "work.txt"
    precious.write_text("uncommitted work")

    with pytest.raises(SnapshotError, match="did not create"):
        download(upstream, tmp_path / "data")

    assert precious.read_text() == "uncommitted work"


def test_refuses_to_overwrite_a_non_empty_directory(upstream: Upstream, tmp_path: Path) -> None:
    occupied = tmp_path / "data" / "repo"
    occupied.mkdir(parents=True)
    (occupied / "notes.txt").write_text("keep me")

    with pytest.raises(SnapshotError, match="not empty"):
        download(upstream, tmp_path / "data")

    assert (occupied / "notes.txt").read_text() == "keep me"


def test_the_commit_of_a_snapshot_on_disk_can_be_read(upstream: Upstream, tmp_path: Path) -> None:
    assert snapshot_commit(tmp_path / "data") is None

    download(upstream, tmp_path / "data", commit=upstream.second_commit)

    assert snapshot_commit(tmp_path / "data") == upstream.second_commit


NAMES_NO_COMMIT = [None, '{"commit": null}', "not json", "[]", '{"commit": 5}', '{"commit": ""}']


@pytest.mark.parametrize("manifest", NAMES_NO_COMMIT)
def test_a_snapshot_whose_manifest_names_no_commit_is_known_by_its_clone(
    upstream: Upstream, tmp_path: Path, manifest: str | None
) -> None:
    download(upstream, tmp_path / "data")
    manifest_path = tmp_path / "data" / "manifest.json"
    if manifest is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(manifest)

    assert snapshot_commit(tmp_path / "data") == upstream.first_commit


@pytest.mark.parametrize("text", ["not json", "[]", '{"commit": 5}', '{"commit": ""}'])
def test_a_manifest_that_cannot_be_read_names_no_commit_and_is_no_error(
    tmp_path: Path, text: str
) -> None:
    # With no clone beside it there is no snapshot to protect, and the next download
    # writes the manifest again. An error here would stop every download for good.
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.json").write_text(text)

    assert snapshot_commit(tmp_path / "data") is None


def test_a_replacement_whose_download_fails_keeps_the_manifest(
    upstream: Upstream, tmp_path: Path
) -> None:
    # The manifest goes only when the files are about to change, not before the download,
    # which a network can fail.
    dest = tmp_path / "data"
    download(upstream, dest)
    manifest = (dest / "manifest.json").read_text()

    with pytest.raises(SnapshotError, match="git fetch failed"):
        download_snapshot(
            repo_url=str(tmp_path / "nowhere"),
            commit=upstream.second_commit,
            content_path="content/handbook",
            licence="CC-BY-SA-4.0",
            dest=dest,
        )

    assert (dest / "manifest.json").read_text() == manifest


def test_a_folder_without_a_clone_of_its_own_has_no_commit_even_inside_a_repository(
    tmp_path: Path,
) -> None:
    # git asked inside such a folder would answer with the surrounding repository's commit.
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "commit", "--quiet", "--allow-empty", "--message", "outer")
    (tmp_path / "data" / "repo").mkdir(parents=True)

    assert snapshot_commit(tmp_path / "data") is None


def test_a_replacement_that_fails_leaves_no_manifest_of_the_old_commit(
    upstream: Upstream, tmp_path: Path
) -> None:
    dest = tmp_path / "data"
    download(upstream, dest)

    with pytest.raises(SnapshotError, match="does not exist"):
        download_snapshot(
            repo_url=upstream.url,
            commit=upstream.second_commit,
            content_path="content/missing",
            licence="CC-BY-SA-4.0",
            dest=dest,
        )

    assert not (dest / "manifest.json").exists()
    assert snapshot_commit(dest) == upstream.second_commit
