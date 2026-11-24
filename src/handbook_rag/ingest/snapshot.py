"""Download a pinned snapshot of the handbook.

The corpus is not stored in this repository. This module fetches the Markdown files of
one directory of the handbook's git repository at one exact commit, so everyone who
runs the pipeline indexes the same text.

Only the files we need are downloaded. A *partial clone* skips file contents until they
are checked out, and a *sparse checkout* limits the checkout to Markdown files under the
content directory.
"""

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 900
_FULL_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
# Written inside the clone's .git directory. Its presence proves this tool created the
# clone, so it is safe to overwrite files in it.
_OWNERSHIP_MARKER = "handbook-rag-snapshot"


class SnapshotError(RuntimeError):
    """The snapshot could not be downloaded or verified."""


@dataclass(frozen=True)
class SnapshotManifest:
    """Where the snapshot came from and what it contains.

    The manifest holds no timestamps or machine-specific paths, so downloading the same
    commit twice produces an identical file.
    """

    source_url: str
    commit: str
    commit_date: str
    content_path: str
    licence: str
    file_count: int
    total_bytes: int
    content_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class Snapshot:
    """A downloaded snapshot on disk."""

    content_dir: Path
    manifest_path: Path
    manifest: SnapshotManifest


def download_snapshot(
    *, repo_url: str, commit: str, content_path: str, licence: str, dest: Path
) -> Snapshot:
    """Fetch the Markdown files under ``content_path`` at ``commit`` into ``dest``.

    Running it again with the same arguments changes nothing. Running it with a different
    commit updates the files in place. Local edits inside the snapshot are discarded, so
    the result always matches the pinned commit exactly.
    """
    if not _FULL_COMMIT_HASH.fullmatch(commit):
        raise SnapshotError(
            f"{commit!r} is not a full commit hash. Pin the snapshot to a 40-character "
            "commit hash, not a branch or tag, so the text cannot change underneath us."
        )
    content_path = content_path.strip("/")
    repo_dir = dest / "repo"

    _prepare_clone(repo_dir, repo_url, content_path)
    if _current_commit(repo_dir) != commit:
        _git(repo_dir, "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    # The manifest describes the files. From here on they may change, so it goes, and a
    # run that fails before writing the new one leaves no manifest naming the old commit
    # beside files of the new one.
    manifest_path = dest / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    _git(repo_dir, "checkout", "--quiet", "--force", "--detach", commit)
    _git(repo_dir, "clean", "--quiet", "-fdx")

    content_dir = repo_dir / content_path
    if not content_dir.is_dir():
        raise SnapshotError(f"Directory {content_path!r} does not exist at commit {commit}.")

    manifest = _build_manifest(
        content_dir,
        source_url=repo_url,
        commit=commit,
        commit_date=_git(repo_dir, "show", "--no-patch", "--format=%cI", commit).strip(),
        content_path=content_path,
        licence=licence,
    )
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return Snapshot(content_dir=content_dir, manifest_path=manifest_path, manifest=manifest)


def snapshot_commit(dest: Path) -> str | None:
    """The commit of the snapshot already in ``dest``, or ``None`` when there is none.

    The manifest says it. When there is no manifest, or it names no commit, the files
    may still be there, for example after a run that failed halfway, so the clone is
    asked which commit it has checked out.
    """
    manifest_path = dest / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SnapshotError(f"{manifest_path} cannot be read: {error}") from error
        if not isinstance(manifest, dict):
            raise SnapshotError(f"{manifest_path} cannot be read: it is not a JSON object.")
        commit = manifest.get("commit")
        if isinstance(commit, str):
            return commit
        if commit is not None:
            raise SnapshotError(
                f"{manifest_path} cannot be read: its commit is {commit!r}, not a hash."
            )
    repo_dir = dest / "repo"
    # Only a clone of its own is asked. A folder without one could lie inside another
    # repository, whose commit git would report instead.
    if not (repo_dir / ".git").is_dir():
        return None
    return _current_commit(repo_dir)


def _prepare_clone(repo_dir: Path, repo_url: str, content_path: str) -> None:
    """Create the local clone if needed and configure what it checks out."""
    git_dir = repo_dir / ".git"
    marker = git_dir / _OWNERSHIP_MARKER

    if git_dir.is_dir():
        if not marker.is_file():
            raise SnapshotError(
                f"{repo_dir} is a git repository this tool did not create. Refusing to "
                "modify it. Point the data directory somewhere else."
            )
        _git(repo_dir, "remote", "set-url", "origin", repo_url)
    else:
        if repo_dir.exists() and any(repo_dir.iterdir()):
            raise SnapshotError(f"{repo_dir} exists and is not empty. Refusing to overwrite it.")
        repo_dir.mkdir(parents=True, exist_ok=True)
        _git(repo_dir, "init", "--quiet")
        _git(repo_dir, "remote", "add", "origin", repo_url)
        marker.write_text("Created by handbook-rag. Safe to delete with its parent.\n")

    settings = {
        # Keep file bytes identical on every platform.
        "core.autocrlf": "false",
        # Check symbolic links out as plain text files, so a link in the upstream
        # repository can never make us read a file from outside the snapshot.
        "core.symlinks": "false",
        "advice.detachedHead": "false",
    }
    for key, value in settings.items():
        _git(repo_dir, "config", key, value)
    _git(repo_dir, "sparse-checkout", "set", "--no-cone", f"/{content_path}/**/*.md")


def _current_commit(repo_dir: Path) -> str | None:
    result = _run_git(repo_dir, "rev-parse", "--verify", "--quiet", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else None


def _build_manifest(
    content_dir: Path,
    *,
    source_url: str,
    commit: str,
    commit_date: str,
    content_path: str,
    licence: str,
) -> SnapshotManifest:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(content_dir.rglob("*.md")):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        relative = path.relative_to(content_dir).as_posix()
        digest.update(f"{relative}\t{hashlib.sha256(data).hexdigest()}\n".encode())
        file_count += 1
        total_bytes += len(data)
    if file_count == 0:
        raise SnapshotError(f"No Markdown files found under {content_path!r} at commit {commit}.")
    return SnapshotManifest(
        source_url=source_url,
        commit=commit,
        commit_date=commit_date,
        content_path=content_path,
        licence=licence,
        file_count=file_count,
        total_bytes=total_bytes,
        content_sha256=digest.hexdigest(),
    )


def _git(repo_dir: Path, *args: str) -> str:
    """Run a git command in the clone and return its output, or raise a clear error."""
    result = _run_git(repo_dir, *args)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["no error output"]
        raise SnapshotError(f"git {args[0]} failed: {detail[0]}")
    return result.stdout


def _run_git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Never wait for a password prompt: the repository is public, so a prompt means
    # something is wrong and the command should fail instead of hanging.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except FileNotFoundError as error:
        raise SnapshotError("git is not installed or not on PATH.") from error
    except subprocess.TimeoutExpired as error:
        raise SnapshotError(
            f"git {args[0]} timed out after {_GIT_TIMEOUT_SECONDS} seconds."
        ) from error
