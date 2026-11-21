"""Command-line interface. Run ``handbook-rag --help`` to list the commands."""

from typing import Annotated

import typer

from handbook_rag import __version__
from handbook_rag.config import get_settings
from handbook_rag.ingest.snapshot import SnapshotError, download_snapshot

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"handbook-rag {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """A knowledge assistant that answers questions from a company handbook."""


@app.command()
def snapshot() -> None:
    """Download the handbook at the pinned commit into the data directory."""
    settings = get_settings()
    try:
        result = download_snapshot(
            repo_url=settings.handbook_repo_url,
            commit=settings.handbook_commit,
            content_path=settings.handbook_content_path,
            licence=settings.handbook_licence,
            dest=settings.data_dir / "handbook",
        )
    except SnapshotError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    manifest = result.manifest
    typer.echo("Handbook snapshot ready")
    typer.echo(f"  source    {manifest.source_url}")
    typer.echo(f"  commit    {manifest.commit[:12]} ({manifest.commit_date[:10]})")
    typer.echo(f"  licence   {manifest.licence}")
    typer.echo(
        f"  files     {manifest.file_count:,} Markdown files, "
        f"{manifest.total_bytes / 1_000_000:.1f} MB"
    )
    typer.echo(f"  location  {result.content_dir}")
    typer.echo(f"  manifest  {result.manifest_path}")
