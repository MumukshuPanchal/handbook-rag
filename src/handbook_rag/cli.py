"""Command-line interface. Run ``handbook-rag --help`` to list the commands."""

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import typer

from handbook_rag import __version__
from handbook_rag.config import get_settings
from handbook_rag.ingest.build import CorpusError, build_corpus
from handbook_rag.ingest.snapshot import SnapshotError, download_snapshot
from handbook_rag.ingest.tokens import token_counter

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


@app.command()
def chunks(
    export: Annotated[
        Path | None,
        typer.Option(help="Also write every chunk to this file, one JSON object per line."),
    ] = None,
) -> None:
    """Clean and chunk the downloaded handbook, and print a summary."""
    settings = get_settings()
    started = time.perf_counter()
    try:
        corpus = build_corpus(settings)
    except CorpusError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    elapsed = time.perf_counter() - started

    sizes = sorted(chunk.token_count for chunk in corpus.chunks)
    if not sizes:
        typer.echo("error: the snapshot produced no chunks.", err=True)
        raise typer.Exit(code=1)

    def percentile(fraction: float) -> int:
        return sizes[round(fraction * (len(sizes) - 1))]

    # What the embedding model receives is the chunk plus its context header, plus two
    # special tokens. Anything beyond the model's limit would be silently cut off.
    embedded_sizes = token_counter(settings.embedding_model)(
        [chunk.text_for_embedding for chunk in corpus.chunks]
    )
    too_long = sum(size + 2 > settings.embedding_max_tokens for size in embedded_sizes)
    skipped = ", ".join(f"{count:,} {reason}" for reason, count in sorted(corpus.skipped.items()))

    typer.echo("Handbook chunks")
    typer.echo(f"  commit    {corpus.source_commit[:12]}")
    typer.echo(f"  pages     {len(corpus.documents):,} loaded; skipped: {skipped or 'none'}")
    typer.echo(f"  chunks    {len(sizes):,} ({corpus.chunking_version})")
    typer.echo(
        f"  tokens    median {percentile(0.5)}, 90th percentile {percentile(0.9)}, "
        f"99th percentile {percentile(0.99)}, largest {sizes[-1]}"
    )
    typer.echo(f"  total     {sum(sizes):,} tokens")
    typer.echo(
        f"  too long  {too_long:,} chunks exceed the embedding model's "
        f"{settings.embedding_max_tokens}-token limit"
    )
    typer.echo(f"  time      {elapsed:.1f} seconds")

    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        with export.open("w", encoding="utf-8") as output:
            for chunk in corpus.chunks:
                output.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False) + "\n")
        typer.echo(f"  exported  {export}")
