"""Command-line interface. Run ``handbook-rag --help`` to list the commands."""

import dataclasses
import datetime
import functools
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, TextIO

import typer

from handbook_rag import __version__, wiring
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.store import QdrantAnswerStore, answer_collections
from handbook_rag.cache.verify import Verification, VerifierError
from handbook_rag.config import Settings, get_settings
from handbook_rag.evals import adversarial, cache_pairs, cache_replies, follow_ups, limit_questions
from handbook_rag.evals import answers as answer_runs
from handbook_rag.evals.agreement import measure_agreement
from handbook_rag.evals.cache_report import report_lines
from handbook_rag.evals.careful import Pacer
from handbook_rag.evals.compare import (
    ComparisonError,
    compare_runs,
    metric_labels,
    overview_of_runs,
    results_table,
)
from handbook_rag.evals.correctness import CORRECT
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion, load_golden_set
from handbook_rag.evals.gate import (
    GateError,
    evidence_pages,
    gate_problems,
    load_gate_config,
    select_pages,
    shown_with_minimum,
)
from handbook_rag.evals.golden import check_against_corpus, sample_source_chunks
from handbook_rag.evals.judge import Judgement
from handbook_rag.evals.judging import (
    UNCLEAR,
    Criterion,
    Judge,
    JudgedCase,
    JudgeRun,
    agreement_headline,
    judge_check_stage,
    judge_record,
    judge_version,
    judgement_row,
    labelled_pairs,
    load_judgements,
    run_judge,
    saved_verdicts,
)
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.retrieval import (
    METRICS,
    RETRIEVAL_DEPTH,
    FailedQuestion,
    evaluate_retrieval,
    several_passages_lines,
    write_run,
)
from handbook_rag.evals.review import render_review_page
from handbook_rag.evals.stage import SavedRunError, Stage
from handbook_rag.evals.support import SUPPORT, SUPPORT_V2
from handbook_rag.evals.two_passage import (
    check_two_passage_set,
    kinds_of,
    load_reasons,
    reading_of,
)
from handbook_rag.generate.base import GenerationError, plain_words
from handbook_rag.generate.models import Answer, CacheHit
from handbook_rag.generate.pricing import cost_usd
from handbook_rag.generate.prompt import CURRENT_INSTRUCTIONS, known_instructions
from handbook_rag.index.compare import (
    DEFAULT_VECTOR_TOLERANCE,
    CollectionComparison,
    compare_collections,
)
from handbook_rag.index.indexer import Reuse, index_chunks
from handbook_rag.index.reuse import (
    PageChanges,
    ReuseError,
    compare_pages,
    find_reuse_source,
    read_previous_index,
)
from handbook_rag.index.store import CollectionMissingError, QdrantVectorStore, VectorStoreError
from handbook_rag.ingest.build import CorpusError, build_corpus, chunk_documents, load_snapshot
from handbook_rag.ingest.models import Chunk
from handbook_rag.ingest.snapshot import SnapshotError, download_snapshot, snapshot_commit
from handbook_rag.ingest.tokens import token_counter
from handbook_rag.measuring import measuring_the_model
from handbook_rag.pipeline import FollowUps, SearchedFor, answer_question
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.retrieve.spread import PageSpread
from handbook_rag.tracing import record_query, record_results, stage, trace_id, tracing_session
from handbook_rag.wiring import RetrievalMode

app = typer.Typer(no_args_is_help=True, add_completion=False)

# The things a judge decides about an answer. The first is the one a command means when
# none is named.
CRITERIA: tuple[Criterion[Any], ...] = (SUPPORT, CORRECT)
DEFAULT_CRITERION = SUPPORT
CriterionName = Literal["support", "correct"]
# Every judge a command can build: the judge of each criterion, in each of its versions.
# A criterion above stands for the first version of its judge, which is the one a
# command means when it names no version, and the one every saved run was made by. Each
# judge keeps its verdicts on an answer run in files of its own, and a run's scores are
# read from those of the judges above.
JUDGES: tuple[Criterion[Any], ...] = (SUPPORT, SUPPORT_V2, CORRECT)


_INSTRUCTIONS_OPTION = typer.Option(
    "--instructions",
    min=1,
    help=(
        "Which version of the answering instructions the answer model is sent. The default "
        "is the one the assistant uses; an earlier one is kept so that the old and the new "
        "assistant can be compared on the same day. Only an evaluation can choose it. A run "
        "is not continued with other instructions than it was made with."
    ),
)


def _instructions(version: int) -> int:
    """A version of the answering instructions, or a usage error if there is none."""
    try:
        return known_instructions(version)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--instructions") from error


def _instructions_note(version: int) -> str:
    """Which answering instructions a run sends, as an evaluation prints it."""
    if version == CURRENT_INSTRUCTIONS:
        return f"version {version}, the assistant's own"
    return f"version {version}, kept for comparison: the assistant uses {CURRENT_INSTRUCTIONS}"


def _instructions_suffix(version: int) -> str:
    """What a run's description adds for its instructions: nothing for version 1, so that
    a run made with them is described as every run before there was a second version."""
    return "" if version == 1 else f" under the answering instructions of version {version}"


def _criterion(name: str, version: int = 1) -> Criterion[Any]:
    """A criterion with one version of its judge, or a usage error if it has no such one."""
    versions = {judge.version: judge for judge in JUDGES if judge.name == name}
    if version not in versions:
        known = ", ".join(str(number) for number in sorted(versions))
        raise typer.BadParameter(
            f"The judge for '{name}' has no version {version}. Its versions: {known}.",
            param_hint="--judge-version",
        )
    return versions[version]


# How long an evaluation waits after a request fails. An overloaded model usually
# recovers within a minute or two, and waiting costs no request.
REST_AFTER_FAILURE_SECONDS = 30.0

_RUN_DESCRIPTIONS = {
    "dense": "Dense retrieval only.",
    "sparse": "Keyword search only (BM25).",
    "hybrid": "Dense and BM25 merged with reciprocal rank fusion.",
}


def measures[**P, R](command: Callable[P, R]) -> Callable[P, R]:
    """Mark a command as one that measures: while it runs, no answer cache can exist.

    An evaluation measures what the model writes, and the answer cache serves what it
    wrote before. Inside a command marked like this, building a cache and handing one
    to the entry point of an answer both fail, however the code that tries is written
    (``measuring.py``). A test asks of every command whether it is marked or is listed
    as one that is not an evaluation, so a new command has to be put on one side.
    """

    @functools.wraps(command)
    def measured(*args: P.args, **kwargs: P.kwargs) -> R:
        with measuring_the_model():
            return command(*args, **kwargs)

    measured.__dict__["measures_the_model"] = True
    return measured


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
def snapshot(
    replace: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="Replace a snapshot of another commit that is already in the data directory.",
        ),
    ] = False,
) -> None:
    """Download the handbook at the pinned commit into the data directory."""
    settings = get_settings()
    dest = settings.data_dir / "handbook"
    try:
        # Downloading a newer commit beside the pinned one is how the index is updated
        # (ADR 0019). Forgetting to point the data directory elsewhere would replace the
        # pinned snapshot, which the test questions are checked against, so that takes
        # a flag.
        on_disk = snapshot_commit(dest)
        if on_disk is not None and on_disk != settings.handbook_commit and not replace:
            raise SnapshotError(
                f"{dest} holds the snapshot of commit {on_disk[:12]}. Downloading commit "
                f"{settings.handbook_commit[:12]} there would replace it. To keep both, set "
                "HANDBOOK_RAG_DATA_DIR to another directory. To replace it, run again with "
                "--replace."
            )
        result = download_snapshot(
            repo_url=settings.handbook_repo_url,
            commit=settings.handbook_commit,
            content_path=settings.handbook_content_path,
            licence=settings.handbook_licence,
            dest=dest,
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


def _progress(verb: str) -> Callable[[int, int], None]:
    """Print a line at every tenth of the way, and at the end."""
    last_reported = 0

    def report_progress(done: int, total: int) -> None:
        nonlocal last_reported
        percent = 100 * done // total
        if percent >= last_reported + 10 or done == total:
            last_reported = percent
            typer.echo(f"  {verb:<10}  {done:,} of {total:,} ({percent}%)")

    return report_progress


@app.command()
def index(
    reuse_from: Annotated[
        str | None,
        typer.Option(
            help=(
                "An older commit whose collection lends its vectors: a chunk whose embedded "
                "text is unchanged takes them instead of being embedded. That collection is "
                "only read. Its name follows from the commit with these settings' models "
                "and chunking."
            ),
        ),
    ] = None,
) -> None:
    """Embed every chunk and store it in the vector store. Safe to run again."""
    settings = get_settings()
    try:
        store = wiring.make_store(settings)
        source = None if reuse_from is None else find_reuse_source(store, settings, reuse_from)
        started = time.perf_counter()
        corpus = build_corpus(settings)
        chunked = time.perf_counter() - started
    except (CorpusError, VectorStoreError, ReuseError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    embedder = wiring.make_embedder(settings)
    sparse_encoder = wiring.make_sparse_encoder(settings)
    typer.echo(f"Indexing {len(corpus.chunks):,} chunks")
    typer.echo(f"  collection  {store.collection}")
    if source is not None:
        typer.echo(f"  reuse from  {source.collection}, which is only read")
    typer.echo(f"  dense       {embedder.name} on {embedder.device}")
    typer.echo(f"  sparse      {sparse_encoder.name}")

    try:
        previous = None if source is None else read_previous_index(source)
        if previous is not None:
            typer.echo(
                f"  read        {previous.chunks:,} stored chunks of {len(previous.pages):,} "
                f"pages in {previous.seconds:.1f} seconds"
            )
        report = index_chunks(
            corpus.chunks,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            store=store,
            on_progress=_progress("embedded"),
            reuse=None if source is None or previous is None else Reuse(source, previous.points),
            on_reuse_progress=_progress("reused"),
        )
    except VectorStoreError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    # With reuse, the run's time includes copying vectors, which is not embedding.
    seconds = report.seconds if previous is None else report.embed_seconds
    rate = report.embedded / seconds if seconds > 0 else 0.0
    typer.echo("Index ready")
    typer.echo(f"  chunks      {report.total:,} in the collection")
    if previous is not None:
        _echo_page_changes(compare_pages(previous, corpus.chunks))
        typer.echo(
            f"  reused      {report.reused:,} chunks' vectors, in "
            f"{report.reuse_seconds:.1f} seconds"
        )
    typer.echo(f"  embedded    {report.embedded:,} new ({rate:.0f} per second)")
    typer.echo(f"  skipped     {report.already_indexed:,} already stored")
    typer.echo(f"  removed     {report.removed:,} no longer produced")
    if report.replaced:
        typer.echo(f"  replaced    {report.replaced:,} stored under the same id with other details")
    # A chunk that was replaced was embedded or took its vectors from the older
    # collection, so the three counts below cover it.
    if report.embedded or report.reused or report.removed:
        _empty_the_answer_cache(settings, store)
    typer.echo(f"  time        {report.seconds:.1f} seconds")
    if previous is not None:
        typer.echo(
            f"              chunking {chunked:.1f}, reading the older collection "
            f"{previous.seconds:.1f}, reusing {report.reuse_seconds:.1f}, embedding "
            f"{report.embed_seconds:.1f}"
        )


def _empty_the_answer_cache(settings: Settings, store: QdrantVectorStore) -> None:
    """Remove every answer cached from this commit's index, after a run that changed it.

    The cache's collections are named by the settings, and an index can change under the
    same settings: while it is first built, when a run that stopped is finished, or when
    pages are read another way. An answer written from the index as it was, a decline
    from a half-built one above all, must not be served from the index as it is.

    Every collection of the commit is emptied, and not only the one of the settings this
    command ran with. One index serves many of them: the answer model, reranking, the
    kind of search and the number of passages change the cache's name and not the
    index's. That also empties a cache whose index, made with another embedding model,
    did not change: too much is removed sooner than too little.

    The collections are emptied and kept, so a service that is running does not meet a
    collection that is being made again. It happens only after a run that finished: a
    run that stopped halfway has changed the index and emptied nothing, and the run
    that finishes it empties the caches if anything was left for it to change.
    Failing to empty a cache does not undo the indexing, so it is reported and no more.
    """
    try:
        removed = sum(
            QdrantAnswerStore(store.client, name).clear()
            for name in answer_collections(
                store.client, commit=wiring.answer_cache_commit(settings)
            )
        )
    except Exception as error:
        typer.echo(
            f"note: the answers cached from the index as it was could not be removed "
            f"({type(error).__name__}). Run `handbook-rag cache-clear --all`.",
            err=True,
        )
        return
    if removed:
        typer.echo(
            f"  cache       {removed:,} cached answers removed: the index they were written "
            "from has changed"
        )


def _echo_page_changes(changes: PageChanges) -> None:
    typer.echo(
        f"  pages       {len(changes.added):,} added, {len(changes.changed):,} changed, "
        f"{len(changes.removed):,} removed, {len(changes.unchanged):,} unchanged"
    )


def _usable_tolerance(tolerance: float) -> float:
    """Refuse a tolerance that would make the comparison meaningless, before any work.

    A tolerance that is not a number, or infinite, would let every difference through,
    and a negative one would fail two identical collections. ``check-reuse`` embeds for
    minutes before it compares, so it is checked when the command line is read.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise typer.BadParameter(f"must be a finite number of at least 0, not {tolerance}.")
    return tolerance


@app.command(name="check-reuse")
def check_reuse(
    reuse_from: Annotated[
        str,
        typer.Option(help="The older commit whose collection lends its vectors. It is only read."),
    ],
    dataset: Annotated[
        Path, typer.Option(help="Golden set: the pages that hold its evidence are the slice.")
    ] = Path("evals/golden/questions.jsonl"),
    config: Annotated[
        Path, typer.Option(help="The retrieval gate's settings, which say how many pages to add.")
    ] = Path("evals/gate.json"),
    tolerance: Annotated[
        float,
        typer.Option(
            help="How far an entry of one vector may be from the same entry of the other.",
            callback=_usable_tolerance,
        ),
    ] = DEFAULT_VECTOR_TOLERANCE,
) -> None:
    """Show on a slice of the handbook that reusing vectors gives the index a full run gives.

    The slice is the retrieval gate's: the pages that hold the golden set's evidence. It
    is indexed in memory twice, once by embedding every chunk and once by taking vectors
    from the older commit's collection where the embedded text is unchanged, and the
    two are compared. Nothing is written to the vector store server.
    """
    settings = get_settings()
    try:
        gate = load_gate_config(config)
        questions = _load_questions(dataset)
        source = find_reuse_source(wiring.make_store(settings), settings, reuse_from)
        snapshot = load_snapshot(settings)
        pages = select_pages(
            snapshot.documents, questions, distractors=gate.distractor_pages, seed=gate.seed
        )
        chunks = chunk_documents(pages, settings)
        embedder = wiring.make_embedder(settings)
        sparse_encoder = wiring.make_sparse_encoder(settings)
        full = wiring.make_memory_store("reuse-check-full")
        full_report = index_chunks(
            chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=full
        )
        # The older side is read for the same pages, by name, so a page of the slice that
        # the newer commit no longer has is counted as removed.
        names = evidence_pages(questions) | {page.doc_id for page in pages}
        previous = read_previous_index(source, pages=names)
        changes = compare_pages(previous, chunks)
        reused = wiring.make_memory_store("reuse-check-reused")
        reused_report = index_chunks(
            chunks,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            store=reused,
            reuse=Reuse(source, previous.points),
        )
        comparison = compare_collections(full, reused, tolerance=tolerance)
    except (GateError, DatasetError, CorpusError, VectorStoreError, ReuseError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo("Reuse check on a slice of the handbook")
    typer.echo(
        f"  pages       {len(pages):,} of {len(snapshot.documents):,} at commit "
        f"{snapshot.source_commit[:12]}, {len(chunks):,} chunks"
    )
    typer.echo(
        f"  full        {full_report.embedded:,} embedded in {full_report.seconds:.1f} seconds"
    )
    typer.echo(f"  reuse from  {source.collection}, {previous.chunks:,} chunks of the same pages")
    _echo_page_changes(changes)
    typer.echo(
        f"  reused      {reused_report.reused:,} chunks' vectors, {reused_report.embedded:,} "
        f"embedded, in {reused_report.seconds:.1f} seconds (reading the older "
        f"collection {previous.seconds:.1f})"
    )
    _echo_comparison(comparison)
    if not comparison.equal:
        typer.echo("Reuse check failed: the two indexes differ.", err=True)
        raise typer.Exit(code=1)
    if reused_report.reused == 0:
        # Both indexes were then embedded in full, so their being the same says nothing
        # about reuse.
        typer.echo(
            "Reuse check failed: no vector was reused, so nothing about reuse was checked. "
            f"Does {source.collection} hold the pages of the slice?",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("Reuse check passed: the two indexes are the same.")


@app.command(name="compare-collections")
def compare_collections_command(
    reference: Annotated[str, typer.Argument(help="The collection taken as right.")],
    candidate: Annotated[str, typer.Argument(help="The collection checked against it.")],
    tolerance: Annotated[
        float,
        typer.Option(
            help="How far an entry of one vector may be from the same entry of the other.",
            callback=_usable_tolerance,
        ),
    ] = DEFAULT_VECTOR_TOLERANCE,
) -> None:
    """Compare two collections point by point: ids, payloads and vectors. Both are only read."""
    settings = get_settings()
    try:
        store = wiring.make_store(settings)
        sides = [store.with_collection(reference), store.with_collection(candidate)]
        for side in sides:
            if not side.exists():
                raise CollectionMissingError(f"Collection {side.collection} does not exist.")
        comparison = compare_collections(sides[0], sides[1], tolerance=tolerance)
    except VectorStoreError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"{candidate} against {reference}")
    _echo_comparison(comparison)
    if not comparison.equal:
        raise typer.Exit(code=1)


def _echo_comparison(comparison: CollectionComparison, shown: int = 20) -> None:
    typer.echo(f"  compared    {comparison.compared:,} points that both collections hold")
    typer.echo(
        f"  ids         {len(comparison.missing):,} missing from the second, "
        f"{len(comparison.extra):,} only in the second"
    )
    typer.echo(f"  payloads    {len(comparison.payloads):,} fields differ")
    typer.echo(
        f"  vectors     {len(comparison.vectors):,} differ by more than {comparison.tolerance:g}; "
        f"largest difference {comparison.largest_dense_difference:.3g} (dense), "
        f"{comparison.largest_sparse_difference:.3g} (keyword); smallest cosine similarity "
        f"{comparison.smallest_cosine:.7f}"
    )
    differences = comparison.differences()
    for line in differences[:shown]:
        typer.echo(f"    {line}")
    if len(differences) > shown:
        typer.echo(f"    and {len(differences) - shown:,} more")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="The question to search for.")],
    limit: Annotated[int | None, typer.Option(help="How many chunks to return.")] = None,
    section: Annotated[
        str | None,
        typer.Option(help="Search only one handbook section, such as 'engineering'."),
    ] = None,
    mode: Annotated[
        RetrievalMode | None,
        typer.Option(help="dense (by meaning), sparse (by keyword) or hybrid. Default: settings."),
    ] = None,
    rerank: Annotated[
        bool | None,
        typer.Option(
            "--rerank/--no-rerank",
            help="Re-score the candidates with a cross-encoder. Default: settings.",
        ),
    ] = None,
) -> None:
    """Find the chunks most relevant to a question."""
    settings = get_settings()
    traced: str | None = None
    # Leaving the session sends the trace, also when the search fails.
    with tracing_session(settings):
        try:
            # Built outside the trace: loading models is a cost of starting the
            # process, not of answering this question.
            retriever = wiring.make_retriever(settings, mode, rerank)
            with stage("search", mode=mode or settings.retrieval_mode) as span:
                traced = trace_id(span)
                record_query(span, query)
                results = retriever.retrieve(
                    query, limit=limit or settings.search_limit, section=section
                )
                record_results(span, results)
        except VectorStoreError as error:
            typer.echo(f"error: {error}", err=True)
            if traced is not None:
                typer.echo(f"trace {traced}", err=True)
            raise typer.Exit(code=1) from error

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.chunk.text.split())
        typer.echo(f"{rank}. {result.score:.3f}  {result.chunk.context_header}")
        typer.echo(f"   {result.chunk.url}")
        typer.echo(f"   {preview[:220]}{'...' if len(preview) > 220 else ''}")
    if not results:
        typer.echo("No results.")
    if traced is not None:
        # The id to look this request up by in the trace backend.
        typer.echo(f"trace {traced}")
    if not results:
        raise typer.Exit(code=1)


def _load_questions(dataset: Path) -> list[GoldenQuestion]:
    """Load the golden set, reporting a missing file as what it is.

    Only this call is guarded. A model file that is missing when working offline is
    also a "file not found" to Python, and must not be reported as a missing golden set.
    """
    try:
        return load_golden_set(dataset)
    except FileNotFoundError as error:
        raise DatasetError(f"golden set not found: {dataset}") from error


@app.command()
def serve(
    host: Annotated[
        str | None, typer.Option(help="Address to listen on. Default: settings.")
    ] = None,
    port: Annotated[int | None, typer.Option(help="Port to listen on. Default: settings.")] = None,
) -> None:
    """Run the HTTP API. Its documentation is served at /docs."""
    # Imported here so that other commands do not load the web server.
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "handbook_rag.api:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


@app.command(name="mcp")
def mcp_server() -> None:
    """Offer handbook search to AI assistants: an MCP server on standard input and output.

    An assistant or a coding tool starts this command itself, and speaks the Model Context
    Protocol to it. Nothing but the protocol is written to standard output; the log goes
    to standard error.
    """
    # First, before the server's module and the libraries it brings are imported, so
    # that nothing they write on the way reaches the client.
    wire = _keep_standard_output_for_the_protocol()
    # Imported here so that other commands do not load the protocol's SDK.
    from handbook_rag.mcp_server import serve_stdio

    serve_stdio(wire, get_settings())


def _keep_standard_output_for_the_protocol() -> TextIO:
    """Give the protocol a copy of standard output of its own, and send the rest elsewhere.

    On an MCP server's standard output, one line that is not the protocol breaks the
    client's connection. From this call until the process ends, standard output is
    standard error: a stray print, a library that writes there, even text Python still
    holds in its buffer, all land in the log. The SDK's transport would divert standard
    output only while it serves, and point it back at the client when the client hangs
    up, while a thread may still be loading a model. Returns the copy.
    """
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return wire


def _page_spread(per_page: int | None, within: int | None) -> PageSpread | None:
    """The rule that spreads results over pages, if the command line asks for it.

    The rule has two numbers and neither has a default, so the command that made a run
    always says which rule it measured.
    """
    if per_page is None and within is None:
        return None
    if per_page is None or within is None:
        raise typer.BadParameter(
            "--spread-per-page and --spread-within are given together: at most so many "
            "chunks of one page, among so many of the first results."
        )
    return PageSpread(per_page=per_page, within=within)


def _search_variant(
    fusion_candidates: int | None, rrf_k: int | None, spread: PageSpread | None
) -> list[str]:
    """What a run changes about hybrid search, in words. Empty for the search as it is set."""
    changed: list[str] = []
    if fusion_candidates is not None:
        changed.append(f"{fusion_candidates} candidates from each search")
    if rrf_k is not None:
        changed.append(f"fusion constant {rrf_k}")
    if spread is not None:
        changed.append(
            f"at most {spread.per_page} chunk{'' if spread.per_page == 1 else 's'} of a page "
            f"among the first {spread.within} results"
        )
    return changed


def _refuse_a_variant_that_would_not_be_measured(
    mode: str,
    *,
    reranked: bool,
    wanted: int,
    fusion_candidates: int | None,
    spread: PageSpread | None,
) -> None:
    """Stop before any work when a run would carry a variant's name and measure something else.

    Dense search and keyword search merge nothing, so the option would change nothing
    there. A merged list that is spread over pages and then reranked comes out in the
    reranker's order, not the rule's. And each of the two searches is asked for at least
    as many candidates as hybrid search is asked for results, which is ``wanted``, so a
    smaller number of candidates would be measured as that many.
    """
    if fusion_candidates is not None and fusion_candidates < wanted:
        raise typer.BadParameter(
            f"--fusion-candidates below {wanted} cannot be measured: hybrid search is asked "
            f"for {wanted} results here, and each of its two searches for at least as many."
        )
    if mode != "hybrid":
        raise typer.BadParameter(
            "--fusion-candidates, --rrf-k and --spread-per-page change how hybrid search "
            f"merges its two lists, and this run is of {mode} search."
        )
    if spread is not None and reranked:
        raise typer.BadParameter(
            "--spread-per-page cannot be combined with reranking, which would put the "
            "results in another order again. Add --no-rerank."
        )


@app.command(name="eval-retrieval")
@measures
def eval_retrieval(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    description: Annotated[
        str | None,
        typer.Option(help="One line saying what this variant is. Default: its retrieval mode."),
    ] = None,
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/retrieval"),
    mode: Annotated[
        RetrievalMode | None,
        typer.Option(help="dense (by meaning), sparse (by keyword) or hybrid. Default: settings."),
    ] = None,
    rerank: Annotated[
        bool | None,
        typer.Option(
            "--rerank/--no-rerank",
            help="Re-score the candidates with a cross-encoder. Default: settings.",
        ),
    ] = None,
    reviewed_only: Annotated[
        bool,
        typer.Option(
            "--reviewed-only",
            help=(
                "Leave out the questions whose review is still pending: those no independent "
                "checker has accepted, or one has disputed. The run names them."
            ),
        ),
    ] = False,
    fusion_candidates: Annotated[
        int | None,
        typer.Option(
            help=(
                "A variant of hybrid search: how many candidates each of the two searches "
                "hands to the merge. At least as many as hybrid search is asked for: the 10 "
                "results that are scored, or what the reranker takes. Default: settings."
            ),
        ),
    ] = None,
    rrf_k: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "A variant of hybrid search: the constant of reciprocal rank fusion. "
                "Default: settings."
            ),
        ),
    ] = None,
    spread_per_page: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "A variant of hybrid search, off unless given: at most this many chunks of "
                "one page among the first results. A chunk over the limit moves down, and "
                "none is dropped. Needs --spread-within."
            ),
        ),
    ] = None,
    spread_within: Annotated[
        int | None,
        typer.Option(
            min=1,
            # A run scores the first results only, and the rule fills those places one by
            # one from the top: a limit that holds further down would be measured as this.
            max=RETRIEVAL_DEPTH,
            help=(
                "How many of the first results the limit of --spread-per-page holds for: "
                f"at most {RETRIEVAL_DEPTH}, the results a run scores."
            ),
        ),
    ] = None,
) -> None:
    """Score retrieval on the golden set: is the right passage found, and how high?"""
    settings = get_settings()
    chosen_mode = mode or settings.retrieval_mode
    reranked = settings.rerank if rerank is None else rerank
    spread = _page_spread(spread_per_page, spread_within)
    changed = _search_variant(fusion_candidates, rrf_k, spread)
    if changed:
        _refuse_a_variant_that_would_not_be_measured(
            chosen_mode,
            reranked=reranked,
            # The evaluation scores ten results. A reranker takes more than that to choose from.
            wanted=max(RETRIEVAL_DEPTH, settings.rerank_candidates if reranked else 0),
            fusion_candidates=fusion_candidates,
            spread=spread,
        )
        # A variant is the settings of this one run. Nothing else reads them.
        settings = settings.model_copy(
            update={
                "fusion_candidates": fusion_candidates or settings.fusion_candidates,
                "rrf_k": rrf_k or settings.rrf_k,
            }
        )
    if description is None:
        description = _RUN_DESCRIPTIONS[chosen_mode]
        if changed:
            description += f" Variant: {'; '.join(changed)}."
        if reranked:
            description += " Then reranked by a cross-encoder."
    try:
        questions = _load_questions(dataset)
        corpus = build_corpus(settings)
        retriever = wiring.make_retriever(settings, mode, rerank, spread=spread)
        report = evaluate_retrieval(
            questions, corpus.documents, retriever, reviewed_only=reviewed_only
        )
    except (DatasetError, CorpusError, VectorStoreError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if not report.results:
        typer.echo("error: no question could be scored.", err=True)
        raise typer.Exit(code=1)
    config = wiring.describe_retrieval(settings, mode, rerank, spread=spread)
    run_dir = write_run(report, output, variant, description, config)

    typer.echo(f"Retrieval evaluation: {variant}")
    typer.echo(
        f"  mode        {chosen_mode}"
        f"{', reranked with ' + settings.rerank_model if reranked else ''}"
    )
    if changed:
        typer.echo(f"  variant     {'; '.join(changed)}")
    typer.echo(f"  collection  {settings.collection_name}")
    typer.echo(
        f"  questions   {len(report.results)} scored, {len(report.failures)} failed, "
        f"{report.not_scored} unanswerable (not scored)"
    )
    if report.left_out:
        typer.echo(
            f"  left out    {len(report.left_out)} with review pending: "
            f"{', '.join(report.left_out)}"
        )
    for metric in METRICS:
        estimate = report.summary[str(metric["id"])]
        typer.echo(
            f"  {metric['label']!s:<10}  {estimate.mean:.3f}  "
            f"(95% interval {estimate.low:.3f} to {estimate.high:.3f})"
        )
    if report.several is not None:
        for line in several_passages_lines(report.several):
            typer.echo(line)
    for origin, estimates in report.by_group["origin"].items():
        hit = estimates["hit@5"]
        typer.echo(f"  Hit@5 on {origin} questions: {hit.mean:.3f} (n={hit.count})")
    typer.echo(f"  written to  {run_dir}")
    if report.failures:
        raise typer.Exit(code=1)


@app.command(name="eval-compare")
@measures
def eval_compare(
    base: Annotated[str, typer.Argument(help="The run to compare against, such as 'baseline'.")],
    candidate: Annotated[str, typer.Argument(help="The run being judged, such as 'v1'.")],
    runs: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per run.")
    ] = Path("evals/runs/retrieval"),
) -> None:
    """Compare two runs question by question: is the candidate really better?"""
    try:
        comparison = compare_runs(runs / base, runs / candidate)
    except ComparisonError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"{candidate} against {base}, on the same {comparison.questions} questions")
    typer.echo(f"  {base:<10}  {comparison.base.description}")
    typer.echo(f"  {candidate:<10}  {comparison.candidate.description}")
    typer.echo("")
    typer.echo(
        f"  {'metric':<9} {base[:9]:>9} {candidate[:9]:>9}   "
        f"{'difference (95% interval)':<27} better  worse   same"
    )
    labels = metric_labels(runs) or {name: name for name in comparison.metrics}
    for metric_id, label in labels.items():
        if metric_id not in comparison.metrics:
            continue
        metric = comparison.metrics[metric_id]
        change = metric.change.difference
        interval = f"{change.mean:+.3f} ({change.low:+.3f} to {change.high:+.3f})"
        scored = metric.change.difference.count
        typer.echo(
            f"  {label:<9} {metric.base.mean:>9.3f} {metric.candidate.mean:>9.3f}   "
            f"{interval:<27} {metric.change.better:>6} {metric.change.worse:>6} "
            f"{metric.change.same:>6}  {'clear' if metric.change.is_clear else 'unclear'}"
            + (
                f"  (on {scored} question{'' if scored == 1 else 's'})"
                if scored != comparison.questions
                else ""
            )
        )
    if comparison.not_compared:
        names = ", ".join(labels.get(name, name) for name in comparison.not_compared)
        typer.echo(f"  not compared, because no question has the score in both runs: {names}")
    typer.echo("")
    typer.echo("  clear    the interval excludes zero: chance is an unlikely explanation")
    typer.echo("  unclear  the two runs cannot be told apart with this many questions")


@app.command(name="eval-table")
@measures
def eval_table(
    runs: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per run.")
    ] = Path("evals/runs/retrieval"),
    metric: Annotated[
        list[str] | None,
        typer.Option(help="Metric to show, in order. Repeat for several. Default: all."),
    ] = None,
) -> None:
    """Print a Markdown table of every run, for pasting into a write-up."""
    try:
        overviews = overview_of_runs(runs)
    except ComparisonError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    labels = metric_labels(runs) or {name: name for name in overviews[0].metrics}
    # Asked for nothing in particular: every score that at least one run has. A score
    # the flow declares and no run has yet, such as a judged one, is simply not shown.
    chosen = metric or [name for name in labels if any(name in o.metrics for o in overviews)]
    unknown = [name for name in chosen if all(name not in o.metrics for o in overviews)]
    if unknown:
        typer.echo(f"error: no run has the metric: {', '.join(unknown)}", err=True)
        raise typer.Exit(code=1)
    typer.echo(results_table(overviews, labels, chosen), nl=False)


@app.command(name="eval-answers")
@measures
def eval_answers(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    description: Annotated[
        str | None,
        typer.Option(help="One line saying what this variant is. Default: its answer model."),
    ] = None,
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/answers"),
    max_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Stop after this many requests. Run the command again to continue the run.",
        ),
    ] = None,
    restart: Annotated[
        bool,
        typer.Option("--restart", help="Discard the replies this run has saved, and start again."),
    ] = False,
    instructions: Annotated[int, _INSTRUCTIONS_OPTION] = CURRENT_INSTRUCTIONS,
) -> None:
    """Answer every golden question and score what code can check. One real request each.

    Replies are saved as they arrive, so a run that stops, or is stopped, continues where
    it left off when the command is run again.
    """
    settings = get_settings()
    instructions = _instructions(instructions)
    config = {
        **wiring.describe_retrieval(settings),
        **wiring.describe_generation(settings, instructions),
    }
    run_dir = output / variant
    try:
        if not restart:
            # Before the models are loaded, which takes a while.
            answer_runs.check_run(run_dir, config)
        questions = _load_questions(dataset)
        corpus = build_corpus(settings)
        retriever = wiring.make_retriever(settings)
        # One attempt per request: no request is then sent a second time unseen, and the
        # run tries a failed question again itself.
        generator = wiring.make_generator(settings, attempts=1, instructions=instructions)
        # Last, so that a run that cannot start leaves nothing behind.
        saved = answer_runs.open_run(run_dir, config, restart=restart)
        if restart:
            # Verdicts on the discarded replies go with them, those of every version.
            for judge in JUDGES:
                judge.run_stage(run_dir).discard()
    except (
        DatasetError,
        CorpusError,
        VectorStoreError,
        GenerationError,
        SavedRunError,
    ) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    reusable = sum(
        1 for q in questions if q.id in saved and saved[q.id].fingerprint == q.fingerprint
    )
    to_ask = len(questions) - reusable
    if max_requests is not None:
        to_ask = min(to_ask, max_requests)
    pace = settings.eval_requests_per_minute
    typer.echo(f"Answer evaluation: {variant}")
    typer.echo(f"  model        {settings.answer_model}")
    typer.echo(f"  instructions {_instructions_note(instructions)}")
    typer.echo(
        f"  questions    {len(questions)} in the golden set, {reusable} already answered, "
        f"{to_ask} to ask now"
    )
    if to_ask:
        typer.echo(
            f"  pace         at most {pace:g} requests a minute, so at least "
            f"{max(1, round(to_ask / pace))} min"
        )

    asked = 0

    def keep(question: GoldenQuestion, record: answer_runs.AnswerRecord) -> None:
        nonlocal asked
        asked += 1
        answer_runs.append_record(run_dir, record)
        typer.echo(
            f"  {asked:>4}/{to_ask}  {question.id}  "
            f"{'answered' if record.answered else 'declined'}  {record.seconds:.1f} s"
        )

    def show_failure(failure: FailedQuestion) -> None:
        typer.echo(f"        failed  {failure.question_id}  {failure.message}")

    def ask_one(question: str) -> Answer:
        return answer_question(
            question,
            retriever=retriever,
            generator=generator,
            limit=settings.answer_context_chunks,
        )

    pacer = Pacer(pace)

    def before_each_request() -> None:
        pacer.wait()
        # Counted before the request is sent, so an interrupted run has still counted it.
        answer_runs.add_attempts(run_dir)

    try:
        run = answer_runs.run_answers(
            questions,
            corpus.documents,
            ask_one,
            saved=saved,
            wait=before_each_request,
            on_record=keep,
            on_failure=show_failure,
            rest=lambda: pacer.rest(REST_AFTER_FAILURE_SECONDS),
            max_attempts=max_requests,
        )
    except (DatasetError, answer_runs.ReplyError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    attempts = answer_runs.attempts_so_far(run_dir)

    try:
        report, judged = _score_answers(
            run_dir, questions, run.records, run.failures, corpus.chunks
        )
    except (DatasetError, SavedRunError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not report.scored:
        for failure in report.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        typer.echo(f"error: no question was answered. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    judges = _judge_records(run_dir, judged)
    answer_runs.write_answer_run(
        report,
        output,
        variant,
        description
        or (
            f"Answers by {settings.answer_model} from {settings.answer_context_chunks} passages"
            f"{_instructions_suffix(instructions)}."
        ),
        config,
        attempts=attempts,
        judges=judges,
    )

    typer.echo(
        f"  replies      {len(report.scored)} scored, {len(report.failures)} failed, "
        f"{report.not_asked} not asked yet"
    )
    typer.echo(f"  requests     at most {run.attempts} now, {attempts} for this run in all")
    _echo_answer_scores(report, judges)
    typer.echo(f"  written to   {run_dir}")
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    for failure in report.failures[:5]:
        typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
    if report.not_asked or report.failures:
        typer.echo("  Run the same command again to continue this run.")
    # A limit the caller set is a planned stop. Any other early end is a failure.
    planned = max_requests is not None and run.attempts >= max_requests
    if report.failures or (run.stopped is not None and not planned):
        raise typer.Exit(code=1)


def _scored_by(criterion: Criterion[Any]) -> tuple[Criterion[Any], ...]:
    """The judges whose verdicts give a run's judged scores, with this one in its place.

    For the first version of a judge, these are the judges whose scores a run's files hold.
    A later version stands in for the first one only in what a command prints.
    """
    return tuple(criterion if other.name == criterion.name else other for other in CRITERIA)


def _score_answers(
    run_dir: Path,
    questions: Sequence[GoldenQuestion],
    records: Mapping[str, answer_runs.AnswerRecord],
    failures: Sequence[FailedQuestion],
    chunks: Sequence[Chunk],
    criteria: Sequence[Criterion[Any]] = CRITERIA,
) -> tuple[answer_runs.AnswerReport, dict[str, dict[str, Judgement]]]:
    """Score a run's replies, with the judges' verdicts on them where they have any.

    The verdicts are read from the files of ``criteria``: by default the first version of
    each judge, which every score in a run's files comes from. They come back by the id
    of the score they give, and then by question.
    """
    records = answer_runs.replies_to_current_labels(questions, records)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    judged: dict[str, dict[str, Judgement]] = {}
    for criterion in criteria:
        if criterion.run_stage(run_dir).rows_path.is_file():
            cases = criterion.cases(questions, records, by_id)
            verdicts = saved_verdicts(criterion, run_dir, cases)
            if verdicts:
                judged[criterion.metric] = verdicts
    passed = {
        criterion.metric: {
            question_id: criterion.passed(judgement)
            for question_id, judgement in judged[criterion.metric].items()
        }
        for criterion in criteria
        if criterion.metric in judged
    }
    report = answer_runs.summarise_answers(questions, records, failures, judged=passed or None)
    return report, judged


def _judge_records(
    run_dir: Path,
    judged: Mapping[str, Mapping[str, Judgement]],
    criteria: Sequence[Criterion[Any]] = CRITERIA,
) -> dict[str, dict[str, object]]:
    """For each judged score of a run: who judged, what it cost, and how it was checked."""
    records: dict[str, dict[str, object]] = {}
    for criterion in criteria:
        record = judge_record(criterion, run_dir, judged.get(criterion.metric, {}))
        if record is not None:
            records[criterion.metric] = record
    return records


def _echo_estimate(label: str, estimate: Estimate | None) -> None:
    if estimate is not None:
        typer.echo(
            f"  {label:<28} {estimate.mean:.3f}  "
            f"(95% interval {estimate.low:.3f} to {estimate.high:.3f}, n={estimate.count})"
        )


def _echo_answer_scores(
    report: answer_runs.AnswerReport,
    judges: Mapping[str, Mapping[str, Any]],
    criteria: Sequence[Criterion[Any]] = CRITERIA,
) -> None:
    for metric in answer_runs.METRICS:
        _echo_estimate(metric["label"], report.summary.get(metric["id"]))
    for kind, estimates in report.by_group["unanswerable_kind"].items():
        declined = estimates["declined"]
        typer.echo(f"  Declined, {kind} questions: {declined.mean:.3f} (n={declined.count})")
    for name, estimates in report.by_group["evidence"].items():
        answered = estimates["answered"]
        typer.echo(f"  Answered, evidence {name}: {answered.mean:.3f} (n={answered.count})")
    if report.timing:
        first = report.timing.get("median_seconds_to_first_text")
        typer.echo(
            f"  time         median {report.timing['median_seconds']:.1f} s, "
            f"95% within {report.timing['p95_seconds']:.1f} s"
            + (f", first words after {first:.1f} s" if first is not None else "")
        )
        typer.echo(
            f"  tokens       {report.tokens['input_mean']:,.0f} in and "
            f"{report.tokens['output_mean']:,.0f} out per question"
        )
    for criterion in criteria:
        if criterion.metric in judges:
            _echo_judge_record(criterion, judges[criterion.metric])


def _version_option(version: int) -> str:
    """The option that names a version of a judge, as it is added to a command.

    Version 1 needs none, and is written without one, as it was before there was a second.
    """
    return "" if version == 1 else f" --judge-version {version}"


def _version_note(version: int) -> str:
    """Which version of a judge ruled, as it is printed after the name of its model.

    Nothing for version 1, so that what a command prints for it stays as it was.
    """
    return "" if version == 1 else f" (version {version})"


def _check_command(name: str, version: int = 1) -> str:
    """The command that checks one judge of a criterion, as a person types it."""
    criterion = "" if name == DEFAULT_CRITERION.name else f" --criterion {name}"
    return f"eval-judge{criterion}{_version_option(version)}"


def _echo_judge_record(criterion: Criterion[Any], judge: Mapping[str, Any]) -> None:
    """Say who judged, and how far that judge agreed with labelled answers.

    Which version of the judge it was is read from the settings saved with the verdicts,
    not from ``criterion``, which stands for the criterion whatever judge ruled on it.
    """
    counts = judge["verdicts"]
    version = judge_version(judge["config"])
    typer.echo(
        f"  judge        {judge['config']['judge_model']}{_version_note(version)} read "
        f"{judge['answers_judged']} answers: "
        + ", ".join(
            f"{counts[verdict]} {criterion.verdict_words[verdict]}"
            for verdict in criterion.verdicts
        )
    )
    check = judge.get("check")
    if not check:
        typer.echo(
            "  judge check  none. This judge has not been checked against labelled answers, "
            f"so its score cannot be trusted yet. Run {_check_command(criterion.name, version)}, "
            f"then eval-{criterion.name}{_version_option(version)} --checked."
        )
        return
    agreement = check["agreement"]
    recall, precision = agreement["recall"], agreement["precision"]
    if recall is None:
        caught = f"the labels it was checked against held no {criterion.faults_phrase}"
    else:
        caught = (
            f"it caught {round(recall['mean'] * recall['count'])} of {recall['count']} "
            f"{criterion.faults_phrase}"
        )
    if precision is None:
        alarms = "it raised no alarm"
    else:
        wrong = precision["count"] - round(precision["mean"] * precision["count"])
        alarms = (
            f"{wrong} of its {precision['count']} alarms {'was' if wrong == 1 else 'were'} false"
        )
    typer.echo(f"  judge check  {caught}, and {alarms} ({check['run']})")


def _judge_carefully[C: JudgedCase](
    criterion: Criterion[C],
    stage: Stage,
    cases: Sequence[C],
    judge: Judge[C],
    *,
    pace: float,
    max_requests: int | None,
    reuse: Mapping[str, Judgement],
) -> JudgeRun:
    """Judge the cases that have no saved verdict, saving each verdict as it arrives.

    A verdict taken from another run is copied into this stage's own file, so that the
    stage can be read without the run it borrowed from.
    """
    own = load_judgements(stage.rows_path, criterion.verdicts)
    known = {**reuse, **own}
    to_judge = sum(1 for case in cases if case.fingerprint not in known)
    typer.echo(
        f"  answers      {len(cases)} to read, {len(cases) - to_judge} already judged, "
        f"{to_judge if max_requests is None else min(to_judge, max_requests)} to judge now"
    )
    if to_judge:
        typer.echo(
            f"  pace         at most {pace:g} requests a minute, so at least "
            f"{max(1, round(min(to_judge, max_requests or to_judge) / pace))} min"
        )
    pacer = Pacer(pace)
    judged_now = 0

    def before_each_request() -> None:
        pacer.wait()
        # Counted before the request is sent, so an interrupted run has still counted it.
        stage.add_attempts()

    def keep(case: C, judgement: Judgement, new: bool) -> None:
        nonlocal judged_now
        if case.fingerprint not in own:
            stage.append(judgement_row(case, judgement))
        if new:
            judged_now += 1
            passed = criterion.passed(judgement)
            typer.echo(
                f"  {judged_now:>4}  {case.case_id}  "
                f"{criterion.passed_word if passed else criterion.failed_word}  "
                f"{judgement.seconds:.1f} s"
            )

    def show_failure(failure: FailedQuestion) -> None:
        typer.echo(f"        failed  {failure.question_id}  {failure.message}")

    return run_judge(
        cases,
        judge,
        saved=known,
        wait=before_each_request,
        on_judgement=keep,
        on_failure=show_failure,
        rest=lambda: pacer.rest(REST_AFTER_FAILURE_SECONDS),
        max_attempts=max_requests,
    )


def _judge_check_to_quote(
    criterion: Criterion[Any], checked: Path, config: Mapping[str, object]
) -> tuple[dict[str, Judgement], dict[str, Any]]:
    """The verdicts and the agreement of a judge check, if it may be quoted for this judge.

    It may be quoted only if it was made with the same judge, and only if it is whole: a
    check that judged part of the labelled answers would be quoted as if it were all.
    """
    command = _check_command(criterion.name, criterion.version)
    check_stage = judge_check_stage(checked)
    summary_file = checked / "summary.json"
    if not check_stage.state_path.is_file() or not summary_file.is_file():
        raise SavedRunError(f"there is no judge check in {checked}. Make one with {command}.")
    if check_stage.config() != dict(config):
        raise SavedRunError(
            f"the check in {checked} was made with another judge. Check this judge with "
            f"{command}, or leave --checked out."
        )
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        answers = {"labelled": summary["answers_labelled"], "judged": summary["answers_judged"]}
        agreement = summary["agreement"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SavedRunError(f"{summary_file} cannot be read as a judge check: {error}") from error
    if answers["judged"] != answers["labelled"]:
        raise SavedRunError(
            f"the check in {checked} is not finished: it judged {answers['judged']} of "
            f"{answers['labelled']} labelled answers. Finish it with {command}."
        )
    check = {
        "run": str(checked),
        "config": dict(config),
        "answers": answers,
        "agreement": agreement,
    }
    return load_judgements(check_stage.rows_path, criterion.verdicts), check


def _refuse_the_check_of_another_judge(
    criterion: Criterion[Any], run_dir: Path, settings: Settings
) -> None:
    """Refuse a directory that holds a check of another judge.

    That is the judge of another criterion, or another version of this criterion's
    judge. All checks use the same file names. Without this, restarting a check with the
    directory of another one would discard verdicts that took a share of a day's
    requests to make.

    A check says in its summary which criterion it is of. It is also known by its
    settings, which hold the hash of the judge's instructions: an older version wrote
    summaries that do not say, and a check that was stopped early has no summary. Either
    sign is enough to refuse. Which version of a judge made a check is told by its
    settings alone, since the hashes in them are what sets one version apart from another.

    A check whose settings record another version than the one asked for is refused as
    well, even when no judge the code builds today made it. After the judge's model
    changes, the settings of every saved check match no judge, and a restart with the
    other version would discard the check.

    One case is not caught: a check of another criterion's judge that says nothing in its
    summary and was made under instructions that judge no longer has. Such a check has
    lapsed already, since its agreement can no longer be quoted for any judge the code
    builds.
    """
    signs: list[object] = []
    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            loaded = json.loads(summary.read_text(encoding="utf-8"))
        except ValueError:  # not JSON, or not text at all
            loaded = None
        signs.append(loaded.get("criterion") if isinstance(loaded, dict) else None)
    stage = judge_check_stage(run_dir)
    try:
        stored = stage.config() if stage.state_path.is_file() else None
    except SavedRunError:
        stored = None
    made_by = [other for other in JUDGES if stored == wiring.describe_judge(settings, other)]
    signs.extend(other.name for other in made_by)
    for held in signs:
        if isinstance(held, str) and held != criterion.name:
            raise SavedRunError(
                f"{run_dir} holds a check of the judge for '{held}', not for "
                f"'{criterion.name}'. Give this check another --output or --variant."
            )
    for other in made_by:
        if other.version != criterion.version:
            raise SavedRunError(
                f"{run_dir} holds a check of version {other.version} of the judge for "
                f"'{other.name}', not of version {criterion.version}. Give this check another "
                "--output or --variant."
            )
    held = None if stored is None else judge_version(stored)
    if held is not None and held != criterion.version:
        raise SavedRunError(
            f"{run_dir} holds a check of version {held} of a judge, not of version "
            f"{criterion.version}. Give this check another --output or --variant."
        )


def _judge_stopped_badly(run: JudgeRun, max_requests: int | None) -> bool:
    """Whether a judge run ended in a way the caller did not plan."""
    planned = max_requests is not None and run.attempts >= max_requests
    return bool(run.failures) or (run.stopped is not None and not planned)


_JUDGE_VERSION_OPTION = typer.Option(
    "--judge-version",
    min=1,
    help=(
        "Which version of the judge to use. 1 is the judge that every saved run was made "
        "by. 2, which only 'support' has, is told the whole rubric and how to look for a "
        "limit that an answer leaves out. Neither version continues a check or a run of the "
        "other. On an answer run, version 2 keeps its verdicts in files of its own, and the "
        "run's scores stay those of version 1."
    ),
)


@app.command(name="eval-judge")
@measures
def eval_judge(
    criterion_name: Annotated[
        CriterionName,
        typer.Option(
            "--criterion",
            help=(
                "Which judge to check: 'support' (is each statement backed by the passage "
                "it cites?) or 'correct' (does the answer agree with the reference answer?)."
            ),
        ),
    ] = "support",
    version: Annotated[int, _JUDGE_VERSION_OPTION] = 1,
    cases: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Labelled answers: a JSON Lines file with one answer per line. Default: the "
                "labelled answers of the criterion, under evals/judge."
            )
        ),
    ] = None,
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this check. The default, 'current', is a scratch run that git "
                "ignores. Name a check to keep it: 'baseline', then 'v1' and so on."
            )
        ),
    ] = "current",
    output: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory that holds one sub-directory per check. Default: "
                "evals/runs/judge for 'support', evals/runs/judge-correct for 'correct'."
            )
        ),
    ] = None,
    max_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Stop after this many requests. Run the command again to continue the check.",
        ),
    ] = None,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart", help="Discard the verdicts this check has saved, and start again."
        ),
    ] = False,
) -> None:
    """Check a judge against labelled answers. One real request for each answer.

    The labelled answers hold real answers and copies with one planted fault. The check
    reports how many faults the judge catches and how many false alarms it raises.
    """
    settings = get_settings()
    criterion = _criterion(criterion_name, version)
    config = wiring.describe_judge(settings, criterion)
    cases = cases or criterion.labelled_path
    run_dir = (output or criterion.checks_dir) / variant
    stage = judge_check_stage(run_dir)
    try:
        _refuse_the_check_of_another_judge(criterion, run_dir, settings)
        if not restart:
            stage.check(config)
        # Only a judge that is shown passages needs the handbook to be read and cut up.
        chunks = build_corpus(settings).chunks if criterion.reads_passages else []
        labelled = criterion.load_labelled(cases, {chunk.chunk_id: chunk for chunk in chunks})
        judge = wiring.make_judge(settings, criterion, attempts=1)
        stage.open(config, restart=restart)
    except (DatasetError, CorpusError, GenerationError, SavedRunError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Judge check: {variant}")
    typer.echo(f"  criterion    {criterion.name}")
    typer.echo(f"  judge        {settings.judge_model}{_version_note(criterion.version)}")
    run = _judge_carefully(
        criterion,
        stage,
        [case.case for case in labelled],
        judge,
        pace=settings.eval_requests_per_minute,
        max_requests=max_requests,
        reuse={},
    )
    pairs = labelled_pairs(labelled, run.judgements)
    if not pairs:
        for failure in run.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        typer.echo(f"error: no answer was judged. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    agreement = measure_agreement(pairs, verdicts=criterion.verdicts, faults=criterion.faults)
    unclear = sum(
        1
        for case in labelled
        if case.case.case_id in run.judgements
        for verdict in case.expected.values()
        if verdict == UNCLEAR
    )
    disagreements = [
        {
            "case": pair.case_id,
            "statement": pair.statement,
            "kind": pair.kind,
            "expected": pair.expected,
            "judged": pair.judged,
            "reason": next(
                verdict.reason
                for verdict in run.judgements[pair.case_id].verdicts
                if verdict.number == pair.statement
            ),
        }
        for pair in pairs
        if pair.expected != pair.judged
    ]
    (run_dir / "disagreements.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in disagreements),
        encoding="utf-8",
    )
    judged = list(run.judgements.values())
    summary = {
        "description": (
            f"{settings.judge_model}{_version_note(criterion.version)} judging the labelled "
            f"answers in {cases}."
        ),
        "criterion": criterion.name,
        "config": config,
        "answers_labelled": len(labelled),
        "answers_judged": len(judged),
        "answers_failed": len(run.failures),
        "attempts": stage.attempts(),
        "left_out_as_unclear": unclear,
        "agreement": agreement_headline(agreement),
        "input_tokens": sum(judgement.input_tokens for judgement in judged),
        "output_tokens": sum(judgement.output_tokens for judgement in judged),
        "median_seconds": statistics.median(judgement.seconds for judgement in judged),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    typer.echo(
        f"  judged       {len(judged)} of {len(labelled)} answers, {len(run.failures)} failed"
    )
    typer.echo(
        f"  requests     at most {run.attempts} now, {stage.attempts()} for this check in all"
    )
    typer.echo(
        f"  {criterion.unit:<12} {agreement.statements} compared with their labels, "
        f"{unclear} left out because their label is unclear"
    )
    _echo_estimate("Same decision", agreement.same_decision)
    if agreement.recall is None:
        # A set with no fault in it, such as the reference answers given as answers,
        # checks only that the judge passes what it should.
        typer.echo(
            f"  {'Faults caught (recall)':<28} nothing to catch: the labels hold no "
            f"{criterion.faults_phrase}"
        )
    _echo_estimate("Faults caught (recall)", agreement.recall)
    _echo_estimate("Alarms that were right", agreement.precision)
    if agreement.labels_hold_both_decisions:
        typer.echo(f"  {'Kappa':<28} {agreement.kappa:.3f}")
    else:
        typer.echo(f"  {'Kappa':<28} not measured: every label makes the same decision")
    for kind, (caught, total) in agreement.caught.items():
        typer.echo(f"  Caught, {kind}: {caught} of {total}")
    _echo_estimate("Same verdict on the answer", agreement.same_answer)
    typer.echo(f"  written to   {run_dir}")
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    if len(judged) < len(labelled):
        typer.echo("  Run the same command again to continue this check.")
    if _judge_stopped_badly(run, max_requests):
        raise typer.Exit(code=1)


def _judge_answers(
    criterion: Criterion[Any],
    *,
    dataset: Path,
    variant: str,
    runs: Path,
    checked: Path | None,
    max_requests: int | None,
    restart: bool,
) -> None:
    """Judge a run's saved answers by one criterion, and add its score to the run.

    A later version of a judge keeps its verdicts in files of its own, and its score is
    printed, not added: the run's scores are those of version 1 (see :func:`_scored_by`).
    """
    settings = get_settings()
    config = wiring.describe_judge(settings, criterion)
    run_dir = runs / variant
    stage = criterion.run_stage(run_dir)
    try:
        saved_replies = answer_runs.load_records(run_dir / answer_runs.ANSWERS_FILE)
        if not saved_replies:
            raise SavedRunError(
                f"{run_dir} holds no replies. Make the answer run first, with eval-answers."
            )
        if not restart:
            stage.check(config)
        reuse: dict[str, Judgement] = {}
        check: dict[str, Any] | None = None
        if checked is not None:
            reuse, check = _judge_check_to_quote(criterion, checked, config)
        questions = _load_questions(dataset)
        # A reply to a question that has since been relabelled is a reply to another test.
        records = answer_runs.replies_to_current_labels(questions, saved_replies)
        outdated = len({q.id for q in questions} & set(saved_replies)) - len(records)
        # The passages are needed even by a judge that is not shown them: the run's
        # scores are written anew, the other judge's among them.
        corpus = build_corpus(settings)
        cases = criterion.cases(
            questions, records, {chunk.chunk_id: chunk for chunk in corpus.chunks}
        )
        judge = wiring.make_judge(settings, criterion, attempts=1)
        stage.open(config, restart=restart)
    except (DatasetError, CorpusError, GenerationError, SavedRunError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if check is not None:
        (run_dir / criterion.check_file).write_text(
            json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    typer.echo(f"{criterion.title}: {variant}")
    typer.echo(f"  judge        {settings.judge_model}{_version_note(criterion.version)}")
    if outdated:
        typer.echo(
            f"  left out     {outdated} {'reply' if outdated == 1 else 'replies'} to a question "
            "that has since been relabelled. Ask again with eval-answers."
        )
    run = _judge_carefully(
        criterion,
        stage,
        cases,
        judge,
        pace=settings.eval_requests_per_minute,
        max_requests=max_requests,
        reuse=reuse,
    )
    criteria = _scored_by(criterion)
    report, all_judged = _score_answers(
        run_dir, questions, records, answer_runs.load_failures(run_dir), corpus.chunks, criteria
    )
    judged = all_judged.get(criterion.metric, {})
    judges = _judge_records(run_dir, all_judged, criteria)
    if criterion.version == 1:
        answers_stage = answer_runs.answers_stage(run_dir)
        # Written whether or not this judge ruled on anything. After a restart, or when no
        # saved reply is to a question as it is labelled now, the run's files must not go
        # on showing a score that nothing stands behind.
        answer_runs.write_answer_run(
            report,
            runs,
            variant,
            answer_runs.saved_description(run_dir)
            or f"Answers by {answers_stage.config().get('answer_model', 'a model')}.",
            answers_stage.config(),
            attempts=answers_stage.attempts(),
            judges=judges,
        )
    # A later version writes none of the run's own files. No score in them is read from
    # its verdicts, so nothing it does can leave one of them standing on nothing.
    if not judged:
        for failure in run.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        if not cases:
            typer.echo("error: the run holds no answer to judge.", err=True)
        else:
            typer.echo(f"error: no answer was judged. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"  judged       {len(judged)} of {len(cases)} answers, {run.reused} of them from "
        f"verdicts saved earlier, {len(run.failures)} failed"
    )
    typer.echo(
        f"  requests     at most {run.attempts} now, {stage.attempts()} for this stage in all"
    )
    _echo_answer_scores(report, judges, criteria)
    if criterion.version == 1:
        typer.echo(f"  written to   {run_dir}")
    else:
        own = [
            name
            for name in (stage.rows_file, stage.state_file, *stage.outputs)
            if (run_dir / name).is_file()
        ]
        label = next(m["label"] for m in answer_runs.METRICS if m["id"] == criterion.metric)
        typer.echo(
            f"  written to   {run_dir}, in the files of version {criterion.version} alone: "
            f"{', '.join(own)}"
        )
        typer.echo(
            f"  kept apart   '{label}' is version {criterion.version}'s score here. The run's "
            "summary and the files of version 1 were left as they were."
        )
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    for failure in run.failures[:5]:
        typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
    if len(judged) < len(cases):
        typer.echo("  Run the same command again to continue.")
    if _judge_stopped_badly(run, max_requests):
        raise typer.Exit(code=1)


_DATASET_OPTION = typer.Option(help="Golden set: a JSON Lines file with one question per line.")
_VARIANT_OPTION = typer.Option(help="Name of the answer run whose replies are judged.")
_RUNS_OPTION = typer.Option(help="Directory that holds one sub-directory per answer run.")
_CHECKED_OPTION = typer.Option(
    help=(
        "A check of this judge made with eval-judge. Its agreement with the labels is "
        "reported beside the score, and its verdicts are used again where it read exactly "
        "the same answer."
    )
)
_MAX_REQUESTS_OPTION = typer.Option(
    min=1, help="Stop after this many requests. Run the command again to continue."
)
_RESTART_OPTION = typer.Option(
    "--restart", help="Discard the verdicts this judge has saved for this run, and start again."
)


@app.command(name="eval-support")
@measures
def eval_support(
    dataset: Annotated[Path, _DATASET_OPTION] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[str, _VARIANT_OPTION] = "current",
    runs: Annotated[Path, _RUNS_OPTION] = Path("evals/runs/answers"),
    checked: Annotated[Path | None, _CHECKED_OPTION] = None,
    max_requests: Annotated[int | None, _MAX_REQUESTS_OPTION] = None,
    restart: Annotated[bool, _RESTART_OPTION] = False,
    version: Annotated[int, _JUDGE_VERSION_OPTION] = 1,
) -> None:
    """Judge a run's saved answers: is every statement backed by the passage it cites?

    One real request for each answer that has no saved verdict. The replies themselves
    are not asked again.
    """
    _judge_answers(
        _criterion(SUPPORT.name, version),
        dataset=dataset,
        variant=variant,
        runs=runs,
        checked=checked,
        max_requests=max_requests,
        restart=restart,
    )


@app.command(name="eval-correct")
@measures
def eval_correct(
    dataset: Annotated[Path, _DATASET_OPTION] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[str, _VARIANT_OPTION] = "current",
    runs: Annotated[Path, _RUNS_OPTION] = Path("evals/runs/answers"),
    checked: Annotated[Path | None, _CHECKED_OPTION] = None,
    max_requests: Annotated[int | None, _MAX_REQUESTS_OPTION] = None,
    restart: Annotated[bool, _RESTART_OPTION] = False,
) -> None:
    """Judge a run's saved answers: does each one agree with the reference answer?

    One real request for each answer to a question the handbook answers, unless it has a
    saved verdict. The replies themselves are not asked again.
    """
    _judge_answers(
        CORRECT,
        dataset=dataset,
        variant=variant,
        runs=runs,
        checked=checked,
        max_requests=max_requests,
        restart=restart,
    )


def _passages_for_injected_cases(settings: Settings) -> dict[str, Chunk]:
    """The corpus by chunk id, which the cases with a planted instruction read from."""
    try:
        chunks = build_corpus(settings).chunks
    except CorpusError as error:
        raise CorpusError(
            f"the cases with a planted instruction read their passages from the handbook, "
            f"and it cannot be read: {error}"
        ) from error
    return {chunk.chunk_id: chunk for chunk in chunks}


def _retriever_for_out_of_scope_cases(settings: Settings) -> Retriever:
    """The retriever for the questions outside the handbook, once the index is known to be there.

    A missing index would otherwise show only as a failure of the first questions, each
    tried twice before the run stops itself.
    """
    try:
        store = wiring.make_store(settings)
        if store.count() == 0:
            raise VectorStoreError(
                f"Collection {store.collection} holds no passages. Run `handbook-rag index`."
            )
    except VectorStoreError as error:
        raise VectorStoreError(
            f"the questions outside the handbook are searched for in the index, and it "
            f"cannot be used: {error}"
        ) from error
    return wiring.make_retriever(settings, store=store)


@app.command(name="eval-adversarial")
@measures
def eval_adversarial(
    cases: Annotated[
        Path, typer.Option(help="Adversarial cases: a JSON Lines file with one case per line.")
    ] = adversarial.CASES_PATH,
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/adversarial"),
    max_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Stop after this many requests. Run the command again to continue the run.",
        ),
    ] = None,
    pace: Annotated[
        float | None,
        typer.Option(help="Send at most this many requests a minute. Default: settings."),
    ] = None,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Discard the replies this run has saved and their readings, and start again.",
        ),
    ] = False,
    instructions: Annotated[int, _INSTRUCTIONS_OPTION] = CURRENT_INSTRUCTIONS,
) -> None:
    """Test the assistant with planted instructions and questions outside the handbook.

    One real request for each case. Replies are saved as they arrive, so a run that stops
    continues where it left off. A reply that code cannot decide is set aside for a
    reader: each reading, which names who made it, goes into readings.jsonl in the run's
    directory, and the next run of the command takes them in.
    """
    settings = get_settings()
    instructions = _instructions(instructions)
    if pace is not None and pace <= 0:
        typer.echo("error: --pace needs a positive number of requests a minute.", err=True)
        raise typer.Exit(code=1)
    per_minute = pace or settings.eval_requests_per_minute
    config = {
        **wiring.describe_retrieval(settings),
        **wiring.describe_generation(settings, instructions),
    }
    run_dir = output / variant
    stage = adversarial.adversarial_stage(run_dir)
    try:
        if not restart:
            # Before the corpus, the index and the models are loaded, which takes a while.
            stage.check(config)
        loaded = adversarial.load_cases(cases)
        canaries = adversarial.instruction_canaries(
            wiring.answer_instructions(settings, instructions)
        )
        for case in loaded:
            adversarial.refuse_canaries_in_what_is_sent(case, canaries)
        saved = {} if restart else adversarial.load_replies(run_dir)
        readings = [] if restart else adversarial.load_readings(run_dir / adversarial.READINGS_FILE)
        # A reading that is not of a saved reply set aside is refused before any request.
        adversarial.summarise_adversarial(loaded, saved, canaries, readings)
        answered = adversarial.replies_to_current_cases(loaded, saved)
        pending = [c for c in adversarial.in_run_order(loaded) if c.case_id not in answered]
        to_ask = pending if max_requests is None else pending[:max_requests]
        # Only what the cases about to be asked need is loaded. A run that only takes in
        # readings needs no corpus, no index and no key.
        planted: dict[str, list[Chunk]] = {}
        if any(case.injected for case in to_ask):
            by_id = _passages_for_injected_cases(settings)
            planted = {
                case.case_id: adversarial.planted_passages(case, by_id, canaries)
                for case in pending
                if case.injected
            }
        retriever = None
        if any(not case.injected for case in to_ask):
            retriever = _retriever_for_out_of_scope_cases(settings)
        # One attempt per request, as for eval-answers: the run tries a failed case again.
        generator = (
            wiring.make_generator(settings, attempts=1, instructions=instructions)
            if to_ask
            else None
        )
        # Last, so that a run that cannot start leaves nothing behind.
        stage.open(config, restart=restart)
        saved = adversarial.load_replies(run_dir)
    except (
        DatasetError,
        CorpusError,
        VectorStoreError,
        GenerationError,
        SavedRunError,
        adversarial.ReadingError,
    ) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    injected = sum(case.injected for case in loaded)
    typer.echo(f"Adversarial evaluation: {variant}")
    typer.echo(f"  model        {settings.answer_model}")
    typer.echo(f"  instructions {_instructions_note(instructions)}")
    typer.echo(
        f"  cases        {len(loaded)} in the file ({injected} with a planted instruction, "
        f"{len(loaded) - injected} outside the handbook), {len(answered)} already answered, "
        f"{len(to_ask)} to ask now"
    )
    if to_ask:
        typer.echo(
            f"  pace         at most {per_minute:g} requests a minute, so at least "
            f"{max(1, round(len(to_ask) / per_minute))} min"
        )

    asked = 0

    def keep(case: adversarial.AdversarialCase, record: answer_runs.AnswerRecord) -> None:
        nonlocal asked
        asked += 1
        stage.append(dataclasses.asdict(record))
        by_code, _, reason = adversarial.decide(case, record, canaries)
        typer.echo(
            f"  {asked:>4}/{len(to_ask)}  {case.case_id}  "
            f"{'answered' if record.answered else 'declined'}  {by_code}: {reason}  "
            f"{record.seconds:.1f} s"
        )

    def show_failure(failure: FailedQuestion) -> None:
        typer.echo(f"        failed  {failure.question_id}  {failure.message}")

    pacer = Pacer(per_minute)

    def before_each_request() -> None:
        pacer.wait()
        # Counted before the request is sent, so an interrupted run has still counted it.
        stage.add_attempts()

    run = adversarial.AdversarialRun(records=saved)
    if generator is not None:
        try:
            run = adversarial.run_cases(
                loaded,
                adversarial.asker(
                    generator=generator,
                    retriever=retriever,
                    limit=settings.answer_context_chunks,
                    planted=planted,
                ),
                saved=saved,
                wait=before_each_request,
                on_record=keep,
                on_failure=show_failure,
                rest=lambda: pacer.rest(REST_AFTER_FAILURE_SECONDS),
                max_attempts=max_requests,
            )
        except answer_runs.ReplyError as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(code=1) from error

    try:
        report = adversarial.summarise_adversarial(
            loaded, run.records, canaries, readings, run.failures
        )
    except adversarial.ReadingError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not report.results:
        for failure in report.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        typer.echo(f"error: no case was answered. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    attempts = stage.attempts()
    adversarial.write_adversarial_run(
        report,
        run_dir,
        description=(
            f"Adversarial cases answered by {settings.answer_model}"
            f"{_instructions_suffix(instructions)}: {injected} with a "
            f"planted instruction, {len(loaded) - injected} outside the handbook."
        ),
        config=config,
        attempts=attempts,
        canaries=canaries,
    )

    typer.echo(
        f"  replies      {len(report.results)} decided or set aside, {len(report.failures)} "
        f"failed, {report.not_asked} not asked yet"
    )
    typer.echo(f"  requests     at most {run.attempts} now, {attempts} for this run in all")
    _echo_adversarial_figures(report)
    if report.tokens:
        typer.echo(
            f"  tokens       {report.tokens['input_mean']:,.0f} in and "
            f"{report.tokens['output_mean']:,.0f} out per case that reached the model, "
            f"{report.tokens['input_total']:,.0f} and {report.tokens['output_total']:,.0f} "
            "in all"
        )
    if report.waiting:
        typer.echo(
            f"  final        not yet: {len(report.waiting)} "
            f"{'reply waits' if len(report.waiting) == 1 else 'replies wait'} for a reading. "
            f"Write each to {run_dir / adversarial.READINGS_FILE} and run the same command again:"
        )
        for result in report.waiting:
            typer.echo(
                f"    {result.case.case_id:<28} reply "
                f"{adversarial.reply_fingerprint(result.record)}  {result.reason}"
            )
    else:
        typer.echo("  final        yes: every reply that code set aside has a reading")
    typer.echo(f"  written to   {run_dir}")
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    for failure in report.failures[:5]:
        typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
    if report.not_asked or report.failures:
        typer.echo("  Run the same command again to continue this run.")
    # A limit the caller set is a planned stop. Any other early end is a failure.
    planned = max_requests is not None and run.attempts >= max_requests
    if report.failures or (run.stopped is not None and not planned):
        raise typer.Exit(code=1)


def _echo_adversarial_figures(report: adversarial.AdversarialReport) -> None:
    """The counts by family and kind, and the share that held once they are final."""
    typer.echo(
        f"  {'':<24}{'cases':>6}{'replies':>8}{'code held':>10}{'code broke':>11}"
        f"{'set aside':>10}{'read held':>10}{'read broke':>11}{'kept':>7}   "
        "share held (95% interval)"
    )
    for name, figures in report.figures.items():
        label = f"  {name.split('/')[1]}" if "/" in name else name
        kept = f"{figures.kept_answer}/{figures.injected_replies}" if figures.kept else ""
        if figures.held is not None:
            held = f"{figures.held.mean:.3f} ({figures.held.low:.3f} to {figures.held.high:.3f})"
        else:
            held = "no reply yet" if not figures.replies else "waits for readings"
        typer.echo(
            f"  {label:<24}{figures.cases:>6}{figures.replies:>8}{figures.held_by_code:>10}"
            f"{figures.broke_by_code:>11}{figures.set_aside:>10}{figures.read_held:>10}"
            f"{figures.read_broke:>11}{kept:>7}   {held}"
        )
    injected = report.figures.get(adversarial.INJECTED)
    if injected is not None:
        _echo_estimate("Kept its answer", injected.kept)


@app.command(name="eval-limits")
@measures
def eval_limits(
    cases: Annotated[
        Path,
        typer.Option(
            help=(
                "The labelled answers of the limit set. The question of each passage's "
                "'dropped' answer is asked, with that passage."
            )
        ),
    ] = limit_questions.CASES_PATH,
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/limits"),
    instructions: Annotated[int, _INSTRUCTIONS_OPTION] = CURRENT_INSTRUCTIONS,
    max_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Stop after this many requests, to the answer model and the judge together. "
                "Run the command again to continue the run."
            ),
        ),
    ] = None,
    pace: Annotated[
        float | None,
        typer.Option(help="Send at most this many requests a minute. Default: settings."),
    ] = None,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Discard the replies this run has saved and their verdicts, and start again.",
        ),
    ] = False,
) -> None:
    """Ask the 40 questions of the limit set, each with its passage, and judge the answers.

    One real request for each question, then one for each answer to version 2 of the
    support judge: 80 when none fails and none declines. There is no search. Replies and
    verdicts are saved as they arrive, so a run that stops continues where it left off.
    """
    settings = get_settings()
    instructions = _instructions(instructions)
    if pace is not None and pace <= 0:
        typer.echo("error: --pace needs a positive number of requests a minute.", err=True)
        raise typer.Exit(code=1)
    per_minute = pace or settings.eval_requests_per_minute
    judge_criterion = limit_questions.JUDGE
    # The passages depend on the snapshot and the chunking, and nothing else of the search.
    config = {
        "handbook_commit": settings.handbook_commit,
        "chunking": settings.chunking.version,
        **wiring.describe_generation(settings, instructions),
    }
    judge_config = wiring.describe_judge(settings, judge_criterion)
    run_dir = output / variant
    stage = limit_questions.answers_stage(run_dir)
    judge_stage = judge_criterion.run_stage(run_dir)
    try:
        if not restart:
            # Before the corpus is read, which takes a while.
            stage.check(config)
            judge_stage.check(judge_config)
        questions = limit_questions.load_limit_questions(cases)
        saved = {} if restart else limit_questions.load_replies(run_dir)
        answered = limit_questions.replies_to_current_questions(questions, saved)
        pending = [q for q in limit_questions.in_run_order(questions) if q.case_id not in answered]
        to_ask = pending if max_requests is None else pending[:max_requests]
        # The judge reads the passages too, so the corpus is read whatever is left to do.
        chunks = {chunk.chunk_id: chunk for chunk in build_corpus(settings).chunks}
        passages = limit_questions.passages_of(questions, chunks)
        generator = (
            wiring.make_generator(settings, attempts=1, instructions=instructions)
            if to_ask
            else None
        )
        saved_verdicts_now = (
            {} if restart else load_judgements(judge_stage.rows_path, judge_criterion.verdicts)
        )
        unjudged = [
            case
            for case in limit_questions.judged_cases(questions, saved, chunks)
            if case.fingerprint not in saved_verdicts_now
        ]
        # A judge is built when it may be asked: for new answers, or saved ones not judged.
        judge = (
            wiring.make_judge(settings, judge_criterion, attempts=1) if to_ask or unjudged else None
        )
        # Last, so that a run that cannot start leaves nothing behind.
        stage.open(config, restart=restart)
        judge_stage.open(judge_config, restart=restart)
        saved = limit_questions.load_replies(run_dir)
    except (DatasetError, CorpusError, GenerationError, SavedRunError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Limit questions: {variant}")
    typer.echo(f"  model        {settings.answer_model}")
    typer.echo(f"  instructions {_instructions_note(instructions)}")
    typer.echo(
        f"  questions    {len(questions)} in the file, {len(answered)} already answered, "
        f"{len(to_ask)} to ask now"
    )
    if to_ask:
        typer.echo(
            f"  pace         at most {per_minute:g} requests a minute, so at least "
            f"{max(1, round(len(to_ask) / per_minute))} min"
        )

    asked = 0

    def keep(question: limit_questions.LimitQuestion, record: answer_runs.AnswerRecord) -> None:
        nonlocal asked
        asked += 1
        stage.append(dataclasses.asdict(record))
        typer.echo(
            f"  {asked:>4}/{len(to_ask)}  {question.case_id}  "
            f"{'answered' if record.answered else 'declined'}  {record.seconds:.1f} s"
        )

    def show_failure(failure: FailedQuestion) -> None:
        typer.echo(f"        failed  {failure.question_id}  {failure.message}")

    pacer = Pacer(per_minute)

    def before_each_request() -> None:
        pacer.wait()
        # Counted before the request is sent, so an interrupted run has still counted it.
        stage.add_attempts()

    run = answer_runs.AnswerRun(records=saved)
    if generator is not None:
        try:
            run = limit_questions.run_questions(
                questions,
                limit_questions.asker(generator, passages),
                saved=saved,
                wait=before_each_request,
                on_record=keep,
                on_failure=show_failure,
                rest=lambda: pacer.rest(REST_AFTER_FAILURE_SECONDS),
                max_attempts=max_requests,
            )
        except answer_runs.ReplyError as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(code=1) from error
    planned = max_requests is not None and run.attempts >= max_requests
    answers_stopped_badly = bool(run.failures) or (run.stopped is not None and not planned)

    judge_cases = limit_questions.judged_cases(questions, run.records, chunks)
    left = None if max_requests is None else max_requests - run.attempts
    judge_run: JudgeRun | None = None
    if answers_stopped_badly:
        typer.echo("  judge        not asked: the answers did not all come")
    elif judge is not None and (left is None or left > 0):
        typer.echo(f"  judge        {settings.judge_model}{_version_note(judge_criterion.version)}")
        judge_run = _judge_carefully(
            judge_criterion,
            judge_stage,
            judge_cases,
            judge,
            pace=per_minute,
            max_requests=left,
            reuse={},
        )
    verdicts = saved_verdicts(judge_criterion, run_dir, judge_cases)

    report = limit_questions.summarise_limits(
        questions, run.records, judge_cases, verdicts, run.failures
    )
    if not report.results:
        for failure in report.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        typer.echo(f"error: no question was answered. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    limit_questions.write_limit_run(
        report,
        run_dir,
        description=(
            f"The {len(questions)} questions of the limit set, each with its passage, answered "
            f"by {settings.answer_model}{_instructions_suffix(instructions)}, and judged by "
            f"{settings.judge_model}{_version_note(judge_criterion.version)}."
        ),
        config=config,
        attempts=stage.attempts(),
        judge_config=judge_config,
        judge_attempts=judge_stage.attempts(),
    )

    typer.echo(
        f"  replies      {len(report.results)} saved, {len(report.failures)} failed, "
        f"{report.not_asked} not asked yet"
    )
    judge_attempts = 0 if judge_run is None else judge_run.attempts
    typer.echo(
        f"  requests     at most {run.attempts + judge_attempts} now; for this run in all, "
        f"{stage.attempts()} to the answer model and {judge_stage.attempts()} to the judge"
    )
    _echo_limit_figures(report)
    if report.waiting_for_the_judge:
        typer.echo(
            f"  final        not yet: {len(report.waiting_for_the_judge)} answers wait for "
            "the judge"
        )
    elif report.final:
        typer.echo("  final        yes: every question has a reply, and every answer a verdict")
    typer.echo(f"  written to   {run_dir}")
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    if judge_run is not None and judge_run.stopped is not None:
        typer.echo(f"  judge stopped: {judge_run.stopped}")
    for failure in [*report.failures, *([] if judge_run is None else judge_run.failures)][:5]:
        typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
    if not report.final:
        typer.echo("  Run the same command again to continue this run.")
    judge_stopped_badly = judge_run is not None and _judge_stopped_badly(judge_run, left)
    if answers_stopped_badly or judge_stopped_badly:
        raise typer.Exit(code=1)


def _echo_limit_figures(report: limit_questions.LimitReport) -> None:
    """The counts in all and by kind of limit, and the share faulted once it is final."""
    typer.echo(
        f"  {'':<14}{'questions':>10}{'replies':>8}{'answered':>9}{'judged':>7}{'faulted':>8}"
        "   share faulted (95% interval)"
    )
    for name, figures in report.figures.items():
        label = name if name == "all" else f"  {name}"
        if figures.share_faulted is not None:
            share = figures.share_faulted
            faulted = f"{share.mean:.3f} ({share.low:.3f} to {share.high:.3f})"
        else:
            faulted = "not final" if figures.replied else "no reply yet"
        typer.echo(
            f"  {label:<14}{figures.questions:>10}{figures.replied:>8}{figures.answered:>9}"
            f"{figures.judged:>7}{figures.faulted:>8}   {faulted}"
        )


@app.command(name="eval-follow-ups")
@measures
def eval_follow_ups(
    cases: Annotated[
        Path,
        typer.Option(help="Follow-up cases: a JSON Lines file with one conversation per line."),
    ] = follow_ups.CASES_PATH,
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/follow-ups"),
    max_requests: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Stop after this many requests. Run the command again to continue the run.",
        ),
    ] = None,
    pace: Annotated[
        float | None,
        typer.Option(help="Send at most this many requests a minute. Default: settings."),
    ] = None,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Discard the replies this run has saved and their readings, and start again.",
        ),
    ] = False,
) -> None:
    """Rewrite follow-up questions so that they stand alone, and measure the rewrites.

    One real request for each case, to the answer model, and none for a case that has a
    saved reply. Replies are saved as they arrive, so a run that stops continues where it
    left off. A rewrite that code cannot decide is set aside for a reader: each reading,
    which names who made it, goes into readings.jsonl in the run's directory, and the
    next run of the command takes them in. The verifier of the answer cache runs on this
    machine to read some rewrites, and no answer cache is built.
    """
    settings = get_settings()
    if pace is not None and pace <= 0:
        typer.echo("error: --pace needs a positive number of requests a minute.", err=True)
        raise typer.Exit(code=1)
    per_minute = pace or settings.eval_requests_per_minute
    claude = not settings.answer_model.startswith("gemini-")
    config = follow_ups.describe_rewriting(
        settings.answer_model, settings.answer_effort if claude else None
    )
    run_dir = output / variant
    stage = follow_ups.follow_up_stage(run_dir)
    try:
        if not restart:
            stage.check(config)
        loaded = follow_ups.load_cases(cases)
        phrases = follow_ups.step_phrases()
        follow_ups.refuse_phrases_in_cases(loaded, phrases)
        saved = {} if restart else follow_ups.load_replies(stage.rows_path)
        readings = [] if restart else follow_ups.load_readings(run_dir / follow_ups.READINGS_FILE)
        # A reading that is not of a saved reply is refused before any request.
        follow_ups.check_readings(loaded, saved, readings)
        answered = follow_ups.replies_to_current_cases(loaded, saved)
        pending = [c for c in follow_ups.in_run_order(loaded) if c.case_id not in answered]
        to_ask = pending if max_requests is None else pending[:max_requests]
        # A run that only takes in readings needs no key.
        rewriter = wiring.make_rewriter(settings) if to_ask else None
        # Last, so that a run that cannot start leaves nothing behind.
        stage.open(config, restart=restart)
        saved = follow_ups.load_replies(stage.rows_path)
    except (DatasetError, GenerationError, SavedRunError, follow_ups.ReadingError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    groups = {group: sum(case.group == group for case in loaded) for group in follow_ups.KINDS}
    typer.echo(f"Follow-up questions: {variant}")
    typer.echo(f"  model        {settings.answer_model}")
    typer.echo(
        f"  cases        {len(loaded)} in the file ({groups[follow_ups.FOLLOW_UP]} follow-ups, "
        f"{groups[follow_ups.STANDALONE]} that stand alone, {groups[follow_ups.PLANTED]} with "
        f"a planted instruction), {len(answered)} already answered, {len(to_ask)} to ask now"
    )
    if to_ask:
        typer.echo(
            f"  pace         at most {per_minute:g} requests a minute, so at least "
            f"{max(1, round(len(to_ask) / per_minute))} min"
        )

    asked = 0

    def keep(case: follow_ups.FollowUpCase, reply: follow_ups.FollowUpReply) -> None:
        nonlocal asked
        asked += 1
        stage.append(dataclasses.asdict(reply))
        rewrite = follow_ups.read_case_reply(case, reply)
        typer.echo(
            f"  {asked:>4}/{len(to_ask)}  {case.case_id:<26} {rewrite.outcome:<10} "
            f"{reply.seconds:.1f} s"
        )

    def show_failure(failure: FailedQuestion) -> None:
        typer.echo(f"        failed  {failure.question_id}  {failure.message}")

    pacer = Pacer(per_minute)

    def before_each_request() -> None:
        pacer.wait()
        # Counted before the request is sent, so an interrupted run has still counted it.
        stage.add_attempts()

    run = follow_ups.FollowUpRun(replies=saved)
    if rewriter is not None:
        run = follow_ups.run_cases(
            loaded,
            follow_ups.asker(rewriter),
            saved=saved,
            wait=before_each_request,
            on_reply=keep,
            on_failure=show_failure,
            rest=lambda: pacer.rest(REST_AFTER_FAILURE_SECONDS),
            max_attempts=max_requests,
        )

    try:
        # The verifier is loaded only when a rewrite has the words of no acceptable form.
        pairs = follow_ups.pairs_to_verify(loaded, run.replies)
        verified: dict[tuple[str, str], Verification] = {}
        if pairs:
            verified = dict(zip(pairs, wiring.make_verifier(settings).score(pairs), strict=True))
        report = follow_ups.summarise_follow_ups(
            loaded, run.replies, verified, phrases, readings, run.failures
        )
    except (VerifierError, follow_ups.ReadingError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not report.results:
        for failure in report.failures[:5]:
            typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
        typer.echo(f"error: no case was answered. {run.stopped or ''}".rstrip(), err=True)
        raise typer.Exit(code=1)
    attempts = stage.attempts()
    summary = follow_ups.write_follow_up_run(
        report,
        run_dir,
        description=(
            f"Follow-up questions rewritten by {settings.answer_model} so that they stand "
            "alone, read by code, by the verifier of the answer cache and by readers."
        ),
        config=config,
        attempts=attempts,
        verifier=settings.cache_verifier_model,
        phrases=phrases,
    )

    typer.echo(
        f"  replies      {len(report.results)} read, {len(report.failures)} failed, "
        f"{report.not_asked} not asked yet"
    )
    typer.echo(f"  requests     at most {run.attempts} now, {attempts} for this run in all")
    for line in follow_ups.report_lines(summary):
        typer.echo(line)
    if report.waiting:
        typer.echo(
            f"  final        not yet: {len(report.waiting)} "
            f"{'case waits' if len(report.waiting) == 1 else 'cases wait'} for a reading. "
            f"Write each to {run_dir / follow_ups.READINGS_FILE} and run the same command again:"
        )
        for result in report.waiting:
            waits = [
                *(["the question"] if result.meaning_waits else []),
                *(["the instruction"] if result.instruction_waits else []),
            ]
            typer.echo(
                f"    {result.case.case_id:<26} reply "
                f"{follow_ups.reply_fingerprint(result.reply)}  {' and '.join(waits)}"
            )
    elif report.final:
        typer.echo(
            "  final        yes: every case has a reply, and every rewrite that code set "
            "aside has a reading"
        )
    else:
        missing = report.not_asked + len(report.failures)
        typer.echo(
            f"  final        not yet: {missing} {'case has' if missing == 1 else 'cases have'} "
            "no reply"
        )
    typer.echo(f"  written to   {run_dir}")
    if run.stopped is not None:
        typer.echo(f"  stopped: {run.stopped}")
    for failure in report.failures[:5]:
        typer.echo(f"  failed: {failure.question_id}: {failure.message}", err=True)
    if report.not_asked or report.failures:
        typer.echo("  Run the same command again to continue this run.")
    # A limit the caller set is a planned stop. Any other early end is a failure. With the
    # careful loop as it is, an early end that was not planned leaves a failure behind, so
    # the first test already decides; the second is for a loop that stops for another
    # reason one day.
    planned = max_requests is not None and run.attempts >= max_requests
    if report.failures or (run.stopped is not None and not planned):
        raise typer.Exit(code=1)


@app.command(name="eval-gate")
@measures
def eval_gate(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    config: Annotated[
        Path, typer.Option(help="The gate's settings: the size of the slice and the minimums.")
    ] = Path("evals/gate.json"),
) -> None:
    """Score the default retriever on a slice of the handbook, and fail below fixed minimums.

    Made for CI. It needs the snapshot and the models but no vector store server: the
    pages that hold the golden set's evidence are indexed in memory, with a fixed
    sample of other pages if the settings ask for one.
    """
    settings = get_settings()
    try:
        gate = load_gate_config(config)
        questions = _load_questions(dataset)
        snapshot = load_snapshot(settings)
        pages = select_pages(
            snapshot.documents, questions, distractors=gate.distractor_pages, seed=gate.seed
        )
        chunks = chunk_documents(pages, settings)
        store = wiring.make_memory_store("retrieval-gate")
        embedder = wiring.make_embedder(settings)
        sparse_encoder = wiring.make_sparse_encoder(settings)
        indexed = index_chunks(
            chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=store
        )
        retriever = wiring.make_retriever(
            settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
        )
        report = evaluate_retrieval(questions, pages, retriever)
    except (GateError, DatasetError, CorpusError, VectorStoreError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    with_evidence = len(evidence_pages(questions) & {page.doc_id for page in pages})
    typer.echo("Retrieval gate")
    typer.echo(
        f"  pages       {len(pages):,} of {len(snapshot.documents):,}: {with_evidence:,} with "
        f"evidence, {len(pages) - with_evidence:,} without (seed {gate.seed})"
    )
    typer.echo(f"  chunks      {len(chunks):,}, indexed in memory in {indexed.seconds:.0f} s")
    typer.echo(
        f"  mode        {settings.retrieval_mode}"
        f"{', reranked with ' + settings.rerank_model if settings.rerank else ''}"
    )
    typer.echo(
        f"  questions   {len(report.results)} scored, {len(report.failures)} failed, "
        f"{report.not_scored} unanswerable (not scored)"
    )
    labels = {str(metric["id"]): str(metric["label"]) for metric in METRICS}
    for metric, minimum in gate.minimums.items():
        estimate = report.summary.get(metric)
        if estimate is not None:
            verdict = "ok" if estimate.mean >= minimum else "BELOW"
            score, floor = shown_with_minimum(estimate.mean, minimum)
            typer.echo(f"  {labels[metric]:<10}  {score}  minimum {floor}  {verdict}")
    missed = [result.question.id for result in report.results if not result.scores["hit@5"]]
    if missed:
        typer.echo(f"  not found in the top five: {', '.join(missed)}")

    problems = gate_problems(report, gate.minimums)
    if problems:
        typer.echo("Gate failed:", err=True)
        for problem in problems:
            typer.echo(f"  {problem}", err=True)
        raise typer.Exit(code=1)
    typer.echo("Gate passed.")


@app.command(name="eval-cache")
@measures
def eval_cache(
    pairs: Annotated[
        Path, typer.Option(help="Cache question pairs: a JSON Lines file with one pair per line.")
    ] = cache_pairs.PAIRS_PATH,
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[
        str,
        typer.Option(
            help=(
                "Name of this run. The default, 'current', is a scratch run that git "
                "ignores. Name a run to keep it: 'baseline', then 'v1', 'v2' and so on."
            )
        ),
    ] = "current",
    output: Annotated[
        Path, typer.Option(help="Directory that holds one sub-directory per variant.")
    ] = Path("evals/runs/cache"),
    candidates: Annotated[
        int,
        typer.Option(
            min=1, help="How many of the nearest stored questions a question is checked against."
        ),
    ] = cache_pairs.CANDIDATES,
    floor: Annotated[
        float,
        typer.Option(
            min=-1.0, max=1.0, help="The cosine below which a stored question is no candidate."
        ),
    ] = cache_pairs.FLOOR,
    decline_candidates: Annotated[
        bool,
        typer.Option(
            "--declines/--no-declines",
            help=(
                "Whether a stored question the handbook does not answer can be a candidate. "
                "--no-declines leaves those out by the golden set's label. To leave out what "
                "the answer cache itself leaves out, use --replies."
            ),
        ),
    ] = True,
    replies: Annotated[
        Path | None,
        typer.Option(
            help=(
                "A saved answer run, such as evals/runs/answers/baseline. The simulated "
                "cache is filled with its replies, and the answer cache's own rule decides "
                "which of them may be served to a question worded differently: the cache as "
                "it is built."
            ),
        ),
    ] = None,
) -> None:
    """Measure what a semantic cache would serve, on labelled pairs of questions.

    No cache is built, and no request goes to any model API: the embedding model and the
    verifier run on this machine. A cache holding the golden questions is simulated, a
    threshold is chosen for each score on one half of the pairs, and the result is
    reported on the other half. The answer cache itself is not read or written.
    """
    if replies is not None and not decline_candidates:
        raise typer.BadParameter(
            "--replies and --no-declines are two ways to leave stored questions out. "
            "Give one of them.",
            param_hint="--replies",
        )
    settings = get_settings()
    try:
        questions = _load_questions(dataset)
        loaded = cache_pairs.load_pairs(pairs, questions)
        held = None if replies is None else cache_replies.stored_replies(questions, replies)
        # A cache would sit in the service, which embeds one question at a time on the CPU.
        embedder = wiring.make_embedder(wiring.settings_for_serving(settings))
        verifier = wiring.make_verifier(settings)
        measured = cache_pairs.measure(
            loaded,
            questions,
            embedder=embedder,
            verifier=verifier,
            candidates=candidates,
            floor=floor,
            decline_candidates=decline_candidates,
            held=None if held is None else held.may_be_candidates,
        )
        description = (
            "The cosine of two question embeddings, and an entailment check in both "
            "directions, as the test of a semantic cache."
        )
        if not decline_candidates:
            description += " A stored decline is no candidate, as in the answer cache."
        if held is not None:
            description += (
                f" The cache holds the replies of the answer run {held.source}, and the "
                "answer cache's own rule decides which of them are candidates."
            )
        summary = cache_pairs.write_cache_run(
            measured,
            output / variant,
            description=description,
            config={
                "embedding_model": settings.embedding_model,
                "embedding_query_prefix": settings.embedding_query_prefix,
                "embedding_device": embedder.device,
                "verifier_model": settings.cache_verifier_model,
                "verifier_device": wiring.VERIFIER_DEVICE,
                **verifier.describe(),
                **cache_pairs.data_fingerprints(loaded, questions),
                # Recorded only for a run that was filled from replies, as the option above.
                **(
                    {}
                    if held is None
                    else {"replies_from": held.source, "replies_fingerprint": held.fingerprint}
                ),
            },
            measured_on=datetime.date.today(),
            stored_replies=None if held is None else held.figures(questions),
        )
    except (DatasetError, VerifierError, SavedRunError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    for line in report_lines(summary):
        typer.echo(line)
    typer.echo(f"  written to   {output / variant}")


@app.command(name="golden-candidates")
@measures
def golden_candidates(
    output: Annotated[Path, typer.Argument(help="Where to write the sampled passages.")],
    count: Annotated[int, typer.Option(help="How many passages to sample.")] = 180,
    seed: Annotated[int, typer.Option(help="Seed, so the same sample can be drawn again.")] = 7,
) -> None:
    """Sample passages to write golden set questions from, spread across sections."""
    try:
        corpus = build_corpus(get_settings())
    except CorpusError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    sampled = sample_source_chunks(corpus.chunks, count=count, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as lines:
        for number, chunk in enumerate(sampled, start=1):
            candidate = {
                "candidate_id": f"c{number:03d}",
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "heading_path": list(chunk.heading_path),
                "text": chunk.text,
            }
            lines.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    sections = len({chunk.section for chunk in sampled})
    typer.echo(f"Sampled {len(sampled)} passages from {sections} sections (seed {seed})")
    typer.echo(f"  written to  {output}")


@app.command(name="golden-check")
@measures
def golden_check(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    reasons: Annotated[
        Path | None,
        typer.Option(
            help=(
                "For a set of two-passage questions: the file that says, for each question, "
                "why it needs both passages. The set is then also held to two passages on "
                "two pages for every question."
            )
        ),
    ] = None,
) -> None:
    """Check every golden set label against the corpus, and summarise the set."""
    why = None
    if reasons is not None:
        # Read first, and guarded by itself: a missing file here is not a missing golden set.
        try:
            why = load_reasons(reasons)
        except FileNotFoundError as error:
            typer.echo(f"error: reasons not found: {reasons}", err=True)
            raise typer.Exit(code=1) from error
        except DatasetError as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(code=1) from error
    try:
        questions = load_golden_set(dataset)
        corpus = build_corpus(get_settings())
    except FileNotFoundError as error:
        typer.echo(f"error: golden set not found: {dataset}", err=True)
        raise typer.Exit(code=1) from error
    except (DatasetError, CorpusError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    report = check_against_corpus(questions, corpus.documents)
    problems = [*report.problems]
    if why is not None:
        problems += check_two_passage_set(questions, why)

    def counts(counter: dict[str, int]) -> str:
        return ", ".join(f"{count} {name}" for name, count in sorted(counter.items()))

    typer.echo(f"Golden set: {report.total} questions")
    typer.echo(f"  type      {counts(report.by_type)}")
    if why is not None:
        typer.echo(f"  kinds     {counts(kinds_of(questions, why))}")
        typer.echo(f"  reading   {counts(reading_of(questions, why))}")
    if report.by_unanswerable_kind:
        typer.echo(f"  declining {counts(report.by_unanswerable_kind)}")
    typer.echo(f"  origin    {counts(report.by_origin)}")
    typer.echo(f"  review    {counts(report.by_review)}")
    typer.echo(f"  sections  {len(report.by_section)} covered")
    typer.echo(
        f"  wording   questions share {report.mean_overlap:.0%} of their words with their "
        f"evidence on average; {len(report.high_overlap)} share 80% or more"
    )
    if report.overlap_by_origin:
        shares = ", ".join(f"{o} {share:.0%}" for o, share in report.overlap_by_origin.items())
        typer.echo(f"            by origin: {shares}")
    if problems:
        typer.echo(f"  labels    {len(problems)} broken:")
        for problem in problems[:20]:
            typer.echo(f"    - {problem}")
        raise typer.Exit(code=1)
    typer.echo("  labels    every quote was found in its page, and on no other page")
    if why is not None:
        typer.echo("  passages  every question has two, on two pages, and a reason for each")


@app.command(name="golden-review")
@measures
def golden_review(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
    output: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the page. Defaults to golden-review.html in the data dir."
        ),
    ] = None,
) -> None:
    """Write a page that shows every question next to its evidence, for a person to review."""
    settings = get_settings()
    try:
        questions = load_golden_set(dataset)
        corpus = build_corpus(settings)
    except FileNotFoundError as error:
        typer.echo(f"error: golden set not found: {dataset}", err=True)
        raise typer.Exit(code=1) from error
    except (DatasetError, CorpusError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    target = output or settings.data_dir / "golden-review.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_review_page(questions, corpus.documents), encoding="utf-8")
    # A model's check is not a person's review, so a checked question still waits for one.
    waiting = sum(question.review != "verified" for question in questions)
    typer.echo(
        f"Review page for {len(questions)} questions ({waiting} not yet verified by a person)"
    )
    typer.echo(f"  written to  {target}")
    typer.echo("  open it in a browser, then set review to verified for each question you accept")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    section: Annotated[
        str | None,
        typer.Option(help="Use only one handbook section, such as 'engineering'."),
    ] = None,
    cache: Annotated[
        bool,
        typer.Option(
            "--cache/--no-cache",
            help=(
                "With --no-cache the answer model is asked, also when a stored answer would "
                "do. The new answer takes the stored one's place if it is one the cache keeps."
            ),
        ),
    ] = True,
    earlier: Annotated[
        list[str] | None,
        typer.Option(
            "--earlier",
            help=(
                "An earlier question of the conversation. Repeat it for each, oldest first. "
                "With one or more, the question is first rewritten so that it stands alone, "
                "with one more request to the answer model, and the command says which "
                "question it searched for (HANDBOOK_RAG_FOLLOW_UPS). Questions only, never "
                "answers."
            ),
        ),
    ] = None,
) -> None:
    """Answer a question from the handbook, with the passages that support the answer.

    An answer stored earlier for the same question, or for one that means the same, is
    printed without a request to the model, and the command says so
    (HANDBOOK_RAG_CACHE_MODE).

    With --earlier, the question is a follow-up: it is rewritten into one that stands
    alone, which is then looked up, searched for and answered (ADR 0024).
    """
    settings = get_settings()
    traced: str | None = None
    wrote = False
    answers: AnswerCache | None = None
    searched: SearchedFor | None = None
    earlier_questions = list(earlier or [])

    def note_trace(found: str) -> None:
        nonlocal traced
        traced = found

    def write(text: str) -> None:
        nonlocal wrote
        wrote = True
        typer.echo(text, nl=False)

    def note_searched(found: SearchedFor) -> None:
        nonlocal searched
        searched = found
        # Before the answer, so that the reader knows what it answers.
        line = _searched_line(found, given=any(plain_words(q) for q in earlier_questions))
        if line is not None:
            typer.echo(line + "\n")
        if found.rewrite.failure is not None:
            # Its own words go to the error output, as an answer's failure does: a step
            # that is set up wrong is seen there, and not only as one word.
            typer.echo(f"note: rewriting the follow-up failed: {found.rewrite.failure}", err=True)

    # Leaving the session sends the trace, also when the answer fails.
    with tracing_session(settings):
        try:
            # Built outside the trace: loading models is a cost of starting the
            # process, not of answering this question. The cache's second model is not
            # loaded here: only a question that has a stored one near it needs it.
            retriever, answers = wiring.make_retriever_and_cache(
                settings, wiring.make_store(settings)
            )
            generator = wiring.make_generator(settings)
            # Built only for a follow-up: without earlier questions nothing is rewritten.
            follow_ups: FollowUps | None = None
            if earlier_questions:
                follow_ups = wiring.make_follow_ups(settings)
            if answers is not None:
                # The collection the answers are kept in is made here, before the
                # question is looked up, and not by the answer that is stored at the
                # end: making it can take longer than this command waits for that.
                # When it is there, this is one question to the store.
                answers.prepare()
            answer = answer_question(
                question,
                retriever=retriever,
                generator=generator,
                limit=settings.answer_context_chunks,
                section=section,
                on_text=write,
                on_trace=note_trace,
                cache=answers,
                fresh=not cache,
                earlier=earlier_questions,
                follow_ups=follow_ups,
                on_searched=note_searched,
            )
        except (VectorStoreError, GenerationError) as error:
            if wrote:
                # The reply broke off. The error starts on a line of its own.
                typer.echo("")
            typer.echo(f"error: {error}", err=True)
            if isinstance(error, GenerationError):
                _show_what_was_found(error, broke_off=wrote)
            if traced is not None:
                typer.echo(f"trace {traced}", err=True)
            raise typer.Exit(code=1) from error
    typer.echo("")

    # A reply that is not an answer lists no sources, even if the model attached one to it.
    if answer.answered and answer.sources:
        typer.echo("\nSources")
        for number, chunk in enumerate(answer.sources, start=1):
            typer.echo(f"  [{number}] {chunk.context_header}")
            typer.echo(f"      {chunk.url}")
    elif answer.answered:
        typer.echo("\nNo passages were cited for this answer. Treat it with caution.")

    if answer.cached is not None:
        _show_cache_hit(answer.cached)
        # No model was asked, so there are no tokens, no price and no time to print.
        details = [answer.model, "no request to the model"]
    else:
        cost = cost_usd(answer.model, answer.usage)
        details = [
            answer.model,
            f"{answer.usage.input_tokens:,} tokens in",
            f"{answer.usage.output_tokens:,} out",
            f"{answer.seconds:.1f} s",
        ]
        if cost is not None:
            details.insert(3, f"${cost:.5f}")
    if answer.requests > 1:
        # Failed requests were sent first. Each one counts against the key's allowance.
        # The tokens and the time above are those of the request that wrote the reply.
        details.append(f"tokens and time of the last of {answer.requests} requests")
    if answer.fallback:
        details.append("written by the fallback model")
    if searched is not None and searched.rewrite.requests:
        details.append(_rewrite_cost(searched))
    if answer.stop_reason == "max_tokens":
        details.append("cut off at the token limit")
    elif answer.stop_reason == "refusal":
        details.append("withheld by the model's provider")
    elif answer.stop_reason not in ("end_turn", "no_context"):
        details.append(f"stopped early ({answer.stop_reason})")
    typer.echo("\n" + " | ".join(details))
    if traced is not None:
        # The id to look this answer up by in the trace backend.
        typer.echo(f"trace {traced}")
    # Last, after everything is printed: the answer is stored by a thread of its own,
    # and the command ends only when that is done, or has taken too long.
    if answers is not None and not answers.close():
        typer.echo(
            "note: the answer was not stored in the cache, because the vector store did "
            "not answer in time.",
            err=True,
        )


def _searched_line(searched: SearchedFor, *, given: bool) -> str | None:
    """What the command says, before the answer, of the question it searched for.

    Nothing when no earlier question was ``given``. A follow-up that could not be
    rewritten is searched for as it was typed, and the line says so: it finds little.
    """
    rewrite = searched.rewrite
    if rewrite.outcome == "not_asked":
        if not given:
            return None
        return (
            "Searched for the question as it was typed: follow-up questions are switched "
            "off (HANDBOOK_RAG_FOLLOW_UPS)."
        )
    if rewrite.outcome == "rewritten":
        return f"Searched for: {searched.question}"
    if rewrite.outcome == "unchanged":
        return "Searched for the question as it was asked: it stands alone."
    why = "the model's reply was not usable" if rewrite.outcome == "unusable" else "it failed"
    if rewrite.failure is not None:
        why = f"it failed ({rewrite.failure.reason})"
    return (
        f"Searched for the question as it was typed: rewriting it to stand alone did not "
        f"work, because {why}."
    )


def _rewrite_cost(searched: SearchedFor) -> str:
    """What rewriting the follow-up cost, for the line of details after the answer."""
    rewrite = searched.rewrite
    parts = ["rewritten with 1 request" if rewrite.changed else "rewriting took 1 request"]
    if rewrite.reply is not None:
        usage = rewrite.reply.usage
        parts.append(f"{usage.input_tokens:,} tokens in, {usage.output_tokens:,} out")
    parts.append(f"{rewrite.seconds:.1f} s")
    return ", ".join(parts)


def _show_cache_hit(hit: CacheHit) -> None:
    """Say that an answer came from the cache, and which question it was written for.

    An answer matched by meaning was written for another wording. Printing that wording
    is what lets the person asking see a wrong match, and the last line says what to do
    about one.
    """
    how = "an exact match" if hit.tier == "exact" else "matched by meaning"
    typer.echo(f"\nFrom the cache: {how}, stored {hit.stored_at}. No request to the model.")
    typer.echo(f"  written for the question: {hit.question}")
    if hit.similarity is not None and hit.score is not None:
        typer.echo(f"  similarity {hit.similarity:.3f}, second check {hit.score:.3f}")
    if hit.tier != "exact":
        typer.echo("  If that is another question than yours, ask again with --no-cache.")


@app.command(name="cache-status")
def cache_status() -> None:
    """Say how many answers the answer cache holds for the current settings."""
    settings = get_settings()
    try:
        store = wiring.make_store(settings)
        answers = wiring.make_answer_store(settings, store)
        held = answers.held()
        others = {
            name: QdrantAnswerStore(store.client, name).held()
            for name in answer_collections(store.client)
            if name != answers.collection
        }
    except VectorStoreError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    exact_only = held.answers - held.by_meaning - held.declines
    typer.echo("Answer cache")
    typer.echo(f"  mode         {settings.cache_mode} (HANDBOOK_RAG_CACHE_MODE)")
    typer.echo(f"  collection   {answers.collection}")
    typer.echo(f"  answers      {held.answers:,} stored")
    typer.echo(
        f"               {held.by_meaning:,} may be served to a question that means the same"
    )
    typer.echo(
        f"               {held.declines:,} declines and {exact_only:,} answers that a guard "
        "found to say something is missing, or that cite nothing: served to the same question only"
    )
    if others:
        stored = sum(other.answers for other in others.values())
        typer.echo(
            f"  left behind  {stored:,} answers in {len(others):,} collections stored under "
            "other settings. They are not served under these."
        )
        typer.echo("               `handbook-rag cache-clear --all` removes them.")


@app.command(name="cache-clear")
def cache_clear(
    everything: Annotated[
        bool,
        typer.Option(
            "--all",
            help=(
                "Also remove the collections of answers stored under other settings. "
                "They are not served under the current ones."
            ),
        ),
    ] = False,
) -> None:
    """Empty the answer cache of the current settings. The handbook's index is not touched.

    The answers are removed and their collection is kept, so a service that is running
    goes on without a pause. With --all, the collections left behind under other
    settings are removed whole. Only collections named as this project names them are
    touched.
    """
    settings = get_settings()
    try:
        store = wiring.make_store(settings)
        answers = wiring.make_answer_store(settings, store)
        emptied = answers.clear()
        left_behind = {
            name: QdrantAnswerStore(store.client, name).remove()
            for name in (answer_collections(store.client) if everything else [])
            if name != answers.collection
        }
    except VectorStoreError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Removed {emptied:,} cached answers from {answers.collection}")
    for name, count in left_behind.items():
        typer.echo(f"Removed {count:,} cached answers and their collection {name}")


def _show_what_was_found(error: GenerationError, *, broke_off: bool) -> None:
    """Say what else failed, then show the passages the search found.

    No whole answer was written, but the search did its job, and its passages may be all
    the person asking needs. They are the ones the model would have answered from.
    ``broke_off`` says that part of a reply was printed before the failure.
    """
    if error.before_fallback is not None:
        typer.echo(
            f"error: before the fallback model, the answer model failed: {error.before_fallback}",
            err=True,
        )
    if not error.retrieved:
        return
    what = "The reply broke off." if broke_off else "No answer was written."
    typer.echo(f"\n{what} The search found these passages:")
    for result in error.retrieved:
        typer.echo(f"  {result.chunk.context_header}")
        typer.echo(f"      {result.chunk.url}")
