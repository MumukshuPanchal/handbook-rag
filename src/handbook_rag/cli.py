"""Command-line interface. Run ``handbook-rag --help`` to list the commands."""

import dataclasses
import json
import time
from pathlib import Path
from typing import Annotated

import typer

from handbook_rag import __version__, wiring
from handbook_rag.config import get_settings
from handbook_rag.evals.compare import (
    ComparisonError,
    compare_runs,
    metric_labels,
    overview_of_runs,
    results_table,
)
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
from handbook_rag.evals.retrieval import METRICS, evaluate_retrieval, write_run
from handbook_rag.evals.review import render_review_page
from handbook_rag.generate.claude import GenerationError
from handbook_rag.generate.pricing import cost_usd
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import VectorStoreError
from handbook_rag.ingest.build import CorpusError, build_corpus, chunk_documents, load_snapshot
from handbook_rag.ingest.snapshot import SnapshotError, download_snapshot
from handbook_rag.ingest.tokens import token_counter
from handbook_rag.pipeline import answer_question
from handbook_rag.tracing import record_query, record_results, stage, trace_id, tracing_session
from handbook_rag.wiring import RetrievalMode

app = typer.Typer(no_args_is_help=True, add_completion=False)

_RUN_DESCRIPTIONS = {
    "dense": "Dense retrieval only.",
    "sparse": "Keyword search only (BM25).",
    "hybrid": "Dense and BM25 merged with reciprocal rank fusion.",
}


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


@app.command()
def index() -> None:
    """Embed every chunk and store it in the vector store. Safe to run again."""
    settings = get_settings()
    try:
        store = wiring.make_store(settings)
        corpus = build_corpus(settings)
    except (CorpusError, VectorStoreError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    embedder = wiring.make_embedder(settings)
    sparse_encoder = wiring.make_sparse_encoder(settings)
    typer.echo(f"Indexing {len(corpus.chunks):,} chunks")
    typer.echo(f"  collection  {store.collection}")
    typer.echo(f"  dense       {embedder.name} on {embedder.device}")
    typer.echo(f"  sparse      {sparse_encoder.name}")

    last_reported = 0

    def report_progress(done: int, total: int) -> None:
        nonlocal last_reported
        percent = 100 * done // total
        if percent >= last_reported + 10 or done == total:
            last_reported = percent
            typer.echo(f"  embedded    {done:,} of {total:,} ({percent}%)")

    try:
        report = index_chunks(
            corpus.chunks,
            embedder=embedder,
            sparse_encoder=sparse_encoder,
            store=store,
            on_progress=report_progress,
        )
    except VectorStoreError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    rate = report.embedded / report.seconds if report.seconds > 0 else 0.0
    typer.echo("Index ready")
    typer.echo(f"  chunks      {report.total:,} in the collection")
    typer.echo(f"  embedded    {report.embedded:,} new ({rate:.0f} per second)")
    typer.echo(f"  skipped     {report.already_indexed:,} already stored")
    typer.echo(f"  removed     {report.removed:,} no longer produced")
    typer.echo(f"  time        {report.seconds:.1f} seconds")


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


@app.command(name="eval-retrieval")
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
) -> None:
    """Score retrieval on the golden set: is the right passage found, and how high?"""
    settings = get_settings()
    chosen_mode = mode or settings.retrieval_mode
    reranked = settings.rerank if rerank is None else rerank
    if description is None:
        description = _RUN_DESCRIPTIONS[chosen_mode]
        if reranked:
            description += " Then reranked by a cross-encoder."
    try:
        questions = _load_questions(dataset)
        corpus = build_corpus(settings)
        retriever = wiring.make_retriever(settings, mode, rerank)
        report = evaluate_retrieval(questions, corpus.documents, retriever)
    except (DatasetError, CorpusError, VectorStoreError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    if not report.results:
        typer.echo("error: no question could be scored.", err=True)
        raise typer.Exit(code=1)
    config = wiring.describe_retrieval(settings, mode, rerank)
    run_dir = write_run(report, output, variant, description, config)

    typer.echo(f"Retrieval evaluation: {variant}")
    typer.echo(
        f"  mode        {chosen_mode}"
        f"{', reranked with ' + settings.rerank_model if reranked else ''}"
    )
    typer.echo(f"  collection  {settings.collection_name}")
    typer.echo(
        f"  questions   {len(report.results)} scored, {len(report.failures)} failed, "
        f"{report.not_scored} unanswerable (not scored)"
    )
    for metric in METRICS:
        estimate = report.summary[str(metric["id"])]
        typer.echo(
            f"  {metric['label']!s:<10}  {estimate.mean:.3f}  "
            f"(95% interval {estimate.low:.3f} to {estimate.high:.3f})"
        )
    for origin, estimates in report.by_group["origin"].items():
        hit = estimates["hit@5"]
        typer.echo(f"  Hit@5 on {origin} questions: {hit.mean:.3f} (n={hit.count})")
    typer.echo(f"  written to  {run_dir}")
    if report.failures:
        raise typer.Exit(code=1)


@app.command(name="eval-compare")
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
        typer.echo(
            f"  {label:<9} {metric.base.mean:>9.3f} {metric.candidate.mean:>9.3f}   "
            f"{interval:<27} {metric.change.better:>6} {metric.change.worse:>6} "
            f"{metric.change.same:>6}  {'clear' if metric.change.is_clear else 'unclear'}"
        )
    typer.echo("")
    typer.echo("  clear    the interval excludes zero: chance is an unlikely explanation")
    typer.echo("  unclear  the two runs cannot be told apart with this many questions")


@app.command(name="eval-table")
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
    chosen = metric or list(labels)
    unknown = [name for name in chosen if any(name not in o.metrics for o in overviews)]
    if unknown:
        typer.echo(f"error: no such metric in every run: {', '.join(unknown)}", err=True)
        raise typer.Exit(code=1)
    typer.echo(results_table(overviews, labels, chosen), nl=False)


@app.command(name="eval-gate")
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


@app.command(name="golden-candidates")
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
def golden_check(
    dataset: Annotated[
        Path, typer.Option(help="Golden set: a JSON Lines file with one question per line.")
    ] = Path("evals/golden/questions.jsonl"),
) -> None:
    """Check every golden set label against the corpus, and summarise the set."""
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

    def counts(counter: dict[str, int]) -> str:
        return ", ".join(f"{count} {name}" for name, count in sorted(counter.items()))

    typer.echo(f"Golden set: {report.total} questions")
    typer.echo(f"  type      {counts(report.by_type)}")
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
    if report.problems:
        typer.echo(f"  labels    {len(report.problems)} broken:")
        for problem in report.problems[:20]:
            typer.echo(f"    - {problem}")
        raise typer.Exit(code=1)
    typer.echo("  labels    every quote was found in its page, and on no other page")


@app.command(name="golden-review")
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
) -> None:
    """Answer a question from the handbook, with the passages that support the answer."""
    settings = get_settings()
    try:
        answer = answer_question(
            question,
            retriever=wiring.make_retriever(settings),
            generator=wiring.make_generator(settings),
            limit=settings.answer_context_chunks,
            section=section,
            on_text=lambda text: typer.echo(text, nl=False),
        )
    except (VectorStoreError, GenerationError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("")

    if answer.sources:
        typer.echo("\nSources")
        for number, chunk in enumerate(answer.sources, start=1):
            typer.echo(f"  [{number}] {chunk.context_header}")
            typer.echo(f"      {chunk.url}")
    elif answer.answered:
        typer.echo("\nNo passages were cited for this answer. Treat it with caution.")

    cost = cost_usd(answer.model, answer.usage)
    details = [
        answer.model,
        f"{answer.usage.input_tokens:,} tokens in",
        f"{answer.usage.output_tokens:,} out",
        f"{answer.seconds:.1f} s",
    ]
    if cost is not None:
        details.insert(3, f"${cost:.5f}")
    if answer.stop_reason == "max_tokens":
        details.append("cut off at the token limit")
    typer.echo("\n" + " | ".join(details))
