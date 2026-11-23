"""Command-line interface. Run ``handbook-rag --help`` to list the commands."""

import dataclasses
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from handbook_rag import __version__, wiring
from handbook_rag.config import get_settings
from handbook_rag.evals import answers as answer_runs
from handbook_rag.evals.agreement import measure_agreement
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
    judgement_row,
    labelled_pairs,
    load_judgements,
    run_judge,
    saved_verdicts,
)
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.retrieval import (
    METRICS,
    FailedQuestion,
    evaluate_retrieval,
    write_run,
)
from handbook_rag.evals.review import render_review_page
from handbook_rag.evals.stage import SavedRunError, Stage
from handbook_rag.evals.support import SUPPORT
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import Answer
from handbook_rag.generate.pricing import cost_usd
from handbook_rag.index.indexer import index_chunks
from handbook_rag.index.store import VectorStoreError
from handbook_rag.ingest.build import CorpusError, build_corpus, chunk_documents, load_snapshot
from handbook_rag.ingest.models import Chunk
from handbook_rag.ingest.snapshot import SnapshotError, download_snapshot
from handbook_rag.ingest.tokens import token_counter
from handbook_rag.pipeline import answer_question
from handbook_rag.tracing import record_query, record_results, stage, trace_id, tracing_session
from handbook_rag.wiring import RetrievalMode

app = typer.Typer(no_args_is_help=True, add_completion=False)

# The things a judge decides about an answer. The first is the one a command means when
# none is named.
CRITERIA: tuple[Criterion[Any], ...] = (SUPPORT, CORRECT)
DEFAULT_CRITERION = SUPPORT
CriterionName = Literal["support", "correct"]


def _criterion(name: str) -> Criterion[Any]:
    return next(criterion for criterion in CRITERIA if criterion.name == name)


# How long an evaluation waits after a request fails. An overloaded model usually
# recovers within a minute or two, and waiting costs no request.
REST_AFTER_FAILURE_SECONDS = 30.0

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
) -> None:
    """Answer every golden question and score what code can check. One real request each.

    Replies are saved as they arrive, so a run that stops, or is stopped, continues where
    it left off when the command is run again.
    """
    settings = get_settings()
    config = {**wiring.describe_retrieval(settings), **wiring.describe_generation(settings)}
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
        generator = wiring.make_generator(settings, attempts=1)
        # Last, so that a run that cannot start leaves nothing behind.
        saved = answer_runs.open_run(run_dir, config, restart=restart)
        if restart:
            # Verdicts on the discarded replies go with them.
            for criterion in CRITERIA:
                criterion.run_stage(run_dir).discard()
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
        or f"Answers by {settings.answer_model} from {settings.answer_context_chunks} passages.",
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


def _score_answers(
    run_dir: Path,
    questions: Sequence[GoldenQuestion],
    records: Mapping[str, answer_runs.AnswerRecord],
    failures: Sequence[FailedQuestion],
    chunks: Sequence[Chunk],
) -> tuple[answer_runs.AnswerReport, dict[str, dict[str, Judgement]]]:
    """Score a run's replies, with the judges' verdicts on them where they have any.

    The verdicts come back by the id of the score they give, and then by question.
    """
    records = answer_runs.replies_to_current_labels(questions, records)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    judged: dict[str, dict[str, Judgement]] = {}
    for criterion in CRITERIA:
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
        for criterion in CRITERIA
        if criterion.metric in judged
    }
    report = answer_runs.summarise_answers(questions, records, failures, judged=passed or None)
    return report, judged


def _judge_records(
    run_dir: Path, judged: Mapping[str, Mapping[str, Judgement]]
) -> dict[str, dict[str, object]]:
    """For each judged score of a run: who judged, what it cost, and how it was checked."""
    records: dict[str, dict[str, object]] = {}
    for criterion in CRITERIA:
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
    report: answer_runs.AnswerReport, judges: Mapping[str, Mapping[str, Any]]
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
    for criterion in CRITERIA:
        if criterion.metric in judges:
            _echo_judge_record(criterion, judges[criterion.metric])


def _check_command(criterion: Criterion[Any]) -> str:
    """The command that checks the judge of a criterion, as a person types it."""
    if criterion is DEFAULT_CRITERION:
        return "eval-judge"
    return f"eval-judge --criterion {criterion.name}"


def _echo_judge_record(criterion: Criterion[Any], judge: Mapping[str, Any]) -> None:
    """Say who judged, and how far that judge agreed with labelled answers."""
    counts = judge["verdicts"]
    typer.echo(
        f"  judge        {judge['config']['judge_model']} read {judge['answers_judged']} "
        "answers: "
        + ", ".join(
            f"{counts[verdict]} {criterion.verdict_words[verdict]}"
            for verdict in criterion.verdicts
        )
    )
    check = judge.get("check")
    if not check:
        typer.echo(
            "  judge check  none. This judge has not been checked against labelled answers, "
            f"so its score cannot be trusted yet. Run {_check_command(criterion)}, then "
            f"eval-{criterion.name} --checked."
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
    command = _check_command(criterion)
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


def _refuse_the_check_of_another_judge(criterion: Criterion[Any], run_dir: Path) -> None:
    """Refuse a directory that holds a check of the judge of another criterion.

    The two checks use the same file names. Without this, restarting a check with the
    directory of the other one would discard verdicts that took a share of a day's
    requests to make.
    """
    summary = run_dir / "summary.json"
    if not summary.is_file():
        return
    try:
        held = json.loads(summary.read_text(encoding="utf-8")).get("criterion")
    except (json.JSONDecodeError, AttributeError):
        return
    if isinstance(held, str) and held != criterion.name:
        raise SavedRunError(
            f"{run_dir} holds a check of the judge for '{held}', not for "
            f"'{criterion.name}'. Give this check another --output or --variant."
        )


def _judge_stopped_badly(run: JudgeRun, max_requests: int | None) -> bool:
    """Whether a judge run ended in a way the caller did not plan."""
    planned = max_requests is not None and run.attempts >= max_requests
    return bool(run.failures) or (run.stopped is not None and not planned)


@app.command(name="eval-judge")
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
    criterion = _criterion(criterion_name)
    config = wiring.describe_judge(settings, criterion)
    cases = cases or criterion.labelled_path
    run_dir = (output or criterion.checks_dir) / variant
    stage = judge_check_stage(run_dir)
    try:
        _refuse_the_check_of_another_judge(criterion, run_dir)
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
    typer.echo(f"  judge        {settings.judge_model}")
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
        "description": f"{settings.judge_model} judging the labelled answers in {cases}.",
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
    _echo_estimate("Faults caught (recall)", agreement.recall)
    _echo_estimate("Alarms that were right", agreement.precision)
    typer.echo(f"  {'Kappa':<28} {agreement.kappa:.3f}")
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
    """Judge a run's saved answers by one criterion, and add its score to the run."""
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
    typer.echo(f"  judge        {settings.judge_model}")
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
    report, all_judged = _score_answers(
        run_dir, questions, records, answer_runs.load_failures(run_dir), corpus.chunks
    )
    judged = all_judged.get(criterion.metric, {})
    judges = _judge_records(run_dir, all_judged)
    answers_stage = answer_runs.answers_stage(run_dir)
    # Written whether or not this judge ruled on anything. After a restart, or when no
    # saved reply is to a question as it is labelled now, the run's files must not go on
    # showing a score that nothing stands behind.
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
    _echo_answer_scores(report, judges)
    typer.echo(f"  written to   {run_dir}")
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
    "--restart", help="Discard the verdicts saved for this run, and start again."
)


@app.command(name="eval-support")
def eval_support(
    dataset: Annotated[Path, _DATASET_OPTION] = Path("evals/golden/questions.jsonl"),
    variant: Annotated[str, _VARIANT_OPTION] = "current",
    runs: Annotated[Path, _RUNS_OPTION] = Path("evals/runs/answers"),
    checked: Annotated[Path | None, _CHECKED_OPTION] = None,
    max_requests: Annotated[int | None, _MAX_REQUESTS_OPTION] = None,
    restart: Annotated[bool, _RESTART_OPTION] = False,
) -> None:
    """Judge a run's saved answers: is every statement backed by the passage it cites?

    One real request for each answer that has no saved verdict. The replies themselves
    are not asked again.
    """
    _judge_answers(
        SUPPORT,
        dataset=dataset,
        variant=variant,
        runs=runs,
        checked=checked,
        max_requests=max_requests,
        restart=restart,
    )


@app.command(name="eval-correct")
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

    # A reply that is not an answer lists no sources, even if the model attached one to it.
    if answer.answered and answer.sources:
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
    elif answer.stop_reason == "refusal":
        details.append("withheld by the model's provider")
    elif answer.stop_reason not in ("end_turn", "no_context"):
        details.append(f"stopped early ({answer.stop_reason})")
    typer.echo("\n" + " | ".join(details))
