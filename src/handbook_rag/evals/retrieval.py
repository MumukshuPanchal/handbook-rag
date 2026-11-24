"""Run the retrieval evaluation: ask every question, and score what comes back.

This measures retrieval alone. No language model is involved, so a run is free,
deterministic and fast, and can be repeated after every change.
"""

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from handbook_rag.evals.dataset import GoldenQuestion, locate_evidence
from handbook_rag.evals.metrics import (
    Estimate,
    all_at_k,
    covered_evidence,
    first_ranks,
    hit_at_k,
    mean_with_interval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from handbook_rag.ingest.models import Document
from handbook_rag.retrieve.dense import Retriever

HIT_CUTOFFS = (1, 3, 5, 10)
# Passages on two pages are never both at rank 1, and for a question with one passage
# All@1 is Hit@1, so "every passage found" starts at three results.
ALL_CUTOFFS = (3, 5, 10)
RANKING_CUTOFF = 10
RETRIEVAL_DEPTH = max(*HIT_CUTOFFS, *ALL_CUTOFFS, RANKING_CUTOFF)

# The headline metric comes first. Hit@5 matches what the model is given: five chunks.
# The All@k scores came later and stand beside the Hit@k scores: for a question that needs
# two passages, a hit means that one was found, and All@k that both were.
METRICS: list[dict[str, str | int]] = [
    {"id": "hit@5", "label": "Hit@5", "kind": "binary"},
    {"id": "hit@1", "label": "Hit@1", "kind": "binary"},
    {"id": "hit@3", "label": "Hit@3", "kind": "binary"},
    {"id": "hit@10", "label": "Hit@10", "kind": "binary"},
    {"id": "all@5", "label": "All@5", "kind": "binary"},
    {"id": "all@3", "label": "All@3", "kind": "binary"},
    {"id": "all@10", "label": "All@10", "kind": "binary"},
    {"id": "recall@5", "label": "Recall@5", "kind": "float", "scale": 1},
    {"id": "mrr@10", "label": "MRR@10", "kind": "float", "scale": 1},
    {"id": "ndcg@10", "label": "nDCG@10", "kind": "float", "scale": 1},
]
METRIC_IDS = tuple(str(metric["id"]) for metric in METRICS)


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One retrieved chunk, with the evidence it covers."""

    rank: int
    chunk_id: str
    doc_id: str
    context_header: str
    score: float
    covers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """The scores for one question."""

    question: GoldenQuestion
    scores: dict[str, float]
    retrieved: tuple[RetrievedItem, ...]
    latency_s: float
    evidence_ranks: tuple[int | None, ...] = ()
    """For each piece of evidence, in the order the question lists them, the rank of the
    first retrieved chunk that covers it. ``None``: not among the retrieved chunks."""
    page_ranks: tuple[int | None, ...] = ()
    """The same for the page of each piece: the rank of the first retrieved chunk from
    that page, whether or not it covers the quote. Never later than the rank above."""


@dataclass(frozen=True, slots=True)
class FailedQuestion:
    """A question that could not be scored because the harness failed, not the retriever."""

    question_id: str
    error_class: str
    message: str


@dataclass(frozen=True, slots=True)
class SeveralPassages:
    """How the search did on the questions whose answer needs two or more passages.

    The means of All@k and Hit@k say the same thing as shares. These are the counts
    behind them, and the one thing the means cannot say: how far down the last passage
    came when it was found late.
    """

    questions: int
    found_in_top: dict[int, dict[str, int]]
    """For each cutoff k: the questions with ``all`` their passages among the first k
    results, with ``some`` but not all, and with ``none``."""
    all_found_by_rank: dict[int, int]
    """The rank at which the results first held every passage, to the number of questions."""
    not_all_retrieved: int
    """Questions with a passage that is not among the retrieved chunks at all."""
    pages_in_top: dict[int, dict[str, int]]
    """The counts of ``found_in_top`` again, with any chunk of a passage's page counted
    as finding it. A passage is credited by its quote, and a page sometimes says the same
    thing in a second place, so ``found_in_top`` can count too few. This counts too many:
    another chunk of the right page usually does not hold the fact. What the model could
    have used lies between the two."""


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Everything one evaluation run produced."""

    results: list[QuestionResult]
    failures: list[FailedQuestion]
    not_scored: int
    """Unanswerable questions. They have no evidence, so retrieval cannot be scored on them."""
    summary: dict[str, Estimate] = field(default_factory=dict)
    by_group: dict[str, dict[str, dict[str, Estimate]]] = field(default_factory=dict)
    several: SeveralPassages | None = None
    """``None`` when no scored question needs more than one passage."""


def evaluate_retrieval(
    questions: Sequence[GoldenQuestion],
    documents: Sequence[Document],
    retriever: Retriever,
) -> RetrievalReport:
    """Score the retriever on every answerable question.

    A harness failure, such as the vector store being unreachable, is recorded separately
    and never counted as a miss: a score must say something about the retriever, not
    about the plumbing around it. A broken label, such as a quote that is not in its
    page, stops the run, because every number that depends on it would be wrong.
    """
    pages = {document.doc_id: document for document in documents}
    answerable = [q for q in questions if q.type == "answerable"]
    spans_by_question = {q.id: locate_evidence(q, pages) for q in answerable}

    results: list[QuestionResult] = []
    failures: list[FailedQuestion] = []
    for question in answerable:
        spans = spans_by_question[question.id]
        started = time.perf_counter()
        try:
            retrieved = retriever.retrieve(question.question, limit=RETRIEVAL_DEPTH)
        except Exception as error:
            failures.append(FailedQuestion(question.id, type(error).__name__, str(error)))
            continue
        latency = time.perf_counter() - started

        coverage = [covered_evidence(result.chunk, spans) for result in retrieved]
        scored, needed = coverage, len(spans)
        if question.evidence_mode == "any":
            # Each piece answers the question on its own, so together they count as one.
            scored = [frozenset({0}) if covered else frozenset() for covered in coverage]
            needed = 1
        scores = {f"hit@{k}": hit_at_k(scored, k) for k in HIT_CUTOFFS}
        scores.update({f"all@{k}": all_at_k(scored, needed, k) for k in ALL_CUTOFFS})
        scores["recall@5"] = recall_at_k(scored, needed, 5)
        scores[f"mrr@{RANKING_CUTOFF}"] = reciprocal_rank_at_k(scored, RANKING_CUTOFF)
        scores[f"ndcg@{RANKING_CUTOFF}"] = ndcg_at_k(scored, needed, RANKING_CUTOFF)
        items = tuple(
            RetrievedItem(
                rank=rank,
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                context_header=result.chunk.context_header,
                score=result.score,
                covers=tuple(sorted(covered)),
            )
            for rank, (result, covered) in enumerate(zip(retrieved, coverage, strict=True), 1)
        )
        same_page = [
            frozenset(index for index, span in enumerate(spans) if span.doc_id == item.doc_id)
            for item in items
        ]
        results.append(
            QuestionResult(
                question,
                scores,
                items,
                latency,
                evidence_ranks=first_ranks(coverage, len(spans)),
                page_ranks=first_ranks(same_page, len(spans)),
            )
        )

    return RetrievalReport(
        results=results,
        failures=failures,
        not_scored=len(questions) - len(answerable),
        summary=_summarise(results),
        by_group={
            "origin": _summarise_groups(results, lambda r: r.question.origin),
            "section": _summarise_groups(results, lambda r: r.question.section),
        },
        several=_summarise_several(results),
    )


def needs_several_passages(question: GoldenQuestion) -> bool:
    """Whether the answer needs two or more passages, and not any one of several."""
    return question.evidence_mode == "all" and len(question.evidence) > 1


def _summarise_several(results: Sequence[QuestionResult]) -> SeveralPassages | None:
    needing = [result for result in results if needs_several_passages(result.question)]
    if not needing:
        return None
    all_found_by_rank: Counter[int] = Counter()
    not_all_retrieved = 0
    for result in needing:
        ranks = [rank for rank in result.evidence_ranks if rank is not None]
        if len(ranks) < len(result.evidence_ranks):
            not_all_retrieved += 1
        else:
            # The results hold every passage from the rank of the one that came last.
            all_found_by_rank[max(ranks)] += 1
    return SeveralPassages(
        questions=len(needing),
        found_in_top=_found_in_top([result.evidence_ranks for result in needing]),
        all_found_by_rank=dict(sorted(all_found_by_rank.items())),
        not_all_retrieved=not_all_retrieved,
        pages_in_top=_found_in_top([result.page_ranks for result in needing]),
    )


def _found_in_top(ranks: Sequence[Sequence[int | None]]) -> dict[int, dict[str, int]]:
    """For each cutoff, the questions with all, some or none of their ranks within it."""
    found_in_top: dict[int, dict[str, int]] = {}
    for k in ALL_CUTOFFS:
        counts = {"all": 0, "some": 0, "none": 0}
        for of_question in ranks:
            found = sum(rank is not None and rank <= k for rank in of_question)
            if found == len(of_question):
                counts["all"] += 1
            else:
                counts["some" if found else "none"] += 1
        found_in_top[k] = counts
    return found_in_top


def several_passages_lines(several: SeveralPassages) -> list[str]:
    """What a command prints about the questions that need two or more passages."""
    lines = [f"  questions that need several passages: {several.questions}"]
    for k, counts in several.found_in_top.items():
        lines.append(
            f"    in the top {k:<3} all found {counts['all']}, some {counts['some']}, "
            f"none {counts['none']}"
        )
    by_rank = ", ".join(f"{rank}: {count}" for rank, count in several.all_found_by_rank.items())
    lines.append(
        f"    all found by rank  {by_rank or 'never'}; "
        f"not all in the top {RETRIEVAL_DEPTH}: {several.not_all_retrieved}"
    )
    pages = ", ".join(f"top {k}: {counts['all']}" for k, counts in several.pages_in_top.items())
    lines.append(f"    a chunk of every passage's page, whatever it says, in the {pages}")
    return lines


def _summarise(results: Sequence[QuestionResult]) -> dict[str, Estimate]:
    if not results:
        return {}
    return {
        metric: mean_with_interval([result.scores[metric] for result in results])
        for metric in METRIC_IDS
    }


def _summarise_groups(
    results: Sequence[QuestionResult], key: Callable[[QuestionResult], str]
) -> dict[str, dict[str, Estimate]]:
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {name: _summarise(members) for name, members in sorted(groups.items())}


def write_run(
    report: RetrievalReport,
    flow_dir: Path,
    variant: str,
    description: str,
    config: Mapping[str, object] | None = None,
) -> Path:
    """Write one run to ``flow_dir/variant`` and return that directory.

    The layout is one directory per variant (``baseline``, ``v1``, ...) under a shared
    flow directory, with one row per question in ``results.jsonl``. Failed questions go
    to ``errors.jsonl`` so they can never be mistaken for scored ones.

    ``results.jsonl`` holds only what a comparison needs, so it stays small enough to
    commit. The ranked chunks behind each score go to ``retrieved.jsonl`` and ``traces/``,
    which are for debugging on the machine that ran the evaluation.

    A number is only useful if it can be traced to what produced it. The summary records
    ``config`` (which models and settings were used) and a fingerprint of the labels that
    were scored, and every row carries the fingerprint of its own label.

    When some question needs two or more passages, the summary also counts for how many
    of them the results held all, some or none of the passages, and at which rank the
    last one came. Every row says where each of its passages first came, and where the
    first chunk of that passage's page came.
    """
    run_dir = flow_dir / variant
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (flow_dir / "_state.json").write_text(
        json.dumps(
            {"metrics": METRICS, "perf_fields": [{"id": "latency_s", "unit": "s"}]}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        (run_dir / "results.jsonl").open("w", encoding="utf-8") as output,
        (run_dir / "retrieved.jsonl").open("w", encoding="utf-8") as details,
    ):
        for result in report.results:
            question = result.question
            row = {
                "prompt_id": question.id,
                "prompt": question.question,
                "fingerprint": question.fingerprint,
                "tags": [question.section, question.origin, f"review:{question.review}"],
                "status": "ok",
                "grade": result.scores,
                # Not scores: where each piece of evidence first came, and where the first
                # chunk of its page came, so that "found late" can be read from a committed
                # file. null: not among the results.
                "evidence_ranks": list(result.evidence_ranks),
                "page_ranks": list(result.page_ranks),
                "latency_s": round(result.latency_s, 4),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            detail = {
                "prompt_id": question.id,
                "retrieved": [asdict(item) for item in result.retrieved],
            }
            details.write(json.dumps(detail, ensure_ascii=False) + "\n")
            trace = [
                {"role": "user", "content": question.question},
                {
                    "role": "tool_result",
                    "content": "\n".join(
                        f"{item.rank}. {'RELEVANT' if item.covers else 'not relevant'}  "
                        f"{item.score:.3f}  {item.context_header}  ({item.doc_id})"
                        for item in result.retrieved
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Expected evidence:\n"
                    + "\n".join(f"- {e.doc_id}: {e.quote}" for e in question.evidence),
                },
            ]
            (run_dir / "traces" / f"{question.id}_rep0.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as output:
        for failure in report.failures:
            output.write(json.dumps(asdict(failure), ensure_ascii=False) + "\n")

    labels = sorted(f"{r.question.id}:{r.question.fingerprint}" for r in report.results)
    summary: dict[str, object] = {
        "description": description,
        "config": dict(config or {}),
        "labels_fingerprint": hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()[:16],
        "questions_scored": len(report.results),
        "questions_failed": len(report.failures),
        "questions_not_scored": report.not_scored,
        "metrics": {name: asdict(estimate) for name, estimate in report.summary.items()},
        "by_group": {
            group: {
                name: {metric: asdict(estimate) for metric, estimate in estimates.items()}
                for name, estimates in members.items()
            }
            for group, members in report.by_group.items()
        },
    }
    if report.several is not None:
        # Counts, for the questions whose answer needs two or more passages. JSON turns
        # the cutoffs and the ranks into text.
        summary["needs_several_passages"] = {
            **asdict(report.several),
            "retrieved": RETRIEVAL_DEPTH,
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "change.md").write_text(f"{description}\n", encoding="utf-8")
    return run_dir
