"""Run the retrieval evaluation: ask every question, and score what comes back.

This measures retrieval alone. No language model is involved, so a run is free,
deterministic and fast, and can be repeated after every change.
"""

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from handbook_rag.evals.dataset import GoldenQuestion, locate_evidence
from handbook_rag.evals.metrics import (
    Estimate,
    covered_evidence,
    hit_at_k,
    mean_with_interval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from handbook_rag.ingest.models import Document
from handbook_rag.retrieve.dense import Retriever

HIT_CUTOFFS = (1, 3, 5, 10)
RANKING_CUTOFF = 10
RETRIEVAL_DEPTH = max(*HIT_CUTOFFS, RANKING_CUTOFF)

# The headline metric comes first. Hit@5 matches what the model is given: five chunks.
METRICS: list[dict[str, str | int]] = [
    {"id": "hit@5", "label": "Hit@5", "kind": "binary"},
    {"id": "hit@1", "label": "Hit@1", "kind": "binary"},
    {"id": "hit@3", "label": "Hit@3", "kind": "binary"},
    {"id": "hit@10", "label": "Hit@10", "kind": "binary"},
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


@dataclass(frozen=True, slots=True)
class FailedQuestion:
    """A question that could not be scored because the harness failed, not the retriever."""

    question_id: str
    error_class: str
    message: str


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Everything one evaluation run produced."""

    results: list[QuestionResult]
    failures: list[FailedQuestion]
    not_scored: int
    """Unanswerable questions. They have no evidence, so retrieval cannot be scored on them."""
    summary: dict[str, Estimate] = field(default_factory=dict)
    by_group: dict[str, dict[str, dict[str, Estimate]]] = field(default_factory=dict)


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
        scores = {f"hit@{k}": hit_at_k(coverage, k) for k in HIT_CUTOFFS}
        scores["recall@5"] = recall_at_k(coverage, len(spans), 5)
        scores[f"mrr@{RANKING_CUTOFF}"] = reciprocal_rank_at_k(coverage, RANKING_CUTOFF)
        scores[f"ndcg@{RANKING_CUTOFF}"] = ndcg_at_k(coverage, len(spans), RANKING_CUTOFF)
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
        results.append(QuestionResult(question, scores, items, latency))

    return RetrievalReport(
        results=results,
        failures=failures,
        not_scored=len(questions) - len(answerable),
        summary=_summarise(results),
        by_group={
            "origin": _summarise_groups(results, lambda r: r.question.origin),
            "section": _summarise_groups(results, lambda r: r.question.section),
        },
    )


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
    summary = {
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
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "change.md").write_text(f"{description}\n", encoding="utf-8")
    return run_dir
