"""A retrieval check that is quick enough for CI: a slice of the corpus, the real models.

The full evaluation needs the whole handbook embedded, which takes too long to run on
every pull request. The gate keeps the pages that hold the golden set's evidence, adds a
fixed sample of other pages, indexes that slice in memory and scores the default
retriever on it.

A slice is easier to search than the whole handbook, so the gate's scores say nothing
about quality in absolute terms. What the gate can say is that a change broke retrieval:
its scores are compared with minimums recorded when retrieval was known to work.
"""

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from handbook_rag.evals.dataset import GoldenQuestion
from handbook_rag.evals.retrieval import METRIC_IDS, RetrievalReport
from handbook_rag.ingest.models import Document


class GateError(ValueError):
    """The gate's settings file is missing or malformed."""


class GateConfig(BaseModel):
    """What the gate indexes and what it demands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distractor_pages: int = Field(ge=0)
    """How many pages without evidence are added, so the search has something to confuse."""
    seed: int
    """Decides which pages those are. The same seed always picks the same pages."""
    minimums: dict[str, float]
    """The lowest acceptable mean for each metric, for example ``{"hit@5": 0.95}``."""
    note: str = ""
    """Free text: where the minimums come from. JSON has no comments, so it lives here."""

    @field_validator("minimums")
    @classmethod
    def _minimums_are_usable(cls, minimums: dict[str, float]) -> dict[str, float]:
        if not minimums:
            raise ValueError("a gate without minimums checks nothing")
        for metric, minimum in minimums.items():
            if metric not in METRIC_IDS:
                raise ValueError(f"{metric} is not a metric; choose from {', '.join(METRIC_IDS)}")
            if not 0 < minimum <= 1:
                raise ValueError(f"the minimum for {metric} must be above 0 and at most 1")
        return minimums


def load_gate_config(path: Path) -> GateConfig:
    """Read the gate's settings from a JSON file."""
    try:
        return GateConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise GateError(f"gate settings not found: {path}") from error
    except (json.JSONDecodeError, ValidationError) as error:
        raise GateError(f"{path} is not a valid gate settings file: {error}") from error


def evidence_pages(questions: Sequence[GoldenQuestion]) -> set[str]:
    """Every page that some question's evidence points to."""
    return {evidence.doc_id for question in questions for evidence in question.evidence}


def select_pages(
    documents: Sequence[Document],
    questions: Sequence[GoldenQuestion],
    *,
    distractors: int,
    seed: int,
) -> list[Document]:
    """The pages the gate indexes: every page with evidence, and a fixed sample of the rest.

    The sample is drawn with a seed from the sorted page names, so the same snapshot
    always gives the same slice, on any machine.
    """
    needed = evidence_pages(questions)
    others = sorted(document.doc_id for document in documents if document.doc_id not in needed)
    sampled = set(random.Random(seed).sample(others, min(distractors, len(others))))
    return [
        document
        for document in documents
        if document.doc_id in needed or document.doc_id in sampled
    ]


def gate_problems(report: RetrievalReport, minimums: Mapping[str, float]) -> list[str]:
    """Every reason the gate fails, one line each. An empty list means it passes.

    A question that could not be searched fails the gate. The evaluation keeps such
    failures apart from the scores, so that a score describes the retriever and not the
    plumbing. A gate that passed while part of the set went unmeasured would be worse
    than no gate, so here they count.
    """
    problems: list[str] = []
    if report.failures:
        first = report.failures[0]
        problems.append(
            f"{len(report.failures)} of {len(report.failures) + len(report.results)} questions "
            f"could not be searched; the first was {first.question_id}: "
            f"{first.error_class}: {first.message}"
        )
    if not report.results:
        return problems or ["no question could be scored"]
    for metric, minimum in minimums.items():
        estimate = report.summary.get(metric)
        if estimate is None:
            problems.append(f"{metric} was not measured")
        elif estimate.mean < minimum:
            problems.append(f"{metric} is {estimate.mean:.3f}, below the minimum of {minimum:.3f}")
    return problems
