"""Retrieval metrics.

All of them start from the same question: for each retrieved chunk, in ranked order,
which pieces of evidence does it cover? From that, each metric answers something
different:

- **Hit@k**: did at least one relevant chunk appear in the top k? The model can only
  answer from what it is given, so this is the ceiling on answer quality.
- **All@k**: did the top k cover every piece of the evidence? For a question that needs
  two passages, Hit@k is already 1 when one of them is found, and the model then answers
  from half of what it needs. All@k is 1 only when both are there. For a question with
  one passage it is the same as Hit@k.
- **Recall@k**: what share of the evidence was covered by the top k? It differs from
  Hit@k only for questions that need more than one passage.
- **MRR@k**: how high was the first relevant chunk? 1 for rank 1, 1/2 for rank 2, and so
  on, 0 if none in the top k. It rewards putting the right passage first.
- **nDCG@k**: a ranking score between 0 and 1 that rewards relevant chunks more the
  higher they appear, compared with the best possible ordering.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from handbook_rag.evals.dataset import EvidenceSpan
from handbook_rag.ingest.models import Chunk

# A chunk counts as covering a piece of evidence when it contains at least this share of
# the quoted text. Requiring all of it would punish a chunk boundary that falls inside
# the quote; requiring any overlap would reward a chunk that only clips its edge.
MIN_COVERAGE = 0.5


def covered_evidence(chunk: Chunk, spans: Sequence[EvidenceSpan]) -> frozenset[int]:
    """Indices of the evidence spans this chunk covers."""
    covered: set[int] = set()
    for index, span in enumerate(spans):
        if chunk.doc_id != span.doc_id:
            continue
        overlap = min(chunk.end_char, span.end) - max(chunk.start_char, span.start)
        if overlap >= MIN_COVERAGE * (span.end - span.start):
            covered.add(index)
    return frozenset(covered)


def hit_at_k(coverage: Sequence[frozenset[int]], k: int) -> float:
    return 1.0 if any(coverage[:k]) else 0.0


def all_at_k(coverage: Sequence[frozenset[int]], evidence_count: int, k: int) -> float:
    """Whether the top k cover every piece of evidence, and not just one of them."""
    if evidence_count == 0:
        raise ValueError("All@k is undefined for a question with no evidence.")
    found: set[int] = set().union(*coverage[:k])
    return 1.0 if len(found) == evidence_count else 0.0


def first_ranks(coverage: Sequence[frozenset[int]], evidence_count: int) -> tuple[int | None, ...]:
    """For each piece of evidence, the rank of the first chunk that covers it.

    ``None`` for a piece that no chunk in the ranking covers. The ranking is as long as
    the search was asked for, so ``None`` means "not among those results", and says
    nothing about where the piece would have come further down.
    """
    ranks: list[int | None] = [None] * evidence_count
    for rank, covered in enumerate(coverage, start=1):
        for index in covered:
            if ranks[index] is None:
                ranks[index] = rank
    return tuple(ranks)


def recall_at_k(coverage: Sequence[frozenset[int]], evidence_count: int, k: int) -> float:
    if evidence_count == 0:
        raise ValueError("Recall is undefined for a question with no evidence.")
    found: set[int] = set().union(*coverage[:k])
    return len(found) / evidence_count


def reciprocal_rank_at_k(coverage: Sequence[frozenset[int]], k: int) -> float:
    for rank, covered in enumerate(coverage[:k], start=1):
        if covered:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(coverage: Sequence[frozenset[int]], evidence_count: int, k: int) -> float:
    """Normalised discounted cumulative gain with one point of gain per piece of evidence.

    A chunk earns gain only for evidence that no higher-ranked chunk already covered.
    Overlapping chunks can cover the same passage, and counting it twice would reward
    returning the same information repeatedly.

    The ideal ranking brings one new piece of evidence per position. A single chunk that
    covers several pieces does better than that, so the score is capped at 1.
    """
    if evidence_count == 0:
        raise ValueError("nDCG is undefined for a question with no evidence.")
    seen: set[int] = set()
    gain = 0.0
    for rank, covered in enumerate(coverage[:k], start=1):
        gain += len(covered - seen) / math.log2(rank + 1)
        seen |= covered
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(evidence_count, k) + 1))
    return min(1.0, gain / ideal)


@dataclass(frozen=True, slots=True)
class Estimate:
    """A mean with a 95% confidence interval."""

    mean: float
    low: float
    high: float
    count: int


def mean_with_interval(
    values: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> Estimate:
    """The mean of per-question scores, with a bootstrap 95% confidence interval.

    A golden set is a sample of the questions people could ask. The interval shows how
    much the mean would move with a different sample of the same size. A difference
    between two systems that is smaller than this is not evidence of anything.

    The bootstrap resamples the questions with replacement many times and takes the
    middle 95% of the resulting means. It is seeded, so the same scores always give the
    same interval.
    """
    if not values:
        raise ValueError("Cannot average an empty list of scores.")
    mean = sum(values) / len(values)
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choices(values, k=len(values))) / len(values) for _ in range(resamples)
    )
    return Estimate(
        mean=mean,
        low=means[int(0.025 * resamples)],
        high=means[int(0.975 * resamples) - 1],
        count=len(values),
    )


def share_with_interval(successes: int, total: int) -> Estimate:
    """A share of a count, with a 95% Wilson score interval.

    This is for a plain share, such as 41 faults caught of 41. The bootstrap above gives
    such a result an interval with no width, as if nothing were left in doubt, because
    every resample of 41 successes is again 41 successes. The Wilson interval stays
    honest at 0 and at 1: for 41 of 41 it runs from 0.914 to 1, which says that a judge
    missing one fault in twelve could still have produced this result.
    """
    if total <= 0:
        raise ValueError("A share needs a count to be a share of.")
    if not 0 <= successes <= total:
        raise ValueError("A share cannot be below nothing or above the whole.")
    z = 1.959964  # the 97.5th percentile of the normal distribution
    share = successes / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    scale = 1 + z * z / total
    # At the two ends the formula gives exactly 0 and exactly 1. They are written out,
    # because the arithmetic leaves a remainder of rounding there.
    return Estimate(
        mean=share,
        low=0.0 if successes == 0 else (centre - spread) / scale,
        high=1.0 if successes == total else (centre + spread) / scale,
        count=total,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """How one system's scores differ from another's on the same questions."""

    difference: Estimate
    """Mean of candidate minus base, per question, with a 95% confidence interval."""
    better: int
    """Questions on which the candidate scored higher."""
    worse: int
    same: int

    @property
    def is_clear(self) -> bool:
        """Whether the interval excludes zero, so chance is an unlikely explanation."""
        return self.difference.low > 0 or self.difference.high < 0


def paired_difference(
    base: Sequence[float], candidate: Sequence[float], *, resamples: int = 2000, seed: int = 0
) -> Comparison:
    """Compare two systems that were scored on the same questions, in the same order.

    Most of the variation in a score comes from the questions: some are easy for every
    system and some are hard for every system. Comparing two separate confidence
    intervals carries that variation twice. Looking at the difference per question
    removes it, which is why a paired comparison can detect an improvement that the two
    separate intervals, overlapping, would hide.

    The interval is a bootstrap over the per-question differences. If it excludes zero,
    the difference is unlikely to be an accident of which questions were sampled.
    """
    if len(base) != len(candidate):
        raise ValueError("A paired comparison needs the same questions on both sides.")
    differences = [after - before for before, after in zip(base, candidate, strict=True)]
    return Comparison(
        difference=mean_with_interval(differences, resamples=resamples, seed=seed),
        better=sum(difference > 0 for difference in differences),
        worse=sum(difference < 0 for difference in differences),
        same=sum(difference == 0 for difference in differences),
    )
