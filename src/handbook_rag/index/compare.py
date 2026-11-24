"""Compare two collections point by point: ids, payloads, and vectors within a tolerance.

Made to check an updated index against a full one. Ids and payloads must be equal. A
vector cannot be expected to be equal to the last bit: a model that embeds the same text
in another batch can give a slightly different vector, and the store held in memory
rounds a vector again when it stores it. So vectors are compared within a tolerance, and
the largest difference found is reported whatever the tolerance, to show how close they
are.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import ReadableStore, StoredPoint

DEFAULT_VECTOR_TOLERANCE = 1e-4
"""How far apart two entries of a dense vector may be and still count as the same.

A dense vector of the embedding model has 384 entries and a length of 1, so an entry is
about 0.05 on average. Over the whole handbook, the largest difference between an updated
index and a full one was 5e-09 (ADR 0019); the reuse check prints the largest it finds.
"""


@dataclass(frozen=True, slots=True)
class CollectionComparison:
    """What differs between a reference collection and a candidate."""

    reference: str
    candidate: str
    tolerance: float
    compared: int
    """How many points are in both collections."""
    missing: list[str]
    """Points in the reference only."""
    extra: list[str]
    """Points in the candidate only."""
    payloads: list[tuple[str, str]]
    """A point and a payload field whose value differs."""
    vectors: list[tuple[str, str, float]]
    """A point, ``dense`` or ``sparse``, and how far apart its two vectors are."""
    largest_dense_difference: float
    largest_sparse_difference: float
    smallest_cosine: float
    """The least similar pair of dense vectors, as the cosine of the angle between them."""

    @property
    def equal(self) -> bool:
        return not (self.missing or self.extra or self.payloads or self.vectors)

    def differences(self) -> list[str]:
        """Every difference, one line each, naming the point and what differs."""
        return [
            *(f"missing  {point}: in {self.reference} only" for point in self.missing),
            *(f"extra    {point}: in {self.candidate} only" for point in self.extra),
            *(f"payload  {point}: {field} differs" for point, field in self.payloads),
            *(
                f"{vector:<8} {point}: differs by up to {size:.3g}"
                for point, vector, size in self.vectors
            ),
        ]


def compare_collections(
    reference: ReadableStore,
    candidate: ReadableStore,
    *,
    tolerance: float,
    batch_size: int = 256,
) -> CollectionComparison:
    """Compare ``candidate`` with ``reference``. Both are only read.

    Two vectors are the same when no entry of one is further than ``tolerance`` from the
    same entry of the other. An entry that is not a number is never the same as
    anything, and it makes the largest difference reported not a number either, so it
    cannot pass unseen. For a keyword vector, a word that only one of them holds counts
    as a weight of zero in the other, so it differs by its whole weight; that is caught
    by any tolerance smaller than the smallest weight.

    Points are read a batch at a time, so two whole collections are never held at once.
    """
    if not (math.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(f"The tolerance must be a finite number of at least 0, not {tolerance}.")
    ours, theirs = set(reference.point_ids()), set(candidate.point_ids())
    shared = sorted(ours & theirs)
    payloads: list[tuple[str, str]] = []
    vectors: list[tuple[str, str, float]] = []
    largest_dense = largest_sparse = 0.0
    smallest_cosine = 1.0
    for start in range(0, len(shared), batch_size):
        batch = shared[start : start + batch_size]
        expected, found = reference.points(batch), candidate.points(batch)
        for point_id in batch:
            first, second = expected[point_id], found[point_id]
            payloads.extend((point_id, field) for field in _payload_differences(first, second))
            dense = _dense_difference(first.dense, second.dense)
            sparse = _sparse_difference(first.sparse, second.sparse)
            largest_dense = _worst(largest_dense, dense, max)
            largest_sparse = _worst(largest_sparse, sparse, max)
            smallest_cosine = _worst(smallest_cosine, _cosine(first.dense, second.dense), min)
            # Asked this way round, a difference that is not a number is reported.
            if not dense <= tolerance:
                vectors.append((point_id, "dense", dense))
            if not sparse <= tolerance:
                vectors.append((point_id, "sparse", sparse))
    return CollectionComparison(
        reference=reference.collection,
        candidate=candidate.collection,
        tolerance=tolerance,
        compared=len(shared),
        missing=sorted(ours - theirs),
        extra=sorted(theirs - ours),
        payloads=payloads,
        vectors=vectors,
        largest_dense_difference=largest_dense,
        largest_sparse_difference=largest_sparse,
        smallest_cosine=smallest_cosine,
    )


def _payload_differences(first: StoredPoint, second: StoredPoint) -> list[str]:
    """The payload fields whose values differ, including a field only one point has."""
    fields = sorted(set(first.payload) | set(second.payload))
    absent = object()
    return [
        field
        for field in fields
        if first.payload.get(field, absent) != second.payload.get(field, absent)
    ]


def _dense_difference(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        return math.inf
    return _largest(abs(a - b) for a, b in zip(first, second, strict=True))


def _sparse_difference(first: SparseVector, second: SparseVector) -> float:
    ours = dict(zip(first.indices, first.values, strict=True))
    theirs = dict(zip(second.indices, second.values, strict=True))
    words = set(ours) | set(theirs)
    return _largest(abs(ours.get(word, 0.0) - theirs.get(word, 0.0)) for word in words)


def _largest(differences: Iterable[float]) -> float:
    """The largest difference, or not a number if one of them is not.

    ``max`` alone would keep or drop a value that is not a number depending on where it
    stands, because every comparison with one is false.
    """
    largest = 0.0
    for difference in differences:
        if math.isnan(difference):
            return math.nan
        largest = max(largest, difference)
    return largest


def _worst(current: float, value: float, choose: Callable[[float, float], float]) -> float:
    """What ``choose`` makes of the two, except that a value that is not a number wins."""
    if math.isnan(current) or math.isnan(value):
        return math.nan
    return choose(current, value)


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        return 0.0
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    lengths = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    # A zero vector has no direction, so it says nothing about an angle. How far its
    # entries are from the other vector's is still reported as its difference.
    return dot / lengths if lengths else 1.0
