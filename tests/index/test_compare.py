"""Comparing two collections: what a check of an updated index against a full one reports.

Most tests use a collection held in a dictionary, so that a vector can be set to exactly
what a case needs; a real store would divide it by its length first. The last test runs
on the real store.
"""

import dataclasses
import math
from collections.abc import Collection, Iterator, Sequence
from typing import Any

import pytest
from qdrant_client import QdrantClient

from handbook_rag.index.compare import compare_collections
from handbook_rag.index.sparse import SparseVector
from handbook_rag.index.store import QdrantVectorStore, StoredPoint
from handbook_rag.ingest.models import Chunk
from tests.support import make_chunk


class Points:
    """A collection held in a dictionary, readable as the comparison reads a store."""

    def __init__(self, collection: str, points: Sequence[StoredPoint]) -> None:
        self.collection = collection
        self._points = {point.point_id: point for point in points}
        self.batches: list[list[str]] = []

    def point_ids(self) -> Iterator[str]:
        return iter(self._points)

    def scroll_chunks(self, pages: Collection[str] | None = None) -> Iterator[Chunk]:
        raise AssertionError("a comparison reads points, not chunks")

    def points(self, ids: Sequence[str]) -> dict[str, StoredPoint]:
        self.batches.append(list(ids))
        return {point_id: self._points[point_id] for point_id in ids if point_id in self._points}


def point(
    point_id: str,
    dense: Sequence[float] = (1.0, 0.0),
    sparse: SparseVector | None = None,
    **payload: Any,
) -> StoredPoint:
    return StoredPoint(
        point_id=point_id,
        payload={"title": "Leave", "text": "Take 25 days.", **payload},
        dense=list(dense),
        sparse=sparse or SparseVector([1, 2], [1.0, 0.5]),
    )


def compare(
    reference: Sequence[StoredPoint], candidate: Sequence[StoredPoint], tolerance: float = 1e-4
) -> Any:
    return compare_collections(
        Points("full", reference), Points("updated", candidate), tolerance=tolerance
    )


def test_two_equal_collections_have_no_differences() -> None:
    points = [point("p1"), point("p2", dense=(0.6, 0.8))]

    comparison = compare(points, points, tolerance=0.0)

    assert comparison.equal
    assert comparison.differences() == []
    assert (comparison.reference, comparison.candidate) == ("full", "updated")
    assert (comparison.compared, comparison.tolerance) == (2, 0.0)
    assert comparison.largest_dense_difference == comparison.largest_sparse_difference == 0.0
    assert comparison.smallest_cosine == pytest.approx(1.0)


def test_a_point_the_candidate_lacks_is_named_as_missing() -> None:
    comparison = compare([point("p1"), point("p2")], [point("p2")])

    assert not comparison.equal
    assert comparison.missing == ["p1"]
    assert comparison.compared == 1
    assert comparison.differences() == ["missing  p1: in full only"]


def test_a_point_only_the_candidate_holds_is_named_as_extra() -> None:
    comparison = compare([point("p1")], [point("p1"), point("p9")])

    assert not comparison.equal
    assert comparison.extra == ["p9"]
    assert comparison.differences() == ["extra    p9: in updated only"]


def test_a_payload_field_that_differs_is_named() -> None:
    comparison = compare([point("p1")], [point("p1", title="Time off")])

    assert not comparison.equal
    assert comparison.payloads == [("p1", "title")]
    assert comparison.differences() == ["payload  p1: title differs"]


def test_a_payload_field_only_one_point_has_is_named() -> None:
    comparison = compare([point("p1")], [point("p1", fingerprint="abc")])

    assert comparison.payloads == [("p1", "fingerprint")]


def test_a_dense_vector_outside_the_tolerance_is_named_and_one_inside_is_not() -> None:
    reference = [point("near"), point("far")]
    candidate = [point("near", dense=(1.0, 0.00005)), point("far", dense=(0.6, 0.8))]

    comparison = compare(reference, candidate, tolerance=1e-4)

    assert not comparison.equal
    assert comparison.vectors == [("far", "dense", pytest.approx(0.8))]
    assert comparison.largest_dense_difference == pytest.approx(0.8)
    assert comparison.smallest_cosine == pytest.approx(0.6)
    assert comparison.differences() == ["dense    far: differs by up to 0.8"]


def test_a_difference_exactly_at_the_tolerance_is_allowed() -> None:
    comparison = compare([point("p1")], [point("p1", dense=(1.0, 0.25))], tolerance=0.25)

    assert comparison.equal
    assert comparison.largest_dense_difference == 0.25


def test_a_keyword_weight_outside_the_tolerance_is_named() -> None:
    reference = [point("p1", sparse=SparseVector([1, 2], [1.0, 0.5]))]
    candidate = [point("p1", sparse=SparseVector([2, 1], [0.75, 1.0]))]

    comparison = compare(reference, candidate)

    assert comparison.vectors == [("p1", "sparse", 0.25)]
    assert comparison.largest_sparse_difference == 0.25
    assert comparison.largest_dense_difference == 0.0
    assert comparison.differences() == ["sparse   p1: differs by up to 0.25"]


def test_a_word_only_one_keyword_vector_holds_differs_by_its_whole_weight() -> None:
    reference = [point("p1", sparse=SparseVector([1, 2], [1.0, 0.5]))]
    candidate = [point("p1", sparse=SparseVector([1, 2, 3], [1.0, 0.5, 0.3]))]

    comparison = compare(reference, candidate)

    assert comparison.vectors == [("p1", "sparse", pytest.approx(0.3))]


def test_dense_vectors_of_different_sizes_always_differ() -> None:
    comparison = compare([point("p1")], [point("p1", dense=(1.0, 0.0, 0.0))], tolerance=1.0)

    assert comparison.vectors == [("p1", "dense", math.inf)]
    assert comparison.smallest_cosine == 0.0


def test_a_zero_vector_does_not_lower_the_smallest_cosine() -> None:
    comparison = compare([point("p1", dense=(0.0, 0.0))], [point("p1", dense=(0.0, 0.0))])

    assert comparison.equal
    assert comparison.smallest_cosine == 1.0


def test_two_keyword_vectors_without_words_are_the_same() -> None:
    no_words = SparseVector([], [])

    comparison = compare([point("p1", sparse=no_words)], [point("p1", sparse=no_words)], 0.0)

    assert comparison.equal
    assert comparison.largest_sparse_difference == 0.0


def test_the_smallest_cosine_and_the_largest_difference_come_from_the_worst_pair() -> None:
    reference = [point("a"), point("b"), point("c")]
    candidate = [point("a", dense=(0.8, 0.6)), point("b"), point("c", dense=(0.6, 0.8))]

    comparison = compare(reference, candidate, tolerance=1.0)

    assert comparison.equal
    assert comparison.smallest_cosine == pytest.approx(0.6)
    assert comparison.largest_dense_difference == pytest.approx(0.8)


def test_points_are_read_a_batch_at_a_time_from_both_collections() -> None:
    points = [point(f"p{n}") for n in range(5)]
    reference, candidate = Points("full", points), Points("updated", points)

    comparison = compare_collections(reference, candidate, tolerance=0.0, batch_size=2)

    assert comparison.equal
    assert comparison.compared == 5
    expected = [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert reference.batches == candidate.batches == expected


def test_every_kind_of_difference_is_listed_in_order() -> None:
    reference = [point("gone"), point("same"), point("moved")]
    candidate = [point("same", title="Holidays"), point("moved", dense=(0.0, 1.0)), point("new")]

    lines = compare(reference, candidate).differences()

    assert lines == [
        "missing  gone: in full only",
        "extra    new: in updated only",
        "payload  same: title differs",
        "dense    moved: differs by up to 1",
    ]


def test_two_collections_in_the_real_store_are_compared_point_by_point() -> None:
    client = QdrantClient(":memory:")
    full, updated = QdrantVectorStore(client, "full"), QdrantVectorStore(client, "updated")
    for store in (full, updated):
        store.ensure_collection(dimension=2)
    leave, laptop = make_chunk("Take 25 days."), make_chunk("Expense a laptop.")
    words = SparseVector([1], [1.0])
    full.upsert([leave, laptop], [[1.0, 0.0], [0.0, 1.0]], [words, words])
    retitled = dataclasses.replace(leave, title="Time off")
    updated.upsert([retitled], [[0.6, 0.8]], [words])

    comparison = compare_collections(full, updated, tolerance=1e-4)

    assert comparison.missing == [laptop.chunk_id]
    assert comparison.payloads == [(leave.chunk_id, "title")]
    ((point_id, vector, size),) = comparison.vectors
    assert (point_id, vector) == (leave.chunk_id, "dense")
    assert size == pytest.approx(0.8)
