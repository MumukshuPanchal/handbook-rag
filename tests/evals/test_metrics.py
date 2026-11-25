"""Metric tests use tiny hand-made rankings whose scores can be worked out on paper."""

import math

import pytest

from handbook_rag.evals.dataset import EvidenceSpan
from handbook_rag.evals.metrics import (
    all_at_k,
    covered_evidence,
    first_ranks,
    hit_at_k,
    mean_with_interval,
    ndcg_at_k,
    paired_difference,
    recall_at_k,
    reciprocal_rank_at_k,
    share_with_interval,
    sign_test,
)
from handbook_rag.ingest.models import Chunk

MISS: frozenset[int] = frozenset()
FIRST = frozenset({0})
SECOND = frozenset({1})


def chunk(doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{start}",
        doc_id=doc_id,
        title="T",
        url="https://example.com/",
        section="s",
        source_commit="a" * 40,
        heading_path=(),
        start_char=start,
        end_char=end,
        text="x" * (end - start),
        token_count=1,
    )


SPAN = EvidenceSpan("people/leave.md", start=100, end=200)


def test_chunk_containing_the_quote_covers_it() -> None:
    assert covered_evidence(chunk("people/leave.md", 50, 400), [SPAN]) == {0}


def test_chunk_from_another_page_never_covers_it() -> None:
    assert covered_evidence(chunk("finance/expenses.md", 50, 400), [SPAN]) == set()


@pytest.mark.parametrize(
    ("start", "end", "covers"),
    [
        (0, 150, True),  # exactly half of the quote
        (0, 149, False),  # just under half
        (151, 400, False),  # just under half, from the other side
        (150, 400, True),
        (0, 100, False),  # ends where the quote starts
        (120, 180, True),  # inside the quote, 60 of 100 characters
    ],
)
def test_partial_overlap_needs_at_least_half_of_the_quote(
    start: int, end: int, covers: bool
) -> None:
    assert bool(covered_evidence(chunk("people/leave.md", start, end), [SPAN])) is covers


def test_one_chunk_can_cover_several_pieces_of_evidence() -> None:
    spans = [SPAN, EvidenceSpan("people/leave.md", 250, 300), EvidenceSpan("other.md", 0, 10)]

    assert covered_evidence(chunk("people/leave.md", 90, 310), spans) == {0, 1}


def test_hit_looks_only_at_the_top_k() -> None:
    ranking = [MISS, MISS, FIRST]

    assert hit_at_k(ranking, 2) == 0.0
    assert hit_at_k(ranking, 3) == 1.0
    assert hit_at_k([], 5) == 0.0


def test_recall_is_the_share_of_evidence_found() -> None:
    ranking = [FIRST, MISS, SECOND]

    assert recall_at_k(ranking, evidence_count=2, k=1) == 0.5
    assert recall_at_k(ranking, evidence_count=2, k=3) == 1.0
    assert recall_at_k([FIRST, FIRST], evidence_count=2, k=2) == 0.5


def test_all_is_one_only_when_the_top_k_hold_every_piece_of_evidence() -> None:
    ranking = [FIRST, MISS, MISS, SECOND]

    # One of two is a hit, and it is not all.
    assert hit_at_k(ranking, 3) == 1.0
    assert all_at_k(ranking, evidence_count=2, k=3) == 0.0
    assert all_at_k(ranking, evidence_count=2, k=4) == 1.0
    # The same piece found twice is still one piece.
    assert all_at_k([FIRST, FIRST], evidence_count=2, k=2) == 0.0
    assert all_at_k([frozenset({0, 1})], evidence_count=2, k=1) == 1.0
    assert all_at_k([], evidence_count=2, k=5) == 0.0


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ranking", [[FIRST], [MISS, FIRST], [MISS, MISS, MISS, FIRST], [MISS]])
def test_all_is_hit_for_a_question_with_one_piece_of_evidence(
    ranking: list[frozenset[int]], k: int
) -> None:
    assert all_at_k(ranking, evidence_count=1, k=k) == hit_at_k(ranking, k)


def test_first_ranks_say_where_each_piece_of_evidence_first_came() -> None:
    assert first_ranks([FIRST, MISS, SECOND, FIRST], evidence_count=2) == (1, 3)
    assert first_ranks([MISS, SECOND], evidence_count=2) == (None, 2)
    assert first_ranks([frozenset({0, 1})], evidence_count=2) == (1, 1)
    assert first_ranks([], evidence_count=2) == (None, None)
    assert first_ranks([MISS], evidence_count=0) == ()


def test_all_and_first_ranks_agree_at_every_cutoff() -> None:
    rankings = [[FIRST, SECOND], [SECOND, MISS, MISS, MISS, MISS, FIRST], [FIRST, FIRST], []]

    for ranking in rankings:
        ranks = first_ranks(ranking, evidence_count=2)
        for k in range(1, 8):
            by_ranks = all(rank is not None and rank <= k for rank in ranks)
            assert all_at_k(ranking, evidence_count=2, k=k) == float(by_ranks)


def test_reciprocal_rank_is_one_over_the_first_relevant_position() -> None:
    assert reciprocal_rank_at_k([FIRST], 10) == 1.0
    assert reciprocal_rank_at_k([MISS, MISS, FIRST], 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k([MISS, MISS, FIRST], 2) == 0.0
    assert reciprocal_rank_at_k([MISS], 10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_falls_with_rank() -> None:
    assert ndcg_at_k([FIRST, MISS], evidence_count=1, k=10) == 1.0
    assert ndcg_at_k([MISS, FIRST], evidence_count=1, k=10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([MISS, MISS], evidence_count=1, k=10) == 0.0


def test_ndcg_with_two_pieces_of_evidence() -> None:
    ideal = 1 / math.log2(2) + 1 / math.log2(3)

    assert ndcg_at_k([FIRST, SECOND], evidence_count=2, k=10) == pytest.approx(1.0)
    assert ndcg_at_k([FIRST, MISS, SECOND], evidence_count=2, k=10) == pytest.approx(
        (1 / math.log2(2) + 1 / math.log2(4)) / ideal
    )


def test_ndcg_is_one_when_a_single_chunk_covers_every_piece() -> None:
    both = frozenset({0, 1})

    assert ndcg_at_k([both, MISS], evidence_count=2, k=10) == 1.0
    assert ndcg_at_k([MISS, both], evidence_count=2, k=10) < 1.0


def test_ndcg_does_not_reward_the_same_evidence_twice() -> None:
    once = ndcg_at_k([FIRST, MISS], evidence_count=2, k=10)
    twice = ndcg_at_k([FIRST, FIRST], evidence_count=2, k=10)

    assert twice == once


def test_metrics_that_need_evidence_reject_questions_without_any() -> None:
    with pytest.raises(ValueError, match="no evidence"):
        recall_at_k([MISS], evidence_count=0, k=5)
    with pytest.raises(ValueError, match="no evidence"):
        ndcg_at_k([MISS], evidence_count=0, k=5)
    with pytest.raises(ValueError, match="no evidence"):
        all_at_k([MISS], evidence_count=0, k=5)


def test_mean_with_interval_brackets_the_mean() -> None:
    scores = [1.0] * 70 + [0.0] * 30

    estimate = mean_with_interval(scores)

    assert estimate.mean == pytest.approx(0.70)
    assert estimate.count == 100
    assert 0.55 < estimate.low < 0.70 < estimate.high < 0.85


def test_interval_narrows_with_more_questions() -> None:
    small = mean_with_interval([1.0, 0.0] * 10)
    large = mean_with_interval([1.0, 0.0] * 200)

    assert (large.high - large.low) < (small.high - small.low) / 2


def test_interval_is_reproducible_and_collapses_when_all_scores_agree() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    assert mean_with_interval(scores) == mean_with_interval(scores)
    assert mean_with_interval([1.0] * 20) == mean_with_interval([1.0] * 20)
    perfect = mean_with_interval([1.0] * 20)
    assert (perfect.low, perfect.mean, perfect.high) == (1.0, 1.0, 1.0)


def test_mean_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean_with_interval([])


def test_paired_difference_counts_the_questions_that_got_better_and_worse() -> None:
    base = [1.0, 0.0, 0.0, 1.0, 0.5]
    candidate = [1.0, 1.0, 0.0, 0.0, 1.0]

    comparison = paired_difference(base, candidate)

    assert (comparison.better, comparison.worse, comparison.same) == (2, 1, 2)
    assert comparison.difference.mean == pytest.approx(0.1)
    assert comparison.difference.count == 5


def test_pairing_reveals_a_gain_that_two_separate_intervals_hide() -> None:
    # Half the questions are hard for the base system. The candidate fixes 12 of them and
    # breaks nothing.
    base = [1.0] * 50 + [0.0] * 50
    candidate = [1.0] * 62 + [0.0] * 38

    before, after = mean_with_interval(base), mean_with_interval(candidate)
    comparison = paired_difference(base, candidate)

    assert before.high > after.low, "the separate intervals overlap"
    assert comparison.difference.mean == pytest.approx(0.12)
    assert comparison.difference.low > 0
    assert comparison.is_clear


def test_a_difference_is_unclear_when_gains_and_losses_balance() -> None:
    base = [1.0, 0.0] * 30
    candidate = [0.0, 1.0] * 29 + [1.0, 1.0]

    comparison = paired_difference(base, candidate)

    assert comparison.better == 30
    assert comparison.worse == 29
    assert comparison.difference.low < 0 < comparison.difference.high
    assert not comparison.is_clear


def test_identical_systems_show_no_difference() -> None:
    scores = [1.0, 0.0, 1.0]

    comparison = paired_difference(scores, scores)

    assert comparison.same == 3
    assert not comparison.is_clear


def test_paired_comparison_needs_the_same_questions_on_both_sides() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_difference([1.0, 0.0], [1.0])


@pytest.mark.parametrize(
    ("successes", "total", "low", "high"),
    [
        # Checked against the published formula by hand, and against the usual tables.
        (41, 41, 0.9143, 1.0),
        (0, 41, 0.0, 0.0857),
        (41, 42, 0.8768, 0.9958),
        (1, 1, 0.2065, 1.0),
        (50, 100, 0.4038, 0.5962),
    ],
)
def test_a_share_has_an_interval_that_stays_open_at_the_ends(
    successes: int, total: int, low: float, high: float
) -> None:
    estimate = share_with_interval(successes, total)

    assert estimate.mean == successes / total
    assert estimate.count == total
    assert estimate.low == pytest.approx(low, abs=0.0005)
    assert estimate.high == pytest.approx(high, abs=0.0005)
    assert 0.0 <= estimate.low <= estimate.mean <= estimate.high <= 1.0


@pytest.mark.parametrize("total", [1, 7, 20, 21, 22, 41, 1000])
def test_the_interval_of_a_share_never_leaves_the_range_from_nothing_to_all(total: int) -> None:
    # Rounding in the arithmetic would otherwise put the ends a hair outside: above 1
    # for totals such as 20, 21 and 22, and above 0 for a total of 1000.
    assert share_with_interval(0, total).low == 0.0
    assert share_with_interval(total, total).high == 1.0


def test_the_ends_of_an_interval_stay_in_order_for_every_small_count() -> None:
    for total in range(1, 120):
        for successes in range(total + 1):
            estimate = share_with_interval(successes, total)
            assert 0.0 <= estimate.low <= estimate.mean <= estimate.high <= 1.0, (successes, total)


@pytest.mark.parametrize(("successes", "total"), [(0, 0), (1, 0), (-1, 5), (6, 5)])
def test_a_share_needs_a_count_it_fits_in(successes: int, total: int) -> None:
    with pytest.raises(ValueError, match="share"):
        share_with_interval(successes, total)


@pytest.mark.parametrize(
    ("better", "worse", "p"),
    [
        (7, 1, 0.0703125),
        (1, 7, 0.0703125),
        (5, 1, 0.21875),
        (2, 1, 1.0),
        (0, 0, 1.0),
        (6, 0, 0.03125),
    ],
)
def test_the_sign_test_is_exact_and_two_sided(better: int, worse: int, p: float) -> None:
    assert sign_test(better, worse) == p
    # The comparison of two runs gives it too, and the questions that did not change take
    # no part in it.
    base = [0.0] * better + [1.0] * worse + [1.0, 0.0]
    candidate = [1.0] * better + [0.0] * worse + [1.0, 0.0]
    assert paired_difference(base, candidate).sign_test == p
