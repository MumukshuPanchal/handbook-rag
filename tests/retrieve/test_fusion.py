import pytest

from handbook_rag.index.store import ScoredChunk
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.fusion import reciprocal_rank_fusion
from tests.support import make_chunk

A, B, C, D = (make_chunk(name) for name in ("alpha", "bravo", "charlie", "delta"))


def ranked(*chunks: Chunk) -> list[ScoredChunk]:
    return [ScoredChunk(chunk, 1.0 - 0.1 * position) for position, chunk in enumerate(chunks)]


def test_a_chunk_ranked_well_by_both_lists_beats_one_ranked_first_by_only_one() -> None:
    dense = ranked(A, B, C)
    sparse = ranked(D, B, C)

    fused = reciprocal_rank_fusion([dense, sparse], limit=4)

    assert [result.chunk for result in fused[:2]] == [B, C]
    assert {result.chunk.text for result in fused[2:]} == {"alpha", "delta"}


def test_scores_are_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([ranked(A, B), ranked(B, A)], limit=2, k=60)

    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 61 + 1 / 62)


def test_original_scores_on_different_scales_do_not_matter() -> None:
    dense = [ScoredChunk(A, 0.91), ScoredChunk(B, 0.90)]
    keyword = [ScoredChunk(B, 250.0), ScoredChunk(A, 3.0)]

    fused = reciprocal_rank_fusion([dense, keyword], limit=2)

    assert fused[0].score == pytest.approx(fused[1].score)


def test_a_chunk_found_by_only_one_list_is_still_returned() -> None:
    fused = reciprocal_rank_fusion([ranked(A), ranked(B)], limit=5)

    assert {result.chunk.text for result in fused} == {"alpha", "bravo"}


def test_limit_is_respected_and_ties_are_broken_the_same_way_every_time() -> None:
    first = reciprocal_rank_fusion([ranked(A, B, C), ranked(C, B, A)], limit=2)
    again = reciprocal_rank_fusion([ranked(A, B, C), ranked(C, B, A)], limit=2)

    assert len(first) == 2
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in again]


def test_a_smaller_k_rewards_top_ranks_more() -> None:
    top_of_one = [ranked(A, B, C, D), ranked(B, C, D)]

    with_small_k = reciprocal_rank_fusion(top_of_one, limit=2, k=1)
    with_large_k = reciprocal_rank_fusion(top_of_one, limit=2, k=1000)

    assert with_large_k[0].chunk == B
    assert with_small_k[0].chunk == B
    assert (
        with_small_k[0].score - with_small_k[1].score
        > with_large_k[0].score - with_large_k[1].score
    )


def test_no_rankings_gives_no_results() -> None:
    assert reciprocal_rank_fusion([], limit=5) == []
    assert reciprocal_rank_fusion([[], []], limit=5) == []
