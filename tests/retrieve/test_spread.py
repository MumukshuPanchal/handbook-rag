"""The rule that spreads a ranked list over pages, and hybrid search with the rule on.

A chunk is named by its page and its place on it: ``a2`` is the second chunk of page
``a``. A ranked list is written from the top, with scores that fall.
"""

import random
from collections import Counter

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from handbook_rag.index.store import ScoredChunk
from handbook_rag.retrieve.fusion import reciprocal_rank_fusion
from handbook_rag.retrieve.hybrid import HybridRetriever
from handbook_rag.retrieve.spread import PageSpread, spread_over_pages
from tests.retrieve.test_hybrid import FixedRetriever
from tests.support import make_chunk


def ranked(*names: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(make_chunk(name, doc_id=f"{name[0]}.md"), 1.0 - 0.01 * position)
        for position, name in enumerate(names)
    ]


def names(results: list[ScoredChunk]) -> list[str]:
    return [result.chunk.text for result in results]


def spread(results: list[ScoredChunk], per_page: int, within: int) -> list[str]:
    return names(spread_over_pages(results, PageSpread(per_page=per_page, within=within)))


def by_the_definition(results: list[ScoredChunk], per_page: int, within: int) -> list[ScoredChunk]:
    """The rule, one place at a time and slowly, to compare the real one with.

    Each of the first places goes to the highest chunk left whose page still has room
    there. When no chunk left has room, the first places stay as they are, and
    everything left follows in its order.
    """
    left = list(results)
    first: list[ScoredChunk] = []
    while left and len(first) < within:
        taken = Counter(result.chunk.doc_id for result in first)
        with_room = [result for result in left if taken[result.chunk.doc_id] < per_page]
        if not with_room:
            break
        first.append(with_room[0])
        left.remove(with_room[0])
    return first + left


def test_a_chunk_over_the_limit_moves_down_and_the_next_page_moves_up() -> None:
    results = ranked("a1", "a2", "a3", "b1", "c1", "d1", "e1", "f1")

    assert spread(results, 1, 5) == ["a1", "b1", "c1", "d1", "e1", "a2", "a3", "f1"]
    assert spread(results, 2, 5) == ["a1", "a2", "b1", "c1", "d1", "a3", "e1", "f1"]
    # Three chunks of a page were allowed, and the list had no more than three.
    assert spread(results, 3, 5) == names(results)


def test_nothing_is_dropped_and_nothing_is_added() -> None:
    results = ranked("a1", "a2", "b1", "a3", "a4", "c1", "b2", "a5", "d1", "e1")

    for per_page in (1, 2, 3):
        for within in (1, 3, 5, 10, 50):
            after = spread_over_pages(results, PageSpread(per_page, within))
            assert len(after) == len(results)
            assert sorted(names(after)) == sorted(names(results)), (per_page, within)


def test_two_chunks_of_one_page_never_change_places() -> None:
    results = ranked("a1", "b1", "a2", "b2", "a3", "c1", "b3", "a4", "d1", "c2")

    for per_page in (1, 2):
        for within in (2, 4, 5, 8):
            after = spread(results, per_page, within)
            for page in "abcd":
                of_page = [name for name in after if name[0] == page]
                assert of_page == sorted(of_page), (per_page, within, page)


def test_the_limit_is_kept_among_the_first_results_when_the_list_has_pages_enough() -> None:
    results = ranked("a1", "a2", "a3", "a4", "a5", "b1", "b2", "c1", "d1", "e1", "f1")

    for per_page, within in ((1, 5), (2, 5), (1, 3), (2, 8)):
        after = spread(results, per_page, within)
        first = Counter(name[0] for name in after[:within])
        assert max(first.values()) <= per_page, (per_page, within)
        assert len(after[:within]) == within


def test_the_chunks_held_back_follow_the_first_places_ahead_of_all_that_stood_below() -> None:
    results = ranked("a1", "a2", "b1", "a3", "c1", "b2", "d1", "e1", "f1")

    # The first four places go to four pages. Then come the three chunks that were held
    # back, in their order, and then the two the rule never reached.
    assert spread(results, 1, 4) == ["a1", "b1", "c1", "d1", "a2", "a3", "b2", "e1", "f1"]


def test_with_too_few_pages_the_chunks_held_back_fill_the_first_places() -> None:
    two_pages = ranked("a1", "a2", "a3", "b1")
    one_page = ranked("a1", "a2", "a3")

    # Five places and two pages: one chunk of each, and then the rest in its order. The
    # first five now hold three chunks of page a, because nothing may be dropped.
    assert spread(two_pages, 1, 5) == ["a1", "b1", "a2", "a3"]
    assert spread(one_page, 1, 5) == ["a1", "a2", "a3"]
    assert spread(one_page, 2, 3) == ["a1", "a2", "a3"]


def test_a_list_within_the_limit_comes_back_as_it_was() -> None:
    assert spread(ranked("a1", "b1", "c1", "a2", "b2"), 1, 3) == ["a1", "b1", "c1", "a2", "b2"]
    assert spread(ranked("a1", "a2", "b1", "b2", "c1"), 2, 5) == ["a1", "a2", "b1", "b2", "c1"]
    # A limit that is as wide as the window limits nothing.
    assert spread(ranked("a1", "a2", "a3", "b1"), 3, 3) == ["a1", "a2", "a3", "b1"]
    assert spread(ranked("a1", "a2", "a3", "b1"), 9, 2) == ["a1", "a2", "a3", "b1"]
    assert spread_over_pages([], PageSpread(1, 5)) == []


def test_equal_scores_stay_in_the_order_given_because_no_score_is_read() -> None:
    chunks = [result.chunk for result in ranked("a1", "a2", "b1", "c1", "b2", "d1")]
    tied = [ScoredChunk(chunk, 0.5) for chunk in chunks]
    rising = [ScoredChunk(chunk, 0.1 * position) for position, chunk in enumerate(chunks)]

    for results in (tied, rising):
        after = spread_over_pages(results, PageSpread(1, 3))
        assert names(after) == ["a1", "b1", "c1", "a2", "b2", "d1"]


def test_every_chunk_keeps_its_score_so_the_scores_no_longer_fall() -> None:
    results = ranked("a1", "a2", "b1", "c1")
    score_of = {result.chunk.text: result.score for result in results}

    after = spread_over_pages(results, PageSpread(1, 3))

    assert names(after) == ["a1", "b1", "c1", "a2"]
    assert {result.chunk.text: result.score for result in after} == score_of
    # The chunk that moved down kept the score that had put it second.
    assert [result.score for result in after] == pytest.approx([1.0, 0.98, 0.97, 0.99])


def test_the_first_results_are_the_same_however_far_the_limit_reaches() -> None:
    results = ranked("a1", "a2", "b1", "a3", "b2", "c1", "c2", "d1", "a4", "e1", "f1", "g1")

    for per_page in (1, 2):
        within_five = spread(results, per_page, 5)
        within_ten = spread(results, per_page, 10)
        assert within_five[:5] == within_ten[:5], per_page
        # Further down they differ: a limit that reaches further holds more back.
        assert within_five != within_ten, per_page


def test_applying_the_rule_twice_changes_nothing_more() -> None:
    results = ranked("a1", "a2", "b1", "a3", "b2", "c1", "c2", "d1", "a4", "e1")

    for per_page, within in ((1, 5), (2, 5), (1, 10), (2, 3), (1, 50)):
        rule = PageSpread(per_page, within)
        once = spread_over_pages(results, rule)
        assert spread_over_pages(once, rule) == once, (per_page, within)


def test_the_rule_does_what_its_definition_says_on_many_lists() -> None:
    generator = random.Random(78)

    for _ in range(500):
        pages = generator.randint(1, 5)
        results = [
            ScoredChunk(
                make_chunk(f"chunk {number}", doc_id=f"page-{generator.randrange(pages)}.md"),
                generator.choice([0.5, 0.4, 0.3]),
            )
            for number in range(generator.randint(0, 14))
        ]
        per_page, within = generator.randint(1, 3), generator.randint(1, 12)

        after = spread_over_pages(results, PageSpread(per_page, within))

        assert after == by_the_definition(results, per_page, within)
        assert sorted(names(after)) == sorted(names(results))
        first = Counter(result.chunk.doc_id for result in after[:within])
        if max(first.values(), default=0) > per_page:
            # Only when no chunk outside the first places has a page with room there.
            assert all(first[result.chunk.doc_id] >= per_page for result in after[within:])


@pytest.mark.parametrize(("per_page", "within"), [(0, 5), (-1, 5), (1, 0), (2, -3)])
def test_a_rule_needs_room_for_one_chunk_and_at_least_one_result(
    per_page: int, within: int
) -> None:
    with pytest.raises(ValueError, match="at least"):
        PageSpread(per_page=per_page, within=within)


# Hybrid search with the rule


def hybrid(*, spread: PageSpread | None = None) -> tuple[HybridRetriever, FixedRetriever]:
    """Hybrid search over two searches that agree on one ranking of twelve chunks."""
    ranking = ranked("a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2", "c1", "d1", "e1", "f1")
    dense, sparse = FixedRetriever(ranking), FixedRetriever(ranking)
    return HybridRetriever(dense, sparse, spread=spread), dense


def test_hybrid_without_the_rule_returns_the_merged_list_cut_to_the_limit() -> None:
    retriever, dense = hybrid()

    results = retriever.retrieve("q", limit=5)

    assert names(results) == ["a1", "a2", "a3", "a4", "a5"]
    assert results == reciprocal_rank_fusion([dense.retrieve("q", limit=30)] * 2, limit=5, k=60)


def test_hybrid_with_the_rule_brings_up_a_page_from_below_what_was_asked_for() -> None:
    one_of_a_page, _ = hybrid(spread=PageSpread(per_page=1, within=5))
    two_of_a_page, _ = hybrid(spread=PageSpread(per_page=2, within=5))

    # Page b stood seventh in the merged list, c ninth, d tenth and e eleventh.
    assert names(one_of_a_page.retrieve("q", limit=5)) == ["a1", "b1", "c1", "d1", "e1"]
    assert names(two_of_a_page.retrieve("q", limit=5)) == ["a1", "a2", "b1", "b2", "c1"]


def test_hybrid_with_the_rule_asks_each_search_for_the_same_candidates() -> None:
    retriever, dense = hybrid(spread=PageSpread(per_page=1, within=5))
    plain, dense_of_plain = hybrid()

    retriever.retrieve("q", limit=5, section="people")
    plain.retrieve("q", limit=5, section="people")

    assert dense.limits == dense_of_plain.limits == [30]
    assert dense.sections == ["people"]


def test_the_first_five_results_are_the_same_whether_five_or_ten_are_asked_for() -> None:
    retriever, _ = hybrid(spread=PageSpread(per_page=2, within=5))

    ten = names(retriever.retrieve("q", limit=10))

    assert ten[:5] == names(retriever.retrieve("q", limit=5))
    # Below the first five, the chunks held back come first, and nothing is lost: the
    # ten results are ten of the twelve chunks.
    assert ten == ["a1", "a2", "b1", "b2", "c1", "a3", "a4", "a5", "a6", "d1"]


def test_fewer_results_than_the_window_are_the_first_of_the_spread_list() -> None:
    retriever, _ = hybrid(spread=PageSpread(per_page=1, within=5))

    assert names(retriever.retrieve("q", limit=3)) == ["a1", "b1", "c1"]
    assert names(retriever.retrieve("q", limit=12))[5:] == [
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "b2",
        "f1",
    ]


def test_hybrid_with_the_rule_reads_every_chunk_of_both_searches() -> None:
    # Page b is in one search only, third, and sixth of the six chunks the two return.
    # A merged list as long as one search's list would end before it.
    retriever = HybridRetriever(
        FixedRetriever(ranked("a1", "a2", "b1")),
        FixedRetriever(ranked("a3", "a4", "a5")),
        spread=PageSpread(per_page=1, within=2),
    )

    results = retriever.retrieve("q", limit=2)

    assert [result.chunk.doc_id for result in results] == ["a.md", "b.md"]
    assert names(results)[1] == "b1"


def test_hybrid_with_the_rule_returns_nothing_when_neither_search_finds_anything() -> None:
    nothing = HybridRetriever(
        FixedRetriever([]), FixedRetriever([]), spread=PageSpread(per_page=1, within=5)
    )

    assert nothing.retrieve("  ", limit=5) == []


def test_hybrid_with_the_rule_keeps_the_merged_scores() -> None:
    retriever, _ = hybrid(spread=PageSpread(per_page=1, within=5))
    plain, _ = hybrid()
    merged = {result.chunk.text: result.score for result in plain.retrieve("q", limit=12)}

    results = retriever.retrieve("q", limit=5)

    assert [result.score for result in results] == [merged[name] for name in names(results)]


def test_the_trace_shows_the_rule_only_when_it_is_on(spans: InMemorySpanExporter) -> None:
    plain, _ = hybrid()
    retriever, _ = hybrid(spread=PageSpread(per_page=2, within=5))

    plain.retrieve("q", limit=5)
    without = [span.name for span in spans.get_finished_spans()]
    spans.clear()
    retriever.retrieve("q", limit=5)
    with_the_rule = {span.name: span for span in spans.get_finished_spans()}

    assert without == ["fuse", "retrieve.hybrid"]
    assert list(with_the_rule) == ["fuse", "spread", "retrieve.hybrid"]
    attributes = with_the_rule["spread"].attributes or {}
    assert attributes["handbook_rag.per_page"] == 2
    assert attributes["handbook_rag.within"] == 5
    # The search's own span records what the caller got: the list after the rule.
    recorded = (with_the_rule["retrieve.hybrid"].attributes or {})["handbook_rag.results.doc_ids"]
    assert recorded == ("a.md", "a.md", "b.md", "b.md", "c.md")
