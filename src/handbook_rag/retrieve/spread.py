"""Spread a ranked list over pages.

A search ranks every chunk by itself. When one page fits a question well, several of its
chunks can fill the first results, and a second page the answer also needs is left below
them. This rule limits how many chunks of one page may stand among the first results. A
chunk over the limit moves down, and none is dropped.

It is a variant, and off unless a caller asks for it. The default search does not use
it. It was built to be measured on questions that need two passages (issue 78), and
``evals/two-passage/search-variants.md`` says what the measurement found.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from handbook_rag.index.store import ScoredChunk


@dataclass(frozen=True, slots=True)
class PageSpread:
    """At most ``per_page`` chunks of one page among the first ``within`` results."""

    per_page: int
    within: int

    def __post_init__(self) -> None:
        if self.per_page < 1:
            raise ValueError("A page needs room for at least one of its chunks.")
        if self.within < 1:
            raise ValueError("The limit has to hold for at least the first result.")


def spread_over_pages(results: Sequence[ScoredChunk], spread: PageSpread) -> list[ScoredChunk]:
    """Reorder ``results`` so that its first ``spread.within`` hold few chunks of one page.

    The list is read from the top. A chunk takes the next of the first places, unless its
    page already has ``spread.per_page`` chunks there. Such a chunk is held back. Once
    the first places are taken, the chunks held back follow, and after them everything
    that stood below.

    - The result is the same chunks. None is dropped and none is added.
    - Two chunks of one page never change places.
    - The chunks in the first places keep the order they had, and so do all the others.
    - Only the order is read. No score is compared, so chunks with equal scores stay as
      they were given. Each chunk keeps its score, so the scores of the result are no
      longer in descending order.

    When the list has too few pages to fill the first places within the limit, the
    chunks held back fill the rest. The limit is then not kept, because nothing may be
    dropped: a list of one page comes back as it was.
    """
    first: list[ScoredChunk] = []
    held_back: list[ScoredChunk] = []
    of_page: Counter[str] = Counter()
    below = iter(results)
    for result in below:
        if of_page[result.chunk.doc_id] >= spread.per_page:
            held_back.append(result)
            continue
        first.append(result)
        of_page[result.chunk.doc_id] += 1
        if len(first) == spread.within:
            break
    # What was held back stood above everything the loop did not reach, so the two
    # together are still in the order they were given in.
    return [*first, *held_back, *below]
