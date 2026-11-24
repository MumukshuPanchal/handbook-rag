"""The answer cache: its two tiers, what it keeps, and what it never serves by meaning.

The store is a Qdrant collection in this process's memory, and the embedding model and
the verifier are stand-ins that give each question the vector and the scores a test
names. Nothing is downloaded or loaded.
"""

import dataclasses
import datetime
import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest

from handbook_rag.cache import answers as cache_answers
from handbook_rag.cache import rules
from handbook_rag.cache.answers import (
    DECIDED_BY,
    AnswerCache,
    Lookup,
    comparable,
    deciding_score,
    entry_id,
    scope_of,
    servable_by_meaning,
    storable,
)
from handbook_rag.cache.rules import (
    MISSING_PHRASES,
    kept_from_the_second_tier,
    says_something_is_missing,
)
from handbook_rag.cache.store import Entry, QdrantAnswerStore
from handbook_rag.cache.verify import LazyVerifier, Verification
from handbook_rag.evals.cache_pairs import Scores
from handbook_rag.generate.models import Answer, CacheHit, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from tests.cache.support import TableVerifier, memory_answers
from tests.evals.test_cache_pairs import TableEmbedder, towards
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
LAPTOP = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")
STORED = "How many weeks of parental leave do I get?"
REWORDED = "How long is my parental leave?"
NEAR_MISS = "How many weeks of parental leave does GitHub give?"
WHEN = datetime.datetime(2031, 2, 3, 4, 5, 6, tzinfo=datetime.UTC)
THRESHOLD = 0.8


def written(text: str = "Sixteen weeks.", *, cite: bool = True, **changes: Any) -> Answer:
    """A whole answer as the answer model writes it: one statement, citing one passage."""
    answer = Answer(
        question="the question it was asked",
        segments=(Segment(text, (Citation(LEAVE, "sixteen weeks"),) if cite else ()),),
        answered=True,
        model="stand-in-model-002",
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1200, output_tokens=40),
        seconds=2.5,
        seconds_to_first_text=1.5,
    )
    return dataclasses.replace(answer, **changes)


DECLINE = written(NOT_FOUND_MESSAGE, cite=False, answered=False)


class Vectors(TableEmbedder):
    """A stand-in embedding model: a question has the vector a test gives it, and any
    other question the same vector, far from all of those."""

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vectors.get(text, [0.0, 0.0, 1.0])


def cache_over(
    store: QdrantAnswerStore,
    vectors: dict[str, list[float]],
    verifier: TableVerifier | LazyVerifier | None = None,
    **options: Any,
) -> AnswerCache:
    return AnswerCache(
        store,
        embedder=Vectors(vectors),
        verifier=verifier,
        threshold=options.pop("threshold", THRESHOLD),
        now=lambda: WHEN,
        **options,
    )


def keep(cache: AnswerCache, question: str, answer: Answer, section: str | None = None) -> None:
    """Store an answer as the pipeline does: after a lookup that missed."""
    lookup = cache.lookup(question, section=section)
    assert lookup.answer is None
    cache.store(lookup, answer)


# --- the exact tier ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "asked",
    [
        STORED,
        "how many weeks of parental leave do i get?",
        "HOW MANY WEEKS OF PARENTAL LEAVE DO I GET",
        "  How many   weeks of\tparental leave\ndo I get ?  ",
        "How many weeks of parental leave do I get",
        "How many weeks of parental leave do I get?!",
        "How many weeks of parental leave do I get...",
        "How many weeks of parental leave do I get ? ! .",
    ],
)
def test_case_spacing_and_the_marks_at_the_end_of_a_question_do_not_count(asked: str) -> None:
    assert comparable(asked) == "how many weeks of parental leave do i get"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("What is C#?", "What is C?"),
        ("What is C++?", "What is C?"),
        ("Is the limit 3.5?", "Is the limit 35?"),
        ("What does U.S. leave cover?", "What does US leave cover?"),
        ("Who approves a laptop?", "Who approves, a laptop?"),
        ("What is the parental-leave policy?", "What is the parental leave policy?"),
        ("Is it “done”?", 'Is it "done"?'),
        ("Who is the DRI (for leave)?", "Who is the DRI (for leave?"),
    ],
)
def test_nothing_else_about_a_question_is_changed(first: str, second: str) -> None:
    # Punctuation inside a question, and any other sign at its end, can carry meaning.
    assert comparable(first) != comparable(second)


def test_a_letter_typed_with_a_separate_accent_is_the_same_letter() -> None:
    composed, with_accent = "Where is the café?", "Where is the café?"

    assert composed != with_accent
    assert comparable(composed) == comparable(with_accent) == "where is the café"
    # Folding the case of this letter gives two characters, which are composed again.
    assert comparable("ǰ") == comparable("ǰ")


@pytest.mark.parametrize("question", ["", "   ", "?", " ?! . ", "..."])
def test_a_question_with_nothing_left_has_no_comparable_form(question: str) -> None:
    assert comparable(question) == ""


def test_the_same_question_is_served_its_stored_answer_without_embedding_or_verifying() -> None:
    verifier = TableVerifier()
    cache = cache_over(memory_answers(), {STORED: towards(1.0)}, verifier)
    keep(cache, STORED, written())
    embedder = cache._embedder
    assert isinstance(embedder, Vectors)
    embedded = list(embedder.queries)

    found = cache.lookup("  how many weeks of parental leave do I get  ")

    served = found.answer
    assert served is not None
    assert served.cached == CacheHit(
        tier="exact", question=STORED, stored_at="2031-02-03T04:05:06+00:00"
    )
    assert (served.cached.similarity, served.cached.score) == (None, None)
    assert found.tier == "exact"
    # The question asked now, and the stored reply word for word with its citation.
    assert served.question == "  how many weeks of parental leave do I get  "
    assert served.segments == (Segment("Sixteen weeks.", (Citation(LEAVE, "sixteen weeks"),)),)
    assert served.sources == (LEAVE,)
    assert (served.answered, served.model, served.stop_reason) == (
        True,
        "stand-in-model-002",
        "end_turn",
    )
    # No model was asked for it: no request, no tokens, no time, no search.
    assert (served.requests, served.fallback) == (0, False)
    assert served.usage == TokenUsage()
    assert (served.seconds, served.seconds_to_first_text, served.retrieved) == (0.0, None, ())
    # The exact tier compares no embeddings and asks no verifier.
    assert embedder.queries == embedded
    assert verifier.calls == []


def test_an_exact_match_is_served_whatever_it_holds_a_decline_too() -> None:
    cache = cache_over(memory_answers(), {})
    keep(cache, STORED, DECLINE)

    served = cache.lookup(STORED.upper()).answer

    assert served is not None
    assert (served.text, served.answered) == (NOT_FOUND_MESSAGE, False)
    assert served.cached is not None
    assert served.cached.tier == "exact"


def test_the_same_words_asked_of_another_part_of_the_handbook_are_another_question() -> None:
    cache = cache_over(memory_answers(), {})
    keep(cache, STORED, written("Of all of it."))
    keep(cache, STORED, written("Of people only."), section="people")

    def served(section: str | None) -> str | None:
        answer = cache.lookup(STORED, section=section).answer
        return None if answer is None else answer.text

    assert served(None) == "Of all of it."
    assert served("people") == "Of people only."
    assert served("finance") is None
    # A section that is the empty text is a section, and not all of the handbook.
    assert served("") is None
    assert len({scope_of(None), scope_of(""), scope_of("all"), scope_of("people")}) == 4
    assert entry_id("a question", scope_of(None)) != entry_id("a question", scope_of("people"))


def test_a_newer_answer_takes_the_place_of_the_one_stored_for_the_same_question() -> None:
    store = memory_answers()
    cache = cache_over(store, {})
    keep(cache, STORED, written("The first answer."))

    # A caller who asks for a fresh answer: nothing is read, and the new answer is kept.
    fresh = cache.lookup(f"{STORED} ", read=False)
    assert (fresh.answer, fresh.read) == (None, False)
    cache.store(fresh, written("The second answer."))

    served = cache.lookup(STORED).answer
    assert served is not None
    assert served.text == "The second answer."
    assert store.held().answers == 1


def test_an_entry_under_the_right_id_that_is_of_another_question_is_not_served() -> None:
    # The id is a hash of the question. What is stored under it has to say so itself.
    store = memory_answers()
    cache = cache_over(store, {})
    keep(cache, STORED, written())
    (stolen,) = [store.get(entry_id(comparable(STORED), scope_of(None)))]
    assert stolen is not None
    other = entry_id(comparable(REWORDED), scope_of(None))
    store.put(dataclasses.replace(stolen, entry_id=other), towards(1.0))

    assert cache.lookup(REWORDED).answer is None


# --- the tier by meaning -------------------------------------------------------------------


def test_a_question_that_means_the_same_is_served_the_answer_with_both_scores() -> None:
    verifier = TableVerifier({(STORED, REWORDED): (0.95, 0.9)})
    cache = cache_over(
        memory_answers(), {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]}, verifier
    )
    keep(cache, STORED, written())
    verifier.calls.clear()

    found = cache.lookup(REWORDED)

    served = found.answer
    assert served is not None
    assert served.cached is not None
    assert served.cached.tier == found.tier == "semantic"
    assert served.cached.question == STORED, "the wording the answer was written for"
    assert served.cached.similarity == pytest.approx(1.0)
    assert served.cached.score == 0.9, "the weaker of the two directions"
    assert served.cached.stored_at == "2031-02-03T04:05:06+00:00"
    assert (served.question, served.text, served.requests) == (REWORDED, "Sixteen weeks.", 0)
    assert served.sources == (LEAVE,)
    # One call, the stored question first and the new one second, as they were measured.
    assert verifier.calls == [[(STORED, REWORDED)]]
    assert [(c.question, c.entails) for c in found.checked] == [(STORED, Verification(0.95, 0.9))]


@pytest.mark.parametrize(
    ("scores", "served"),
    [
        ((0.8, 0.8), True),
        ((0.95, 0.8), True),
        ((0.7999, 1.0), False),
        ((1.0, 0.7999), False),
        ((0.0, 1.0), False),
        ((float("nan"), 1.0), False),
        ((1.0, float("nan")), False),
    ],
)
def test_the_weaker_direction_has_to_reach_the_threshold(
    scores: tuple[float, float], served: bool
) -> None:
    verifier = TableVerifier({(STORED, REWORDED): scores})
    cache = cache_over(
        memory_answers(), {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]}, verifier
    )
    keep(cache, STORED, written())

    assert (cache.lookup(REWORDED).answer is not None) is served


def test_the_score_that_decides_is_the_one_the_measurement_chose() -> None:
    entails = Verification(forward=0.9, backward=0.3)

    assert DECIDED_BY == "entails_min"
    assert deciding_score(entails) == Scores(0.5, entails).of(DECIDED_BY) == 0.3


def test_a_near_miss_the_verifier_turns_down_is_a_miss_and_its_scores_are_kept() -> None:
    verifier = TableVerifier({(STORED, NEAR_MISS): (0.99, 0.05)})
    cache = cache_over(
        memory_answers(), {STORED: towards(0.98), NEAR_MISS: [1.0, 0.0, 0.0]}, verifier
    )
    keep(cache, STORED, written())

    found = cache.lookup(NEAR_MISS)

    assert (found.answer, found.tier, found.failure) == (None, None, None)
    (checked,) = found.checked
    assert checked.similarity == pytest.approx(0.98)
    assert checked.entails == Verification(0.99, 0.05)
    # The vector is kept, so that storing the new answer does not embed the question again.
    assert found.vector is not None
    assert list(found.vector) == [1.0, 0.0, 0.0]


def stored_near(*entries: tuple[str, float, Answer], **options: Any) -> tuple[AnswerCache, Any]:
    """A cache holding these questions, each at this cosine from the question "new"."""
    verifier = TableVerifier({(question, "new"): (1.0, 1.0) for question, _, _ in entries})
    vectors = {question: towards(cosine) for question, cosine, _ in entries}
    cache = cache_over(memory_answers(), {**vectors, "new": [1.0, 0.0, 0.0]}, verifier, **options)
    for question, _, answer in entries:
        keep(cache, question, answer)
    verifier.calls.clear()
    return cache, verifier


def test_the_candidates_are_the_nearest_and_the_first_that_passes_is_served() -> None:
    cache, verifier = stored_near(
        ("far", 0.6, written("far")),
        ("nearest", 0.95, written("nearest")),
        ("near", 0.9, written("near")),
        ("nearer", 0.92, written("nearer")),
        candidates=3,
    )
    # The nearest is turned down. The next two would both pass.
    verifier.scores[("nearest", "new")] = (0.2, 1.0)

    found = cache.lookup("new")

    assert verifier.calls == [[("nearest", "new"), ("nearer", "new"), ("near", "new")]]
    assert found.answer is not None
    assert found.answer.text == "nearer", "in the order of nearness, not the highest score"
    assert found.answer.cached is not None
    assert found.answer.cached.similarity == pytest.approx(0.92)
    assert [c.question for c in found.checked] == ["nearest", "nearer", "near"]


def test_no_more_stored_questions_are_verified_than_the_setting_says() -> None:
    cache, verifier = stored_near(
        ("nearest", 0.95, written("nearest")),
        ("nearer", 0.92, written("nearer")),
        candidates=1,
    )
    verifier.scores[("nearest", "new")] = (0.0, 0.0)

    assert cache.lookup("new").answer is None, "the second nearest would pass, and is not asked"
    assert verifier.calls == [[("nearest", "new")]]
    with pytest.raises(ValueError, match="at least one candidate"):
        cache_over(memory_answers(), {}, candidates=0)


def test_a_stored_question_below_the_floor_is_no_candidate_and_the_verifier_is_not_asked() -> None:
    loads: list[str] = []

    def load() -> TableVerifier:
        loads.append("loaded")
        return TableVerifier({("far", "new"): (1.0, 1.0)})

    lazy = LazyVerifier(load, name="lazy")
    cache = cache_over(
        memory_answers(), {"far": towards(0.49), "new": [1.0, 0.0, 0.0]}, lazy, floor=0.5
    )
    keep(cache, "far", written())

    found = cache.lookup("new")

    assert (found.answer, found.checked) == (None, ())
    assert loads == [], "with no candidate the verifier is never loaded"
    assert lazy.loaded is False
    # Just above the floor it is a candidate.
    cache = cache_over(
        memory_answers(), {"far": towards(0.51), "new": [1.0, 0.0, 0.0]}, lazy, floor=0.5
    )
    keep(cache, "far", written())
    assert cache.lookup("new").answer is not None
    assert loads == ["loaded"]


PARTIAL = written(f"Sixteen weeks in the US. For other countries: {NOT_FOUND_MESSAGE}")
UNCITED = written("Sixteen weeks, I believe.", cite=False)


@pytest.mark.parametrize(
    ("name", "answer"),
    [("a decline", DECLINE), ("an answer that declines in part", PARTIAL), ("uncited", UNCITED)],
)
def test_some_stored_replies_are_never_served_to_a_question_worded_differently(
    name: str, answer: Answer
) -> None:
    cache, verifier = stored_near(("stored", 0.99, answer))

    found = cache.lookup("new")

    assert found.answer is None, name
    assert found.checked == (), "it is no candidate, so the verifier is not even asked"
    assert verifier.calls == []
    assert servable_by_meaning(answer) is False
    # The very same question still gets it.
    again = cache.lookup("Stored?").answer
    assert again is not None
    assert again.text == answer.text


def test_a_reply_that_is_no_candidate_takes_no_candidates_place() -> None:
    # Three declines nearer than the one answer, and room for one candidate only.
    cache, verifier = stored_near(
        ("decline one", 0.99, DECLINE),
        ("decline two", 0.98, DECLINE),
        ("partial", 0.97, PARTIAL),
        ("answered", 0.9, written("The answer.")),
        candidates=1,
    )

    found = cache.lookup("new")

    assert verifier.calls == [[("answered", "new")]]
    assert found.answer is not None
    assert found.answer.text == "The answer."


def test_a_question_asked_of_one_section_is_matched_among_answers_of_that_section_only() -> None:
    verifier = TableVerifier({(STORED, REWORDED): (1.0, 1.0)})
    vectors = {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]}
    cache = cache_over(memory_answers(), vectors, verifier)
    keep(cache, STORED, written("From all of the handbook."))

    assert cache.lookup(REWORDED, section="people").answer is None
    assert cache.lookup(REWORDED, section="").answer is None
    assert verifier.calls == []

    keep(cache, STORED, written("From the people section."), section="people")
    of_people = cache.lookup(REWORDED, section="people").answer
    of_all = cache.lookup(REWORDED).answer
    assert of_people is not None
    assert of_all is not None
    assert (of_people.text, of_all.text) == (
        "From the people section.",
        "From all of the handbook.",
    )


class Unfiltered:
    """A store whose search ignores every filter and limit, as a faulty one might."""

    def __init__(self, entries: Sequence[tuple[Entry, float]]) -> None:
        self._entries = list(entries)

    def ensure(self, dimension: int) -> None:
        return None

    def get(self, entry_id: str) -> Entry | None:
        return None

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        return list(self._entries)

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        raise AssertionError("nothing is stored in this test")


def test_what_a_store_returns_is_checked_again_before_anything_is_verified() -> None:
    def entry(question: str, **changes: Any) -> Entry:
        made = Entry(
            entry_id=entry_id(comparable(question), scope_of(None)),
            question=question,
            comparable=comparable(question),
            scope=scope_of(None),
            segments=written().segments,
            answered=True,
            model="stand-in-model-002",
            stop_reason="end_turn",
            stored_at="2031-02-03T04:05:06+00:00",
            by_meaning=True,
        )
        return dataclasses.replace(made, **changes)

    verifier = TableVerifier()
    store = Unfiltered(
        [
            (entry("a decline", by_meaning=False), 0.99),
            (entry("of another section", scope=scope_of("people")), 0.98),
            (entry("too far"), 0.1),
            (entry("first"), 0.9),
            (entry("second"), 0.8),
            (entry("third"), 0.7),
        ]
    )
    cache = AnswerCache(
        store,
        embedder=Vectors({}),
        verifier=verifier,
        threshold=THRESHOLD,
        candidates=2,
        floor=0.5,
    )

    cache.lookup("new")

    assert verifier.calls == [[("first", "new"), ("second", "new")]]


def test_without_a_verifier_the_cache_has_the_exact_tier_only() -> None:
    cache = cache_over(memory_answers(), {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]})
    keep(cache, STORED, written())
    embedder = cache._embedder
    assert isinstance(embedder, Vectors)
    embedder.queries.clear()

    assert cache.mode == "exact"
    assert cache.lookup(REWORDED).answer is None
    assert cache.lookup(STORED).answer is not None
    assert embedder.queries == [], "looking a question up by its key embeds nothing"
    assert cache_over(memory_answers(), {}, TableVerifier()).mode == "semantic"


def test_an_entry_stored_by_the_exact_tier_can_be_matched_by_meaning_later() -> None:
    # Every entry is stored with its question's embedding, in both modes.
    store = memory_answers()
    vectors = {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]}
    keep(cache_over(store, vectors), STORED, written())

    both_tiers = cache_over(store, vectors, TableVerifier({(STORED, REWORDED): (1.0, 1.0)}))

    assert both_tiers.lookup(REWORDED).answer is not None


# --- what is kept --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("written by the fallback model", written(fallback=True, requests=2)),
        ("cut off at the length limit", written(stop_reason="max_tokens")),
        ("withheld by the provider", written(stop_reason="refusal", answered=False)),
        ("stopped for another reason", written(stop_reason="recitation")),
        ("stopped for an unknown reason", written(stop_reason="unknown")),
        (
            "declined without a model, because nothing was found",
            written(
                NOT_FOUND_MESSAGE, cite=False, answered=False, stop_reason="no_context", requests=0
            ),
        ),
        ("a model's reply that counts no request", written(requests=0)),
        ("without text", written("", cite=False, answered=False)),
        ("only white space", written(" \n ", cite=False)),
        (
            "an answer that came from the cache",
            written(cached=CacheHit("exact", STORED, "2031-02-03T04:05:06+00:00"), requests=0),
        ),
    ],
)
def test_a_reply_that_is_not_whole_or_not_the_answer_models_is_not_kept(
    name: str, answer: Answer
) -> None:
    store = memory_answers()
    cache = cache_over(store, {})

    keep(cache, STORED, answer)

    assert storable(answer) is False, name
    assert store.held().answers == 0, name
    assert cache.lookup(STORED).answer is None, name


@pytest.mark.parametrize(
    ("answer", "by_meaning"),
    [
        (written(), True),
        (written(requests=3), True),
        (DECLINE, False),
        (PARTIAL, False),
        (UNCITED, False),
        # The words of the fixed sentence are compared, as when a decline is recognised.
        (written("**i could not find this in the handbook**"), False),
    ],
)
def test_a_whole_reply_of_the_answer_model_is_kept_and_marked(
    answer: Answer, by_meaning: bool
) -> None:
    store = memory_answers()
    cache = cache_over(store, {})

    keep(cache, STORED, answer)

    assert storable(answer) is True
    assert servable_by_meaning(answer) is by_meaning
    stored = store.get(entry_id(comparable(STORED), scope_of(None)))
    assert stored is not None
    assert (stored.by_meaning, stored.answered) == (by_meaning, answer.answered)
    assert (stored.question, stored.comparable, stored.scope) == (
        STORED,
        comparable(STORED),
        "all",
    )


def test_the_answer_of_a_question_with_no_comparable_form_is_neither_looked_up_nor_kept() -> None:
    store = memory_answers()
    cache = cache_over(store, {}, TableVerifier())
    embedder = cache._embedder
    assert isinstance(embedder, Vectors)

    found = cache.lookup(" ?! ")
    cache.store(found, written())

    assert (found.key, found.answer) == ("", None)
    assert embedder.queries == []
    assert store.held().answers == 0


def test_the_question_is_embedded_once_when_the_tier_by_meaning_looked_it_up() -> None:
    cache = cache_over(memory_answers(), {STORED: towards(1.0)}, TableVerifier())
    embedder = cache._embedder
    assert isinstance(embedder, Vectors)

    keep(cache, STORED, written())

    assert embedder.queries == [STORED], "by the lookup, and not again to store the answer"
    # The exact tier embeds nothing to look up, so the question is embedded to be stored.
    exact = cache_over(memory_answers(), {STORED: towards(1.0)})
    embedder = exact._embedder
    assert isinstance(embedder, Vectors)
    keep(exact, STORED, written())
    assert embedder.queries == [STORED]


# --- when something fails ------------------------------------------------------------------


class Down:
    """A store that cannot be reached, whatever is asked of it."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def _fail(self, what: str) -> Any:
        self.asked.append(what)
        raise ConnectionError("connection refused by 10.1.2.3:6333")

    def ensure(self, dimension: int) -> None:
        self._fail("ensure")

    def get(self, entry_id: str) -> Entry | None:
        return self._fail("get")  # type: ignore[no-any-return]

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        return self._fail("nearest")  # type: ignore[no-any-return]

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        self._fail("put")


def test_a_store_that_is_down_is_a_miss_and_keeps_nothing_and_neither_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Down()
    cache = AnswerCache(store, embedder=Vectors({}), verifier=TableVerifier(), threshold=THRESHOLD)

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(STORED)
        cache.store(found, written())

    assert found.answer is None
    assert found.failure == "builtins.ConnectionError", "the kind of error, for the trace"
    # Asked once to read: a store that did not answer is not asked again by the next tier.
    assert store.asked == ["get", "put"]
    assert "the answer cache could not be read: builtins.ConnectionError" in caplog.text
    assert "the answer cache could not be stored: builtins.ConnectionError" in caplog.text


def test_a_store_that_fails_only_when_searched_is_a_miss_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class SearchFails(Down):
        def get(self, entry_id: str) -> Entry | None:
            return None

    cache = AnswerCache(
        SearchFails(), embedder=Vectors({}), verifier=TableVerifier(), threshold=THRESHOLD
    )

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(STORED)

    assert (found.answer, found.failure) == (None, "builtins.ConnectionError")
    assert "the answer cache could not be searched" in caplog.text


def test_a_verifier_that_fails_is_a_miss_and_the_exact_tier_still_serves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    verifier = TableVerifier(error=RuntimeError("the model file is corrupt"))
    cache = cache_over(
        memory_answers(), {STORED: towards(1.0), REWORDED: [1.0, 0.0, 0.0]}, verifier
    )
    keep(cache, STORED, written())

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(REWORDED)

    assert found.answer is None
    assert found.failure == "builtins.RuntimeError"
    assert "the answer cache could not be verified: builtins.RuntimeError" in caplog.text
    # The candidate is still named, with no verdict.
    assert [(c.question, c.entails) for c in found.checked] == [(STORED, None)]
    # The new answer is kept all the same, and the same question is served by its key.
    cache.store(found, written("A new answer."))
    assert cache.lookup(STORED).answer is not None
    again = cache.lookup(REWORDED).answer
    assert again is not None
    assert again.text == "A new answer."


def test_a_verifier_that_scores_another_number_of_pairs_serves_nothing() -> None:
    class OneTooFew(TableVerifier):
        def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
            return [Verification(1.0, 1.0)] * (len(pairs) - 1)

    cache, _ = stored_near(("one", 0.9, written()), ("two", 0.8, written()))
    cache._verifier = OneTooFew()

    found = cache.lookup("new")

    assert (found.answer, found.failure) == (None, "builtins.ValueError")


def test_an_entry_the_code_cannot_read_is_a_miss_and_is_replaced_by_the_next_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = memory_answers()
    cache = cache_over(store, {})
    keep(cache, STORED, written())
    point = entry_id(comparable(STORED), scope_of(None))
    store._client.set_payload(
        store.collection, payload={"chunks": "of another shape"}, points=[point]
    )

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(STORED)
        cache.store(found, written("Written again."))

    assert (found.answer, found.failure) == (None, "builtins.AttributeError")
    again = cache.lookup(STORED).answer
    assert again is not None
    assert again.text == "Written again."


def test_a_question_that_cannot_be_made_into_a_key_is_answered_and_not_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Half of a surrogate pair is a valid Python string that UTF-8 cannot encode. A
    # command line can hand one over, and the key of an entry is made from UTF-8.
    store = memory_answers()
    cache = cache_over(store, {}, TableVerifier())
    question = "How long is leave \ud800?"

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(question)
        cache.store(found, written())
        cache.store(cache.lookup(question, read=False), written())

    assert (found.answer, found.failure) == (None, "builtins.UnicodeEncodeError")
    assert store.held().answers == 0
    assert "the answer cache could not be stored: builtins.UnicodeEncodeError" in caplog.text


def test_a_service_that_cannot_load_its_verifier_says_so_when_it_warms_up() -> None:
    def load() -> TableVerifier:
        raise OSError("cross-encoder/nli-deberta-v3-small is not on disk")

    cache = cache_over(memory_answers(), {}, LazyVerifier(load, name="missing"))

    with pytest.raises(OSError, match="is not on disk"):
        cache.warm_up()


def test_warming_up_loads_the_verifier_and_asks_it_once() -> None:
    verifier = TableVerifier()
    lazy = LazyVerifier(lambda: verifier, name="lazy")
    cache = cache_over(memory_answers(), {}, lazy)

    assert lazy.loaded is False
    cache.warm_up()

    assert lazy.loaded is True
    assert len(verifier.calls) == 1
    assert len(verifier.calls[0]) == 1
    assert lazy.asked is True


def test_warming_up_runs_both_models_once_and_neither_without_the_second_tier() -> None:
    embedder, verifier = Vectors({}), TableVerifier()
    semantic = AnswerCache(
        memory_answers(), embedder=embedder, verifier=verifier, threshold=THRESHOLD
    )

    semantic.warm_up()

    # The pair the verifier reads, and the first of its two questions for the embedding.
    assert verifier.calls == [[cache_answers._WARM_UP]]
    assert embedder.queries == [cache_answers._WARM_UP[0]]
    # With the exact tier only, a lookup runs no model, so there is nothing to run first.
    alone = Vectors({})
    AnswerCache(memory_answers(), embedder=alone, threshold=THRESHOLD).warm_up()
    assert alone.queries == []


def test_an_embedding_model_that_fails_at_the_warm_up_is_logged_and_the_verifier_is_still_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken(Vectors):
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("out of memory")

    verifier = TableVerifier()
    cache = AnswerCache(
        memory_answers(), embedder=Broken({}), verifier=verifier, threshold=THRESHOLD
    )

    with caplog.at_level(logging.WARNING):
        cache.warm_up()

    assert "the answer cache could not be warmed up: builtins.RuntimeError" in caplog.text
    assert len(verifier.calls) == 1


def test_a_verifier_is_loaded_once_however_many_ask_it_at_the_same_time() -> None:
    loads: list[str] = []
    release = threading.Event()

    def load() -> TableVerifier:
        loads.append("loading")
        assert release.wait(5.0)
        return TableVerifier()

    lazy = LazyVerifier(load, name="some/nli-model")
    threads = [threading.Thread(target=lazy.score, args=([("a", "b")],)) for _ in range(4)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(5.0)

    assert loads == ["loading"]
    assert (lazy.name, lazy.loaded) == ("some/nli-model", True)


# --- storing in the background -------------------------------------------------------------


class Slow(QdrantAnswerStore):
    """A store whose writes wait until the test lets them go on."""

    def __init__(self) -> None:
        inner = memory_answers()
        super().__init__(inner._client, inner.collection)
        self.release = threading.Event()
        self.writing = threading.Event()
        self.puts = 0

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        self.writing.set()
        assert self.release.wait(5.0), "the test never let the store go on"
        self.puts += 1
        super().put(entry, vector)


def test_in_the_background_storing_returns_at_once_and_closing_waits_for_it() -> None:
    store = Slow()
    cache = cache_over(store, {}, background=True)
    found = cache.lookup(STORED)

    cache.store(found, written())

    assert store.writing.wait(5.0), "the write has begun, in a thread of its own"
    assert store.puts == 0, "and the caller did not wait for it"
    assert cache.close(timeout=0.05) is False, "closing waits its time and no longer"
    store.release.set()
    # The thread ends by itself once the write is through.
    for thread in threading.enumerate():
        if thread.name == "handbook-rag-cache":
            thread.join(5.0)
    assert store.puts == 1
    assert cache.lookup(STORED).answer is not None


def test_closing_lets_the_answers_still_waiting_be_stored_and_ends_the_thread() -> None:
    store = Slow()
    store.release.set()
    cache = cache_over(store, {}, background=True)
    for number in range(5):
        cache.store(cache.lookup(f"question {number}"), written())

    assert cache.close() is True

    assert store.held().answers == 5
    assert not [t for t in threading.enumerate() if t.name == "handbook-rag-cache" and t.is_alive()]
    # An answer stored after that is written before store returns.
    cache.store(cache.lookup("one more"), written())
    assert store.held().answers == 6


def test_more_answers_than_may_wait_are_dropped_and_logged_not_held_without_end(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cache_answers, "PENDING_WRITES", 2)
    store = Slow()
    cache = cache_over(store, {}, background=True)

    with caplog.at_level(logging.WARNING):
        for number in range(4):
            cache.store(Lookup(f"question {number}", None, f"question {number}"), written())

    assert "an answer was not cached: 2 others are still waiting to be stored" in caplog.text
    store.release.set()
    assert cache.close() is True
    assert store.held().answers == 2


def test_a_write_that_fails_in_the_background_is_logged_and_the_next_one_still_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailsOnce(Slow):
        def put(self, entry: Entry, vector: Sequence[float]) -> None:
            if self.puts == 0:
                self.puts += 1
                raise ConnectionError("connection reset")
            super().put(entry, vector)

    store = FailsOnce()
    store.release.set()
    cache = cache_over(store, {}, background=True)

    with caplog.at_level(logging.WARNING):
        cache.store(cache.lookup("first"), written())
        cache.store(cache.lookup("second"), written())
        assert cache.close() is True

    assert "the answer cache could not be stored: builtins.ConnectionError" in caplog.text
    assert store.held().answers == 1


# --- what a review's mutants got past ------------------------------------------------------


def stored_entry(question: str, **changes: Any) -> Entry:
    made = Entry(
        entry_id=entry_id(comparable(question), scope_of(None)),
        question=question,
        comparable=comparable(question),
        scope=scope_of(None),
        segments=written().segments,
        answered=True,
        model="stand-in-model-002",
        stop_reason="end_turn",
        stored_at="2031-02-03T04:05:06+00:00",
        by_meaning=True,
    )
    return dataclasses.replace(made, **changes)


def test_an_entry_under_the_right_id_that_is_of_another_part_of_the_handbook_is_not_served() -> (
    None
):
    # The id is a hash of the question and the part asked of. An entry under it that says
    # it was written from one section is not the answer to the question asked of all.
    store = memory_answers()
    cache = cache_over(store, {})
    store.put(stored_entry(STORED, scope=scope_of("people")), towards(1.0))

    assert store.get(entry_id(comparable(STORED), scope_of(None))) is not None
    assert cache.lookup(STORED).answer is None


def test_a_candidate_exactly_at_the_floor_is_read_by_the_verifier() -> None:
    verifier = TableVerifier({("at the floor", "new"): (1.0, 1.0)})
    cache = AnswerCache(
        Unfiltered([(stored_entry("at the floor"), 0.5), (stored_entry("under it"), 0.4999)]),
        embedder=Vectors({}),
        verifier=verifier,
        threshold=THRESHOLD,
        floor=0.5,
    )

    found = cache.lookup("new")

    assert verifier.calls == [[("at the floor", "new")]]
    assert found.answer is not None


def test_an_answer_that_came_from_the_cache_is_not_kept_though_it_is_whole() -> None:
    # Whole, written by the answer model, one request: all that sets it apart is its mark.
    whole = written(requests=1)
    served = dataclasses.replace(whole, cached=CacheHit("semantic", STORED, "2031-02-03T04:05:06"))

    assert storable(whole) is True
    assert storable(served) is False


# --- an answer that says in plain words that something is missing --------------------------

# What the answer model wrote in the saved answer run where part of an answer was missing
# (evals/runs/answers/baseline: q092, q112, u004 and u006), sentence by sentence.
SAID_WHEN_PART_WAS_MISSING = [
    "The handbook does not cover any additional data sources beyond these.",
    "The search results do not specify what *must* be included in the message to prevent "
    "it from being overlooked, other than ensuring there is an identified DRI.",
    "However, the specific monetary amount allocated is not provided in the handbook.",
    "The search results do not cover insurance payouts specifically for flight delays.",
]
# What it wrote in answers that lack nothing: a no that is the answer.
A_NO_THAT_IS_THE_ANSWER = [
    "Pods do not have a separate budget line.",
    "The Digital Experience team does not perform content changes, updates, or moving content.",
    "GitLab does not currently require subsequent background screenings for current team members.",
    "Your order will be paid for centrally, so you do not need to submit an expense report.",
    "No employee contribution is required.",
    "If an epic has no `section::` label, it will not show up on the AI org status dashboard.",
    "Quotations, order forms, agreements, and contracts are not valid.",
    "New features that do not leverage FIPS-validated modules cannot be made available to "
    "customers who use GitLab's FIPS releases.",
    "A login does not mean the account is active.",
    "They do not covertly record calls, and the statement of work lists no penalty.",
]


@pytest.mark.parametrize("sentence", SAID_WHEN_PART_WAS_MISSING)
def test_an_answer_that_says_in_plain_words_that_something_is_missing_is_no_candidate(
    sentence: str,
) -> None:
    answer = written(f"Each member has a budget for three quarters. {sentence}")
    cache, verifier = stored_near(("stored", 0.99, answer))

    assert storable(answer) is True, "it is kept, for the very same question"
    assert says_something_is_missing(answer) is True
    assert kept_from_the_second_tier(answer) == rules.SAYS_SOMETHING_IS_MISSING
    assert servable_by_meaning(answer) is False
    assert cache.lookup("new").answer is None
    assert verifier.calls == [], "no candidate, so the verifier is not asked"
    assert cache.lookup("stored").answer is not None


@pytest.mark.parametrize("sentence", A_NO_THAT_IS_THE_ANSWER)
def test_a_no_that_is_the_answer_keeps_an_answer_a_candidate(sentence: str) -> None:
    answer = written(sentence)

    assert says_something_is_missing(answer) is False
    assert kept_from_the_second_tier(answer) is None
    assert servable_by_meaning(answer) is True


def test_the_guard_errs_towards_keeping_an_answer_out_and_a_reply_can_get_past_it() -> None:
    # A plain answer about a benefit, in words the list holds: kept out, which costs a saving.
    assert servable_by_meaning(written("Dental care is not covered by the plan.")) is False
    # A decline in part, in words the list does not hold: a candidate like any other.
    silent = written("The budget runs for three quarters. The handbook is silent on the amount.")
    assert says_something_is_missing(silent) is False
    assert servable_by_meaning(silent) is True
    # So is one with more between the words than the list allows for.
    wordy = written("The handbook does not, as far as I can see, state the amount.")
    assert says_something_is_missing(wordy) is False


@pytest.mark.parametrize(
    "text",
    [
        "The handbook DOES NOT SPECIFY the amount.",
        "The handbook **doesn't** specify the amount.",
        "The handbook does\nnot   specify the amount.",
        "the amount is *not provided*.",
        "There is no information about the amount.",
        "I could not find the amount.",
        "The amount is not in the handbook.",
        # The instructions' own word for it, and the shortened form of "is not".
        "The search results do not contain the amount.",
        "The amount isn't mentioned in the search results.",
        # With the apostrophe a model may write: a typographic one.
        "The amounts aren\u2019t listed anywhere.",
        # With a word in between, as a model likes to put one.
        "The handbook does not explicitly state the amount.",
        "The amount is not specifically given.",
        "There is no specific information about the amount.",
        "It is not clear from the search results how much it is.",
    ],
)
def test_case_markup_spacing_and_an_apostrophe_do_not_hide_a_phrase(text: str) -> None:
    assert says_something_is_missing(written(text)) is True


def test_the_phrases_are_words_as_a_reply_is_compared_and_each_is_found_as_whole_words() -> None:
    assert len(MISSING_PHRASES) == len(set(MISSING_PHRASES))
    for phrase in MISSING_PHRASES:
        # Lower case, single spaces, no apostrophe: as plain_words leaves a reply.
        assert phrase == " ".join(phrase.split()) == phrase.lower(), phrase
        assert "'" not in phrase, phrase
        assert says_something_is_missing(written(f"Sixteen weeks. It is so: {phrase}.")), phrase
    for phrase in ("does not specify", "do not cover", "not provided", "not covered"):
        assert phrase in MISSING_PHRASES
    # The four the model used in the saved answer run are in it as they were written.
    assert {"does not cover", "do not cover", "do not specify", "not provided"} <= set(
        MISSING_PHRASES
    )
    assert len(MISSING_PHRASES) == 1093
    # A phrase inside a longer word is another word.
    assert says_something_is_missing(written("The teams do not specifying anything.")) is False
    assert says_something_is_missing(written("It is cannot provided.")) is False


def test_the_reason_a_reply_is_kept_from_the_second_tier_is_the_first_that_holds() -> None:
    missing = "The handbook does not specify the amount."

    assert kept_from_the_second_tier(DECLINE_IN_OTHER_WORDS) == rules.DECLINE
    assert kept_from_the_second_tier(DECLINE) == rules.DECLINE
    assert kept_from_the_second_tier(PARTIAL) == rules.DECLINES_IN_PART
    assert (
        kept_from_the_second_tier(written(missing, cite=False)) == rules.SAYS_SOMETHING_IS_MISSING
    )
    assert kept_from_the_second_tier(UNCITED) == rules.CITES_NOTHING
    assert kept_from_the_second_tier(written()) is None


DECLINE_IN_OTHER_WORDS = written("The handbook does not specify this.", answered=False)


# --- a lookup has a time limit -------------------------------------------------------------


class Hangs:
    """A store that stops at one of its calls until the test lets it go on."""

    def __init__(self, at: str) -> None:
        self.at = at
        self.release = threading.Event()
        self.entries = [(stored_entry(STORED), 0.99)]
        self.asked: list[str] = []

    def _wait(self, call: str) -> None:
        self.asked.append(call)
        if call == self.at:
            assert self.release.wait(10.0), "the test never let the store go on"

    def ensure(self, dimension: int) -> None:
        return None

    def get(self, entry_id: str) -> Entry | None:
        self._wait("get")
        return None

    def nearest(
        self, vector: Sequence[float], *, scope: str, limit: int, floor: float
    ) -> list[tuple[Entry, float]]:
        self._wait("nearest")
        return list(self.entries)

    def put(self, entry: Entry, vector: Sequence[float]) -> None:
        return None


class SlowVerifier(TableVerifier):
    """A verifier that reads for as long as the test makes it."""

    def __init__(self) -> None:
        super().__init__({(STORED, REWORDED): (1.0, 1.0)})
        self.release = threading.Event()

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        assert self.release.wait(10.0), "the test never let the verifier go on"
        return super().score(pairs)


@pytest.mark.parametrize("at", ["get", "nearest", "the verifier"])
def test_a_lookup_that_takes_too_long_is_given_up_and_counts_as_a_miss(
    at: str, caplog: pytest.LogCaptureFixture
) -> None:
    store, verifier = Hangs(at), SlowVerifier()
    if at != "the verifier":
        verifier.release.set()
    cache = AnswerCache(
        store, embedder=Vectors({}), verifier=verifier, threshold=THRESHOLD, lookup_seconds=0.1
    )

    began = time.monotonic()
    with caplog.at_level(logging.WARNING):
        found = cache.lookup(REWORDED)
    waited = time.monotonic() - began

    assert (found.answer, found.failure) == (None, "timeout")
    assert 0.1 <= waited < 2.0, "the caller waits its time and no longer"
    assert "the answer cache was given up after 0.1 s" in caplog.text
    # The answer the model then writes is kept as after any other miss.
    cache.store(found, written())
    store.release.set()
    verifier.release.set()


def test_within_its_time_a_lookup_that_goes_through_every_step_is_served() -> None:
    store, verifier = Hangs("nothing"), SlowVerifier()
    verifier.release.set()
    cache = AnswerCache(
        store, embedder=Vectors({}), verifier=verifier, threshold=THRESHOLD, lookup_seconds=5.0
    )

    found = cache.lookup(REWORDED)

    assert found.answer is not None
    assert (found.tier, found.failure) == ("semantic", None)
    with pytest.raises(ValueError, match="above 0 seconds"):
        cache_over(memory_answers(), {}, lookup_seconds=0)


def test_loading_the_verifier_is_not_counted_in_the_time_a_lookup_may_take() -> None:
    # A command loads the model the first time a stored question is near, which takes
    # longer than a lookup may. That is start-up, and the answer is still served.
    loaded = threading.Event()

    def load() -> TableVerifier:
        time.sleep(0.3)
        loaded.set()
        return TableVerifier({(STORED, REWORDED): (1.0, 1.0)})

    lazy = LazyVerifier(load, name="slow to load")
    cache = AnswerCache(
        Hangs("nothing"),
        embedder=Vectors({}),
        verifier=lazy,
        threshold=THRESHOLD,
        lookup_seconds=0.2,
    )

    found = cache.lookup(REWORDED)

    assert loaded.is_set()
    assert found.answer is not None
    assert found.failure is None


class SlowAtTimes(Vectors):
    """A stand-in embedding model whose calls take as long as the test says, one by one."""

    def __init__(self, seconds: Sequence[float]) -> None:
        super().__init__({})
        self._seconds = list(seconds)

    def embed_query(self, text: str) -> list[float]:
        if self._seconds:
            time.sleep(self._seconds.pop(0))
        return super().embed_query(text)


class ReadsSlowlyAtTimes(TableVerifier):
    """A stand-in verifier whose readings take as long as the test says, one by one."""

    def __init__(self, seconds: Sequence[float]) -> None:
        super().__init__({(STORED, REWORDED): (1.0, 1.0)})
        self._seconds = list(seconds)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        if self._seconds:
            time.sleep(self._seconds.pop(0))
        return super().score(pairs)


def test_the_first_embedding_of_a_process_is_not_counted_and_a_later_one_is() -> None:
    # The first time a process runs a model is slow, as loading it is. A command is a new
    # process for each question, so its one lookup would pay for that inside its limit.
    store = Hangs("nothing")
    cache = AnswerCache(
        store,
        embedder=SlowAtTimes([0.3, 0.3]),
        verifier=TableVerifier({(STORED, REWORDED): (1.0, 1.0)}),
        threshold=THRESHOLD,
        lookup_seconds=0.1,
    )

    first = cache.lookup(REWORDED)

    assert (first.tier, first.failure) == ("semantic", None), "three times the limit, and served"
    assert store.asked == ["get", "nearest"]
    # The next embedding takes as long, and is counted like everything else from now on.
    second = cache.lookup(REWORDED)
    assert (second.answer, second.failure) == (None, "timeout")
    # The step was still embedding when its caller left. It ends that, and does not go
    # on to ask the store.
    time.sleep(0.4)
    assert store.asked == ["get", "nearest", "get"]


def test_the_first_reading_of_a_verifier_just_loaded_is_not_counted_and_a_later_one_is() -> None:
    lazy = LazyVerifier(lambda: ReadsSlowlyAtTimes([0.3, 0.3]), name="slow the first time")
    cache = AnswerCache(
        Hangs("nothing"),
        embedder=Vectors({}),
        verifier=lazy,
        threshold=THRESHOLD,
        lookup_seconds=0.1,
    )
    assert (lazy.loaded, lazy.asked) == (False, False)

    first = cache.lookup(REWORDED)

    assert (first.tier, first.failure) == ("semantic", None), "three times the limit, and served"
    assert (lazy.loaded, lazy.asked) == (True, True)
    # The second reading takes as long, and is counted.
    second = cache.lookup(REWORDED)
    assert (second.answer, second.failure) == (None, "timeout")


@pytest.mark.parametrize("slow", ["the embedding", "the reading"])
def test_a_cache_that_was_warmed_up_counts_everything_from_its_first_question(slow: str) -> None:
    # A service runs both models once when it starts. After that nothing is slow for the
    # first time, and a slow embedding or a slow reading is given up like any other.
    embedder = SlowAtTimes([0.0, 0.3] if slow == "the embedding" else [])
    lazy = LazyVerifier(
        lambda: ReadsSlowlyAtTimes([0.0, 0.3] if slow == "the reading" else []), name="lazy"
    )
    cache = AnswerCache(
        Hangs("nothing"), embedder=embedder, verifier=lazy, threshold=THRESHOLD, lookup_seconds=0.1
    )

    cache.warm_up()
    found = cache.lookup(REWORDED)

    assert lazy.asked is True
    assert (found.answer, found.failure) == (None, "timeout")


def lookups_at_work(before: Sequence[threading.Thread] = ()) -> list[threading.Thread]:
    """The threads in which steps of lookups are running now, but for those named."""
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "handbook-rag-cache-lookup" and thread not in before
    ]


def test_a_lookup_that_was_given_up_does_not_take_a_turn_at_the_verifier() -> None:
    # The verifier reads for one lookup at a time. Three questions arrive one after
    # another while the first reading is still going on, and each is given up.
    before = lookups_at_work()
    verifier = SlowVerifier()
    cache = AnswerCache(
        Hangs("nothing"),
        embedder=Vectors({}),
        verifier=verifier,
        threshold=THRESHOLD,
        lookup_seconds=0.1,
    )

    given_up = [cache.lookup(REWORDED) for _ in range(3)]

    assert [found.failure for found in given_up] == ["timeout"] * 3
    # The two that waited for their turn stopped waiting when their callers did. Only
    # the one that is reading is still at work.
    patience = time.monotonic() + 2.0
    while len(lookups_at_work(before)) > 1 and time.monotonic() < patience:
        time.sleep(0.01)
    assert len(lookups_at_work(before)) == 1
    # And only that one reading was ever started: nothing reads for a caller who has left.
    verifier.release.set()
    patience = time.monotonic() + 2.0
    while lookups_at_work(before) and time.monotonic() < patience:
        time.sleep(0.01)
    assert len(verifier.calls) == 1
    # So the next question finds the verifier free, and is served within its time.
    served = cache.lookup(REWORDED)
    assert (served.tier, served.failure) == ("semantic", None)
    assert len(verifier.calls) == 2


def test_a_reading_whose_caller_has_left_is_not_started_when_its_turn_comes() -> None:
    # The turn can come in the moment the caller leaves. This is that moment by itself:
    # a lookup whose time is up, and a verifier that is free.
    verifier = TableVerifier({(STORED, REWORDED): (1.0, 1.0)})
    cache = AnswerCache(
        Hangs("nothing"), embedder=Vectors({}), verifier=verifier, threshold=THRESHOLD
    )
    near = cache_answers._Near(((stored_entry(STORED), 0.99),), (0.0, 0.0, 1.0))
    wait = cache_answers._Wait(0.05)
    with pytest.raises(cache_answers._TookTooLong):
        wait.run(time.sleep, 0.3)
    assert wait.over is True
    # Time enough to take the turn, were it only a matter of time.
    wait.not_counted(5.0)

    with pytest.raises(cache_answers._TookTooLong):
        cache._verify(REWORDED, None, comparable(REWORDED), near, wait)

    assert verifier.calls == []
    # The turn was given back: the next lookup is served.
    assert cache.lookup(REWORDED).tier == "semantic"
    # A lookup that still has its caller, and time, reads as always.
    in_time = cache_answers._Wait(5.0)
    assert in_time.over is False
    assert cache._verify(REWORDED, None, comparable(REWORDED), near, in_time).answer is not None


def test_a_step_is_not_started_once_the_time_of_its_lookup_is_up() -> None:
    wait = cache_answers._Wait(0.01)
    time.sleep(0.05)
    started: list[str] = []

    with pytest.raises(cache_answers._TookTooLong):
        wait.run(started.append, "a step")

    assert started == []
    assert wait.over is True
    assert wait.left < 0


def test_a_verifier_that_cannot_be_loaded_at_the_first_question_is_a_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def load() -> TableVerifier:
        raise OSError("the model is not on disk and the Hub cannot be reached")

    cache = AnswerCache(
        Hangs("nothing"),
        embedder=Vectors({}),
        verifier=LazyVerifier(load, name="missing"),
        threshold=THRESHOLD,
    )

    with caplog.at_level(logging.WARNING):
        found = cache.lookup(REWORDED)

    assert (found.answer, found.failure) == (None, "builtins.OSError")
    assert [(c.question, c.entails) for c in found.checked] == [(STORED, None)]
    assert "the answer cache could not be verified: builtins.OSError" in caplog.text


# --- the place for the answers is made before anything is looked up ------------------------


def test_preparing_makes_the_collection_in_the_size_of_the_embedding_model() -> None:
    store = memory_answers()
    cache = cache_over(store, {})
    assert not store._client.collection_exists(store.collection)

    assert cache.prepare() is True

    made = store._client.get_collection(store.collection).config.params.vectors
    assert made.size == Vectors({}).dimension == 3  # type: ignore[union-attr]
    # Nothing is in it, and reading it fails nothing.
    found = cache.lookup(STORED)
    assert (found.answer, found.failure) == (None, None)
    assert cache.prepare() is True, "a second time changes nothing"


def test_a_store_that_cannot_be_prepared_is_logged_and_nothing_else(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Down()
    cache = AnswerCache(store, embedder=Vectors({}), threshold=THRESHOLD)

    with caplog.at_level(logging.WARNING):
        assert cache.prepare() is False

    assert store.asked == ["ensure"]
    assert "the answer cache could not be prepared: builtins.ConnectionError" in caplog.text


def test_answers_still_waiting_when_the_cache_is_closed_are_counted_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Slow()
    cache = cache_over(store, {}, background=True)
    for number in range(3):
        cache.store(Lookup(f"question {number}", None, f"question {number}"), written())
    assert store.writing.wait(5.0)

    with caplog.at_level(logging.WARNING):
        assert cache.close(timeout=0.05) is False

    assert "3 answers were not stored in the answer cache" in caplog.text
    store.release.set()
