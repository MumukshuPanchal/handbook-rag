"""What a semantic cache would serve, measured before one is built.

A semantic cache answers a new question with the answer stored for an earlier one, when
the two mean the same. Its danger is a *false hit*: the stored answer served to a
question that only looks the same. The reply carries citations and looks like any other
answer, so the asker has no sign of it. A miss costs one ordinary answer.

The measurement uses pairs of a golden question and another question, labelled ``same``
or ``different`` (``evals/cache/pairs.jsonl``, with a card beside it), and two scores of
a stored question against a new one: the cosine of their embeddings, which is all a
plain semantic cache has, and a verifier that reads the two questions together
(``handbook_rag.cache.verify``). It has two views:

- **Pair view.** Each pair by itself: how each score spreads over the two kinds. It
  describes the scores, and says nothing final about a cache.
- **Cache view.** A cache that holds every golden question is asked each pair's other
  question. The nearest stored questions are its candidates, and the first that passes
  the threshold is served. This is what a cache would do, so it is the view that counts.

A threshold chosen and scored on the same pairs can flatter. So the pairs are split in
two by a rule fixed in code, a threshold is chosen for each score on one half, and the
result is reported on the other. Pairs that name the same golden question stay on one
side. The cache holds every golden question for both halves, so a question can still
meet a stored question that a pair of the other half names.

Nothing here asks an answer model anything. The pairs were written and labelled by one
model session and labelled again by a second, as their card describes. No person has read
them.
"""

import datetime
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from handbook_rag.cache.verify import Verification, Verifier
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.metrics import share_with_interval
from handbook_rag.evals.stage import write_json
from handbook_rag.index.embedder import Embedder

SAME = "same"
DIFFERENT = "different"
KINDS = (SAME, DIFFERENT)
HOWS = ("reworded", "short", "long", "synonyms", "indirect")
"""The ways the other question of a ``same`` pair differs from its golden question."""
WHAT_DIFFERS = ("region", "number", "entity", "negation", "scope", "aspect")
"""What makes the answers of a ``different`` pair differ."""
SOURCES = ("golden", "written")
"""How a ``different`` pair was made: from two golden questions on neighbouring subjects,
or from one golden question and a question written to change one decisive detail of it."""

COSINE = "cosine"
ENTAILS_FORWARD = "entails_forward"
ENTAILS_BACKWARD = "entails_backward"
ENTAILS_MIN = "entails_min"
ENTAILS_MEAN = "entails_mean"
SCORES = (COSINE, ENTAILS_FORWARD, ENTAILS_BACKWARD, ENTAILS_MIN, ENTAILS_MEAN)
"""Every score of a stored question against a new one, as the pair view reports them."""
POLICIES = (COSINE, ENTAILS_MIN, ENTAILS_MEAN)
"""The scores a cache could decide by, in the order of preference when two serve equally
many: the cosine needs no second model, and the smaller direction is the stricter of the
verifier's two."""
VERIFIER_POLICIES = (ENTAILS_MIN, ENTAILS_MEAN)
"""The two ways to decide by the verifier."""

RIGHT = "right"
"""A ``same`` question was served the answer of its own golden question."""
WRONG = "wrong"
"""A question was served an answer its label does not allow: a ``different`` question
anything at all, a ``same`` question the answer of another stored question."""
MISSED = "missed"
"""A ``same`` question was served nothing: a saving the cache did not make."""
NOT_SERVED = "not_served"
"""A ``different`` question was served nothing, which is right."""

CHOOSING = "choosing"
REPORTING = "reporting"

CANDIDATES = 3
"""How many of the nearest stored questions a new question is checked against."""
FLOOR = 0.5
"""The cosine below which a stored question is no candidate at all. It is there to keep a
cache from verifying questions that have nothing to do with each other. These pairs
cannot choose it: every question in them is near a stored one by construction."""
SPLIT_SEED = "cache-pairs-split-1"
"""Names the one split the results are reported for. It is fixed here, and the command
has no option for it. Whether a split is a typical one cannot be seen from a seed, which
is why the same rules are also run on other splits."""
OTHER_SPLITS = 200
"""How many other splits the same rules are run on, to show how far the result depends
on the split."""
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999)
"""The thresholds the descriptive curves are printed at. No result is chosen from them."""
HARD_NEAR_MISS = 0.8
"""The near-misses the verifier would serve at this score are listed by name, with the
half each is on. Whether a half tests a threshold depends on whether it holds one of
them. The score was set after the first run had been read, to describe that run, and no
result is chosen by it."""

PAIRS_PATH = Path("evals/cache/pairs.jsonl")
PAIRS_FILE = "pairs.jsonl"
QUESTIONS_FILE = "questions.jsonl"
SUMMARY_FILE = "summary.json"


class SecondLabel(BaseModel):
    """The label a second model session gave a pair without seeing the first."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    label: Literal["same", "different", "unsure"]
    confidence: Literal["high", "medium", "low"]
    basis: Literal["wording", "handbook", "golden"]
    reason: str = Field(min_length=1)
    flag: str | None = Field(default=None, min_length=1)


class CachePair(BaseModel):
    """One pair: a golden question, and another question that means the same or does not.

    Every field has to be there, also those that are ``null`` for the pair's kind. A
    field left out or misspelt would otherwise go unnoticed, and the measurement would
    be of other pairs than the card describes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    pair: str = Field(pattern=r"^[sd]\d{3}$")
    kind: Literal["same", "different"]
    """The maker's label."""
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    other: str = Field(min_length=1)
    other_id: str | None
    """The id of the other question when it is a golden question itself."""
    how: Literal["reworded", "short", "long", "synonyms", "indirect"] | None
    what_differs: Literal["region", "number", "entity", "negation", "scope", "aspect"] | None
    source: Literal["golden", "written"] | None
    note: str = Field(min_length=1)
    second: SecondLabel
    agreed: bool
    """Whether the second labelling gave the maker's label. Only such pairs are measured."""

    @property
    def group(self) -> str:
        """The way a ``same`` pair was reworded, or what makes a ``different`` pair differ."""
        group = self.how if self.kind == SAME else self.what_differs
        assert group is not None  # load_pairs refuses a pair without it
        return group


def load_pairs(path: Path, golden: Sequence[GoldenQuestion]) -> list[CachePair]:
    """Read the pairs, and refuse any pair that could not measure what it says.

    A pair is held to the golden set it was written from: its golden question word for
    word, and an other question that is a golden question exactly when the pair says so.
    The simulated cache stores the golden questions, so a pair that names one wrongly
    would be scored against the wrong entry.
    """
    if not path.is_file():
        raise DatasetError(f"cache question pairs not found: {path}")
    by_id = {question.id: question for question in golden}
    ids_by_text = {question.question: question.id for question in golden}
    pairs: list[CachePair] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            pair = CachePair.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise DatasetError(f"{path}:{number}: not a cache question pair: {error}") from error
        problem = _fault_in_pair(pair, by_id, ids_by_text)
        if problem:
            raise DatasetError(f"{path}:{number}: {pair.pair} {problem}")
        if pair.pair in seen:
            raise DatasetError(f"{path}:{number}: the id {pair.pair} is used twice")
        seen.add(pair.pair)
        pairs.append(pair)
    if not pairs:
        raise DatasetError(f"{path} holds no cache question pair")
    return pairs


def _fault_in_pair(
    pair: CachePair, golden: Mapping[str, GoldenQuestion], ids_by_text: Mapping[str, str]
) -> str | None:
    """What makes a pair unable to measure what it says, if anything."""
    if pair.pair.startswith("s") != (pair.kind == SAME):
        return f"is labelled {pair.kind}, which its id does not say"
    if pair.question_id not in golden:
        return f"names the golden question {pair.question_id}, which the golden set does not have"
    if pair.question != golden[pair.question_id].question:
        return f"does not give the golden question {pair.question_id} word for word"
    if not pair.other.strip() or not pair.note.strip():
        return "has a blank other question or note"
    if pair.other == pair.question:
        return "asks the golden question itself as its other question"
    if pair.agreed != (pair.second.label == pair.kind):
        # The measurement keeps the pairs marked as agreed, so the mark has to be true.
        return (
            f"says agreed is {str(pair.agreed).lower()}, and its second label is "
            f"{pair.second.label} against {pair.kind}"
        )
    if pair.kind == SAME:
        if pair.how is None or pair.what_differs is not None or pair.source is not None:
            return "is a same pair, which has how and neither what_differs nor source"
    elif pair.how is not None or pair.what_differs is None or pair.source is None:
        return "is a different pair, which has what_differs and source and no how"
    if (pair.source == "golden") != (pair.other_id is not None):
        return "has an other_id exactly when its source is golden, and this one does not"
    if pair.other_id is None:
        if pair.other in ids_by_text:
            # Its own entry would then sit in the simulated cache without being left out.
            return f"has the golden question {ids_by_text[pair.other]} as its other question"
        return None
    if pair.other_id == pair.question_id or pair.other_id not in golden:
        return f"names {pair.other_id!r} as its other question, which is no other golden question"
    if pair.other != golden[pair.other_id].question:
        return f"does not give the golden question {pair.other_id} word for word as its other"
    return None


def agreed_pairs(pairs: Iterable[CachePair]) -> list[CachePair]:
    """The pairs both labellings agree on, which are the ones every figure is about."""
    return [pair for pair in pairs if pair.agreed]


@dataclass(frozen=True, slots=True)
class Scores:
    """The scores of one stored question against one new question."""

    cosine: float
    entails: Verification
    """The stored question is the first of the two, so ``forward`` is the probability
    that the stored question implies the new one."""

    def of(self, score: str) -> float:
        """One of :data:`SCORES`, by its name."""
        if score == COSINE:
            return self.cosine
        if score == ENTAILS_MIN:
            return self.entails.smaller
        if score == ENTAILS_MEAN:
            return self.entails.mean
        if score == ENTAILS_FORWARD:
            return self.entails.forward
        if score == ENTAILS_BACKWARD:
            return self.entails.backward
        raise ValueError(f"There is no score named {score!r}.")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A stored question near enough to a new one to be checked."""

    question_id: str
    scores: Scores


@dataclass(frozen=True, slots=True)
class Asked:
    """One pair's other question, put to the simulated cache."""

    pair: CachePair
    own: Scores
    """The scores of the pair's own golden question against the other question: the pair
    view. They are kept whether or not that question is among the candidates."""
    candidates: tuple[Candidate, ...]
    """The nearest stored questions above the floor, nearest first."""
    above_floor: int
    """How many stored questions were above the floor, before the nearest were taken."""
    excluded: str | None
    """The stored question left out because it has the very words asked (see
    :func:`measure`)."""


@dataclass(frozen=True, slots=True)
class Measured:
    """Every pair with its scores and candidates, and what computing them took."""

    asked: list[Asked]
    stored: int
    """How many questions the simulated cache held."""
    unanswered: frozenset[str]
    """Those among them that the handbook does not answer."""
    candidates: int
    floor: float
    embedded: int
    verified: int
    embedding_seconds: float
    verifying_seconds: float
    decline_candidates: bool = True
    """Whether a stored question the handbook does not answer could be a candidate (see
    :func:`measure`)."""


def measure(
    pairs: Sequence[CachePair],
    golden: Sequence[GoldenQuestion],
    *,
    embedder: Embedder,
    verifier: Verifier,
    candidates: int = CANDIDATES,
    floor: float = FLOOR,
    decline_candidates: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> Measured:
    """Embed every question, find each new question's candidates, and verify them.

    The cache holds every golden question, also those the handbook does not answer: a
    stored decline served to a question that has an answer is a false hit too. Each
    question is embedded by itself with ``embed_query``, which is how a question is
    embedded for search, so the vectors are the ones a cache would compare.

    With ``decline_candidates`` off, the cache simulated is the one the application
    builds (``handbook_rag.cache.answers``). It keeps a decline for the very question
    that was declined and never offers it for another, so a stored question the handbook
    does not answer is left out of every question's candidates, and the next nearest
    takes its place. That such a question holds a decline is taken from the golden
    set's label. No model was asked, and a model that answers one of them anyway would
    leave an answer in a real cache.

    A stored question that has the very words asked is never a candidate. That happens
    in the pairs whose other question is a golden question: its own entry would be the
    nearest, with a cosine of 1, and serving it would be an exact match, which is another
    mechanism and says nothing about telling two different wordings apart.

    This is the only function here that runs a model. Everything after it is counting.
    """
    if candidates < 1:
        raise ValueError("A question needs at least one candidate to be checked against.")
    vectors: dict[str, tuple[list[float], float]] = {}
    embedding_seconds = 0.0

    def vector(text: str) -> tuple[list[float], float]:
        nonlocal embedding_seconds
        if text not in vectors:
            started = clock()
            values = embedder.embed_query(text)
            embedding_seconds += clock() - started
            vectors[text] = (values, math.sqrt(sum(value * value for value in values)))
        return vectors[text]

    stored = [(question.id, question.question, vector(question.question)) for question in golden]
    texts = {question_id: text for question_id, text, _ in stored}
    declines = frozenset() if decline_candidates else unanswered_questions(golden)
    near: dict[str, list[tuple[float, str]]] = {}
    own_cosine: dict[str, float] = {}
    above: dict[str, int] = {}
    excluded: dict[str, str | None] = {}
    for pair in pairs:
        if texts.get(pair.question_id) != pair.question or pair.other == pair.question:
            raise DatasetError(
                f"{pair.pair} cannot be asked: its golden question {pair.question_id} is not "
                "stored word for word, or is the question asked."
            )
        asked = vector(pair.other)
        excluded[pair.pair] = None
        ranked: list[tuple[float, str]] = []
        for question_id, text, entry in stored:
            if text == pair.other:
                excluded[pair.pair] = question_id
                continue
            cosine = _cosine(entry, asked)
            if question_id == pair.question_id:
                own_cosine[pair.pair] = cosine
            if cosine >= floor and question_id not in declines:
                ranked.append((cosine, question_id))
        # Nearest first. Two stored questions at the same distance keep the order of their ids.
        ranked.sort(key=lambda found: (-found[0], found[1]))
        above[pair.pair] = len(ranked)
        near[pair.pair] = ranked[:candidates]

    # The pair's own golden question is verified whether or not it is a candidate. The
    # stored question goes first and the new one second.
    wanted: dict[tuple[str, str], None] = {}
    for pair in pairs:
        wanted[(pair.question, pair.other)] = None
        for _, question_id in near[pair.pair]:
            wanted[(texts[question_id], pair.other)] = None
    started = clock()
    verifications = dict(zip(wanted, verifier.score(list(wanted)), strict=True))
    verifying_seconds = clock() - started

    def scores(pair: CachePair, question_id: str, cosine: float) -> Scores:
        return Scores(cosine, verifications[(texts[question_id], pair.other)])

    return Measured(
        asked=[
            Asked(
                pair=pair,
                own=scores(pair, pair.question_id, own_cosine[pair.pair]),
                candidates=tuple(
                    Candidate(question_id, scores(pair, question_id, cosine))
                    for cosine, question_id in near[pair.pair]
                ),
                above_floor=above[pair.pair],
                excluded=excluded[pair.pair],
            )
            for pair in pairs
        ],
        stored=len(stored),
        unanswered=unanswered_questions(golden),
        candidates=candidates,
        floor=floor,
        embedded=len(vectors),
        verified=len(wanted),
        embedding_seconds=embedding_seconds,
        verifying_seconds=verifying_seconds,
        decline_candidates=decline_candidates,
    )


def _cosine(first: tuple[list[float], float], second: tuple[list[float], float]) -> float:
    """The cosine of two vectors, each given with its length."""
    if not first[1] or not second[1]:
        return 0.0
    dot = sum(x * y for x, y in zip(first[0], second[0], strict=True))
    return max(-1.0, min(1.0, dot / (first[1] * second[1])))


@dataclass(frozen=True, slots=True)
class Split:
    """The agreed pairs in two halves: one to choose a threshold on, one to report on."""

    choosing: frozenset[str]
    reporting: frozenset[str]
    groups: dict[str, str]
    """For each pair, the group it was dealt with: pairs that share a golden question."""

    def half_of(self, pair_id: str) -> str | None:
        if pair_id in self.choosing:
            return CHOOSING
        return REPORTING if pair_id in self.reporting else None


def split_pairs(pairs: Sequence[CachePair], *, seed: str = SPLIT_SEED) -> Split:
    """Deal the pairs into two halves so that no golden question is on both sides.

    Pairs are not independent: many golden questions are in several pairs, and a quirk of
    one question's wording moves all of its pairs at once. A threshold chosen with some of
    them and scored on the others would have seen part of its test. So two pairs are in
    one group when they name the same golden question, as the question or as the other
    one, directly or through a third pair, and a group is never divided.

    The groups are put in an order that the seed fixes and no score can influence. Each
    goes to the half that it leaves the two halves most alike with, counting ``same``
    and ``different`` pairs apart, and to the choosing half when it makes no difference.
    Only the labels are looked at.
    """
    leader: dict[str, str] = {}

    def find(question_id: str) -> str:
        leader.setdefault(question_id, question_id)
        while leader[question_id] != question_id:
            leader[question_id] = leader[leader[question_id]]
            question_id = leader[question_id]
        return question_id

    for pair in pairs:
        first = find(pair.question_id)
        if pair.other_id is not None:
            second = find(pair.other_id)
            # The smaller id leads, so a group is named the same in whatever order it is met.
            leader[max(first, second)] = min(first, second)
    members: dict[str, list[CachePair]] = {}
    for pair in pairs:
        members.setdefault(find(pair.question_id), []).append(pair)

    def place(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()

    dealt: dict[str, list[str]] = {CHOOSING: [], REPORTING: []}
    held: dict[str, Counter[str]] = {CHOOSING: Counter(), REPORTING: Counter()}
    for group in sorted(members, key=place):
        joining: Counter[str] = Counter(pair.kind for pair in members[group])
        # How far apart the halves are, kind by kind, once the group has joined each.
        apart = {
            half: sum(abs(held[half][kind] + joining[kind] - held[other][kind]) for kind in KINDS)
            for half, other in ((CHOOSING, REPORTING), (REPORTING, CHOOSING))
        }
        half = CHOOSING if apart[CHOOSING] <= apart[REPORTING] else REPORTING
        held[half] += joining
        dealt[half] += [pair.pair for pair in members[group]]
    return Split(
        choosing=frozenset(dealt[CHOOSING]),
        reporting=frozenset(dealt[REPORTING]),
        groups={pair.pair: group for group, found in members.items() for pair in found},
    )


def _of_kind(pairs: Iterable[CachePair], kind: str) -> int:
    return sum(pair.kind == kind for pair in pairs)


def served(asked: Asked, policy: str, threshold: float | None) -> str | None:
    """The stored question whose answer a cache would serve, or ``None`` for a miss.

    The candidates are tried nearest first, and the first whose score reaches the
    threshold is served. Deciding by the cosine, that can only be the nearest one.
    ``threshold`` is ``None`` for a cache that serves nothing.
    """
    if threshold is None:
        return None
    for candidate in asked.candidates:
        if candidate.scores.of(policy) >= threshold:
            return candidate.question_id
    return None


def outcome(pair: CachePair, served_id: str | None) -> str:
    """What the labels say of serving a stored question's answer for a pair's other question.

    A label is about one stored question only, the pair's own. So one outcome rests on
    no label at all: a question served the answer of *another* stored question. It is
    counted as wrong, for both kinds of pair, because no label says it is right. Nobody
    checked those answers, and the run lists each such case so that they can be read.
    """
    if served_id is None:
        return MISSED if pair.kind == SAME else NOT_SERVED
    if pair.kind == SAME and served_id == pair.question_id:
        return RIGHT
    return WRONG


@dataclass(frozen=True, slots=True)
class Tally:
    """What a cache served to a set of questions, counted by what the labels say of it."""

    same: int
    right: int
    wrong_same: int
    """``same`` questions served the answer of another stored question."""
    missed: int
    different: int
    wrong_own: int
    """``different`` questions served the answer of their own golden question: the false
    hit each of them was written to be."""
    wrong_other: int
    """``different`` questions served the answer of another stored question."""

    @property
    def wrong_different(self) -> int:
        return self.wrong_own + self.wrong_other

    @property
    def wrong(self) -> int:
        return self.wrong_same + self.wrong_different

    @property
    def not_served(self) -> int:
        return self.different - self.wrong_different


def tally(asked: Iterable[Asked], policy: str, threshold: float | None) -> Tally:
    """Count what a cache with this score and threshold serves to these questions."""
    counts: Counter[str] = Counter()
    for question in asked:
        pair = question.pair
        served_id = served(question, policy, threshold)
        counts[pair.kind] += 1
        result = outcome(pair, served_id)
        if result != WRONG:
            counts[result] += 1
        elif pair.kind == SAME:
            counts["wrong_same"] += 1
        else:
            counts["wrong_own" if served_id == pair.question_id else "wrong_other"] += 1
    return Tally(
        same=counts[SAME],
        right=counts[RIGHT],
        wrong_same=counts["wrong_same"],
        missed=counts[MISSED],
        different=counts[DIFFERENT],
        wrong_own=counts["wrong_own"],
        wrong_other=counts["wrong_other"],
    )


@dataclass(frozen=True, slots=True)
class WrongServe:
    """A stored question served wrongly for a pair's other question."""

    pair: str
    served: str
    score: float
    """The highest threshold at which it is served."""


@dataclass(frozen=True, slots=True)
class Threshold:
    """The threshold chosen for one score on a set of questions."""

    value: float | None
    """``None`` when no threshold serves anything without serving something wrong."""
    decided_by: WrongServe | None
    """The wrong serve with the highest score, which the threshold has to stay above.
    ``None`` when nothing wrong could be served at any threshold."""
    halfway_to: float | None = None
    """The next score seen above that wrong serve. The threshold is put halfway between
    the two, so a question that scores between the threshold and this is served only
    because of where in that gap the rule puts the threshold."""


def wrong_serves(question: Asked, policy: str) -> Iterator[WrongServe]:
    """Each wrong answer a question would be served at some threshold.

    A candidate is served where its score reaches the threshold and no nearer candidate's
    does: at every threshold above the best score of those before it, up to its own. One
    that a nearer candidate outscores is never reached, since the nearer one passes at
    every threshold it passes.
    """
    ahead = -math.inf
    for candidate in question.candidates:
        score = candidate.scores.of(policy)
        if score <= ahead:
            continue
        ahead = score
        if outcome(question.pair, candidate.question_id) == WRONG:
            yield WrongServe(question.pair.pair, candidate.question_id, score)


def highest_wrong_serve(asked: Iterable[Asked], policy: str) -> WrongServe | None:
    """The wrong serve with the highest score among these questions.

    No threshold at or below that score is safe for them, so it is the one that sets a
    threshold, and the one a set of questions tests a threshold with. Of two with the
    same score, the first met is named.
    """
    worst: WrongServe | None = None
    for question in asked:
        for serve in wrong_serves(question, policy):
            if worst is None or serve.score > worst.score:
                worst = serve
    return worst


def choose_threshold(asked: Sequence[Asked], policy: str) -> Threshold:
    """The loosest threshold from which upwards nothing wrong is served to these questions.

    A false hit costs far more than a miss, so the rule allows none on the questions it
    sees: the threshold is put above the highest score at which a wrong answer would be
    served, halfway to the next score seen. Above it nothing wrong is served, however
    strict the threshold is made. If that highest score is the highest of all, no
    threshold serves anything safely, and the cache serves nothing.

    Raising a threshold can bring a farther candidate into play, which is why the rule
    looks at every threshold above the one it chooses and not at that one alone (see
    :func:`wrong_serves`).

    The rule sits exactly on the edge of what these questions show, so on other
    questions something wrong can still be served. That is what the other half is for.
    """
    seen = {c.scores.of(policy) for question in asked for c in question.candidates}
    worst = highest_wrong_serve(asked, policy)
    if worst is None:
        return Threshold(min(seen) if seen else None, None)
    higher = [score for score in seen if score > worst.score]
    if not higher:
        return Threshold(None, worst)
    above = min(higher)
    middle = (worst.score + above) / 2
    # Two scores can be so close that their middle is the lower one again.
    return Threshold(middle if middle > worst.score else above, worst, halfway_to=above)


def choose_policy(right: Mapping[str, int]) -> str | None:
    """The score that served the most ``same`` questions right at its own threshold.

    ``right`` holds that count for each of :data:`POLICIES`. Of two that serve equally
    many, the one named first there is taken. When none of them serves a single question
    right, there is nothing to choose between, and the answer is ``None``: by these
    rules a cache would serve nothing.
    """
    best = max(POLICIES, key=lambda policy: (right[policy], -POLICIES.index(policy)))
    return best if right[best] else None


def area_under_curve(same: Sequence[float], different: Sequence[float]) -> float:
    """The chance that a ``same`` pair scores above a ``different`` pair, a tie counting half.

    1 means a threshold exists that tells the two kinds apart without a mistake, and
    0.5 that the score is no better than a coin.
    """
    if not same or not different:
        raise ValueError("The area under the curve needs pairs of both kinds.")
    above = sum((s > d) + 0.5 * (s == d) for s in same for d in different)
    return above / (len(same) * len(different))


def fingerprint(value: object) -> str:
    """A short hash of what a run measured, so that a saved run names its data."""
    encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def data_fingerprints(
    pairs: Sequence[CachePair], golden: Sequence[GoldenQuestion]
) -> dict[str, str]:
    """Hashes of what the figures depend on: each pair's two questions and labels, and the
    stored questions with whether the handbook answers them. A note or a reason can be
    reworded without making a run stale."""
    return {
        "pairs_fingerprint": fingerprint(
            [
                [
                    p.pair,
                    p.kind,
                    p.question_id,
                    p.other,
                    p.other_id,
                    p.how,
                    p.what_differs,
                    p.source,
                    p.agreed,
                ]
                for p in pairs
            ]
        ),
        "stored_questions_fingerprint": fingerprint([[q.id, q.question, q.type] for q in golden]),
    }


def unanswered_questions(golden: Iterable[GoldenQuestion]) -> frozenset[str]:
    """The stored questions the handbook does not answer. What a cache holds for one of
    them is a decline, and a decline served to a question that has an answer is a false
    hit like any other."""
    return frozenset(question.id for question in golden if question.type == "unanswerable")


def summarise(
    asked: Sequence[Asked],
    *,
    stored: int,
    candidates: int,
    floor: float,
    unanswered: Collection[str] = (),
    seed: str = SPLIT_SEED,
    other_splits: int = OTHER_SPLITS,
    decline_candidates: bool = True,
) -> dict[str, Any]:
    """Every figure of the measurement, from the scores and candidates of each pair.

    It runs no model, so the figures of a saved run can be worked out again from its
    rows, and a test does that. ``unanswered`` names the stored questions the handbook
    does not answer (:func:`unanswered_questions`); the rows do not hold that.

    ``decline_candidates`` says how the candidates of the rows were found (see
    :func:`measure`). The counting is the same either way. With it off, the figures say
    so and count what was left out. With it on they are exactly what they were before
    the option existed, so the figures of the first saved run can still be worked out
    again from its rows.
    """
    agreed = [question for question in asked if question.pair.agreed]
    disputed = [question for question in asked if not question.pair.agreed]
    pairs = [question.pair for question in agreed]
    if not _of_kind(pairs, SAME) or not _of_kind(pairs, DIFFERENT):
        raise DatasetError("The measurement needs agreed pairs of both kinds.")
    by_id = {pair.pair: pair for pair in pairs}
    split = split_pairs(pairs, seed=seed)
    halves = {
        half: [question for question in agreed if split.half_of(question.pair.pair) == half]
        for half in (CHOOSING, REPORTING)
    }
    thresholds = {policy: choose_threshold(halves[CHOOSING], policy) for policy in POLICIES}
    on_choosing = {
        policy: tally(halves[CHOOSING], policy, thresholds[policy].value) for policy in POLICIES
    }
    chosen = choose_policy({policy: on_choosing[policy].right for policy in POLICIES})
    reported = {
        policy: tally(halves[REPORTING], policy, thresholds[policy].value) for policy in POLICIES
    }
    near_misses = _of_kind((question.pair for question in halves[REPORTING]), DIFFERENT)
    by_group = Counter(split.groups.values())

    def named(serve: WrongServe | None) -> dict[str, Any] | None:
        """A wrong serve as it is saved: the pair, and what it was served."""
        if serve is None:
            return None
        pair = by_id[serve.pair]
        return {
            "pair": pair.pair,
            "kind": pair.kind,
            "group": pair.group,
            "served": serve.served,
            "own": serve.served == pair.question_id,
            # A stored question the handbook does not answer: the answer served is a decline.
            "stored_unanswered": serve.served in unanswered,
            "score": serve.score,
        }

    def of_policy(policy: str) -> dict[str, Any]:
        threshold = thresholds[policy]
        return {
            "threshold": threshold.value,
            "decided_by": named(threshold.decided_by),
            "halfway_to": threshold.halfway_to,
            CHOOSING: _counts(on_choosing[policy]),
            # A cache that serves nothing by rule has no rate to estimate.
            REPORTING: _counts(reported[policy], intervals=threshold.value is not None),
            **{
                f"{half}_by_how": _by_group(halves[half], policy, threshold.value, SAME)
                for half in halves
            },
            **{
                f"{half}_by_what_differs": _by_group(
                    halves[half], policy, threshold.value, DIFFERENT
                )
                for half in halves
            },
            "wrong_on_reporting": [
                named(serve) for serve in _served_wrong(halves[REPORTING], policy, threshold.value)
            ],
            "right_only_by_the_margin": _right_only_by_the_margin(
                halves[REPORTING], policy, threshold
            ),
            # What each half could test a threshold with: the wrong serve it holds with
            # the highest score, for near-misses and for rewordings served another answer.
            "highest_wrong": {
                half: {
                    kind: named(
                        highest_wrong_serve(
                            (q for q in halves[half] if q.pair.kind == kind), policy
                        )
                    )
                    for kind in KINDS
                }
                for half in halves
            },
        }

    def hard_near_misses(policy: str) -> list[WrongServe]:
        """The near-misses this score would serve at :data:`HARD_NEAR_MISS`, hardest first."""
        found = [
            highest_wrong_serve([question], policy)
            for question in agreed
            if question.pair.kind == DIFFERENT
            and served(question, policy, HARD_NEAR_MISS) is not None
        ]
        return sorted((serve for serve in found if serve), key=lambda serve: -serve.score)

    hard = {policy: hard_near_misses(policy) for policy in VERIFIER_POLICIES}
    others = _other_splits(
        agreed,
        seed=seed,
        count=other_splits,
        hard={policy: [serve.pair for serve in found] for policy, found in hard.items()},
    )
    figures: dict[str, Any] = {
        "pairs": {
            "in_file": len(asked),
            "agreed": len(agreed),
            SAME: _of_kind(pairs, SAME),
            DIFFERENT: _of_kind(pairs, DIFFERENT),
            "disputed": [question.pair.pair for question in disputed],
            "other_is_a_stored_question": sum(q.excluded is not None for q in agreed),
            "different_by_source": _by_source(pairs),
        },
        "pair_view": {
            "what": (
                "Each agreed pair by itself: the scores of its golden question against its "
                "other question. A description of all agreed pairs, not a held-out result."
            ),
            "scores": {score: _pair_view(agreed, score) for score in SCORES},
        },
        "cache_view": {
            "stored_questions": stored,
            "stored_unanswered": len(unanswered),
            "candidates": candidates,
            "floor": floor,
            "own_question": {
                kind: {
                    "pairs": _of_kind(pairs, kind),
                    "nearest": sum(_place_of_own(q) == 1 for q in agreed if q.pair.kind == kind),
                    "among_candidates": sum(
                        _place_of_own(q) is not None for q in agreed if q.pair.kind == kind
                    ),
                }
                for kind in KINDS
            },
            "questions_with_fewer_candidates": sum(len(q.candidates) < candidates for q in agreed),
            "stored_above_floor": _range([q.above_floor for q in agreed]),
            "split": {
                "seed": seed,
                "groups": len(by_group),
                "largest_group": max(by_group.values()),
                **{
                    half: {
                        "pairs": len(halves[half]),
                        SAME: _of_kind((q.pair for q in halves[half]), SAME),
                        DIFFERENT: _of_kind((q.pair for q in halves[half]), DIFFERENT),
                        "groups": len({split.groups[q.pair.pair] for q in halves[half]}),
                        "groups_of": {
                            kind: _groups_of(split, halves[half], kind) for kind in KINDS
                        },
                        "different_by_source": _by_source(q.pair for q in halves[half]),
                        "candidates": _candidates_across(halves[half], halves[other]),
                    }
                    for half, other in ((CHOOSING, REPORTING), (REPORTING, CHOOSING))
                },
            },
            "policies": {policy: of_policy(policy) for policy in POLICIES},
            "chosen_policy": chosen,
            "zero_wrong_allows": {
                "near_misses": near_misses,
                "up_to": share_with_interval(0, near_misses).high if near_misses else None,
            },
            "near_misses_above": {
                "what": (
                    "The near-misses the second check would serve at this score, by name. "
                    "The score was set after the first run was read, to describe it. No "
                    "result is chosen by it."
                ),
                "score": HARD_NEAR_MISS,
                "policies": {
                    policy: {
                        "pairs": [
                            {**(named(serve) or {}), "half": split.half_of(serve.pair)}
                            for serve in found
                        ],
                        "other_splits_with_all_on_the_choosing_half": others["hard_on_choosing"][
                            policy
                        ],
                    }
                    for policy, found in hard.items()
                },
            },
            "whole_set": {
                "what": (
                    "All agreed pairs at fixed thresholds, and at the threshold the rule "
                    "would choose with all of them in view. A description: these "
                    "thresholds are scored on the pairs they are read from."
                ),
                "curves": {
                    policy: [
                        {"threshold": threshold, **_counts(tally(agreed, policy, threshold))}
                        for threshold in THRESHOLDS
                    ]
                    for policy in POLICIES
                },
                "chosen_on_all_pairs": {
                    policy: {
                        **_chosen_on(agreed, policy),
                        "decided_by": named(highest_wrong_serve(agreed, policy)),
                    }
                    for policy in POLICIES
                },
            },
            "other_splits": others["figures"],
        },
        "disputed": [
            {
                "pair": question.pair.pair,
                "kind": question.pair.kind,
                "second_label": question.pair.second.label,
                **_scores(question.own),
                "served": {
                    policy: served(question, policy, thresholds[policy].value)
                    for policy in POLICIES
                },
            }
            for question in disputed
        ],
    }
    if not decline_candidates:
        figures["cache_view"]["stored_declines"] = {
            "what": (
                "A stored question the handbook does not answer holds a decline, and a "
                "decline is no candidate for a match by meaning, as in the answer cache "
                "the application builds. That it holds a decline is the golden set's "
                "label, not a model's reply."
            ),
            "left_out": len(unanswered),
            "candidates_from": stored - len(unanswered),
            # The pairs whose own golden question is one of them: the wrong serve such a
            # pair was written to provoke cannot happen by meaning.
            "pairs_whose_golden_question_is_one": {
                kind: sum(p.kind == kind and p.question_id in unanswered for p in pairs)
                for kind in KINDS
            },
        }
    return figures


def _place_of_own(question: Asked) -> int | None:
    """Where the pair's own golden question stands among the candidates, counted from 1."""
    for place, candidate in enumerate(question.candidates, start=1):
        if candidate.question_id == question.pair.question_id:
            return place
    return None


def _spread(values: Sequence[float]) -> dict[str, Any]:
    return {
        "pairs": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _by_source(pairs: Iterable[CachePair]) -> dict[str, int]:
    """How many ``different`` pairs are of each source: two golden questions, or a golden
    question and one written to change a decisive detail of it."""
    counted: Counter[str | None] = Counter(pair.source for pair in pairs if pair.kind == DIFFERENT)
    return {source: counted[source] for source in SOURCES}


def _groups_of(split: Split, asked: Iterable[Asked], kind: str) -> dict[str, int]:
    """How many groups the pairs of one kind come from, and how many the largest gives.

    Pairs of one group share a golden question, so they are not independent, and an
    interval that counts each pair as one draw is too narrow.
    """
    counted = Counter(split.groups[q.pair.pair] for q in asked if q.pair.kind == kind)
    return {"groups": len(counted), "largest": max(counted.values(), default=0)}


def _candidates_across(asked: Sequence[Asked], other_half: Sequence[Asked]) -> dict[str, int]:
    """How many candidates of one half are stored questions that the other half names.

    The split keeps apart the golden questions that pairs name. The cache holds every
    golden question for both halves, so a question can still meet, as a candidate, a
    stored question that a pair of the other half is about.
    """
    named = {q.pair.question_id for q in other_half} | {
        q.pair.other_id for q in other_half if q.pair.other_id
    }
    others = [
        candidate.question_id
        for q in asked
        for candidate in q.candidates
        if candidate.question_id != q.pair.question_id
    ]
    return {
        "not_own": len(others),
        "named_by_the_other_half": sum(question_id in named for question_id in others),
    }


def _pair_view(agreed: Sequence[Asked], score: str) -> dict[str, Any]:
    """How one score spreads over the pairs, each compared with its own golden question."""

    def values(kind: str, group: str | None = None, source: str | None = None) -> list[float]:
        return [
            q.own.of(score)
            for q in agreed
            if q.pair.kind == kind
            and (group is None or q.pair.group == group)
            and (source is None or q.pair.source == source)
        ]

    same, different = values(SAME), values(DIFFERENT)
    return {
        "area_under_curve": area_under_curve(same, different),
        "by_kind": {SAME: _spread(same), DIFFERENT: _spread(different)},
        "by_how": {how: _spread(values(SAME, how)) for how in HOWS if values(SAME, how)},
        "by_what_differs": {
            what: _spread(values(DIFFERENT, what))
            for what in WHAT_DIFFERS
            if values(DIFFERENT, what)
        },
        "by_source": {
            source: _spread(values(DIFFERENT, source=source))
            for source in SOURCES
            if values(DIFFERENT, source=source)
        },
        "at_or_above": [
            {
                "threshold": threshold,
                SAME: sum(value >= threshold for value in same),
                DIFFERENT: sum(value >= threshold for value in different),
            }
            for threshold in THRESHOLDS
        ],
    }


def _estimate(successes: int, total: int) -> dict[str, Any] | None:
    return asdict(share_with_interval(successes, total)) if total else None


def _counts(counted: Tally, *, intervals: bool = False) -> dict[str, Any]:
    """A tally as it is saved. With ``intervals``, each share has a 95% Wilson interval."""
    same: dict[str, Any] = {
        "pairs": counted.same,
        RIGHT: counted.right,
        WRONG: counted.wrong_same,
        MISSED: counted.missed,
    }
    different: dict[str, Any] = {
        "pairs": counted.different,
        WRONG: counted.wrong_different,
        "wrong_own": counted.wrong_own,
        "wrong_other": counted.wrong_other,
        NOT_SERVED: counted.not_served,
    }
    if intervals:
        for name in (RIGHT, WRONG, MISSED):
            same[f"{name}_share"] = _estimate(same[name], counted.same)
        different[f"{WRONG}_share"] = _estimate(counted.wrong_different, counted.different)
    return {SAME: same, DIFFERENT: different}


def _by_group(
    asked: Sequence[Asked], policy: str, threshold: float | None, kind: str
) -> dict[str, dict[str, Any]]:
    """The counts of one kind of pair by its group. Counts only: the groups are too small
    for an interval to say anything."""
    groups = HOWS if kind == SAME else WHAT_DIFFERS
    rows: dict[str, dict[str, Any]] = {}
    for group in groups:
        members = [q for q in asked if q.pair.kind == kind and q.pair.group == group]
        if members:
            rows[group] = _counts(tally(members, policy, threshold))[kind]
    return rows


def _served_wrong(asked: Sequence[Asked], policy: str, threshold: float | None) -> list[WrongServe]:
    """Each wrong serve at one threshold, so that it can be named and read: only the
    pair's own golden question is covered by its label."""
    found = []
    for question in asked:
        served_id = served(question, policy, threshold)
        if served_id is not None and outcome(question.pair, served_id) == WRONG:
            score = next(
                candidate.scores.of(policy)
                for candidate in question.candidates
                if candidate.question_id == served_id
            )
            found.append(WrongServe(question.pair.pair, served_id, score))
    return found


def _right_only_by_the_margin(
    asked: Sequence[Asked], policy: str, threshold: Threshold
) -> list[dict[str, Any]]:
    """The questions served right at the threshold that would be missed at the next score.

    The rule puts the threshold halfway between the wrong serve it has to stay above and
    the next score seen. Putting it at that next score would be as safe on the choosing
    half, so whatever scores in the upper half of the gap is served by that choice alone.
    """
    if threshold.value is None or threshold.halfway_to is None:
        return []
    return [
        {
            "pair": question.pair.pair,
            "group": question.pair.group,
            "score": next(
                candidate.scores.of(policy)
                for candidate in question.candidates
                if candidate.question_id == question.pair.question_id
            ),
        }
        for question in asked
        if outcome(question.pair, served(question, policy, threshold.value)) == RIGHT
        and outcome(question.pair, served(question, policy, threshold.halfway_to)) != RIGHT
    ]


def _chosen_on(asked: Sequence[Asked], policy: str) -> dict[str, Any]:
    threshold = choose_threshold(asked, policy)
    return {"threshold": threshold.value, **_counts(tally(asked, policy, threshold.value))}


def _other_splits(
    agreed: Sequence[Asked], *, seed: str, count: int, hard: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    """The same rules on other splits of the same pairs.

    One split is one draw: with about fifty pairs of each kind on a side, a single hard
    pair landing on one side or the other moves the result. Running the rules on many
    splits shows how much. It is still the same pairs each time, so it shows the
    dependence on the split, and not what other questions would give.

    The results are also given apart by where one pair fell: the pair whose wrong serve
    scores highest with every pair in view. On the choosing half it sets the threshold.
    On the reporting half it is the hardest test a threshold meets.

    ``hard`` names, for each score it is given for, the near-misses to look for on the
    choosing half. Returned under ``hard_on_choosing``: in how many splits all of them
    are there, or ``None`` where there are none to look for.
    """
    pairs = [question.pair for question in agreed]
    hardest = {policy: highest_wrong_serve(agreed, policy) for policy in POLICIES}
    splits: list[Split] = []
    runs: dict[str, list[Tally]] = {policy: [] for policy in POLICIES}
    without: Counter[str] = Counter()
    chosen: list[str | None] = []
    of_choice: list[Tally] = []
    for number in range(1, count + 1):
        split = split_pairs(pairs, seed=f"{seed}/other-{number}")
        splits.append(split)
        choosing = [q for q in agreed if q.pair.pair in split.choosing]
        reporting = [q for q in agreed if q.pair.pair in split.reporting]
        right: dict[str, int] = {}
        for policy in POLICIES:
            threshold = choose_threshold(choosing, policy).value
            without[policy] += threshold is None
            right[policy] = tally(choosing, policy, threshold).right
            runs[policy].append(tally(reporting, policy, threshold))
        chosen.append(choose_policy(right))
        # Where the rules choose no score, the cache serves nothing to anyone.
        served_nothing = tally(reporting, COSINE, None)
        of_choice.append(runs[chosen[-1]][-1] if chosen[-1] is not None else served_nothing)

    def across(tallies: Sequence[Tally]) -> dict[str, Any]:
        return {
            "right_share": _range([t.right / t.same for t in tallies if t.same]),
            "wrong_share_of_different": _range(
                [t.wrong_different / t.different for t in tallies if t.different]
            ),
            "splits_with_nothing_wrong": sum(t.wrong == 0 for t in tallies),
            "splits_with_something_wrong": sum(t.wrong > 0 for t in tallies),
        }

    def where_it_fell(policy: str) -> dict[str, Any] | None:
        pair = hardest[policy]
        if pair is None:
            return None
        fell: dict[str, list[Tally]] = {CHOOSING: [], REPORTING: []}
        for split, counted in zip(splits, runs[policy], strict=True):
            fell[CHOOSING if pair.pair in split.choosing else REPORTING].append(counted)
        return {
            "pair": pair.pair,
            "score": pair.score,
            **{half: {"splits": len(found), **across(found)} for half, found in fell.items()},
        }

    return {
        "figures": {
            "splits": count,
            "policies": {
                policy: {
                    "chosen": chosen.count(policy),
                    "splits_without_a_threshold": without[policy],
                    **across(runs[policy]),
                    "by_where_the_hardest_pair_fell": where_it_fell(policy),
                }
                for policy in POLICIES
            },
            "chosen_policy": {
                "splits_without_a_choice": chosen.count(None),
                **across(of_choice),
            },
        },
        "hard_on_choosing": {
            policy: sum(all(pair in split.choosing for pair in found) for split in splits)
            if found
            else None
            for policy, found in hard.items()
        },
    }


def _range(values: Sequence[float]) -> dict[str, float] | None:
    """The lowest, middle and highest of some values, or ``None`` where there are none."""
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _scores(scores: Scores) -> dict[str, float]:
    return {score: scores.of(score) for score in SCORES}


def write_cache_run(
    measured: Measured,
    run_dir: Path,
    *,
    description: str,
    config: Mapping[str, object],
    measured_on: datetime.date,
    seed: str = SPLIT_SEED,
    other_splits: int = OTHER_SPLITS,
) -> dict[str, Any]:
    """Write a score for every pair, what the cache did with every question, and the figures.

    The rows hold each score to its last digit, so the figures can be worked out again
    from them. All three files are small and are committed for a run that is kept.
    ``config`` names the models and the data. What the figures were computed with is
    added to it here, so that a saved run cannot name other settings than it used. The
    day of the run is recorded too, so that a document that gives it can be held to it,
    and so is a hash of each file of rows, so that the figures name the rows they are of.
    """
    figures = summarise(
        measured.asked,
        stored=measured.stored,
        candidates=measured.candidates,
        floor=measured.floor,
        unanswered=measured.unanswered,
        seed=seed,
        other_splits=other_splits,
        decline_candidates=measured.decline_candidates,
    )
    split = split_pairs(agreed_pairs(q.pair for q in measured.asked), seed=seed)
    thresholds = {
        policy: figures["cache_view"]["policies"][policy]["threshold"] for policy in POLICIES
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for question in measured.asked:
        pair = question.pair
        half = split.half_of(pair.pair)
        pair_rows.append(
            {
                "pair": pair.pair,
                "kind": pair.kind,
                "how": pair.how,
                "what_differs": pair.what_differs,
                "question_id": pair.question_id,
                "other_id": pair.other_id,
                "agreed": pair.agreed,
                "half": half,
                **_scores(question.own),
            }
        )
        gave = {policy: served(question, policy, thresholds[policy]) for policy in POLICIES}
        question_rows.append(
            {
                "pair": pair.pair,
                "kind": pair.kind,
                "agreed": pair.agreed,
                "half": half,
                "question_id": pair.question_id,
                "excluded": question.excluded,
                "stored_above_floor": question.above_floor,
                "candidates": [
                    {
                        "question_id": candidate.question_id,
                        "own": candidate.question_id == pair.question_id,
                        **_scores(candidate.scores),
                    }
                    for candidate in question.candidates
                ],
                # What each score would serve at the threshold chosen for it.
                "served": gave,
                "outcome": {policy: outcome(pair, gave[policy]) for policy in POLICIES},
            }
        )
    _write_rows(run_dir / PAIRS_FILE, pair_rows)
    _write_rows(run_dir / QUESTIONS_FILE, question_rows)
    summary = {
        "description": description,
        "measured_on": measured_on.isoformat(),
        "config": {
            **config,
            "candidates": measured.candidates,
            "floor": measured.floor,
            "split_seed": seed,
            "other_splits": other_splits,
            # Recorded only when it is off, so that a run made as the first one was is
            # recorded as the first one was.
            **({} if measured.decline_candidates else {"decline_candidates": False}),
        },
        "rows": rows_fingerprints(run_dir),
        **figures,
        # On this machine, in this run. The verifier read its pairs in batches, so the
        # time for a pair is not the time one question would wait for.
        "timing": {
            "questions_embedded": measured.embedded,
            "embedding_seconds": round(measured.embedding_seconds, 1),
            "embedding_ms_per_question": _per_item_ms(
                measured.embedding_seconds, measured.embedded
            ),
            "pairs_verified": measured.verified,
            "verifying_seconds": round(measured.verifying_seconds, 1),
            "verifying_ms_per_pair": _per_item_ms(measured.verifying_seconds, measured.verified),
        },
    }
    write_json(run_dir / SUMMARY_FILE, summary)
    return summary


def rows_fingerprints(run_dir: Path) -> dict[str, str]:
    """A short hash of each file of rows in a run directory, byte for byte."""
    return {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()[:16]
        for name in (PAIRS_FILE, QUESTIONS_FILE)
    }


def _per_item_ms(seconds: float, items: int) -> float | None:
    return round(1000 * seconds / items, 1) if items else None


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def load_cache_run(run_dir: Path, pairs: Sequence[CachePair]) -> list[Asked]:
    """Read the scores and candidates a run saved, for working its figures out again.

    The rows name their pairs by id, so the pairs file they were measured on has to be
    given. A run of other pairs is refused, not matched as far as it goes.
    """
    by_id = {pair.pair: pair for pair in pairs}
    own = _rows(run_dir / PAIRS_FILE)
    questions = _rows(run_dir / QUESTIONS_FILE)
    if list(own) != list(by_id) or list(questions) != list(by_id):
        raise DatasetError(f"The run in {run_dir} was not made from the pairs given.")
    return [
        Asked(
            pair=pair,
            own=_read_scores(own[pair_id]),
            candidates=tuple(
                Candidate(str(candidate["question_id"]), _read_scores(candidate))
                for candidate in questions[pair_id]["candidates"]
            ),
            above_floor=int(questions[pair_id]["stored_above_floor"]),
            excluded=questions[pair_id]["excluded"],
        )
        for pair_id, pair in by_id.items()
    ]


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise DatasetError(f"saved run file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["pair"]): row for row in rows}


def _read_scores(row: Mapping[str, Any]) -> Scores:
    return Scores(
        float(row[COSINE]),
        Verification(float(row[ENTAILS_FORWARD]), float(row[ENTAILS_BACKWARD])),
    )
