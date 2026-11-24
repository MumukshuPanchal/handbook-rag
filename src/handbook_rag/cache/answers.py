"""The answer cache: an answer written before, served again without asking the model.

A question asked a second time needs no second request to the answer model, which costs
part of the key's allowance and a wait of seconds. The cache has two tiers, tried in
this order:

- **Exact.** The question is made comparable (:func:`comparable`) and looked up by a
  key. A match is served whatever it holds, a decline too: the same question has the
  same right reply.
- **By meaning.** The stored questions nearest to the new one, by the cosine of their
  embeddings, are candidates. A second model, the verifier, reads each of them with the
  new question in both directions (``cache/verify.py``). The first candidate whose
  weaker direction reaches the threshold is served.

The second tier is the one that can be wrong: a question that only looks like a stored
one will sometimes be served its answer. How often was measured on labelled pairs before
this was built (``evals/cache_pairs.py``, ADR 0020). What is here reduces it and makes
it visible, and does not end it:

- **Some stored replies are never candidates** (:func:`servable_by_meaning`): a decline,
  an answer that declines part of the question, and an answer that cites no passage.
  They are kept, and served to the very same question only.
- **A candidate was written from the same part of the handbook**: a question asked of
  one section never gets an answer written from another, or from all of it.
- **A cached answer says that it is one** (``Answer.cached``): which tier served it,
  the stored question it was written for, and the two scores. A reader who sees the
  other wording can tell when the match was wrong.
- **A caller can ask for a fresh answer.** It is stored under the question's own key,
  and the exact tier, which comes first, serves it from then on.

Not every reply is kept (:func:`storable`), and the cache never stands between a
question and its answer: a store that is down or a verifier that fails is a miss, and
a reply that cannot be stored still goes out. Both are logged.
"""

import datetime
import logging
import math
import queue
import threading
import unicodedata
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Literal

from handbook_rag.cache.store import AnswerStore, Entry
from handbook_rag.cache.verify import Verification, Verifier
from handbook_rag.generate.base import holds_the_decline
from handbook_rag.generate.models import Answer, CacheHit
from handbook_rag.index.embedder import Embedder
from handbook_rag.tracing import stage

EXACT: Final = "exact"
SEMANTIC: Final = "semantic"
Mode = Literal["exact", "semantic"]

ANSWER_CACHE_VERSION = 1
"""Part of the name of every collection of cached answers (``wiring.py``). Raise it when
the code changes what an answer to a question is, or which answers are kept or may be
served by meaning, in a way no setting shows: how a model's reply is read into
statements and citations, for one. Entries written before are then left behind under
their old name. Nothing checks that it is raised: it is a rule for whoever changes that
code."""

DECIDED_BY: Final = "entails_min"
"""The verifier's weaker direction decides, by its name in the measurement
(``evals/cache_pairs.py``). The measurement's rule chose it over the mean of the two
directions, and a test holds this to the saved run the default threshold comes from."""

PENDING_WRITES = 64
"""How many answers may wait to be stored. One more is dropped, and logged: a store too
slow to keep up must not hold answers in memory without end."""

# The verifier is asked once when a service starts, so that no question waits for it.
_WARM_UP = ("How many days of leave do I get?", "How much leave am I given?")
# Question marks, exclamation marks, full stops and the spaces between them, at the end.
_END_MARKS = "?!. "
_ID_NAMESPACE = uuid.UUID("b3a1c2d4-6e5f-4a7b-8c9d-0e1f2a3b4c5d")
_ALL = "all"

logger = logging.getLogger(__name__)


def comparable(question: str) -> str:
    """A question as the exact tier compares it.

    The rule, in this order:

    1. Unicode's composed form (NFC), so that a letter typed as one character and the
       same letter typed with a separate accent are one letter.
    2. Every run of white space becomes one space, and none is left at either end.
    3. Case does not count (``str.casefold``).
    4. Question marks, exclamation marks, full stops and spaces at the end are dropped.

    Nothing else is changed. Punctuation inside the question stays, and so does any
    other sign at its end: "What is C#?" and "What is C?" are two questions. Case can
    carry meaning, as in "US" and "us", and the exact tier does not see it.

    A question with nothing left, such as "?", has no comparable form: the empty text.
    """
    text = " ".join(unicodedata.normalize("NFC", question).split())
    # Folding the case of some letters gives a letter and a separate accent again.
    return unicodedata.normalize("NFC", text.casefold()).rstrip(_END_MARKS)


def scope_of(section: str | None) -> str:
    """Which part of the handbook a question was asked of, as an entry records it.

    An answer is written from the passages a search finds, and a search of one section
    finds others than a search of all of it. So the part is in an entry's key, and the
    tier by meaning looks among entries of the same part only.
    """
    return _ALL if section is None else f"section:{section}"


def entry_id(key: str, scope: str) -> str:
    """The id an answer to this question, asked of this part, is stored under."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{scope}\n{key}"))


def storable(answer: Answer) -> bool:
    """Whether a reply is kept: only a whole reply written by the answer model.

    Not kept:

    - A reply the fallback model wrote. Its answers are not evaluated (ADR 0018), and a
      stored one would be served long after the answer model is back.
    - A reply that did not come to a proper end: cut off at the length limit, withheld
      by the provider, or stopped for another reason.
    - A decline made without asking a model, because the search found nothing. That
      happens while an index is empty or half built, and it must not outlive that.
    - A reply with no text.
    - An answer that came from the cache itself.

    A decline the model wrote is a whole reply, and is kept. A failure raises before
    there is a reply, so nothing of it ever arrives here.
    """
    return (
        answer.cached is None
        and not answer.fallback
        and answer.requests >= 1
        and answer.stop_reason == "end_turn"
        and bool(answer.text.strip())
    )


def servable_by_meaning(answer: Answer) -> bool:
    """Whether a stored reply may be served to a question worded differently.

    - **Not a decline.** The two near-misses the verifier scored highest in the first
      measurement were both a stored decline served to a question the handbook answers
      (``docs/evaluation.md``). The reader is then told the handbook lacks something it
      holds, and nothing in the reply shows it.
    - **Not an answer that declines part of the question.** It is a decline in part, and
      the part it could not find may be exactly what a neighbouring question asks. No
      pair of the measurement tests this: it is a caution, not a finding.
    - **Not an answer that cites no passage.** It rests on nothing a reader can check,
      which is reason enough not to hand it to another question. Not measured either.

    Each of them is still served to the very same question, by the exact tier.
    """
    return answer.answered and bool(answer.sources) and not holds_the_decline(answer.segments)


@dataclass(frozen=True, slots=True)
class Checked:
    """One candidate of the tier by meaning, as it was scored."""

    question: str
    """The stored question. A trace holds it only when content is switched on."""
    similarity: float
    """The cosine of its embedding and the new question's."""
    entails: Verification | None
    """What the verifier said, the stored question first. ``None`` when it failed."""


@dataclass(frozen=True, slots=True)
class Lookup:
    """What looking one question up gave, and what storing its answer needs afterwards."""

    question: str
    section: str | None
    key: str
    """The question in its comparable form. Empty when it has none: the cache then
    neither looks the question up nor stores its answer."""
    read: bool = True
    """False when the caller asked for a fresh answer, so that nothing was looked up."""
    answer: Answer | None = None
    """The stored answer to serve, or ``None`` for a miss."""
    checked: tuple[Checked, ...] = ()
    """The candidates of the tier by meaning, nearest first. Empty when the exact tier
    answered, when nothing was near enough, or when that tier is switched off."""
    vector: tuple[float, ...] | None = None
    """The question's embedding, when the lookup made one. Storing uses it again."""
    failure: str | None = None
    """The kind of error that kept a tier from being asked, if one did."""

    @property
    def tier(self) -> str | None:
        """The tier that served the answer, or ``None`` for a miss."""
        if self.answer is None or self.answer.cached is None:
            return None
        return self.answer.cached.tier


class AnswerCache:
    """Stored answers in two tiers: for the same question, and for one that means the same."""

    def __init__(
        self,
        store: AnswerStore,
        *,
        embedder: Embedder,
        threshold: float,
        verifier: Verifier | None = None,
        candidates: int = 3,
        floor: float = 0.5,
        background: bool = False,
        now: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        """Build a cache over ``store``.

        With a ``verifier`` the cache has both tiers; without one, the exact tier only.
        ``threshold`` is what the verifier's weaker direction has to reach,
        ``candidates`` how many of the nearest stored questions it is asked about, and
        ``floor`` the cosine below which a stored question is no candidate.

        Every entry is stored with the embedding of its question, in both modes, so the
        tier by meaning can be switched on later without emptying the cache.

        With ``background``, answers are stored from a thread of their own, and
        :meth:`store` returns at once: a slow store then delays no answer. Whoever built
        the cache calls :meth:`close` before the process ends.
        """
        if candidates < 1:
            raise ValueError("The tier by meaning needs at least one candidate to check.")
        self._store = store
        self._embedder = embedder
        self._verifier = verifier
        self._threshold = threshold
        self._candidates = candidates
        self._floor = floor
        self._writer = _Writer() if background else None
        self._now = now if now is not None else _utc_now

    @property
    def mode(self) -> Mode:
        return EXACT if self._verifier is None else SEMANTIC

    def warm_up(self) -> None:
        """Load the verifier and run it once, so that no question waits for either.

        A failure leaves this method: a service that cannot load a model it is set to
        use should say so when it starts, not at the first question that needs it.
        """
        if self._verifier is not None:
            self._verifier.score([_WARM_UP])

    def lookup(self, question: str, *, section: str | None = None, read: bool = True) -> Lookup:
        """Look for a stored answer to ``question``. Never raises.

        ``read`` is False when the caller wants a fresh answer: nothing is looked up, and
        what comes back is a miss that :meth:`store` can still keep the new answer for.
        A tier that cannot be asked, because the store is down or the verifier failed,
        is a miss too, and the failure is logged.
        """
        key = comparable(question)
        if not key or not read:
            return Lookup(question, section, key, read=read)
        scope = scope_of(section)
        try:
            entry = self._store.get(entry_id(key, scope))
        except Exception as error:
            return Lookup(question, section, key, failure=_failed("read", error))
        # The id is a hash. The entry is the question's own only if it says so itself.
        if entry is not None and (entry.comparable, entry.scope) == (key, scope):
            hit = CacheHit(tier=EXACT, question=entry.question, stored_at=entry.stored_at)
            return Lookup(question, section, key, answer=_served(entry, question, hit))
        if self._verifier is None:
            return Lookup(question, section, key)
        return self._by_meaning(question, section, key, scope, self._verifier)

    def _by_meaning(
        self, question: str, section: str | None, key: str, scope: str, verifier: Verifier
    ) -> Lookup:
        """The second tier: the nearest stored questions, each checked by the verifier."""
        try:
            with stage("embed.query", model=self._embedder.name):
                vector = tuple(self._embedder.embed_query(question))
            found = self._store.nearest(
                vector, scope=scope, limit=self._candidates, floor=self._floor
            )
        except Exception as error:
            return Lookup(question, section, key, failure=_failed("searched", error))
        # The store filters, and what it returns is checked once more here: an entry that
        # may not be served by meaning must not be, whatever store stands behind this.
        near = [
            (entry, similarity)
            for entry, similarity in found
            if entry.by_meaning and entry.scope == scope and similarity >= self._floor
        ][: self._candidates]
        if not near:
            # Nothing to verify, so the verifier is not asked. A command that loads it
            # only when it is first asked never loads it for such a question.
            return Lookup(question, section, key, vector=vector)
        try:
            with stage("cache.verify", model=verifier.name, pairs=len(near)):
                # The stored question first and the new one second, as they were measured.
                verified = verifier.score([(entry.question, question) for entry, _ in near])
            if len(verified) != len(near):
                raise ValueError(
                    f"The verifier scored {len(verified)} pairs when asked about {len(near)}."
                )
        except Exception as error:
            return Lookup(
                question,
                section,
                key,
                checked=tuple(
                    Checked(entry.question, similarity, None) for entry, similarity in near
                ),
                vector=vector,
                failure=_failed("verified", error),
            )
        checked = tuple(
            Checked(entry.question, similarity, entails)
            for (entry, similarity), entails in zip(near, verified, strict=True)
        )
        for (entry, similarity), entails in zip(near, verified, strict=True):
            # Nearest first, and the first that passes is served. A score that is not a
            # number passes nothing.
            if deciding_score(entails) >= self._threshold:
                hit = CacheHit(
                    tier=SEMANTIC,
                    question=entry.question,
                    stored_at=entry.stored_at,
                    similarity=similarity,
                    score=deciding_score(entails),
                )
                return Lookup(
                    question,
                    section,
                    key,
                    answer=_served(entry, question, hit),
                    checked=checked,
                    vector=vector,
                )
        return Lookup(question, section, key, checked=checked, vector=vector)

    def store(self, lookup: Lookup, answer: Answer) -> None:
        """Keep ``answer`` for the question of ``lookup``, if it is one to keep. Never raises.

        The entry takes the place of any entry of the same question, so a fresh answer
        replaces what the cache held. In the background it is written by a thread of its
        own; otherwise before this returns.
        """
        if not lookup.key or not storable(answer):
            return
        try:
            scope = scope_of(lookup.section)
            entry = Entry(
                entry_id=entry_id(lookup.key, scope),
                question=lookup.question,
                comparable=lookup.key,
                scope=scope,
                segments=answer.segments,
                answered=answer.answered,
                model=answer.model,
                stop_reason=answer.stop_reason,
                stored_at=self._now().isoformat(timespec="seconds"),
                by_meaning=servable_by_meaning(answer),
            )
        except Exception as error:
            # A question that cannot be made into a key, such as one with half of a
            # surrogate pair in it, has been answered all the same.
            _failed("stored", error)
            return
        write = partial(self._write, entry, lookup.vector)
        writer = self._writer
        if writer is None:
            write()
        elif not writer.submit(write):
            logger.warning(
                "an answer was not cached: %d others are still waiting to be stored",
                PENDING_WRITES,
            )

    def _write(self, entry: Entry, vector: Sequence[float] | None) -> None:
        try:
            if vector is None:
                # The exact tier embeds nothing when it looks a question up.
                vector = self._embedder.embed_query(entry.question)
            self._store.put(entry, vector)
        except Exception as error:
            _failed("stored", error)

    def close(self, timeout: float = 5.0) -> bool:
        """Let the answers still waiting be stored, for ``timeout`` seconds at most.

        Returns whether all of them were. A command calls this after it has printed its
        answer, and a service when it shuts down. Without a background thread there is
        nothing to wait for. An answer stored after this is written before
        :meth:`store` returns.
        """
        writer, self._writer = self._writer, None
        return True if writer is None else writer.close(timeout)


def deciding_score(entails: Verification) -> float:
    """The one number the threshold is compared with: the weaker of the two directions.

    It is high only when each question implies the other. :data:`DECIDED_BY` is its name
    in the measurement.

    A direction that is not a number makes the whole score not a number, which reaches
    no threshold. Left to ``min``, such a direction would be passed over without a word,
    and the other one would decide alone.
    """
    if math.isnan(entails.forward) or math.isnan(entails.backward):
        return math.nan
    return entails.smaller


def _served(entry: Entry, question: str, hit: CacheHit) -> Answer:
    """A stored answer as the reply to ``question``: no request, no tokens, no search."""
    return Answer(
        question=question,
        segments=entry.segments,
        answered=entry.answered,
        model=entry.model,
        stop_reason=entry.stop_reason,
        requests=0,
        cached=hit,
    )


def _failed(what: str, error: Exception) -> str:
    """Log that the cache could not be read, searched, verified or stored, and name the
    kind of error. The message goes to the log only: it can name internal addresses."""
    kind = f"{type(error).__module__}.{type(error).__qualname__}"
    logger.warning("the answer cache could not be %s: %s: %s", what, kind, error)
    return kind


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class _Writer:
    """One thread that stores answers, so that no answer waits for the store.

    The thread is started with the first answer to store. It is a daemon: a store that
    hangs cannot keep a command from ending once :meth:`close` has waited its time.
    """

    def __init__(self) -> None:
        self._writes: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._changed = threading.Condition()
        self._waiting = 0
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, write: Callable[[], None]) -> bool:
        """Hand over one write. False when too many are waiting already: it is dropped.

        After :meth:`close` there is no thread to hand it to, and it is written here.
        """
        with self._changed:
            if not self._closed:
                if self._waiting >= PENDING_WRITES:
                    return False
                self._waiting += 1
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="handbook-rag-cache", daemon=True
                    )
                    self._thread.start()
                self._writes.put(write)
                return True
        write()
        return True

    def close(self, timeout: float) -> bool:
        """Wait for the writes handed over so far, then end the thread. Takes no more."""
        with self._changed:
            self._closed = True
            done = self._changed.wait_for(lambda: self._waiting == 0, timeout)
            # After every write that was handed over, so that none is left behind it.
            self._writes.put(None)
            thread = self._thread
        if done and thread is not None:
            # It has nothing left to do but to end.
            thread.join(timeout)
        return done

    def _run(self) -> None:
        while True:
            write = self._writes.get()
            if write is None:
                return
            try:
                write()
            except Exception:  # a write logs its own failure; nothing may end the thread
                logger.exception("an answer was not cached")
            finally:
                with self._changed:
                    self._waiting -= 1
                    self._changed.notify_all()
