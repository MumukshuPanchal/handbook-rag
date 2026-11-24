"""The answer cache: an answer written before, served again without asking the model.

A question asked a second time needs no second request to the answer model, which costs
part of the key's allowance and a wait of seconds. The cache has two tiers, tried in
this order:

- **Exact.** The question is made comparable (``rules.comparable``) and looked up by a
  key. A match is served whatever it holds, a decline too: the same question has the
  same right reply.
- **By meaning.** The stored questions nearest to the new one, by the cosine of their
  embeddings, are candidates. A second model, the verifier, reads each of them with the
  new question in both directions (``cache/verify.py``). The first candidate whose
  weaker direction reaches the threshold is served.

The second tier is the one that can be wrong: a question that only looks like a stored
one will sometimes be served its answer. How often was measured on labelled pairs
(``evals/cache_pairs.py``, ADR 0020). What is here reduces it and makes it visible, and
does not end it:

- **Some stored replies are never candidates** (``rules.kept_from_the_second_tier``):
  a decline, an answer in which one of two guards finds that something is missing, and
  an answer that cites no passage. They are kept, and served to the very same question
  only. The guards know a decline in part by its words, so one in other words passes.
- **A candidate was written from the same part of the handbook**: a question asked of
  one section never gets an answer written from another, or from all of it.
- **A cached answer says that it is one** (``Answer.cached``): which tier served it,
  the stored question it was written for, and the two scores. A reader who sees the
  other wording can tell when the match was wrong.
- **A caller can ask for a fresh answer.** If it is one the cache keeps, it is stored
  under the question's own key, and the exact tier, which comes first, serves it from
  then on.

Not every reply is kept (``rules.storable``). And the cache stands before an answer for
a bounded time only: a lookup that has not ended after :data:`LOOKUP_SECONDS` is given
up, a store that is down or a verifier that fails is a miss, and a reply that cannot be
stored still goes out. Each of these is logged. One wait is not bounded: a command
loads the verifier the first time a stored question is near its own, as it loads its
other models when it starts.

The rules themselves are in ``cache/rules.py``, which needs no store and no model. An
evaluation may apply them, and may not build what is in this module (``measuring.py``).
"""

import contextvars
import datetime
import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Literal

from handbook_rag.cache.rules import (
    ANSWER_CACHE_VERSION,
    DECIDED_BY,
    comparable,
    deciding_score,
    entry_id,
    scope_of,
    servable_by_meaning,
    storable,
)
from handbook_rag.cache.store import AnswerStore, Entry
from handbook_rag.cache.verify import LazyVerifier, Verification, Verifier
from handbook_rag.generate.models import Answer, CacheHit
from handbook_rag.index.embedder import Embedder
from handbook_rag.measuring import refuse_a_cache
from handbook_rag.tracing import stage

__all__ = [
    "ANSWER_CACHE_VERSION",
    "DECIDED_BY",
    "LOOKUP_SECONDS",
    "PENDING_WRITES",
    "AnswerCache",
    "Checked",
    "Lookup",
    "comparable",
    "deciding_score",
    "entry_id",
    "scope_of",
    "servable_by_meaning",
    "storable",
]

EXACT: Final = "exact"
SEMANTIC: Final = "semantic"
Mode = Literal["exact", "semantic"]

LOOKUP_SECONDS = 2.0
"""How long a question waits for the cache at most. After that the lookup is given up
and the question is answered as if there were no cache.

Without it, what stands before every answer is the store's own time limit, 10 seconds in
the service and 60 on the command line, and the wait for the verifier, which reads for
one question at a time. In the check with the real service the slowest lookup took 1.2
seconds, on a machine that had paged the models out, and the others under 0.3. Loading
the verifier is not counted: a command loads it once, when the first question needs it,
as it loads its other models when it starts."""

TIMED_OUT: Final = "timeout"
"""What a lookup records as its failure when it was given up for taking too long."""

PENDING_WRITES = 64
"""How many answers may wait to be stored. One more is dropped, and logged: a store too
slow to keep up must not hold answers in memory without end."""

# The verifier is asked once when a service starts, so that no question waits for it.
_WARM_UP = ("How many days of leave do I get?", "How much leave am I given?")

logger = logging.getLogger(__name__)


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
    """What kept a tier from being asked, if anything did: the kind of error, or
    :data:`TIMED_OUT`."""

    @property
    def tier(self) -> str | None:
        """The tier that served the answer, or ``None`` for a miss."""
        if self.answer is None or self.answer.cached is None:
            return None
        return self.answer.cached.tier


@dataclass(frozen=True, slots=True)
class _Near:
    """The stored questions near a new one, found and not yet read by the verifier."""

    entries: tuple[tuple[Entry, float], ...]
    vector: tuple[float, ...]


class _TookTooLong(Exception):
    """A step of a lookup had not ended when the time for the whole lookup was up."""


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
        floor: float = 0.7,
        background: bool = False,
        lookup_seconds: float = LOOKUP_SECONDS,
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
        the cache calls :meth:`close` before the process ends. ``lookup_seconds`` is how
        long :meth:`lookup` waits at most.

        While an evaluation runs, no cache can be built (``measuring.py``).
        """
        refuse_a_cache("built")
        if candidates < 1:
            raise ValueError("The tier by meaning needs at least one candidate to check.")
        if not lookup_seconds > 0:
            raise ValueError("A lookup needs some time: its limit has to be above 0 seconds.")
        self._store = store
        self._embedder = embedder
        self._verifier = verifier
        self._threshold = threshold
        self._candidates = candidates
        self._floor = floor
        self._lookup_seconds = lookup_seconds
        self._writer = _Writer() if background else None
        self._now = now if now is not None else _utc_now

    @property
    def mode(self) -> Mode:
        return EXACT if self._verifier is None else SEMANTIC

    def prepare(self) -> bool:
        """Make the place the answers are kept, before any question is looked up.

        A service does this when it starts and a command before it asks, so that no
        lookup meets a collection that the first write is still making. Returns whether
        it is there. A store that cannot be reached is logged and no more: the cache
        then makes its collection with the first answer it stores.
        """
        try:
            self._store.ensure(self._embedder.dimension)
        except Exception as error:
            _failed("prepared", error)
            return False
        return True

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
        is a miss too, and so is a lookup that has not ended after the cache's time
        limit. Each is logged. The limit covers the store and the verifier's reading.
        It does not cover loading the verifier, which a command does here the first
        time it needs it, and which is a download the first time ever.

        The work is done in a thread of its own, which is what lets the caller stop
        waiting. A lookup that was given up goes on there to its end, and what it finds
        is thrown away.
        """
        key = comparable(question)
        if not key or not read:
            return Lookup(question, section, key, read=read)
        deadline = time.monotonic() + self._lookup_seconds
        try:
            found = _within(deadline, self._find, question, section, key)
            if isinstance(found, _Near):
                loading = time.monotonic()
                failure = self._load_the_verifier()
                if failure is not None:
                    return Lookup(
                        question,
                        section,
                        key,
                        checked=_unverified(found),
                        vector=found.vector,
                        failure=failure,
                    )
                # Loading a model is not part of looking a question up.
                deadline += time.monotonic() - loading
                found = _within(deadline, self._verify, question, section, key, found)
        except _TookTooLong:
            logger.warning(
                "the answer cache was given up after %.1f s, and the question is answered "
                "without it",
                self._lookup_seconds,
            )
            return Lookup(question, section, key, failure=TIMED_OUT)
        return found

    def _find(self, question: str, section: str | None, key: str) -> Lookup | _Near:
        """The exact tier, and for the second tier the stored questions near enough."""
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
        near = tuple(
            (entry, similarity)
            for entry, similarity in found
            if entry.by_meaning and entry.scope == scope and similarity >= self._floor
        )[: self._candidates]
        if not near:
            # Nothing to verify, so the verifier is not asked. A command that loads it
            # only when it is first asked never loads it for such a question.
            return Lookup(question, section, key, vector=vector)
        return _Near(near, vector)

    def _load_the_verifier(self) -> str | None:
        """Load the verifier if this process has not yet. The kind of failure, or ``None``.

        A service has loaded it when it started. A command loads it here, the first time
        a stored question is near the one asked, and waits for it as it waits for its
        other models: the first time ever, that is a download.
        """
        verifier = self._verifier
        if not isinstance(verifier, LazyVerifier) or verifier.loaded:
            return None
        try:
            with stage("cache.load_verifier", model=verifier.name):
                verifier.load()
        except Exception as error:
            return _failed("verified", error)
        return None

    def _verify(self, question: str, section: str | None, key: str, near: _Near) -> Lookup:
        """The second tier: each candidate read by the verifier, nearest first."""
        verifier = self._verifier
        assert verifier is not None  # candidates are looked for only when there is one
        try:
            with stage("cache.verify", model=verifier.name, pairs=len(near.entries)):
                # The stored question first and the new one second, as they were measured.
                verified = verifier.score([(entry.question, question) for entry, _ in near.entries])
            if len(verified) != len(near.entries):
                raise ValueError(
                    f"The verifier scored {len(verified)} pairs when asked about "
                    f"{len(near.entries)}."
                )
        except Exception as error:
            return Lookup(
                question,
                section,
                key,
                checked=_unverified(near),
                vector=near.vector,
                failure=_failed("verified", error),
            )
        checked = tuple(
            Checked(entry.question, similarity, entails)
            for (entry, similarity), entails in zip(near.entries, verified, strict=True)
        )
        for (entry, similarity), entails in zip(near.entries, verified, strict=True):
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
                    vector=near.vector,
                )
        return Lookup(question, section, key, checked=checked, vector=near.vector)

    def store(self, lookup: Lookup, answer: Answer) -> None:
        """Keep ``answer`` for the question of ``lookup``, if it is one to keep. Never raises.

        The entry takes the place of any entry of the same question, so a fresh answer
        that is kept replaces what the cache held. One that is not kept, a reply cut off
        at the length limit for one, leaves the older entry where it is. In the
        background the entry is written by a thread of its own; otherwise before this
        returns.
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

        Returns whether all of them were, and logs how many were not. A command calls
        this after it has printed its answer, and a service when it shuts down. Without
        a background thread there is nothing to wait for. An answer stored after this is
        written before :meth:`store` returns.
        """
        writer, self._writer = self._writer, None
        if writer is None:
            return True
        done = writer.close(timeout)
        if not done:
            logger.warning(
                "%d answers were not stored in the answer cache: the store had not taken "
                "them %.1f s after the cache was closed",
                writer.waiting,
                timeout,
            )
        return done


def _within[T](deadline: float, step: Callable[..., T], *arguments: object) -> T:
    """Run one step of a lookup in a thread of its own, and stop waiting at ``deadline``.

    The thread starts with a copy of the caller's context, so the spans the step opens
    belong to the caller's trace. It is a daemon: a step that hangs on a store cannot
    keep a command from ending.
    """
    left = deadline - time.monotonic()
    if left <= 0:
        raise _TookTooLong
    results: list[T] = []
    ended = threading.Event()
    context = contextvars.copy_context()

    def run() -> None:
        try:
            results.append(context.run(step, *arguments))
        finally:
            ended.set()

    threading.Thread(target=run, name="handbook-rag-cache-lookup", daemon=True).start()
    if not ended.wait(left) or not results:
        # Not in time, or ended without a result, which no step does by itself.
        raise _TookTooLong
    return results[0]


def _unverified(near: _Near) -> tuple[Checked, ...]:
    """The candidates as they are recorded when the verifier could not be asked."""
    return tuple(Checked(entry.question, similarity, None) for entry, similarity in near.entries)


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
    """Log that the cache could not be prepared, read, searched, verified or stored, and
    name the kind of error. The message goes to the log only: it can name internal
    addresses."""
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

    @property
    def waiting(self) -> int:
        """How many writes were handed over and are not done."""
        with self._changed:
            return self._waiting

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
