"""Send real requests carefully: counted, spaced out, tried twice, and stopped in time.

Every stage of the answer evaluation sends requests on a key that allows a fixed number
a day. This module holds the loop they share, so that the rules for spending requests are
written, and tested, once.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from handbook_rag.evals.retrieval import FailedQuestion


class Pacer:
    """Keeps requests a fixed time apart, so a run stays under a limit per minute."""

    def __init__(
        self,
        per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("A pace needs a positive number of requests per minute.")
        self._gap = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Return when the next request may start. The first one starts at once."""
        if self._last is not None:
            remaining = self._last + self._gap - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()

    def rest(self, seconds: float) -> None:
        """Pause after a failed request, to give a busy service time to recover."""
        self._sleep(seconds)


@dataclass(slots=True)
class CarefulRun:
    """What one pass over the pending items did."""

    attempts: int = 0
    """Items that were tried. Each try sends at most one request."""
    failures: list[FailedQuestion] = field(default_factory=list)
    """Items that were tried and failed, after one more try."""
    stopped: str | None = None
    """Why the pass ended before every item was tried, if it did."""


def run_carefully[T](
    pending: Sequence[T],
    attempt: Callable[[T], None],
    *,
    name: Callable[[T], str],
    errors: tuple[type[Exception], ...],
    things: str = "questions",
    wait: Callable[[], None] | None = None,
    on_failure: Callable[[FailedQuestion], None] | None = None,
    rest: Callable[[], None] | None = None,
    max_attempts: int | None = None,
    failures_before_stopping: int = 3,
) -> CarefulRun:
    """Try every pending item, and the ones that fail once more.

    ``attempt`` sends the request for one item and keeps its result. It raises one of
    ``errors`` when the request fails for a reason outside our code, such as a model or a
    store that cannot be reached. Any other exception is a fault in our code, and is
    passed on. ``wait`` is called before each try and returns when the next request may
    start: it is where the pace is kept and the attempt is counted. ``on_failure``
    receives each failure as it happens, including one that a second try then makes good.
    ``rest`` is called after each failure, before anything else is tried: a pause there
    lets a service that is briefly overloaded recover, and costs no request.

    An item that fails is tried once more after all the others. When several items fail
    one after another, the pass stops: the cause is then almost certainly the day's
    allowance, the network or the key, and each further try would only spend a request to
    learn the same thing.
    """
    run = CarefulRun()
    failed: dict[str, FailedQuestion] = {}
    failures_in_a_row = 0
    # The second round gives each item that failed one more try.
    for _ in range(2):
        again: list[T] = []
        for item in pending:
            if max_attempts is not None and run.attempts >= max_attempts:
                run.stopped = f"the limit of {max_attempts} requests for this run was reached"
                break
            if wait is not None:
                wait()
            run.attempts += 1
            try:
                attempt(item)
            except errors as error:
                failed[name(item)] = FailedQuestion(name(item), type(error).__name__, str(error))
                if on_failure is not None:
                    on_failure(failed[name(item)])
                again.append(item)
                failures_in_a_row += 1
                if failures_in_a_row >= failures_before_stopping:
                    run.stopped = (
                        f"{failures_in_a_row} {things} in a row failed. The last error: {error}"
                    )
                    break
                if rest is not None:
                    rest()
                continue
            failures_in_a_row = 0
            failed.pop(name(item), None)
        if run.stopped is not None or not again:
            break
        pending = again
    run.failures = list(failed.values())
    return run
