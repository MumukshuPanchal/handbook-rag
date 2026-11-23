"""Ask the model again when that may help, then a second model, within a deadline.

The SDKs can send a failed request again by themselves, but they do it unseen: the
caller waits without a sign of life, nobody counts the requests, and no trace shows
them. So for answers their own repeats are switched off (see ``wiring.py``), and the
wrapper here does the repeating instead. Every request it sends has a span of its own,
is counted on the answer, and starts before a deadline.

What happens after a failure is decided by the error's reason (ADR 0018):

- ``unavailable`` and ``timeout``: the same model is asked again, after a pause that
  grows with each attempt and has some randomness in it, so that many callers who failed
  together do not all come back at the same moment.
- ``rate_limit``: asked again after the wait the provider names, if that wait ends
  before the deadline. Otherwise the answer model gives up at once.
- ``quota``: never again on the same model today.
- ``configuration``: never again, and no fallback either. The person who runs the
  service has to fix it, and a fallback would hide it.

When the answer model gives up for any reason but the last, the fallback model, if one
is set, gets one attempt. Once any text has been passed on to the caller, a failure is
final: asking again would send the beginning of the answer a second time.
"""

import dataclasses
import random
import threading
import time
from collections.abc import Callable, Sequence

from handbook_rag.generate.base import GenerationError, Generator, decline_without_context
from handbook_rag.generate.models import Answer
from handbook_rag.ingest.models import Chunk
from handbook_rag.tracing import model_call, record_failure, record_reply

# The longest pause after the first failed attempt. Each later pause may be twice as long
# as the one before, up to the longest of all. A pause is between half of that length and
# all of it, chosen at random.
FIRST_PAUSE_SECONDS = 1.0
LONGEST_PAUSE_SECONDS = 8.0

# The failures after which the same model is asked again, after a pause of our choosing.
_PAUSE_AND_ASK_AGAIN = frozenset({"unavailable", "timeout"})


def traced_request(
    generator: Generator,
    question: str,
    chunks: Sequence[Chunk],
    *,
    on_text: Callable[[str], None] | None = None,
    attempt: int = 1,
    fallback: bool = False,
) -> Answer:
    """Send one request through ``generator``, in a span of its own.

    ``attempt`` is the request's place among those of one answer, counted from 1, and
    ``fallback`` says whether it goes to the fallback model. Both are recorded on the
    span. The answer that comes back carries them, and so does an error that leaves:
    the attempt's number is then how many requests the answer has made. The span of a
    failed request is the only record of it, so it also notes the failure's reason.
    """
    with model_call(
        provider=generator.provider,
        operation=generator.operation,
        model=generator.model,
        passages=len(chunks),
        attempt=attempt,
        fallback=fallback,
    ) as calling:
        try:
            answer = generator.generate(question, chunks, on_text=on_text)
        except GenerationError as failure:
            record_failure(calling, failure.reason)
            failure.requests = attempt
            failure.fallback = fallback
            raise
        record_reply(calling, answer)
    return dataclasses.replace(answer, requests=attempt, fallback=fallback)


class RetryingGenerator:
    """An answer generator that asks its model again when that may help, then a fallback.

    It is a generator like the ones it wraps, so the rest of the application does not
    need to know it is there. Unlike them, it opens the span of each request itself,
    because only it knows how many it sends.
    """

    def __init__(
        self,
        answer_model: Generator,
        *,
        attempts: int,
        deadline_seconds: float,
        fallback: Callable[[], Generator] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        chance: Callable[[], float] | None = None,
    ) -> None:
        """Wrap ``answer_model``.

        ``attempts`` is how often the answer model may be asked for one answer, the first
        time included. ``deadline_seconds`` is how long after the first attempt began a
        new attempt may still start. ``fallback`` builds the fallback model's generator.
        It is called the first time the fallback is needed, and only then. ``chance``
        returns a number from 0 up to 1, which places a pause within its range.
        """
        if attempts < 1:
            raise ValueError("A request has to be sent at least once.")
        if not deadline_seconds > 0:
            raise ValueError("The deadline has to be a positive number of seconds.")
        self._answer_model = answer_model
        self._attempts = attempts
        self._deadline_seconds = deadline_seconds
        self._make_fallback = fallback
        self._fallback: Generator | None = None
        # Several answers are written at once, each in its own thread, and the fallback
        # is built once for all of them.
        self._building = threading.Lock()
        # Looked up here, when a wrapper is made, and not in the signature above. A
        # default in a signature is fixed when the module is first imported, so a test
        # that replaces `time.sleep` afterwards would be ignored, and would really wait.
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._chance = chance if chance is not None else random.random

    @property
    def provider(self) -> str:
        return self._answer_model.provider

    @property
    def operation(self) -> str:
        return self._answer_model.operation

    @property
    def model(self) -> str:
        return self._answer_model.model

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Write an answer grounded in ``chunks``. ``on_text`` receives text as it arrives.

        The answer says how many requests it took and whether the fallback model wrote
        it. When no answer can be written, the last failure is raised, with the same two
        facts on it.
        """
        if not chunks:
            # Nothing to answer from: the assistant declines without a request, so there
            # is nothing to repeat and no span to open.
            return decline_without_context(question, self.model, on_text)

        passed_on = False

        def pass_on(text: str) -> None:
            nonlocal passed_on
            passed_on = True
            if on_text is not None:
                on_text(text)

        listener = pass_on if on_text is not None else None
        deadline = self._clock() + self._deadline_seconds
        attempt = 1
        while True:
            try:
                return traced_request(
                    self._answer_model, question, chunks, on_text=listener, attempt=attempt
                )
            except GenerationError as failure:
                if passed_on:
                    # Part of the answer has reached the caller. Asking again, here or
                    # of the fallback, would send its beginning a second time.
                    raise
                pause = self._pause_before_next(failure, attempt=attempt, deadline=deadline)
                if pause is None:
                    return self._fall_back(failure, question, chunks, listener)
                self._sleep(pause)
            attempt += 1

    def _pause_before_next(
        self, failure: GenerationError, *, attempt: int, deadline: float
    ) -> float | None:
        """How long to wait before asking the answer model again, or ``None`` not to."""
        if attempt >= self._attempts:
            return None
        left = deadline - self._clock()
        if left <= 0:
            return None
        if failure.reason in _PAUSE_AND_ASK_AGAIN:
            # Our own pause, so the deadline may cut it short. The decision to ask again
            # is made here, before the pause, so an attempt still follows a pause that
            # was cut short: it starts at the deadline.
            return min(self._pause(after=attempt), left)
        if failure.reason == "rate_limit" and failure.retry_after is not None:
            # The provider's wait cannot be cut short: a request sent sooner would be
            # refused again. So it has to end before the deadline, or it is not waited.
            return failure.retry_after if failure.retry_after <= left else None
        return None

    def _pause(self, *, after: int) -> float:
        """The pause after the attempt numbered ``after``, with some randomness in it."""
        longest = min(LONGEST_PAUSE_SECONDS, FIRST_PAUSE_SECONDS * 2.0 ** (after - 1))
        return longest * (1 + self._chance()) / 2

    def _fall_back(
        self,
        failure: GenerationError,
        question: str,
        chunks: Sequence[Chunk],
        on_text: Callable[[str], None] | None,
    ) -> Answer:
        """Give the fallback model its one attempt, if the answer model's failure allows it.

        Its request has the same time limit as any other, also when the deadline has
        passed: the deadline is for the answer model's attempts. If it fails too, its
        failure is the one the caller sees, because it is the latest news about whether
        an answer can be had, and the answer model's failure goes with it.
        """
        if failure.reason == "configuration" or self._make_fallback is None:
            raise failure
        try:
            fallback = self._fallback_model(self._make_fallback)
            return traced_request(
                fallback,
                question,
                chunks,
                on_text=on_text,
                attempt=failure.requests + 1,
                fallback=True,
            )
        except GenerationError as second:
            # A fallback that could not even be built has sent no request of its own.
            second.requests = max(second.requests, failure.requests)
            second.before_fallback = failure
            raise

    def _fallback_model(self, make: Callable[[], Generator]) -> Generator:
        with self._building:
            if self._fallback is None:
                self._fallback = make()
            return self._fallback
