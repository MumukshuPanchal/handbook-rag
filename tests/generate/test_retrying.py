"""The wrapper that asks again, waits, and falls back, tested with stand-in models.

No test waits or sends a request. The clock is a number that the stand-in models and the
stand-in sleep move on, so every pause and every deadline is exact.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from handbook_rag.generate.base import FailureReason, GenerationError, Generator
from handbook_rag.generate.models import Answer, Citation, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.generate.retrying import (
    FIRST_PAUSE_SECONDS,
    LONGEST_PAUSE_SECONDS,
    RetryingGenerator,
)
from handbook_rag.ingest.models import Chunk
from tests.support import StandInModel, make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.")
QUESTION = "How long is parental leave?"
# Not zero, so that a deadline counted from zero instead of from the first attempt shows.
START = 1000.0


class Clock:
    """A clock that only moves when it is told to, and a sleep that moves it."""

    def __init__(self) -> None:
        self.now = START
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@dataclass(frozen=True)
class Step:
    """What a stand-in model does for one request: take some time, write, then fail or not."""

    text: str = ""
    error: GenerationError | None = None
    takes: float = 0.0


# A request that writes the whole answer.
ANSWERS = Step(text="Sixteen weeks.")


class Scripted(StandInModel):
    """A stand-in model that follows its script, one step for each request it is sent."""

    def __init__(self, *steps: Step, clock: Clock, model: str = "stand-in-model") -> None:
        self.steps = steps
        self.clock = clock
        self.model = model
        self.asked = 0
        self.started_at: list[float] = []

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        step = self.steps[self.asked]
        self.asked += 1
        self.started_at.append(self.clock.now)
        self.clock.now += step.takes
        if step.text and on_text is not None:
            on_text(step.text)
        if step.error is not None:
            raise step.error
        return Answer(
            question=question,
            segments=(Segment(step.text, (Citation(chunks[0]),)),),
            answered=True,
            model=f"{self.model}-001",
            stop_reason="end_turn",
        )


class Factory:
    """Builds the fallback model when asked, and counts how often it was asked."""

    def __init__(self, model: Generator) -> None:
        self.model = model
        self.built = 0

    def __call__(self) -> Generator:
        self.built += 1
        return self.model


def failure(reason: FailureReason, retry_after: float | None = None) -> GenerationError:
    return GenerationError(f"It failed: {reason}.", reason=reason, retry_after=retry_after)


def wrap(
    model: Generator,
    clock: Clock,
    *,
    attempts: int = 3,
    deadline: float = 30.0,
    fallback: Factory | None = None,
    chance: float = 0.0,
) -> RetryingGenerator:
    return RetryingGenerator(
        model,
        attempts=attempts,
        deadline_seconds=deadline,
        fallback=fallback,
        clock=clock,
        sleep=clock.sleep,
        chance=lambda: chance,
    )


# Asking again.


def test_an_answer_comes_after_one_failed_attempt_and_says_it_took_two_requests() -> None:
    clock = Clock()
    model = Scripted(Step(error=failure("unavailable")), ANSWERS, clock=clock)

    answer = wrap(model, clock).generate(QUESTION, [LEAVE])

    assert answer.text == "Sixteen weeks."
    assert (answer.requests, answer.fallback) == (2, False)
    assert model.asked == 2
    assert clock.slept == [FIRST_PAUSE_SECONDS / 2], "half the first pause, by the chance of 0"


def test_an_answer_at_the_first_attempt_took_one_request_and_no_pause() -> None:
    clock = Clock()
    model = Scripted(ANSWERS, clock=clock)

    answer = wrap(model, clock).generate(QUESTION, [LEAVE])

    assert (answer.requests, answer.fallback, answer.model) == (1, False, "stand-in-model-001")
    assert clock.slept == []


def test_a_model_that_went_silent_is_asked_again_like_one_that_was_unavailable() -> None:
    clock = Clock()
    model = Scripted(Step(error=failure("timeout"), takes=20.0), ANSWERS, clock=clock)

    answer = wrap(model, clock).generate(QUESTION, [LEAVE])

    assert answer.requests == 2
    assert clock.slept == [0.5]


@pytest.mark.parametrize(
    ("chance", "pauses"),
    [
        (0.0, [0.5, 1.0, 2.0, 4.0, 4.0]),
        (0.5, [0.75, 1.5, 3.0, 6.0, 6.0]),
    ],
)
def test_the_pauses_grow_with_each_attempt_until_the_longest(
    chance: float, pauses: list[float]
) -> None:
    clock = Clock()
    model = Scripted(*[Step(error=failure("unavailable"))] * 6, clock=clock)

    with pytest.raises(GenerationError):
        wrap(model, clock, attempts=6, deadline=600.0, chance=chance).generate(QUESTION, [LEAVE])

    # Each pause is between half its length and all of it, and the length doubles from
    # the first pause up to the longest.
    assert clock.slept == pauses
    assert (FIRST_PAUSE_SECONDS, LONGEST_PAUSE_SECONDS) == (1.0, 8.0)
    assert model.asked == 6


def test_a_pause_is_cut_short_by_the_deadline_and_one_last_attempt_starts_at_it() -> None:
    clock = Clock()
    slow = Step(error=failure("unavailable"), takes=4.5)
    model = Scripted(slow, slow, ANSWERS, clock=clock)

    answer = wrap(model, clock, deadline=10.0).generate(QUESTION, [LEAVE])

    # 4.5 s, a pause of 0.5, 4.5 s more: 0.5 s are left of the 1 s the pause would take.
    assert clock.slept == [0.5, 0.5]
    assert model.started_at == [START, START + 5.0, START + 10.0]
    assert answer.requests == 3


@pytest.mark.parametrize("takes", [10.0, 12.0])
def test_no_new_attempt_starts_once_the_deadline_has_come(takes: float) -> None:
    clock = Clock()
    model = Scripted(Step(error=failure("timeout"), takes=takes), ANSWERS, clock=clock)

    with pytest.raises(GenerationError) as raised:
        wrap(model, clock, deadline=10.0).generate(QUESTION, [LEAVE])

    assert raised.value.reason == "timeout"
    assert raised.value.requests == 1
    assert model.asked == 1
    assert clock.slept == []


def test_the_answer_model_is_asked_no_more_often_than_its_attempts() -> None:
    clock = Clock()
    model = Scripted(*[Step(error=failure("unavailable"))] * 3, clock=clock)

    with pytest.raises(GenerationError) as raised:
        wrap(model, clock, attempts=2).generate(QUESTION, [LEAVE])

    assert model.asked == 2
    assert raised.value.requests == 2
    assert clock.slept == [0.5], "no pause after the last attempt"


# Waiting as long as the provider says.


@pytest.mark.parametrize(("takes", "wait"), [(0.0, 5.0), (25.0, 5.0)])
def test_a_rate_limit_is_waited_out_when_the_wait_the_provider_names_ends_by_the_deadline(
    takes: float, wait: float
) -> None:
    clock = Clock()
    limited = Step(error=failure("rate_limit", retry_after=wait), takes=takes)
    model = Scripted(limited, ANSWERS, clock=clock)

    answer = wrap(model, clock, deadline=30.0, chance=0.5).generate(QUESTION, [LEAVE])

    assert clock.slept == [wait], "the provider's wait, not a pause of our own"
    assert answer.requests == 2


@pytest.mark.parametrize(("takes", "wait"), [(0.0, 36.0), (25.0, 5.5)])
def test_a_rate_limit_whose_wait_ends_after_the_deadline_ends_the_attempts_at_once(
    takes: float, wait: float
) -> None:
    clock = Clock()
    limited = failure("rate_limit", retry_after=wait)
    model = Scripted(Step(error=limited, takes=takes), ANSWERS, clock=clock)

    with pytest.raises(GenerationError) as raised:
        wrap(model, clock, deadline=30.0).generate(QUESTION, [LEAVE])

    assert raised.value is limited, "with its wait, which the caller is told"
    assert raised.value.retry_after == wait
    assert model.asked == 1
    assert clock.slept == []


def test_a_rate_limit_without_a_wait_ends_the_attempts_at_once() -> None:
    clock = Clock()
    model = Scripted(Step(error=failure("rate_limit")), ANSWERS, clock=clock)

    with pytest.raises(GenerationError, match="rate_limit"):
        wrap(model, clock).generate(QUESTION, [LEAVE])

    assert model.asked == 1
    assert clock.slept == []


# What is never tried again.


@pytest.mark.parametrize("reason", ["quota", "configuration"])
def test_a_used_up_allowance_or_a_wrong_setting_is_not_tried_again(reason: FailureReason) -> None:
    clock = Clock()
    model = Scripted(Step(error=failure(reason)), ANSWERS, clock=clock)

    with pytest.raises(GenerationError) as raised:
        wrap(model, clock).generate(QUESTION, [LEAVE])

    assert raised.value.reason == reason
    assert model.asked == 1
    assert clock.slept == []


def test_once_text_has_reached_the_caller_a_failure_is_final() -> None:
    clock = Clock()
    broken = Step(text="New parents get", error=failure("unavailable"))
    model = Scripted(broken, ANSWERS, clock=clock)
    backup = Factory(Scripted(ANSWERS, clock=clock, model="fallback-model"))
    received: list[str] = []

    with pytest.raises(GenerationError) as raised:
        wrap(model, clock, fallback=backup).generate(QUESTION, [LEAVE], on_text=received.append)

    assert received == ["New parents get"], "the beginning was sent once"
    assert model.asked == 1
    assert backup.built == 0, "not the fallback either"
    assert (raised.value.requests, raised.value.fallback) == (1, False)


def test_text_that_reached_nobody_does_not_stop_another_attempt() -> None:
    clock = Clock()
    broken = Step(text="New parents get", error=failure("unavailable"))
    model = Scripted(broken, ANSWERS, clock=clock)

    answer = wrap(model, clock).generate(QUESTION, [LEAVE])

    assert answer.requests == 2


def test_text_reaches_the_caller_from_the_attempt_that_wrote_it() -> None:
    clock = Clock()
    model = Scripted(Step(error=failure("unavailable")), ANSWERS, clock=clock)
    received: list[str] = []

    wrap(model, clock).generate(QUESTION, [LEAVE], on_text=received.append)

    assert received == ["Sixteen weeks."]


# The fallback model.


@pytest.mark.parametrize(
    ("first", "requests"),
    [
        ([Step(error=failure("quota"))], 2),
        ([Step(error=failure("rate_limit", retry_after=60.0))], 2),
        ([Step(error=failure("unavailable"))] * 3, 4),
        ([Step(error=failure("timeout"))] * 3, 4),
    ],
)
def test_the_fallback_gets_one_attempt_once_the_answer_model_has_given_up(
    first: list[Step], requests: int
) -> None:
    clock = Clock()
    model = Scripted(*first, clock=clock)
    backup = Factory(Scripted(ANSWERS, clock=clock, model="fallback-model"))
    received: list[str] = []

    answer = wrap(model, clock, fallback=backup).generate(
        QUESTION, [LEAVE], on_text=received.append
    )

    assert (answer.requests, answer.fallback) == (requests, True)
    assert answer.model == "fallback-model-001"
    assert received == ["Sixteen weeks."]
    assert backup.built == 1


def test_a_wrong_setting_of_the_answer_model_is_not_hidden_by_the_fallback() -> None:
    clock = Clock()
    wrong = failure("configuration")
    backup = Factory(Scripted(ANSWERS, clock=clock, model="fallback-model"))

    with pytest.raises(GenerationError) as raised:
        wrap(Scripted(Step(error=wrong), clock=clock), clock, fallback=backup).generate(
            QUESTION, [LEAVE]
        )

    assert raised.value is wrong
    assert backup.built == 0


def test_the_fallback_is_asked_also_when_the_deadline_has_passed() -> None:
    clock = Clock()
    silent = Step(error=failure("timeout"), takes=40.0)
    fallback = Scripted(ANSWERS, clock=clock, model="fallback-model")

    answer = wrap(Scripted(silent, clock=clock), clock, fallback=Factory(fallback)).generate(
        QUESTION, [LEAVE]
    )

    assert answer.fallback is True
    assert fallback.started_at == [START + 40.0]


def test_when_the_fallback_fails_too_its_failure_is_raised_with_the_first_ones() -> None:
    clock = Clock()
    used_up, overloaded = failure("quota"), failure("unavailable")
    backup = Factory(Scripted(Step(error=overloaded), clock=clock, model="fallback-model"))

    with pytest.raises(GenerationError) as raised:
        wrap(Scripted(Step(error=used_up), clock=clock), clock, fallback=backup).generate(
            QUESTION, [LEAVE]
        )

    assert raised.value is overloaded
    assert raised.value.before_fallback is used_up
    assert (raised.value.requests, raised.value.fallback) == (2, True)
    assert used_up.before_fallback is None


def test_a_fallback_that_cannot_be_built_counts_no_request_of_its_own() -> None:
    clock = Clock()
    used_up, missing = failure("quota"), failure("configuration")

    def broken() -> Generator:
        raise missing

    generator = RetryingGenerator(
        Scripted(Step(error=used_up), clock=clock),
        attempts=3,
        deadline_seconds=30.0,
        fallback=broken,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(GenerationError) as raised:
        generator.generate(QUESTION, [LEAVE])

    assert raised.value is missing
    assert raised.value.before_fallback is used_up
    assert (raised.value.requests, raised.value.fallback) == (1, False)


def test_the_fallback_is_built_the_first_time_it_is_needed_and_then_kept() -> None:
    clock = Clock()
    model = Scripted(
        ANSWERS, Step(error=failure("quota")), Step(error=failure("quota")), clock=clock
    )
    backup = Factory(Scripted(ANSWERS, ANSWERS, clock=clock, model="fallback-model"))
    generator = wrap(model, clock, fallback=backup)

    generator.generate(QUESTION, [LEAVE])
    assert backup.built == 0, "not needed yet"
    generator.generate(QUESTION, [LEAVE])
    generator.generate(QUESTION, [LEAVE])
    assert backup.built == 1


# Declining, the names, and what the wrapper is made with.


def test_with_nothing_to_answer_from_no_model_is_asked() -> None:
    clock = Clock()
    model = Scripted(ANSWERS, clock=clock)
    backup = Factory(Scripted(ANSWERS, clock=clock, model="fallback-model"))
    received: list[str] = []

    answer = wrap(model, clock, fallback=backup).generate(QUESTION, [], on_text=received.append)

    assert (answer.text, answer.stop_reason, answer.answered) == (
        NOT_FOUND_MESSAGE,
        "no_context",
        False,
    )
    assert (answer.requests, answer.fallback, answer.model) == (0, False, "stand-in-model")
    assert received == [NOT_FOUND_MESSAGE]
    assert (model.asked, backup.built) == (0, 0)


def test_the_wrapper_goes_by_the_names_of_the_answer_model() -> None:
    generator = wrap(Scripted(clock=Clock(), model="gemini-9"), Clock())

    assert (generator.provider, generator.operation, generator.model) == (
        "stand-in",
        "chat",
        "gemini-9",
    )


@pytest.mark.parametrize(
    ("attempts", "deadline", "message"),
    [
        (0, 30.0, "at least once"),
        (-1, 30.0, "at least once"),
        (1, 0.0, "positive number of seconds"),
        (1, -5.0, "positive number of seconds"),
        (1, float("nan"), "positive number of seconds"),
    ],
)
def test_a_wrapper_needs_one_attempt_and_a_deadline_in_the_future(
    attempts: int, deadline: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RetryingGenerator(Scripted(clock=Clock()), attempts=attempts, deadline_seconds=deadline)


def test_the_clock_the_sleep_and_the_chance_are_looked_up_when_a_wrapper_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("time.monotonic", clock)
    monkeypatch.setattr("time.sleep", clock.sleep)
    monkeypatch.setattr("random.random", lambda: 0.5)
    model = Scripted(
        Step(error=failure("unavailable")),
        Step(error=failure("unavailable"), takes=29.0),
        ANSWERS,
        clock=clock,
    )

    # Made after the three were replaced, and given none of them.
    RetryingGenerator(model, attempts=3, deadline_seconds=30.0).generate(QUESTION, [LEAVE])

    # The first pause is 0.75 s by the chance of 0.5. The second would be 1.5 s, and is
    # cut to the 0.25 s that the clock says are left of the deadline.
    assert clock.slept == [0.75, 0.25]


# The trace: one span for each request, and none for anything else.


def test_each_request_has_a_span_with_its_number_and_why_it_failed(
    spans: InMemorySpanExporter,
) -> None:
    clock = Clock()
    model = Scripted(
        Step(error=failure("unavailable")),
        Step(error=failure("timeout")),
        Step(error=failure("quota")),
        clock=clock,
    )
    backup = Factory(Scripted(ANSWERS, clock=clock, model="fallback-model"))

    wrap(model, clock, fallback=backup).generate(QUESTION, [LEAVE])

    recorded = sorted(spans.get_finished_spans(), key=lambda span: span.start_time or 0)
    assert [span.name for span in recorded] == [
        "chat stand-in-model",
        "chat stand-in-model",
        "chat stand-in-model",
        "chat fallback-model",
    ]
    facts = [dict(span.attributes or {}) for span in recorded]
    assert [fact["handbook_rag.attempt"] for fact in facts] == [1, 2, 3, 4]
    assert [fact["handbook_rag.fallback"] for fact in facts] == [False, False, False, True]
    assert [fact.get("handbook_rag.failure") for fact in facts] == [
        "unavailable",
        "timeout",
        "quota",
        None,
    ]
    assert [span.status.status_code for span in recorded] == [
        StatusCode.ERROR,
        StatusCode.ERROR,
        StatusCode.ERROR,
        StatusCode.UNSET,
    ]
    assert facts[3]["gen_ai.request.model"] == "fallback-model"
    assert facts[3]["gen_ai.response.model"] == "fallback-model-001"
    assert "gen_ai.response.model" not in facts[0], "a failed request had no reply"


def test_no_span_is_opened_for_a_request_that_was_never_sent(
    spans: InMemorySpanExporter,
) -> None:
    clock = Clock()
    limited = Step(error=failure("rate_limit", retry_after=60.0))
    model = Scripted(limited, clock=clock)

    with pytest.raises(GenerationError):
        wrap(model, clock).generate(QUESTION, [LEAVE])
    wrap(Scripted(clock=clock), clock).generate(QUESTION, [])

    assert [span.name for span in spans.get_finished_spans()] == ["chat stand-in-model"]
