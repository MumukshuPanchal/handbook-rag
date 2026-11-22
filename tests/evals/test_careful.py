"""The loop that spends requests, used directly. The answer run's tests cover it in use."""

import pytest

from handbook_rag.evals.careful import Pacer, run_carefully


class Busy(Exception):
    """A failure from outside our code, such as a model that cannot be reached."""


def test_every_item_is_tried_and_a_failed_one_once_more_after_the_others() -> None:
    tried: list[str] = []
    seen: list[tuple[str, str, str]] = []

    def attempt(item: str) -> None:
        tried.append(item)
        if item == "b" and tried.count("b") == 1:
            raise Busy("try later")

    run = run_carefully(
        ["a", "b", "c"],
        attempt,
        name=str.upper,
        errors=(Busy,),
        on_failure=lambda f: seen.append((f.question_id, f.error_class, f.message)),
    )

    assert tried == ["a", "b", "c", "b"]
    assert (run.attempts, run.failures, run.stopped) == (4, [], None)
    # A failure is reported under the item's name, and as it happens.
    assert seen == [("B", "Busy", "try later")]


def test_the_stop_names_the_things_that_failed() -> None:
    def attempt(item: str) -> None:
        raise Busy("the day's allowance is used up")

    answers = run_carefully(list("abcdef"), attempt, name=str, errors=(Busy,), things="answers")
    two = run_carefully(
        list("abcdef"), attempt, name=str, errors=(Busy,), failures_before_stopping=2
    )

    assert answers.stopped == (
        "3 answers in a row failed. The last error: the day's allowance is used up"
    )
    assert (answers.attempts, [f.question_id for f in answers.failures]) == (3, ["a", "b", "c"])
    assert two.stopped is not None
    assert two.stopped.startswith("2 questions in a row failed.")
    assert two.attempts == 2


def test_only_the_named_errors_count_as_failures_of_a_request() -> None:
    def attempt(item: str) -> None:
        raise KeyError("a mistake in our own code")

    with pytest.raises(KeyError):
        run_carefully(["a"], attempt, name=str, errors=(Busy,))


def test_nothing_pending_means_nothing_is_tried() -> None:
    waits: list[str] = []

    run = run_carefully(
        [], lambda item: None, name=str, errors=(Busy,), wait=lambda: waits.append("w")
    )

    assert (run.attempts, run.failures, run.stopped, waits) == (0, [], None, [])


def test_the_run_rests_after_a_failure_but_not_before_it_gives_up() -> None:
    events: list[str] = []

    def attempt(item: str) -> None:
        events.append(f"try {item}")
        if item != "d":
            raise Busy("overloaded")

    def rest() -> None:
        events.append("rest")

    recovered = run_carefully(["a", "d"], attempt, name=str, errors=(Busy,), rest=rest)
    assert events == ["try a", "rest", "try d", "try a", "rest"]
    assert [failure.question_id for failure in recovered.failures] == ["a"]

    events.clear()
    stopped = run_carefully(["a", "b", "c", "e"], attempt, name=str, errors=(Busy,), rest=rest)

    # A rest after the first and the second failure. The third ends the run at once.
    assert events == ["try a", "rest", "try b", "rest", "try c"]
    assert stopped.stopped is not None


def test_a_rest_is_a_sleep_of_the_given_length() -> None:
    slept: list[float] = []

    Pacer(10, sleep=slept.append).rest(30)

    assert slept == [30]
