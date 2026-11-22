"""The loop that spends requests, used directly. The answer run's tests cover it in use."""

import pytest

from handbook_rag.evals.careful import Pacer, run_carefully


class Busy(Exception):
    """A failure from outside our code, such as a model that cannot be reached."""


def test_every_item_is_tried_and_a_failed_one_once_more_after_the_others() -> None:
    tried: list[str] = []
    kept: list[tuple[str, str]] = []
    seen: list[tuple[str, str, str]] = []

    def send(item: str) -> str:
        tried.append(item)
        if item == "b" and tried.count("b") == 1:
            raise Busy("try later")
        return item.upper()

    run = run_carefully(
        ["a", "b", "c"],
        send,
        lambda item, result: kept.append((item, result)),
        name=str.upper,
        errors=(Busy,),
        on_failure=lambda f: seen.append((f.question_id, f.error_class, f.message)),
    )

    assert tried == ["a", "b", "c", "b"]
    # Each result is handed over with its item, in the order the requests worked.
    assert kept == [("a", "A"), ("c", "C"), ("b", "B")]
    assert (run.attempts, run.failures, run.stopped) == (4, [], None)
    # A failure is reported under the item's name, and as it happens.
    assert seen == [("B", "Busy", "try later")]


def test_the_stop_names_the_things_that_failed() -> None:
    def send(item: str) -> None:
        raise Busy("the day's allowance is used up")

    def keep(item: str, result: None) -> None:
        raise AssertionError("nothing worked, so nothing is kept")

    answers = run_carefully(list("abcdef"), send, keep, name=str, errors=(Busy,), things="answers")
    two = run_carefully(
        list("abcdef"), send, keep, name=str, errors=(Busy,), failures_before_stopping=2
    )

    assert answers.stopped == (
        "3 answers in a row failed. The last error: the day's allowance is used up"
    )
    assert (answers.attempts, [f.question_id for f in answers.failures]) == (3, ["a", "b", "c"])
    assert two.stopped is not None
    assert two.stopped.startswith("2 questions in a row failed.")
    assert two.attempts == 2


def test_only_the_named_errors_from_sending_count_as_failures_of_a_request() -> None:
    def broken(item: str) -> None:
        raise KeyError("a mistake in our own code")

    with pytest.raises(KeyError):
        run_carefully(["a"], broken, lambda item, result: None, name=str, errors=(Busy,))


def test_an_error_while_keeping_a_result_is_never_counted_as_a_failed_request() -> None:
    # The request worked and was paid for. Whatever goes wrong afterwards is our fault,
    # even if it happens to be an error of the kind a failed request raises.
    def keep(item: str, result: str) -> None:
        raise Busy("raised while saving, not while asking")

    with pytest.raises(Busy, match="while saving"):
        run_carefully(["a", "b"], str.upper, keep, name=str, errors=(Busy,))


def test_nothing_pending_means_nothing_is_tried() -> None:
    waits: list[str] = []

    run = run_carefully(
        [],
        str.upper,
        lambda item, result: None,
        name=str,
        errors=(Busy,),
        wait=lambda: waits.append("w"),
    )

    assert (run.attempts, run.failures, run.stopped, waits) == (0, [], None, [])


def test_the_run_rests_between_a_failure_and_the_next_try_and_not_at_the_end() -> None:
    events: list[str] = []

    def send(item: str) -> str:
        events.append(f"try {item}")
        if item != "d":
            raise Busy("overloaded")
        return item

    def rest() -> None:
        events.append("rest")

    def wait() -> None:
        events.append("wait")

    recovered = run_carefully(
        ["a", "d"], send, lambda item, result: None, name=str, errors=(Busy,), rest=rest, wait=wait
    )

    # A rest before the try that follows a failure, taken before the pace is kept. None
    # after the last failure: nothing is left to try, so there is nothing to wait for.
    assert events == ["wait", "try a", "rest", "wait", "try d", "wait", "try a"]
    assert [failure.question_id for failure in recovered.failures] == ["a"]

    events.clear()
    stopped = run_carefully(
        ["a", "b", "c", "e"], send, lambda item, result: None, name=str, errors=(Busy,), rest=rest
    )

    # A rest after the first and the second failure. The third ends the run at once.
    assert events == ["try a", "rest", "try b", "rest", "try c"]
    assert stopped.stopped is not None


def test_a_request_limit_ends_the_run_without_a_rest() -> None:
    events: list[str] = []

    def send(item: str) -> str:
        events.append(f"try {item}")
        raise Busy("overloaded")

    run = run_carefully(
        ["a", "b"],
        send,
        lambda item, result: None,
        name=str,
        errors=(Busy,),
        rest=lambda: events.append("rest"),
        max_attempts=1,
    )

    assert events == ["try a"]
    assert run.stopped == "the limit of 1 requests for this run was reached"


def test_a_rest_is_a_sleep_of_the_given_length() -> None:
    slept: list[float] = []

    Pacer(10, sleep=slept.append).rest(30)

    assert slept == [30]
