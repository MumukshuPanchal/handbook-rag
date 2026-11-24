"""A switch for the time an evaluation runs: no stored answer may stand in for the model.

An evaluation measures what the answer model writes. An answer served from the answer
cache was written on another day, perhaps for another question, and scored it would say
nothing about the model. So every evaluation command runs inside
:func:`measuring_the_model`. While one does, building an answer cache, and handing one
to the entry point of an answer, is an error, wherever in the code it is tried and under
whatever name: the two places every cache has to pass through ask this module first
(``AnswerCache.__init__`` and ``pipeline.answer_question``).

The switch is for the whole process, not for one thread: an evaluation that started a
thread of its own would be covered too. It counts how often it was switched on, so that
one evaluation inside another leaves it on until both have ended.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_on = 0
_changing = threading.Lock()


class CacheNotAllowed(RuntimeError):
    """An answer cache was built or used while an evaluation was measuring the model."""


@contextmanager
def measuring_the_model() -> Iterator[None]:
    """Switch every answer cache off for what runs inside, and back as it was afterwards."""
    global _on
    with _changing:
        _on += 1
    try:
        yield
    finally:
        with _changing:
            _on -= 1


def measuring() -> bool:
    """Whether an evaluation is running in this process."""
    return _on > 0


def refuse_a_cache(doing: str) -> None:
    """Fail if an evaluation is running. ``doing`` says what was tried, for the message."""
    if measuring():
        raise CacheNotAllowed(
            f"An answer cache cannot be {doing} while an evaluation runs: an evaluation "
            "measures what the model writes, and a stored answer says nothing about that."
        )
