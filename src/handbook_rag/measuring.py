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

The rewriting step in front of an answer is refused the same way, at the entry point of
an answer: an evaluation of answers measures the answer to a question as it was typed,
and a rewrite would put another request and another question in front of it.
``eval-follow-ups`` measures the step by itself, without the entry point, and is not
refused.

It works the other way round for answering instructions other than the product's own
(``generate/prompt.py``). They are kept so that an evaluation can ask the old and the new
assistant on the same day. Outside an evaluation, the Gemini and the Claude generator
refuse to be built with any instructions but the current ones, so the service and the
``ask`` command, which build theirs outside an evaluation, cannot answer with them.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_on = 0
_changing = threading.Lock()


class CacheNotAllowed(RuntimeError):
    """An answer cache was built or used while an evaluation was measuring the model."""


class RewriteNotAllowed(RuntimeError):
    """The rewriting step was handed to the entry point while an evaluation was measuring."""


class OldInstructionsNotAllowed(RuntimeError):
    """Answering instructions other than the current ones were asked for outside an evaluation."""


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


def refuse_a_rewrite() -> None:
    """Fail if an evaluation is running and the rewriting step is handed to an answer."""
    if measuring():
        raise RewriteNotAllowed(
            "The rewriting step cannot run in front of an answer while an evaluation runs: "
            "an evaluation of answers measures the answer to the question as it was typed. "
            "`handbook-rag eval-follow-ups` measures the step by itself."
        )


def refuse_old_instructions(version: int, current: int) -> None:
    """Fail if instructions other than the current ones are asked for and no evaluation runs."""
    if version != current and not measuring():
        raise OldInstructionsNotAllowed(
            f"The answering instructions of version {version} are kept for evaluations only. "
            f"Every answer outside an evaluation is written with version {current}."
        )
