"""A stand-in model for the rewriting step, shared by its tests and by the evaluation's."""

from collections.abc import Callable, Sequence

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import TokenUsage
from handbook_rag.rewrite.models import RewriteReply, Turn

Script = Callable[[Sequence[Turn], str], "str | RewriteReply | BaseException"]


class StandInRewriter:
    """Replies as its script says, and records every request it was sent.

    The script is given the earlier turns and the question, and returns the reply's text,
    a whole reply, or an error to raise. A text comes back as a reply that came to its
    own end, with token counts and a time, as a real one does.
    """

    provider = "stand-in"
    operation = "chat"

    def __init__(self, script: Script, *, model: str = "stand-in-rewriter") -> None:
        self._script = script
        self._model = model
        self.calls: list[tuple[tuple[Turn, ...], str]] = []

    @property
    def model(self) -> str:
        return self._model

    def rewrite(self, turns: Sequence[Turn], question: str) -> RewriteReply:
        self.calls.append((tuple(turns), question))
        outcome = self._script(turns, question)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, RewriteReply):
            return outcome
        return RewriteReply(
            text=outcome,
            model=f"{self._model}-001",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=420, output_tokens=18),
            seconds=0.75,
        )


def replying(text: str) -> StandInRewriter:
    """A stand-in that replies with one text to every request."""
    return StandInRewriter(lambda _turns, _question: text)


def failing(error: GenerationError) -> StandInRewriter:
    """A stand-in whose every request fails with one error."""
    return StandInRewriter(lambda _turns, _question: error)
