"""What the rewriting step is given and what it returns."""

from dataclasses import dataclass, field
from typing import Literal

from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import TokenUsage

MAX_QUESTION_CHARS = 500
"""The longest question the step returns. It is the length ``POST /ask`` accepts for a
question (``MAX_QUERY_CHARS`` in ``api.py``), written out here so that this package does
not load the web service to learn one number. A test holds the two together."""

MAX_EARLIER_TURNS = 6
"""How many earlier turns the model is given at most: the latest ones. What a follow-up
points at is nearly always in the last turn or two, and a conversation of any length
then costs a request of a bounded size. The number is a choice. No conversation longer
than two earlier turns has been measured."""

MAX_REPLY_TOKENS = 1000
"""The upper limit on the tokens of one reply. A question of the longest length allowed
is about 125 tokens, so this leaves room several times over. The default answer model
spends no tokens on reasoning (its replies in the answer evaluation count about one
token for every four characters), and a model that does could run past this limit. A
reply cut off at the limit is not usable, and the evaluation counts it."""

Outcome = Literal["not_asked", "unchanged", "rewritten", "unusable", "failed"]
"""How the step ended for one question.

- ``not_asked``: there were no earlier turns, so no model was asked. The entry point of an
  answer says the same when it was given no rewriting step, as when follow-ups are
  switched off (``pipeline.py``).
- ``unchanged``: the model gave the question back with the words it had.
- ``rewritten``: the model gave another question, and it passed every check.
- ``unusable``: the model replied, and the reply was not one usable question.
- ``failed``: the request failed or ran out of time.

After the last two the step returns the question as it was asked: see
:func:`handbook_rag.rewrite.base.standalone_question` for why.
"""


@dataclass(frozen=True, slots=True)
class Turn:
    """One earlier exchange of a conversation: a question, and perhaps its answer.

    The rewriting step reads the question only. An answer may be carried here, because
    whoever holds a conversation holds its answers too, but it is never sent to the
    model (``prompt.py`` says why, and a test holds the step to that).
    """

    question: str
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class RewriteReply:
    """What the model sent back for one request, before any of it is checked."""

    text: str
    model: str
    """The model that served the reply, which may be named more exactly than the one asked."""
    stop_reason: str
    """Why the model stopped, in the words the answer generators use: ``end_turn``,
    ``max_tokens``, ``refusal`` and so on."""
    usage: TokenUsage = field(default_factory=TokenUsage)
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Rewrite:
    """The question to search for, and how the step came to it."""

    question: str
    """The question to search for. After ``rewritten`` it is the model's. After every
    other outcome it is the question exactly as it was asked."""
    outcome: Outcome
    problem: str | None = None
    """What was wrong with the model's reply, after ``unusable``."""
    failure: GenerationError | None = None
    """Why the request failed, after ``failed``."""
    reply: RewriteReply | None = None
    """What the model sent, whenever it sent something: also a reply that was not usable."""
    requests: int = 0
    """How many requests to a model the step made: 0 or 1."""
    seconds: float = 0.0
    """How long the step made its caller wait: the time of its one request, also when
    that request failed."""

    @property
    def changed(self) -> bool:
        """Whether the question to search for differs from the one that was asked."""
        return self.outcome == "rewritten"

    @property
    def fell_back(self) -> bool:
        """Whether the question is the one asked because the step could not do its work.

        Then nothing says that the question stands alone, and whoever shows the answer
        has to say so: a follow-up searched for as it was typed finds little.
        """
        return self.outcome in ("unusable", "failed")
