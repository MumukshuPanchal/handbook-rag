"""Stand-ins shared by the tests of the answer cache and of its measurement.

They are here and not in ``tests/support.py``: that module is also imported by a child
process that one test starts, which should not need the cache package.
"""

import re
from collections.abc import Mapping, Sequence

from qdrant_client import QdrantClient

from handbook_rag.cache.store import QdrantAnswerStore
from handbook_rag.cache.verify import Verification


class FakeVerifier:
    """A stand-in verifier: one question implies another by the share of the other's words
    that it has."""

    name = "fake-verifier"

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        self.calls.append(list(pairs))
        return [
            Verification(self._implies(first, second), self._implies(second, first))
            for first, second in pairs
        ]

    def describe(self) -> dict[str, object]:
        return {
            "verifier_outputs": ["contradiction", "entailment", "neutral"],
            "verifier_entailment_output": 1,
        }

    @staticmethod
    def _implies(premise: str, hypothesis: str) -> float:
        said = set(re.findall(r"[a-z0-9]+", premise.lower()))
        asked = set(re.findall(r"[a-z0-9]+", hypothesis.lower()))
        return len(said & asked) / len(asked) if asked else 0.0


class TableVerifier:
    """A stand-in verifier that scores a pair by a table: (stored question, new question)
    to the two directions. A pair that is not in the table implies nothing either way."""

    name = "table-verifier"

    def __init__(
        self,
        scores: Mapping[tuple[str, str], tuple[float, float]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.scores = dict(scores or {})
        self.calls: list[list[tuple[str, str]]] = []
        self._error = error

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        self.calls.append(list(pairs))
        if self._error is not None:
            raise self._error
        return [Verification(*self.scores.get(pair, (0.0, 0.0))) for pair in pairs]


def memory_answers(collection: str = "answers-test") -> QdrantAnswerStore:
    """A place for cached answers in this process's memory, which needs no server."""
    return QdrantAnswerStore(QdrantClient(":memory:"), collection)
