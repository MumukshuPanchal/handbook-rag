"""Stand-ins shared by the tests of the cache measurement.

They are here and not in ``tests/support.py``: that module is also imported by a child
process that one test starts, which should not need the cache package.
"""

import re
from collections.abc import Sequence

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
