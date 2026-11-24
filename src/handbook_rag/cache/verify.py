"""A second check on a stored question before its answer is served for a new one.

A semantic cache finds a stored question by comparing two embeddings. Each embedding is
made from one question alone, so two questions that differ in one decisive word, twins
against triplets, land almost on the same spot. A *verifier* reads the stored question
and the new one together, as the reranker reads a question with a passage, and says how
far each implies the other. It is too slow to compare a new question with every stored
one, so it is asked only about the few candidates the embeddings put nearest.

This module holds the check and nothing else. No cache uses it yet: what it is worth was
measured first, by ``handbook-rag eval-cache`` (see ``evals/cache_pairs.py``).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from handbook_rag.local_models import load_local_first, quiet_model_libraries

ENTAILMENT = "entailment"


class VerifierError(ValueError):
    """A model cannot be used as a verifier: it does not say which output is entailment."""


@dataclass(frozen=True, slots=True)
class Verification:
    """How far two questions imply each other: two probabilities between 0 and 1.

    Both directions are kept, because they differ. "Who is eligible for a card?" may
    follow from "How do I get a card?" and not the other way round. Whoever uses the
    check decides how to combine them.
    """

    forward: float
    """That the first question implies the second."""
    backward: float
    """That the second question implies the first."""

    @property
    def smaller(self) -> float:
        """The weaker direction. It is high only when each question implies the other."""
        return min(self.forward, self.backward)

    @property
    def mean(self) -> float:
        """The middle of the two directions, which forgives one weak direction in part."""
        return (self.forward + self.backward) / 2


class Verifier(Protocol):
    """Anything that can say how far two questions imply each other."""

    @property
    def name(self) -> str: ...

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        """One verification for each pair of questions, in the order given.

        Several pairs go in at once because a new question is checked against each of
        its few candidates, and a model reads them as one batch.
        """
        ...


class _PairClassifier(Protocol):
    """What the verifier needs of a model: its configuration, and class scores for pairs."""

    @property
    def config(self) -> Any: ...

    def predict(
        self,
        inputs: Any,
        *,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        apply_softmax: bool | None = ...,
    ) -> Any: ...


def entailment_output(labels: Mapping[Any, Any] | None, *, model: str) -> int:
    """Which of a model's outputs means "entailment", read from its own configuration.

    Models trained for natural language inference do not agree on the order of their
    three outputs. One that is read in the wrong order still gives numbers between 0 and
    1, of contradiction or of neither, and nothing would fail. So the place is never
    assumed, and a model whose configuration names no output "entailment", or names two,
    is refused. The project's reranker is refused the same way: its configuration names
    one output, ``LABEL_0``, which is a score of something else.
    """
    named = _in_order(labels)
    found = [place for place, label in named if label.strip().casefold() == ENTAILMENT]
    if len(found) != 1:
        outputs = ", ".join(label for _, label in named) or "none"
        raise VerifierError(
            f"The model {model} cannot be used as a verifier: its configuration has to name "
            f"exactly one output {ENTAILMENT!r}, and it names these outputs: {outputs}."
        )
    return found[0]


def _in_order(labels: Mapping[Any, Any] | None) -> list[tuple[int, str]]:
    """A model's outputs as (place, name), by place.

    A configuration read from JSON has its places as text ("0", "1"), and one built by
    the library as numbers. Both are taken.
    """
    try:
        return sorted((int(place), str(label)) for place, label in (labels or {}).items())
    except (TypeError, ValueError) as error:
        raise VerifierError(
            f"A model's outputs are numbered, and these are not: {labels!r}"
        ) from error


class EntailmentVerifier:
    """A cross-encoder trained for natural language inference, run on this machine.

    Such a model reads a premise and a hypothesis and scores three classes: the premise
    entails the hypothesis, contradicts it, or neither. Each pair of questions is read
    twice, once with each question as the premise, and the probability of entailment is
    kept for both.

    The model was trained on statements, not on questions. How well entailment between
    two questions tells whether one answer serves both is what has to be measured, not
    something the model promises.
    """

    def __init__(self, model: _PairClassifier, *, name: str, batch_size: int = 16) -> None:
        """Wrap a model that is already loaded. :meth:`load` loads one by its name.

        The two are apart so that the reading of the label order can be tested without
        loading a model.
        """
        config = model.config
        labels = None if config is None else getattr(config, "id2label", None)
        self._entailment = entailment_output(labels, model=name)
        self._outputs = tuple(label for _, label in _in_order(labels))
        self._model = model
        self._name = name
        self._batch_size = batch_size

    @classmethod
    def load(cls, model_name: str, *, batch_size: int = 16, device: str | None = None) -> Self:
        """Load the model from the download cache, and from the Hub only when it is not there."""
        # Imported here because loading PyTorch takes seconds.
        quiet_model_libraries()
        from sentence_transformers import CrossEncoder

        model = load_local_first(CrossEncoder, model_name, device=device)
        return cls(model, name=model_name, batch_size=batch_size)

    @property
    def name(self) -> str:
        return self._name

    def describe(self) -> dict[str, object]:
        """How the model's outputs are read, for recording with a run."""
        return {
            "verifier_outputs": list(self._outputs),
            "verifier_entailment_output": self._entailment,
        }

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Verification]:
        if not pairs:
            return []
        # Every pair in the given order, then every pair turned round.
        both_ways = [*pairs, *((second, first) for first, second in pairs)]
        rows = self._model.predict(
            both_ways,
            batch_size=self._batch_size,
            show_progress_bar=False,
            # The probabilities of the three classes, which sum to 1 for each pair.
            apply_softmax=True,
        )
        if len(rows) != len(both_ways):
            raise VerifierError(
                f"The model {self._name} scored {len(rows)} pairs when asked about "
                f"{len(both_ways)}."
            )
        entailed = [float(row[self._entailment]) for row in rows]
        return [
            Verification(forward=entailed[place], backward=entailed[len(pairs) + place])
            for place in range(len(pairs))
        ]
