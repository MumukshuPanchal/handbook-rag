"""The second check on a stored question: which output is entailment, and both directions.

Every test but the last uses a stand-in for the model, so nothing is downloaded or loaded.
"""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from handbook_rag import local_models, wiring
from handbook_rag.cache.verify import (
    EntailmentVerifier,
    Verification,
    VerifierError,
    entailment_output,
)
from handbook_rag.config import Settings

NLI = {0: "contradiction", 1: "entailment", 2: "neutral"}


class StandInClassifier:
    """A stand-in for a model trained for inference: it scores a pair by a table.

    Its three outputs are in the order its configuration gives, as a real model's are.
    A pair that is not in the table is scored as neither entailed nor contradicted.
    """

    def __init__(
        self,
        labels: dict[Any, str] | None,
        entailed: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.config = SimpleNamespace(id2label=labels)
        self._entailed = entailed or {}
        self.calls: list[dict[str, Any]] = []

    def predict(
        self,
        inputs: Any,
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        apply_softmax: bool | None = False,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "inputs": list(inputs),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "apply_softmax": apply_softmax,
            }
        )
        order = [label.lower() for _, label in sorted((self.config.id2label or {}).items())]
        rows = []
        for pair in inputs:
            entailed = self._entailed.get(pair, 0.0)
            by_label = {"entailment": entailed, "contradiction": 0.0, "neutral": 1 - entailed}
            rows.append([by_label[label] for label in order])
        return rows


def test_the_two_directions_are_kept_apart_and_combined_two_ways() -> None:
    verification = Verification(forward=0.9, backward=0.3)

    assert verification.smaller == 0.3
    assert verification.mean == pytest.approx(0.6)


@pytest.mark.parametrize(
    "labels",
    [
        {0: "contradiction", 1: "entailment", 2: "neutral"},
        {0: "entailment", 1: "neutral", 2: "contradiction"},
        {0: "neutral", 1: "contradiction", 2: "entailment"},
    ],
)
def test_the_entailment_output_is_found_wherever_the_configuration_puts_it(
    labels: dict[Any, str],
) -> None:
    model = StandInClassifier(labels, {("stored", "new"): 0.8, ("new", "stored"): 0.2})

    (verification,) = EntailmentVerifier(model, name="stand-in").score([("stored", "new")])

    # A verifier that assumed a place would read contradiction or neither for two of these.
    assert verification == Verification(forward=0.8, backward=0.2)


def test_the_place_is_read_from_text_keys_and_any_letter_case() -> None:
    # A configuration read from JSON has its places as text.
    assert (
        entailment_output({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "Entailment"}, model="m") == 2
    )
    assert entailment_output(NLI, model="m") == 1
    # A name with spaces around it is the same name.
    assert entailment_output({0: "neutral", 1: " entailment\n"}, model="m") == 1


@pytest.mark.parametrize(
    "labels",
    [
        None,
        {},
        {0: "LABEL_0"},
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        {0: "entailment", 1: "entailment", 2: "neutral"},
        {0: "not_entailment", 1: "neutral"},
    ],
)
def test_a_model_that_does_not_name_one_entailment_output_is_refused(
    labels: dict[Any, str] | None,
) -> None:
    with pytest.raises(VerifierError, match="exactly one output 'entailment'"):
        EntailmentVerifier(StandInClassifier(labels), name="some/model")


def test_the_refusal_names_the_model_and_the_outputs_it_has() -> None:
    with pytest.raises(VerifierError) as refused:
        entailment_output({0: "LABEL_0"}, model="cross-encoder/ms-marco-MiniLM-L6-v2")

    assert "cross-encoder/ms-marco-MiniLM-L6-v2" in str(refused.value)
    assert "LABEL_0" in str(refused.value)


def test_a_model_without_a_configuration_is_refused() -> None:
    model = StandInClassifier(NLI)
    model.config = None  # type: ignore[assignment]

    with pytest.raises(VerifierError):
        EntailmentVerifier(model, name="some/model")


def test_outputs_that_are_not_numbered_are_refused() -> None:
    with pytest.raises(VerifierError, match="numbered"):
        entailment_output({"first": "entailment"}, model="m")


def test_every_pair_is_read_both_ways_in_one_call_as_probabilities() -> None:
    model = StandInClassifier(
        NLI, {("a", "b"): 0.9, ("b", "a"): 0.1, ("c", "d"): 0.4, ("d", "c"): 0.7}
    )
    verifier = EntailmentVerifier(model, name="stand-in", batch_size=4)

    scores = verifier.score([("a", "b"), ("c", "d")])

    assert scores == [Verification(0.9, 0.1), Verification(0.4, 0.7)]
    (call,) = model.calls
    # The pairs as given, then each turned round.
    assert call["inputs"] == [("a", "b"), ("c", "d"), ("b", "a"), ("d", "c")]
    assert call == {**call, "batch_size": 4, "show_progress_bar": False, "apply_softmax": True}


def test_no_pairs_ask_the_model_nothing() -> None:
    model = StandInClassifier(NLI)

    assert EntailmentVerifier(model, name="stand-in").score([]) == []
    assert model.calls == []


def test_a_model_that_scores_another_number_of_pairs_is_an_error() -> None:
    class Short(StandInClassifier):
        def predict(self, inputs: Any, **options: Any) -> list[list[float]]:
            return super().predict(inputs, **options)[:1]

    with pytest.raises(VerifierError, match="scored 1 pairs when asked about 2"):
        EntailmentVerifier(Short(NLI), name="stand-in").score([("a", "b")])


def test_it_says_how_it_reads_the_model_for_the_record_of_a_run() -> None:
    verifier = EntailmentVerifier(
        StandInClassifier({2: "entailment", 0: "neutral", 1: "contradiction"}), name="stand-in"
    )

    assert verifier.name == "stand-in"
    assert verifier.describe() == {
        "verifier_outputs": ["neutral", "contradiction", "entailment"],
        "verifier_entailment_output": 2,
    }


class RecordingCrossEncoder(StandInClassifier):
    """Stands in for the library's class: notes how it is loaded, and fails where told to."""

    loads: list[tuple[str, dict[str, Any]]]
    on_disk: bool

    def __init__(self, model_name: str, **options: Any) -> None:
        type(self).loads.append((model_name, options))
        if options.get("local_files_only") and not type(self).on_disk:
            raise OSError("couldn't find them in the cached files")
        super().__init__(NLI)


@pytest.fixture
def cross_encoder(monkeypatch: pytest.MonkeyPatch) -> type[RecordingCrossEncoder]:
    """The model library replaced by a stand-in, so that loading imports no PyTorch."""

    class Recording(RecordingCrossEncoder):
        loads: list[tuple[str, dict[str, Any]]] = []  # noqa: RUF012
        on_disk = True

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Recording)
    )
    monkeypatch.setattr(local_models, "_no_progress_bars", nullcontext)
    return Recording


def test_a_model_on_disk_is_loaded_without_asking_the_hub(
    cross_encoder: type[RecordingCrossEncoder],
) -> None:
    verifier = EntailmentVerifier.load("some/nli-model", device="cpu")

    assert verifier.name == "some/nli-model"
    assert cross_encoder.loads == [("some/nli-model", {"local_files_only": True, "device": "cpu"})]


def test_a_model_that_is_not_on_disk_is_fetched_from_the_hub(
    cross_encoder: type[RecordingCrossEncoder],
) -> None:
    cross_encoder.on_disk = False

    EntailmentVerifier.load("some/nli-model")

    assert cross_encoder.loads == [
        ("some/nli-model", {"local_files_only": True, "device": None}),
        ("some/nli-model", {"device": None}),
    ]


def test_the_settings_name_the_verifier_and_it_runs_on_the_cpu(
    cross_encoder: type[RecordingCrossEncoder], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert Settings().cache_verifier_model == "cross-encoder/nli-deberta-v3-small"
    monkeypatch.setenv("HANDBOOK_RAG_CACHE_VERIFIER_MODEL", "another/nli-model")

    verifier = wiring.make_verifier(Settings())

    assert verifier.name == "another/nli-model"
    assert cross_encoder.loads == [
        ("another/nli-model", {"local_files_only": True, "device": "cpu"})
    ]


@pytest.mark.slow
def test_the_real_model_names_its_entailment_output_and_follows_a_plain_inference() -> None:
    verifier = wiring.make_verifier(Settings())

    described = verifier.describe()
    labels = described["verifier_outputs"]
    assert isinstance(labels, list)
    assert sorted(labels) == ["contradiction", "entailment", "neutral"]
    assert labels[int(str(described["verifier_entailment_output"]))] == "entailment"

    implied, unrelated = verifier.score(
        [
            ("A man is eating a slice of pizza.", "A man is eating food."),
            ("A man is eating a slice of pizza.", "The stock market fell on Tuesday."),
        ]
    )
    # Pizza is food, and food need not be pizza: the two directions differ.
    assert implied.forward > 0.9
    assert implied.forward > implied.backward
    assert unrelated.smaller < 0.1
    for verification in (implied, unrelated):
        assert 0.0 <= verification.smaller <= verification.mean <= 1.0
