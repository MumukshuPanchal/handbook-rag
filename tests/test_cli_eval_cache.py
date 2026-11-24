"""The ``eval-cache`` command, with stand-ins for the two models.

It is in a file of its own, beside ``test_cli.py``, because it needs none of that file's
fixtures: no snapshot, no index and no answer model.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from handbook_rag import cli
from handbook_rag.cache.verify import VerifierError
from handbook_rag.cli import app
from handbook_rag.config import Settings
from handbook_rag.evals import cache_pairs
from tests.cache.support import FakeVerifier
from tests.evals.test_cache_pairs import a_small_world
from tests.support import FakeEmbedder

runner = CliRunner()
# The colour codes a terminal gets. In CI the usage errors come with them, and they can
# stand in the middle of an option's name.
COLOUR_CODES = re.compile(r"\x1b\[[0-9;]*m")


class StoppedClock:
    """The datetime module as the command sees it, with a clock that stands on one day."""

    class date(datetime.date):
        @classmethod
        def today(cls) -> "StoppedClock.date":
            return cls(2031, 2, 3)

    def __getattr__(self, name: str) -> Any:
        return getattr(datetime, name)


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for the embedding model and the verifier, and a guard on everything else.

    The command must not build an answer model, a judge or a vector store. Each of those
    fails the test if the command asks for it.
    """
    used: dict[str, Any] = {"embedder": FakeEmbedder(), "verifier": FakeVerifier()}

    def embedder(settings: Settings) -> FakeEmbedder:
        used["embedding_settings"] = settings
        return used["embedder"]  # type: ignore[no-any-return]

    def never(*_arguments: object, **_options: object) -> None:
        raise AssertionError("eval-cache must not ask for an answer model, a judge or a store")

    monkeypatch.setattr("handbook_rag.wiring.make_embedder", embedder)
    monkeypatch.setattr("handbook_rag.wiring.make_verifier", lambda _settings: used["verifier"])
    for name in ("make_generator", "make_judge", "make_store", "make_retriever", "make_reranker"):
        monkeypatch.setattr(f"handbook_rag.wiring.{name}", never)
    return used


@pytest.fixture
def files(tmp_path: Path) -> list[str]:
    """A small golden set and its pairs on disk, as the options that name them."""
    golden, pairs = a_small_world()
    dataset, pairs_file = tmp_path / "questions.jsonl", tmp_path / "pairs.jsonl"
    dataset.write_text("".join(q.model_dump_json() + "\n" for q in golden), encoding="utf-8")
    pairs_file.write_text("".join(p.model_dump_json() + "\n" for p in pairs), encoding="utf-8")
    return ["--dataset", str(dataset), "--pairs", str(pairs_file)]


def test_eval_cache_measures_the_pairs_and_saves_a_scratch_run(
    models: dict[str, Any], files: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "datetime", StoppedClock())
    result = runner.invoke(app, ["eval-cache", *files])

    assert result.exit_code == 0, result.output
    run = Path("evals/runs/cache/current")
    assert sorted(path.name for path in run.iterdir()) == [
        "pairs.jsonl",
        "questions.jsonl",
        "summary.json",
    ]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["pairs"]["agreed"] == 24
    assert summary["pairs"]["disputed"] == ["s090", "s091"]
    assert summary["config"] == {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_query_prefix": "Represent this sentence for searching relevant passages: ",
        "embedding_device": "cpu",
        "verifier_model": "cross-encoder/nli-deberta-v3-small",
        "verifier_device": "cpu",
        "verifier_outputs": ["contradiction", "entailment", "neutral"],
        "verifier_entailment_output": 1,
        "pairs_fingerprint": summary["config"]["pairs_fingerprint"],
        "stored_questions_fingerprint": summary["config"]["stored_questions_fingerprint"],
        "candidates": 3,
        "floor": 0.5,
        "split_seed": cache_pairs.SPLIT_SEED,
        "other_splits": cache_pairs.OTHER_SPLITS,
    }
    # The day of the run, by the clock of the machine that made it, which stands still here.
    assert summary["measured_on"] == "2031-02-03"
    assert summary["rows"] == cache_pairs.rows_fingerprints(run)
    assert (
        f"Cache measurement of {summary['measured_on']}: 24 agreed pairs (12 same, 12 different)"
    ) in result.output
    assert "none to any model API" in result.output
    assert "On the reporting half, which chose nothing" in result.output
    assert "not a held-out result" in result.output
    assert f"written to   {run}" in result.output
    # 12 stored questions and 26 others, each embedded once, as a question.
    assert len(models["embedder"].queries) == 38
    assert models["embedder"].embedded == []
    assert len(models["verifier"].calls) == 1


def test_eval_cache_embeds_on_the_cpu_as_the_service_does(
    models: dict[str, Any], files: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["eval-cache", *files])
    assert models["embedding_settings"].embedding_device == "cpu"

    # The run records the device the embedding model says it ran on, not an assumption.
    models["embedder"].device = "mps"
    runner.invoke(app, ["eval-cache", *files])
    recorded = json.loads(Path("evals/runs/cache/current/summary.json").read_text(encoding="utf-8"))
    assert recorded["config"]["embedding_device"] == "mps"
    assert recorded["config"]["verifier_device"] == "cpu"

    # A device named in the settings is kept, as it is for the service.
    monkeypatch.setenv("HANDBOOK_RAG_EMBEDDING_DEVICE", "mps")
    from handbook_rag.config import get_settings

    get_settings.cache_clear()
    runner.invoke(app, ["eval-cache", *files])
    assert models["embedding_settings"].embedding_device == "mps"


def test_eval_cache_takes_a_name_a_directory_and_the_two_numbers_of_the_cache(
    models: dict[str, Any], files: list[str], tmp_path: Path
) -> None:
    command = ["eval-cache", *files, "--variant", "v1", "--output", str(tmp_path / "runs")]

    result = runner.invoke(app, [*command, "--candidates", "1", "--floor", "0.25"])

    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "runs" / "v1" / "summary.json").read_text(encoding="utf-8"))
    assert (summary["config"]["candidates"], summary["config"]["floor"]) == (1, 0.25)
    assert summary["cache_view"]["candidates"] == 1
    lines = (tmp_path / "runs" / "v1" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(len(json.loads(line)["candidates"]) <= 1 for line in lines)
    assert "meets its 1 nearest above a cosine of 0.25" in result.output


def test_eval_cache_gives_the_same_figures_each_time(
    models: dict[str, Any], files: list[str], tmp_path: Path
) -> None:
    for name in ("first", "second"):
        runner.invoke(app, ["eval-cache", *files, "--variant", name, "--output", str(tmp_path)])
    first, second = (
        json.loads((tmp_path / name / "summary.json").read_text(encoding="utf-8"))
        for name in ("first", "second")
    )

    assert {**first, "timing": None} == {**second, "timing": None}
    assert first["rows"] == second["rows"]
    for name in ("pairs.jsonl", "questions.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


@pytest.mark.usefixtures("models")
def test_eval_cache_explains_missing_and_malformed_inputs(files: list[str]) -> None:
    dataset, pairs_file = files[1], Path(files[3])

    no_pairs = runner.invoke(app, ["eval-cache", "--dataset", dataset, "--pairs", "none.jsonl"])
    no_golden = runner.invoke(
        app, ["eval-cache", "--dataset", "none.jsonl", "--pairs", str(pairs_file)]
    )
    lines = pairs_file.read_text(encoding="utf-8").splitlines()
    pairs_file.write_text("\n".join([*lines[:3], '{"pair": "s999"}', *lines[3:]]) + "\n")
    malformed = runner.invoke(app, ["eval-cache", *files])

    assert no_pairs.exit_code == no_golden.exit_code == malformed.exit_code == 1
    assert "error: cache question pairs not found: none.jsonl" in no_pairs.output
    assert "error: golden set not found: none.jsonl" in no_golden.output
    assert f"error: {pairs_file}:4: not a cache question pair" in malformed.output
    # Nothing is written for a run that measured nothing.
    assert not Path("evals/runs/cache").exists()


def test_eval_cache_refuses_a_verifier_that_names_no_entailment_output(
    models: dict[str, Any], files: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_settings: Settings) -> None:
        raise VerifierError("The model x cannot be used as a verifier: it names LABEL_0.")

    monkeypatch.setattr("handbook_rag.wiring.make_verifier", refuse)

    result = runner.invoke(app, ["eval-cache", *files])

    assert result.exit_code == 1
    assert "error: The model x cannot be used as a verifier" in result.output
    assert not Path("evals/runs/cache").exists()


@pytest.mark.usefixtures("models")
def test_eval_cache_needs_at_least_one_candidate_and_a_floor_that_is_a_cosine(
    files: list[str],
) -> None:
    for option, value in (("--candidates", "0"), ("--floor", "1.5")):
        result = runner.invoke(app, ["eval-cache", *files, option, value])

        assert result.exit_code == 2, option
        assert option in COLOUR_CODES.sub("", result.output)


def test_eval_cache_has_no_option_for_the_split() -> None:
    # The split is fixed in code, so that no run can pick the one that suits it.
    result = runner.invoke(app, ["eval-cache", "--help"])

    assert result.exit_code == 0
    assert "--seed" not in result.output
    assert "--split" not in result.output
    assert "No cache is built" in result.output
