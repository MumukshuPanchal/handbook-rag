"""The ``eval-cache`` command, with stand-ins for the two models.

It is in a file of its own, beside ``test_cli.py``, because it needs none of that file's
fixtures: no snapshot, no index and no answer model.
"""

import datetime
import hashlib
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
from handbook_rag.evals.answers import ANSWERS_FILE, AnswerRecord, Passage, Statement, append_record
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.measuring import measuring
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


def test_eval_cache_can_leave_stored_declines_out_as_the_answer_cache_does(
    models: dict[str, Any], files: list[str], tmp_path: Path
) -> None:
    command = ["eval-cache", *files, "--output", str(tmp_path / "runs")]

    as_before = runner.invoke(app, [*command, "--variant", "kept"])
    left_out = runner.invoke(app, [*command, "--variant", "left-out", "--no-declines"])

    assert as_before.exit_code == left_out.exit_code == 0, left_out.output
    kept, without = (
        json.loads((tmp_path / "runs" / name / "summary.json").read_text(encoding="utf-8"))
        for name in ("kept", "left-out")
    )
    # A run made without the option is recorded as it always was.
    assert "decline_candidates" not in kept["config"]
    assert "stored_declines" not in kept["cache_view"]
    assert "a stored decline is no candidate" not in as_before.output
    assert kept["description"].endswith("as the test of a semantic cache.")
    # With it, the run says so in its settings, its figures, its description and its report.
    assert without["config"] == {**kept["config"], "decline_candidates": False}
    # Every golden question of the small world is one the handbook does not answer.
    assert without["cache_view"]["stored_declines"]["left_out"] == 12
    assert without["description"].endswith(
        "A stored decline is no candidate, as in the answer cache."
    )
    assert "a stored decline is no candidate: the 12 are left out" in left_out.output
    lines = (tmp_path / "runs" / "left-out" / "questions.jsonl").read_text(encoding="utf-8")
    assert all(json.loads(line)["candidates"] == [] for line in lines.splitlines())


def an_answer_run(run_dir: Path) -> Path:
    """The replies of a model to the small world's golden questions, as an answer run saves them.

    It answered g0 to g7 from a passage. It declined g8 and g9, and to g10 it wrote part
    of an answer and said what was missing. Nothing was saved for g11.
    """
    golden, _ = a_small_world()
    answer = (Statement("Ask your manager, who approves it within a week.", (1,)),)
    written: dict[str, tuple[Statement, ...]] = {question.id: answer for question in golden[:8]}
    written |= {question.id: (Statement(NOT_FOUND_MESSAGE, ()),) for question in golden[8:10]}
    written["g10"] = (*answer, Statement(" The handbook does not say who pays.", ()))
    run_dir.mkdir(parents=True)
    for question in golden[:11]:
        append_record(
            run_dir,
            AnswerRecord(
                question_id=question.id,
                fingerprint=question.fingerprint,
                passages=(
                    Passage(1, "people/approvals.md#0", "people/approvals.md", "Approvals", ()),
                ),
                statements=written[question.id],
                answered=question.id not in ("g8", "g9"),
                stop_reason="end_turn",
                model="answer-model",
                input_tokens=1800,
                output_tokens=40,
                seconds=2.0,
            ),
        )
    return run_dir


def test_eval_cache_can_fill_its_cache_with_the_replies_of_an_answer_run(
    models: dict[str, Any], files: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = an_answer_run(tmp_path / "answers" / "a-run")
    switched_on: list[bool] = []
    measure = cache_pairs.measure

    def watched(*arguments: Any, **options: Any) -> cache_pairs.Measured:
        switched_on.append(measuring())
        return measure(*arguments, **options)

    monkeypatch.setattr(cache_pairs, "measure", watched)
    command = ["eval-cache", *files, "--output", str(tmp_path / "runs")]

    by_the_label = runner.invoke(app, [*command, "--variant", "by-the-label"])
    result = runner.invoke(app, [*command, "--variant", "replies", "--replies", str(answers)])

    assert by_the_label.exit_code == result.exit_code == 0, result.output
    # It reads saved replies and asks no model: the fixture fails the test if it builds
    # one, and the cache is switched off while it runs, as for every evaluation.
    assert switched_on == [True, True]
    plain, summary = (
        json.loads((tmp_path / "runs" / name / "summary.json").read_text(encoding="utf-8"))
        for name in ("by-the-label", "replies")
    )
    saved = (answers / ANSWERS_FILE).read_bytes()
    assert summary["config"] == {
        **plain["config"],
        "replies_from": answers.as_posix(),
        "replies_fingerprint": hashlib.sha256(saved).hexdigest()[:16],
    }
    assert summary["description"].endswith(
        f"The cache holds the replies of the answer run {answers.as_posix()}, and the answer "
        "cache's own rule decides which of them are candidates."
    )
    cache = summary["cache_view"]
    replies = cache["stored_replies"]
    assert (replies["from"], replies["fingerprint"]) == (
        summary["config"]["replies_from"],
        summary["config"]["replies_fingerprint"],
    )
    assert (replies["kept"], replies["candidates"]) == (11, 8)
    assert replies["kept_from_the_second_tier"] == {
        "decline": ["g8", "g9"],
        "declines_in_part": [],
        "says_something_is_missing": ["g10"],
        "cites_nothing": [],
    }
    assert (replies["not_kept"], replies["without_a_reply"]) == ([], ["g11"])
    # By the label the handbook answers none of the twelve, and the model answered eight.
    assert replies["candidates_the_handbook_does_not_answer"] == [f"g{n}" for n in range(8)]
    assert (cache["stored_questions"], plain["cache_view"]["stored_questions"]) == (11, 12)
    assert "stored_replies" not in plain["cache_view"]
    assert (
        f"filled with the replies of {answers.as_posix()}: 11 kept, and 8 of them candidates "
        "by the cache's own rule"
    ) in result.output
    assert (
        "kept from the second tier: 2 decline, 0 declines_in_part, 1 says_something_is_missing, "
        "0 cites_nothing"
    ) in result.output
    assert "filled with the replies" not in by_the_label.output
    # No question is offered a reply that may not be served, and some are offered one.
    rows = (tmp_path / "runs" / "replies" / "questions.jsonl").read_text(encoding="utf-8")
    offered = {
        c["question_id"] for line in rows.splitlines() for c in json.loads(line)["candidates"]
    }
    assert offered
    assert offered <= {f"g{n}" for n in range(8)}


@pytest.mark.usefixtures("models")
def test_eval_cache_takes_the_replies_or_the_label_to_leave_questions_out_and_not_both(
    files: list[str], tmp_path: Path
) -> None:
    answers = an_answer_run(tmp_path / "a-run")

    result = runner.invoke(
        app,
        [
            "eval-cache",
            *files,
            "--output",
            str(tmp_path / "runs"),
            "--replies",
            str(answers),
            "--no-declines",
        ],
    )

    assert result.exit_code == 2
    # In CI a usage error comes with colour codes, and its box breaks the lines.
    said = " ".join(COLOUR_CODES.sub("", result.output).replace("│", " ").split())
    assert "--replies and --no-declines are two ways to leave stored questions out" in said
    assert "Give one of them." in said
    assert not (tmp_path / "runs").exists()


@pytest.mark.usefixtures("models")
def test_eval_cache_explains_replies_it_cannot_use(files: list[str], tmp_path: Path) -> None:
    command = ["eval-cache", *files, "--output", str(tmp_path / "runs"), "--replies"]
    # A run in which the model was cut off every time leaves a cache nothing to keep.
    answers = an_answer_run(tmp_path / "a-run")
    lines = (answers / ANSWERS_FILE).read_text(encoding="utf-8").splitlines()
    cut_off = tmp_path / "cut-off"
    cut_off.mkdir()
    (cut_off / ANSWERS_FILE).write_text(
        "".join(
            json.dumps({**json.loads(line), "stop_reason": "max_tokens"}) + "\n" for line in lines
        ),
        encoding="utf-8",
    )
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ANSWERS_FILE).write_text(lines[0] + '\n{"question_id": "g1"}\n', encoding="utf-8")

    no_run = runner.invoke(app, [*command, str(tmp_path / "no-such-run")])
    nothing_kept = runner.invoke(app, [*command, str(cut_off)])
    not_replies = runner.invoke(app, [*command, str(broken)])

    assert no_run.exit_code == nothing_kept.exit_code == not_replies.exit_code == 1
    assert (
        f"error: saved replies not found: {tmp_path / 'no-such-run' / ANSWERS_FILE}"
        in no_run.output
    )
    assert "holds no reply that an answer cache would keep" in nothing_kept.output
    assert f"error: {broken / ANSWERS_FILE}:2 is not a saved reply" in not_replies.output
    assert not (tmp_path / "runs").exists()


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
    assert "--no-declines" in COLOUR_CODES.sub("", result.output)
    assert "--replies" in COLOUR_CODES.sub("", result.output)
