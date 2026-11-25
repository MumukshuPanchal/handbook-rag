"""The ``eval-limits`` command, with a stand-in answer model and a stand-in judge.

A real run sends 80 requests: one for each of the 40 questions, and one to the judge for
each answer. These tests ask three questions of a two-page handbook on disk, and send
nothing. The command reads the handbook but needs no index.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.evals.careful import Pacer
from handbook_rag.evals.judge import JudgeError
from handbook_rag.evals.support import passage_hash
from handbook_rag.generate.base import GenerationError, is_answered
from handbook_rag.generate.models import Answer, Citation, Segment, TokenUsage
from handbook_rag.generate.prompt import CURRENT_INSTRUCTIONS, NOT_FOUND_MESSAGE
from handbook_rag.ingest.build import build_corpus
from handbook_rag.ingest.models import Chunk
from handbook_rag.measuring import measuring
from tests.support import StandInModel, count_words, write_snapshot
from tests.test_cli import FakeJudge

runner = CliRunner()
PAGES = {
    "people/leave.md": (
        '---\ntitle: "Leave"\n---\n\n## Parental leave\n\n'
        "In Canada, new parents get sixteen weeks of paid leave."
    ),
    "finance/expenses.md": (
        "## Laptops\n\nManagers may expense a laptop up to one thousand dollars."
    ),
    "facilities/parking.md": "## Parking\n\nStaff at the Berlin office may park in the yard.",
}


class LimitModel(StandInModel):
    """Answers from its one passage, says "eighteen" about laptops, declines on parking."""

    def __init__(self, *, error: GenerationError | None = None) -> None:
        self.sent: list[tuple[str, list[str]]] = []
        self.built_with: list[dict[str, Any]] = []
        self.measuring: list[bool] = []
        self._error = error

    def build(self, _settings: Settings, **options: Any) -> "LimitModel":
        self.built_with.append(options)
        return self

    def generate(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.sent.append((question, [chunk.text for chunk in chunks]))
        self.measuring.append(measuring())
        if self._error is not None:
            raise self._error
        (chunk,) = chunks
        if "park" in question:
            segments: tuple[Segment, ...] = (Segment(NOT_FOUND_MESSAGE),)
        elif "laptop" in question:
            segments = (
                Segment(
                    "You may expense a laptop up to eighteen hundred dollars.", (Citation(chunk),)
                ),
            )
        else:
            segments = (
                Segment("New parents get sixteen weeks of paid leave.", (Citation(chunk),)),
            )
        return Answer(
            question=question,
            segments=segments,
            answered=is_answered(segments, refused=False),
            model="fake",
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=400, output_tokens=30),
            seconds=1.0,
        )


def labelled(case: str, kind: str, chunk: Chunk, question: str, limit: str) -> dict[str, Any]:
    return {
        "case": case,
        "kind": kind,
        "section": chunk.section,
        "limit_kind": "where",
        "limit": limit,
        "limit_where": "text",
        "question": question,
        "passages": [
            {"number": 2, "chunk_id": chunk.chunk_id, "text_sha256": passage_hash(chunk.text)}
        ],
        "statements": [{"number": 1, "text": "A statement.", "cites": [2]}],
    }


@pytest.fixture
def limit_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A handbook on disk, a limit set of three passages, stand-ins, and no waiting."""
    commit = "a" * 40
    write_snapshot(tmp_path / "data", commit, PAGES)
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HANDBOOK_RAG_HANDBOOK_COMMIT", commit)
    monkeypatch.setenv("HANDBOOK_RAG_MIN_PAGE_CHARS", "20")
    monkeypatch.setattr("handbook_rag.ingest.build.token_counter", lambda _model: count_words)
    monkeypatch.setattr(
        "handbook_rag.cli.Pacer", lambda per_minute: Pacer(per_minute, sleep=lambda _s: None)
    )

    def never(*_arguments: object, **_options: object) -> None:
        raise AssertionError("eval-limits searches nothing")

    for name in ("make_store", "make_retriever", "make_embedder"):
        monkeypatch.setattr(f"handbook_rag.wiring.{name}", never)
    chunks = {chunk.doc_id: chunk for chunk in build_corpus(get_settings()).chunks}
    leave, laptop = chunks["people/leave.md"], chunks["finance/expenses.md"]
    parking = chunks["facilities/parking.md"]
    rows = [
        labelled("lim-001", "kept", leave, "How long is parental leave in Canada?", "In Canada"),
        labelled("lim-002", "dropped", leave, "How long is parental leave?", "In Canada"),
        labelled("lim-003", "dropped", laptop, "Can I expense a laptop?", "Managers"),
        labelled("lim-004", "dropped", parking, "Where can I park?", "the Berlin office"),
    ]
    cases = tmp_path / "limits.jsonl"
    cases.write_text("".join(json.dumps(line) + "\n" for line in rows))
    model, judge = LimitModel(), FakeJudge()
    monkeypatch.setattr("handbook_rag.wiring.make_generator", model.build)
    monkeypatch.setattr("handbook_rag.wiring.make_judge", judge.build)
    return {
        "command": ["eval-limits", "--cases", str(cases), "--output", str(tmp_path / "runs")],
        "cases": cases,
        "run_dir": tmp_path / "runs" / "current",
        "model": model,
        "judge": judge,
        "leave": leave,
    }


def results(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = map(json.loads, (run_dir / "results.jsonl").read_text().splitlines())
    return {row["case"]: row for row in rows}


def test_eval_limits_asks_each_question_with_its_passage_then_judges_each_answer(
    limit_run: dict[str, Any],
) -> None:
    model, judge = limit_run["model"], limit_run["judge"]

    result = runner.invoke(app, limit_run["command"])

    assert result.exit_code == 0, result.output
    assert model.built_with == [{"attempts": 1, "instructions": CURRENT_INSTRUCTIONS}]
    assert sorted(question for question, _ in model.sent) == [
        "Can I expense a laptop?",
        "How long is parental leave?",
        "Where can I park?",
    ]
    # Each question with its one passage, and inside the measuring switch.
    leave = next(texts for question, texts in model.sent if "leave" in question)
    assert leave == [limit_run["leave"].text]
    assert model.measuring == [True, True, True]
    # Version 2 of the support judge, one attempt for each request; the decline is not judged.
    assert (judge.built_for, judge.built_versions) == (["support"], [2])
    assert judge.built_with == [{"attempts": 1}]
    assert sorted(judge.asked) == ["lim-002", "lim-003"]
    assert "3 in the file, 0 already answered, 3 to ask now" in result.output
    assert f"instructions version {CURRENT_INSTRUCTIONS}, the assistant's own" in result.output
    assert "at most 5 now; for this run in all, 3 to the answer model and 2 to the judge" in (
        result.output
    )
    assert "final        yes: every question has a reply, and every answer a verdict" in (
        result.output
    )
    assert "Run the same command again" not in result.output

    run_dir = limit_run["run_dir"]
    rows = results(run_dir)
    assert {case: (row["answered"], row["faulted"]) for case, row in rows.items()} == {
        "lim-002": (True, False),
        "lim-003": (True, True),
        "lim-004": (False, None),
    }
    assert rows["lim-003"]["verdicts"][0]["verdict"] == "unsupported"
    assert rows["lim-003"]["limit"] == "Managers"
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 3
    assert len((run_dir / "support-v2.jsonl").read_text().splitlines()) == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    figures = summary["figures"]["all"]
    assert (figures["answered"], figures["judged"], figures["faulted"]) == (2, 2, 1)
    assert figures["share_faulted"]["mean"] == 0.5
    assert (summary["attempts"], summary["judge"]["attempts"], summary["final"]) == (3, 2, True)
    assert summary["judge"]["config"]["judge_version"] == 2
    assert summary["config"]["handbook_commit"] == "a" * 40
    assert summary["description"].startswith("The 3 questions of the limit set, each with its")


def test_eval_limits_goes_on_where_it_stopped_without_asking_again(
    limit_run: dict[str, Any],
) -> None:
    model, judge = limit_run["model"], limit_run["judge"]

    first = runner.invoke(app, [*limit_run["command"], "--max-requests", "2"])

    # Two answers, and no request left for the judge: a planned stop.
    assert first.exit_code == 0, first.output
    assert (len(model.sent), judge.asked) == (2, [])
    assert "Run the same command again to continue this run." in first.output
    assert "not final" in first.output

    second = runner.invoke(app, limit_run["command"])

    assert second.exit_code == 0, second.output
    assert len(model.sent) == 3
    assert sorted(judge.asked) == ["lim-002", "lim-003"]
    assert "2 already answered, 1 to ask now" in second.output

    # With every answer and verdict saved, nothing is asked and nothing is built.
    third = runner.invoke(app, limit_run["command"])
    assert third.exit_code == 0, third.output
    assert (len(model.sent), len(judge.asked)) == (3, 2)
    assert (len(model.built_with), len(judge.built_with)) == (2, 2)
    assert "final        yes" in third.output


def test_eval_limits_does_not_judge_when_the_answers_did_not_all_come(
    monkeypatch: pytest.MonkeyPatch, limit_run: dict[str, Any]
) -> None:
    failing = LimitModel(error=GenerationError("The model is overloaded."))
    monkeypatch.setattr("handbook_rag.wiring.make_generator", failing.build)

    result = runner.invoke(app, limit_run["command"])

    assert result.exit_code == 1
    assert limit_run["judge"].asked == []
    assert "judge        not asked: the answers did not all come" in result.output


def test_eval_limits_fails_when_the_judge_fails_and_judges_only_what_is_left_next_time(
    monkeypatch: pytest.MonkeyPatch, limit_run: dict[str, Any]
) -> None:
    model = limit_run["model"]
    broken = FakeJudge(error=JudgeError("The judge's reply could not be read."))
    monkeypatch.setattr("handbook_rag.wiring.make_judge", broken.build)

    result = runner.invoke(app, limit_run["command"])

    assert result.exit_code == 1
    assert len(model.sent) == 3
    assert "not yet: 2 answers wait for the judge" in result.output

    judge = FakeJudge()
    monkeypatch.setattr("handbook_rag.wiring.make_judge", judge.build)
    again = runner.invoke(app, limit_run["command"])

    assert again.exit_code == 0, again.output
    assert len(model.sent) == 3
    assert sorted(judge.asked) == ["lim-002", "lim-003"]


def test_eval_limits_refuses_a_passage_that_reads_otherwise_before_any_request(
    limit_run: dict[str, Any],
) -> None:
    cases: Path = limit_run["cases"]
    rows = [json.loads(line) for line in cases.read_text().splitlines()]
    rows[1]["passages"][0]["text_sha256"] = "0" * 16
    cases.write_text("".join(json.dumps(line) + "\n" for line in rows))

    result = runner.invoke(app, limit_run["command"])

    assert result.exit_code == 1
    assert "has changed since lim-002 was written" in result.output
    assert limit_run["model"].built_with == []
    assert not limit_run["run_dir"].exists()


def test_eval_limits_does_not_go_on_with_a_run_made_with_other_settings(
    monkeypatch: pytest.MonkeyPatch, limit_run: dict[str, Any]
) -> None:
    assert runner.invoke(app, limit_run["command"]).exit_code == 0
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_MAX_TOKENS", "123")
    get_settings.cache_clear()

    result = runner.invoke(app, limit_run["command"])

    assert result.exit_code == 1
    assert "made with other settings (answer_max_tokens)" in result.output

    restarted = runner.invoke(app, [*limit_run["command"], "--restart"])
    assert restarted.exit_code == 0, restarted.output
    assert len(limit_run["model"].sent) == 6
    assert len(limit_run["judge"].asked) == 4
    run_dir = limit_run["run_dir"]
    assert len((run_dir / "answers.jsonl").read_text().splitlines()) == 3
    assert len((run_dir / "support-v2.jsonl").read_text().splitlines()) == 2
