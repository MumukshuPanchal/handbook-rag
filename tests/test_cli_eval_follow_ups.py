"""The ``eval-follow-ups`` command, with a stand-in model and a stand-in verifier.

It is in a file of its own, beside ``test_cli.py``, because it needs none of that file's
fixtures: no snapshot, no index and no answer model.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from handbook_rag.cache.verify import VerifierError
from handbook_rag.cli import app
from handbook_rag.config import Settings, get_settings
from handbook_rag.evals.careful import Pacer
from handbook_rag.generate.base import GenerationError
from handbook_rag.measuring import measuring
from handbook_rag.rewrite.models import Turn
from handbook_rag.rewrite.prompt import INSTRUCTION_PHRASES
from handbook_rag.wiring import make_rewriter as REAL_MAKE_REWRITER
from tests.cache.support import TableVerifier
from tests.evals.test_follow_ups import FINLAND, YUBIKEY, row
from tests.rewrite.support import StandInRewriter

runner = CliRunner()
# The colour codes a terminal gets. In CI the usage errors come with them, and they can
# stand in the middle of an option's name.
COLOUR_CODES = re.compile(r"\x1b\[[0-9;]*m")
VERIFIED = "How much annual leave do Finnish staff get?"
# What the stand-in model replies to each question it is sent.
REPLIES = {
    "And in Finland?": FINLAND,
    "And for Finnish staff?": VERIFIED,
    "Finland?": "What about Finland?",
    YUBIKEY: YUBIKEY,
    "And in Finland, then?": "OSPREY-1234?",
}


class Model:
    """The stand-in model, and what it saw while the command ran."""

    def __init__(self, *, fails: Sequence[str] = ()) -> None:
        self.fails = set(fails)
        self.measuring: list[bool] = []
        self.rewriter = StandInRewriter(self.reply)

    def reply(self, _turns: Sequence[Turn], question: str) -> str | BaseException:
        self.measuring.append(measuring())
        if question in self.fails:
            return GenerationError("The model is overloaded.")
        return REPLIES[question]

    @property
    def asked(self) -> list[str]:
        return [question for _, question in self.rewriter.calls]


def never(*_arguments: object, **_options: object) -> None:
    raise AssertionError("eval-follow-ups must not build this")


@pytest.fixture
def follow_up_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Five cases on disk, a stand-in model and verifier, and a guard on everything else."""
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "".join(
            json.dumps(line) + "\n"
            for line in (
                row("c-words"),
                row("c-verifier", question="And for Finnish staff?"),
                row("c-aside", question="Finland?"),
                row("c-alone", "change_of_subject"),
                row("c-planted", "planted_in_turn", question="And in Finland, then?"),
            )
        ),
        encoding="utf-8",
    )
    model = Model()
    verifier = TableVerifier({(FINLAND, VERIFIED): (0.999, 0.995)})
    # The pace of a real run is kept, and each wait is recorded instead of waited.
    waited: list[float] = []
    monkeypatch.setattr(
        "handbook_rag.cli.Pacer", lambda per_minute: Pacer(per_minute, sleep=waited.append)
    )
    monkeypatch.setattr("handbook_rag.wiring.make_rewriter", lambda _settings: model.rewriter)
    monkeypatch.setattr("handbook_rag.wiring.make_verifier", lambda _settings: verifier)
    for name in (
        "make_generator",
        "make_judge",
        "make_store",
        "make_retriever",
        "make_answer_cache",
        "make_retriever_and_cache",
    ):
        monkeypatch.setattr(f"handbook_rag.wiring.{name}", never)
    monkeypatch.setattr("handbook_rag.cache.answers.AnswerCache.__init__", never)
    run_dir = tmp_path / "runs" / "current"
    return {
        "command": ["eval-follow-ups", "--cases", str(cases), "--output", str(tmp_path / "runs")],
        "cases": cases,
        "run_dir": run_dir,
        "model": model,
        "verifier": verifier,
        "waited": waited,
    }


def results_by_case(run_dir: Path) -> dict[str, dict[str, Any]]:
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {line["case"]: line for line in map(json.loads, lines)}


def write_reading(run_dir: Path, case: str, reply: str, label: str) -> None:
    with (run_dir / "readings.jsonl").open("a", encoding="utf-8") as readings:
        readings.write(
            json.dumps(
                {"case": case, "reply": reply, "label": label, "reader": "a test", "note": "n"}
            )
            + "\n"
        )


def test_eval_follow_ups_asks_each_case_once_inside_the_measuring_switch(
    follow_up_run: dict[str, Any],
) -> None:
    model: Model = follow_up_run["model"]

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 0, result.output
    assert sorted(model.asked) == sorted(REPLIES)
    assert model.measuring == [True] * 5
    assert not measuring(), "and the switch is off again afterwards"
    assert (
        "5 in the file (3 follow-ups, 1 that stand alone, 1 with a planted instruction), "
        "0 already answered, 5 to ask now"
    ) in result.output
    assert "requests     at most 5 now, 5 for this run in all" in result.output
    run_dir = follow_up_run["run_dir"]
    assert json.loads((run_dir / "run.json").read_text())["attempts"] == 5
    assert len((run_dir / "replies.jsonl").read_text().splitlines()) == 5
    rows = results_by_case(run_dir)
    assert (rows["c-words"]["meaning_how"], rows["c-words"]["meaning"]) == ("words", "same")
    assert rows["c-verifier"]["meaning_how"] == "verifier"
    assert (rows["c-aside"]["meaning_by_code"], rows["c-aside"]["meaning"]) == ("to_read", None)
    assert rows["c-alone"]["outcome"] == "unchanged"
    assert rows["c-planted"]["instruction"] == "broke"
    # The verifier read only the rewrites that have the words of no acceptable form.
    (asked,) = follow_up_run["verifier"].calls
    assert sorted({second for _, second in asked}) == sorted(
        [VERIFIED, "What about Finland?", "OSPREY-1234?"]
    )
    assert "final        not yet: 2 cases wait for a reading" in result.output
    # Each waiting case names what waits: a case without a planted instruction never
    # waits for a reading of one.
    assert (
        f"    c-aside                    reply {rows['c-aside']['reply_fingerprint']}  "
        "the question\n"
    ) in result.output
    assert (
        f"    c-planted                  reply {rows['c-planted']['reply_fingerprint']}  "
        "the question\n"
    ) in result.output
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["config"]["rewrite_model"] == "gemini-3.5-flash-lite"
    assert summary["verifier"] == "cross-encoder/nli-deberta-v3-small"
    assert summary["final"] is False


def test_eval_follow_ups_goes_on_with_a_stopped_run_without_asking_again(
    follow_up_run: dict[str, Any],
) -> None:
    model: Model = follow_up_run["model"]

    first = runner.invoke(app, [*follow_up_run["command"], "--max-requests", "2"])
    second = runner.invoke(app, follow_up_run["command"])

    assert first.exit_code == 0, "a limit the caller set is a planned stop"
    assert "0 already answered, 2 to ask now" in first.output
    assert "stopped: the limit of 2 requests for this run was reached" in first.output
    assert "Run the same command again to continue this run." in first.output
    assert second.exit_code == 0, second.output
    assert "2 already answered, 3 to ask now" in second.output
    assert sorted(model.asked) == sorted(REPLIES)
    run = json.loads((follow_up_run["run_dir"] / "run.json").read_text())
    assert run["attempts"] == 5


def test_eval_follow_ups_does_not_go_on_with_a_run_made_with_other_settings(
    follow_up_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    model: Model = follow_up_run["model"]
    assert runner.invoke(app, [*follow_up_run["command"], "--max-requests", "1"]).exit_code == 0
    run_dir = follow_up_run["run_dir"]
    write_reading(run_dir, "c-words", "0" * 16, "same")

    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_MODEL", "gemini-9-other")
    get_settings.cache_clear()
    refused = runner.invoke(app, follow_up_run["command"])
    restarted = runner.invoke(app, [*follow_up_run["command"], "--restart"])

    assert refused.exit_code == 1
    assert "was made with other settings (rewrite_model)" in refused.output
    assert restarted.exit_code == 0, restarted.output
    assert "0 already answered, 5 to ask now" in restarted.output
    assert len(model.asked) == 6
    # The readings were of replies that are gone.
    assert not (run_dir / "readings.jsonl").exists()
    config = json.loads((run_dir / "run.json").read_text())["config"]
    assert config["rewrite_model"] == "gemini-9-other"


def test_eval_follow_ups_gives_final_figures_once_every_rewrite_set_aside_is_read(
    follow_up_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, follow_up_run["command"]).exit_code == 0
    run_dir = follow_up_run["run_dir"]
    rows = results_by_case(run_dir)
    write_reading(run_dir, "c-aside", rows["c-aside"]["reply_fingerprint"], "same")
    write_reading(run_dir, "c-planted", rows["c-planted"]["reply_fingerprint"], "different")
    write_reading(run_dir, "c-verifier", rows["c-verifier"]["reply_fingerprint"], "different")
    # With every case answered, no key and no model are needed to take the readings in.
    monkeypatch.setattr("handbook_rag.wiring.make_rewriter", never)

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 0, result.output
    assert (
        "final        yes: every case has a reply, and every rewrite that code set aside has "
        "a reading"
    ) in result.output
    assert re.search(
        r"  follow_up\s+3\s+3\s+0\s+3\s+0\s+2\s+0\s+1\s+1/0\s+1\.000 \(", result.output
    )
    assert re.search(
        r"  standalone\s+1\s+1\s+1\s+0\s+0\s+1\s+0\s+0\s+0/0\s+1\.000 \(", result.output
    )
    assert "1 broke by code" in result.output
    assert "share held 0.000" in result.output
    # A reading of what the verifier decided checks it, and does not overturn it.
    assert (
        "checks       1 readings of code's own decisions: 0 agree, 1 disagree; of the "
        "verifier's, 1 read, 1 disagree"
    ) in result.output
    assert "c-verifier                   code same, reading different" in result.output
    summary = json.loads((run_dir / "summary.json").read_text())
    assert (summary["final"], summary["waiting_for_readings"]) == (True, [])
    assert summary["figures"]["follow_up"]["same"]["mean"] == 1.0
    assert (summary["checks"]["verifier_read"], summary["checks"]["verifier_disagree"]) == (1, 1)


@pytest.mark.parametrize(
    ("case", "reply", "label", "problem"),
    [
        ("c-aside", "0" * 16, "same", "the reading of c-aside is of another reply"),
        ("c-nobody", "0" * 16, "same", "reading of c-nobody, which has no saved reply"),
        ("c-words", "from-results", "held", "c-words, which has no planted instruction"),
        ("c-words", "from-results", "maybe", "labelled same, different, held or broke"),
    ],
)
def test_eval_follow_ups_refuses_a_reading_before_it_asks_anything(
    follow_up_run: dict[str, Any], case: str, reply: str, label: str, problem: str
) -> None:
    model: Model = follow_up_run["model"]
    assert runner.invoke(app, [*follow_up_run["command"], "--max-requests", "4"]).exit_code == 0
    run_dir = follow_up_run["run_dir"]
    if reply == "from-results":
        reply = results_by_case(run_dir)[case]["reply_fingerprint"]
    summary_before = (run_dir / "summary.json").read_text()
    write_reading(run_dir, case, reply, label)

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert problem in result.output
    # Refused before the fifth case was asked, and nothing written is rewritten.
    assert len(model.asked) == 4
    assert (run_dir / "summary.json").read_text() == summary_before


def test_eval_follow_ups_fails_when_a_case_failed_though_others_were_answered(
    follow_up_run: dict[str, Any],
) -> None:
    model: Model = follow_up_run["model"]
    model.fails = {"Finland?"}

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert model.asked.count("Finland?") == 2, "tried once more after the others"
    assert "failed: c-aside: The model is overloaded." in result.output
    assert "replies      4 read, 1 failed, 0 not asked yet" in result.output
    assert len(results_by_case(follow_up_run["run_dir"])) == 4


def test_eval_follow_ups_is_not_final_while_a_case_has_no_reply(
    follow_up_run: dict[str, Any],
) -> None:
    model: Model = follow_up_run["model"]
    model.fails = {"And in Finland?"}
    assert runner.invoke(app, follow_up_run["command"]).exit_code == 1
    run_dir = follow_up_run["run_dir"]
    rows = results_by_case(run_dir)
    write_reading(run_dir, "c-aside", rows["c-aside"]["reply_fingerprint"], "same")
    write_reading(run_dir, "c-planted", rows["c-planted"]["reply_fingerprint"], "different")

    result = runner.invoke(app, follow_up_run["command"])

    # Every reply it has is decided, and c-words has none: no share stands for it.
    assert result.exit_code == 1
    assert "final        not yet: 1 case has no reply" in result.output
    assert re.search(r"  follow_up\s+3\s+2\s+.*waits for replies", result.output)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert (summary["final"], summary["waiting_for_readings"]) == (False, [])
    assert summary["figures"]["follow_up"]["same"] is None


def test_eval_follow_ups_stops_when_cases_fail_one_after_another(
    follow_up_run: dict[str, Any],
) -> None:
    model: Model = follow_up_run["model"]
    model.fails = set(REPLIES)

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert len(model.asked) == 3
    assert "error: no case was answered. 3 cases in a row failed" in result.output
    # Each failure is shown as it happens, and a rest of thirty seconds follows each of
    # the first two; none follows the last.
    assert result.output.count("failed  ") == 3
    assert follow_up_run["waited"].count(30.0) == 2
    assert json.loads((follow_up_run["run_dir"] / "run.json").read_text())["attempts"] == 3
    assert not (follow_up_run["run_dir"] / "summary.json").exists()


def test_eval_follow_ups_without_a_key_says_where_to_put_one(
    follow_up_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real wiring, which looks for the Gemini key of the answer model. Only it is put
    # back: the test still runs in an empty directory and sees no key of the machine's.
    monkeypatch.setattr("handbook_rag.wiring.make_rewriter", REAL_MAKE_REWRITER)

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert "error: No Gemini API key was found" in result.output
    assert not follow_up_run["run_dir"].exists(), "a run that cannot start leaves nothing"


def test_eval_follow_ups_refuses_cases_that_hold_a_phrase_of_the_instructions(
    follow_up_run: dict[str, Any],
) -> None:
    cases: Path = follow_up_run["cases"]
    with cases.open("a", encoding="utf-8") as lines:
        lines.write(json.dumps(row("c-leak", question=f"Why {INSTRUCTION_PHRASES[1]}?")) + "\n")

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert "c-leak holds 'take the latest one that fits'" in result.output
    assert follow_up_run["model"].asked == []


def test_eval_follow_ups_reports_a_verifier_it_cannot_use(
    follow_up_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_settings: Settings) -> None:
        raise VerifierError("The model x cannot be used as a verifier: it names LABEL_0.")

    monkeypatch.setattr("handbook_rag.wiring.make_verifier", refuse)

    result = runner.invoke(app, follow_up_run["command"])

    assert result.exit_code == 1
    assert "error: The model x cannot be used as a verifier" in result.output
    # The replies it paid for are kept, and read next time.
    assert len((follow_up_run["run_dir"] / "replies.jsonl").read_text().splitlines()) == 5


def test_eval_follow_ups_keeps_its_own_pace_and_refuses_one_that_is_not(
    follow_up_run: dict[str, Any],
) -> None:
    refused = runner.invoke(app, [*follow_up_run["command"], "--pace", "0"])
    no_limit = runner.invoke(app, [*follow_up_run["command"], "--max-requests", "0"])
    paced = runner.invoke(app, [*follow_up_run["command"], "--pace", "2"])

    assert refused.exit_code == 1
    assert "error: --pace needs a positive number of requests a minute." in refused.output
    assert no_limit.exit_code == 2
    assert "--max-requests" in COLOUR_CODES.sub("", no_limit.output)
    assert paced.exit_code == 0, paced.output
    assert "at most 2 requests a minute, so at least 2 min" in paced.output
    # Four pauses between five requests, each up to thirty seconds.
    waited = follow_up_run["waited"]
    assert len(waited) == 4
    assert all(25 < seconds <= 30 for seconds in waited)


def test_eval_follow_ups_says_what_it_costs_in_its_help() -> None:
    result = runner.invoke(app, ["eval-follow-ups", "--help"])

    assert result.exit_code == 0
    said = " ".join(COLOUR_CODES.sub("", result.output).replace("│", " ").split())
    assert "One real request for each case" in said
    assert "no answer cache is built" in said
    for option in ("--cases", "--variant", "--output", "--max-requests", "--pace", "--restart"):
        assert option in said
