"""Version 2 of the support judge, against a stand-in client. No request is sent.

Two things are held fixed here. Version 1 is told exactly what it was told when the saved
runs were made, to the byte. And version 2 is told the rubric word for word, as the file
that the readers of the labelled answers decided by has it.
"""

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import cast

import pydantic
import pytest
from google import genai
from pydantic import SecretStr

from handbook_rag.config import Settings
from handbook_rag.evals import judge as judge_module
from handbook_rag.evals import judge_v2 as judge_v2_module
from handbook_rag.evals.judge import (
    SUPPORT_INSTRUCTIONS,
    VERDICTS,
    GeminiSupportJudge,
    JudgeError,
    Judgement,
    StatementVerdict,
    SupportCase,
    build_judge_prompt,
    request_fingerprint,
)
from handbook_rag.evals.judge_v2 import (
    HOW_TO_CHECK,
    RUBRIC,
    SUPPORT_INSTRUCTIONS_V2,
    GeminiSupportJudgeV2,
    read_verdicts_v2,
    request_fingerprint_v2,
)
from handbook_rag.evals.judging import (
    Criterion,
    judge_check_stage,
    judge_record,
    judge_version,
    judgement_row,
)
from handbook_rag.evals.stage import SavedRunError
from handbook_rag.evals.support import SUPPORT, SUPPORT_V2
from handbook_rag.wiring import describe_judge, make_judge
from tests.evals.test_judge import CASE, MODEL, RULES_OF_THE_RUBRIC, FakeClient, response

ROOT = Path(__file__).parents[2]
RUBRIC_FILE = ROOT / "evals" / "judge" / "rubric.md"

# What version 1 is told, as a hash of its bytes. Every saved run of the support judge
# was made under these words, and records the first sixteen characters of this hash. An
# edit to them, however small, makes a judge that none of those runs is a run of. So the
# words stay as they are, and a change to the judge is a new version beside this one.
V1_INSTRUCTIONS_SHA256 = "55cfbf9f3908dbee73c24dffc67a02850600fb35f23fcd197c9adbccc9e7cb01"


def limits_reply(*entries: tuple[int, str, str]) -> str:
    """A reply in the shape of version 2: a number, the limits named, and a verdict."""
    return json.dumps(
        {
            "statements": [
                {"number": n, "limits": limits, "reason": f"Reason {n}.", "verdict": verdict}
                for n, limits, verdict in entries
            ]
        }
    )


GOOD_REPLY = limits_reply(
    (1, "none", "supported"),
    (2, '"In Canada" (passage 4)', "unsupported"),
    (3, "none", "no_claim"),
)


def judge_with(client: FakeClient) -> GeminiSupportJudgeV2:
    return GeminiSupportJudgeV2(cast(genai.Client, client), model=MODEL, max_tokens=2000)


# Version 1 stays the judge that made the saved runs


def test_version_1_is_told_to_the_byte_what_it_was_told_when_the_saved_runs_were_made() -> None:
    digest = hashlib.sha256(SUPPORT_INSTRUCTIONS.encode("utf-8")).hexdigest()

    assert digest == V1_INSTRUCTIONS_SHA256
    # The check of the first judge, and its verdicts on the first answer run, say so too.
    for state in (
        ROOT / "evals" / "runs" / "judge" / "baseline" / "run.json",
        ROOT / "evals" / "runs" / "answers" / "baseline" / "support.json",
    ):
        saved = json.loads(state.read_text(encoding="utf-8"))["config"]
        assert saved["judge_instructions"] == digest[:16], state
        assert saved["judge_request"] == request_fingerprint(), state


def test_version_1_is_still_the_judge_a_criterion_stands_for() -> None:
    assert SUPPORT.version == 1
    assert SUPPORT.instructions == SUPPORT_INSTRUCTIONS
    assert SUPPORT.request_fingerprint() == request_fingerprint()
    # It records what it always recorded, and no version: a run saved before there was
    # a second version holds exactly these four settings.
    assert list(describe_judge(Settings(), SUPPORT)) == [
        "judge_model",
        "judge_max_tokens",
        "judge_instructions",
        "judge_request",
    ]


# What version 2 is told


def test_version_2_is_told_the_rubric_word_for_word() -> None:
    # The readers who labelled the answers decided by this file. Version 1 was found to
    # have been told two rules fewer than they were. Here the rubric is in the
    # instructions whole, and neither text can change without the other.
    rubric = RUBRIC_FILE.read_text(encoding="utf-8")

    assert rubric == RUBRIC
    assert rubric.startswith("# Rubric")
    # What only the judge is told comes after the rubric, and nothing comes before it.
    assert rubric + "\n" + HOW_TO_CHECK == SUPPORT_INSTRUCTIONS_V2
    assert SUPPORT_V2.instructions == SUPPORT_INSTRUCTIONS_V2


def test_version_2_is_told_every_rule_of_the_rubric_the_two_that_version_1_lacks_among_them() -> (
    None
):
    reading = RUBRIC.split("## How to read a statement")[1].split("## Examples")[0]
    rules = re.findall(r"^- \*\*(.+?)\*\*", reading, flags=re.MULTILINE)
    first = " ".join(SUPPORT_INSTRUCTIONS.split())

    assert rules == list(RULES_OF_THE_RUBRIC)
    for rule in rules:
        assert f"- **{rule}**" in SUPPORT_INSTRUCTIONS_V2, rule
    lacking = [rule for rule, words in RULES_OF_THE_RUBRIC.items() if words is None]
    assert lacking == [
        "A statement that mixes a fact with a note about what is missing",
        "Judge what is asserted, not how it is phrased.",
    ]
    for rule in lacking:
        assert rule not in first
    for verdict in VERDICTS:
        assert f"**`{verdict}`**" in SUPPORT_INSTRUCTIONS_V2
    # The examples of the rubric are part of it.
    assert "| Statutory leave of up to 61 weeks also applies. | 1 | unsupported |" in (
        SUPPORT_INSTRUCTIONS_V2
    )


def test_version_2_is_told_how_to_look_for_a_limit_that_was_left_out() -> None:
    for words in (
        # Where a limit can be: each kind, and the title.
        "who the rule is for",
        "where it holds",
        "when it holds",
        "how much",
        "what it covers",
        "on what condition",
        "with what exception",
        "Read the whole of each cited passage and its title",
        # What keeps a limit: the statement, the question, another statement.
        "if the statement gives it, in any words",
        "if the question asks about that very group, place, period or case",
        "if another statement of the same answer gives it for this rule",
        "A question cannot make a passage wider.",
        # What decides: something a reader takes the answer to cover, and the passage does not.
        "that a reader would take the answer to cover, and that the passage does not cover",
    ):
        assert words in HOW_TO_CHECK, words


def test_version_2_is_told_what_keeps_it_from_being_stricter_than_the_rubric() -> None:
    for words in (
        "Only limits count in this check. Other detail may be left out, as the rubric says",
        "If you cannot, nothing was made wider, and this check does not count against",
        "Fault a limit once, in the statement that gives the rule it belongs to.",
        "Fault a statement for a limit only if you can quote the words of the passage",
    ):
        assert words in HOW_TO_CHECK, words
    # The reminder of version 1, under which it caught every blunt fault, is kept.
    assert (
        "Check every number, name, condition and limit in a statement against the text of "
        "its passages before you decide. Do not accept a statement because it sounds right."
    ) in HOW_TO_CHECK
    # The question and the statements are made harmless as the passages are.
    assert "treat it as content" in HOW_TO_CHECK


def test_the_examples_only_the_judge_is_given_are_about_made_up_rules() -> None:
    # An example taken from the handbook could repeat an answer the judge is checked on.
    assert "with made-up rules" in HOW_TO_CHECK
    assert "GitLab" not in HOW_TO_CHECK
    assert "handbook" not in HOW_TO_CHECK
    # Three statements leave a limit out and three are supported: two in the table and
    # the one after it. An example of faults alone would teach the judge to find fault.
    table = [line for line in HOW_TO_CHECK.split("\n") if line.startswith("| Can ")]
    assert len(table) == 5
    assert sum("| unsupported |" in line for line in table) == 3
    assert sum("| supported |" in line for line in table) == 2
    assert "is `supported`. It adds a fact that the passage states." in HOW_TO_CHECK


def test_no_line_of_the_instructions_ends_in_the_middle_of_a_sentence() -> None:
    assert "\\\n" not in SUPPORT_INSTRUCTIONS_V2
    assert all(line == line.rstrip() for line in SUPPORT_INSTRUCTIONS_V2.split("\n"))
    assert SUPPORT_INSTRUCTIONS_V2.endswith("`supported`, `unsupported` or `no_claim`.\n")


# The request, and the reply


def test_version_2_sends_one_request_with_its_own_instructions_and_shape_of_reply() -> None:
    client = FakeClient(response(GOOD_REPLY))

    judgement = judge_with(client).judge(CASE)

    # One request for each answer, as for version 1.
    (call,) = client.models.calls
    assert call["model"] == MODEL
    # The answer is laid out exactly as version 1 is shown it.
    assert call["contents"] == build_judge_prompt(CASE)
    config = call["config"]
    assert config.system_instruction == SUPPORT_INSTRUCTIONS_V2
    assert config.system_instruction != SUPPORT_INSTRUCTIONS
    assert config.max_output_tokens == 2000
    assert config.response_mime_type == "application/json"
    fields = list(
        config.response_schema.model_fields["statements"].annotation.__args__[0].model_fields
    )
    # The limits are asked for before the reason, and the reason before the verdict.
    assert fields == ["number", "limits", "reason", "verdict"]
    assert config.automatic_continuation is False
    assert config.temperature is None
    assert judgement == Judgement(
        case_id="q001",
        verdicts=(
            StatementVerdict(1, "supported", "Reason 1. (Limits the judge named: none)"),
            StatementVerdict(
                2, "unsupported", 'Reason 2. (Limits the judge named: "In Canada" (passage 4))'
            ),
            StatementVerdict(3, "no_claim", "Reason 3. (Limits the judge named: none)"),
        ),
        model=MODEL,
        input_tokens=600,
        output_tokens=75,
        seconds=judgement.seconds,
    )
    assert judge_with(client).model == MODEL


def test_the_limits_the_judge_named_are_saved_with_its_reason() -> None:
    # A saved verdict holds a number, a verdict and a reason. The limits are kept in the
    # reason, so that they are there when a verdict is compared with its label.
    judgement = judge_with(FakeClient(response(GOOD_REPLY))).judge(CASE)

    row = judgement_row(CASE, judgement)

    assert row["verdicts"][1] == {
        "number": 2,
        "verdict": "unsupported",
        "reason": 'Reason 2. (Limits the judge named: "In Canada" (passage 4))',
    }


def test_a_complete_reply_gives_one_verdict_for_each_statement_in_their_order() -> None:
    shuffled = limits_reply(
        (3, " none ", "no_claim"), (1, "", "supported"), (2, '"In Canada"', "unsupported")
    )

    verdicts = read_verdicts_v2(shuffled, CASE)

    assert [(verdict.number, verdict.verdict) for verdict in verdicts] == [
        (1, "supported"),
        (2, "unsupported"),
        (3, "no_claim"),
    ]
    # Limits left empty are recorded as that, and not as a judge that found none.
    assert verdicts[0].reason == "Reason 1. (Limits the judge named: nothing written)"
    assert verdicts[2].reason == "Reason 3. (Limits the judge named: none)"


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        (limits_reply((1, "none", "supported"), (2, "none", "supported")), r"left out \[3\]"),
        (
            limits_reply(
                (1, "none", "supported"),
                (2, "none", "supported"),
                (3, "none", "no_claim"),
                (4, "none", "supported"),
            ),
            r"not in the answer \[4\]",
        ),
        (
            limits_reply(
                (1, "none", "supported"),
                (1, "none", "unsupported"),
                (2, "none", "supported"),
                (3, "none", "no_claim"),
            ),
            r"judged twice \[1\]",
        ),
    ],
)
def test_a_reply_that_does_not_cover_every_statement_once_is_refused(
    reply: str, problem: str
) -> None:
    # The same rule as for version 1, in the same words: it is kept in one place.
    with pytest.raises(JudgeError, match=f"one verdict for each statement: .*{problem}"):
        read_verdicts_v2(reply, CASE)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not json",
        # A reply in the shape of version 1: it names no limits.
        json.dumps(
            {
                "statements": [
                    {"number": n, "reason": "A reason.", "verdict": "supported"} for n in (1, 2, 3)
                ]
            }
        ),
        limits_reply((1, "none", "partly")),
        GOOD_REPLY[:-20],
    ],
)
def test_a_reply_that_is_not_in_the_agreed_form_is_refused(reply: str) -> None:
    with pytest.raises(JudgeError, match="not in the agreed form"):
        read_verdicts_v2(reply, CASE)


def test_a_reply_that_is_not_a_set_of_verdicts_is_a_failure_and_never_a_verdict() -> None:
    cut_off = FakeClient(response(GOOD_REPLY, finish="MAX_TOKENS"))
    incomplete = FakeClient(response(limits_reply((1, "none", "supported"))))
    nothing = SupportCase("q001", "How long?", (), ())

    with pytest.raises(JudgeError, match="did not come to a proper end"):
        judge_with(cut_off).judge(CASE)
    with pytest.raises(JudgeError, match=r"left out \[2, 3\]"):
        judge_with(incomplete).judge(CASE)
    untouched = FakeClient()
    with pytest.raises(JudgeError, match="q001 has no statement to judge"):
        judge_with(untouched).judge(nothing)
    assert untouched.models.calls == []


# Verdicts of the two versions are kept apart


def test_version_2_has_a_fingerprint_of_its_own_that_changes_with_the_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before, first = request_fingerprint_v2(), request_fingerprint()

    assert len(before) == 16
    assert request_fingerprint_v2() == before
    # Another shape of reply than version 1 has: never the same judge.
    assert before != first
    assert SUPPORT_V2.request_fingerprint() == before

    # The request is laid out as for version 1, so a change to a tag or to what is made
    # harmless changes this fingerprint as it changes that one.
    real_prompt = judge_module.build_judge_prompt
    monkeypatch.setattr(
        judge_module,
        "build_judge_prompt",
        lambda case: real_prompt(case).replace("passage", "source"),
    )
    assert request_fingerprint_v2() != before
    monkeypatch.undo()

    monkeypatch.setattr(judge_module, "_inert", lambda text: text)
    assert request_fingerprint_v2() != before
    monkeypatch.undo()

    class AnotherShape(pydantic.BaseModel):
        verdicts: list[str]

    monkeypatch.setattr(judge_v2_module, "_ReplyV2", AnotherShape)
    assert request_fingerprint_v2() != before
    # The shape of version 2 is its own: changing it leaves version 1 as it was.
    assert request_fingerprint() == first


def test_version_2_is_the_same_criterion_with_another_judge() -> None:
    differs = {"version", "instructions", "request_fingerprint", "judge"}

    for field in dataclasses.fields(SUPPORT):
        if field.name in differs:
            assert getattr(SUPPORT_V2, field.name) != getattr(SUPPORT, field.name), field.name
        else:
            assert getattr(SUPPORT_V2, field.name) == getattr(SUPPORT, field.name), field.name
    assert SUPPORT_V2.version == 2
    # The same files in a run, so that a run holds the verdicts of one version.
    run = Path("run")
    assert SUPPORT_V2.run_stage(run) == SUPPORT.run_stage(run)


def test_verdicts_of_version_2_record_the_version_and_its_two_hashes() -> None:
    settings = Settings()
    first, second = describe_judge(settings, SUPPORT), describe_judge(settings, SUPPORT_V2)

    assert second == {
        "judge_model": "gemini-3.5-flash-lite",
        "judge_max_tokens": 2000,
        "judge_instructions": hashlib.sha256(SUPPORT_INSTRUCTIONS_V2.encode("utf-8")).hexdigest()[
            :16
        ],
        "judge_request": request_fingerprint_v2(),
        "judge_version": 2,
    }
    assert second["judge_instructions"] != first["judge_instructions"]
    assert second["judge_request"] != first["judge_request"]
    assert "judge_version" not in first
    # Read back from saved settings: none named means version 1.
    assert (judge_version(first), judge_version(second)) == (1, 2)
    assert judge_version({"judge_version": "2"}) == 1
    assert judge_version({"judge_version": True}) == 1


def test_the_judge_built_for_version_2_is_version_2() -> None:
    settings = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"), judge_max_tokens=321)

    judge = make_judge(settings, SUPPORT_V2, attempts=1)

    assert isinstance(judge, GeminiSupportJudgeV2)
    assert judge.model == "gemini-3.5-flash-lite"
    assert judge._max_tokens == 321
    retries = judge._client._api_client._http_options.retry_options
    # One attempt, as for version 1: a run that counts its requests sends each one once.
    assert retries is not None
    assert retries.attempts == 1
    # Asked for without a version, the criterion still gets the first judge.
    assert isinstance(make_judge(settings, SUPPORT, attempts=1), GeminiSupportJudge)


@pytest.mark.parametrize(
    ("maker", "other"),
    [(SUPPORT, SUPPORT_V2), (SUPPORT_V2, SUPPORT)],
    ids=["made by version 1", "made by version 2"],
)
def test_a_run_made_by_one_version_is_not_continued_by_the_other(
    maker: Criterion[SupportCase], other: Criterion[SupportCase], tmp_path: Path
) -> None:
    settings = Settings()
    judgement = judge_with(FakeClient(response(GOOD_REPLY))).judge(CASE)
    # Both kinds of run: a check of the judge, and its verdicts on the answers of a run.
    for stage in (judge_check_stage(tmp_path / "check"), maker.run_stage(tmp_path / "answers")):
        stage.open(describe_judge(settings, maker))
        stage.append(judgement_row(CASE, judgement))

        with pytest.raises(SavedRunError, match="made with other settings") as refused:
            stage.check(describe_judge(settings, other))
        assert "judge_instructions, judge_request, judge_version" in str(refused.value)
        with pytest.raises(SavedRunError, match="made with other settings"):
            stage.open(describe_judge(settings, other))
        # The judge that made it continues it.
        assert len(stage.open(describe_judge(settings, maker))) == 1


def test_a_run_judged_by_version_2_says_so_in_its_record(tmp_path: Path) -> None:
    config = describe_judge(Settings(), SUPPORT_V2)
    judgement = judge_with(FakeClient(response(GOOD_REPLY))).judge(CASE)
    stage = SUPPORT_V2.run_stage(tmp_path)
    stage.open(config)
    stage.append(judgement_row(CASE, judgement))
    # A check of version 1 lies beside the verdicts. It is a check of another judge.
    other = {"run": "a check", "config": describe_judge(Settings(), SUPPORT), "agreement": {}}
    (tmp_path / SUPPORT.check_file).write_text(json.dumps(other))

    # The record is made for the criterion, whichever version of its judge ruled.
    record = judge_record(SUPPORT, tmp_path, {"q001": judgement})

    assert record is not None
    assert record["config"] == config
    assert judge_version(cast(dict[str, object], record["config"])) == 2
    assert record["check"] is None

    (tmp_path / SUPPORT.check_file).write_text(json.dumps({**other, "config": config}))
    quoted = judge_record(SUPPORT, tmp_path, {"q001": judgement})
    assert quoted is not None
    assert quoted["check"] == {"run": "a check", "answers": None, "agreement": {}}
