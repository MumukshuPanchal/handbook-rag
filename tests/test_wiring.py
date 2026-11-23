import dataclasses
import json
from pathlib import Path
from typing import Any

import anthropic
import pytest
from pydantic import SecretStr

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.evals.correctness import (
    CORRECT,
    GeminiCorrectnessJudge,
    correctness_request_fingerprint,
)
from handbook_rag.evals.judge import (
    SUPPORT_INSTRUCTIONS,
    GeminiSupportJudge,
    request_fingerprint,
)
from handbook_rag.evals.judging import Criterion
from handbook_rag.evals.support import SUPPORT
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT
from handbook_rag.wiring import describe_generation, describe_judge, make_generator, make_judge

CLAUDE = "claude-haiku-5-5"


def test_answers_come_from_gemini_unless_configured_otherwise(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"))

    generator = make_generator(settings)

    assert isinstance(generator, GeminiGenerator)
    assert generator._model == "gemini-3.5-flash-lite"
    assert generator._client._api_client.api_key == "test-key-not-a-real-one"
    # The key reaches the client directly, never through the process environment.
    assert "GEMINI_API_KEY" not in monkeypatch_environment
    assert "GOOGLE_API_KEY" not in monkeypatch_environment


def test_gemini_requests_have_a_time_limit_and_retry_only_what_a_retry_can_fix() -> None:
    generator = make_generator(Settings(gemini_api_key=SecretStr("test-key-not-a-real-one")))

    assert isinstance(generator, GeminiGenerator)
    options = generator._client._api_client._http_options
    assert options.timeout == 60_000  # milliseconds
    assert options.retry_options is not None
    assert options.retry_options.attempts == 3
    # A busy or failing server is worth another try. A rate limit (429) is not in the
    # list: on a free key it lasts a minute or a day, and trying again at once only
    # spends another request, so it is reported instead.
    assert options.retry_options.http_status_codes == [408, 500, 502, 503, 504]


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_a_gemini_model_without_a_key_is_a_clear_error(key: SecretStr | None) -> None:
    with pytest.raises(GenerationError, match="GEMINI_API_KEY") as raised:
        make_generator(Settings(gemini_api_key=key))
    # The HTTP API starts without an answer model on this, and serves searches.
    assert raised.value.reason == "configuration"


def test_an_empty_key_copied_from_the_example_file_is_a_clear_error(
    monkeypatch_environment: dict[str, str],
) -> None:
    monkeypatch_environment["GEMINI_API_KEY"] = ""

    with pytest.raises(GenerationError, match="No Gemini API key was found") as raised:
        make_generator(Settings())
    assert raised.value.reason == "configuration"


def test_a_claude_model_name_selects_the_claude_generator(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(
        answer_model=CLAUDE, anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key")
    )

    generator = make_generator(settings)

    assert isinstance(generator, ClaudeGenerator)
    assert generator._model == CLAUDE
    assert generator._client.api_key == "sk-ant-test-not-a-real-key"
    assert generator._client.timeout == 60.0
    assert "ANTHROPIC_API_KEY" not in monkeypatch_environment


def test_claude_generator_can_be_built_without_a_key_in_the_settings() -> None:
    # The Claude SDK then looks in its own standard places, and a Gemini key is not needed.
    generator = make_generator(Settings(answer_model=CLAUDE))

    assert isinstance(generator, ClaudeGenerator)
    assert generator._client.timeout == 60.0


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", CLAUDE])
def test_the_answer_length_limit_comes_from_the_settings(model: str) -> None:
    settings = Settings(
        answer_model=model,
        answer_max_tokens=123,
        gemini_api_key=SecretStr("test-key-not-a-real-one"),
    )

    generator = make_generator(settings)

    assert isinstance(generator, GeminiGenerator | ClaudeGenerator)
    assert generator._max_tokens == 123


@pytest.mark.parametrize(
    ("variable", "field"),
    [("GEMINI_API_KEY", "gemini_api_key"), ("ANTHROPIC_API_KEY", "anthropic_api_key")],
)
def test_keys_are_read_from_their_standard_variables_and_never_shown(
    monkeypatch_environment: dict[str, str], variable: str, field: str
) -> None:
    monkeypatch_environment[variable] = "test-key-not-a-real-one"

    settings = Settings()

    key = getattr(settings, field)
    assert key is not None
    assert key.get_secret_value() == "test-key-not-a-real-one"
    assert "test-key-not-a-real-one" not in repr(settings)
    assert "test-key-not-a-real-one" not in str(settings.model_dump())


def test_an_evaluation_can_ask_for_one_attempt_per_request() -> None:
    generator = make_generator(
        Settings(gemini_api_key=SecretStr("test-key-not-a-real-one")), attempts=1
    )

    assert isinstance(generator, GeminiGenerator)
    options = generator._client._api_client._http_options
    assert options.retry_options is not None
    # One attempt means the request is never sent a second time behind the caller's back.
    assert options.retry_options.attempts == 1


def test_one_attempt_for_a_claude_model_means_no_retries() -> None:
    settings = Settings(
        answer_model=CLAUDE, anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key")
    )

    assert make_generator(settings, attempts=1)._client.max_retries == 0  # type: ignore[attr-defined]
    assert make_generator(settings, attempts=3)._client.max_retries == 2  # type: ignore[attr-defined]
    # Left out, the SDK's own default stands.
    assert make_generator(settings)._client.max_retries == anthropic.DEFAULT_MAX_RETRIES  # type: ignore[attr-defined]


@pytest.mark.parametrize("attempts", [0, -1])
def test_a_request_cannot_be_sent_fewer_than_one_time(attempts: int) -> None:
    with pytest.raises(ValueError, match="at least once"):
        make_generator(Settings(gemini_api_key=SecretStr("k")), attempts=attempts)


def test_a_run_records_what_decides_how_an_answer_looks() -> None:
    described = describe_generation(Settings())

    assert described == {
        "answer_model": "gemini-3.5-flash-lite",
        "answer_max_tokens": 4000,
        "answer_context_chunks": 5,
        "citations": "numbered markers",
        "instructions": described["instructions"],
    }
    assert len(str(described["instructions"])) == 16


def test_reworded_instructions_are_recorded_as_a_different_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = describe_generation(Settings())["instructions"]

    monkeypatch.setattr(wiring, "MARKER_SYSTEM_PROMPT", MARKER_SYSTEM_PROMPT + " Be kind.")

    assert describe_generation(Settings())["instructions"] != before


def test_a_claude_run_records_its_own_instructions_and_effort() -> None:
    gemini = describe_generation(Settings())
    claude = describe_generation(Settings(answer_model=CLAUDE, answer_effort="high"))

    assert claude["citations"] == "from the provider's API"
    assert claude["answer_effort"] == "high"
    assert claude["instructions"] != gemini["instructions"]
    assert "answer_effort" not in gemini


def test_the_judge_is_built_for_the_configured_model_with_the_key_passed_directly() -> None:
    settings = Settings(
        gemini_api_key=SecretStr("test-key-not-a-real-one"),
        judge_model="gemini-9-judge",
        judge_max_tokens=321,
    )

    judge = make_judge(settings, SUPPORT, attempts=1)

    assert isinstance(judge, GeminiSupportJudge)
    assert judge.model == "gemini-9-judge"
    assert judge._max_tokens == 321
    assert judge._client._api_client.api_key == "test-key-not-a-real-one"
    options = judge._client._api_client._http_options
    assert options.timeout == 60_000
    assert options.retry_options is not None
    # One attempt: a run that counts its requests cannot have them sent again unseen.
    assert options.retry_options.attempts == 1
    retried = make_judge(settings, SUPPORT)
    assert retried._client._api_client._http_options.retry_options.attempts == 3  # type: ignore[attr-defined]


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_a_judge_without_a_key_is_a_clear_error(key: SecretStr | None) -> None:
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        make_judge(Settings(gemini_api_key=key), SUPPORT)


def test_a_judge_request_cannot_be_sent_fewer_than_one_time() -> None:
    with pytest.raises(ValueError, match="at least once"):
        make_judge(Settings(gemini_api_key=SecretStr("k")), CORRECT, attempts=0)


def test_verdicts_record_the_judge_that_made_them() -> None:
    described = describe_judge(Settings(), SUPPORT)

    assert described == {
        "judge_model": "gemini-3.5-flash-lite",
        "judge_max_tokens": 2000,
        "judge_instructions": described["judge_instructions"],
        "judge_request": request_fingerprint(),
    }
    assert len(str(described["judge_instructions"])) == 16

    # Reworded instructions are another judge, whose agreement has to be measured again.
    reworded = dataclasses.replace(SUPPORT, instructions=SUPPORT_INSTRUCTIONS + " Be strict.")
    assert (
        describe_judge(Settings(), reworded)["judge_instructions"]
        != described["judge_instructions"]
    )
    other_model = describe_judge(Settings(judge_model="gemini-9-judge"), SUPPORT)
    assert other_model["judge_model"] == "gemini-9-judge"


def test_each_criterion_has_a_judge_of_its_own() -> None:
    settings = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"))

    judge = make_judge(settings, CORRECT, attempts=1)
    support, correct = describe_judge(settings, SUPPORT), describe_judge(settings, CORRECT)

    assert isinstance(judge, GeminiCorrectnessJudge)
    assert judge.model == "gemini-3.5-flash-lite"
    assert judge._max_tokens == 2000
    # Other instructions and another layout of the request: verdicts of one are never
    # taken for verdicts of the other.
    assert support["judge_instructions"] != correct["judge_instructions"]
    assert support["judge_request"] != correct["judge_request"]
    assert correct["judge_request"] == correctness_request_fingerprint()


def test_the_saved_checks_are_of_the_judges_the_code_builds_today() -> None:
    # A check of a judge is kept beside the scores that judge gave. If the instructions,
    # the layout of a request or the model change, the judge is another one, and this
    # fails until it has been checked again and the new check is saved.
    root = Path(__file__).parents[1] / "evals" / "runs"
    criteria: tuple[Criterion[Any], ...] = (SUPPORT, CORRECT)
    for criterion in criteria:
        check = root / criterion.checks_dir.name / "baseline" / "run.json"
        saved = json.loads(check.read_text(encoding="utf-8"))["config"]
        assert saved == describe_judge(Settings(), criterion), criterion.name
        scored = root / "answers" / "baseline" / criterion.state_file
        assert json.loads(scored.read_text(encoding="utf-8"))["config"] == saved, criterion.name
