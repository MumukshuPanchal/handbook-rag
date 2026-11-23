import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import anthropic
import pytest
from google.genai import types as genai_types
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
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT
from handbook_rag.generate.prompt import SYSTEM_PROMPT
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.wiring import describe_generation, describe_judge, make_generator, make_judge

CLAUDE = "claude-haiku-5-5"


def wrapped(generator: Generator) -> Generator:
    """The model's own generator inside the wrapper that the service and the command use."""
    assert isinstance(generator, RetryingGenerator)
    return generator._answer_model


def gemini_options(generator: Generator) -> genai_types.HttpOptions:
    assert isinstance(generator, GeminiGenerator)
    return generator._client._api_client._http_options


def test_answers_come_from_gemini_unless_configured_otherwise(
    monkeypatch_environment: dict[str, str],
) -> None:
    settings = Settings(gemini_api_key=SecretStr("test-key-not-a-real-one"))

    generator = wrapped(make_generator(settings))

    assert isinstance(generator, GeminiGenerator)
    assert generator._model == "gemini-3.5-flash-lite"
    assert generator._client._api_client.api_key == "test-key-not-a-real-one"
    # The key reaches the client directly, never through the process environment.
    assert "GEMINI_API_KEY" not in monkeypatch_environment
    assert "GOOGLE_API_KEY" not in monkeypatch_environment


def test_the_sdk_sends_each_request_of_an_answer_once_with_the_time_limit_of_the_settings() -> None:
    key = SecretStr("test-key-not-a-real-one")

    default = gemini_options(wrapped(make_generator(Settings(gemini_api_key=key))))
    shorter = gemini_options(
        wrapped(make_generator(Settings(gemini_api_key=key, answer_timeout_seconds=7.25)))
    )

    assert default.timeout == 20_000  # milliseconds
    assert shorter.timeout == 7_250
    # The repeats are the wrapper's, where each one is counted and traced.
    for options in (default, shorter):
        assert options.retry_options is not None
        assert options.retry_options.attempts == 1


def test_a_time_limit_too_short_for_a_millisecond_is_still_a_time_limit() -> None:
    settings = Settings(gemini_api_key=SecretStr("k"), answer_timeout_seconds=0.0001)

    # The SDK would read 0 as no time limit at all.
    assert gemini_options(wrapped(make_generator(settings))).timeout == 1


def test_the_wrapper_has_three_attempts_and_the_deadline_of_the_settings() -> None:
    settings = Settings(gemini_api_key=SecretStr("k"), answer_deadline_seconds=12.5)

    generator = make_generator(settings)
    more = make_generator(settings, attempts=5)

    assert isinstance(generator, RetryingGenerator)
    assert (generator._attempts, generator._deadline_seconds) == (3, 12.5)
    assert wiring.ANSWER_ATTEMPTS == 3
    assert isinstance(more, RetryingGenerator)
    assert more._attempts == 5
    assert generator._make_fallback is None, "no fallback unless the settings name one"


@pytest.mark.parametrize(
    ("answer_model", "fallback_model", "kind"),
    [
        ("gemini-3.5-flash-lite", "gemini-3.8-flash", GeminiGenerator),
        ("gemini-3.5-flash-lite", CLAUDE, ClaudeGenerator),
        (CLAUDE, "gemini-3.5-flash-lite", GeminiGenerator),
    ],
)
def test_the_fallback_is_built_only_when_it_is_needed_like_an_answer_model(
    answer_model: str, fallback_model: str, kind: type[Generator]
) -> None:
    settings = Settings(
        answer_model=answer_model,
        answer_fallback_model=fallback_model,
        answer_timeout_seconds=9.0,
        answer_max_tokens=321,
        gemini_api_key=SecretStr("test-key-not-a-real-one"),
        anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key"),
    )

    generator = make_generator(settings)

    assert isinstance(generator, RetryingGenerator)
    assert generator._fallback is None, "not built yet"
    assert generator._make_fallback is not None
    fallback = generator._make_fallback()
    assert isinstance(fallback, kind)
    assert isinstance(fallback, GeminiGenerator | ClaudeGenerator)
    assert (fallback.model, fallback._max_tokens) == (fallback_model, 321)
    if isinstance(fallback, GeminiGenerator):
        options = gemini_options(fallback)
        assert options.timeout == 9_000
        assert options.retry_options is not None
        assert options.retry_options.attempts == 1
    else:
        assert (fallback._client.timeout, fallback._client.max_retries) == (9.0, 0)
        assert fallback._client.api_key == "sk-ant-test-not-a-real-key"


@pytest.mark.parametrize(
    ("answer_model", "fallback_model", "variable"),
    [
        (CLAUDE, "gemini-3.5-flash-lite", "GEMINI_API_KEY"),
        ("gemini-3.5-flash-lite", CLAUDE, "ANTHROPIC_API_KEY"),
    ],
)
def test_a_fallback_without_a_key_is_found_when_the_generator_is_built(
    answer_model: str, fallback_model: str, variable: str
) -> None:
    settings = Settings(
        answer_model=answer_model,
        answer_fallback_model=fallback_model,
        # Only the answer model's key, never the fallback's.
        gemini_api_key=SecretStr("k") if answer_model.startswith("gemini-") else None,
        anthropic_api_key=SecretStr("k") if answer_model.startswith("claude-") else None,
    )

    with pytest.raises(GenerationError, match=variable) as raised:
        make_generator(settings)
    assert raised.value.reason == "configuration"
    assert fallback_model in str(raised.value)
    assert "HANDBOOK_RAG_ANSWER_FALLBACK_MODEL" in str(raised.value)


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", CLAUDE])
def test_the_evaluation_gets_the_models_own_generator_exactly_as_before(model: str) -> None:
    # Whatever the settings say about repeats, deadlines and a fallback.
    settings = Settings(
        answer_model=model,
        answer_fallback_model="gemini-3.8-flash" if model == CLAUDE else CLAUDE,
        answer_timeout_seconds=5.0,
        answer_deadline_seconds=5.0,
        gemini_api_key=SecretStr("test-key-not-a-real-one"),
        anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key"),
    )

    generator = make_generator(settings, attempts=1)

    assert not isinstance(generator, RetryingGenerator)
    if isinstance(generator, GeminiGenerator):
        options = gemini_options(generator)
        assert options.timeout == 60_000
        assert options.retry_options is not None
        assert options.retry_options.attempts == 1
        assert options.retry_options.http_status_codes == [408, 500, 502, 503, 504]
    else:
        assert isinstance(generator, ClaudeGenerator)
        assert (generator._client.timeout, generator._client.max_retries) == (60.0, 0)
        # The SDK's own tracing as the client has it by default: an evaluation does not trace.
        assert generator._client._open_telemetry["traces"]["enabled"] is True
    assert generator.model == model


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

    generator = wrapped(make_generator(settings))

    assert isinstance(generator, ClaudeGenerator)
    assert generator._model == CLAUDE
    assert generator._client.api_key == "sk-ant-test-not-a-real-key"
    # One request for each call of the SDK, with the time limit of the settings.
    assert generator._client.timeout == 20.0
    assert generator._client.max_retries == 0
    # Each request has a span of our own, and the SDK's would count its tokens again.
    assert generator._client._open_telemetry["traces"]["enabled"] is False
    assert "ANTHROPIC_API_KEY" not in monkeypatch_environment


def test_claude_generator_can_be_built_without_a_key_in_the_settings() -> None:
    # The Claude SDK then looks in its own standard places, and a Gemini key is not needed.
    generator = wrapped(make_generator(Settings(answer_model=CLAUDE, answer_timeout_seconds=8.0)))

    assert isinstance(generator, ClaudeGenerator)
    assert generator._client.timeout == 8.0


@pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", CLAUDE])
def test_the_answer_length_limit_comes_from_the_settings(model: str) -> None:
    settings = Settings(
        answer_model=model,
        answer_max_tokens=123,
        gemini_api_key=SecretStr("test-key-not-a-real-one"),
    )

    generator = wrapped(make_generator(settings))

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


def test_the_claude_sdk_never_repeats_a_request_by_itself() -> None:
    settings = Settings(
        answer_model=CLAUDE, anthropic_api_key=SecretStr("sk-ant-test-not-a-real-key")
    )

    assert anthropic.DEFAULT_MAX_RETRIES > 0, "left alone, it would"
    for generator in (
        make_generator(settings, attempts=1),
        wrapped(make_generator(settings, attempts=3)),
        wrapped(make_generator(settings)),
    ):
        assert isinstance(generator, ClaudeGenerator)
        assert generator._client.max_retries == 0


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


def test_the_instructions_of_the_answer_model_are_those_its_generator_sends() -> None:
    gemini, claude = Settings(), Settings(answer_model=CLAUDE)

    # A Gemini model cites by numbered markers, and is told how to write them.
    assert wiring.answer_instructions(gemini) == MARKER_SYSTEM_PROMPT
    assert wiring.answer_instructions(claude) == SYSTEM_PROMPT
    # What is recorded with a run is a hash of exactly these.
    for settings in (gemini, claude):
        expected = hashlib.sha256(wiring.answer_instructions(settings).encode("utf-8"))
        assert describe_generation(settings)["instructions"] == expected.hexdigest()[:16]


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
