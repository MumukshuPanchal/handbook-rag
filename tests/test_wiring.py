import dataclasses
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import anthropic
import pytest
from google.genai import types as genai_types
from pydantic import SecretStr

from handbook_rag import wiring
from handbook_rag.cache.answers import ANSWER_CACHE_VERSION, AnswerCache
from handbook_rag.cache.store import COLLECTION_PREFIX, QdrantAnswerStore
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
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT, build_numbered_prompt
from handbook_rag.generate.models import Answer, Segment
from handbook_rag.generate.prompt import CURRENT_INSTRUCTIONS, SYSTEM_PROMPT
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.index.store import QdrantVectorStore
from handbook_rag.wiring import (
    answer_cache_collection,
    answer_request_fingerprint,
    describe_answer_cache,
    describe_generation,
    describe_judge,
    make_answer_cache,
    make_generator,
    make_judge,
)
from tests.cache.support import TableVerifier
from tests.support import FakeEmbedder, FakeSparseEncoder

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

    monkeypatch.setattr(
        wiring, "MARKER_SYSTEM_PROMPTS", {CURRENT_INSTRUCTIONS: MARKER_SYSTEM_PROMPT + " Be kind."}
    )

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


# The answer cache. Its collection is named by everything that decides what the right
# answer to a question is, so that an answer never outlives what it was written from.

# Each setting that decides an answer under the default settings, with another value.
DECIDES_AN_ANSWER: dict[str, Any] = {
    "handbook_repo_url": "https://gitlab.example.com/another/handbook.git",
    "handbook_commit": "b" * 40,
    "handbook_content_path": "content/other",
    "handbook_base_url": "https://handbook.example.com/",
    "min_page_chars": 100,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "embedding_query_prefix": "query: ",
    "sparse_model": "Qdrant/bm42-all-minilm-l6-v2-attentions",
    "chunk_max_tokens": 200,
    "chunk_min_tokens": 50,
    "chunk_overlap_tokens": 20,
    "retrieval_mode": "dense",
    "fusion_candidates": 20,
    "rrf_k": 30,
    "rerank": True,
    "answer_model": "gemini-3.8-flash",
    "answer_max_tokens": 2000,
    "answer_context_chunks": 3,
}
# Each setting that decides an answer only while another setting has a certain value:
# that other setting, a value of its own, and the setting under which it decides nothing.
DECIDES_UNDER: dict[str, tuple[dict[str, Any], Any, dict[str, Any]]] = {
    "rerank_model": ({"rerank": True}, "cross-encoder/another-reranker", {"rerank": False}),
    "rerank_candidates": ({"rerank": True}, 10, {"rerank": False}),
    "answer_effort": ({"answer_model": CLAUDE}, "high", {}),
}
# Each setting that decides only whether an answer is served, how fast, or where things
# are kept, with another value. An answer stored under one value is right under the other.
DECIDES_NO_ANSWER: dict[str, Any] = {
    "data_dir": Path("elsewhere"),
    "log_level": "DEBUG",
    "handbook_licence": "MIT",
    "embedding_max_tokens": 256,
    "embedding_batch_size": 8,
    "embedding_device": "cpu",
    "qdrant_url": "http://another-host:6333",
    "search_limit": 9,
    "cache_mode": "exact",
    "cache_threshold": 0.99,
    "cache_candidates": 5,
    "cache_floor": 0.6,
    "cache_verifier_model": "another/nli-model",
    "api_host": "0.0.0.0",
    "api_port": 9000,
    "api_max_answers_in_flight": 2,
    "api_demo": False,
    "tracing": True,
    "otlp_endpoint": "http://another-host:4318",
    "service_name": "another-name",
    "trace_content": True,
    "gemini_api_key": SecretStr("another-key-not-a-real-one"),
    "anthropic_api_key": SecretStr("another-key-not-a-real-one"),
    "answer_timeout_seconds": 10.0,
    "answer_deadline_seconds": 20.0,
    # Its answers are never stored (cache/answers.py).
    "answer_fallback_model": "gemini-3.8-flash",
    "eval_requests_per_minute": 5.0,
    # They decide which question is searched for, and the cache is keyed on that question:
    # an answer stored for it is its answer under any of them (ADR 0024).
    "follow_ups": False,
    "follow_up_earlier_questions": 5,
    "follow_up_timeout_seconds": 5.0,
    "judge_model": "gemini-3.8-flash",
    "judge_max_tokens": 1000,
}


def test_every_setting_is_listed_as_deciding_an_answer_or_not() -> None:
    """A setting added later has to be put in one of the three lists above.

    That is the moment to ask whether an answer stored before it changed is still right,
    and, if it is not, to add it to what names the cache's collection.
    """
    listed = [*DECIDES_AN_ANSWER, *DECIDES_UNDER, *DECIDES_NO_ANSWER]

    assert len(listed) == len(set(listed)), "a setting is in one list only"
    assert set(listed) == set(Settings.model_fields)


@pytest.mark.parametrize("setting", DECIDES_AN_ANSWER)
def test_a_setting_that_decides_an_answer_changes_the_name_of_the_caches_collection(
    setting: str,
) -> None:
    changed = Settings(**{setting: DECIDES_AN_ANSWER[setting]})

    assert getattr(changed, setting) != getattr(Settings(), setting)
    assert answer_cache_collection(changed) != answer_cache_collection(Settings())


@pytest.mark.parametrize("setting", DECIDES_UNDER)
def test_a_setting_that_decides_an_answer_under_another_changes_the_name_only_then(
    setting: str,
) -> None:
    under, value, otherwise = DECIDES_UNDER[setting]

    assert answer_cache_collection(Settings(**under, **{setting: value})) != (
        answer_cache_collection(Settings(**under))
    )
    # Where the search or the model does not use it, a stored answer is still right.
    assert answer_cache_collection(Settings(**otherwise, **{setting: value})) == (
        answer_cache_collection(Settings(**otherwise))
    )


@pytest.mark.parametrize("setting", ["fusion_candidates", "rrf_k"])
def test_the_settings_of_merging_two_searches_decide_nothing_when_one_search_is_used(
    setting: str,
) -> None:
    dense = Settings(retrieval_mode="dense")
    changed = Settings(retrieval_mode="dense", **{setting: DECIDES_AN_ANSWER[setting]})

    assert answer_cache_collection(changed) == answer_cache_collection(dense)


@pytest.mark.parametrize("setting", DECIDES_NO_ANSWER)
def test_a_setting_that_decides_no_answer_leaves_the_name_as_it_is(setting: str) -> None:
    changed = Settings(**{setting: DECIDES_NO_ANSWER[setting]})

    assert getattr(changed, setting) != getattr(Settings(), setting)
    assert answer_cache_collection(changed) == answer_cache_collection(Settings())


def test_the_caches_collection_is_named_for_a_person_to_read_and_apart_from_the_handbooks() -> None:
    settings = Settings()
    name = answer_cache_collection(settings)

    assert re.fullmatch(r"answers-e645aac6-[0-9a-f]{16}", name)
    assert name == answer_cache_collection(Settings()), "the same settings, the same name"
    assert name != settings.collection_name
    assert name.startswith(COLLECTION_PREFIX)
    assert not settings.collection_name.startswith(COLLECTION_PREFIX)
    # What is hashed names each thing by its own word, for whoever wonders why a name changed.
    described = describe_answer_cache(settings)
    assert described["cache_version"] == ANSWER_CACHE_VERSION
    assert described["handbook_commit"] == settings.handbook_commit, "whole, not its first eight"
    assert described["collection"] == settings.collection_name
    assert described["instructions"] == describe_generation(settings)["instructions"]
    assert described["request"] == answer_request_fingerprint(settings)
    json.dumps(described)


def test_other_instructions_another_layout_of_the_request_or_a_new_version_give_another_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini, claude = Settings(), Settings(answer_model=CLAUDE)
    before = (answer_cache_collection(gemini), answer_cache_collection(claude))

    def renamed() -> tuple[bool, bool]:
        now = (answer_cache_collection(gemini), answer_cache_collection(claude))
        return (now[0] != before[0], now[1] != before[1])

    with monkeypatch.context() as patch:
        patch.setattr(
            wiring,
            "MARKER_SYSTEM_PROMPTS",
            {CURRENT_INSTRUCTIONS: MARKER_SYSTEM_PROMPT + "\n- Be brief."},
        )
        assert renamed() == (True, False), "the instructions a Gemini model is given"
    with monkeypatch.context() as patch:
        patch.setattr(
            wiring, "SYSTEM_PROMPTS", {CURRENT_INSTRUCTIONS: SYSTEM_PROMPT + "\n- Be brief."}
        )
        assert renamed() == (False, True), "the instructions a Claude model is given"
    with monkeypatch.context() as patch:
        patch.setattr(wiring, "build_numbered_prompt", lambda question, chunks: "laid out anew")
        assert renamed() == (True, False)
    with monkeypatch.context() as patch:
        patch.setattr(wiring, "build_user_content", lambda question, chunks: ["laid out anew"])
        assert renamed() == (False, True)
    with monkeypatch.context() as patch:
        patch.setattr(wiring, "ANSWER_CACHE_VERSION", ANSWER_CACHE_VERSION + 1)
        assert renamed() == (True, True)
    assert renamed() == (False, False)


def test_the_layout_of_a_request_is_named_from_the_text_a_model_would_be_sent() -> None:
    sample = build_numbered_prompt(wiring._SAMPLE_QUESTION, wiring._SAMPLE_PASSAGES)

    # The sample holds what the layout has to make harmless, so a change to that shows.
    assert "&lt;/search_result>" in sample
    assert sample.count('<search_result number="') == 2
    assert sample.endswith("Question: A question?")
    assert len(answer_request_fingerprint(Settings())) == 16
    assert answer_request_fingerprint(Settings()) != (
        answer_request_fingerprint(Settings(answer_model=CLAUDE))
    )


def stores() -> QdrantVectorStore:
    return wiring.make_memory_store("handbook")


def test_the_cache_is_built_for_the_mode_of_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[Settings] = []
    verifier = TableVerifier()

    def make_verifier(settings: Settings) -> TableVerifier:
        built.append(settings)
        return verifier

    monkeypatch.setattr(wiring, "make_verifier", make_verifier)
    store, embedder = stores(), FakeEmbedder()

    def build(**settings: Any) -> AnswerCache | None:
        return make_answer_cache(Settings(**settings), store=store, embedder=embedder)

    assert build(cache_mode="off") is None
    exact = build(cache_mode="exact")
    assert exact is not None
    assert exact.mode == "exact"
    chosen = Settings(cache_threshold=0.95, cache_candidates=2, cache_floor=0.6)
    semantic = make_answer_cache(chosen, store=store, embedder=embedder)
    assert semantic is not None
    assert semantic.mode == "semantic", "the default"
    assert (semantic._threshold, semantic._candidates, semantic._floor) == (0.95, 2, 0.6)
    assert semantic._embedder is embedder
    # In a collection of its own, in the same store as the handbook's.
    kept = semantic._store
    assert isinstance(kept, QdrantAnswerStore)
    assert kept.collection == answer_cache_collection(chosen)
    assert kept._client is store.client
    # The verifier is not built until it is first asked.
    assert built == []
    semantic.warm_up()
    semantic.warm_up()
    assert built == [chosen], "once, with the settings that name its model"
    assert len(verifier.calls) == 2
    assert semantic.close() is True


def test_the_retriever_and_the_cache_share_one_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []
    embedder = FakeEmbedder()

    def make_embedder(_settings: Settings) -> FakeEmbedder:
        built.append("embedder")
        return embedder

    monkeypatch.setattr(wiring, "make_embedder", make_embedder)
    monkeypatch.setattr(wiring, "make_sparse_encoder", lambda _settings: FakeSparseEncoder())
    store = stores()

    def build(**settings: Any) -> tuple[AnswerCache | None, int]:
        built.clear()
        _, cache = wiring.make_retriever_and_cache(Settings(**settings), store)
        return cache, len(built)

    both, loads = build()
    assert both is not None
    assert (both.mode, loads) == ("semantic", 1), "one model for the search and the cache"
    assert both._embedder is embedder
    # Keyword search loads no embedding model. A cache needs one for its own vectors.
    assert build(retrieval_mode="sparse", cache_mode="off") == (None, 0)
    keyword, loads = build(retrieval_mode="sparse", cache_mode="exact")
    assert keyword is not None
    assert (keyword.mode, loads) == ("exact", 1)
    # Switched off, there is no cache, and the search loads what it needs itself.
    assert build(cache_mode="off") == (None, 1)


def test_the_cache_the_settings_build_stores_an_answer_without_making_its_caller_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command and the service both get this cache. Stored in the foreground, an
    answer would wait for the store before it reached whoever asked."""
    release, writing, stored = threading.Event(), threading.Event(), []

    def slow_put(self: QdrantAnswerStore, entry: Any, vector: Any) -> None:
        writing.set()
        assert release.wait(5.0), "the test never let the store go on"
        stored.append(entry.question)

    monkeypatch.setattr(QdrantAnswerStore, "put", slow_put)
    cache = make_answer_cache(Settings(cache_mode="exact"), store=stores(), embedder=FakeEmbedder())
    assert cache is not None
    answer = Answer(
        question="How long is leave?",
        segments=(Segment("Sixteen weeks."),),
        answered=True,
        model="stand-in",
        stop_reason="end_turn",
    )

    cache.store(cache.lookup("How long is leave?"), answer)

    assert writing.wait(5.0), "the write has begun, in a thread of its own"
    assert stored == [], "and store() came back before it ended"
    release.set()
    assert cache.close() is True
    assert stored == ["How long is leave?"]


def test_the_commit_of_a_caches_name_is_how_every_cache_of_that_commit_is_found() -> None:
    settings = Settings()
    names = {
        answer_cache_collection(settings),
        answer_cache_collection(Settings(rerank=True)),
        answer_cache_collection(Settings(answer_context_chunks=3)),
    }

    assert wiring.answer_cache_commit(settings) == "e645aac6"
    assert len(names) == 3
    for name in names:
        assert name.startswith(f"answers-{wiring.answer_cache_commit(settings)}-")
    # A commit named by a word is held as a collection's name can hold it.
    odd = Settings(handbook_commit="feature/new-pages")
    assert wiring.answer_cache_commit(odd) == "feature"
    assert re.fullmatch(r"answers-feature-[0-9a-f]{16}", answer_cache_collection(odd))
