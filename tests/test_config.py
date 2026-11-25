import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from handbook_rag.config import Settings, get_settings


def test_defaults_work_without_any_configuration() -> None:
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.log_level == "INFO"


def test_environment_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_DATA_DIR", "/tmp/handbook")
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.data_dir == Path("/tmp/handbook")
    assert settings.log_level == "DEBUG"


def test_dotenv_file_is_read() -> None:
    Path(".env").write_text("HANDBOOK_RAG_LOG_LEVEL=WARNING\n")

    assert Settings().log_level == "WARNING"


def test_environment_variable_wins_over_dotenv_file(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text("HANDBOOK_RAG_LOG_LEVEL=WARNING\n")
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "ERROR")

    assert Settings().log_level == "ERROR"


def test_unrelated_keys_in_dotenv_file_are_ignored() -> None:
    Path(".env").write_text("SOME_OTHER_TOOL_KEY=value\nHANDBOOK_RAG_LOG_LEVEL=DEBUG\n")

    assert Settings().log_level == "DEBUG"


def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_LOG_LEVEL", "LOUD")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_returns_the_same_object_each_time() -> None:
    assert get_settings() is get_settings()


def test_handbook_is_pinned_to_a_full_commit_hash_by_default() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", Settings().handbook_commit)


def test_collection_name_identifies_corpus_model_and_chunking() -> None:
    assert Settings().collection_name == (
        "handbook-e645aac6-baai-bge-small-en-v1-5-qdrant-bm25-v1-max320-min60-overlap40"
    )


def test_collection_name_changes_with_any_of_its_parts() -> None:
    default = Settings().collection_name

    assert Settings(handbook_commit="b" * 40).collection_name != default
    assert Settings(embedding_model="BAAI/bge-base-en-v1.5").collection_name != default
    assert Settings(chunk_max_tokens=200).collection_name != default
    assert Settings(sparse_model="Qdrant/bm42").collection_name != default


def test_search_is_hybrid_and_not_reranked_unless_configured() -> None:
    from handbook_rag.config import Settings

    settings = Settings()

    assert settings.retrieval_mode == "hybrid"
    assert settings.rerank is False


def test_answers_are_written_by_an_exact_gemini_model_by_default() -> None:
    model = Settings().answer_model

    assert model.startswith("gemini-")
    # An alias such as gemini-flash-latest would change models without notice.
    assert "latest" not in model


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "claude-haiku-5-5"])
def test_answer_model_may_be_a_gemini_or_a_claude_model(model: str) -> None:
    assert Settings(answer_model=model).answer_model == model


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "",
        "Gemini-3.8-flash",
        "models/gemini-3.8-flash",
        "geminiX",  # the dash is part of the name
        "gemini-",  # a provider and no model
        "claude-",
    ],
)
def test_answer_model_from_an_unknown_provider_is_rejected(model: str) -> None:
    with pytest.raises(ValidationError, match=r"must be a Gemini model .* or a Claude model"):
        Settings(answer_model=model)


@pytest.mark.parametrize("model", ["gemini-flash-latest", "gemini-flash-lite-latest"])
def test_answer_model_must_be_an_exact_name_not_an_alias(model: str) -> None:
    with pytest.raises(ValidationError, match="must be an exact model name"):
        Settings(answer_model=model)


def test_the_model_is_given_five_chunks_by_default() -> None:
    assert Settings().answer_context_chunks == 5


def test_an_evaluation_sends_ten_requests_a_minute_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().eval_requests_per_minute == 10

    monkeypatch.setenv("HANDBOOK_RAG_EVAL_REQUESTS_PER_MINUTE", "4.5")

    assert Settings().eval_requests_per_minute == 4.5


@pytest.mark.parametrize("rate", [0, -1, 601])
def test_the_evaluation_pace_must_be_a_usable_rate(rate: float) -> None:
    with pytest.raises(ValidationError):
        Settings(eval_requests_per_minute=rate)


def test_the_judge_is_the_answer_model_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().judge_model == "gemini-3.5-flash-lite"
    assert Settings().judge_max_tokens == 2000

    monkeypatch.setenv("HANDBOOK_RAG_JUDGE_MODEL", "gemini-9-judge")

    assert Settings().judge_model == "gemini-9-judge"


@pytest.mark.parametrize(
    "model", ["claude-haiku-5-5", "gpt-9", "gemini-", "gemini", "gemini-flash-latest", ""]
)
def test_the_judge_must_be_an_exact_gemini_model(model: str) -> None:
    with pytest.raises(ValidationError):
        Settings(judge_model=model)


def test_an_answer_request_may_be_silent_for_20_seconds_and_tried_again_for_30() -> None:
    settings = Settings()

    assert settings.answer_timeout_seconds == 20.0
    assert settings.answer_deadline_seconds == 30.0
    assert settings.answer_fallback_model is None


def test_the_time_limits_and_the_fallback_can_be_set_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_DEADLINE_SECONDS", "45")
    monkeypatch.setenv("HANDBOOK_RAG_ANSWER_FALLBACK_MODEL", "gemini-3.8-flash")

    settings = Settings()

    assert settings.answer_timeout_seconds == 12.5
    assert settings.answer_deadline_seconds == 45.0
    assert settings.answer_fallback_model == "gemini-3.8-flash"


@pytest.mark.parametrize("field", ["answer_timeout_seconds", "answer_deadline_seconds"])
@pytest.mark.parametrize("seconds", [0, -1, 600.5, 20_000, float("inf"), float("nan")])
def test_a_time_limit_must_be_a_number_of_seconds_up_to_ten_minutes(
    field: str, seconds: float
) -> None:
    # 20,000 is twenty seconds written in milliseconds, as the Gemini SDK counts them.
    options: dict[str, Any] = {field: seconds}
    with pytest.raises(ValidationError, match=field):
        Settings(**options)


@pytest.mark.parametrize("field", ["answer_timeout_seconds", "answer_deadline_seconds"])
@pytest.mark.parametrize("seconds", [0.5, 600])
def test_a_time_limit_from_half_a_second_to_ten_minutes_is_accepted(
    field: str, seconds: float
) -> None:
    options: dict[str, Any] = {field: seconds}
    assert getattr(Settings(**options), field) == seconds


def test_a_deadline_shorter_than_the_time_limit_of_one_request_is_allowed() -> None:
    # The deadline only stops new attempts. With one this short, only a quick failure is
    # tried again, and a silent request still has its whole time limit.
    settings = Settings(answer_timeout_seconds=20, answer_deadline_seconds=5)

    assert (settings.answer_timeout_seconds, settings.answer_deadline_seconds) == (20, 5)


@pytest.mark.parametrize("model", ["gemini-3.8-flash", "claude-haiku-5-5"])
def test_the_fallback_may_be_a_gemini_or_a_claude_model(model: str) -> None:
    assert Settings(answer_fallback_model=model).answer_fallback_model == model


def test_an_empty_fallback_copied_from_the_example_file_means_none() -> None:
    Path(".env").write_text("HANDBOOK_RAG_ANSWER_FALLBACK_MODEL=\n")

    assert Settings().answer_fallback_model is None
    assert Settings(answer_fallback_model="").answer_fallback_model is None


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("gpt-4o", r"must be a Gemini model .* or a Claude model"),
        ("gemini-", r"must be a Gemini model .* or a Claude model"),
        ("gemini-flash-latest", "must be an exact model name"),
    ],
)
def test_the_fallback_is_named_by_the_same_rule_as_the_answer_model(
    model: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(answer_fallback_model=model)


def test_the_fallback_must_be_another_model_than_the_answer_model() -> None:
    with pytest.raises(ValidationError, match="must be another model than answer_model"):
        Settings(answer_model="gemini-3.8-flash", answer_fallback_model="gemini-3.8-flash")


def test_the_collection_of_another_commit_has_these_models_and_this_chunking() -> None:
    settings = Settings(embedding_model="BAAI/bge-base-en-v1.5", chunk_max_tokens=200)

    assert settings.collection_for("c" * 40) == (
        "handbook-cccccccc-baai-bge-base-en-v1-5-qdrant-bm25-v1-max200-min60-overlap40"
    )
    assert settings.collection_for(settings.handbook_commit) == settings.collection_name


def test_the_answer_cache_matches_by_meaning_unless_told_otherwise() -> None:
    settings = Settings()

    assert settings.cache_mode == "semantic", "the project owner's decision (ADR 0020)"
    assert settings.cache_threshold == 0.91
    assert (settings.cache_candidates, settings.cache_floor) == (3, 0.7)
    assert settings.cache_verifier_model == "cross-encoder/nli-deberta-v3-small"


def test_the_answer_cache_can_be_set_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDBOOK_RAG_CACHE_MODE", "exact")
    monkeypatch.setenv("HANDBOOK_RAG_CACHE_THRESHOLD", "0.95")
    monkeypatch.setenv("HANDBOOK_RAG_CACHE_CANDIDATES", "5")
    monkeypatch.setenv("HANDBOOK_RAG_CACHE_FLOOR", "0.7")

    settings = Settings()

    assert settings.cache_mode == "exact"
    assert (settings.cache_threshold, settings.cache_candidates, settings.cache_floor) == (
        0.95,
        5,
        0.7,
    )
    assert Settings(cache_mode="off").cache_mode == "off"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache_mode", "on"),
        ("cache_mode", "fuzzy"),
        ("cache_mode", ""),
        # A threshold of 0 would serve the nearest stored question whatever the verifier says.
        ("cache_threshold", 0),
        ("cache_threshold", -0.1),
        ("cache_threshold", 1.01),
        ("cache_threshold", float("nan")),
        ("cache_candidates", 0),
        ("cache_candidates", 11),
        ("cache_floor", -1.01),
        ("cache_floor", 1.01),
    ],
)
def test_a_cache_setting_that_makes_no_sense_is_rejected(field: str, value: object) -> None:
    options: dict[str, Any] = {field: value}
    with pytest.raises(ValidationError, match=field):
        Settings(**options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache_threshold", 1.0),
        ("cache_threshold", 0.001),
        ("cache_candidates", 1),
        ("cache_candidates", 10),
        ("cache_floor", -1.0),
        ("cache_floor", 1.0),
    ],
)
def test_a_cache_setting_at_the_edge_of_its_range_is_accepted(field: str, value: float) -> None:
    options: dict[str, Any] = {field: value}
    assert getattr(Settings(**options), field) == value


def test_the_example_file_lists_the_cache_settings_with_their_defaults() -> None:
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    settings = Settings()

    for line in (
        f"HANDBOOK_RAG_CACHE_MODE={settings.cache_mode}",
        f"HANDBOOK_RAG_CACHE_THRESHOLD={settings.cache_threshold}",
        f"HANDBOOK_RAG_CACHE_CANDIDATES={settings.cache_candidates}",
        f"HANDBOOK_RAG_CACHE_FLOOR={settings.cache_floor}",
        f"HANDBOOK_RAG_CACHE_VERIFIER_MODEL={settings.cache_verifier_model}",
    ):
        assert f"\n{line}\n" in example, line


def test_the_example_file_lists_the_follow_up_settings_with_their_defaults() -> None:
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    settings = Settings()

    for line in (
        f"HANDBOOK_RAG_FOLLOW_UPS={str(settings.follow_ups).lower()}",
        f"HANDBOOK_RAG_FOLLOW_UP_EARLIER_QUESTIONS={settings.follow_up_earlier_questions}",
        f"HANDBOOK_RAG_FOLLOW_UP_TIMEOUT_SECONDS={settings.follow_up_timeout_seconds:g}",
    ):
        assert f"\n{line}\n" in example, line


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("follow_up_earlier_questions", 0),
        ("follow_up_earlier_questions", 7),
        ("follow_up_timeout_seconds", 0),
        ("follow_up_timeout_seconds", 15000),
    ],
)
def test_a_follow_up_setting_out_of_its_range_is_refused(field: str, value: float) -> None:
    options: dict[str, Any] = {field: value}

    with pytest.raises(ValidationError):
        Settings(**options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("follow_up_earlier_questions", 1),
        ("follow_up_earlier_questions", 6),
        ("follow_up_timeout_seconds", 0.001),
        ("follow_up_timeout_seconds", 600),
    ],
)
def test_a_follow_up_setting_at_the_edge_of_its_range_is_accepted(field: str, value: float) -> None:
    options: dict[str, Any] = {field: value}
    assert getattr(Settings(**options), field) == value


def test_a_follow_up_time_limit_just_over_ten_minutes_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(follow_up_timeout_seconds=600.1)


def test_the_step_reads_no_more_earlier_questions_than_it_can_send() -> None:
    from handbook_rag.rewrite.models import MAX_EARLIER_TURNS

    field = Settings.model_fields["follow_up_earlier_questions"]
    assert any(getattr(rule, "le", None) == MAX_EARLIER_TURNS for rule in field.metadata)
