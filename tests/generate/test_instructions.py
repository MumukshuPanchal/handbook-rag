"""The versions of the answering instructions, and who may answer with which.

The product answers with the current version. An earlier version is kept so that an
evaluation can ask the old and the new assistant on the same day, and only an evaluation
may build a generator with it. While there is one version, the tests that need another
register a made-up one beside it: whatever is not the current version is refused the same
way.
"""

import ast
from pathlib import Path
from types import MappingProxyType
from typing import cast

import anthropic
import pytest
from google import genai
from pydantic import SecretStr

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.generate import claude, gemini, prompt
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPTS
from handbook_rag.generate.prompt import CURRENT_INSTRUCTIONS, SYSTEM_PROMPTS, known_instructions
from handbook_rag.generate.retrying import RetryingGenerator
from handbook_rag.measuring import OldInstructionsNotAllowed, measuring_the_model
from handbook_rag.wiring import describe_generation, make_generator
from tests.generate.test_gemini import LEAVE, MODEL, FakeClient, answer_pieces

SOURCE = Path(__file__).parents[2] / "src" / "handbook_rag"
KEY = SecretStr("test-key-not-a-real-one")
OTHER_TEXT = "Instructions of another version."
OTHER_MARKERS = "Instructions of another version, with markers."


@pytest.fixture
def other(monkeypatch: pytest.MonkeyPatch) -> int:
    """A version beside the current one, in every place that reads the versions."""
    version = max(SYSTEM_PROMPTS) + 1
    plain = MappingProxyType({**SYSTEM_PROMPTS, version: OTHER_TEXT})
    markers = MappingProxyType({**MARKER_SYSTEM_PROMPTS, version: OTHER_MARKERS})
    for module in (prompt, claude, wiring):
        monkeypatch.setattr(module, "SYSTEM_PROMPTS", plain)
    for module in (gemini, wiring):
        monkeypatch.setattr(module, "MARKER_SYSTEM_PROMPTS", markers)
    return version


def gemini_generator(instructions: int) -> GeminiGenerator:
    return GeminiGenerator(
        cast(genai.Client, FakeClient(answer_pieces("Sixteen weeks [1]."))),
        model=MODEL,
        max_tokens=4000,
        instructions=instructions,
    )


def claude_generator(instructions: int) -> ClaudeGenerator:
    return ClaudeGenerator(
        anthropic.Anthropic(api_key="sk-ant-test-not-a-real-key"),
        model="claude-haiku-5-5",
        max_tokens=4000,
        effort="low",
        instructions=instructions,
    )


def test_every_version_has_instructions_for_both_ways_of_citing() -> None:
    assert set(MARKER_SYSTEM_PROMPTS) == set(SYSTEM_PROMPTS)
    assert CURRENT_INSTRUCTIONS in SYSTEM_PROMPTS
    assert known_instructions(CURRENT_INSTRUCTIONS) == CURRENT_INSTRUCTIONS


def test_a_version_that_does_not_exist_is_refused_wherever_it_is_named() -> None:
    missing = max(SYSTEM_PROMPTS) + 1
    for build in (
        lambda: known_instructions(missing),
        lambda: gemini_generator(missing),
        lambda: claude_generator(missing),
        lambda: describe_generation(Settings(), missing),
        lambda: make_generator(Settings(gemini_api_key=KEY), attempts=1, instructions=missing),
    ):
        with pytest.raises(ValueError, match=f"no answering instructions of version {missing}"):
            build()


def test_the_current_instructions_are_what_a_generator_sends_when_none_is_named() -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))
    generator = GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000)

    generator.generate("How long is parental leave?", [LEAVE])

    assert generator.instructions == CURRENT_INSTRUCTIONS
    (call,) = client.models.calls
    assert call["config"].system_instruction == MARKER_SYSTEM_PROMPTS[CURRENT_INSTRUCTIONS]
    assert claude_generator(CURRENT_INSTRUCTIONS).instructions == CURRENT_INSTRUCTIONS


def test_other_instructions_cannot_be_built_outside_an_evaluation(other: int) -> None:
    for build in (gemini_generator, claude_generator):
        with pytest.raises(OldInstructionsNotAllowed, match="for evaluations only"):
            build(other)
        with measuring_the_model():
            assert build(other).instructions == other


def test_a_generator_of_another_version_sends_that_version_s_instructions(other: int) -> None:
    client = FakeClient(answer_pieces("Sixteen weeks [1]."))
    with measuring_the_model():
        generator = GeminiGenerator(
            cast(genai.Client, client), model=MODEL, max_tokens=4000, instructions=other
        )
    generator.generate("How long is parental leave?", [LEAVE])

    (call,) = client.models.calls
    assert call["config"].system_instruction == OTHER_MARKERS


def test_make_generator_gives_other_instructions_to_an_evaluation_only(other: int) -> None:
    for settings in (
        Settings(gemini_api_key=KEY),
        Settings(answer_model="claude-haiku-5-5", anthropic_api_key=SecretStr("sk-ant-test")),
    ):
        # Not to the generator that the service and the ask command build, even inside
        # an evaluation: only one attempt for each request is an evaluation's.
        for attempts in (None, 3):
            with measuring_the_model(), pytest.raises(ValueError, match="for an evaluation"):
                make_generator(settings, attempts=attempts, instructions=other)
        with pytest.raises(OldInstructionsNotAllowed):
            make_generator(settings, attempts=1, instructions=other)
        with measuring_the_model():
            built = make_generator(settings, attempts=1, instructions=other)
        assert isinstance(built, GeminiGenerator | ClaudeGenerator)
        assert built.instructions == other
        # Without a version named, the product's own.
        product = make_generator(settings)
        assert isinstance(product, RetryingGenerator)
        own = product._answer_model
        assert isinstance(own, GeminiGenerator | ClaudeGenerator)
        assert own.instructions == CURRENT_INSTRUCTIONS


def test_version_1_is_recorded_as_every_run_before_it_and_another_by_its_number(
    other: int,
) -> None:
    first = describe_generation(Settings(), 1)
    later = describe_generation(Settings(), other)

    assert "instructions_version" not in first
    assert later["instructions_version"] == other
    assert later["instructions"] != first["instructions"]
    assert {key: value for key, value in later.items() if key != "instructions_version"}.keys() == (
        first.keys()
    )
    assert wiring.answer_instructions(Settings(), other) == OTHER_MARKERS
    assert wiring.answer_instructions(Settings(answer_model="claude-haiku-5-5"), other) == (
        OTHER_TEXT
    )
    # The answer cache is named by the instructions the product answers with.
    assert (
        wiring.describe_answer_cache(Settings())["instructions"]
        == (describe_generation(Settings(), CURRENT_INSTRUCTIONS)["instructions"])
    )


def calls_naming_instructions(tree: ast.Module) -> set[str]:
    """The top-level functions that call make_generator with an ``instructions`` keyword."""
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute | ast.Name)
                and (call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id)
                == "make_generator"
                and any(keyword.arg in ("instructions", None) for keyword in call.keywords)
            ):
                found.add(node.name)
    return found


def test_only_evaluation_commands_name_the_instructions_when_they_build_an_answer_model() -> None:
    """A second net under the generator's own refusal: the service, the ask command and the
    MCP server never pass a version, so they get the current one."""
    naming = {
        (str(path.relative_to(SOURCE)), name)
        for path in SOURCE.rglob("*.py")
        for name in calls_naming_instructions(ast.parse(path.read_text(encoding="utf-8")))
    }

    assert naming == {
        ("cli.py", "eval_answers"),
        ("cli.py", "eval_adversarial"),
        ("cli.py", "eval_limits"),
    }
    from handbook_rag.cli import eval_adversarial, eval_answers, eval_limits

    for command in (eval_answers, eval_adversarial, eval_limits):
        assert getattr(command, "measures_the_model", False), command.__name__


def test_the_first_instructions_are_kept_for_evaluations_only() -> None:
    settings = Settings(gemini_api_key=KEY)

    assert 1 in SYSTEM_PROMPTS
    assert CURRENT_INSTRUCTIONS > 1
    with pytest.raises(OldInstructionsNotAllowed, match="version 1 are kept for evaluations only"):
        make_generator(settings, attempts=1, instructions=1)
    with pytest.raises(ValueError, match="for an evaluation"):
        make_generator(settings, instructions=1)
    with measuring_the_model():
        old = make_generator(settings, attempts=1, instructions=1)
    assert isinstance(old, GeminiGenerator)
    assert old.instructions == 1
