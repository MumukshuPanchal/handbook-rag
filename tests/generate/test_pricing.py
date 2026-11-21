import pytest

from handbook_rag.generate.models import TokenUsage
from handbook_rag.generate.pricing import cost_usd


def test_cost_is_input_and_output_tokens_at_the_model_price() -> None:
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.60)
    assert cost_usd("claude-opus-5-5", usage) == pytest.approx(24.00)


def test_cache_writes_cost_more_and_cache_reads_cost_less_than_input() -> None:
    written = TokenUsage(cache_creation_input_tokens=1_000_000)
    read = TokenUsage(cache_read_input_tokens=1_000_000)

    assert cost_usd("claude-sonnet-5-5", written) == pytest.approx(2.50)
    assert cost_usd("claude-sonnet-5-5", read) == pytest.approx(0.20)


def test_a_typical_answer_on_haiku_costs_a_fraction_of_a_cent() -> None:
    cost = cost_usd("claude-haiku-5-5", TokenUsage(input_tokens=3_000, output_tokens=400))

    assert cost == pytest.approx(0.0005)


def test_unknown_model_has_no_price() -> None:
    assert cost_usd("some-other-model", TokenUsage(input_tokens=10)) is None
