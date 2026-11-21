"""Turn token counts into dollars.

Cost is always derived from the token usage the API reports and the model that actually
served the request, never from an estimate.
"""

from handbook_rag.generate.models import TokenUsage

# US dollars per million tokens: (input, output). Anthropic list prices as of 2026-11-03.
# Claude Haiku 5.5 costs more for prompts over 100,000 tokens; ours are a few thousand.
_PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
# Tokens written to the prompt cache cost more than normal input; tokens read from it
# cost much less.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10


def cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Cost of one request in US dollars, or ``None`` for a model with no known price."""
    prices = next(
        (price for name, price in _PRICES_PER_MILLION.items() if model.startswith(name)), None
    )
    if prices is None:
        return None
    input_price, output_price = prices
    input_cost = (
        usage.input_tokens
        + usage.cache_creation_input_tokens * _CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * _CACHE_READ_MULTIPLIER
    ) * input_price
    return (input_cost + usage.output_tokens * output_price) / 1_000_000
