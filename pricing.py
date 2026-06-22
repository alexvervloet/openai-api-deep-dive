"""
Cost estimation for OpenAI chat models.
=======================================

OpenAI bills you per *token*, and charges a different rate for tokens you send
(input / "prompt" tokens) versus tokens the model generates back (output /
"completion" tokens). Output tokens are usually several times more expensive
than input tokens, which is why a chatty model can cost more than you expect.

Prices are quoted per 1,000,000 tokens. We store them that way below and divide
when we estimate.

⚠️  PRICES CHANGE. The numbers below are a snapshot and may be out of date by the
    time you read this. Always confirm against the official pricing page:
        https://platform.openai.com/docs/pricing
    Treat this module as a *teaching tool*, not a billing source of truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Price in US dollars per 1,000,000 tokens."""
    input_per_1m: float
    output_per_1m: float


# A small, representative slice of the catalog. Add more as you explore.
# (input $/1M, output $/1M)
PRICING: dict[str, ModelPrice] = {
    "gpt-4o":          ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini":     ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4-turbo":     ModelPrice(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-3.5-turbo":   ModelPrice(input_per_1m=0.50, output_per_1m=1.50),
}

# Embedding models (see examples/11_embeddings.py) are billed differently: there
# is no "output" to generate, so you only pay for the tokens you send in. We keep
# them in their own table with a single price per 1M tokens.
EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the estimated cost in USD for a request/response.

    Raises KeyError with a helpful message if we don't know the model's price.
    """
    if model not in PRICING:
        known = ", ".join(sorted(PRICING))
        raise KeyError(
            f"No pricing on file for {model!r}. "
            f"Known models: {known}. "
            f"Add it to PRICING in pricing.py (check the pricing page first)."
        )
    price = PRICING[model]
    input_cost = input_tokens / 1_000_000 * price.input_per_1m
    output_cost = output_tokens / 1_000_000 * price.output_per_1m
    return input_cost + output_cost


def estimate_embedding_cost(model: str, input_tokens: int) -> float:
    """Return the estimated cost in USD for embedding `input_tokens` tokens.

    Embeddings have no output tokens, so cost depends only on the input.
    """
    if model not in EMBEDDING_PRICING:
        known = ", ".join(sorted(EMBEDDING_PRICING))
        raise KeyError(
            f"No embedding pricing on file for {model!r}. "
            f"Known models: {known}. "
            f"Add it to EMBEDDING_PRICING in pricing.py (check the pricing page first)."
        )
    return input_tokens / 1_000_000 * EMBEDDING_PRICING[model]


def format_cost(usd: float) -> str:
    """Pretty-print a cost. Tiny amounts get more decimal places so they don't
    just show up as ``$0.00`` and look free (they aren't!)."""
    if usd < 0.01:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


if __name__ == "__main__":
    # Run `python pricing.py` for a quick demo / sanity check.
    demo_model = "gpt-4o-mini"
    cost = estimate_cost(demo_model, input_tokens=1_000, output_tokens=500)
    print(f"{demo_model}: 1,000 in + 500 out  ->  {format_cost(cost)}")

    embed_model = "text-embedding-3-small"
    embed_cost = estimate_embedding_cost(embed_model, input_tokens=1_000)
    print(f"{embed_model}: 1,000 in  ->  {format_cost(embed_cost)}")
