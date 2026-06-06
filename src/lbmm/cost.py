"""Price tables ($USD per 1M tokens, input/output) per provider+model.

Updated 2024-Q4. Keep up to date manually; cost surprises in production
are usually a stale price table. The numbers below are list prices and may
not reflect committed-use discounts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def usd_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok / 1_000_000
            + completion_tokens * self.output_per_mtok / 1_000_000
        )


PRICES: dict[str, dict[str, Price]] = {
    "anthropic": {
        "claude-3-5-haiku-latest": Price(0.80, 4.00),
        "claude-3-5-sonnet-latest": Price(3.00, 15.00),
        "claude-3-opus-latest": Price(15.00, 75.00),
    },
    "openai": {
        "gpt-4o-mini": Price(0.15, 0.60),
        "gpt-4o": Price(2.50, 10.00),
    },
    "google": {
        "gemini-1.5-flash": Price(0.075, 0.30),
        "gemini-1.5-pro": Price(1.25, 5.00),
    },
    "local": {
        # local inference is free in $ terms; we still compute "compute time"
        # via latency_ms in the runner so cost-per-token is comparable.
        "qwen2.5-0.5b-instruct": Price(0.0, 0.0),
        "qwen2.5-1.5b-instruct": Price(0.0, 0.0),
    },
}


def lookup(provider: str, model: str) -> Price | None:
    return PRICES.get(provider, {}).get(model)
