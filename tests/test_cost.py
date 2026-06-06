from __future__ import annotations

from lbmm.cost import PRICES, lookup


def test_anthropic_haiku_price() -> None:
    p = lookup("anthropic", "claude-3-5-haiku-latest")
    assert p is not None
    # at 2024-Q4 list: input 0.80 / output 4.00 per 1M
    cost = p.usd_for(1_000_000, 1_000_000)
    assert abs(cost - (0.80 + 4.00)) < 1e-9


def test_local_models_are_free() -> None:
    p = lookup("local", "qwen2.5-0.5b-instruct")
    assert p is not None
    assert p.usd_for(99_999, 99_999) == 0.0


def test_unknown_returns_none() -> None:
    assert lookup("anthropic", "no-such-model") is None
    assert lookup("nobody", "claude-3-5-haiku-latest") is None


def test_all_listed_models_have_nonnegative_prices() -> None:
    for vendor, table in PRICES.items():
        for model, price in table.items():
            assert price.input_per_mtok >= 0, f"{vendor}:{model} input"
            assert price.output_per_mtok >= 0, f"{vendor}:{model} output"
