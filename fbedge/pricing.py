"""Odds arithmetic: turning prices into probabilities and back.

A bookmaker's prices imply probabilities that sum to more than one. The excess
is the margin - the overround - and it is the reason betting is a losing
proposition by default. To compare a model against the market you first have to
strip that margin out, and *how* you strip it changes the answer, particularly
for longshots.

Two methods are provided. Neither is obviously right, so Phase 3 tests which
one produces better-calibrated benchmarks rather than the choice being made
here on aesthetics.

**Multiplicative** divides every implied probability by the overround. Simple,
and it assumes the margin is applied proportionally across outcomes.

**Additive** subtracts an equal share of the margin from each outcome. This
takes proportionally more away from longshots, which matches the well-documented
favourite-longshot bias: bookmakers load more margin onto the outsider.
"""

from __future__ import annotations

import numpy as np


def implied_probability(price: float | np.ndarray):
    """Raw implied probability of a decimal price. Includes the margin."""
    price = np.asarray(price, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(price > 1.0, 1.0 / price, np.nan)


def overround(prices) -> float:
    """Sum of implied probabilities across a complete market.

    A sharp book's 1X2 market lands around 1.02-1.04; a soft one is higher.
    A value at or below 1.00 means the prices are wrong or the market is
    incomplete, not that free money has been found.
    """
    return float(np.nansum(implied_probability(np.asarray(prices, dtype=float))))


def remove_margin(prices, method: str = "multiplicative") -> np.ndarray:
    """Convert a complete set of prices into probabilities summing to one.

    Args:
        prices: every price in the market, e.g. all three of a 1X2.
        method: "multiplicative" or "additive".
    """
    raw = implied_probability(np.asarray(prices, dtype=float))
    total = np.nansum(raw)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Prices do not form a usable market.")

    if method == "multiplicative":
        return raw / total
    if method == "additive":
        # Spread the excess equally, then repair any negative that creates.
        adjusted = raw - (total - 1.0) / np.count_nonzero(~np.isnan(raw))
        if np.nanmin(adjusted) <= 0:
            return raw / total
        return adjusted / np.nansum(adjusted)
    raise ValueError(f"Unknown method {method!r}; use 'multiplicative' or 'additive'.")


def fair_price(probability: float) -> float:
    """The break-even decimal price for a probability."""
    return float("inf") if probability <= 0 else 1.0 / probability


def expected_value(probability: float, price: float, push_probability: float = 0.0) -> float:
    """Expected profit per unit staked.

    Zero means break-even, +0.05 means an expected five percent return on the
    stake. This is the number Phase 4's scanner sorts on, and it is only as
    trustworthy as the probability fed into it: a model that is 3% too
    confident will manufacture a 3% edge out of nothing.
    """
    return probability * price + push_probability - 1.0


def edge_over_market(model_probability: float, market_probability: float) -> float:
    """How much more likely the model thinks an outcome is than the market.

    Expressed as a ratio minus one, so +0.10 means the model has it ten percent
    more likely than the margin-free market price implies.
    """
    if market_probability <= 0:
        return float("nan")
    return model_probability / market_probability - 1.0
