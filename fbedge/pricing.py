"""Odds arithmetic: turning prices into probabilities and back.

A bookmaker's prices imply probabilities that sum to more than one. The excess
is the margin - the overround - and it is the reason betting is a losing
proposition by default. To compare a model against the market you first have to
strip that margin out, and *how* you strip it changes the answer, particularly
for longshots.

Three methods are provided, and the choice between them has now been **tested
against ground truth rather than argued from aesthetics**, which is what the
earlier version of this docstring said ought to happen.

**Multiplicative** divides every implied probability by the overround. Simple,
and it assumes the margin is applied proportionally across outcomes.

**Additive** subtracts an equal share of the margin from each outcome. This
takes proportionally more away from longshots, which matches the well-documented
favourite-longshot bias: bookmakers load more margin onto the outsider.

**Shin** models the margin as the bookmaker's defence against better-informed
traders, and solves for the implied share of such trading, `z`, one market at a
time. It is the only one of the three with a free parameter, so it can land
between the other two instead of being stuck at whatever correction its fixed
rule happens to imply.

**The test.** football-data.co.uk carries Betfair Exchange closing prices from
2024-25. An exchange charges commission instead of building in a margin, so it
closes near a 0.56% overround against Pinnacle's ~2.9% and is the closest thing
to a margin-free truth in the data. On the 1770 selections where both books
priced the same match, reconstructing the exchange's probabilities from
Pinnacle's prices gives mean error by probability band:

| band | multiplicative | additive | Shin |
|---|---|---|---|
| 0.00-0.15 | +0.0060 | -0.0012 | +0.0007 |
| 0.15-0.25 | +0.0023 | -0.0015 | -0.0006 |
| 0.25-0.35 | +0.0005 | -0.0009 | -0.0005 |
| 0.35-0.50 | -0.0011 | +0.0014 | +0.0008 |
| 0.50-1.00 | -0.0064 | +0.0030 | +0.0006 |

Multiplicative has exactly the favourite-longshot gradient this docstring
predicted, and it is large: it overstates a longshot by more than half a point.
Additive corrects it but overshoots. Shin is flat across the whole range.

**Why this matters more than it looks.** Closing line value is measured by
multiplying the price taken by one of these probabilities, and the model bets
longshots - the median backed selection closes around 0.22. Measured on the
same bets, multiplicative sits +1.75 points above the exchange (11 SE) and
additive -0.77 points below it (5.1 SE), while Shin is within -0.12 points
(0.9 SE). A whole apparent era of positive CLV in this project turned out to be
the multiplicative bias rather than an edge.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


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


def remove_margin(prices, method: str = "shin") -> np.ndarray:
    """Convert a complete set of prices into probabilities summing to one.

    Args:
        prices: every price in the market, e.g. all three of a 1X2.
        method: "shin" (default), "multiplicative" or "additive". See the
            module docstring for the measurement that decided the default.
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
    if method == "shin":
        return _shin(raw, total)
    raise ValueError(
        f"Unknown method {method!r}; use 'shin', 'multiplicative' or 'additive'."
    )


def _shin(raw: np.ndarray, total: float) -> np.ndarray:
    """Shin's margin removal, solving for the insider-trading share z.

    The model treats the overround as the bookmaker protecting itself against
    traders who know more than it does. If a fraction `z` of turnover comes
    from such traders, the true probabilities satisfy

        p_i = [sqrt(z^2 + 4(1 - z) * r_i^2 / R) - z] / [2(1 - z)]

    with `r_i` the raw implied probabilities and `R` their sum. Larger `z`
    pulls more out of the longshots than the favourites, which is the shape the
    data actually shows; `z` is solved per market so the result sums to one.

    Falls back to multiplicative whenever the solve is not well posed - a
    market already at or below 1.00, or no sign change in the bracket. Those
    are degenerate prices rather than a reason to fail, and the caller has no
    better option to offer.
    """
    if not np.isfinite(total) or total <= 1.0:
        return raw / total

    finite = ~np.isnan(raw)
    values = raw[finite]

    def implied(z: float) -> np.ndarray:
        return (
            np.sqrt(z**2 + 4.0 * (1.0 - z) * values**2 / total) - z
        ) / (2.0 * (1.0 - z))

    try:
        z = optimize.brentq(lambda v: float(implied(v).sum() - 1.0), 1e-12, 0.4)
    except ValueError:
        # No root in the bracket: the margin is outside what the model can
        # explain as insider trading. Proportional is the honest fallback.
        return raw / total

    out = np.full_like(raw, np.nan)
    solved = implied(z)
    out[finite] = solved / solved.sum()
    return out


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
