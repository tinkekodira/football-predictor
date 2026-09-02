"""Scoring a backtest.

Four questions, in descending order of how much they should influence what you
do next.

**Is the model calibrated?** When it says 30%, does it happen 30% of the time?
A model can have a decent log loss and still be systematically overconfident in
one band, and a few percent of overconfidence is more than enough to turn an
apparent edge into a real loss.

**Does it beat the closing line?** The closing price at a sharp bookmaker
aggregates an enormous amount of information. Beating it is the whole game.
Closing line value is measurable over a few hundred bets, whereas profit is
not, which is why it is the primary metric here.

**Does it beat the market on log loss?** A direct comparison of the model's
probabilities against the margin-free closing probabilities, on the same
matches. This is the cleanest single number for "does the model know
something the market does not".

**What would it have returned?** Last, and reported with a confidence interval
rather than a point estimate, because over a few hundred bets the interval is
wide enough to contain both a healthy profit and a serious loss. A backtest
that reports "+7% ROI" without saying "give or take fifteen points" is not
telling you anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-12


# --------------------------------------------------------------------------
# Probability scoring
# --------------------------------------------------------------------------

def binary_log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log probability assigned to what happened.

    Punishes confident mistakes far harder than cautious ones, which is the
    right shape: the confident mistake is what empties a bankroll.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2))


def calibration_table(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10, min_count: int = 10
) -> pd.DataFrame:
    """Predicted probability against observed frequency, by band.

    Read the `gap` column. Consistently positive means the model is
    underconfident, consistently negative means overconfident, and a sign that
    flips across bands means it is badly shaped rather than merely biased.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.sum() < min_count:
            continue
        rows.append(
            {
                "band": f"{low:.2f}-{high:.2f}",
                "n": int(mask.sum()),
                "predicted": float(probabilities[mask].mean()),
                "observed": float(outcomes[mask].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["gap"] = frame["observed"] - frame["predicted"]
    return frame


def score_market(predictions: pd.DataFrame, market: str) -> dict:
    """Model versus market versus base rate, on one market.

    Two things keep the comparison honest. Selections that actually pushed are
    dropped, because a returned stake is not a prediction that came true or
    false. And the model's probability is converted to the same conditional
    scale the bookmaker's normalised prices are already on - without that,
    handicaps and whole-number totals would be compared on different footings
    and the model would look worse than it is.
    """
    frame = predictions[predictions["market"] == market].copy()
    frame = frame[frame["push_fraction"] < 0.5]
    if "model_conditional" in frame.columns:
        frame = frame[frame["model_conditional"].notna()]
    if frame.empty:
        return {"market": market, "n": 0}

    outcomes = (frame["win_fraction"] > 0.5).astype(float).to_numpy()
    model = (
        frame["model_conditional"] if "model_conditional" in frame.columns
        else frame["model_probability"]
    ).to_numpy()

    result = {
        "market": market,
        "n": int(len(frame)),
        "model_log_loss": binary_log_loss(model, outcomes),
        "model_brier": brier_score(model, outcomes),
        "base_rate_log_loss": binary_log_loss(
            np.full_like(model, outcomes.mean()), outcomes
        ),
    }

    priced = frame[frame["market_probability"].notna()]
    if len(priced) >= 30:
        priced_outcomes = (priced["win_fraction"] > 0.5).astype(float).to_numpy()
        priced_model = (
            priced["model_conditional"] if "model_conditional" in priced.columns
            else priced["model_probability"]
        ).to_numpy()
        model_score = binary_log_loss(priced_model, priced_outcomes)
        market_score = binary_log_loss(
            priced["market_probability"].to_numpy(), priced_outcomes
        )
        result |= {
            "n_priced": int(len(priced)),
            "model_log_loss_priced": model_score,
            "market_log_loss": market_score,
            "log_loss_gap": model_score - market_score,
        }
    return result


# --------------------------------------------------------------------------
# Closing line value
# --------------------------------------------------------------------------

def closing_line_value(bets: pd.DataFrame) -> dict:
    """Did the prices taken beat the closing line?

    The headline measure is the expected value of each bet evaluated at the
    *margin-free closing probability*: if you took 2.10 on something the close
    said was worth 2.30, you got closing line value whether or not it won.
    That is the right comparison, because it strips out the difference between
    a soft book's margin and a sharp book's before comparing anything.

    Raw price movement at the same bookmaker is reported alongside as a
    secondary check. Used on its own it conflates a moving line with a
    changing margin, which is why it is not the headline.

    Positive CLV over a decent number of bets is the strongest evidence
    available that a model has found something real, and it shows up long
    before profit does. Negative CLV alongside a profit means the profit was
    luck and will not survive.
    """
    frame = bets.dropna(subset=["clv"]) if "clv" in bets.columns else bets.iloc[0:0]
    if frame.empty:
        return {"n": 0}
    values = frame["clv"].to_numpy()
    result = {
        "n": int(len(frame)),
        "mean_clv": float(values.mean()),
        "median_clv": float(np.median(values)),
        "beat_close_rate": float((values > 0).mean()),
        "clv_standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
    }
    movement = frame["price_movement"].dropna() if "price_movement" in frame else pd.Series(dtype=float)
    if len(movement) >= 20:
        result["mean_price_movement"] = float(movement.mean())
        result["n_price_movement"] = int(len(movement))
    return result


def bookmaker_breakdown(predictions: pd.DataFrame, min_selections: int = 100) -> pd.DataFrame:
    """Margin and beatability, for every bookmaker in the data - not just
    whichever one happened to supply the best price on a given selection.

    The practical question for anyone who cannot get on at a sharp book is not
    "does the model work" but "does it work against prices I can actually
    take", and different books deserve different answers. This explodes the
    per-selection `book_prices` dictionary so that every bookmaker's own price
    is compared against the same fair closing line, independent of which price
    ended up chosen for staking.

    `mean_clv` is the average expected value of that bookmaker's own prices,
    measured against the margin-free closing line, using every priced
    selection rather than only the ones that cleared a betting threshold. For
    a model with no edge it should land close to minus that book's margin; a
    book sitting well above the pack is either sharp or genuinely beatable.
    """
    if predictions.empty or "book_prices" not in predictions.columns:
        return pd.DataFrame()

    rows = []
    for entry in predictions["book_prices"]:
        if isinstance(entry, dict):
            rows.extend(entry.items())
    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows, columns=["bookmaker", "clv"])
    grouped = exploded.groupby("bookmaker").agg(
        selections=("clv", "size"),
        mean_clv=("clv", "mean"),
        beat_close=("clv", lambda values: float((values > 0).mean())),
    )
    grouped = grouped[grouped["selections"] >= min_selections]
    return grouped.sort_values("mean_clv", ascending=False)


def market_margins(con, league: str | None = None) -> pd.DataFrame:
    """Average overround per bookmaker on 1X2 closing prices.

    A sharp book lands near 1.02, a soft one nearer 1.07. The gap is what you
    are paying to bet somewhere convenient, and it is usually larger than any
    edge a model of this kind will find.
    """
    sql = """
        SELECT o.bookmaker, o.phase,
               COUNT(*) AS markets,
               ROUND(AVG(implied), 4) AS mean_overround
        FROM (
            SELECT o.match_id, o.bookmaker, o.phase, SUM(1.0 / o.price) AS implied
            FROM odds o
            JOIN matches m USING (match_id)
            WHERE o.market = '1x2'
            {league_filter}
            GROUP BY o.match_id, o.bookmaker, o.phase
            HAVING COUNT(*) = 3
        ) o
        GROUP BY o.bookmaker, o.phase
        HAVING COUNT(*) >= 50
        ORDER BY mean_overround
    """
    filter_clause = "AND m.league = ?" if league else ""
    return con.execute(
        sql.format(league_filter=filter_clause), [league] if league else []
    ).df()


# --------------------------------------------------------------------------
# Staking
# --------------------------------------------------------------------------

def kelly_stake(
    probability: float, price: float, fraction: float = 0.25, cap: float = 0.02
) -> float:
    """Fractional Kelly stake as a share of bankroll.

    Full Kelly is the growth-optimal stake only if the probability is exactly
    right, which it never is. A model that is a little overconfident staking
    full Kelly is not aggressive, it is ruinous, so the default here is a
    quarter Kelly with a hard cap of two percent of bankroll.
    """
    edge = probability * price - 1.0
    if edge <= 0 or price <= 1.0:
        return 0.0
    return float(min(cap, fraction * edge / (price - 1.0)))


def simulate_staking(
    bets: pd.DataFrame,
    method: str = "flat",
    starting_bankroll: float = 1000.0,
    flat_stake: float = 1.0,
    kelly_fraction: float = 0.25,
    kelly_cap: float = 0.02,
) -> pd.DataFrame:
    """Run the bets in date order and track the result.

    The two methods are tracked differently on purpose.

    **Flat** stakes a fixed number of units on every bet, so the running total
    is cumulative profit in stake units, starting at zero - the standard
    convention in sports-betting backtests, and the reason a loss shows up as
    "-12.4 units" rather than a percentage of some starting bankroll picked out
    of the air. It stops the choice of `starting_bankroll` from silently
    determining whether the result even fits the metric: a fixed dollar stake
    against a small starting bankroll can push the running total negative,
    which earlier versions of this function reported as a bankroll going
    negative and a drawdown beyond -100%. That was a bug, not a real result.

    **Kelly** genuinely needs a bankroll, since the stake is a fraction of it.
    Here the stake is capped at whatever bankroll remains, and betting stops
    once the bankroll is exhausted, which is what an actual bettor would do.
    """
    if bets.empty:
        return pd.DataFrame()

    frame = bets.sort_values("date").copy()
    rows = []

    if method == "flat":
        cumulative = 0.0
        turnover = 0.0
        for row in frame.itertuples():
            profit = flat_stake * row.profit_at_taken
            cumulative += profit
            turnover += flat_stake
            rows.append(
                {
                    "date": row.date,
                    "market": row.market,
                    "stake": flat_stake,
                    "profit": profit,
                    "bankroll": cumulative,   # cumulative profit, in units
                    "turnover": turnover,
                }
            )
    elif method == "kelly":
        bankroll = starting_bankroll
        for row in frame.itertuples():
            if bankroll <= 0:
                break
            stake = min(
                bankroll,
                bankroll * kelly_stake(
                    row.model_probability, row.price_taken, kelly_fraction, kelly_cap
                ),
            )
            if stake <= 0:
                continue
            profit = stake * row.profit_at_taken
            bankroll = max(0.0, bankroll + profit)
            rows.append(
                {
                    "date": row.date,
                    "market": row.market,
                    "stake": stake,
                    "profit": profit,
                    "bankroll": bankroll,
                }
            )
    else:
        raise ValueError(f"Unknown staking method {method!r}")

    return pd.DataFrame(rows)


def staking_summary(ledger: pd.DataFrame, starting_bankroll: float = 1000.0) -> dict:
    """Headline numbers from a staking run.

    Auto-detects flat versus Kelly from the ledger's columns, since the two
    need different drawdown treatments. Flat's "bankroll" column is cumulative
    profit starting at zero and can go negative, so a running-peak percentage
    is not well-defined there - dividing by a peak of zero or a negative peak
    flips the sign and produces nonsense. Drawdown for flat staking is instead
    reported as a fraction of the money actually staked so far, which is
    always positive and always well-defined. Kelly's bankroll cannot go
    negative by construction, so a conventional percentage drawdown applies.
    """
    if ledger.empty:
        return {"bets": 0}

    turnover = float(ledger["stake"].sum())
    profit = float(ledger["profit"].sum())
    result = {
        "bets": int(len(ledger)),
        "turnover": turnover,
        "profit": profit,
        "roi": profit / turnover if turnover else float("nan"),
        "win_rate": float((ledger["profit"] > 0).mean()),
    }

    is_flat = "turnover" in ledger.columns
    running_peak = ledger["bankroll"].cummax()
    drawdown_units = ledger["bankroll"] - running_peak

    if is_flat:
        denominator = ledger["turnover"].replace(0, np.nan)
        drawdown_pct = (drawdown_units / denominator)
        result["max_drawdown"] = float(drawdown_pct.min(skipna=True) or 0.0)
        result["max_drawdown_units"] = float(drawdown_units.min())
        result["drawdown_basis"] = "fraction of turnover staked so far"
    else:
        denominator = running_peak.replace(0, np.nan)
        drawdown_pct = drawdown_units / denominator
        result["max_drawdown"] = float(drawdown_pct.min(skipna=True) or 0.0)
        result["final_bankroll"] = float(ledger["bankroll"].iloc[-1])
        result["growth"] = float(ledger["bankroll"].iloc[-1] / starting_bankroll - 1.0)
        result["busted"] = bool((ledger["bankroll"] <= 0).any())
        result["drawdown_basis"] = "fraction of peak bankroll"
    return result


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

def bootstrap_roi(
    bets: pd.DataFrame,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Confidence interval for ROI, resampling whole matchdays.

    Resampling by date rather than by bet matters: several bets on the same
    day share a model fit and often the same match, so treating them as
    independent draws would understate the uncertainty. The interval is
    normally wide, and that width is the finding, not a defect of the method.
    """
    frame = bets.dropna(subset=["profit_at_taken"])
    if len(frame) < 20:
        return {"n": int(len(frame))}

    by_date = [group["profit_at_taken"].to_numpy() for _, group in frame.groupby("date")]
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations)
    n_days = len(by_date)
    for i in range(iterations):
        chosen = rng.integers(0, n_days, n_days)
        pooled = np.concatenate([by_date[j] for j in chosen])
        samples[i] = pooled.mean()

    tail = (1.0 - confidence) / 2.0
    observed = float(frame["profit_at_taken"].mean())
    return {
        "n": int(len(frame)),
        "n_days": n_days,
        "roi": observed,
        "roi_low": float(np.quantile(samples, tail)),
        "roi_high": float(np.quantile(samples, 1.0 - tail)),
        "probability_profitable": float((samples > 0).mean()),
    }


def summarise(result, markets_to_score: tuple[str, ...] | None = None) -> dict:
    """Everything worth knowing about a backtest, in one dictionary."""
    predictions = result.predictions
    bets = result.bets
    markets_to_score = markets_to_score or tuple(
        predictions["market"].unique() if not predictions.empty else ()
    )
    return {
        "config": result.config,
        "refits": result.refits,
        "matches": result.matches,
        "selections": int(len(predictions)),
        "market_scores": [score_market(predictions, m) for m in markets_to_score],
        "clv": closing_line_value(bets),
        "bets": int(len(bets)),
    }
