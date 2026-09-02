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

    **Do not point this at a two-sided market without picking a side first.**
    Over and under at one line are exact complements, so feeding both in makes
    every band a mixture of two different kinds of fixture: the 0.30-0.40 band
    of a totals market holds the *overs* from defensive fixtures and the
    *unders* from open ones. The table then reports a sign flip across adjacent
    bands that is an artifact of that mixing rather than a property of the
    model, and its `n` column looks twice as informative as it is. Use
    `calibration_by_line` for those markets; it splits by line and takes one
    side. This function is right for a market whose selections are not
    complements of each other.
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


# --------------------------------------------------------------------------
# Season and era breakdown
# --------------------------------------------------------------------------

def season_labels(dates, cutover_month: int = 8) -> pd.Series:
    """Label each date with the calendar year its season started in.

    A European league season spans two calendar years, so a raw year is the
    wrong grouping: it splits a single season in half at the winter break and
    pools the end of one season with the start of the next.

    The cutover defaults to August because that is when the top-5 leagues
    start and, more importantly, because it puts the COVID-delayed finish of
    2019-20 (matches played in June and July 2020) back in the 2019 season
    where it belongs. A January cutover, or the more obvious "July" guess,
    would file those matches under 2020-21 and smear the one regime this
    breakdown exists to isolate across two rows.

    No top-5 league plays league fixtures in July in a normal year, so the
    boundary is unambiguous in practice. Where the caller has the authoritative
    `season_start_year` column from the database, pass that instead of relying
    on this.
    """
    stamps = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    years = stamps.dt.year.to_numpy()
    months = stamps.dt.month.to_numpy()
    return pd.Series(np.where(months >= cutover_month, years, years - 1), name="season")


def clustered_mean(values, clusters) -> dict:
    """Mean with a standard error that allows for clustering.

    Every selection on one match is priced off the same model fit and settled
    against the same closing-line move, so bets within a match are not
    independent draws. Treating them as independent, which is what
    `closing_line_value` does, understates the standard error by roughly the
    square root of the number of bets per match. On this data that is a factor
    of about 1.2 to 1.7 - not enough to overturn a 7-sigma result, but easily
    enough to manufacture a 2-sigma one, and the whole point of a per-season
    table is that each row is thin enough for that to matter.

    Uses the standard cluster-robust variance for a mean, with the usual
    G/(G-1) small-sample correction. When every observation is its own cluster
    this reduces exactly to the naive `std(ddof=1) / sqrt(n)`, which is what
    `test_clustered_se_matches_the_naive_se_without_clusters` pins down.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    n = len(values)
    if n == 0:
        return {"n": 0, "n_clusters": 0, "mean": float("nan"), "se": float("nan")}

    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "n_clusters": 1, "mean": mean, "se": float("nan")}

    residuals = values - mean
    frame = pd.DataFrame({"cluster": clusters, "residual": residuals})
    group_sums = frame.groupby("cluster")["residual"].sum().to_numpy()
    groups = len(group_sums)
    if groups < 2:
        return {"n": n, "n_clusters": groups, "mean": mean, "se": float("nan")}

    variance = (groups / (groups - 1)) * float((group_sums ** 2).sum()) / (n ** 2)
    return {
        "n": n,
        "n_clusters": groups,
        "mean": mean,
        "se": float(np.sqrt(variance)),
    }


def season_breakdown(
    predictions: pd.DataFrame,
    market: str | None = None,
    season: pd.Series | None = None,
    edge_threshold: float = 0.02,
    cutover_month: int = 8,
) -> pd.DataFrame:
    """Closing line value and probability quality, one row per season.

    A single pooled CLV number assumes the thing being measured held still for
    the length of the window. That assumption is worth checking rather than
    making. If CLV is strongly positive in some seasons and negative in
    others, the pooled figure is an average over regimes that no longer exist,
    and both the tuning window and the evaluation window are answering a
    question about a mixture nobody asked about.

    `mean_clv` is measured on selections that cleared `edge_threshold`, so it
    matches the headline number in `scripts/backtest.py`. Log loss is measured
    on every priced selection via `score_market`, so the two columns answer
    different questions on purpose: whether the model's probabilities got
    worse, and whether the prices it could take got worse. Those can move
    independently, and which one moved says what changed.
    """
    if predictions.empty:
        return pd.DataFrame()

    frame = predictions.copy().reset_index(drop=True)
    if market is not None:
        frame = frame[frame["market"] == market].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    if season is not None:
        frame["season"] = pd.Series(season).reset_index(drop=True)
    else:
        frame["season"] = season_labels(frame["date"], cutover_month)

    rows = []
    for label, group in frame.groupby("season"):
        bets = group[
            group["expected_value"].notna()
            & (group["expected_value"] >= edge_threshold)
            & group["clv"].notna()
        ]
        clv = clustered_mean(bets["clv"], bets["match_id"]) if not bets.empty else {}
        # `closing_line_value` uses std(ddof=1), which is undefined on a single
        # observation. A pooled window never has one bet in it; a per-season
        # row easily can, in a season the data barely covers.
        naive = closing_line_value(bets) if len(bets) > 1 else {}
        if len(bets) == 1:
            naive = {"beat_close_rate": float(bets["clv"].iloc[0] > 0)}

        row = {
            "season": int(label),
            "matches": int(group["match_id"].nunique()),
            "priced": int(len(group)),
            "bets": int(clv.get("n", 0)),
            "mean_clv": clv.get("mean", float("nan")),
            "se_clustered": clv.get("se", float("nan")),
            "se_naive": naive.get("clv_standard_error", float("nan")),
            "beat_close": naive.get("beat_close_rate", float("nan")),
        }
        row["z"] = (
            row["mean_clv"] / row["se_clustered"]
            if np.isfinite(row.get("se_clustered", np.nan)) and row["se_clustered"] > 0
            else float("nan")
        )

        if market is not None:
            score = score_market(group, market)
            row["model_log_loss"] = score.get("model_log_loss", float("nan"))
            row["market_log_loss"] = score.get("market_log_loss", float("nan"))
            row["log_loss_gap"] = score.get("log_loss_gap", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


# --------------------------------------------------------------------------
# Distribution shape
# --------------------------------------------------------------------------

def calibration_by_line(
    predictions: pd.DataFrame,
    market: str,
    side: str,
    bins: int = 10,
    min_count: int = 25,
) -> pd.DataFrame:
    """Calibration for one side of a market, split by the bookmaker's line.

    Two problems with pooling, and this fixes both.

    **The complement problem.** Over and under at the same line always sum to
    one, so a pooled table double-counts every match and mixes opposite kinds
    of fixture into the same probability band. Taking one `side` removes the
    duplication and the mixing together; the other side carries no independent
    information, so nothing is lost by dropping it.

    **The line problem.** A handicap at -0.5 and one at -2.0 are different
    questions, and pooling them hides a model that is fine at one and wrong at
    the other. Splitting by line asks them separately. For a market the source
    only ever quotes at a single line this split is a no-op, which is itself
    worth knowing before inventing an explanation that needs several.

    `gap` is observed frequency minus predicted, and `se` is the standard error
    of that gap *under the model* - the Poisson-binomial spread of the number
    of winners, which is the right null for "is this discrepancy real". The
    existing `calibration_table` leaves that arithmetic to the reader, which is
    how a 3.5-sigma band and a 1-sigma band end up looking equally alarming.
    """
    frame = predictions[predictions["market"] == market]
    frame = frame[frame["selection"] == side]
    frame = frame[frame["push_fraction"] < 0.5]
    if "model_conditional" in frame.columns:
        frame = frame[frame["model_conditional"].notna()]
    if frame.empty:
        return pd.DataFrame()

    probability_column = (
        "model_conditional" if "model_conditional" in frame.columns
        else "model_probability"
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for line, group in frame.groupby("line", dropna=False):
        probabilities = group[probability_column].to_numpy(dtype=float)
        outcomes = (group["win_fraction"] > 0.5).to_numpy(dtype=float)
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (probabilities >= low) & (probabilities < high)
            n = int(mask.sum())
            if n < min_count:
                continue
            predicted = probabilities[mask]
            observed = float(outcomes[mask].mean())
            gap = observed - float(predicted.mean())
            se = float(np.sqrt((predicted * (1.0 - predicted)).sum())) / n
            rows.append(
                {
                    "line": line,
                    "band": f"{low:.2f}-{high:.2f}",
                    "n": n,
                    "predicted": float(predicted.mean()),
                    "observed": observed,
                    "gap": gap,
                    "se": se,
                    "z": gap / se if se > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def calibration_slope(probabilities, outcomes) -> dict:
    """How much too extreme - or too timid - the probabilities are, as one number.

    Regresses the outcome on the *log-odds* of the model's probability. A
    perfectly calibrated model gives slope 1 and intercept 0.

    - **slope below 1** means the probabilities are spread too far apart: the
      model separates fixtures more confidently than the results justify, so
      the fix is more shrinkage, not a different distribution.
    - **slope above 1** means the opposite, that it is hedging toward the base
      rate and could afford to be bolder.
    - **intercept away from 0** is a plain bias, the same thing a consistently
      signed `gap` column shows.

    This is worth having next to a calibration table because the table cannot
    distinguish the two failures by eye. A sign flip across bands looks like
    "badly shaped" but is the *expected* appearance of a slope below one, and
    reading it as unfixable shape when it is really excess spread points the
    next piece of work in the wrong direction.

    Fitted by plain maximum likelihood on the two parameters, with a
    cluster-free standard error on the slope from the inverse Fisher
    information. Selections that share a match are not independent, so treat a
    borderline result here the way `clustered_mean` treats a borderline CLV.
    """
    from scipy import optimize

    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    outcomes = np.asarray(outcomes, dtype=float)
    keep = np.isfinite(probabilities) & np.isfinite(outcomes)
    probabilities, outcomes = probabilities[keep], outcomes[keep]
    if len(outcomes) < 30 or len(np.unique(outcomes)) < 2:
        return {"n": int(len(outcomes))}

    logit = np.log(probabilities / (1.0 - probabilities))

    def negative_log_likelihood(theta):
        intercept, slope = theta
        eta = intercept + slope * logit
        # log(1 + exp(eta)) computed stably for large |eta|.
        log_denominator = np.logaddexp(0.0, eta)
        return -float(np.sum(outcomes * eta - log_denominator))

    result = optimize.minimize(
        negative_log_likelihood, x0=np.array([0.0, 1.0]), method="BFGS"
    )
    intercept, slope = float(result.x[0]), float(result.x[1])

    eta = intercept + slope * logit
    fitted = 1.0 / (1.0 + np.exp(-eta))
    weight = fitted * (1.0 - fitted)
    design = np.column_stack([np.ones_like(logit), logit])
    information = design.T @ (design * weight[:, None])
    try:
        covariance = np.linalg.inv(information)
        slope_se = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:  # pragma: no cover - singular only if logit is constant
        slope_se = float("nan")

    return {
        "n": int(len(outcomes)),
        "intercept": intercept,
        "slope": slope,
        "slope_se": slope_se,
        "slope_z": (slope - 1.0) / slope_se if slope_se > 0 else float("nan"),
    }


def goal_total_fit(match_totals: pd.DataFrame, max_bucket: int = 6) -> dict:
    """Does the realised spread of match totals match the predicted spread?

    `models/counts.py` uses a negative binomial for corners and cards because a
    Poisson is too narrow for them: the variance of a real count exceeds its
    mean, so a Poisson fit is overconfident about the middle and underprices
    the tails. The goals model never had that reasoning applied to it. This
    tests it directly, against the fitted distribution rather than against a
    textbook Poisson, so the answer accounts for the Dixon-Coles correction and
    for the fact that every match has its own rates.

    Two views, because they fail differently.

    **`dispersion`** is the mean squared Pearson residual, `(observed - mean) /
    sd`, averaged over matches. A correctly shaped predictive distribution puts
    it at 1.0. Above 1.0 means realised totals scatter more widely than the
    model expects, which is the corners-and-cards failure. Note honestly what
    else inflates it: error in the *fitted rates* also widens the residuals, so
    a ratio above one is evidence that the predictive distribution is too
    narrow, not proof that the Poisson assumption specifically is the culprit.

    **`buckets`** is the same question without assuming the failure is
    symmetric. It compares how often each total actually occurred against how
    often the model said it would, with a Poisson-binomial standard error for
    each bucket. Overdispersion shows up here as too few matches in the middle
    and too many in both tails; a mean that is simply misplaced shows up as a
    monotone drift instead. The distinction decides whether the fix is a
    dispersion parameter or a better rate.

    The variance is also split into its two sources, because "the predictive
    distribution is too wide" has two different causes and two different
    repairs. `within_match_variance` is how wide each match's own distribution
    is, and it is a Poisson property that a dispersion parameter would change.
    `between_match_variance` is how far the fitted rates move from fixture to
    fixture, and it is a shrinkage property that ridge and the half-life
    control. Their sum is what should match `observed_variance`.
    """
    if match_totals.empty:
        return {"n": 0}

    pmfs = np.vstack([np.asarray(p, dtype=float) for p in match_totals["total_pmf"]])
    observed = match_totals["observed_total"].to_numpy(dtype=int)
    support = np.arange(pmfs.shape[1])

    predicted_mean = pmfs @ support
    predicted_second = pmfs @ (support ** 2)
    predicted_variance = predicted_second - predicted_mean ** 2

    usable = predicted_variance > 0
    residual = (observed[usable] - predicted_mean[usable]) / np.sqrt(
        predicted_variance[usable]
    )
    squared = residual ** 2
    n = int(len(squared))
    ratio = float(squared.mean())
    ratio_se = float(squared.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    # Law of total variance, so that "the distribution is too wide" can be
    # attributed. Var(T) = E[Var(T|x)] + Var(E[T|x]): the first term is how
    # wide each match's own distribution is, the second is how much the fitted
    # rates move between matches. They call for different repairs - a
    # dispersion parameter for the first, more shrinkage for the second - so
    # reporting only their sum would leave the useful part out.
    within = float(predicted_variance.mean())
    between = float(predicted_mean.var(ddof=1))

    # Buckets: every total up to max_bucket, then one tail bucket holding the
    # rest, so the comparison stays well-powered where the data is thin.
    rows = []
    for bucket in range(max_bucket + 1):
        column = pmfs[:, bucket] if bucket < pmfs.shape[1] else np.zeros(len(pmfs))
        rows.append(
            _bucket_row(str(bucket), (observed == bucket), column)
        )
    tail = pmfs[:, max_bucket + 1:].sum(axis=1)
    rows.append(_bucket_row(f"{max_bucket + 1}+", (observed > max_bucket), tail))

    return {
        "n": n,
        "dispersion": ratio,
        "dispersion_se": ratio_se,
        "dispersion_z": (ratio - 1.0) / ratio_se if ratio_se > 0 else float("nan"),
        "observed_mean": float(observed.mean()),
        "predicted_mean": float(predicted_mean.mean()),
        "observed_variance": float(observed.var(ddof=1)),
        "within_match_variance": within,
        "between_match_variance": between,
        "predicted_variance": within + between,
        "buckets": pd.DataFrame(rows),
    }


def _bucket_row(label: str, hit: np.ndarray, probabilities: np.ndarray) -> dict:
    """One row of the observed-versus-expected table for a count bucket.

    The count of matches landing in a bucket is a sum of independent Bernoulli
    draws with *different* probabilities - a Poisson-binomial - so its variance
    is the sum of p(1-p) rather than n*p_bar*(1-p_bar). Using the latter would
    overstate the spread and quietly hide real discrepancies.
    """
    observed = int(hit.sum())
    expected = float(probabilities.sum())
    variance = float((probabilities * (1.0 - probabilities)).sum())
    se = float(np.sqrt(variance))
    return {
        "total": label,
        "observed": observed,
        "expected": expected,
        "difference": observed - expected,
        "se": se,
        "z": (observed - expected) / se if se > 0 else float("nan"),
    }


def fair_line_sources(
    predictions: pd.DataFrame,
    season: pd.Series | None = None,
    cutover_month: int = 8,
) -> pd.DataFrame:
    """Which bookmaker supplied the closing benchmark, season by season.

    Run this before reading anything into a change in CLV over time. Closing
    line value is a comparison against a benchmark, so a season where the
    benchmark changed is not comparable with the seasons around it, and nothing
    in a CLV table says that it changed.

    This is not a hypothetical failure. `betfair_exchange` first appears in
    football-data.co.uk in 2024-25, and it heads `FAIR_LINE_PREFERENCE`, so the
    benchmark switched from Pinnacle to Betfair in exactly the season CLV
    appeared to fall off a cliff. Measured on the seasons where both exist, the
    exchange scores the same bets about 1.75 points lower than Pinnacle does -
    which is most of the apparent collapse. The market moved much less than the
    ruler did.

    `share` is the fraction of that season's priced selections the book
    supplied, so a season split between two books is visible rather than
    rounded to whichever won.
    """
    if predictions.empty or "fair_line_source" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions.copy().reset_index(drop=True)
    frame = frame[frame["fair_line_source"].notna()]
    if frame.empty:
        return pd.DataFrame()

    if season is not None:
        frame["season"] = pd.Series(season).reset_index(drop=True).loc[frame.index]
    else:
        frame["season"] = season_labels(frame["date"], cutover_month).loc[frame.index]

    counts = (
        frame.groupby(["season", "fair_line_source"]).size().rename("selections")
    ).reset_index()
    totals = counts.groupby("season")["selections"].transform("sum")
    counts["share"] = counts["selections"] / totals
    return counts.sort_values(
        ["season", "selections"], ascending=[True, False]
    ).reset_index(drop=True)


def benchmark_changed(sources: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Seasons where the dominant benchmark differs from the season before.

    Returns human-readable warnings rather than a bare flag, because the only
    useful response is to say which seasons stopped being comparable and to
    whom. `threshold` is the share a book needs before it counts as the one
    that supplied the season.
    """
    if sources.empty:
        return []
    dominant = {}
    for season, group in sources.groupby("season"):
        top = group.sort_values("share", ascending=False).iloc[0]
        if top["share"] >= threshold:
            dominant[int(season)] = str(top["fair_line_source"])

    warnings_out = []
    ordered = sorted(dominant)
    for previous, current in zip(ordered, ordered[1:]):
        if dominant[previous] != dominant[current]:
            warnings_out.append(
                f"{current}: benchmark changed from {dominant[previous]} to "
                f"{dominant[current]}; CLV before and after is not comparable."
            )
    return warnings_out


def era_comparison(
    predictions: pd.DataFrame,
    split_season: int,
    market: str | None = None,
    season: pd.Series | None = None,
    edge_threshold: float = 0.02,
    cutover_month: int = 8,
) -> dict:
    """Pool seasons either side of one pre-specified split and compare CLV.

    Per-season rows are thin. Pooling into two eras buys back the power, but
    only if the split point is chosen before looking. Scanning every possible
    split and reporting the one with the largest gap is the same uncorrected
    multiple-comparisons error as tuning on five leagues and reporting the
    best: with six candidate splits, a two-sigma "regime change" is what you
    should expect to see even when nothing changed.

    So: pick the split from a reason, not from the data. `2021` is defensible
    because crowds returned for 2021-22 and home advantage is known to have
    moved while they were absent. If a different split is used, it needs a
    reason of the same kind, stated first.

    The two eras are disjoint sets of matches, so the clusters do not overlap
    and the variance of the difference is just the sum of the two variances.
    """
    table_input = predictions.copy().reset_index(drop=True)
    if market is not None:
        table_input = table_input[table_input["market"] == market].reset_index(drop=True)
    if table_input.empty:
        return {"n_before": 0, "n_after": 0}

    if season is not None:
        table_input["season"] = pd.Series(season).reset_index(drop=True)
    else:
        table_input["season"] = season_labels(table_input["date"], cutover_month)

    bets = table_input[
        table_input["expected_value"].notna()
        & (table_input["expected_value"] >= edge_threshold)
        & table_input["clv"].notna()
    ]
    before = bets[bets["season"] < split_season]
    after = bets[bets["season"] >= split_season]
    if before.empty or after.empty:
        return {"n_before": len(before), "n_after": len(after)}

    left = clustered_mean(before["clv"], before["match_id"])
    right = clustered_mean(after["clv"], after["match_id"])
    difference = right["mean"] - left["mean"]
    se = float(np.sqrt(left["se"] ** 2 + right["se"] ** 2))

    return {
        "split_season": int(split_season),
        "n_before": left["n"], "mean_clv_before": left["mean"], "se_before": left["se"],
        "n_after": right["n"], "mean_clv_after": right["mean"], "se_after": right["se"],
        "difference": float(difference),
        "difference_se": se,
        "difference_z": float(difference / se) if se > 0 else float("nan"),
    }
