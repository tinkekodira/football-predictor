"""Walk-forward backtesting.

The engine steps through history a week at a time. At each step it fits the
models on matches played before that week, prices every selection the
bookmaker actually offered, and settles the result. Nothing is ever fitted on
a match it later predicts.

Two things this produces, and they answer different questions.

**Predictions** - one row per selection, with the model's probability, the
market's, and what happened. This is what calibration and log loss are computed
from, and it is the honest measure of whether the model knows anything.

**Bets** - the subset where the model saw value at the earlier price, with the
closing price recorded alongside. This is where closing line value comes from,
and it is the more important of the two.

The reason CLV matters more than profit: over a few hundred bets, return on
investment is almost entirely noise. A 5% edge and a 5% loss are both
completely ordinary outcomes for a model with no edge at all. Whether the
prices you took consistently beat the closing line, though, is visible in
weeks rather than years. The free data source carries both an earlier price
and a closing price for each match, which is precisely what makes this
measurable without paying anyone.

One limitation worth naming up front: the source has no historical corner or
card prices. Those models can be checked for calibration against outcomes, but
they cannot be backtested as bets, because there is nothing to bet into.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import markets, pricing, settlement
from .models import base
from . import predict as predict_mod

# Which price to treat as the one you could have taken, and which as the
# closing line to measure against.
#
# The closing reference is ordered sharpest first. Betfair Exchange leads
# because an exchange charges commission rather than building a margin into
# the price, so its overround is a fraction of a bookmaker's and its close is
# the most accurate probability estimate in the file. Pinnacle is next. Both
# are worth measuring against even if you cannot bet at either.
#
# The price actually taken defaults to the market maximum: the best number
# available anywhere at the time, which is what someone shopping across
# several accounts would get. Pass a specific bookmaker to model what one
# account would really have offered you.
DEFAULT_PRICE_SOURCE = ("market_max", "bet365", "market_avg")
DEFAULT_CLOSING_SOURCE = ("betfair_exchange", "pinnacle", "market_avg", "market_max")

# Order of preference when deriving the margin-free closing probabilities.
# Reading these off a soft bookmaker would bake its margin into the benchmark.
#
# **This list is a measuring instrument, and changing which entry supplies a
# given season silently rewrites every CLV number for it.** That is not
# hypothetical: `betfair_exchange` first appears in the source in 2024-25, and
# because it sits at the head of this list the benchmark switched from Pinnacle
# to Betfair in exactly the season CLV appeared to collapse. Roughly three
# quarters of that "regime change" was the instrument, not the market. Whatever
# preference is used, check `fair_line_sources` for a season where the source
# changes before reading anything into a change in the numbers.
FAIR_LINE_PREFERENCE = (
    "betfair_exchange", "pinnacle", "market_avg", "market_max", "bet365",
)

BETTABLE_MARKETS = ("1x2", "total_goals", "asian_handicap")


@dataclass
class BacktestConfig:
    """Everything that defines one backtest run."""

    league: str
    start: dt.date
    end: dt.date
    step_days: int = 7
    half_life_days: float = base.DEFAULT_HALF_LIFE_DAYS
    ridge: float = base.DEFAULT_RIDGE
    markets: tuple[str, ...] = BETTABLE_MARKETS
    price_source: tuple[str, ...] = DEFAULT_PRICE_SOURCE
    closing_source: tuple[str, ...] = DEFAULT_CLOSING_SOURCE
    # Which book supplies the margin-free closing line CLV is measured against.
    # Configurable so a run can be pinned to one book that covers the whole
    # window, which is the only way to compare CLV across seasons honestly.
    fair_line_preference: tuple[str, ...] = FAIR_LINE_PREFERENCE
    # "shin" since it is the only one of the three measured to reproduce the
    # exchange's probabilities without a favourite-longshot gradient. This
    # default changed after multiplicative was found to inflate CLV by 1.75
    # points on the selections this model actually bets; numbers recorded
    # before that change are on a different and worse scale.
    margin_method: str = "shin"
    edge_threshold: float = 0.02
    min_price: float = 1.20
    max_price: float = 15.0
    fit_count_models: bool = False


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    config: BacktestConfig
    refits: int = 0
    matches: int = 0
    notes: list[str] = field(default_factory=list)
    # One row per match rather than per selection: the model's whole predicted
    # distribution over the match total, and what the total actually was. Kept
    # apart from `predictions` because it is a property of the match, not of
    # any bookmaker's offer, so it exists even for matches nobody priced, and
    # duplicating it across every selection row would invite double-counting.
    match_totals: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def bets(self) -> pd.DataFrame:
        """Selections where the model saw value at the price it could take."""
        if self.predictions.empty:
            return self.predictions
        frame = self.predictions
        return frame[
            frame["expected_value"].notna()
            & (frame["expected_value"] >= self.config.edge_threshold)
            & frame["price_taken"].between(self.config.min_price, self.config.max_price)
        ].copy()


def _load_fixtures(con, league: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    frame = con.execute(
        """
        SELECT match_id, date, home_team, away_team, referee,
               home_goals, away_goals, total_corners, total_cards
        FROM matches
        WHERE league = ? AND date >= ? AND date <= ?
          AND home_goals IS NOT NULL
        ORDER BY date
        """,
        [league, start, end],
    ).df()
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def _load_odds(con, match_ids: list[str], markets_wanted: tuple[str, ...]) -> pd.DataFrame:
    if not match_ids:
        return pd.DataFrame()
    match_placeholders = ", ".join("?" * len(match_ids))
    market_placeholders = ", ".join("?" * len(markets_wanted))
    return con.execute(
        f"""
        SELECT match_id, bookmaker, phase, market, selection, line, price
        FROM odds
        WHERE match_id IN ({match_placeholders})
          AND market IN ({market_placeholders})
        """,
        list(match_ids) + list(markets_wanted),
    ).df()


def _pick_price(group: pd.DataFrame, phase: str, preference: tuple[str, ...]):
    """First available price for a phase, in order of source preference."""
    subset = group[group["phase"] == phase]
    for bookmaker in preference:
        row = subset[subset["bookmaker"] == bookmaker]
        if not row.empty:
            return float(row["price"].iloc[0]), bookmaker
    return np.nan, None


def _market_probabilities(
    group: pd.DataFrame, method: str, preference: tuple[str, ...] = FAIR_LINE_PREFERENCE
) -> dict[tuple, tuple[float, str]]:
    """Margin-free closing probabilities and the book each came from.

    A market is only usable once every leg of it is priced: two thirds of a
    1X2 tells you nothing about where the margin sits.

    The fallback from closing to opening prices is decided per market, not
    once for the whole match. This source carries closing prices for 1X2 long
    before it carries them for totals, so a global fallback would silently
    leave the totals market unpriced whenever any closing price existed.

    The supplying bookmaker is returned alongside the probability, not thrown
    away. Which book answered is part of the measurement: a CLV series that
    changes benchmark halfway through is not one series, and without this the
    change leaves no trace in the output.
    """
    out: dict[tuple, tuple[float, str]] = {}
    keyed = group.copy()
    # Both halves of a handicap must land in the same group even though they
    # carry opposite lines, so lines are normalised to the home team's view
    # for grouping purposes only.
    keyed["group_line"] = [
        -line if (market == "asian_handicap" and selection == "away") else line
        for market, selection, line in zip(
            keyed["market"], keyed["selection"], keyed["line"]
        )
    ]
    for (market, _group_line), rows in keyed.groupby(["market", "group_line"], dropna=False):
        for phase in ("close", "open"):
            phase_rows = rows[rows["phase"] == phase]
            if phase_rows.empty:
                continue
            if _fill_market(out, market, phase_rows, method, preference):
                break
    return out


def _fill_market(
    out: dict, market, rows: pd.DataFrame, method: str,
    preference: tuple[str, ...] = FAIR_LINE_PREFERENCE,
) -> bool:
    """Add one complete market's fair probabilities. True if any were added.

    The probabilities that come back are *conditional on the bet not pushing*,
    because normalising a market's prices to sum to one is exactly what
    removes the pushed portion. That matters for handicaps and whole-number
    totals, and it is the scale the model must be converted to before the two
    are compared.
    """
    expected = 3 if market == "1x2" else 2
    available = set(rows["bookmaker"])
    ordered = [b for b in preference if b in available]
    # Anything not named in the preference is still tried, in name order, so a
    # market is priced whenever it can be. That maximises coverage at the cost
    # of letting a soft book supply the benchmark, which is why the choice is
    # recorded rather than left implicit.
    ordered += sorted(available - set(ordered))
    for bookmaker in ordered:
        book_rows = rows[rows["bookmaker"] == bookmaker]
        deduped = book_rows.drop_duplicates("selection")
        if len(deduped) != expected:
            continue
        try:
            fair = pricing.remove_margin(deduped["price"].tolist(), method=method)
        except ValueError:
            continue
        for selection, line, probability in zip(
            deduped["selection"], deduped["line"], fair
        ):
            out.setdefault(
                (market, selection, _line_key(line)), (float(probability), bookmaker)
            )
        return True
    return False


def _line_key(line) -> float | None:
    if line is None or (isinstance(line, float) and np.isnan(line)):
        return None
    return round(float(line), 3)


def run_backtest(con, config: BacktestConfig, verbose: bool = True) -> BacktestResult:
    """Refit weekly, price the book's own lines, settle the results."""
    fixtures = _load_fixtures(con, config.league, config.start, config.end)
    if fixtures.empty:
        raise ValueError(
            f"No {config.league} matches with results between "
            f"{config.start} and {config.end}."
        )

    odds = _load_odds(con, fixtures["match_id"].tolist(), config.markets)
    odds_by_match = dict(tuple(odds.groupby("match_id"))) if not odds.empty else {}

    records: list[dict] = []
    total_records: list[dict] = []
    notes: list[str] = []
    cursor = config.start
    refits = 0
    matches_seen = 0

    while cursor <= config.end:
        window_end = cursor + dt.timedelta(days=config.step_days)
        batch = fixtures[(fixtures["date"] >= cursor) & (fixtures["date"] < window_end)]
        cursor = window_end
        if batch.empty:
            continue

        as_of = batch["date"].min()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", base.ConvergenceWarning)
                bundle = predict_mod.build_models(
                    con, config.league, as_of,
                    half_life_days=config.half_life_days, ridge=config.ridge,
                    use_cache=False, fit_counts=config.fit_count_models,
                )
        except base.InsufficientData as exc:
            notes.append(f"{as_of}: skipped, {exc}")
            continue

        refits += 1
        for row in batch.itertuples():
            matches_seen += 1
            match_records, total_record = _score_match(
                row, bundle, odds_by_match.get(row.match_id), config
            )
            records += match_records
            total_records.append(total_record)

    predictions = pd.DataFrame(records)
    if verbose:
        print(
            f"Refitted {refits} times, scored {matches_seen} matches, "
            f"{len(predictions)} selections."
        )
    return BacktestResult(
        predictions=predictions, config=config, refits=refits,
        matches=matches_seen, notes=notes,
        match_totals=pd.DataFrame(total_records),
    )


def _score_match(row, bundle, match_odds: pd.DataFrame | None, config):
    """Price and settle every selection the bookmaker offered for one match.

    Returns the selection records alongside one match-level record holding the
    predicted distribution of the total. The matrix is built before the odds
    are checked so that the second of those survives a match no bookmaker
    priced: whether the goals distribution is the right shape is a question
    about the model, and throwing away matches because nobody quoted them
    would answer it on a sample selected by the bookmaker.
    """
    matrix = bundle.goals.score_matrix(row.home_team, row.away_team)
    total_pmf = markets.total_distribution(matrix)
    total_record = {
        "match_id": row.match_id,
        "date": row.date,
        "observed_total": int(row.home_goals) + int(row.away_goals),
        "total_pmf": total_pmf,
    }

    if match_odds is None or match_odds.empty:
        return [], total_record

    market_probabilities = _market_probabilities(
        match_odds, config.margin_method, config.fair_line_preference
    )
    records: list[dict] = []

    grouped = match_odds.groupby(["market", "selection", "line"], dropna=False)
    for (market, selection, line), group in grouped:
        line_value = _line_key(line)
        priced = markets.price_selection(matrix, market, selection, line_value)
        if priced is None or priced.probability <= 0:
            continue

        outcome = settlement.settle(
            market, selection, line_value,
            int(row.home_goals), int(row.away_goals),
            total_corners=row.total_corners, total_cards=row.total_cards,
        )
        if outcome is None:
            continue

        price_taken, taken_from = _pick_price(group, "open", config.price_source)
        price_close, closed_at = _pick_price(group, "close", config.closing_source)
        same_book_close = (
            _same_book_price(group, "close", taken_from) if taken_from else np.nan
        )
        closing_fair, fair_line_source = market_probabilities.get(
            (market, selection, line_value), (np.nan, None)
        )

        expected_value = (
            pricing.expected_value(priced.probability, price_taken, priced.push_probability)
            if np.isfinite(price_taken) else np.nan
        )
        records.append(
            {
                "match_id": row.match_id,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "market": market,
                "selection": selection,
                "line": line_value,
                "model_probability": priced.probability,
                "push_probability": priced.push_probability,
                "fair_price": priced.fair_price,
                "market_probability": closing_fair,
                # Which book the benchmark came from. CLV is only comparable
                # across seasons where this is the same.
                "fair_line_source": fair_line_source,
                "price_taken": price_taken,
                "price_source": taken_from,
                "price_close": price_close,
                "closing_source": closed_at,
                "same_book_close": same_book_close,
                "expected_value": expected_value,
                "win_fraction": outcome.win,
                "push_fraction": outcome.push,
                "profit_at_taken": (
                    outcome.profit(price_taken) if np.isfinite(price_taken) else np.nan
                ),
                "profit_at_close": (
                    outcome.profit(price_close) if np.isfinite(price_close) else np.nan
                ),
                "model_conditional": _conditional(
                    priced.probability, priced.push_probability
                ),
                # The sound measure: was the price taken better than the
                # margin-free closing line said it was worth? Both sides are
                # conditional on the bet not pushing, which is the scale a
                # normalised set of bookmaker prices is already on.
                "clv": (
                    price_taken * closing_fair - 1.0
                    if np.isfinite(price_taken) and np.isfinite(closing_fair)
                    else np.nan
                ),
                # The raw price movement at the same bookmaker. Useful, but on
                # its own it conflates line movement with margin changes.
                "price_movement": (
                    price_taken / same_book_close - 1.0
                    if np.isfinite(price_taken)
                    and np.isfinite(same_book_close)
                    and same_book_close > 0
                    else np.nan
                ),
                # Every bookmaker's own open price against the same fair
                # closing line, so a per-bookmaker comparison can be built
                # without rerunning the backtest once per book. price_taken
                # above answers "what would I have got shopping for the best
                # price"; this answers "what would each specific book alone
                # have given me".
                "book_prices": _all_book_clv(group, closing_fair),
            }
        )
    return records, total_record


def _all_book_clv(group: pd.DataFrame, closing_fair: float) -> dict[str, float]:
    """Every bookmaker's own open price, converted to CLV against one fair line."""
    if not np.isfinite(closing_fair):
        return {}
    opens = group[group["phase"] == "open"].drop_duplicates("bookmaker")
    return {
        row.bookmaker: float(row.price * closing_fair - 1.0)
        for row in opens.itertuples()
        if row.price > 1.0
    }


def _conditional(probability: float, push: float) -> float:
    """Win probability given that the bet does not push.

    A handicap at -1.0 that wins 45% of the time and pushes 20% of the time is
    a 56% shot on the money actually at risk, and 56% is the number a
    bookmaker's normalised prices are quoting.
    """
    remaining = 1.0 - push
    return probability / remaining if remaining > 1e-9 else np.nan


def _same_book_price(group: pd.DataFrame, phase: str, bookmaker: str | None) -> float:
    """A specific bookmaker's price for a phase, or NaN.

    Comparing an early price at one book against a closing price at another
    measures the difference in their margins as much as any movement in the
    line, so same-book comparisons are kept separate.
    """
    if bookmaker is None:
        return np.nan
    row = group[(group["phase"] == phase) & (group["bookmaker"] == bookmaker)]
    return float(row["price"].iloc[0]) if not row.empty else np.nan
