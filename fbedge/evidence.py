"""What track record does this price actually have?

A price with no track record attached is the most misleading thing this project
could publish. "+6% expected value on Arsenal" reads like a recommendation; the
same line next to "this market has never been backtested as a bet, because the
source carries no historical prices for it" reads like what it is. The README
is already careful never to put a modelled estimate next to a fact about two
matches without saying which is which. This module applies that standard one
level up, to markets rather than to numbers.

**Three statuses, and the difference between them is what data exists.**

*Backtested.* The source carries historical prices, so the model's selections
could be settled as bets against a real market and closing line value is
measurable. Only three markets qualify: 1X2, over/under 2.5 goals and Asian
handicap. **The brief that commissioned this module listed BTTS among them and
that is wrong** - `normalize.extract_odds` produces exactly three markets
because `notes.txt` publishes exactly three, and there has never been a
both-teams-to-score price in this data.

*Calibration only.* No price exists to bet into, but the outcome is recorded,
so "when the model says 30%, does it happen 30% of the time" is answerable and
has been answered. Corners and cards are the headline case and BACKLOG B7 is
about them, but BTTS, double chance, the team totals and the half-time markets
are all in the same position. "The model is well calibrated on corners" and
"there is an edge in corner markets" are different claims and only the first is
supported by anything here. Confirming the second needs a paid odds feed.

*Untested.* Priced by the model, and no calibration run has scored it yet on
this league. A newly added market is untested until `scripts/build_evidence.py`
has been run, and says so rather than borrowing another market's record.

**Evidence is stored, not computed on demand.** Producing it means a
walk-forward refit per league, which is minutes rather than milliseconds, and a
scan that ran one on every invocation would be a scan nobody runs. So it lives
in a table with the date it was computed and the window it covers, and every
consumer reads it. A market with no row in that table is reported as untested -
never silently as a bare number.

**Nothing here is a project-wide average.** Card models in particular are
league-dependent: the referee column is missing in some leagues, and English
and Scottish yellows exclude the first yellow of a second-yellow red while
European competitions count both. So evidence is keyed by (league, market), and
`card_conditions` surfaces which of those apply to the fixture being priced.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import backtest as backtest_mod
from . import evaluation

TABLE_NAME = "market_evidence"

BACKTESTED = "backtested"
CALIBRATION_ONLY = "calibration-only"
UNTESTED = "untested"

# The markets the source prices, and therefore the only ones that can carry a
# closing line value figure. Read from the backtest rather than restated, so
# the two cannot disagree about what is bettable.
PRICED_MARKETS = tuple(backtest_mod.BETTABLE_MARKETS)

# Why each unpriced market has no price, in the source's own terms. Shown to
# the reader instead of a bare "untested", because "nobody has run it yet" and
# "no such price has ever existed" are different situations.
NO_PRICE_REASON = {
    "total_corners": "football-data.co.uk carries no corner prices, ever",
    "home_total_corners": "football-data.co.uk carries no corner prices, ever",
    "away_total_corners": "football-data.co.uk carries no corner prices, ever",
    "corner_handicap": "football-data.co.uk carries no corner prices, ever",
    "total_cards": "football-data.co.uk carries no card prices, ever",
    "home_total_cards": "football-data.co.uk carries no card prices, ever",
    "away_total_cards": "football-data.co.uk carries no card prices, ever",
    "card_handicap": "football-data.co.uk carries no card prices, ever",
    "btts": "the source publishes no both-teams-to-score price",
    "double_chance": "the source publishes no double-chance price",
    "draw_no_bet": "the source publishes no draw-no-bet price",
    "home_goals": "the source publishes no team-total price",
    "away_goals": "the source publishes no team-total price",
    "odd_even_goals": "the source publishes no odd/even price",
    "winning_margin": "the source publishes no winning-margin price",
    "correct_score": "the source publishes no correct-score price",
    "1x2_ht": "the source publishes no half-time price",
    "total_goals_ht": "the source publishes no half-time price",
}


def status(market: str, row: pd.Series | None = None) -> str:
    """Which of the three a market is, given whatever evidence exists.

    A priced market with no evidence row is still `untested` - the price
    existing is not the same as somebody having measured against it.
    """
    if row is None or int(row.get("n", 0) or 0) == 0:
        return UNTESTED
    if market in PRICED_MARKETS and int(row.get("n_bets", 0) or 0) > 0:
        return BACKTESTED
    return CALIBRATION_ONLY


# --------------------------------------------------------------------------
# Computing it
# --------------------------------------------------------------------------

def compute(
    con,
    league: str,
    start: dt.date,
    end: dt.date,
    markets_wanted: tuple[str, ...] = backtest_mod.DEFAULT_CALIBRATION_MARKETS,
    **config_kwargs,
) -> pd.DataFrame:
    """One walk-forward, scored per market. Minutes, not milliseconds.

    Runs both passes at once: the priced markets are settled against real
    bookmaker lines and yield CLV, while every market in `markets_wanted` is
    also priced from the model's own lines and settled with no price, which is
    what makes calibration measurable for the markets nobody quotes.

    Both come from the same fits and the same matches, so the calibration
    figure next to a backtested market describes the same model run that
    produced its CLV.
    """
    settings = backtest_mod.BacktestConfig(
        league=league, start=start, end=end,
        calibration_markets=tuple(markets_wanted),
        **config_kwargs,
    )
    result = backtest_mod.run_backtest(con, settings, verbose=False)
    return summarise(result, league, start, end)


def summarise(result, league: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Turn one backtest into one row per market."""
    predictions = result.predictions
    rows: list[dict] = []
    if predictions.empty:
        return pd.DataFrame(rows, columns=_COLUMNS)

    bets = result.bets
    computed_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    for market in sorted(predictions["market"].unique()):
        # Calibration is read from the priceless pass where one exists, so a
        # backtested market is scored on every match rather than only on the
        # selections a bookmaker chose to offer. Which selections a book
        # quotes is not a random sample of the model's opinions.
        block = predictions[predictions["market"] == market]
        priceless = block[block["priceless"]] if "priceless" in block else block.iloc[0:0]
        scored = priceless if not priceless.empty else block
        scores = evaluation.score_market(scored, market)
        if not scores.get("n"):
            continue

        usable = scored[scored["push_fraction"] < 0.5]
        slope = evaluation.calibration_slope(
            usable["model_conditional"], (usable["win_fraction"] > 0.5).astype(float)
        )

        market_bets = bets[bets["market"] == market] if not bets.empty else bets
        clv = {"n": 0}
        if len(market_bets):
            valid = market_bets.dropna(subset=["clv"])
            if len(valid):
                clv = evaluation.clustered_mean(valid["clv"], valid["match_id"])

        row = {
            "league": league,
            "market": market,
            "n": int(scores["n"]),
            "n_matches": int(scored["match_id"].nunique()),
            "model_log_loss": float(scores["model_log_loss"]),
            "base_rate_log_loss": float(scores["base_rate_log_loss"]),
            "market_log_loss": float(scores.get("market_log_loss", np.nan)),
            "calibration_slope": float(slope.get("slope", np.nan)),
            "calibration_slope_se": float(slope.get("slope_se", np.nan)),
            "n_bets": int(clv.get("n", 0)),
            "mean_clv": float(clv.get("mean", np.nan)),
            "clv_se": float(clv.get("se", np.nan)),
            "window_start": start,
            "window_end": end,
            "computed_at": computed_at,
        }
        row["status"] = status(market, pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows, columns=_COLUMNS)


_COLUMNS = [
    "league", "market", "status", "n", "n_matches",
    "model_log_loss", "base_rate_log_loss", "market_log_loss",
    "calibration_slope", "calibration_slope_se",
    "n_bets", "mean_clv", "clv_se",
    "window_start", "window_end", "computed_at",
]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def create_table(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            league                VARCHAR NOT NULL,
            market                VARCHAR NOT NULL,
            status                VARCHAR NOT NULL,
            n                     INTEGER,
            n_matches             INTEGER,
            model_log_loss        DOUBLE,
            base_rate_log_loss    DOUBLE,
            market_log_loss       DOUBLE,
            calibration_slope     DOUBLE,
            calibration_slope_se  DOUBLE,
            n_bets                INTEGER,
            mean_clv              DOUBLE,
            clv_se                DOUBLE,
            window_start          DATE,
            window_end            DATE,
            computed_at           TIMESTAMP
        )
        """
    )


def write(con, frame: pd.DataFrame) -> int:
    """Replace the evidence for the leagues supplied, keeping the others.

    Scoped per league for the reason `build_xg.py` learned the hard way: a
    single-league run that wiped the other four looked exactly like a
    successful run until somebody opened the app.
    """
    create_table(con)
    if frame.empty:
        return 0
    leagues = sorted({str(v) for v in frame["league"].unique()})
    con.register("incoming_evidence", frame[_COLUMNS])
    con.execute(
        f"DELETE FROM {TABLE_NAME} WHERE league IN "
        f"({','.join(repr(name) for name in leagues)})"
    )
    columns = ", ".join(f'"{c}"' for c in _COLUMNS)
    con.execute(
        f"INSERT INTO {TABLE_NAME} ({columns}) SELECT {columns} FROM incoming_evidence"
    )
    con.unregister("incoming_evidence")
    return len(frame)


def load(con, league: str | None = None) -> pd.DataFrame:
    """Stored evidence, or an empty frame when none has been computed."""
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME]
    ).fetchone()
    if not exists:
        return pd.DataFrame(columns=_COLUMNS)
    if league:
        return con.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE league = ? ORDER BY market", [league]
        ).df()
    return con.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY league, market").df()


# --------------------------------------------------------------------------
# Turning it into something a reader sees
# --------------------------------------------------------------------------

def describe(market: str, row: pd.Series | None) -> str:
    """One line of evidence for one market, fit to sit next to a price.

    Deliberately blunt where the answer is weak. A market nobody has scored
    says so in the first three words rather than in a footnote.
    """
    state = status(market, row)
    if state == UNTESTED:
        reason = NO_PRICE_REASON.get(market)
        if reason:
            return (
                f"UNTESTED - no backtest has scored this market yet, and it can "
                f"never be backtested as a bet: {reason}."
            )
        return "UNTESTED - no backtest has scored this market on this league yet."

    slope = row.get("calibration_slope")
    slope_text = (
        f"calibration slope {slope:.2f}" if slope == slope else "calibration unmeasured"
    )
    sample = f"n={int(row['n'])} over {int(row['n_matches'])} matches"

    if state == CALIBRATION_ONLY:
        reason = NO_PRICE_REASON.get(market, "the source publishes no price for it")
        return (
            f"CALIBRATION ONLY - {slope_text}, {sample}. Never backtested as a "
            f"bet, because {reason}, so there is no evidence of an edge here, "
            "only of the model being about right. Confirming an edge needs a "
            "paid odds feed."
        )

    clv, se = row.get("mean_clv"), row.get("clv_se")
    if clv == clv and se == se and se > 0:
        clv_text = f"CLV {clv * 100:+.2f}% ({clv / se:+.1f} SE) on {int(row['n_bets'])} bets"
    else:
        clv_text = f"CLV not measurable on {int(row.get('n_bets', 0))} bets"
    gap = row.get("model_log_loss") - row.get("market_log_loss")
    gap_text = (
        f", log loss {gap:+.4f} against the closing line" if gap == gap else ""
    )
    return f"BACKTESTED - {clv_text}, {slope_text}, {sample}{gap_text}."


def labels(frame: pd.DataFrame, markets_wanted=None) -> dict[str, str]:
    """`{market: one-line evidence}` for a league, ready to print or render."""
    indexed = (
        frame.set_index("market") if not frame.empty and "market" in frame
        else pd.DataFrame()
    )
    wanted = list(markets_wanted) if markets_wanted is not None else list(indexed.index)
    return {
        market: describe(
            market, indexed.loc[market] if market in indexed.index else None
        )
        for market in wanted
    }


def short_labels(frame: pd.DataFrame, markets_wanted=None) -> dict[str, str]:
    """The same, cut to a tag that fits on a table row."""
    indexed = (
        frame.set_index("market") if not frame.empty and "market" in frame
        else pd.DataFrame()
    )
    wanted = list(markets_wanted) if markets_wanted is not None else list(indexed.index)
    out = {}
    for market in wanted:
        row = indexed.loc[market] if market in indexed.index else None
        state = status(market, row)
        if state == UNTESTED:
            out[market] = "untested"
        elif state == CALIBRATION_ONLY:
            slope = row.get("calibration_slope")
            out[market] = f"calibration only, slope {slope:.2f}, n={int(row['n'])}"
        else:
            clv, se = row.get("mean_clv"), row.get("clv_se")
            piece = (
                f"CLV {clv * 100:+.2f}%" if clv == clv else "CLV n/a"
            )
            out[market] = f"backtested, {piece}, n={int(row.get('n_bets', 0))}"
    return out


# --------------------------------------------------------------------------
# The conditions a card price depends on
# --------------------------------------------------------------------------

def card_conditions(con, league: str, season_start_year: int | None = None) -> dict:
    """What a card model on this league is and is not able to know.

    Two league-dependent facts the README currently states once, far away from
    any card price, which is the wrong place for them.

    **The referee.** `models/counts.py` fits a per-referee card multiplier, and
    the source does not report the referee for every league. A league with no
    referee column gets card prices at a single league-average referee, and the
    largest single driver of a card total is then simply absent from the model.

    **The counting convention**, from the source's own notes: in England and
    Scotland a first yellow is *not* recorded when a second turns it into a
    red, while European competitions record both. English card totals therefore
    run slightly below continental ones for identical on-pitch events. Models
    are fitted per league so the difference is absorbed by the intercept, but a
    card line quoted here is not comparable with one quoted for another
    country, and a reader deserves to be told which convention is in play.
    """
    clauses, params = ["league = ?"], [league]
    if season_start_year is not None:
        clauses.append("season_start_year = ?")
        params.append(season_start_year)
    where = " AND ".join(clauses)
    row = con.execute(
        f"""
        SELECT COUNT(*) AS matches,
               COUNT(referee) AS with_referee,
               COUNT(home_yellows) AS with_cards
        FROM matches WHERE {where}
        """,
        params,
    ).fetchone()
    matches, with_referee, with_cards = (int(v) for v in row)
    country = _COUNTRY_BY_LEAGUE.get(league)
    second_yellow_excluded = country in ("England", "Scotland")
    return {
        "league": league,
        "country": country,
        "matches": matches,
        "referee_coverage": with_referee / matches if matches else 0.0,
        "card_coverage": with_cards / matches if matches else 0.0,
        "has_referee_effects": with_referee > 0,
        "second_yellow_excluded": second_yellow_excluded,
        "note": _card_note(with_referee, matches, second_yellow_excluded, country),
    }


def _card_note(with_referee, matches, second_yellow_excluded, country) -> str:
    parts = []
    if matches == 0:
        return "No matches for this league, so nothing about cards can be said."
    if with_referee == 0:
        parts.append(
            "The source reports no referee for this league, so cards are priced "
            "at the league-average referee and the largest single driver of a "
            "card total is missing from the model."
        )
    else:
        parts.append(
            f"Referee known for {with_referee / matches:.0%} of matches, so the "
            "fitted referee multiplier applies where it is."
        )
    if second_yellow_excluded:
        parts.append(
            f"{country} excludes the first yellow when a second turns it into a "
            "red, so these totals run slightly below continental ones for the "
            "same events and are not comparable across borders."
        )
    else:
        parts.append(
            f"{country} records both yellows when a second turns into a red, "
            "unlike England and Scotland, so these totals are not comparable "
            "with English ones."
        )
    return " ".join(parts)


# Imported lazily rather than at module scope to keep `config` out of the
# import graph of a module the app reloads on every rerun.
def _country_map() -> dict:
    from . import config

    return dict(config.LEAGUE_COUNTRY)


_COUNTRY_BY_LEAGUE = _country_map()
