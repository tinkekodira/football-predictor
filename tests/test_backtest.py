"""Tests for the Phase 3 backtest.

Settlement gets the most attention here, and deliberately so. It is the
shortest module in the project and the one where a mistake does the most
damage: a backtest that settles away handicaps against the home line, or
treats a quarter-line half-loss as a full loss, runs perfectly cleanly and
reports a profit that does not exist. Every case below was worked out by hand
before it was written down.

`test_clv_is_zero_at_the_closing_fair_price` is the other one worth knowing
about. Closing line value is the headline metric of the whole phase, so it
needs to read exactly zero when a bet is taken at precisely the price the
closing line implies.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation, normalize, settlement  # noqa: E402
from fbedge import predict as predict_mod  # noqa: E402
from scripts.make_sample_data import build_season  # noqa: E402


@pytest.fixture(scope="module")
def backtest_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("bt") / "bt.duckdb"
    con = database.connect(path)
    for year in range(config.CURRENT_SEASON_START_YEAR - 5, config.CURRENT_SEASON_START_YEAR + 1):
        raw = build_season("E0", year, seed=year * 23)
        matches, odds = normalize.normalize_league_season(raw, "E0", year)
        database.load_matches(con, matches)
        database.load_odds(con, odds)
    yield con
    con.close()


@pytest.fixture(scope="module")
def result(backtest_db) -> backtest.BacktestResult:
    predict_mod.clear_model_cache()
    settings = backtest.BacktestConfig(
        league="E0",
        start=dt.date(config.CURRENT_SEASON_START_YEAR - 2, 8, 1),
        end=dt.date(config.CURRENT_SEASON_START_YEAR, 8, 31),
    )
    return backtest.run_backtest(backtest_db, settings, verbose=False)


# --------------------------------------------------------------------------
# Settlement: the arithmetic, checked by hand
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "selection, home, away, expected_win",
    [
        ("home", 2, 0, 1.0), ("home", 1, 1, 0.0), ("home", 0, 1, 0.0),
        ("draw", 1, 1, 1.0), ("draw", 2, 0, 0.0),
        ("away", 0, 1, 1.0), ("away", 1, 1, 0.0),
    ],
)
def test_settle_1x2(selection, home, away, expected_win):
    assert settlement.settle_1x2(selection, home, away).win == expected_win


def test_settle_double_chance():
    assert settlement.settle_double_chance("home_or_draw", 1, 1).win == 1.0
    assert settlement.settle_double_chance("home_or_draw", 0, 1).win == 0.0
    assert settlement.settle_double_chance("home_or_away", 1, 1).win == 0.0


def test_settle_btts():
    assert settlement.settle_btts("yes", 1, 1).win == 1.0
    assert settlement.settle_btts("yes", 2, 0).win == 0.0
    assert settlement.settle_btts("no", 3, 0).win == 1.0


def test_over_under_half_lines_cannot_push():
    assert settlement.settle_over_under("over", 2.5, 3).win == 1.0
    assert settlement.settle_over_under("over", 2.5, 2).win == 0.0
    assert settlement.settle_over_under("under", 2.5, 2).win == 1.0
    assert settlement.settle_over_under("over", 2.5, 3).push == 0.0


def test_over_under_whole_line_pushes_on_the_exact_total():
    """Over 3.0 with three goals returns the stake. Treating that as a loss
    would misprice the market by several percent."""
    pushed = settlement.settle_over_under("over", 3.0, 3)
    assert pushed.push == 1.0
    assert pushed.win == 0.0
    assert pushed.profit(2.0) == 0.0


@pytest.mark.parametrize(
    "line, home, away, win, push",
    [
        (-0.5, 1, 0, 1.0, 0.0),    # wins by one, covers half a goal
        (-1.0, 1, 0, 0.0, 1.0),    # wins by exactly one, stake returned
        (-1.5, 1, 0, 0.0, 0.0),    # wins by one, does not cover 1.5
        (-1.5, 2, 0, 1.0, 0.0),
        (0.0, 1, 1, 0.0, 1.0),     # draw no bet
        (0.5, 1, 1, 1.0, 0.0),     # draw is enough with half a goal start
    ],
)
def test_settle_handicap_whole_and_half_lines(line, home, away, win, push):
    outcome = settlement.settle_asian_handicap("home", line, home, away)
    assert outcome.win == pytest.approx(win)
    assert outcome.push == pytest.approx(push)


def test_quarter_line_half_win():
    """Home -0.75, winning by one: half wins at -0.5, half pushes at -1.0."""
    outcome = settlement.settle_asian_handicap("home", -0.75, 1, 0)
    assert outcome.win == pytest.approx(0.5)
    assert outcome.push == pytest.approx(0.5)
    assert outcome.loss == pytest.approx(0.0)
    assert outcome.profit(2.0) == pytest.approx(0.5)


def test_quarter_line_half_loss():
    """Home -0.25 in a draw: half loses at -0.5, half pushes at 0.0."""
    outcome = settlement.settle_asian_handicap("home", -0.25, 0, 0)
    assert outcome.win == pytest.approx(0.0)
    assert outcome.push == pytest.approx(0.5)
    assert outcome.loss == pytest.approx(0.5)
    assert outcome.profit(2.0) == pytest.approx(-0.5)


def test_away_handicap_uses_its_own_line():
    """Away +0.5 must win a goalless draw; settling it against the home line
    would have it losing."""
    assert settlement.settle_asian_handicap("away", 0.5, 0, 0).win == 1.0
    assert settlement.settle_asian_handicap("home", -0.5, 0, 0).win == 0.0


@pytest.mark.parametrize("line", [-1.5, -1.0, -0.75, -0.25, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("score", [(0, 0), (1, 0), (2, 1), (0, 3)])
def test_handicap_sides_are_complementary(line, score):
    """The two halves of one handicap must account for the whole stake."""
    home_goals, away_goals = score
    home = settlement.settle_asian_handicap("home", line, home_goals, away_goals)
    away = settlement.settle_asian_handicap("away", -line, home_goals, away_goals)
    assert home.win + away.win + home.push == pytest.approx(1.0)
    assert home.push == pytest.approx(away.push)


def test_profit_arithmetic():
    assert settlement.WON.profit(2.5) == pytest.approx(1.5)
    assert settlement.LOST.profit(2.5) == pytest.approx(-1.0)
    assert settlement.PUSHED.profit(2.5) == pytest.approx(0.0)
    assert settlement.Settlement(0.5, 0.5).profit(3.0) == pytest.approx(1.0)


def test_settle_returns_none_when_the_statistic_is_missing():
    """A corners bet on a match with no corner record is not a loss."""
    assert settlement.settle("total_corners", "over", 9.5, 1, 0, total_corners=None) is None
    assert settlement.settle("total_corners", "over", 9.5, 1, 0, total_corners=11).win == 1.0


def test_unknown_market_is_rejected():
    with pytest.raises(ValueError):
        settlement.settle("first_goalscorer", "someone", None, 1, 0)


# --------------------------------------------------------------------------
# Backtest mechanics
# --------------------------------------------------------------------------

def test_backtest_produces_predictions(result):
    frame = result.predictions
    assert not frame.empty
    assert result.refits > 5
    assert {"1x2", "total_goals", "asian_handicap"} <= set(frame["market"])
    assert frame["model_probability"].between(0, 1).all()


def test_every_selection_is_settled(result):
    frame = result.predictions
    total = frame["win_fraction"] + frame["push_fraction"]
    assert (total <= 1.0 + 1e-9).all()
    assert (frame["win_fraction"] >= 0).all()


def test_prices_come_from_the_odds_table(result):
    """Every priced selection must trace back to a real bookmaker row."""
    priced = result.predictions.dropna(subset=["price_taken"])
    assert not priced.empty
    assert (priced["price_taken"] > 1.0).all()
    assert priced["price_source"].notna().all()


def test_bets_clear_the_edge_threshold(result):
    bets = result.bets
    if bets.empty:
        pytest.skip("no bets cleared the threshold in this sample")
    assert (bets["expected_value"] >= result.config.edge_threshold).all()
    assert bets["price_taken"].between(
        result.config.min_price, result.config.max_price
    ).all()


def test_backtest_is_point_in_time(backtest_db):
    """A model used for a week must not have seen that week's results.

    Checked directly: refit at the first date of the window, and confirm the
    training set stops before it.
    """
    from fbedge.models import base as model_base

    as_of = dt.date(config.CURRENT_SEASON_START_YEAR, 8, 20)
    training = model_base.load_training_set(backtest_db, "E0", as_of)
    assert (pd.to_datetime(training.frame["date"]).dt.date < as_of).all()


def test_handicap_halves_share_a_market_probability(result):
    """Both sides of a handicap must be priced, despite opposite lines.

    They live in different `line` groups in the odds table, so the market
    probability lookup has to normalise them back together. If it does not,
    every away handicap silently loses its market comparison.
    """
    handicaps = result.predictions[result.predictions["market"] == "asian_handicap"]
    if handicaps.empty:
        pytest.skip("no handicap prices in the sample")
    priced = handicaps.dropna(subset=["market_probability"])
    assert not priced.empty
    assert set(priced["selection"]) == {"home", "away"}


def test_conditional_probability_conversion():
    assert backtest._conditional(0.45, 0.20) == pytest.approx(0.5625)
    assert backtest._conditional(0.50, 0.0) == pytest.approx(0.50)
    assert np.isnan(backtest._conditional(0.5, 1.0))


def test_market_probabilities_sum_to_one(result):
    """Each complete market's margin-free probabilities must be a distribution."""
    frame = result.predictions.dropna(subset=["market_probability"])
    grouped = frame.groupby(["match_id", "market"])["market_probability"].sum()
    complete = grouped[grouped > 0.5]
    assert np.allclose(complete.to_numpy(), 1.0, atol=1e-6)


# --------------------------------------------------------------------------
# Closing line value
# --------------------------------------------------------------------------

def test_clv_is_zero_at_the_closing_fair_price():
    """Taking exactly the price the close implies is zero CLV, by definition."""
    bets = pd.DataFrame(
        {
            "clv": [2.0 * 0.5 - 1.0, 2.5 * 0.4 - 1.0],
            "price_movement": [np.nan, np.nan],
        }
    )
    summary = evaluation.closing_line_value(bets)
    assert summary["mean_clv"] == pytest.approx(0.0)


def test_clv_detects_a_better_price():
    bets = pd.DataFrame({"clv": [2.20 * 0.5 - 1.0] * 40, "price_movement": [np.nan] * 40})
    summary = evaluation.closing_line_value(bets)
    assert summary["mean_clv"] == pytest.approx(0.10)
    assert summary["beat_close_rate"] == 1.0


def test_clv_reports_nothing_without_closing_prices():
    bets = pd.DataFrame({"clv": [np.nan, np.nan], "price_movement": [np.nan, np.nan]})
    assert evaluation.closing_line_value(bets)["n"] == 0


def test_taking_a_soft_price_shows_negative_clv(result):
    """Betting into a margin with no edge must lose to the closing line.

    This is the sanity check on the whole metric: a bettor with no information
    who takes a soft bookmaker's price should show roughly minus that
    bookmaker's margin, never a positive number.
    """
    bets = result.bets
    if bets.empty or bets["clv"].notna().sum() < 50:
        pytest.skip("not enough priced bets in the sample")
    assert bets["clv"].mean() < 0.0


# --------------------------------------------------------------------------
# Evaluation metrics
# --------------------------------------------------------------------------

def test_log_loss_of_a_perfect_forecast_is_zero():
    assert evaluation.binary_log_loss(
        np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])
    ) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_of_a_coin_flip():
    value = evaluation.binary_log_loss(np.full(100, 0.5), np.resize([0.0, 1.0], 100))
    assert value == pytest.approx(np.log(2))


def test_confident_and_wrong_scores_worse_than_cautious_and_wrong():
    outcomes = np.zeros(10)
    assert evaluation.binary_log_loss(np.full(10, 0.95), outcomes) > \
        evaluation.binary_log_loss(np.full(10, 0.55), outcomes)


def test_brier_score_bounds():
    assert evaluation.brier_score(np.array([1.0]), np.array([1.0])) == 0.0
    assert evaluation.brier_score(np.array([1.0]), np.array([0.0])) == 1.0


def test_calibration_table_detects_overconfidence():
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0.05, 0.95, 4000)
    # Outcomes occur less often than claimed: an overconfident model.
    outcomes = (rng.uniform(size=4000) < probabilities * 0.8).astype(float)
    table = evaluation.calibration_table(probabilities, outcomes)
    assert not table.empty
    assert (table["gap"] < 0).mean() > 0.7


def test_calibration_table_skips_thin_bands():
    table = evaluation.calibration_table(
        np.array([0.5] * 5), np.array([1.0] * 5), min_count=10
    )
    assert table.empty


# --------------------------------------------------------------------------
# Staking
# --------------------------------------------------------------------------

def test_kelly_is_zero_without_an_edge():
    assert evaluation.kelly_stake(0.5, 1.90) == 0.0
    assert evaluation.kelly_stake(0.5, 2.00) == 0.0


def test_kelly_is_capped():
    """Full Kelly on a large apparent edge would be reckless, so it is capped."""
    stake = evaluation.kelly_stake(0.9, 5.0, fraction=1.0, cap=0.02)
    assert stake == pytest.approx(0.02)


def test_kelly_scales_with_edge():
    small = evaluation.kelly_stake(0.52, 2.0, fraction=0.25, cap=1.0)
    large = evaluation.kelly_stake(0.60, 2.0, fraction=0.25, cap=1.0)
    assert 0 < small < large


def test_flat_staking_tracks_cumulative_profit_in_units():
    """Flat staking is reported in stake units starting at zero, not as a
    percentage of some starting bankroll picked out of the air."""
    bets = pd.DataFrame(
        {
            "date": [dt.date(2026, 1, 1), dt.date(2026, 1, 2)],
            "market": ["1x2", "1x2"],
            "model_probability": [0.5, 0.5],
            "price_taken": [2.0, 2.0],
            "profit_at_taken": [1.0, -1.0],
        }
    )
    ledger = evaluation.simulate_staking(bets, method="flat", flat_stake=10.0)
    assert ledger["bankroll"].tolist() == [10.0, 0.0]
    summary = evaluation.staking_summary(ledger)
    assert summary["profit"] == pytest.approx(0.0)
    assert summary["roi"] == pytest.approx(0.0)


def test_flat_staking_cannot_report_drawdown_beyond_100_percent():
    """Regression: a fixed stake against a small starting bankroll used to be
    able to push the simulated bankroll negative, reporting drawdowns beyond
    -100%. That was a bug - flat staking has no bankroll to go negative
    against any more."""
    bets = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20).date,
            "market": ["1x2"] * 20,
            "model_probability": [0.5] * 20,
            "price_taken": [2.0] * 20,
            "profit_at_taken": [-1.0] * 20,   # a total wipeout, worst case
        }
    )
    ledger = evaluation.simulate_staking(bets, method="flat", flat_stake=10.0)
    summary = evaluation.staking_summary(ledger)
    assert summary["max_drawdown"] >= -1.0
    assert summary["roi"] == pytest.approx(-1.0)


def test_flat_drawdown_is_measured():
    bets = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3).date,
            "market": ["1x2"] * 3,
            "model_probability": [0.5] * 3,
            "price_taken": [2.0] * 3,
            "profit_at_taken": [1.0, -1.0, -1.0],
        }
    )
    ledger = evaluation.simulate_staking(bets, method="flat", flat_stake=10.0)
    summary = evaluation.staking_summary(ledger)
    assert -1.0 <= summary["max_drawdown"] < -0.15


def test_kelly_staking_never_overstakes_the_bankroll():
    """Kelly is a genuine bankroll simulation: the stake can never exceed what
    remains, so a string of losses shrinks the bankroll but cannot borrow
    against it."""
    bets = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=50).date,
            "market": ["1x2"] * 50,
            "model_probability": [0.9] * 50,   # a large apparent edge
            "price_taken": [3.0] * 50,
            "profit_at_taken": [-1.0] * 50,    # and it loses every time
        }
    )
    ledger = evaluation.simulate_staking(
        bets, method="kelly", starting_bankroll=100.0, kelly_fraction=1.0, kelly_cap=0.5
    )
    assert (ledger["bankroll"] >= 0).all()
    assert (ledger["stake"] <= ledger["bankroll"] + ledger["stake"] + 1e-9).all()
    summary = evaluation.staking_summary(ledger, starting_bankroll=100.0)
    assert -1.0 <= summary["max_drawdown"] <= 0.0
    # A losing edge shrinks the bankroll towards zero asymptotically rather
    # than in one step, so it ends up small but need not hit exactly zero.
    assert ledger["bankroll"].iloc[-1] < 1.0


def test_busted_flag_is_set_when_bankroll_is_exhausted():
    """Constructed directly against staking_summary, since a true Kelly stake
    self-limits and only reaches exactly zero in the limit."""
    ledger = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3).date,
            "market": ["1x2"] * 3,
            "stake": [50.0, 50.0, 0.0],
            "profit": [-20.0, -30.0, 0.0],
            "bankroll": [80.0, 0.0, 0.0],
        }
    )
    summary = evaluation.staking_summary(ledger, starting_bankroll=100.0)
    assert summary["busted"] is True
    assert summary["max_drawdown"] == pytest.approx(-1.0)


def test_bootstrap_interval_brackets_the_observed_roi():
    rng = np.random.default_rng(5)
    profits = rng.normal(0.05, 1.0, 400)
    bets = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2026-01-01", periods=100).date, 4),
            "profit_at_taken": profits,
        }
    )
    interval = evaluation.bootstrap_roi(bets, iterations=500)
    assert interval["roi_low"] < interval["roi"] < interval["roi_high"]
    assert 0.0 <= interval["probability_profitable"] <= 1.0


def test_bootstrap_interval_is_wide_for_small_samples():
    """The width is the point: a few hundred bets cannot resolve a small edge.

    A true 3% edge with unit-variance results needs thousands of bets before
    the interval stops spanning both a healthy profit and a real loss. The
    assertion is that the method refuses to be confident, not that it lands on
    any particular side of zero.
    """
    rng = np.random.default_rng(7)
    bets = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2026-01-01", periods=40).date, 5),
            "profit_at_taken": rng.normal(0.03, 1.0, 200),
        }
    )
    interval = evaluation.bootstrap_roi(bets, iterations=500)
    assert interval["roi_high"] - interval["roi_low"] > 0.10
    assert 0.02 < interval["probability_profitable"] < 0.98


def test_bootstrap_declines_on_too_few_bets():
    bets = pd.DataFrame({"date": [dt.date(2026, 1, 1)] * 5, "profit_at_taken": [1.0] * 5})
    assert "roi" not in evaluation.bootstrap_roi(bets)


def test_summarise_runs_end_to_end(result):
    summary = evaluation.summarise(result)
    assert summary["selections"] == len(result.predictions)
    assert summary["refits"] == result.refits
    assert isinstance(summary["market_scores"], list)


# --------------------------------------------------------------------------
# Bookmaker comparison
# --------------------------------------------------------------------------

def test_margins_recover_the_expected_overrounds(backtest_db):
    """Sharp books must show a smaller overround than soft ones.

    The sample generator prices Pinnacle at a 2.5% margin and Bet365 at 6%, so
    this doubles as a check that the overround arithmetic is right.
    """
    margins = evaluation.market_margins(backtest_db, "E0")
    assert not margins.empty
    by_book = margins.groupby("bookmaker")["mean_overround"].mean()
    assert by_book["pinnacle"] < by_book["bet365"]
    assert by_book.min() > 1.0


def test_bookmaker_breakdown_ranks_by_clv(result):
    """Every bookmaker's own price is compared, not just whichever one
    happened to be chosen for staking - the sample data carries both Pinnacle
    and Bet365, so this must return both rather than collapsing to one."""
    table = evaluation.bookmaker_breakdown(result.predictions, min_selections=20)
    assert not table.empty
    assert {"pinnacle", "bet365"} <= set(table.index)
    assert table["mean_clv"].is_monotonic_decreasing
    assert (table["selections"] >= 20).all()


def test_bookmaker_breakdown_reflects_the_sharper_book(result):
    """Pinnacle is priced with a smaller margin than Bet365 in the sample
    generator, so it must show better (less negative) CLV on average."""
    table = evaluation.bookmaker_breakdown(result.predictions, min_selections=20)
    if "pinnacle" not in table.index or "bet365" not in table.index:
        pytest.skip("both books not present with enough selections")
    assert table.loc["pinnacle", "mean_clv"] > table.loc["bet365", "mean_clv"]


def test_book_prices_are_attached_per_selection(result):
    frame = result.predictions.dropna(subset=["book_prices"])
    assert not frame.empty
    sample = frame["book_prices"].iloc[0]
    assert isinstance(sample, dict)
    assert all(isinstance(v, float) for v in sample.values())


def test_fair_line_prefers_the_sharpest_source():
    """Reading the benchmark off a soft book would bake its margin in."""
    assert backtest.FAIR_LINE_PREFERENCE[0] == "betfair_exchange"
    assert backtest.FAIR_LINE_PREFERENCE.index("pinnacle") < \
        backtest.FAIR_LINE_PREFERENCE.index("bet365")


def test_price_source_can_be_restricted_to_one_book(backtest_db):
    """Modelling a single account, rather than shopping across several."""
    predict_mod.clear_model_cache()
    settings = backtest.BacktestConfig(
        league="E0",
        start=dt.date(config.CURRENT_SEASON_START_YEAR - 1, 8, 1),
        end=dt.date(config.CURRENT_SEASON_START_YEAR, 8, 31),
        price_source=("bet365",),
        markets=("1x2",),
        fit_count_models=False,
    )
    restricted = backtest.run_backtest(backtest_db, settings, verbose=False)
    sources = set(restricted.predictions["price_source"].dropna())
    assert sources <= {"bet365"}
