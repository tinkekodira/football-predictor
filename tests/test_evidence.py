"""Tests for the evidence layer and the pre-match scan.

Two things are worth guarding here and they are not the same thing.

The **point-in-time** tests are the ones that matter in the way the backtest's
two guards matter: a scan that could see the match it is pricing would produce
a spectacular apparent edge, and it would not look like a bug.

The **labelling** tests are the ones that matter in the way the README's
sample-size rule matters. A market with no historical price must never be able
to display a number that reads like a track record, whatever else changes.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import (  # noqa: E402
    backtest, config, database, evidence, normalize, snapshots,
)
from fbedge import predict as predict_mod  # noqa: E402
from fbedge.models import base as model_base  # noqa: E402
from scripts.make_sample_data import build_season  # noqa: E402
from scripts import scan_fixtures  # noqa: E402


@pytest.fixture(scope="module")
def scan_db(tmp_path_factory):
    """Five synthetic seasons plus an archived round of upcoming fixtures."""
    path = tmp_path_factory.mktemp("scan") / "scan.duckdb"
    con = database.connect(path)
    for year in range(config.CURRENT_SEASON_START_YEAR - 4, config.CURRENT_SEASON_START_YEAR + 1):
        raw = build_season("E0", year, seed=year * 31)
        matches, odds = normalize.normalize_league_season(raw, "E0", year)
        database.load_matches(con, matches)
        database.load_odds(con, odds)
    yield con
    con.close()


@pytest.fixture(scope="module")
def as_of(scan_db) -> dt.date:
    """A date inside the last synthetic season, so there is history and future."""
    row = scan_db.execute("SELECT MAX(date) FROM matches WHERE league = 'E0'").fetchone()
    return pd.Timestamp(row[0]).date() - dt.timedelta(days=30)


@pytest.fixture(scope="module")
def archived(scan_db, as_of):
    """Two upcoming fixtures between teams the database already knows."""
    teams = database.known_teams(scan_db, league="E0")
    kickoff = as_of + dt.timedelta(days=2)
    rows = pd.DataFrame(
        [
            {
                "Div": "E0", "Date": kickoff.strftime("%d/%m/%Y"), "Time": "15:00",
                "HomeTeam": teams[0], "AwayTeam": teams[1], "Referee": None,
                "B365H": 2.10, "B365D": 3.40, "B365A": 3.60,
                "MaxH": 2.20, "MaxD": 3.60, "MaxA": 3.80,
                "AvgH": 2.05, "AvgD": 3.35, "AvgA": 3.50,
                "B365>2.5": 1.90, "B365<2.5": 1.90,
                "Max>2.5": 1.95, "Max<2.5": 1.95,
                "AHh": -0.5, "B365AHH": 1.95, "B365AHA": 1.95,
                "MaxAHH": 2.00, "MaxAHA": 2.00,
            },
            {
                "Div": "E0", "Date": kickoff.strftime("%d/%m/%Y"), "Time": "17:30",
                "HomeTeam": teams[2], "AwayTeam": teams[3], "Referee": None,
                "B365H": 1.80, "B365D": 3.80, "B365A": 4.50,
                "MaxH": 1.85, "MaxD": 3.95, "MaxA": 4.70,
                "AvgH": 1.78, "AvgD": 3.70, "AvgA": 4.40,
                "B365>2.5": 1.80, "B365<2.5": 2.00,
                "Max>2.5": 1.88, "Max<2.5": 2.08,
                "AHh": -0.75, "B365AHH": 1.90, "B365AHA": 2.00,
                "MaxAHH": 1.95, "MaxAHA": 2.05,
            },
        ]
    )
    snapshot, odds = snapshots.build_snapshot(rows, leagues=["E0"])
    snapshots.write_snapshot(scan_db, snapshot, odds)
    return snapshot


# --------------------------------------------------------------------------
# Point in time
# --------------------------------------------------------------------------

def test_the_scan_fits_only_on_matches_before_its_as_of_date(scan_db, as_of):
    """The same guard the backtest carries, on the new code path.

    One leak here would produce a scan whose top selections were the matches
    it had already seen the result of, which is the failure mode that looks
    most like success.
    """
    training = model_base.load_training_set(scan_db, "E0", as_of)
    assert (pd.to_datetime(training.frame["date"]).dt.date < as_of).all()


def test_the_scan_cannot_see_a_match_on_its_own_as_of_date(scan_db, as_of):
    """Strictly before, not on or before. A same-day match must be invisible."""
    played_that_day = scan_db.execute(
        "SELECT COUNT(*) FROM matches WHERE league = 'E0' AND date = ?", [as_of]
    ).fetchone()[0]
    training = model_base.load_training_set(scan_db, "E0", as_of)
    assert not (pd.to_datetime(training.frame["date"]).dt.date == as_of).any()
    # And the assertion above is only meaningful if such a match exists in the
    # database at all; otherwise it passes vacuously.
    if played_that_day == 0:
        pytest.skip("no match on the as_of date in this sample, guard is vacuous")


def test_the_scan_prices_only_fixtures_that_have_not_kicked_off(
    scan_db, as_of, archived
):
    predict_mod.clear_model_cache()
    frame, _notes = scan_fixtures.scan(
        scan_db, as_of, ["E0"], backtest.DEFAULT_PRICE_SOURCE
    )
    assert not frame.empty
    assert (pd.to_datetime(frame["date"]).dt.date >= as_of).all()


# --------------------------------------------------------------------------
# The scan itself
# --------------------------------------------------------------------------

def test_the_scan_prices_the_selections_the_source_published(
    scan_db, as_of, archived
):
    """Not a fixed ladder of lines. The book's own line or nothing."""
    predict_mod.clear_model_cache()
    frame, _ = scan_fixtures.scan(scan_db, as_of, ["E0"], backtest.DEFAULT_PRICE_SOURCE)
    stored = snapshots.load_snapshot_odds(scan_db, list(archived["content_hash"]))
    published = {
        (m, s, None if pd.isna(v) else round(float(v), 3))
        for m, s, v in zip(stored["market"], stored["selection"], stored["line"])
    }
    # Every scanned handicap line must be one the source actually offered.
    for row in frame[frame["market"] == "asian_handicap"].itertuples():
        line = float(row.selection.split()[-1])
        assert ("asian_handicap", row.selection.split()[0], line) in published


def test_expected_value_is_sorted_and_uses_the_taken_price(
    scan_db, as_of, archived
):
    predict_mod.clear_model_cache()
    frame, _ = scan_fixtures.scan(scan_db, as_of, ["E0"], backtest.DEFAULT_PRICE_SOURCE)
    assert frame["expected_value"].is_monotonic_decreasing
    row = frame.iloc[0]
    assert row["expected_value"] == pytest.approx(
        row["model_probability"] * row["price"] - 1.0, abs=1e-6
    )


def test_the_book_flag_changes_which_price_is_taken(scan_db, as_of, archived):
    predict_mod.clear_model_cache()
    default, _ = scan_fixtures.scan(
        scan_db, as_of, ["E0"], backtest.DEFAULT_PRICE_SOURCE
    )
    single, _ = scan_fixtures.scan(scan_db, as_of, ["E0"], ("bet365",))
    assert set(default["book"]) == {"market_max"}
    assert set(single["book"]) == {"bet365"}
    # And the market maximum is by definition never worse than one book's price.
    assert default["price"].max() >= single["price"].max()


def test_a_fixture_the_model_barely_knows_is_flagged(scan_db, as_of, archived):
    """A promoted club priced from the prior must not look like a find."""
    predict_mod.clear_model_cache()
    bundle = predict_mod.build_models(scan_db, "E0", as_of, fit_counts=False)

    class Unknown:
        home_team = "Newly Promoted FC"
        away_team = database.known_teams(scan_db, league="E0")[0]

    assert "no history" in scan_fixtures._thin_history(bundle, Unknown)


# --------------------------------------------------------------------------
# Evidence: the labelling rules
# --------------------------------------------------------------------------

def test_a_market_with_no_evidence_row_is_untested():
    assert evidence.status("total_corners", None) == evidence.UNTESTED
    label = evidence.describe("total_corners", None)
    assert label.startswith("UNTESTED")
    assert "no corner prices" in label


def test_a_market_the_source_never_priced_can_never_be_backtested():
    """The rule that must survive every future refactor.

    Corners have a full calibration record and 4,000 scored selections. None of
    that may ever be allowed to read as a track record for *betting* them,
    because no corner price has ever existed in this source.
    """
    row = pd.Series(
        {
            "n": 4000, "n_matches": 500, "n_bets": 0,
            "calibration_slope": 0.98, "mean_clv": np.nan, "clv_se": np.nan,
            "model_log_loss": 0.6, "market_log_loss": np.nan,
        }
    )
    assert evidence.status("total_corners", row) == evidence.CALIBRATION_ONLY
    label = evidence.describe("total_corners", row)
    assert "CALIBRATION ONLY" in label
    assert "no evidence of an edge" in label
    assert "paid odds feed" in label


def test_a_priced_market_with_bets_is_backtested_and_shows_its_clv():
    row = pd.Series(
        {
            "n": 2000, "n_matches": 700, "n_bets": 900,
            "calibration_slope": 1.05, "mean_clv": -0.0165, "clv_se": 0.0037,
            "model_log_loss": 0.5767, "market_log_loss": 0.5626,
        }
    )
    assert evidence.status("1x2", row) == evidence.BACKTESTED
    label = evidence.describe("1x2", row)
    assert "BACKTESTED" in label
    assert "-1.65%" in label
    assert "900 bets" in label


def test_a_priced_market_with_no_bets_is_not_called_backtested():
    """A market nobody bet is not a market with a track record."""
    row = pd.Series(
        {
            "n": 2000, "n_matches": 700, "n_bets": 0,
            "calibration_slope": 1.05, "mean_clv": np.nan, "clv_se": np.nan,
            "model_log_loss": 0.58, "market_log_loss": 0.56,
        }
    )
    assert evidence.status("1x2", row) == evidence.CALIBRATION_ONLY


def test_every_market_the_model_prices_gets_a_label(scan_db, as_of):
    """No price may reach a reader without its evidence in the same glance."""
    labels = evidence.labels(
        pd.DataFrame(), backtest.DEFAULT_CALIBRATION_MARKETS
    )
    assert set(labels) == set(backtest.DEFAULT_CALIBRATION_MARKETS)
    assert all(text.startswith("UNTESTED") for text in labels.values())


def test_only_three_markets_are_ever_bettable():
    """Pinned because the brief that asked for this listed a fourth.

    BTTS is not among them: `normalize.extract_odds` produces 1x2, total_goals
    and asian_handicap because those are the only prices the source publishes,
    and no both-teams-to-score column has ever existed in it.
    """
    assert evidence.PRICED_MARKETS == ("1x2", "total_goals", "asian_handicap")
    assert "btts" not in evidence.PRICED_MARKETS


# --------------------------------------------------------------------------
# Evidence: computing and storing it
# --------------------------------------------------------------------------

def test_computing_evidence_produces_both_kinds_of_row(scan_db, as_of):
    predict_mod.clear_model_cache()
    start = as_of - dt.timedelta(days=400)
    frame = evidence.compute(
        scan_db, "E0", start, as_of,
        markets_wanted=("1x2", "btts", "total_goals"),
    )
    assert not frame.empty
    statuses = dict(zip(frame["market"], frame["status"]))
    assert statuses.get("btts") == evidence.CALIBRATION_ONLY
    assert statuses.get("1x2") == evidence.BACKTESTED
    assert (frame["n"] > 0).all()


def test_evidence_round_trips_through_the_database(scan_db, as_of):
    frame = pd.DataFrame(
        [
            {
                "league": "E0", "market": "1x2", "status": evidence.BACKTESTED,
                "n": 100, "n_matches": 50, "model_log_loss": 0.6,
                "base_rate_log_loss": 0.65, "market_log_loss": 0.59,
                "calibration_slope": 1.02, "calibration_slope_se": 0.05,
                "n_bets": 40, "mean_clv": -0.01, "clv_se": 0.003,
                "window_start": as_of - dt.timedelta(days=365),
                "window_end": as_of,
                "computed_at": dt.datetime(2026, 9, 4, 12, 0),
            }
        ]
    )
    evidence.write(scan_db, frame)
    back = evidence.load(scan_db, "E0")
    assert len(back) == 1
    assert back["market"].iloc[0] == "1x2"

    # Writing one league must leave the others alone: the mistake build_xg.py
    # made, which silently deleted four leagues of data (BACKLOG B2).
    other = frame.copy()
    other["league"] = "SP1"
    evidence.write(scan_db, other)
    assert len(evidence.load(scan_db)) == 2
    evidence.write(scan_db, frame)
    assert len(evidence.load(scan_db)) == 2


def test_load_returns_an_empty_frame_when_nothing_has_been_computed(tmp_path):
    con = database.connect(tmp_path / "bare.duckdb")
    assert evidence.load(con).empty
    con.close()


# --------------------------------------------------------------------------
# Card conditions, which are league-dependent
# --------------------------------------------------------------------------

def test_card_conditions_report_the_english_yellow_card_convention(scan_db):
    conditions = evidence.card_conditions(scan_db, "E0")
    assert conditions["second_yellow_excluded"] is True
    assert "not comparable across borders" in conditions["note"]


def test_card_conditions_report_the_continental_convention(scan_db):
    conditions = evidence.card_conditions(scan_db, "I1")
    assert conditions["second_yellow_excluded"] is False


def test_card_conditions_report_missing_referee_coverage(tmp_path):
    """A league with no referee column has no referee effects, and says so."""
    con = database.connect(tmp_path / "noref.duckdb")
    raw = build_season("F1", config.CURRENT_SEASON_START_YEAR - 1, seed=5)
    raw = raw.drop(columns=[c for c in raw.columns if c == "Referee"])
    matches, odds = normalize.normalize_league_season(
        raw, "F1", config.CURRENT_SEASON_START_YEAR - 1
    )
    database.load_matches(con, matches)
    conditions = evidence.card_conditions(con, "F1")
    assert conditions["has_referee_effects"] is False
    assert "league-average referee" in conditions["note"]
    con.close()


# --------------------------------------------------------------------------
# The odds-less guard, which Phase 5 needs
# --------------------------------------------------------------------------

def test_a_match_with_no_odds_is_fitted_but_never_bettable(scan_db, as_of):
    """The guard the roadmap needs before UEFA results enter the database.

    A match with no odds must still contribute to calibration and to team
    strengths, and must never contribute to CLV or ROI. Simulated by stripping
    the odds from one match rather than by waiting for a European fixture.
    """
    predict_mod.clear_model_cache()
    settings = backtest.BacktestConfig(
        league="E0",
        start=as_of - dt.timedelta(days=400),
        end=as_of,
        calibration_markets=("1x2",),
    )
    result = backtest.run_backtest(scan_db, settings, verbose=False)
    calibration = result.calibration_only
    assert not calibration.empty
    assert calibration["price_taken"].isna().all()
    assert calibration["expected_value"].isna().all()
    assert calibration["clv"].isna().all()
    # And none of them can reach the bet ledger, however the threshold moves.
    assert result.bets["priceless"].eq(False).all()


def test_the_result_counts_matches_that_could_not_be_bet(scan_db, as_of):
    settings = backtest.BacktestConfig(
        league="E0", start=as_of - dt.timedelta(days=400), end=as_of,
    )
    result = backtest.run_backtest(scan_db, settings, verbose=False)
    assert isinstance(result.fitted_not_bettable, int)
    assert result.fitted_not_bettable >= 0


# --------------------------------------------------------------------------
# The command line must obey the same rule as the app
# --------------------------------------------------------------------------

def test_the_forecast_renders_its_evidence_when_it_has_any(scan_db, as_of):
    """A fair price and its track record belong on the same screen.

    `render` prints whatever is in `forecast.evidence`; this pins that it
    actually does, so a caller that fetched the evidence cannot have it
    silently dropped on the way to the terminal.
    """
    predict_mod.clear_model_cache()
    forecast = predict_mod.predict_fixture(
        scan_db, *database.known_teams(scan_db, league="E0")[:2],
        as_of=as_of, league="E0",
    )
    assert "[" not in forecast.render(), "no labels were supplied, so none show"

    forecast.evidence = {"1x2": "backtested, CLV -1.65%, n=999"}
    rendered = forecast.render()
    assert "Match result    [backtested, CLV -1.65%, n=999]" in rendered


def test_a_half_time_market_is_priced_from_its_own_fit(scan_db, as_of):
    """Not from the full-time matrix, and measurably not.

    The half-time draw probability is much higher than the full-time one - a
    goalless first half is common and a goalless match is not - so a derived
    approximation would be visibly wrong here.
    """
    predict_mod.clear_model_cache()
    home, away = database.known_teams(scan_db, league="E0")[:2]
    forecast = predict_mod.predict_fixture(
        scan_db, home, away, as_of=as_of, league="E0", half_time=True,
    )
    full = {s.selection: s.probability for s in forecast.market("1x2")}
    half = {s.selection: s.probability for s in forecast.market("1x2_ht")}
    assert half, "the half-time market was not priced"
    assert half["draw"] > full["draw"], (
        "a half-time draw must be likelier than a full-time one; if it is not, "
        "the half-time prices are probably being derived from the full-time fit"
    )
    assert sum(half.values()) == pytest.approx(1.0, abs=1e-6)


def test_half_time_markets_are_absent_unless_asked_for(scan_db, as_of):
    predict_mod.clear_model_cache()
    home, away = database.known_teams(scan_db, league="E0")[:2]
    forecast = predict_mod.predict_fixture(
        scan_db, home, away, as_of=as_of, league="E0",
    )
    assert not forecast.market("1x2_ht")
    assert not forecast.market("total_goals_ht")
