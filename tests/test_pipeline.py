"""Tests for the Phase 1 data spine.

The two that matter most are `test_no_lookahead_in_profile` and
`test_profile_excludes_matches_on_the_as_of_date`. Everything else here
protects against inconvenience; those two protect against building a backtest
that reports a fictional edge because the profile could see the result of the
match it was profiling.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, normalize, quality  # noqa: E402
from fbedge import profile as profile_mod  # noqa: E402
from scripts.make_sample_data import build_season  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_season() -> pd.DataFrame:
    return build_season("E0", 2024, seed=7)


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    """A database built from several synthetic seasons across two leagues."""
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    con = database.connect(path)
    for year in (2019, 2023, 2024, 2025, config.CURRENT_SEASON_START_YEAR):
        for league in ("E0", "F1"):
            raw = build_season(league, year, seed=year * 10)
            matches, odds = normalize.normalize_league_season(raw, league, year)
            database.load_matches(con, matches)
            database.load_odds(con, odds)
    yield con
    con.close()


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------

def test_season_code_and_label():
    assert config.season_code(2026) == "2627"
    assert config.season_code(1999) == "9900"
    assert config.season_label(2026) == "2026/27"
    assert config.season_years(3) == [2024, 2025, 2026]


def test_season_csv_url_rejects_unknown_league():
    assert config.season_csv_url("E0", 2026).endswith("/2627/E0.csv")
    with pytest.raises(KeyError):
        config.season_csv_url("XX9", 2026)


# --------------------------------------------------------------------------
# Team names
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Manchester United", "Man United"),
        ("Man Utd", "Man United"),
        ("Man United", "Man United"),
        ("Borussia Mönchengladbach", "M'gladbach"),
        ("PSG", "Paris SG"),
        ("Atletico Madrid", "Ath Madrid"),
        ("  Arsenal  ", "Arsenal"),
        ("Espanyol", "Espanol"),
    ],
)
def test_canonical_team(raw, expected):
    assert normalize.canonical_team(raw) == expected


def test_unknown_team_passes_through():
    """A newly promoted club must not break ingestion."""
    assert normalize.canonical_team("Brand New FC") == "Brand New FC"


def test_team_slug():
    assert normalize.team_slug("Nott'm Forest") == "nottm_forest"
    assert normalize.team_slug("M'gladbach") == "mgladbach"
    assert normalize.team_slug("Ath Madrid") == "ath_madrid"


def test_resolve_team():
    known = ["Ath Madrid", "Ath Bilbao", "Arsenal", "Aston Villa"]
    assert normalize.resolve_team("arsenal", known) == "Arsenal"
    assert normalize.resolve_team("Atletico Madrid", known) == "Ath Madrid"
    assert normalize.resolve_team("atletico", known) == "Ath Madrid"
    assert normalize.resolve_team("A", known) is None  # ambiguous, not a guess
    assert normalize.resolve_team("", known) is None


def test_missing_values_do_not_become_strings():
    """Regression: pd.NA was being rendered as the literal string '<NA>'."""
    assert normalize.canonical_referee(pd.NA) is None
    assert normalize.canonical_referee(None) is None
    assert normalize.canonical_referee(float("nan")) is None
    assert normalize.canonical_referee("   ") is None
    assert normalize.canonical_team(pd.NA) == ""
    assert normalize.canonical_referee("M. Oliver") == "M Oliver"


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

def test_parse_dates_handles_both_formats():
    parsed = normalize.parse_dates(pd.Series(["15/08/2026", "15/08/26", "01/12/1999"]))
    assert parsed.iloc[0] == pd.Timestamp("2026-08-15")
    assert parsed.iloc[1] == pd.Timestamp("2026-08-15")
    assert parsed.iloc[2] == pd.Timestamp("1999-12-01")


def test_parse_dates_is_day_first():
    """05/06/2026 must be 5 June, not 6 May."""
    assert normalize.parse_dates(pd.Series(["05/06/2026"])).iloc[0] == pd.Timestamp("2026-06-05")


def test_bad_dates_become_nat_not_exceptions():
    parsed = normalize.parse_dates(pd.Series(["not a date", None, "15/08/2026"]))
    assert parsed.isna().sum() == 2
    assert parsed.notna().sum() == 1


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_derived_metrics(raw_season):
    matches, _ = normalize.normalize_league_season(raw_season, "E0", 2024)
    row = matches.iloc[0]
    assert row["total_goals"] == row["home_goals"] + row["away_goals"]
    assert row["btts"] == (row["home_goals"] > 0 and row["away_goals"] > 0)
    assert row["total_cards"] == row["total_yellows"] + row["total_reds"]
    assert row["booking_points"] == row["total_yellows"] * 10 + row["total_reds"] * 25
    assert row["goals_2h"] == row["total_goals"] - row["goals_ht"]
    assert row["over_2_5"] == (row["total_goals"] > 2.5)


def test_result_matches_score(raw_season):
    matches, _ = normalize.normalize_league_season(raw_season, "E0", 2024)
    expected = matches.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else ("A" if r["home_goals"] < r["away_goals"] else "D"),
        axis=1,
    )
    assert (matches["result"] == expected).all()


def test_padding_rows_are_dropped(raw_season):
    """The source pads files with blank rows; none should survive."""
    matches, _ = normalize.normalize_league_season(raw_season, "E0", 2024)
    assert matches["date"].notna().all()
    assert (matches["home_team"] != "").all()
    assert not matches["home_team"].isin(["<NA>", "nan", "None"]).any()


def test_unplayed_fixtures_are_excluded():
    """The in-progress season's file lists future fixtures with no score."""
    raw = build_season("E0", config.CURRENT_SEASON_START_YEAR, seed=3)
    matches, _ = normalize.normalize_league_season(raw, "E0", config.CURRENT_SEASON_START_YEAR)
    assert len(matches) < len(raw)
    assert matches["home_goals"].notna().all()
    assert (matches["date"] < pd.Timestamp.today()).all()


def test_missing_columns_become_null_not_missing_keys():
    """F1 pre-2021 in the sample has no corner or referee columns."""
    raw = build_season("F1", 2019, seed=11)
    assert "HC" not in raw.columns
    matches, _ = normalize.normalize_league_season(raw, "F1", 2019)
    assert "home_corners" in matches.columns
    assert matches["home_corners"].isna().all()
    assert matches["referee"].isna().all()
    assert matches["home_goals"].notna().all()  # the rest still loads


def test_match_ids_are_unique_and_deterministic(raw_season):
    first, _ = normalize.normalize_league_season(raw_season, "E0", 2024)
    second, _ = normalize.normalize_league_season(raw_season, "E0", 2024)
    assert first["match_id"].is_unique
    assert first["match_id"].tolist() == second["match_id"].tolist()


# --------------------------------------------------------------------------
# Odds
# --------------------------------------------------------------------------

def test_odds_align_with_surviving_matches():
    """Odds must never refer to a match that was filtered out."""
    raw = build_season("E0", config.CURRENT_SEASON_START_YEAR, seed=5)
    matches, odds = normalize.normalize_league_season(
        raw, "E0", config.CURRENT_SEASON_START_YEAR
    )
    assert set(odds["match_id"]) <= set(matches["match_id"])
    assert not odds.empty


def test_odds_are_plausible(raw_season):
    _, odds = normalize.normalize_league_season(raw_season, "E0", 2024)
    assert (odds["price"] > 1.0).all()
    assert odds["market"].isin({"1x2", "total_goals", "asian_handicap"}).all()
    assert odds["phase"].isin({"open", "close"}).all()
    assert odds["selection"].isin({"home", "draw", "away", "over", "under"}).all()


def test_old_pinnacle_column_names_are_read():
    """Pre-2019 files call Pinnacle PH/PD/PA rather than PSH/PSD/PSA."""
    raw = build_season("E0", 2018, seed=13)
    assert "PH" in raw.columns and "PSH" not in raw.columns
    _, odds = normalize.normalize_league_season(raw, "E0", 2018)
    assert not odds[odds["bookmaker"] == "pinnacle"].empty


def test_overround_is_positive(raw_season):
    """Implied probabilities must sum above 1: no free money in the data."""
    _, odds = normalize.normalize_league_season(raw_season, "E0", 2024)
    ones = odds[(odds["market"] == "1x2") & (odds["bookmaker"] == "bet365")]
    sums = ones.groupby("match_id")["price"].apply(lambda s: (1 / s).sum())
    assert (sums > 1.0).all()


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def test_loading_is_idempotent(tmp_path, raw_season):
    con = database.connect(tmp_path / "idem.duckdb")
    matches, odds = normalize.normalize_league_season(raw_season, "E0", 2024)
    for _ in range(3):
        database.load_matches(con, matches)
        database.load_odds(con, odds)
    assert con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == len(matches)
    assert con.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == len(odds)
    con.close()


def test_team_matches_view_doubles_rows(populated_db):
    matches = populated_db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    team_rows = populated_db.execute("SELECT COUNT(*) FROM team_matches").fetchone()[0]
    assert team_rows == matches * 2


def test_team_matches_view_flips_perspective(populated_db):
    """The away row's goals_for must equal the match's away_goals."""
    row = populated_db.execute(
        """
        SELECT m.home_goals, m.away_goals, t.goals_for, t.goals_against, t.outcome
        FROM matches m JOIN team_matches t USING (match_id)
        WHERE t.is_home = FALSE AND m.home_goals > m.away_goals
        LIMIT 1
        """
    ).fetchone()
    home_goals, away_goals, goals_for, goals_against, outcome = row
    assert goals_for == away_goals
    assert goals_against == home_goals
    assert outcome == "L"


def test_integrity_checks_pass(populated_db):
    report = quality.integrity_checks(populated_db)
    failed = report[~report["passed"]]
    assert failed.empty, f"failed checks:\n{failed}"


# --------------------------------------------------------------------------
# Point-in-time correctness - the important ones
# --------------------------------------------------------------------------

def test_profile_excludes_matches_on_the_as_of_date(populated_db):
    """`as_of` is exclusive: a match played that day is not yet knowable."""
    match = populated_db.execute(
        """
        SELECT date, home_team, away_team FROM matches
        WHERE league = 'E0' ORDER BY date DESC LIMIT 1
        """
    ).fetchone()
    match_date, home, away = match

    before = profile_mod.fixture_profile(
        populated_db, home, away, as_of=match_date, league="E0"
    )
    after = profile_mod.fixture_profile(
        populated_db, home, away,
        as_of=match_date + dt.timedelta(days=1), league="E0",
    )
    assert after.home.stat("season", "goals_for").n == \
        before.home.stat("season", "goals_for").n + 1


def test_no_lookahead_in_profile(populated_db):
    """Nothing in a profile may come from after `as_of`."""
    as_of = dt.date(2025, 9, 15)
    rows = profile_mod._team_rows(populated_db, "Arsenal", as_of, league="E0")
    assert (pd.to_datetime(rows["date"]).dt.date < as_of).all()

    prof = profile_mod.fixture_profile(
        populated_db, "Arsenal", "Liverpool", as_of=as_of, league="E0"
    )
    for line in prof.home.form:
        assert pd.Timestamp(line["date"]).date() < as_of
    for meeting in prof.h2h["matches"]:
        assert pd.Timestamp(meeting["date"]).date() < as_of


def test_profile_scopes_are_nested_correctly(populated_db):
    """Venue-restricted samples can never be larger than the full season."""
    prof = profile_mod.fixture_profile(
        populated_db, "Arsenal", "Liverpool",
        as_of=dt.date(2026, 8, 31), league="E0",
    )
    season_n = prof.home.stat("season", "goals_for").n
    venue_n = prof.home.stat("season_venue", "goals_for").n
    assert venue_n <= season_n
    assert prof.home.stat("last6", "goals_for").n <= 6


def test_sample_size_is_per_metric(populated_db):
    """A league with goals but no corners must report n=0 for corners only."""
    prof = profile_mod.fixture_profile(
        populated_db, "Paris SG", "Marseille",
        as_of=dt.date(2020, 6, 1), league="F1", season_start_year=2019,
    )
    assert prof.home.stat("season", "goals_for").n > 0
    assert prof.home.stat("season", "total_corners").n == 0
    assert prof.home.stat("season", "total_corners").value is None


def test_thin_samples_produce_warnings(populated_db):
    """Two matches into a season, the profile must say so."""
    prof = profile_mod.fixture_profile(
        populated_db, "Arsenal", "Liverpool",
        as_of=dt.date(config.CURRENT_SEASON_START_YEAR, 8, 20), league="E0",
    )
    assert prof.warnings
    assert any("noise" in w or "no matches" in w for w in prof.warnings)


def test_profile_renders_without_data(populated_db):
    """An unknown team must produce an empty card, not a traceback."""
    prof = profile_mod.fixture_profile(
        populated_db, "Nonexistent FC", "Arsenal",
        as_of=dt.date(2026, 8, 31), league="E0",
    )
    assert prof.home.stat("season", "goals_for").n == 0
    assert "no matches yet" in " ".join(prof.warnings)
    assert isinstance(profile_mod.format_profile(prof), str)


def test_stat_formatting():
    assert profile_mod.Stat(2.456, 10).format() == "2.46"
    assert profile_mod.Stat(0.5, 10).format(as_percent=True) == "50%"
    assert profile_mod.Stat(None, 0).format() == "-"
    assert profile_mod.Stat(1.0, 9).is_reliable is False
    assert profile_mod.Stat(1.0, 10).is_reliable is True


# --------------------------------------------------------------------------
# Odds column specifications, verified against football-data.co.uk/notes.txt
# --------------------------------------------------------------------------

def test_betfair_exchange_is_extracted():
    """An exchange charges commission rather than a margin, so its close is
    the sharpest benchmark in the file. Missing it would cost the backtest its
    best reference."""
    books = {name for name, *_ in normalize.ODDS_1X2}
    assert "betfair_exchange" in books
    columns = {spec[2] for spec in normalize.ODDS_1X2}
    assert {"BFEH", "BFECH"} <= columns


def test_sharp_and_soft_books_are_both_covered():
    books = {name for name, *_ in normalize.ODDS_1X2}
    assert {"pinnacle", "market_max", "market_avg"} <= books          # references
    assert {"bet365", "william_hill", "ladbrokes", "betvictor"} <= books  # bettable


def test_closing_columns_follow_the_documented_convention():
    """The source inserts a C after the bookmaker prefix: B365H -> B365CH."""
    specs = {(name, phase): home for name, phase, home, _, _ in normalize.ODDS_1X2}
    assert specs[("bet365", "open")] == "B365H"
    assert specs[("bet365", "close")] == "B365CH"


def test_bookmaker_specific_handicap_lines_take_precedence():
    """A book's prices refer to its own line, not the market-wide one."""
    bet365 = [s for s in normalize.ODDS_ASIAN_HANDICAP
              if s[0] == "bet365" and s[1] == "open"]
    assert bet365[0][2] == "B365AH"      # own line first
    assert "AHh" in [spec[2] for spec in bet365]   # market line as fallback


def test_free_kicks_conceded_substitutes_for_fouls():
    """Some competitions report free kicks conceded instead of fouls."""
    assert normalize.COLUMN_FALLBACKS["HFKC"] == "home_fouls"
    raw = pd.DataFrame(
        {
            "Date": ["15/08/2026"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Everton"],
            "FTHG": [2], "FTAG": [0], "FTR": ["H"], "HFKC": [14], "AFKC": [11],
        }
    )
    matches, _ = normalize.normalize_league_season(raw, "E0", 2026)
    assert matches["home_fouls"].iloc[0] == 14
    assert matches["total_fouls"].iloc[0] == 25
