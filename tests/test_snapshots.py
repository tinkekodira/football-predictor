"""Tests for the upcoming-fixtures archive.

The dedupe tests carry the most weight. This archive cannot be rebuilt from
anywhere - the source overwrites the file it comes from - so a bug that stores
a duplicate merely wastes space, while a bug that *fails* to store a genuine
price change loses the observation for ever.

The reconciliation tests are the other half: a snapshot that silently fails to
join to its played match looks exactly like a fixture nobody priced, and the
two failures this project actually expects are named clubs - a newly promoted
side and an accented spelling.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, normalize, snapshots  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

FIXTURE_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam", "Referee",
    "B365H", "B365D", "B365A", "BFEH", "BFED", "BFEA",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
    "B365>2.5", "B365<2.5", "BFE>2.5", "BFE<2.5",
    "AHh", "B365AHH", "B365AHA",
]


def raw_fixture_file(rows: list[dict]) -> pd.DataFrame:
    """A frame shaped exactly like the source's fixtures.csv."""
    defaults = {
        "Div": "E0", "Time": "15:00", "Referee": "M Oliver",
        "B365H": 2.10, "B365D": 3.40, "B365A": 3.60,
        "BFEH": 2.18, "BFED": 3.55, "BFEA": 3.75,
        "MaxH": 2.20, "MaxD": 3.60, "MaxA": 3.80,
        "AvgH": 2.05, "AvgD": 3.35, "AvgA": 3.50,
        "B365>2.5": 1.90, "B365<2.5": 1.90,
        "BFE>2.5": 1.98, "BFE<2.5": 1.96,
        "AHh": -0.25, "B365AHH": 1.95, "B365AHA": 1.95,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])[FIXTURE_COLUMNS]


@pytest.fixture
def one_round() -> pd.DataFrame:
    """Three fixtures, all in leagues the project models."""
    return raw_fixture_file(
        [
            {"Date": "05/09/2026", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool"},
            {"Date": "05/09/2026", "HomeTeam": "Chelsea", "AwayTeam": "Everton"},
            {"Div": "SP1", "Date": "06/09/2026",
             "HomeTeam": "Barcelona", "AwayTeam": "Ath Madrid"},
        ]
    )


@pytest.fixture
def mixed_divisions(one_round) -> pd.DataFrame:
    """The same round with a Championship match, as the real file has.

    The source ships twenty-two divisions in one file and the models price
    five of them.
    """
    return pd.concat(
        [
            one_round,
            raw_fixture_file(
                [{"Div": "E1", "Date": "05/09/2026",
                  "HomeTeam": "Watford", "AwayTeam": "Millwall"}]
            ),
        ],
        ignore_index=True,
    )


@pytest.fixture
def con(tmp_path):
    connection = database.connect(tmp_path / "snap.duckdb")
    yield connection
    connection.close()


# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------

def test_only_the_configured_leagues_are_archived(mixed_divisions):
    frame = snapshots.fixture_frame(mixed_divisions, leagues=list(config.LEAGUES))
    assert set(frame["league"]) == {"E0", "SP1"}
    assert len(frame) == 3


def test_every_division_is_kept_when_no_league_filter_is_given(mixed_divisions):
    frame = snapshots.fixture_frame(mixed_divisions, leagues=None)
    assert set(frame["league"]) == {"E0", "SP1", "E1"}


def test_a_file_with_no_div_column_is_refused(one_round):
    with pytest.raises(ValueError, match="Div"):
        snapshots.fixture_frame(one_round.drop(columns=["Div"]))


def test_team_names_are_canonicalised_on_the_way_in():
    raw = raw_fixture_file(
        [{"Date": "05/09/2026", "HomeTeam": "Manchester United",
          "AwayTeam": "Nottingham Forest"}]
    )
    frame = snapshots.fixture_frame(raw)
    assert frame["home_team"].iloc[0] == "Man United"
    assert frame["away_team"].iloc[0] == "Nott'm Forest"
    # The raw spelling is kept alongside, because a name that later fails to
    # reconcile is only diagnosable from what the source actually wrote.
    assert frame["home_team_raw"].iloc[0] == "Manchester United"


# --------------------------------------------------------------------------
# The collection window
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "date,window",
    [
        ("2026-09-04", "weekend"),   # Friday
        ("2026-09-05", "weekend"),   # Saturday
        ("2026-09-06", "weekend"),   # Sunday
        ("2026-09-07", "weekend"),   # Monday
        ("2026-09-08", "midweek"),   # Tuesday
        ("2026-09-09", "midweek"),   # Wednesday
        ("2026-09-10", "midweek"),   # Thursday
    ],
)
def test_the_collection_window_follows_the_fixture_weekday(date, window):
    assert snapshots.collection_window(date)[0] == window


def test_the_nominal_deadline_is_the_preceding_friday_or_tuesday():
    """The source publishes deadlines, not per-row collection times.

    Friday 17:00 and Tuesday 13:00 British time, which in September is BST,
    an hour ahead of UTC. Storing UTC means the archive does not change
    meaning when the clocks go back.
    """
    _, saturday = snapshots.collection_window("2026-09-05")
    assert saturday == pd.Timestamp("2026-09-04 16:00:00")   # Fri 17:00 BST

    _, wednesday = snapshots.collection_window("2026-09-09")
    assert wednesday == pd.Timestamp("2026-09-08 12:00:00")  # Tue 13:00 BST


def test_a_winter_fixture_uses_gmt_not_a_fixed_offset():
    """January is GMT, so 17:00 British time *is* 17:00 UTC."""
    _, january = snapshots.collection_window("2027-01-16")   # a Saturday
    assert january == pd.Timestamp("2027-01-15 17:00:00")


# --------------------------------------------------------------------------
# Content hashing and dedupe
# --------------------------------------------------------------------------

def test_the_same_file_twice_stores_one_snapshot(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    first = snapshots.write_snapshot(con, snapshot, odds)
    assert first["new_snapshots"] == 3

    again_snapshot, again_odds = snapshots.build_snapshot(one_round)
    second = snapshots.write_snapshot(con, again_snapshot, again_odds)
    assert second["new_snapshots"] == 0
    assert second["repeat_snapshots"] == 3
    assert second["new_odds_rows"] == 0

    stored = con.execute("SELECT COUNT(*) FROM fixture_snapshots").fetchone()[0]
    assert stored == 3


def test_a_moved_price_creates_a_second_row_for_the_same_fixture(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)

    moved = one_round.copy()
    moved.loc[0, "B365H"] = 2.30
    later_snapshot, later_odds = snapshots.build_snapshot(moved)
    counts = snapshots.write_snapshot(con, later_snapshot, later_odds)

    assert counts["new_snapshots"] == 1
    assert counts["repeat_snapshots"] == 2

    history = con.execute(
        "SELECT COUNT(*) FROM fixture_snapshots WHERE home_team = 'Arsenal'"
    ).fetchone()[0]
    assert history == 2, "the earlier price must survive, not be overwritten"

    prices = con.execute(
        """
        SELECT o.price FROM snapshot_odds o
        JOIN fixture_snapshots f USING (content_hash)
        WHERE f.home_team = 'Arsenal' AND o.bookmaker = 'bet365'
          AND o.market = '1x2' AND o.selection = 'home'
        ORDER BY o.price
        """
    ).df()
    assert list(prices["price"]) == [2.10, 2.30]


def test_a_repeat_pull_moves_the_sighting_stamp_forward(con, one_round):
    early = dt.datetime(2026, 9, 4, 16, 0)
    late = dt.datetime(2026, 9, 5, 9, 0)
    snapshot, odds = snapshots.build_snapshot(one_round, pulled_at=early)
    snapshots.write_snapshot(con, snapshot, odds)
    snapshot, odds = snapshots.build_snapshot(one_round, pulled_at=late)
    snapshots.write_snapshot(con, snapshot, odds)

    row = con.execute(
        "SELECT first_pulled_at_utc, last_pulled_at_utc FROM fixture_snapshots "
        "WHERE home_team = 'Arsenal'"
    ).fetchone()
    assert row[0] == early, "the first sighting is the archive's whole point"
    assert row[1] == late, "how long a price stood is worth knowing"


def test_rounding_noise_is_not_a_price_change(con, one_round):
    """2.1 and 2.10 are the same price, and the source writes both."""
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)

    restated = one_round.copy()
    restated.loc[0, "B365H"] = 2.100000001
    snapshot, odds = snapshots.build_snapshot(restated)
    counts = snapshots.write_snapshot(con, snapshot, odds)
    assert counts["new_snapshots"] == 0


def test_a_moved_kickoff_time_counts_as_a_change(one_round):
    original, _ = snapshots.build_snapshot(one_round)
    moved = one_round.copy()
    moved.loc[0, "Time"] = "17:30"
    rescheduled, _ = snapshots.build_snapshot(moved)
    assert original["content_hash"].iloc[0] != rescheduled["content_hash"].iloc[0]


# --------------------------------------------------------------------------
# Odds format
# --------------------------------------------------------------------------

def test_odds_are_stored_long_and_reuse_the_shared_reader(one_round):
    _, odds = snapshots.build_snapshot(one_round)
    assert set(odds.columns) == {
        "content_hash", "bookmaker", "phase", "market", "selection", "line", "price",
    }
    assert set(odds["market"]) == {"1x2", "total_goals", "asian_handicap"}
    # The fixtures file publishes no closing prices - every C-prefixed column
    # in it is empty - so a scan can only ever see opening prices.
    assert set(odds["phase"]) == {"open"}


def test_the_away_handicap_line_is_negated_exactly_as_in_the_season_files(one_round):
    """The bug this project already paid for once, guarded on the new path."""
    _, odds = snapshots.build_snapshot(one_round)
    handicap = odds[odds["market"] == "asian_handicap"]
    home = handicap[handicap["selection"] == "home"]["line"].unique()
    away = handicap[handicap["selection"] == "away"]["line"].unique()
    assert list(home) == [-0.25]
    assert list(away) == [0.25]


def test_betfair_exchange_totals_and_handicaps_are_captured(one_round):
    """They were silently dropped until the fixtures work exposed it (B10)."""
    _, odds = snapshots.build_snapshot(one_round)
    exchange = odds[odds["bookmaker"] == "betfair_exchange"]
    assert set(exchange["market"]) == {"1x2", "total_goals"}


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------

def test_a_file_whose_fixtures_have_been_played_is_stale(one_round):
    snapshot, _ = snapshots.build_snapshot(one_round)
    report = snapshots.staleness(snapshot, now=dt.datetime(2026, 9, 20))
    assert report["stale"]
    assert "already been played" in report["reasons"][0]

    with pytest.raises(snapshots.StaleFixtures, match="already been played"):
        snapshots.check(snapshot, now=dt.datetime(2026, 9, 20))


def test_an_upcoming_file_is_not_stale(one_round):
    snapshot, _ = snapshots.build_snapshot(one_round)
    report = snapshots.staleness(snapshot, now=dt.datetime(2026, 9, 4))
    assert not report["stale"]
    snapshots.check(snapshot, now=dt.datetime(2026, 9, 4))  # must not raise


def test_an_old_file_is_stale_even_when_its_fixtures_are_still_ahead(
    tmp_path, one_round
):
    """The subtle case: a cached copy nobody refreshed, describing next week."""
    path = tmp_path / "fixtures.csv"
    path.write_text("cached", encoding="utf-8")
    stamp = dt.datetime(2026, 9, 1).timestamp()
    import os

    os.utime(path, (stamp, stamp))

    snapshot, _ = snapshots.build_snapshot(one_round)
    report = snapshots.staleness(snapshot, path, now=dt.datetime(2026, 9, 4, 12))
    assert report["stale"]
    assert "cached copy" in report["reasons"][0]


def test_an_empty_file_is_stale_rather_than_silently_scannable():
    report = snapshots.staleness(pd.DataFrame())
    assert report["stale"]


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------

def test_a_played_fixture_is_joined_on_date_and_canonical_names(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-05", "Arsenal", "Liverpool")])

    result = snapshots.reconcile(con)
    assert result["matched"] == 1
    assert result["unmatched"] == 2

    row = con.execute(
        "SELECT match_id FROM fixture_snapshots WHERE home_team = 'Arsenal'"
    ).fetchone()
    assert row[0] is not None


def test_a_postponement_of_one_day_still_joins(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-06", "Arsenal", "Liverpool")])

    assert snapshots.reconcile(con)["matched"] == 1


def test_a_postponement_of_a_fortnight_does_not_join(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-19", "Arsenal", "Liverpool")])

    result = snapshots.reconcile(con)
    assert result["matched"] == 0
    assert result["unmatched"] == 3


def test_a_newly_promoted_club_joins_when_the_spellings_agree(con):
    """Promoted sides are where reconciliation fails in practice.

    Both files come from the same source, so they normally agree; this pins
    that the join does not depend on the club having history in the database.
    """
    raw = raw_fixture_file(
        [{"Date": "05/09/2026", "HomeTeam": "Sunderland", "AwayTeam": "Leeds"}]
    )
    snapshot, odds = snapshots.build_snapshot(raw)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-05", "Sunderland", "Leeds")])

    assert snapshots.reconcile(con)["matched"] == 1


def test_an_accented_spelling_joins_through_canonicalisation(con):
    """`Köln` on one side and `FC Koln` on the other must be one club."""
    raw = raw_fixture_file(
        [{"Div": "D1", "Date": "05/09/2026",
          "HomeTeam": "Köln", "AwayTeam": "Bayern München"}]
    )
    snapshot, odds = snapshots.build_snapshot(raw)
    assert snapshot["home_team"].iloc[0] == "FC Koln"
    assert snapshot["away_team"].iloc[0] == "Bayern Munich"
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("D1", "2026-09-05", "FC Koln", "Bayern Munich")])

    assert snapshots.reconcile(con)["matched"] == 1


def test_an_unresolvable_name_is_reported_not_dropped(con):
    raw = raw_fixture_file(
        [{"Date": "05/09/2026", "HomeTeam": "Wanderers FC", "AwayTeam": "Rovers"}]
    )
    snapshot, odds = snapshots.build_snapshot(raw)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-05", "Bolton", "Blackburn")])

    result = snapshots.reconcile(con)
    assert result["matched"] == 0
    assert result["unmatched"] == 1
    assert "Wanderers FC" in set(result["unmatched_rows"]["home_team"])

    # And it is still in the archive: the prices are the irreplaceable part.
    assert con.execute("SELECT COUNT(*) FROM fixture_snapshots").fetchone()[0] == 1


def test_reconciliation_is_idempotent(con, one_round):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    _insert_matches(con, [("E0", "2026-09-05", "Arsenal", "Liverpool")])

    first = snapshots.reconcile(con)
    second = snapshots.reconcile(con)
    assert first["matched"] == 1
    assert second["matched"] == 0
    assert second["already_matched"] == 1


def test_coverage_separates_a_failed_join_from_a_fixture_still_to_come(con):
    """Only one of these two is a defect, and the report must say which.

    Dates far either side of any plausible run date, because `coverage` reads
    the real clock and a test that only works this month is worse than none.
    """
    raw = raw_fixture_file(
        [
            {"Date": "05/09/2020", "HomeTeam": "Arsenal", "AwayTeam": "Liverpool"},
            {"Date": "31/12/2099", "HomeTeam": "Chelsea", "AwayTeam": "Everton"},
        ]
    )
    snapshot, odds = snapshots.build_snapshot(raw)
    snapshots.write_snapshot(con, snapshot, odds)
    snapshots.reconcile(con)

    table = snapshots.coverage(con).set_index("league")
    assert table.loc["E0", "awaiting_kickoff"] == 1
    assert table.loc["E0", "unmatched_past"] == 1
    assert table.loc["E0", "fixtures"] == 2


def _insert_matches(con, rows: list[tuple]) -> None:
    frame = pd.DataFrame(
        [
            {
                "match_id": normalize.make_match_id(
                    league, pd.Timestamp(date), home, away
                ),
                "league": league,
                "league_name": config.LEAGUES.get(league, league),
                "country": None,
                "season_start_year": 2026,
                "season": "2026/27",
                "date": pd.Timestamp(date).date(),
                "kickoff_time": "15:00",
                "home_team": home,
                "away_team": away,
                "referee": None,
                "home_goals": 1,
                "away_goals": 0,
                "result": "H",
            }
            for league, date, home, away in rows
        ]
    )
    con.register("incoming", frame)
    con.execute(
        "INSERT INTO matches (match_id, league, league_name, country, "
        "season_start_year, season, date, kickoff_time, home_team, away_team, "
        "referee, home_goals, away_goals, result) "
        "SELECT match_id, league, league_name, country, season_start_year, "
        "season, date, kickoff_time, home_team, away_team, referee, "
        "home_goals, away_goals, result FROM incoming"
    )
    con.unregister("incoming")


# --------------------------------------------------------------------------
# The CSV mirror, which is the only backup this archive has
# --------------------------------------------------------------------------
# The database is not tracked in git (BACKLOG B3) because every other table
# rebuilds from static files in two minutes. This one rebuilds from nothing, so
# `data/snapshots/` is committed and is the sole copy of these prices.

def test_the_export_writes_both_halves_of_the_archive(con, one_round, tmp_path):
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)

    written = snapshots.export(con, tmp_path / "mirror")
    assert written["fixtures"] == 3
    assert written["prices"] > 0
    assert (tmp_path / "mirror" / "fixture_snapshots.csv").exists()
    assert (tmp_path / "mirror" / "snapshot_odds.csv").exists()


def test_the_mirror_restores_into_an_empty_database(con, one_round, tmp_path):
    """The half that makes it a backup rather than a gesture.

    A fresh clone rebuilds every other table from the season files. This is
    how it gets the archive, and nothing else can supply it.
    """
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    snapshots.export(con, tmp_path / "mirror")

    restored = database.connect(tmp_path / "restored.duckdb")
    counts = snapshots.import_export(restored, tmp_path / "mirror")
    assert counts["new_snapshots"] == 3

    before = con.execute(
        "SELECT content_hash, home_team, away_team FROM fixture_snapshots "
        "ORDER BY content_hash"
    ).fetchall()
    after = restored.execute(
        "SELECT content_hash, home_team, away_team FROM fixture_snapshots "
        "ORDER BY content_hash"
    ).fetchall()
    assert before == after

    prices_before = con.execute(
        "SELECT COUNT(*) FROM snapshot_odds"
    ).fetchone()[0]
    prices_after = restored.execute(
        "SELECT COUNT(*) FROM snapshot_odds"
    ).fetchone()[0]
    assert prices_before == prices_after
    restored.close()


def test_importing_twice_merges_rather_than_duplicating(con, one_round, tmp_path):
    """It goes through the same content-hash dedupe a live pull does."""
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)
    snapshots.export(con, tmp_path / "mirror")

    restored = database.connect(tmp_path / "restored.duckdb")
    snapshots.import_export(restored, tmp_path / "mirror")
    second = snapshots.import_export(restored, tmp_path / "mirror")
    assert second["new_snapshots"] == 0
    assert second["repeat_snapshots"] == 3
    assert restored.execute(
        "SELECT COUNT(*) FROM fixture_snapshots"
    ).fetchone()[0] == 3
    restored.close()


def test_the_export_is_ordered_so_a_commit_diff_is_only_new_prices(
    con, one_round, tmp_path
):
    """Re-exporting unchanged data must produce a byte-identical file.

    Otherwise every run shows a diff of reordered rows and the week's actual
    new prices are invisible in it.
    """
    snapshot, odds = snapshots.build_snapshot(one_round)
    snapshots.write_snapshot(con, snapshot, odds)

    snapshots.export(con, tmp_path / "a")
    snapshots.export(con, tmp_path / "b")
    for name in ("fixture_snapshots.csv", "snapshot_odds.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_exporting_an_empty_archive_is_harmless(tmp_path):
    con = database.connect(tmp_path / "bare.duckdb")
    written = snapshots.export(con, tmp_path / "mirror")
    assert written["fixtures"] == 0
    con.close()


def test_importing_with_no_mirror_present_is_harmless(tmp_path):
    con = database.connect(tmp_path / "bare.duckdb")
    counts = snapshots.import_export(con, tmp_path / "absent")
    assert counts["new_snapshots"] == 0
    con.close()
