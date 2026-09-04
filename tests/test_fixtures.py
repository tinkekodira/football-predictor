"""Tests for the season calendar.

The one that matters most is `test_unplayed_fixtures_carry_null_not_zero`.
Understat reports an unplayed fixture with an xG of **0.0**, not a missing one,
and `understat.match_frame` exists partly to throw those rows away before they
reach a model. This module deliberately keeps them, so it is the one place in
the project where those zeros are handled rather than avoided, and a regression
there would feed twenty teams of "failed to have a shot" into the ratings.

The timezone tests are the other kind worth having: they are cheap, and getting
local dates wrong scatters one matchday across two days of the calendar in a
way that looks like missing data rather than like a bug.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from fbedge import fixtures


def payload(entries: list[dict]) -> dict:
    return {"dates": entries, "teams": {}}


def entry(
    match_id: str,
    when: str,
    home: str,
    away: str,
    played: bool = False,
    goals=(None, None),
    xg=(0.0, 0.0),
) -> dict:
    return {
        "id": match_id,
        "datetime": when,
        "h": {"title": home},
        "a": {"title": away},
        "isResult": played,
        "goals": {"h": goals[0], "a": goals[1]},
        "xG": {"h": xg[0], "a": xg[1]},
    }


# ----------------------------------------------------------------------
# Reading a season payload
# ----------------------------------------------------------------------


def test_keeps_played_and_unplayed_alike():
    """The whole point of this module, against `understat.match_frame`."""
    frame = fixtures.fixture_frame(
        payload([
            entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool"),
            entry("2", "2026-08-21 19:00:00", "Arsenal", "Chelsea",
                  played=True, goals=(2, 1), xg=(1.8, 0.9)),
        ]),
        "E0", 2026,
    )
    assert len(frame) == 2
    assert set(frame["played"]) == {True, False}


def test_unplayed_fixtures_carry_null_not_zero():
    """Understat reports an unplayed match with xG of 0.0, not as missing.

    Writing that zero through would tell anything downstream that both sides
    genuinely failed to create a chance. NULL is the only honest value, and it
    is what makes this table safe to average over.
    """
    frame = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool",
                       played=False, goals=(None, None), xg=(0.0, 0.0))]),
        "E0", 2026,
    )
    row = frame.iloc[0]
    assert row["home_xg"] is None
    assert row["away_xg"] is None
    assert row["home_goals"] is None


def test_played_fixtures_keep_their_numbers():
    frame = fixtures.fixture_frame(
        payload([entry("9", "2026-08-21 19:00:00", "Arsenal", "Chelsea",
                       played=True, goals=(3, 0), xg=(2.4, 0.7))]),
        "E0", 2026,
    )
    row = frame.iloc[0]
    assert row["home_goals"] == 3
    assert row["home_xg"] == pytest.approx(2.4)


def test_team_names_are_translated_to_the_project_convention():
    """The calendar has to join to the model, which speaks football-data names.

    `Wolverhampton Wanderers` on one side and `Wolves` on the other is a team
    the model has never heard of, and it would show up as a fixture whose
    detail page cannot be priced.
    """
    frame = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00",
                       "Wolverhampton Wanderers", "Manchester City")]),
        "E0", 2026,
    )
    row = frame.iloc[0]
    assert row["home_team"] == "Wolves"
    assert row["home_team_understat"] == "Wolverhampton Wanderers"
    assert row["away_team"] == "Man City"


def test_fixtures_come_back_in_kick_off_order():
    frame = fixtures.fixture_frame(
        payload([
            entry("2", "2026-09-05 16:00:00", "B", "C"),
            entry("1", "2026-09-04 19:00:00", "A", "D"),
        ]),
        "E0", 2026,
    )
    assert list(frame["understat_id"]) == ["1", "2"]


# ----------------------------------------------------------------------
# Timezones and dates
# ----------------------------------------------------------------------


def test_local_time_is_derived_not_stored():
    """Understat gives UTC; a 19:00 kick-off is 21:00 in Zagreb in September."""
    frame = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool")]),
        "E0", 2026,
    )
    localised = fixtures.local_frame(frame, "Europe/Zagreb")
    assert localised.iloc[0]["kickoff_local"].hour == 21
    assert localised.iloc[0]["local_date"] == dt.date(2026, 9, 4)


def test_a_late_kick_off_stays_on_its_local_day():
    """A 22:30 local kick-off is 20:30 UTC and belongs to the same evening.

    The failure this guards against is grouping on the UTC date, which splits
    one matchday in two and looks like half the fixtures went missing.
    """
    frame = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 20:30:00", "Betis", "Real Madrid")]),
        "SP1", 2026,
    )
    localised = fixtures.local_frame(frame, "Europe/Zagreb")
    assert localised.iloc[0]["kickoff_local"].hour == 22
    assert localised.iloc[0]["local_date"] == dt.date(2026, 9, 4)


def test_matches_on_filters_by_local_date():
    frame = fixtures.fixture_frame(
        payload([
            entry("1", "2026-09-04 19:00:00", "A", "B"),
            entry("2", "2026-09-05 11:30:00", "C", "D"),
        ]),
        "E0", 2026,
    )
    day = fixtures.matches_on(frame, dt.date(2026, 9, 5))
    assert list(day["understat_id"]) == ["2"]


def test_dates_with_matches_is_sorted_and_unique():
    frame = fixtures.fixture_frame(
        payload([
            entry("1", "2026-09-05 16:00:00", "A", "B"),
            entry("2", "2026-09-04 19:00:00", "C", "D"),
            entry("3", "2026-09-05 18:30:00", "E", "F"),
        ]),
        "E0", 2026,
    )
    assert fixtures.dates_with_matches(frame) == [
        dt.date(2026, 9, 4), dt.date(2026, 9, 5)
    ]


def test_nearest_date_prefers_going_forwards_on_a_tie():
    """An empty day should send you to the next round, not back into results
    you have already seen."""
    frame = fixtures.fixture_frame(
        payload([
            entry("1", "2026-09-03 19:00:00", "A", "B"),
            entry("2", "2026-09-05 19:00:00", "C", "D"),
        ]),
        "E0", 2026,
    )
    # 4 September is one day from each.
    assert fixtures.nearest_date_with_matches(frame, dt.date(2026, 9, 4)) == \
        dt.date(2026, 9, 5)


def test_nearest_date_on_an_empty_calendar_is_none():
    assert fixtures.nearest_date_with_matches(
        pd.DataFrame(columns=["kickoff_utc"]), dt.date(2026, 9, 4)
    ) is None


def test_local_frame_survives_an_empty_calendar():
    """The home page calls this before it knows whether anything loaded."""
    out = fixtures.local_frame(pd.DataFrame(columns=["kickoff_utc"]))
    assert out.empty
    assert "local_date" in out.columns


# ----------------------------------------------------------------------
# Season boundaries
# ----------------------------------------------------------------------


def test_current_season_cuts_in_july():
    """August would be too late: the calendar is asked about a season before
    its first match, and La Liga has kicked off by mid-August."""
    assert fixtures.current_season(dt.date(2026, 9, 4)) == 2026
    assert fixtures.current_season(dt.date(2026, 7, 1)) == 2026
    assert fixtures.current_season(dt.date(2026, 6, 30)) == 2025
    assert fixtures.current_season(dt.date(2027, 5, 30)) == 2026


def test_default_leagues_are_the_five_with_a_source():
    leagues = fixtures.default_leagues()
    assert set(leagues) == {"E0", "SP1", "I1", "D1", "F1"}


# ----------------------------------------------------------------------
# Cache freshness
# ----------------------------------------------------------------------


def test_a_finished_season_is_never_refetched(tmp_path):
    """Nothing about 2019 can change, so re-downloading it is pure waste."""
    path = tmp_path / "E0_2019.json"
    path.write_text("{}", encoding="utf-8")
    assert not fixtures._is_stale(tmp_path, "E0", 2019, max_age_hours=0.0)


def test_the_current_season_goes_stale(tmp_path):
    """Results land during the day; a calendar showing this morning's state is
    simply wrong."""
    season = fixtures.current_season()
    path = tmp_path / f"E0_{season}.json"
    path.write_text("{}", encoding="utf-8")
    # An explicit clock rather than the wall one: a file written a microsecond
    # ago can carry an mtime marginally in the future on Windows, which makes
    # "is it older than zero hours" genuinely undecidable.
    now = dt.datetime.now()
    assert not fixtures._is_stale(
        tmp_path, "E0", season, max_age_hours=6.0, now=now
    )
    assert fixtures._is_stale(
        tmp_path, "E0", season, max_age_hours=1.0,
        now=now + dt.timedelta(hours=2),
    )


def test_a_missing_cache_is_not_reported_as_stale(tmp_path):
    """`fetch_season` downloads when there is no file, so forcing as well would
    just be a second request."""
    assert not fixtures._is_stale(
        tmp_path, "E0", fixtures.current_season(), max_age_hours=0.0
    )


# ----------------------------------------------------------------------
# Round trip through the database
# ----------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    frame = fixtures.fixture_frame(
        payload([
            entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool"),
            entry("2", "2026-08-21 19:00:00", "Arsenal", "Chelsea",
                  played=True, goals=(2, 1), xg=(1.8, 0.9)),
        ]),
        "E0", 2026,
    )
    assert fixtures.write_calendar(con, frame) == 2
    back = fixtures.load_calendar(con, season=2026)
    assert len(back) == 2
    assert set(back["understat_id"]) == {"1", "2"}


def test_reloading_a_season_replaces_rather_than_duplicates(tmp_path):
    """A result arrives by *changing* a row from unplayed to played.

    An insert-only load would keep the old row too, and the calendar would show
    a finished match as still to come - next to itself.
    """
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    before = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool")]),
        "E0", 2026,
    )
    fixtures.write_calendar(con, before)
    after = fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool",
                       played=True, goals=(0, 2), xg=(0.4, 2.1))]),
        "E0", 2026,
    )
    fixtures.write_calendar(con, after)

    back = fixtures.load_calendar(con, season=2026)
    assert len(back) == 1
    assert bool(back.iloc[0]["played"]) is True
    assert back.iloc[0]["away_goals"] == 2


def test_reloading_one_league_leaves_the_others_alone(tmp_path):
    """Refreshing the Premier League must not empty Serie A."""
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    fixtures.write_calendar(con, fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool")]),
        "E0", 2026,
    ))
    fixtures.write_calendar(con, fixtures.fixture_frame(
        payload([entry("2", "2026-09-04 20:45:00", "Genoa", "Como")]),
        "I1", 2026,
    ))
    fixtures.write_calendar(con, fixtures.fixture_frame(
        payload([entry("1", "2026-09-04 19:00:00", "Ipswich", "Liverpool")]),
        "E0", 2026,
    ))
    back = fixtures.load_calendar(con, season=2026)
    assert set(back["league"]) == {"E0", "I1"}
    assert len(back) == 2


def test_an_empty_payload_is_refused_rather_than_written(tmp_path, monkeypatch):
    """A league silently absent from a calendar looks exactly like a league
    with no matches that week."""
    from fbedge import understat

    monkeypatch.setattr(
        understat, "fetch_season", lambda *a, **k: {"dates": [], "teams": {}}
    )
    with pytest.raises(understat.UnderstatError, match="no fixtures"):
        fixtures.fetch_calendar(["E0"], 2026, tmp_path)
