"""Tests for the multi-source forward calendar.

The one that matters most is `test_every_team_in_a_fetched_calendar_resolves`.
A club name that fails to map produces a fixture that is present, looks
correct, and joins to nothing for the rest of the season - which on a page is
indistinguishable from a fixture that was never scheduled. So the rule is that
an unresolved name raises, and these tests pin that it raises rather than
warns.

The network is never touched here. The openfootball payloads are the shapes the
live files actually have, including the three different things their `score`
field turns out to be, and the football-data.org payload follows their
published v4 schema - which is **not** verified against a live response,
because the endpoint answers 403 without a token and none was available.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import calendar as calendar_mod  # noqa: E402
from fbedge import config, database, normalize  # noqa: E402


KNOWN_E0 = {
    "Arsenal", "Liverpool", "Man City", "Man United", "Brighton", "Tottenham",
    "Nott'm Forest", "Newcastle", "Leeds", "Hull", "Coventry", "Ipswich",
}
KNOWN_SP1 = {
    "Barcelona", "Real Madrid", "Ath Madrid", "Celta", "Espanol", "Vallecano",
    "Sociedad", "Alaves", "Santander", "Betis",
}


def openfootball_payload(matches: list[dict]) -> dict:
    return {"name": "English Premier League 2026/27", "matches": matches}


@pytest.fixture
def cache(tmp_path) -> Path:
    return tmp_path / "calendar"


# --------------------------------------------------------------------------
# Name resolution, which is the whole risk
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "feed_name,expected",
    [
        # Corporate noise, which is 81 of the 96 real cases.
        ("Arsenal FC", "Arsenal"),
        ("Liverpool FC", "Liverpool"),
        ("Hull City AFC", "Hull"),
        ("Coventry City FC", "Coventry"),
        ("Leeds United FC", "Leeds"),
        ("Newcastle United FC", "Newcastle"),
        ("Nottingham Forest FC", "Nott'm Forest"),
        ("Tottenham Hotspur FC", "Tottenham"),
        ("Manchester City FC", "Man City"),
        ("Manchester United FC", "Man United"),
    ],
)
def test_corporate_suffixes_resolve_without_an_alias(feed_name, expected):
    assert normalize.resolve_external_team(feed_name, KNOWN_E0) == expected


@pytest.mark.parametrize(
    "feed_name,expected",
    [
        ("Brighton & Hove Albion FC", "Brighton"),
        ("Club Atlético de Madrid", "Ath Madrid"),
        ("RC Celta de Vigo", "Celta"),
        ("RCD Espanyol de Barcelona", "Espanol"),
        ("Rayo Vallecano de Madrid", "Vallecano"),
        ("Real Sociedad de Fútbol", "Sociedad"),
        ("Deportivo Alavés", "Alaves"),
        ("Real Racing Club de Santander", "Santander"),
    ],
)
def test_the_names_noise_stripping_cannot_reach_need_an_alias(feed_name, expected):
    """The fifteen entries hand-written from what the feed actually said."""
    known = KNOWN_E0 | KNOWN_SP1
    assert normalize.resolve_external_team(
        feed_name, known, calendar_mod.OPENFOOTBALL_ALIASES
    ) == expected


def test_an_accented_name_resolves_through_the_same_path():
    assert normalize.resolve_external_team(
        "FC Bayern München", {"Bayern Munich"}
    ) == "Bayern Munich"


def test_an_unknown_club_returns_none_rather_than_a_near_miss():
    """No fuzzy fallback. None is a real answer and the caller must handle it.

    The Understat integration established why: 41 of 156 names differed between
    two sources and a similarity score would have merged Milan with Inter.
    """
    assert normalize.resolve_external_team("Wanderers FC", KNOWN_E0) is None
    assert normalize.resolve_external_team("Real Madrid CF", KNOWN_E0) is None


def test_every_team_in_a_fetched_calendar_resolves(cache):
    """The test the brief asked for, on a payload shaped like the real file."""
    payload = openfootball_payload(
        [
            {"round": "Matchday 1", "date": "2026-08-21", "time": "20:00",
             "team1": "Arsenal FC", "team2": "Coventry City FC",
             "score": {"ht": [2, 0], "ft": [3, 0]}},
            {"round": "Matchday 1", "date": "2026-08-22", "time": "15:00",
             "team1": "Brighton & Hove Albion FC", "team2": "Hull City AFC",
             "score": None},
            {"round": "Matchday 2", "date": "2026-08-29", "time": "17:30",
             "team1": "Manchester United FC", "team2": "Nottingham Forest FC",
             "score": None},
        ]
    )
    frame = calendar_mod._openfootball_frame(payload, "E0", 2026, KNOWN_E0)
    assert len(frame) == 3
    assert set(frame["home_team"]) <= KNOWN_E0
    assert set(frame["away_team"]) <= KNOWN_E0


def test_an_unresolved_name_raises_and_names_the_club(cache):
    payload = openfootball_payload(
        [
            {"round": "Matchday 1", "date": "2026-08-21", "time": "20:00",
             "team1": "Arsenal FC", "team2": "Wanderers Athletic", "score": None},
        ]
    )
    with pytest.raises(calendar_mod.UnresolvedTeams) as caught:
        calendar_mod._openfootball_frame(payload, "E0", 2026, KNOWN_E0)
    assert "Wanderers Athletic" in caught.value.names
    assert "OPENFOOTBALL_ALIASES" in str(caught.value)


# --------------------------------------------------------------------------
# openfootball's three score shapes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [
        (None, (None, None)),                      # unplayed
        ({"ft": [3, 0], "ht": [2, 0]}, (3, 0)),    # the documented shape
        ([2, 1], (2, 1)),                          # es.1, de.1 and fr.1 do this
        ({}, (None, None)),
        ({"ht": [1, 0]}, (None, None)),            # half-time only, not played out
    ],
)
def test_the_score_field_has_three_shapes_and_all_are_read(score, expected):
    """Found live, not guessed.

    The Spanish, German and French 2026/27 files carry a bare list where the
    English and Italian ones carry a dict. Reading only the dict shape would
    have marked a slice of each of those seasons unplayed - which on a calendar
    means showing finished matches as still to come.
    """
    assert calendar_mod._openfootball_score(score) == expected


def test_a_fixture_with_no_kickoff_time_still_appears():
    """A kick-off nobody has decided is not a fixture that does not exist."""
    payload = openfootball_payload(
        [{"round": "Matchday 30", "date": "2027-04-10", "team1": "Arsenal FC",
          "team2": "Liverpool FC", "score": None}]
    )
    frame = calendar_mod._openfootball_frame(payload, "E0", 2026, KNOWN_E0)
    assert len(frame) == 1
    assert pd.notna(frame["kickoff_utc"].iloc[0])


def test_the_openfootball_season_directory_is_the_hyphenated_form():
    assert calendar_mod._openfootball_season(2026) == "2026-27"
    assert calendar_mod._openfootball_season(1999) == "1999-00"


# --------------------------------------------------------------------------
# football-data.org
# --------------------------------------------------------------------------

def test_football_data_org_parses_its_documented_shape():
    """**Unverified against a live response**, and this test says so.

    No token was available and the endpoint answers 403 without one, so this
    pins the schema the parser expects rather than one that was observed. If
    the real feed differs, correct it here first.
    """
    payload = {
        "matches": [
            {
                "id": 497712,
                "utcDate": "2026-08-21T19:00:00Z",
                "status": "FINISHED",
                "homeTeam": {"id": 57, "name": "Arsenal FC"},
                "awayTeam": {"id": 1076, "name": "Coventry City FC"},
                "score": {"fullTime": {"home": 3, "away": 0}},
            },
            {
                "id": 497713,
                "utcDate": "2026-08-22T14:00:00Z",
                "status": "TIMED",
                "homeTeam": {"id": 397, "name": "Brighton & Hove Albion FC"},
                "awayTeam": {"id": 322, "name": "Hull City AFC"},
                "score": {"fullTime": {"home": None, "away": None}},
            },
        ]
    }
    frame = calendar_mod._football_data_org_frame(payload, "E0", 2026, KNOWN_E0)
    assert list(frame["home_team"]) == ["Arsenal", "Brighton"]
    assert list(frame["played"]) == [True, False]
    assert frame["kickoff_utc"].iloc[0] == pd.Timestamp("2026-08-21 19:00:00")
    assert frame["external_id"].iloc[0] == "497712"


def test_football_data_org_refuses_without_a_token(tmp_path, monkeypatch):
    """No key must be a clear message, not a stack trace or an empty frame."""
    monkeypatch.delenv(config.CALENDAR_TOKEN_ENV, raising=False)
    with pytest.raises(calendar_mod.CalendarError, match="token"):
        calendar_mod.fetch_football_data_org("E0", 2026, tmp_path, force=True)


def test_the_rate_limit_is_enforced_before_the_request_not_after(monkeypatch):
    """Ten a minute is the published free-tier allowance; we honour it locally.

    Relying on the server to reject the eleventh call wastes it, and repeated
    hammering is how access gets withdrawn without warning.
    """
    calendar_mod._LAST_CALLS.clear()
    slept = []
    monkeypatch.setattr(calendar_mod.time, "sleep", slept.append)
    for _ in range(config.CALENDAR_CALLS_PER_MINUTE):
        calendar_mod._respect_rate_limit()
    assert not slept, "the allowance itself must not be throttled"
    calendar_mod._respect_rate_limit()
    assert slept, "the call past the allowance must wait"
    calendar_mod._LAST_CALLS.clear()


# --------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------

def test_auto_picks_the_free_source_when_no_token_is_configured(monkeypatch):
    monkeypatch.delenv(config.CALENDAR_TOKEN_ENV, raising=False)
    assert calendar_mod.resolve_source("auto") == calendar_mod.OPENFOOTBALL


def test_auto_picks_the_keyed_source_when_a_token_is_configured(monkeypatch):
    monkeypatch.setenv(config.CALENDAR_TOKEN_ENV, "a-token")
    assert calendar_mod.resolve_source("auto") == calendar_mod.FOOTBALL_DATA_ORG


def test_an_explicit_source_always_wins(monkeypatch):
    monkeypatch.setenv(config.CALENDAR_TOKEN_ENV, "a-token")
    assert calendar_mod.resolve_source("openfootball") == calendar_mod.OPENFOOTBALL


def test_an_unknown_source_is_rejected_by_name():
    with pytest.raises(calendar_mod.CalendarError, match="Unknown calendar source"):
        calendar_mod.resolve_source("somebody's blog")


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def test_a_finished_season_is_never_refetched(tmp_path):
    path = tmp_path / "cached.json"
    path.write_text("{}", encoding="utf-8")
    import os

    old = dt.datetime(2020, 1, 1).timestamp()
    os.utime(path, (old, old))
    assert not calendar_mod._stale(path, 2019)


def test_a_season_in_progress_is_refetched_once_the_cache_ages(tmp_path):
    path = tmp_path / "cached.json"
    path.write_text("{}", encoding="utf-8")
    import os
    from fbedge import fixtures as fixtures_mod

    current = fixtures_mod.current_season()
    fresh = dt.datetime.now().timestamp()
    os.utime(path, (fresh, fresh))
    assert not calendar_mod._stale(path, current)

    stale = (
        dt.datetime.now() - dt.timedelta(hours=config.CALENDAR_CACHE_HOURS + 1)
    ).timestamp()
    os.utime(path, (stale, stale))
    assert calendar_mod._stale(path, current)


def test_a_missing_cache_is_stale(tmp_path):
    assert calendar_mod._stale(tmp_path / "absent.json", 2019)


# --------------------------------------------------------------------------
# Storage and provenance
# --------------------------------------------------------------------------

@pytest.fixture
def con(tmp_path):
    connection = database.connect(tmp_path / "cal.duckdb")
    yield connection
    connection.close()


def _frame(source: str, league: str = "E0") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": source, "external_id": "1", "league": league,
                "season_start_year": 2026,
                "kickoff_utc": pd.Timestamp("2026-08-21 19:00"),
                "home_team": "Arsenal", "away_team": "Liverpool",
                "played": False, "home_goals": None, "away_goals": None,
            }
        ]
    )


def test_provenance_is_stored_and_queryable(con):
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL))
    stored = calendar_mod.load(con)
    assert list(stored["source"]) == [calendar_mod.OPENFOOTBALL]


def test_two_sources_coexist_rather_than_overwriting_each_other(con):
    """Provenance is only worth storing if the sources can be compared."""
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL))
    calendar_mod.write(con, _frame(calendar_mod.UNDERSTAT))
    assert len(calendar_mod.load(con)) == 2
    assert len(calendar_mod.load(con, source=calendar_mod.OPENFOOTBALL)) == 1


def test_rewriting_one_source_replaces_only_its_own_rows(con):
    """A result arrives by changing a row, so the load replaces rather than appends."""
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL))
    calendar_mod.write(con, _frame(calendar_mod.UNDERSTAT))
    played = _frame(calendar_mod.OPENFOOTBALL)
    played.loc[0, ["played", "home_goals", "away_goals"]] = [True, 2, 1]
    calendar_mod.write(con, played)

    stored = calendar_mod.load(con)
    assert len(stored) == 2
    open_row = stored[stored["source"] == calendar_mod.OPENFOOTBALL].iloc[0]
    assert bool(open_row["played"]) is True
    understat_row = stored[stored["source"] == calendar_mod.UNDERSTAT].iloc[0]
    assert bool(understat_row["played"]) is False


def test_rewriting_one_league_leaves_the_others_alone(con):
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL, "E0"))
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL, "SP1"))
    calendar_mod.write(con, _frame(calendar_mod.OPENFOOTBALL, "E0"))
    assert len(calendar_mod.load(con)) == 2


def test_load_returns_an_empty_frame_before_anything_is_written(tmp_path):
    con = database.connect(tmp_path / "bare.duckdb")
    assert calendar_mod.load(con).empty
    con.close()


# --------------------------------------------------------------------------
# The calendar is additive, never a dependency
# --------------------------------------------------------------------------

def test_the_project_works_with_no_calendar_table_at_all(tmp_path):
    """The requirement that matters most and is easiest to break by accident.

    Nothing outside `build_calendar.py` may need this table to exist.
    """
    con = database.connect(tmp_path / "nocal.duckdb")
    tables = {
        row[0] for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert calendar_mod.TABLE_NAME not in tables
    assert calendar_mod.load(con).empty
    con.close()
