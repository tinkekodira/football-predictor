"""The season calendar: every fixture, played or not.

Everything else in this project works backwards from results. A calendar has to
work forwards, and that turns out to need a different source from the one the
model is built on.

**Why not football-data.co.uk.** It is the project's primary source and it does
publish `fixtures.csv`, which `ingest.download_upcoming_fixtures` already
fetches. That file carries about three days of fixtures - 48 rows on the day
this module was written, of which two were in the top five leagues - because
its purpose is to ship prices for imminent matches, not to be a calendar. It
cannot answer "what is on in October".

**Why Understat.** Its league payload has a `dates` array holding the *whole*
season, played and unplayed alike: 380 Premier League fixtures in 2026/27, of
which 20 had been played. It is already integrated for xG, the team-name
aliases are already derived and tested, and it needs no new dependency.

`understat.match_frame` deliberately throws the unplayed fixtures away, and is
right to - an unplayed fixture carries a scoreline of null and an xG of *zero*
rather than a missing one, and feeding those zeros into the ratings would
quietly tell the model that twenty teams failed to have a shot. This module is
its mirror image: it keeps everything, and takes care to write NULL rather than
zero for anything not yet played.

**Kick-off times are UTC and are stored that way.** Understat gives
`2026-09-04 19:00:00` for a match that starts at 21:00 in Zagreb. Storing local
time would bake one reader's timezone into the database and break the moment
anyone travels or the clocks change, so the conversion happens on the way out,
in `local_frame`. The grouping a calendar needs is by *local* date, because a
20:45 kick-off in Rome belongs to the evening the viewer is looking at.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, understat

# Where the calendar is read for display. A single default rather than a
# per-user setting, because this is a personal project with one user; it is a
# named constant so that the assumption is visible and changeable in one place.
DISPLAY_TIMEZONE = "Europe/Zagreb"

TABLE_NAME = "fixtures"

# A season's fixtures are re-fetched rather than merged, because results arrive
# by *changing* an existing row from unplayed to played. An insert-only load
# would leave the old row behind and show a finished match as still to come.
SEASON_PAYLOAD_KEY = "dates"


def fixture_frame(payload: dict, league: str, season: int) -> pd.DataFrame:
    """Every fixture in a season payload, played or not.

    The mirror of `understat.match_frame`, which keeps only results. Unplayed
    rows carry NULL for goals and xG - never zero, which is what the source
    actually supplies and what would poison anything that averaged them.
    """
    rows = []
    for entry in payload.get(SEASON_PAYLOAD_KEY, []):
        played = bool(entry.get("isResult"))
        kickoff = pd.to_datetime(entry["datetime"], utc=False)
        rows.append(
            {
                "understat_id": str(entry.get("id")),
                "league": league,
                "season_start_year": season,
                "kickoff_utc": kickoff,
                "home_team": understat.to_football_data_name(entry["h"]["title"]),
                "away_team": understat.to_football_data_name(entry["a"]["title"]),
                "home_team_understat": entry["h"]["title"],
                "away_team_understat": entry["a"]["title"],
                "played": played,
                "home_goals": _int_or_none(entry, "goals", "h", played),
                "away_goals": _int_or_none(entry, "goals", "a", played),
                "home_xg": _float_or_none(entry, "xG", "h", played),
                "away_xg": _float_or_none(entry, "xG", "a", played),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("kickoff_utc").reset_index(drop=True)


def _int_or_none(entry: dict, block: str, side: str, played: bool):
    if not played:
        return None
    return understat._to_int(entry.get(block, {}).get(side))


def _float_or_none(entry: dict, block: str, side: str, played: bool):
    if not played:
        return None
    return understat._to_float(entry.get(block, {}).get(side))


# How stale a cached season payload may be before the calendar refetches it.
# `understat.fetch_season` caches for ever on purpose, which is right for a
# finished season and wrong for one in progress: results land, kick-off times
# get moved, and a calendar showing yesterday's state is simply wrong. Six
# hours keeps a day's football current without hammering a free source.
CURRENT_SEASON_MAX_AGE_HOURS = 6.0


def _is_stale(
    cache_dir: Path,
    league: str,
    season: int,
    max_age_hours: float,
    now: dt.datetime | None = None,
) -> bool:
    """Whether a cached season payload is too old to trust for a calendar.

    Only ever true for a season still being played. A finished season cannot
    change, so re-fetching it would be pure waste.
    """
    if season != current_season():
        return False
    path = Path(cache_dir) / f"{league}_{season}.json"
    if not path.exists():
        return False  # no cache at all; fetch_season will download anyway
    age = (now or dt.datetime.now()) - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > dt.timedelta(hours=max_age_hours)


def fetch_calendar(
    leagues: list[str],
    season: int,
    cache_dir: Path,
    refresh: bool = False,
    max_age_hours: float = CURRENT_SEASON_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Fixtures for several leagues in one season, concatenated.

    A league that fails is reported by raising, not by being quietly missing:
    a calendar with a league silently absent looks exactly like a league with
    no matches that week, and this project has been bitten by that shape of
    bug before.

    `refresh` forces a download for every league; without it the in-progress
    season is refetched once its cache passes `max_age_hours` and finished
    seasons are always served from disk.
    """
    frames = []
    for league in leagues:
        force = refresh or _is_stale(cache_dir, league, season, max_age_hours)
        payload = understat.fetch_season(league, season, cache_dir, force=force)
        frame = fixture_frame(payload, league, season)
        if frame.empty:
            raise understat.UnderstatError(
                f"{league} {season}: the season payload held no fixtures. "
                "That is never right for a league in progress; check the feed "
                "before writing anything."
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        "kickoff_utc"
    ).reset_index(drop=True)


def create_table(con) -> None:
    """Create the fixtures table if it is not already there."""
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            understat_id        VARCHAR PRIMARY KEY,
            league              VARCHAR NOT NULL,
            season_start_year   INTEGER NOT NULL,
            kickoff_utc         TIMESTAMP NOT NULL,
            home_team           VARCHAR NOT NULL,
            away_team           VARCHAR NOT NULL,
            home_team_understat VARCHAR,
            away_team_understat VARCHAR,
            played              BOOLEAN NOT NULL,
            home_goals          INTEGER,
            away_goals          INTEGER,
            home_xg             DOUBLE,
            away_xg             DOUBLE
        )
        """
    )


def write_calendar(con, frame: pd.DataFrame) -> int:
    """Replace the stored calendar for every season the frame covers.

    Replace rather than insert, and per season rather than wholesale. A result
    arrives by *changing* a row from unplayed to played, so an insert-only load
    would keep the stale row and show a finished match as still to come.
    Scoping the delete to the seasons actually supplied means refreshing one
    season cannot wipe another.
    """
    if frame.empty:
        return 0
    create_table(con)
    seasons = sorted({int(v) for v in frame["season_start_year"].unique()})
    leagues = sorted({str(v) for v in frame["league"].unique()})
    con.register("incoming_fixtures", frame)
    con.execute(
        f"DELETE FROM {TABLE_NAME} WHERE season_start_year IN "
        f"({','.join(str(s) for s in seasons)}) AND league IN "
        f"({','.join(repr(l) for l in leagues)})"
    )
    con.execute(
        f"INSERT INTO {TABLE_NAME} SELECT * FROM incoming_fixtures"
    )
    con.unregister("incoming_fixtures")
    return len(frame)


def load_calendar(
    con,
    leagues: list[str] | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """Read the stored calendar back, still in UTC."""
    clauses, params = [], []
    if leagues:
        clauses.append(f"league IN ({','.join('?' for _ in leagues)})")
        params.extend(leagues)
    if season is not None:
        clauses.append("season_start_year = ?")
        params.append(season)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return con.execute(
        f"SELECT * FROM {TABLE_NAME} {where} ORDER BY kickoff_utc", params
    ).df()


def local_frame(
    frame: pd.DataFrame, timezone: str = DISPLAY_TIMEZONE
) -> pd.DataFrame:
    """Add local kick-off time and the local date to group by.

    The local date is the one a calendar has to use. A 20:45 kick-off in Rome
    is 18:45 UTC and belongs to the evening the viewer is looking at either
    way, but a 21:00 kick-off in Madrid during a summer month is 19:00 UTC, and
    anything later would start rolling over to the previous UTC day. Grouping
    on the UTC date would then scatter a single matchday across two.
    """
    if frame.empty:
        out = frame.copy()
        out["kickoff_local"] = pd.Series(dtype="datetime64[ns]")
        out["local_date"] = pd.Series(dtype="object")
        return out
    zone = ZoneInfo(timezone)
    utc = pd.to_datetime(frame["kickoff_utc"]).dt.tz_localize("UTC")
    local = utc.dt.tz_convert(zone)
    out = frame.copy()
    out["kickoff_local"] = local
    out["local_date"] = local.dt.date
    return out


def matches_on(
    frame: pd.DataFrame, day: dt.date, timezone: str = DISPLAY_TIMEZONE
) -> pd.DataFrame:
    """Every fixture kicking off on one local date, earliest first."""
    localised = local_frame(frame, timezone)
    if localised.empty:
        return localised
    return localised[localised["local_date"] == day].sort_values(
        ["kickoff_local", "league"]
    ).reset_index(drop=True)


def dates_with_matches(
    frame: pd.DataFrame, timezone: str = DISPLAY_TIMEZONE
) -> list[dt.date]:
    """Sorted local dates that have at least one fixture.

    What the date strip needs in order to mark the days worth landing on. A
    calendar that lets you scroll onto empty days is not wrong, but one that
    cannot tell you which days are empty is annoying.
    """
    localised = local_frame(frame, timezone)
    if localised.empty:
        return []
    return sorted(set(localised["local_date"]))


def nearest_date_with_matches(
    frame: pd.DataFrame,
    target: dt.date,
    timezone: str = DISPLAY_TIMEZONE,
) -> dt.date | None:
    """The day with fixtures closest to `target`, preferring later on a tie.

    Used to pick what the page opens on. Opening on an empty Tuesday in an
    international break is technically correct and useless; landing on the next
    day that has football is what a reader wants, and preferring *later* means
    an empty day sends you forwards to the next round rather than backwards
    into results you have already seen.
    """
    days = dates_with_matches(frame, timezone)
    if not days:
        return None
    return min(days, key=lambda day: (abs((day - target).days), -day.toordinal()))


def current_season(today: dt.date | None = None) -> int:
    """The season year a date falls in, cutting at 1 July.

    July rather than August, because the earliest of these leagues kicks off in
    mid-August and a cut in the quiet month either side of that cannot land in
    the middle of a season. The same reasoning as `evaluation.season_labels`,
    which cuts at 1 August for *results* - this one has to be earlier, because
    a calendar is asked about a season before it starts.
    """
    day = today or dt.date.today()
    return day.year if day.month >= 7 else day.year - 1


def default_leagues() -> list[str]:
    """The five leagues the calendar covers, in the project's usual order."""
    return [league for league in config.LEAGUES if league in understat.LEAGUE_SLUGS]
