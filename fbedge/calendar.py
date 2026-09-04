"""The rest of the season's fixtures, from sources that publish a whole one.

`fixtures.csv` carries the next few days - 198 rows across twenty-two divisions
on the day this module was written - because its purpose is to ship prices for
imminent matches. It cannot answer "what is on in October". This module can.

**Three sources behind one interface, chosen in `config.CALENDAR_SOURCE`.**

*`understat`* is the shipped default and needs no key. `fbedge/fixtures.py`
already reads a whole season out of the same payload the xG integration
fetches, and the home page runs on it. Everything here treats it as source
number one rather than reimplementing it, because a second Understat reader
would be a second place for the "unplayed fixtures carry xG of zero, not null"
trap to be fallen into.

*`football_data_org`* is the keyed option. Its free tier covers twelve
competitions - verified on their pricing page: Champions League, the big five
domestic leagues, Eredivisie, Primeira Liga, the Championship, Brazilian Serie
A, the World Cup and the Euros - at **10 calls per minute**, with scores and
schedules delayed. Delay is irrelevant to a calendar. The token comes from an
environment variable and is never written to disk.

*`openfootball`* is the no-key fallback: plain JSON on raw.githubusercontent
.com, community-maintained. Verified live for 2026/27: 380 fixtures for each of
the Premier League, La Liga and Serie A and 306 for each of the Bundesliga and
Ligue 1, which are exactly the right counts.

**Nothing here is ever a hard dependency.** With no token and no network the
project runs exactly as it does today; the calendar is additive. A source that
fails says so and the caller decides, rather than the pipeline degrading into a
half-populated table.

**An unresolved club name is an error, not a row to drop.** A fixture whose
team does not map to a canonical name never joins to anything, which looks
identical to a fixture that was never scheduled. `fetch` raises
`UnresolvedTeams` listing every name it could not place, and the caller has to
decide - which is the only way the alias tables below get extended from what a
feed actually says rather than from imagination.

**Cache aggressively.** A season calendar changes rarely, football-data.org's
free quota is ten calls a minute, and openfootball is somebody else's GitHub
bandwidth. A finished season is cached for ever; one in progress is refetched
after `config.CALENDAR_CACHE_HOURS`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from . import config, fixtures as fixtures_mod, normalize, understat

TABLE_NAME = "calendar"

UNDERSTAT = "understat"
FOOTBALL_DATA_ORG = "football_data_org"
OPENFOOTBALL = "openfootball"
SOURCES = (UNDERSTAT, FOOTBALL_DATA_ORG, OPENFOOTBALL)


class CalendarError(RuntimeError):
    """A source answered, but not with a calendar."""


class UnresolvedTeams(CalendarError):
    """Club names that could not be mapped onto this project's spelling.

    Raised rather than warned, and it carries the names. A calendar with a
    silently unmapped team is worse than no calendar: the fixture is present,
    looks fine, and joins to nothing for the rest of the season.
    """

    def __init__(self, names: list[str], league: str, source: str):
        self.names = sorted(set(names))
        super().__init__(
            f"{source} {league}: {len(self.names)} club name(s) did not resolve "
            f"to a canonical name: {', '.join(self.names)}. Add them to "
            f"calendar.{source.upper()}_ALIASES rather than guessing at "
            "fetch time - a fuzzy match here would eventually merge two clubs."
        )


# --------------------------------------------------------------------------
# openfootball
# --------------------------------------------------------------------------

OPENFOOTBALL_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"

# openfootball's league file names, keyed by football-data.co.uk division code.
OPENFOOTBALL_LEAGUES = {
    "E0": "en.1",
    "SP1": "es.1",
    "I1": "it.1",
    "D1": "de.1",
    "F1": "fr.1",
}

# Names that survive `normalize.external_name_key` and still disagree.
#
# **Every entry here was produced by running the feed and reading what it
# said**, which is the only way this can be trusted, and is the practice
# `injuries.py` established after the Understat integration showed a fuzzy
# matcher would have merged Milan with Inter. Stripping corporate noise gets 81
# of 96 names on its own; these are the fifteen it cannot.
#
# Keys are already in `external_name_key` form - lowercase, unaccented, no
# punctuation, corporate prefixes and suffixes removed - because that is the
# only form the lookup ever sees.
OPENFOOTBALL_ALIASES: dict[str, str] = {
    # England
    "brighton hove albion": "Brighton",
    # Spain
    "club atletico de madrid": "Ath Madrid",
    "deportivo alaves": "Alaves",
    "celta de vigo": "Celta",
    "espanyol de barcelona": "Espanol",
    "rayo vallecano de madrid": "Vallecano",
    "real racing club de santander": "Santander",
    "real sociedad de futbol": "Sociedad",
    # Italy
    "acf fiorentina": "Fiorentina",
    "internazionale milano": "Inter",
    # Germany
    "1 fsv mainz": "Mainz",
    "hamburger": "Hamburg",
    # France
    "es troyes": "Troyes",
    "strasbourg alsace": "Strasbourg",
    "racing club de lens": "Lens",
}


def _openfootball_season(season: int) -> str:
    """openfootball's directory name: 2026 -> '2026-27'."""
    return f"{season}-{(season + 1) % 100:02d}"


def fetch_openfootball(league: str, season: int, cache_dir: Path, force: bool = False):
    slug = OPENFOOTBALL_LEAGUES.get(league)
    if slug is None:
        raise CalendarError(
            f"openfootball has no file mapped for {league!r}. "
            "See calendar.OPENFOOTBALL_LEAGUES."
        )
    url = f"{OPENFOOTBALL_BASE}/{_openfootball_season(season)}/{slug}.json"
    return _cached_json(
        url, Path(cache_dir) / f"openfootball_{league}_{season}.json",
        season, force, headers={"User-Agent": config.USER_AGENT},
    )


def _openfootball_frame(payload: dict, league: str, season: int, known) -> pd.DataFrame:
    rows, unresolved = [], []
    for entry in payload.get("matches", []):
        home = normalize.resolve_external_team(
            entry.get("team1", ""), known, OPENFOOTBALL_ALIASES
        )
        away = normalize.resolve_external_team(
            entry.get("team2", ""), known, OPENFOOTBALL_ALIASES
        )
        if home is None or away is None:
            unresolved += [
                raw for raw, mapped in
                ((entry.get("team1"), home), (entry.get("team2"), away))
                if mapped is None
            ]
            continue
        home_goals, away_goals = _openfootball_score(entry.get("score"))
        rows.append(
            {
                "source": OPENFOOTBALL,
                "external_id": f"{league}_{season}_{entry.get('round')}_"
                               f"{normalize.team_slug(home)}_{normalize.team_slug(away)}",
                "league": league,
                "season_start_year": season,
                "kickoff_utc": _combine(entry.get("date"), entry.get("time")),
                "home_team": home,
                "away_team": away,
                "played": home_goals is not None,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    if unresolved:
        raise UnresolvedTeams(unresolved, league, OPENFOOTBALL)
    return pd.DataFrame(rows)


def _openfootball_score(score):
    """Full-time goals from openfootball's score field, which has three shapes.

    Confirmed on the live 2026/27 files: `None` for an unplayed fixture, a dict
    carrying `ft` and usually `ht`, and - in the Spanish, German and French
    files - a bare list. Reading only the dict shape would have silently marked
    a slice of each of those seasons unplayed, which for a calendar means
    showing finished matches as still to come.
    """
    if score is None:
        return None, None
    if isinstance(score, dict):
        full_time = score.get("ft")
    elif isinstance(score, list):
        full_time = score
    else:
        return None, None
    if not full_time or len(full_time) < 2:
        return None, None
    try:
        return int(full_time[0]), int(full_time[1])
    except (TypeError, ValueError):
        return None, None


# --------------------------------------------------------------------------
# football-data.org
# --------------------------------------------------------------------------

FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"

# football-data.org's own competition codes. Unrelated to the division codes
# used everywhere else here, and kept in one place because a wrong one fails
# silently - a 404 and a competition with no fixtures look different, but a
# wrong-but-valid code returns somebody else's league.
FOOTBALL_DATA_ORG_COMPETITIONS = {
    "E0": "PL",
    "SP1": "PD",
    "I1": "SA",
    "D1": "BL1",
    "F1": "FL1",
}

# **Unverified against a live response.** No token was available when this was
# written, and the endpoint answers 403 without one, so the parsing below
# follows football-data.org's published v4 schema rather than something
# observed. `test_football_data_org_parses_its_documented_shape` pins the shape
# this expects; if the real feed differs, that test is where to correct it.
FOOTBALL_DATA_ORG_ALIASES: dict[str, str] = {
    "brighton hove albion": "Brighton",
    "club atletico de madrid": "Ath Madrid",
    "deportivo alaves": "Alaves",
    "celta de vigo": "Celta",
    "rcd espanyol de barcelona": "Espanol",
    "espanyol de barcelona": "Espanol",
    "rayo vallecano de madrid": "Vallecano",
    "real sociedad de futbol": "Sociedad",
    "acf fiorentina": "Fiorentina",
    "internazionale milano": "Inter",
    "1 fsv mainz": "Mainz",
    "hamburger": "Hamburg",
    "strasbourg alsace": "Strasbourg",
    "racing club de lens": "Lens",
}

# The free tier allows ten calls a minute. Enforced here rather than relied on
# from the server: a 429 spent from a hundred-a-day-adjacent budget is a
# request wasted, and repeated hammering is how access gets withdrawn without
# warning. Module-level because the limit is per key, not per caller.
_LAST_CALLS: list[float] = []


def token() -> str | None:
    """The configured token, or None. Absence is a normal state, not a failure."""
    return os.environ.get(config.CALENDAR_TOKEN_ENV, "").strip() or None


def _respect_rate_limit(max_per_minute: int | None = None) -> None:
    """Block until another call is within the published allowance.

    A loop rather than recursion, and it drops the oldest call rather than
    trusting the sleep to have advanced far enough. The recursive version this
    replaces could not terminate when `time.sleep` did not actually move the
    clock forward - which is exactly what a test that stubs sleep does, and
    also what a suspended laptop does.
    """
    limit = max_per_minute or config.CALENDAR_CALLS_PER_MINUTE
    while True:
        now = time.monotonic()
        _LAST_CALLS[:] = [t for t in _LAST_CALLS if now - t < 60.0]
        if len(_LAST_CALLS) < limit:
            _LAST_CALLS.append(now)
            return
        wait = max(0.0, 60.0 - (now - _LAST_CALLS[0])) + 0.1
        time.sleep(wait)
        # Retire the call we waited out, so a stubbed or short sleep cannot
        # spin here for ever.
        _LAST_CALLS.pop(0)


def fetch_football_data_org(
    league: str, season: int, cache_dir: Path, force: bool = False
):
    competition = FOOTBALL_DATA_ORG_COMPETITIONS.get(league)
    if competition is None:
        raise CalendarError(
            f"No football-data.org competition code for {league!r}. "
            "See calendar.FOOTBALL_DATA_ORG_COMPETITIONS."
        )
    path = Path(cache_dir) / f"football_data_org_{league}_{season}.json"
    if not force and not _stale(path, season):
        return json.loads(path.read_text(encoding="utf-8"))

    key = token()
    if key is None:
        raise CalendarError(
            f"No football-data.org token. Set {config.CALENDAR_TOKEN_ENV} to a "
            "free key from https://www.football-data.org/, or set "
            f"config.CALENDAR_SOURCE to {OPENFOOTBALL!r}, which needs none."
        )
    _respect_rate_limit()
    url = f"{FOOTBALL_DATA_ORG_BASE}/competitions/{competition}/matches?season={season}"
    request = urllib.request.Request(
        url, headers={"X-Auth-Token": key, "User-Agent": config.USER_AGENT}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise CalendarError(
                f"football-data.org rejected the token (HTTP {error.code}). The "
                "free tier covers twelve competitions; a league outside it "
                "answers 403 with a valid key."
            ) from error
        if error.code == 429:
            raise CalendarError(
                "football-data.org rate limit hit (HTTP 429) despite the "
                "client-side throttle. Something else is using the same key."
            ) from error
        raise CalendarError(f"{url} returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise CalendarError(f"{url} could not be reached: {error.reason}") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _football_data_org_frame(payload: dict, league: str, season: int, known):
    rows, unresolved = [], []
    for entry in payload.get("matches", []):
        home_raw = (entry.get("homeTeam") or {}).get("name")
        away_raw = (entry.get("awayTeam") or {}).get("name")
        home = normalize.resolve_external_team(
            home_raw or "", known, FOOTBALL_DATA_ORG_ALIASES
        )
        away = normalize.resolve_external_team(
            away_raw or "", known, FOOTBALL_DATA_ORG_ALIASES
        )
        if home is None or away is None:
            unresolved += [
                raw for raw, mapped in ((home_raw, home), (away_raw, away))
                if mapped is None
            ]
            continue
        full_time = ((entry.get("score") or {}).get("fullTime") or {})
        home_goals, away_goals = full_time.get("home"), full_time.get("away")
        rows.append(
            {
                "source": FOOTBALL_DATA_ORG,
                "external_id": str(entry.get("id")),
                "league": league,
                "season_start_year": season,
                "kickoff_utc": _parse_utc(entry.get("utcDate")),
                "home_team": home,
                "away_team": away,
                "played": home_goals is not None,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    if unresolved:
        raise UnresolvedTeams(unresolved, league, FOOTBALL_DATA_ORG)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Understat, through the reader that already exists
# --------------------------------------------------------------------------

def fetch_understat(league: str, season: int, cache_dir: Path, force: bool = False):
    return understat.fetch_season(league, season, Path(cache_dir), force=force)


def _understat_frame(payload: dict, league: str, season: int, known):
    """Delegates to `fixtures.fixture_frame`, which already gets this right.

    Its one subtlety is worth restating because it is the reason this is not
    reimplemented: Understat gives an unplayed fixture an xG of *zero* rather
    than a missing one, and `fixture_frame` is careful to write NULL. A second
    reader would eventually get that wrong and quietly tell the model twenty
    teams failed to have a shot.
    """
    frame = fixtures_mod.fixture_frame(payload, league, season)
    if frame.empty:
        return frame
    unresolved = [
        raw for raw, mapped in
        zip(
            list(frame["home_team_understat"]) + list(frame["away_team_understat"]),
            list(frame["home_team"]) + list(frame["away_team"]),
        )
        if mapped not in known
    ]
    if unresolved:
        raise UnresolvedTeams(unresolved, league, UNDERSTAT)
    out = frame[
        ["league", "season_start_year", "kickoff_utc", "home_team", "away_team",
         "played", "home_goals", "away_goals"]
    ].copy()
    out.insert(0, "source", UNDERSTAT)
    out.insert(1, "external_id", frame["understat_id"])
    return out


# --------------------------------------------------------------------------
# One interface
# --------------------------------------------------------------------------

_FETCHERS = {
    UNDERSTAT: (fetch_understat, _understat_frame),
    FOOTBALL_DATA_ORG: (fetch_football_data_org, _football_data_org_frame),
    OPENFOOTBALL: (fetch_openfootball, _openfootball_frame),
}

CALENDAR_COLUMNS = [
    "source", "external_id", "league", "season_start_year", "kickoff_utc",
    "home_team", "away_team", "played", "home_goals", "away_goals",
]


def resolve_source(source: str | None = None) -> str:
    """Which source to use, honouring `config.CALENDAR_SOURCE` and "auto".

    "auto" means the keyed source when a token exists and the free one
    otherwise, so a machine with no key does something sensible rather than
    failing on a configuration it never set.
    """
    choice = source or config.CALENDAR_SOURCE
    if choice == "auto":
        return FOOTBALL_DATA_ORG if token() else OPENFOOTBALL
    if choice not in SOURCES:
        raise CalendarError(
            f"Unknown calendar source {choice!r}. Known: {', '.join(SOURCES)}."
        )
    return choice


def fetch(
    leagues: list[str],
    season: int,
    known_teams: dict[str, set[str]],
    source: str | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """One season's calendar for several leagues, from one source.

    `known_teams` maps a league code to the canonical names that league already
    has in the database. Passed in rather than read here so this module never
    needs a connection, and so a test can supply its own.
    """
    chosen = resolve_source(source)
    fetcher, parser = _FETCHERS[chosen]
    cache = Path(cache_dir) if cache_dir else config.RAW_DIR / "calendar"
    cache.mkdir(parents=True, exist_ok=True)

    frames = []
    for league in leagues:
        payload = fetcher(league, season, cache, force=force)
        frame = parser(payload, league, season, known_teams.get(league, set()))
        if frame.empty:
            raise CalendarError(
                f"{chosen} {league} {season}: the payload held no fixtures. That "
                "is never right for a league in progress; check the feed before "
                "writing anything."
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined[CALENDAR_COLUMNS].sort_values("kickoff_utc").reset_index(drop=True)


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------

def _stale(path: Path, season: int, now: dt.datetime | None = None) -> bool:
    """A finished season's calendar cannot change; an in-progress one can."""
    if not path.exists() or path.stat().st_size == 0:
        return True
    if season < fixtures_mod.current_season():
        return False
    age = (now or dt.datetime.now()) - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > dt.timedelta(hours=config.CALENDAR_CACHE_HOURS)


def _cached_json(url: str, path: Path, season: int, force: bool, headers: dict):
    if not force and not _stale(path, season):
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(
            request, timeout=config.REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise CalendarError(f"{url} returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise CalendarError(f"{url} could not be reached: {error.reason}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalendarError(f"{url} did not return JSON.") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(config.REQUEST_DELAY_SECONDS)
    return payload


def _combine(date_text, time_text):
    """openfootball splits date and time; a missing time is not a missing date.

    A fixture whose kick-off has not been set yet still belongs on the
    calendar, at midnight, rather than being dropped for lacking a detail
    nobody has decided.
    """
    if not date_text:
        return pd.NaT
    stamp = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(stamp) or not time_text:
        return stamp
    parsed = pd.to_datetime(f"{date_text} {time_text}", errors="coerce")
    return stamp if pd.isna(parsed) else parsed


def _parse_utc(value):
    if not value:
        return pd.NaT
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.NaT if pd.isna(stamp) else stamp.tz_localize(None)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def create_table(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            source             VARCHAR NOT NULL,
            external_id        VARCHAR,
            league             VARCHAR NOT NULL,
            season_start_year  INTEGER NOT NULL,
            kickoff_utc        TIMESTAMP,
            home_team          VARCHAR NOT NULL,
            away_team          VARCHAR NOT NULL,
            played             BOOLEAN,
            home_goals         INTEGER,
            away_goals         INTEGER
        )
        """
    )


def write(con, frame: pd.DataFrame) -> int:
    """Replace the calendar for the (source, league, season) triples supplied.

    Scoped by source as well as by league, so two sources can coexist and be
    compared - which is the point of storing provenance at all - and so
    refreshing one never silently empties the other.

    Replace rather than append, because a result arrives by *changing* a row
    from unplayed to played. An insert-only load would keep the stale row and
    show a finished match as still to come.
    """
    create_table(con)
    if frame.empty:
        return 0
    con.register("incoming_calendar", frame[CALENDAR_COLUMNS])
    con.execute(
        f"""
        DELETE FROM {TABLE_NAME} WHERE (source, league, season_start_year) IN
        (SELECT DISTINCT source, league, season_start_year FROM incoming_calendar)
        """
    )
    columns = ", ".join(f'"{c}"' for c in CALENDAR_COLUMNS)
    con.execute(
        f"INSERT INTO {TABLE_NAME} ({columns}) SELECT {columns} FROM incoming_calendar"
    )
    con.unregister("incoming_calendar")
    return len(frame)


def load(
    con,
    leagues: list[str] | None = None,
    season: int | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME]
    ).fetchone()
    if not exists:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    clauses, params = [], []
    if leagues:
        clauses.append(f"league IN ({','.join('?' for _ in leagues)})")
        params.extend(leagues)
    if season is not None:
        clauses.append("season_start_year = ?")
        params.append(season)
    if source:
        clauses.append("source = ?")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return con.execute(
        f"SELECT * FROM {TABLE_NAME} {where} ORDER BY kickoff_utc", params
    ).df()
