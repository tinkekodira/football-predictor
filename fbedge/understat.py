"""Expected goals from Understat.

football-data.co.uk carries results and prices but no shot data, and goals are
a noisy realisation of how a match actually went: a team that creates six good
chances and scores once is stronger than the scoreline says. Expected goals sum
the quality of the chances instead of counting the ones that went in, so team
strength estimated from xG settles down over far fewer matches than strength
estimated from goals. That is the whole reason for this module.

**The site changed and most published scrapers are broken.** Every tutorial and
the `understat` PyPI package look for `var datesData = JSON.parse('...')`
embedded in the league page's HTML. Understat now renders that client-side, so
the HTML no longer contains it and those scrapers return nothing - not an
error, just an empty result, which is the dangerous kind of failure. The data
comes from `getLeagueData/<league>/<season>`, which `js/league.min.js` calls
over AJAX and which returns gzipped JSON. That endpoint is what this module
uses, and `test_understat.py` pins the shape it returns so a future change
fails loudly rather than silently.

**Everything is cached to disk.** Understat is a free service run by someone
else and this project has no claim on it: a backfill of five leagues and nine
seasons is 45 requests, and it should happen once. Re-running any script here
reads the cache.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://understat.com"

# Understat's league slugs, keyed by the football-data.co.uk codes this project
# already uses everywhere else.
LEAGUE_SLUGS = {
    "E0": "EPL",
    "SP1": "La_liga",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}

# Understat's earliest season.
FIRST_SEASON = 2014

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/",
    "Accept-Encoding": "gzip",
}


# Understat's team names translated into football-data.co.uk's, which is the
# vocabulary the database and every model already speak. Derived by diffing the
# two sources across all five leagues and every season from 2017, not guessed:
# 41 of the 156 team names differ, and the differences are not systematic
# enough to normalise with a rule. "Milan" means AC Milan here while "Inter"
# matches exactly, so a fuzzy matcher is a liability rather than a convenience.
TEAM_ALIASES = {
    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
    # Bundesliga
    "Arminia Bielefeld": "Bielefeld",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia M.Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln",
    "FC Heidenheim": "Heidenheim",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "Greuther Fuerth": "Greuther Furth",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "Hertha Berlin": "Hertha",
    "Mainz 05": "Mainz",
    "Nuernberg": "Nurnberg",
    "RasenBallsport Leipzig": "RB Leipzig",
    "St. Pauli": "St Pauli",
    "VfB Stuttgart": "Stuttgart",
    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "Deportivo La Coruna": "La Coruna",
    "Espanyol": "Espanol",
    "Racing Santander": "Santander",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "SD Huesca": "Huesca",
    # Serie A
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",
    "SPAL 2013": "Spal",
    # Ligue 1
    "Clermont Foot": "Clermont",
    "Paris Saint Germain": "Paris SG",
    "Saint-Etienne": "St Etienne",
}


class UnderstatError(RuntimeError):
    """Raised when the endpoint answers with something unusable."""


def to_football_data_name(name: str) -> str:
    """Understat's name for a team, in football-data.co.uk's vocabulary.

    Unknown names pass through unchanged rather than raising, because a team
    Understat carries and football-data does not is a normal occurrence - the
    two sources disagree at the edges about promoted sides and about which
    competitions count. Those rows simply fail to join, and
    `scripts/build_xg.py` reports the join rate so a mapping that has rotted is
    visible as a drop in coverage rather than as silence.
    """
    return TEAM_ALIASES.get(name, name)


def fetch_season(
    league: str,
    season: int,
    cache_dir: Path,
    force: bool = False,
    pause: float = 1.0,
) -> dict:
    """One league-season of Understat data, from cache when possible.

    `season` is the starting calendar year, matching the project's
    `season_start_year` convention rather than Understat's own labelling.

    Raises:
        UnderstatError: on a non-JSON answer or a payload missing the keys the
            rest of this module needs. Failing here is deliberate: a silently
            empty result would flow into the models as "no xG for this season"
            and quietly weaken every rating fitted from it.
    """
    if league not in LEAGUE_SLUGS:
        raise ValueError(f"Unknown league {league!r}; expected one of {sorted(LEAGUE_SLUGS)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{league}_{season}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/getLeagueData/{LEAGUE_SLUGS[league]}/{season}"
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        raise UnderstatError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise UnderstatError(f"{url} could not be reached: {exc.reason}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnderstatError(
            f"{url} did not return JSON. The endpoint has probably changed; "
            "see this module's docstring for how it was found."
        ) from exc

    missing = {"dates", "teams"} - set(payload)
    if missing:
        raise UnderstatError(
            f"{url} returned JSON without {sorted(missing)}. Shape has changed."
        )

    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(pause)  # only reached on a real request, never on a cache hit
    return payload


def match_frame(payload: dict, league: str, season: int) -> pd.DataFrame:
    """Match-level xG, one row per fixture.

    Unplayed fixtures are dropped: Understat lists the whole season including
    fixtures not yet played, and those carry a null scoreline with an xG of
    zero rather than a missing one. Keeping them would feed real zeros into
    the ratings.
    """
    rows = []
    for entry in payload.get("dates", []):
        if not entry.get("isResult"):
            continue
        rows.append(
            {
                "league": league,
                "season_start_year": season,
                "date": pd.to_datetime(entry["datetime"]).date(),
                "understat_id": entry.get("id"),
                "home_team": to_football_data_name(entry["h"]["title"]),
                "away_team": to_football_data_name(entry["a"]["title"]),
                "home_team_understat": entry["h"]["title"],
                "away_team_understat": entry["a"]["title"],
                "home_goals": _to_int(entry["goals"]["h"]),
                "away_goals": _to_int(entry["goals"]["a"]),
                "home_xg": _to_float(entry["xG"]["h"]),
                "away_xg": _to_float(entry["xG"]["a"]),
            }
        )
    return pd.DataFrame(rows)


def npxg_frame(payload: dict, league: str, season: int) -> pd.DataFrame:
    """Non-penalty xG per team per match, from the per-team history blocks.

    Worth the extra join. A penalty is worth about 0.76 xG on its own, so a
    single spot-kick moves a team's xG for the match more than a good open-play
    chance does, and whether a team won a penalty says much less about its
    strength than the rest of its play. Ratings fitted on non-penalty xG are
    the cleaner measure; the penalty component is better handled as its own
    small, roughly constant rate.
    """
    rows = []
    for team in payload.get("teams", {}).values():
        title = team.get("title")
        for match in team.get("history", []):
            rows.append(
                {
                    "league": league,
                    "season_start_year": season,
                    "date": pd.to_datetime(match["date"]).date(),
                    "team": to_football_data_name(title),
                    "team_understat": title,
                    "is_home": match.get("h_a") == "h",
                    "xg": _to_float(match.get("xG")),
                    "npxg": _to_float(match.get("npxG")),
                    "xg_against": _to_float(match.get("xGA")),
                    "npxg_against": _to_float(match.get("npxGA")),
                    "deep": _to_float(match.get("deep")),
                    "scored": _to_int(match.get("scored")),
                }
            )
    return pd.DataFrame(rows)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def season_range(first: int = 2017, last: int | None = None) -> list[int]:
    """Seasons to pull, defaulting to the span this project's database covers."""
    if last is None:
        today = dt.date.today()
        last = today.year if today.month >= 8 else today.year - 1
    return list(range(max(first, FIRST_SEASON), last + 1))


# --------------------------------------------------------------------------
# Per-match rosters
# --------------------------------------------------------------------------
#
# **Read this before using anything below.**
#
# `getMatchData/<id>` returns who actually played, with minutes. That is
# after-the-fact information: confirmed line-ups are published around an hour
# before kick-off, which is long after the opening price this project bets into
# and at or after the closing line it measures against. Feeding "who played"
# into a model that prices a match would be lookahead bias, and because
# availability genuinely matters it would not look like a bug - it would look
# like a large and convincing edge.
#
# So nothing here may be used as a feature of the match it describes. It is
# raw material for features about *earlier* matches: who was missing last week,
# who is a booking away from a ban. `fbedge/availability.py` builds those and
# is where the point-in-time rule is enforced and tested.

MATCH_CACHE_DIRNAME = "matches"


def fetch_match(
    understat_id: str,
    cache_dir: Path,
    force: bool = False,
    pause: float = 0.6,
) -> dict:
    """One match's roster and shot data, from cache when possible.

    Cached one file per match rather than one per season, because a backfill is
    thousands of requests against somebody else's free service and will be
    interrupted. Per-match files make a resumed run skip everything already
    fetched instead of starting the season again.
    """
    cache_dir = Path(cache_dir) / MATCH_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{understat_id}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/getMatchData/{understat_id}"
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.HTTPError as exc:
        raise UnderstatError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise UnderstatError(f"{url} could not be reached: {exc.reason}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnderstatError(f"{url} did not return JSON.") from exc
    if "rosters" not in payload:
        raise UnderstatError(f"{url} returned JSON without 'rosters'.")

    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(pause)
    return payload


def roster_frame(payload: dict, match_id: str) -> pd.DataFrame:
    """One row per player who appeared, for one match.

    `started` comes from the position label rather than from minutes: Understat
    writes a real position for a player in the starting eleven and the literal
    string "Sub" for anyone who came on. Minutes cannot stand in for it, since
    a starter withdrawn after twenty minutes and a substitute who played
    seventy are not distinguished by time alone.
    """
    rows = []
    for side, entries in payload.get("rosters", {}).items():
        for entry in entries.values():
            position = entry.get("position") or ""
            rows.append(
                {
                    "match_id": match_id,
                    "is_home": side == "h",
                    "player_id": str(entry.get("player_id")),
                    "player": _unescape(entry.get("player") or ""),
                    "position": position,
                    "started": position != "Sub",
                    "minutes": _to_int(entry.get("time")) or 0,
                    "yellow_card": _to_int(entry.get("yellow_card")) or 0,
                    "red_card": _to_int(entry.get("red_card")) or 0,
                    "xg": _to_float(entry.get("xG")),
                    "xa": _to_float(entry.get("xA")),
                    "xgchain": _to_float(entry.get("xGChain")),
                }
            )
    return pd.DataFrame(rows)


def _unescape(name: str) -> str:
    """Understat HTML-escapes apostrophes: "Dara O&#039;Shea"."""
    import html

    return html.unescape(name)
