"""Real injury news, from a keyed external feed.

Every other source in this project is a public file or a public JSON endpoint.
Injuries are not: nobody publishes them free and unauthenticated, so this is
the one module that needs an API key, and the one that has to behave sensibly
when there isn't one.

**Why API-Football.** Its `/injuries` endpoint takes a league and a season and
returns everyone currently unavailable in it, so **five requests refresh all
five leagues**. The free plan allows a hundred a day, which leaves an order of
magnitude of headroom. The alternative considered, Big Balls Sports Data,
offers a larger quota but is addressed per player id, which would mean building
and maintaining a player-id mapping before a single injury could be read.

**Nothing here guesses a team name.** The Understat integration learned this
the expensive way: 41 of 156 names differed between two sources, the
differences were not systematic, and a fuzzy matcher would have silently merged
Milan with Inter. So matching is exact, then exact again after a documented
normalisation, and then an explicit alias - and anything still unmatched is
**reported loudly and dropped**, never approximated. `scripts/build_injuries.py`
prints the unmatched names so the alias table can be extended deliberately.

**What this can and cannot tell you.** It is genuine injury news - a player,
what is wrong, and whether the feed rates them out or doubtful. It is a
third-party feed, so it is exactly as current and as correct as its provider,
which is why `fetched_at` travels with every row and the page shows it. It is
also not a betting edge: knowing who is injured is not the same as knowing
something the market does not, and the market has the same feeds.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TABLE_NAME = "injuries"

CACHE_DIRNAME = "injuries"

# The feed's own words for how badly a player is unavailable. Carried through
# rather than collapsed to a boolean, because "doubtful" and "ruled out" are
# genuinely different claims and flattening them would be inventing certainty.
OUT_TYPES = {"missing fixture"}
DOUBTFUL_TYPES = {"questionable"}


class InjuryFeedError(RuntimeError):
    """The feed answered, but not with what it promised.

    Raised rather than returning empty, for the reason this project keeps
    relearning: an empty injury list and a broken injury feed look identical
    on a page, and only one of them is news.
    """


class MissingApiKey(InjuryFeedError):
    """No key configured. Distinct from a failure, because it is not one."""


# ----------------------------------------------------------------------
# Team names
# ----------------------------------------------------------------------

# Keys are already in `normalise` form, because that is the only form the
# lookup ever sees - a key like 'fc barcelona' normalises to 'barcelona'
# and could never be hit. `test_the_alias_table_is_injective` pins this,
# and it caught exactly that mistake in twelve entries when first written.
#
# Entries that merely restate a club's own name are kept on purpose: the
# normalised comparison needs the club to be in the database already, and
# a newly promoted side is not.
#
# Names that survive normalisation but still disagree. Deliberately short and
# deliberately hand-written: every entry here is a decision somebody made after
# seeing both names, which is the only way this can be trusted. Extend it from
# what `scripts/build_injuries.py` reports rather than from imagination.
TEAM_ALIASES: dict[str, str] = {
    # England
    'manchester united': 'Man United',
    'manchester city': 'Man City',
    'newcastle united': 'Newcastle',
    'wolverhampton wanderers': 'Wolves',
    'nottingham forest': "Nott'm Forest",
    'tottenham hotspur': 'Tottenham',
    'west ham united': 'West Ham',
    'leicester city': 'Leicester',
    'leeds united': 'Leeds',
    'norwich city': 'Norwich',
    'brighton hove albion': 'Brighton',
    'sheffield united': 'Sheffield United',
    'west bromwich albion': 'West Brom',
    'queens park rangers': 'QPR',
    # Spain
    'atletico madrid': 'Ath Madrid',
    'athletic club': 'Ath Bilbao',
    'real sociedad': 'Sociedad',
    'celta vigo': 'Celta',
    'deportivo alaves': 'Alaves',
    'rayo vallecano': 'Vallecano',
    'real valladolid': 'Valladolid',
    'barcelona': 'Barcelona',
    'espanyol': 'Espanol',
    'real betis': 'Betis',
    # Italy
    'milan': 'Milan',
    'inter': 'Inter',
    'internazionale': 'Inter',
    'roma': 'Roma',
    'lazio': 'Lazio',
    'hellas verona': 'Verona',
    # Germany
    'bayern munchen': 'Bayern Munich',
    'borussia dortmund': 'Dortmund',
    'borussia monchengladbach': "M'gladbach",
    'bayer leverkusen': 'Leverkusen',
    'eintracht frankfurt': 'Ein Frankfurt',
    '1899 hoffenheim': 'Hoffenheim',
    'stuttgart': 'Stuttgart',
    'wolfsburg': 'Wolfsburg',
    'freiburg': 'Freiburg',
    '1 fc koln': 'FC Koln',
    'koln': 'FC Koln',
    'werder bremen': 'Werder Bremen',
    'augsburg': 'Augsburg',
    'mainz 05': 'Mainz',
    'union berlin': 'Union Berlin',
    'rb leipzig': 'RB Leipzig',
    # France
    'paris saint germain': 'Paris SG',
    'olympique lyonnais': 'Lyon',
    'olympique marseille': 'Marseille',
    'stade rennais': 'Rennes',
    'monaco': 'Monaco',
    'lille': 'Lille',
    'ogc nice': 'Nice',
    'lens': 'Lens',
    'stade brestois 29': 'Brest',
    'nantes': 'Nantes',
    'montpellier': 'Montpellier',
    'toulouse': 'Toulouse',
    'strasbourg': 'Strasbourg',
}

# Corporate and legal noise that carries no information about which club is
# meant. Stripped before matching so that "FC Augsburg" and "Augsburg" are one
# club rather than two. Ordered longest-first so "1. FC" goes before "FC".
_DROPPABLE = (
    "football club", "fussball club", "association", "calcio",
    "1. fc", "1.fc", "afc", "fsv", "tsg", "vfl", "vfb", "sv", "sc", "fc",
    "cf", "ac", "as", "ss", "us", "rc", "sd", "cd", "ud", "cp",
)


def normalise(name: str) -> str:
    """A comparable form of a club name.

    Accents removed, punctuation dropped, corporate prefixes stripped, spacing
    collapsed. The point is only to make two spellings of one club compare
    equal; it is never used to decide that two *different* strings are the same
    club, which is what a similarity score would do.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    words = text.split()
    # Strip the noise words only from the ends, so a club actually called
    # "Milan" is not eaten by the "ac" rule applied in the middle.
    while words and words[0] in _DROPPABLE:
        words = words[1:]
    while words and words[-1] in _DROPPABLE:
        words = words[:-1]
    return " ".join(words)


def to_football_data_name(name: str, known: set[str] | None = None) -> str | None:
    """Map a feed team name onto the project's convention, or return None.

    Three attempts, in decreasing confidence: the name as given, an explicit
    alias, then a normalised comparison against the names we actually know.
    **No fuzzy fallback.** Returning None is a real answer here - the caller
    reports it and drops the row, which is the honest outcome for a club we
    cannot identify.
    """
    known = known or set()
    if name in known:
        return name

    key = normalise(name)
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]

    for candidate in known:
        if normalise(candidate) == key:
            return candidate
    return None


# ----------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------


def api_key() -> str | None:
    """The configured key, or None. Never raises: absence is a normal state."""
    value = os.environ.get(config.INJURY_API_KEY_ENV, "").strip()
    return value or None


# ----------------------------------------------------------------------
# The daily budget
# ----------------------------------------------------------------------
# The free plan allows a hundred requests a day, resetting at 00:00 UTC, and
# ten a minute. Both are enforced here, before the request goes out.
#
# **Why this is a file and not a counter.** The limit is per key per day, and
# this project is a collection of short-lived scripts; an in-memory counter
# resets on every invocation and therefore counts nothing. Two loops in two
# terminals would each believe they had a hundred calls in hand.
#
# **Why a hard stop rather than a warning.** A spent budget is not a loud
# failure at this endpoint - it answers 429 sometimes and 200-with-an-empty-
# list other times - and an empty injury list and a broken injury feed look
# identical on a page. Refusing to make the call is the only way the two stay
# distinguishable. It is also the only protection against the failure that
# actually costs something: repeated hammering gets access withdrawn without
# warning, and this is the project's one keyed source.


class QuotaExhausted(InjuryFeedError):
    """The day's request budget is spent. Not a failure of the feed."""


_MINUTE_CALLS: list[float] = []


def _quota_path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else config.INJURY_QUOTA_PATH


def _utc_day(now: dt.datetime | None = None) -> str:
    """The budget day. UTC because that is when the provider resets it."""
    moment = now or dt.datetime.now(dt.timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc).date().isoformat()


def quota_state(path: Path | None = None, now: dt.datetime | None = None) -> dict:
    """Today's spend. A new day starts at zero rather than carrying over.

    Unused requests are lost at 00:00 UTC - the provider says so - so nothing
    accumulates and a stale file from last week reads as an unspent budget,
    which is correct.
    """
    today = _utc_day(now)
    file = _quota_path(path)
    used = 0
    if file.exists():
        try:
            stored = json.loads(file.read_text(encoding="utf-8"))
            if stored.get("day") == today:
                used = int(stored.get("used", 0))
        except (json.JSONDecodeError, ValueError, OSError):
            # A corrupt counter must not be read as "budget available". Zero
            # is the wrong guess in exactly the dangerous direction, so treat
            # it as spent-so-far-unknown and start from the limit.
            used = config.INJURY_DAILY_QUOTA
    limit = config.INJURY_DAILY_QUOTA - config.INJURY_QUOTA_RESERVE
    return {
        "day": today,
        "used": used,
        "limit": limit,
        "published_limit": config.INJURY_DAILY_QUOTA,
        "remaining": max(0, limit - used),
    }


def record_call(path: Path | None = None, now: dt.datetime | None = None) -> dict:
    """Count one request against today's budget, and persist it."""
    state = quota_state(path, now)
    file = _quota_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        json.dumps({"day": state["day"], "used": state["used"] + 1}),
        encoding="utf-8",
    )
    return quota_state(path, now)


def check_quota(path: Path | None = None, now: dt.datetime | None = None) -> dict:
    """Raise `QuotaExhausted` if there is no budget left. Silent if there is."""
    state = quota_state(path, now)
    if state["remaining"] <= 0:
        raise QuotaExhausted(
            f"The injury feed's daily budget is spent: {state['used']} requests "
            f"used today against a self-imposed ceiling of {state['limit']} "
            f"(the plan allows {state['published_limit']}, and "
            f"{config.INJURY_QUOTA_RESERVE} are held back so a spent budget "
            "can still be diagnosed). It resets at 00:00 UTC. Nothing was "
            f"requested. The counter is {_quota_path(path)}; delete it only if "
            "you know the spend was not real."
        )
    return state


def _respect_minute_limit(limit: int | None = None) -> None:
    """Ten a minute, enforced locally so the eleventh is never wasted."""
    ceiling = limit or config.INJURY_CALLS_PER_MINUTE
    while True:
        now = time.monotonic()
        _MINUTE_CALLS[:] = [t for t in _MINUTE_CALLS if now - t < 60.0]
        if len(_MINUTE_CALLS) < ceiling:
            _MINUTE_CALLS.append(now)
            return
        time.sleep(max(0.0, 60.0 - (now - _MINUTE_CALLS[0])) + 0.1)
        # Retire the call we waited out, so a stubbed or short sleep cannot
        # spin here for ever.
        _MINUTE_CALLS.pop(0)


def _cache_path(cache_dir: Path, league: str, season: int) -> Path:
    return Path(cache_dir) / f"{league}_{season}.json"


def _is_stale(path: Path, max_age_hours: float, now: dt.datetime | None = None) -> bool:
    if not path.exists():
        return True
    # An infinite budget means "any cache will do", which is how a caller with
    # no API key asks to work from whatever is already on disk. `timedelta`
    # cannot represent it, so answer before constructing one.
    if not np.isfinite(max_age_hours):
        return False
    age = (now or dt.datetime.now()) - dt.datetime.fromtimestamp(path.stat().st_mtime)
    return age > dt.timedelta(hours=max_age_hours)


def fetch_league(
    league: str,
    season: int,
    cache_dir: Path,
    max_age_hours: float = config.INJURY_CACHE_HOURS,
    force: bool = False,
    key: str | None = None,
    now: dt.datetime | None = None,
    quota_path: Path | None = None,
) -> dict:
    """One league-season of injuries, from cache when it is fresh enough.

    **Caching is not politeness here, it is the budget.** The free plan allows
    a hundred requests a day and the page re-renders on every click, so a cache
    miss on a page load would spend the day's allowance in a few minutes of
    browsing. One call covers a whole league-season - never call this from
    inside a per-match loop, which is the shape that turns five requests into
    three hundred and eighty.

    A cache hit costs nothing and is not counted. A miss is checked against the
    persistent daily counter and the per-minute limit *before* the request goes
    out, and refused outright when the budget is gone rather than being allowed
    to fail at the server.
    """
    league_id = config.INJURY_LEAGUE_IDS.get(league)
    if league_id is None:
        raise InjuryFeedError(
            f"No API-Football league id for {league!r}. "
            "See config.INJURY_LEAGUE_IDS."
        )

    path = _cache_path(cache_dir, league, season)
    if not force and not _is_stale(path, max_age_hours, now):
        return json.loads(path.read_text(encoding="utf-8"))

    key = key or api_key()
    if key is None:
        raise MissingApiKey(
            f"No injury feed key. Set {config.INJURY_API_KEY_ENV} to a free "
            "API-Football key from https://www.api-football.com/."
        )

    # Both limits, before the socket is opened. A request refused here costs
    # nothing; one refused by the server costs a request from a hundred.
    check_quota(quota_path)
    _respect_minute_limit()

    url = f"{config.INJURY_API_URL}?league={league_id}&season={season}"
    request = urllib.request.Request(
        url,
        headers={
            "x-apisports-key": key,
            "User-Agent": config.USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise InjuryFeedError(
                f"The injury feed rejected the key (HTTP {error.code}). Check "
                f"{config.INJURY_API_KEY_ENV}."
            ) from error
        if error.code == 429:
            raise InjuryFeedError(
                "The injury feed daily quota is spent (HTTP 429). The free "
                "plan allows 100 requests a day and resets at 00:00 UTC."
            ) from error
        raise InjuryFeedError(f"{url} returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise InjuryFeedError(f"{url} could not be reached: {error.reason}") from error

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InjuryFeedError(f"{url} did not return JSON.") from error

    # The API answers 200 with an `errors` block rather than an HTTP error for
    # things like an unknown parameter, so a 200 is not by itself success.
    errors = payload.get("errors")
    if errors:
        raise InjuryFeedError(f"The injury feed reported: {errors}")
    if "response" not in payload:
        raise InjuryFeedError(
            f"{url} returned JSON with no `response` key. The shape has "
            "changed; see this module's docstring."
        )

    # A league-season is thousands of rows - 3,168 for the Premier League in
    # 2024 - and it arrives on one page today. If that ever stops being true,
    # reading only page one would drop most of a league without a word, which
    # is the exact failure this project keeps guarding against. Refuse instead.
    paging = payload.get("paging") or {}
    total_pages = int(paging.get("total") or 1)
    if total_pages > 1:
        raise InjuryFeedError(
            f"{url} came back in {total_pages} pages and this reader only "
            "fetches one, so most of the league would be missing. Add paging "
            "before trusting anything from it."
        )

    # Counted after the response comes back, not before: a request that never
    # reached the provider did not spend anything, and over-counting would
    # eventually refuse work the budget could have covered.
    record_call(quota_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(config.REQUEST_DELAY_SECONDS)
    return payload


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def injury_frame(
    payload: dict,
    league: str,
    season: int,
    known_teams: set[str] | None = None,
    fetched_at: dt.datetime | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalise a payload into rows, and report the names that did not map.

    Returns `(frame, unmatched)`. The second element is not an afterthought: a
    club that fails to map vanishes from the page silently otherwise, and a
    quietly missing team looks exactly like a team with nobody injured.
    """
    stamp = fetched_at or dt.datetime.now()
    rows, unmatched = [], []
    for item in payload.get("response", []):
        player = item.get("player") or {}
        team = item.get("team") or {}
        fixture = item.get("fixture") or {}

        raw_name = team.get("name")
        mapped = to_football_data_name(raw_name, known_teams) if raw_name else None
        if mapped is None:
            if raw_name:
                unmatched.append(raw_name)
            continue

        kind = str(player.get("type") or "").strip()
        rows.append(
            {
                "league": league,
                "season_start_year": season,
                "team": mapped,
                "team_feed": raw_name,
                "player": player.get("name"),
                "status": kind,
                "reason": player.get("reason"),
                "ruled_out": kind.lower() in OUT_TYPES,
                "doubtful": kind.lower() in DOUBTFUL_TYPES,
                "fixture_id": fixture.get("id"),
                "fixture_date": _fixture_date(fixture),
                "fetched_at": stamp,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # One row per player per fixture is what the feed sends; a player can
        # appear against several upcoming fixtures. De-duplicated on the pair
        # so a page listing "who is out" does not repeat a name.
        frame = frame.drop_duplicates(subset=["team", "player", "fixture_date"])
    return frame, sorted(set(unmatched))


def _fixture_date(fixture: dict):
    value = fixture.get("date")
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        return None


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------


def create_table(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            league            VARCHAR NOT NULL,
            season_start_year INTEGER NOT NULL,
            team              VARCHAR NOT NULL,
            team_feed         VARCHAR,
            player            VARCHAR,
            status            VARCHAR,
            reason            VARCHAR,
            ruled_out         BOOLEAN,
            doubtful          BOOLEAN,
            fixture_id        BIGINT,
            fixture_date      DATE,
            fetched_at        TIMESTAMP
        )
        """
    )


def write_injuries(con, frame: pd.DataFrame) -> int:
    """Replace the stored injuries for the leagues and seasons supplied.

    Replace, because an injury ends: a player who has recovered is absent from
    the new payload rather than marked fit in it, so anything additive would
    keep him on the page for ever.
    """
    create_table(con)
    if frame.empty:
        return 0
    seasons = sorted({int(v) for v in frame["season_start_year"].unique()})
    leagues = sorted({str(v) for v in frame["league"].unique()})
    con.register("incoming_injuries", frame)
    con.execute(
        f"DELETE FROM {TABLE_NAME} WHERE season_start_year IN "
        f"({','.join(str(s) for s in seasons)}) AND league IN "
        f"({','.join(repr(l) for l in leagues)})"
    )
    con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM incoming_injuries")
    con.unregister("incoming_injuries")
    return len(frame)


def load_injuries(con, teams: list[str] | None = None) -> pd.DataFrame:
    """Stored injuries, optionally for a set of teams."""
    if teams:
        placeholders = ",".join("?" for _ in teams)
        return con.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE team IN ({placeholders})", teams
        ).df()
    return con.execute(f"SELECT * FROM {TABLE_NAME}").df()


def for_teams(
    frame: pd.DataFrame,
    teams: list[str],
    on_date: dt.date | None = None,
) -> pd.DataFrame:
    """Injuries for a set of teams, worst news first.

    **`on_date` is not optional in practice.** The feed returns one row per
    player *per fixture*, so a league-season is thousands of rows - 3,168 for
    the Premier League in 2024. Without a date filter a team's panel would list
    every absence of the entire season, including players who were back within
    a fortnight, and it would look like a catastrophic injury crisis at every
    club. Filtering on the fixture the row actually refers to is exact, because
    the feed says which fixture it means.

    Ruled out before doubtful, because that is the order a reader cares about,
    and alphabetical within each so the list is stable between refreshes
    instead of reshuffling on every page load.
    """
    if frame.empty:
        return frame
    subset = frame[frame["team"].isin(teams)].copy()
    if on_date is not None and "fixture_date" in subset.columns:
        dates = pd.to_datetime(subset["fixture_date"], errors="coerce").dt.date
        subset = subset[dates == on_date]
    if subset.empty:
        return subset
    return subset.sort_values(
        ["ruled_out", "doubtful", "team", "player"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def freshness(frame: pd.DataFrame) -> dt.datetime | None:
    """When the feed was last read. Shown on the page, never hidden.

    A stale injury list is worse than none, because it looks current.
    """
    if frame.empty or "fetched_at" not in frame:
        return None
    stamp = pd.to_datetime(frame["fetched_at"]).max()
    return None if pd.isna(stamp) else stamp.to_pydatetime()
