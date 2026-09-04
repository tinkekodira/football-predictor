"""Central configuration.

Everything that is a tunable knob lives here rather than being scattered through
the codebase, so that later phases (models, backtests) can import the same
league/season definitions the ingest layer used.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "football.duckdb"


# --------------------------------------------------------------------------
# Competitions
# --------------------------------------------------------------------------
# football-data.co.uk division codes for the top-5 European leagues.
# UEFA competitions are deliberately out of scope for v1 - this source does
# not cover them (see README, Phase 5).

LEAGUES: dict[str, str] = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Countries, used for display and for later work on referee pools (referees
# are national, so a referee's card rate is only comparable within a country).
LEAGUE_COUNTRY: dict[str, str] = {
    "E0": "England",
    "SP1": "Spain",
    "I1": "Italy",
    "D1": "Germany",
    "F1": "France",
}


# --------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------

# The season currently in progress, identified by the year it started.
CURRENT_SEASON_START_YEAR = 2026

# How many seasons of history to ingest by default.
#
# Ten is generous on purpose: the models use exponential time decay, so old
# matches contribute very little weight but cost nothing to store, and having
# the depth available means we can lengthen the half-life during Phase 3
# tuning without re-downloading anything.
DEFAULT_HISTORY_SEASONS = 10


def season_code(start_year: int) -> str:
    """Convert a season's starting year into football-data.co.uk's code.

    >>> season_code(2026)
    '2627'
    >>> season_code(1999)
    '9900'
    """
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Human-readable season label.

    >>> season_label(2026)
    '2026/27'
    """
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_years(n_seasons: int = DEFAULT_HISTORY_SEASONS) -> list[int]:
    """Starting years for the n most recent seasons, oldest first."""
    if n_seasons < 1:
        raise ValueError("n_seasons must be >= 1")
    first = CURRENT_SEASON_START_YEAR - n_seasons + 1
    return list(range(first, CURRENT_SEASON_START_YEAR + 1))


# --------------------------------------------------------------------------
# Source URLs
# --------------------------------------------------------------------------

BASE_URL = "https://www.football-data.co.uk/mmz4281"

USER_AGENT = "football-edge/0.1 (personal analytics project)"

# --------------------------------------------------------------------------
# Upcoming fixtures (Phase 4)
# --------------------------------------------------------------------------
# The source publishes the next few days of fixtures with current prices, and
# **overwrites the file** each time it rebuilds it. Nothing archives the old
# copies, so a price not written down when it is pulled cannot be recovered.
# `fbedge/snapshots.py` exists for that reason; see its docstring.

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
# The Excel twin of the same data. Not used - the CSV needs no extra
# dependency - but recorded here so nobody has to go looking for it.
FIXTURES_XLSX_URL = "https://www.football-data.co.uk/fixtures.xlsx"

# How long a downloaded fixtures file may be reused before re-fetching. Short,
# because the whole point of a snapshot archive is to catch price changes; a
# long cache would archive the same prices repeatedly and miss the moves.
FIXTURES_CACHE_HOURS = 2.0

# How old the file may be before a scan refuses to run. Prices are collected
# twice a week (Friday afternoon for the weekend, Tuesday afternoon for
# midweek), so anything past about three days is either a browser-cache
# problem - the source warns about exactly this on its own page - or a source
# that has stopped updating. Either way, scanning it silently is worse than
# stopping.
FIXTURES_MAX_AGE_HOURS = 72.0

# Where the archive is mirrored as plain CSV, and it is **tracked in git**.
#
# The database is not tracked (BACKLOG B3) because it rebuilds from static
# files in two minutes. The snapshot archive is the one thing in it that does
# not: the source overwrites `fixtures.csv` and keeps no history, so a price
# lost here is lost permanently. That is exactly the argument B3 used for
# keeping the season CSVs tracked, applied to the one table that has a stronger
# claim to it - the season files could at least be re-downloaded, and these
# could not.
#
# Long format and sorted deterministically so a commit diff is the week's new
# prices and nothing else.
SNAPSHOT_EXPORT_DIR = DATA_DIR / "snapshots"

# How far a snapshot's fixture date may drift from the played match's date and
# still be the same match. One day catches postponements and late kick-offs
# that roll over midnight, and is tight enough that two legs of a tie cannot
# be confused: the same pair does not play twice inside three days.
FIXTURE_RECONCILE_TOLERANCE_DAYS = 1

# --------------------------------------------------------------------------
# The forward calendar (Phase 4)
# --------------------------------------------------------------------------
# `fixtures.csv` gives the next few days. A whole remaining season needs a
# source that publishes one. Three are wired up in `fbedge/calendar.py`; this
# picks between them.
#
# "auto" uses football-data.org when a token is configured and openfootball
# when one is not, so a machine with no key does something sensible rather than
# failing on a setting it never chose. "understat" is what the home page has
# always used and needs no key either.
CALENDAR_SOURCE = "auto"

# Never committed, and read from the environment for the same reason the injury
# key is: a key in a tracked file is a key in the history for ever.
CALENDAR_TOKEN_ENV = "FOOTBALL_DATA_ORG_TOKEN"

# football-data.org's published free-tier allowance, confirmed on their pricing
# page: twelve competitions, ten calls a minute, scores and schedules delayed.
# Enforced client-side in `calendar._respect_rate_limit` rather than left to the
# server to reject - a 429 is a wasted request, and repeated hammering is how
# access gets withdrawn without warning.
CALENDAR_CALLS_PER_MINUTE = 10

# A season calendar changes rarely. Twelve hours keeps a postponement current
# without spending quota or somebody else's GitHub bandwidth; a finished season
# is never refetched at all.
CALENDAR_CACHE_HOURS = 12.0

# --------------------------------------------------------------------------
# Injury feed
# --------------------------------------------------------------------------
# The only genuinely external, keyed dependency in the project. Everything
# else here is a public file or a public JSON endpoint; injuries are not
# published free by anyone without a signup, so this one needs a key and
# degrades to "no data" without it rather than failing.

INJURY_API_URL = "https://v3.football.api-sports.io/injuries"

# Read from the environment rather than stored: a key in a tracked file is a
# key in the history for ever, and this repository has no .gitignore.
INJURY_API_KEY_ENV = "FOOTBALL_API_KEY"

# API-Football's own competition ids, which are unrelated to the
# football-data.co.uk division codes used everywhere else. Kept here rather
# than inline so that a wrong id is fixable in one place - and it is the kind
# of thing that is wrong silently, returning an empty list for a league that
# simply has a different number.
INJURY_LEAGUE_IDS: dict[str, int] = {
    "E0": 39,    # Premier League
    "SP1": 140,  # La Liga
    "I1": 135,   # Serie A
    "D1": 78,    # Bundesliga
    "F1": 61,    # Ligue 1
}

# The free plan allows 100 requests a day and one call covers a whole league,
# so five calls refresh everything. Two hours keeps a matchday current without
# coming close to the limit.
INJURY_CACHE_HOURS = 2

# The free plan's published allowances, confirmed 2026-09-04: 100 requests a
# day, resetting at 00:00 UTC with unused requests lost, and 10 a minute.
#
# **Both are enforced locally, before the request goes out.** A spent quota is
# not a soft failure: the endpoint keeps answering 200 with an empty list on
# some errors, so a run that blows the budget can look like a league with
# nobody injured. And repeated hammering is how access gets withdrawn without
# warning, which would cost the only keyed source in the project.
INJURY_DAILY_QUOTA = 100
INJURY_CALLS_PER_MINUTE = 10

# Stop this many requests short of the published daily limit. One careless
# loop is all it takes, and leaving a few in reserve means a spent budget can
# still be diagnosed interactively rather than only tomorrow.
INJURY_QUOTA_RESERVE = 5

# Where the spend is recorded. A file rather than a variable, because the limit
# is per key per day and this project is many short-lived processes: an
# in-memory counter would reset on every script invocation and count nothing.
INJURY_QUOTA_PATH = DATA_DIR / "raw" / "injuries" / "quota.json"

# **The free plan caps the seasons it will serve.** Confirmed empirically:
# 2024 returns 3,168 Premier League rows, the current season returns nothing.
# That is not a bug and not a wrong league id, and the difference matters
# because both look identical - an empty list. Anything asking for a season
# past this on a free key should say so rather than shrug.
INJURY_FREE_PLAN_LAST_SEASON = 2024

# Seconds to wait between downloads. The site is a static file host and has no
# published rate limit, but there is no reason to hammer it.
REQUEST_DELAY_SECONDS = 1.5
REQUEST_TIMEOUT_SECONDS = 30

# The in-progress season's file grows as matches are played, so it is
# re-downloaded when the cached copy is older than this.
CURRENT_SEASON_CACHE_HOURS = 6


def season_csv_url(league: str, start_year: int) -> str:
    """URL of one league-season CSV, e.g. .../mmz4281/2627/E0.csv"""
    if league not in LEAGUES:
        raise KeyError(f"Unknown league code {league!r}. Known: {sorted(LEAGUES)}")
    return f"{BASE_URL}/{season_code(start_year)}/{league}.csv"
