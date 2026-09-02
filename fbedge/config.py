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
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

USER_AGENT = "football-edge/0.1 (personal analytics project)"

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
