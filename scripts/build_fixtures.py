"""Load the season calendar into the database.

    python scripts/build_fixtures.py                 # current season, five leagues
    python scripts/build_fixtures.py --season 2025   # a finished season
    python scripts/build_fixtures.py --refresh       # ignore the cache

Fills the `fixtures` table that the home page reads: every match in the season,
played or not, with kick-off times in UTC. See `fbedge.fixtures` for why this
comes from Understat rather than from the project's primary source.

**Safe to run while the app is open?** No - DuckDB takes one writer at a time.
This script follows the shape `build_rosters.py` settled on after locking the
database for fifty minutes once: the network work happens with no connection
held, and a write connection is opened only for the final insert. It is still
a write, so the app has to be stopped for the few seconds it takes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, fixtures, understat  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="season start year (default: the current one)")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download even if the cache is fresh")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/understat"))
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    season = args.season if args.season is not None else fixtures.current_season()
    leagues = args.leagues or fixtures.default_leagues()

    # Every network request happens here, with no database connection open.
    print(f"Fetching {season}/{str(season + 1)[-2:]} for {', '.join(leagues)}...")
    try:
        calendar = fixtures.fetch_calendar(
            leagues, season, args.cache, refresh=args.refresh
        )
    except understat.UnderstatError as error:
        print(f"  failed: {error}")
        return 1

    played = int(calendar["played"].sum())
    print(f"  {len(calendar)} fixtures, {played} played, "
          f"{len(calendar) - played} still to come")
    for league in leagues:
        rows = calendar[calendar["league"] == league]
        if rows.empty:
            continue
        print(
            f"    {league}: {len(rows):3d} fixtures, "
            f"{rows['kickoff_utc'].min().date()} to {rows['kickoff_utc'].max().date()}"
        )

    unknown = _unmapped_team_names(calendar)
    if unknown:
        # Not fatal, but it means the model cannot find these teams, so the
        # calendar would show a fixture the detail page could not price.
        print(
            f"\n  WARNING: {len(unknown)} team name(s) look untranslated: "
            f"{', '.join(sorted(unknown)[:8])}"
        )
        print("  Check understat.TEAM_ALIASES; a wrong name silently becomes a "
              "team the model has never seen.")

    con = database.connect(args.db, read_only=False)
    try:
        written = fixtures.write_calendar(con, calendar)
    finally:
        con.close()
    print(f"\nWrote {written} rows to `{fixtures.TABLE_NAME}`.")

    today = dt.date.today()
    upcoming = fixtures.matches_on(calendar, today)
    print(f"Today ({today}) has {len(upcoming)} fixture(s) across these leagues.")
    return 0


def _unmapped_team_names(calendar) -> set[str]:
    """Understat names that came through the alias table unchanged.

    A name that is identical on both sides is normal and common - "Inter" is
    the same in both sources. This only flags names carrying the punctuation
    Understat uses and football-data does not, which is the signature of an
    alias that was never added.
    """
    suspicious = set()
    for column in ("home_team", "away_team"):
        for name in calendar[column].unique():
            if any(ch in str(name) for ch in "éüöáí"):
                suspicious.add(str(name))
    return suspicious


if __name__ == "__main__":
    raise SystemExit(main())
