"""Download the remaining season's fixtures from a source that publishes one.

    python scripts/build_calendar.py                        # whichever source is configured
    python scripts/build_calendar.py --source openfootball  # no key needed
    python scripts/build_calendar.py --source football_data_org

`fixtures.csv` carries the next few days, because its job is to ship prices for
imminent matches. This fills in the rest of the season.

**Optional, always.** Nothing else in the project depends on this table. With no
token and no network the pipeline runs exactly as it did before, and
`scripts/build_fixtures.py` remains the calendar the home page reads.

**An unresolved club name stops the run.** A fixture whose team does not map to
a canonical name is present, looks fine, and joins to nothing for the rest of
the season - which is indistinguishable from a fixture that was never
scheduled. The names are printed so the alias table can be extended from what
the feed actually says.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import calendar as calendar_mod  # noqa: E402
from fbedge import config, database, fixtures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), action="append")
    parser.add_argument("--season", type=int, default=None,
                        help="Starting year. Defaults to the season in progress.")
    parser.add_argument("--source", choices=calendar_mod.SOURCES, default=None,
                        help=f"Defaults to config.CALENDAR_SOURCE "
                             f"({config.CALENDAR_SOURCE!r}).")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore the cache. A season calendar changes rarely, "
                             "so this is rarely what you want.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    leagues = args.league or list(config.LEAGUES)
    season = args.season if args.season is not None else fixtures.current_season()
    source = calendar_mod.resolve_source(args.source)

    # Read-only for the fetch, write connection opened afterwards. DuckDB
    # allows one writer and a download can be slow; BACKLOG B2 records the two
    # times this project locked itself out by holding the lock across one.
    read_con = database.connect(args.db, read_only=True)
    known = {
        league: set(database.known_teams(read_con, league=league))
        for league in leagues
    }
    read_con.close()

    if source == calendar_mod.FOOTBALL_DATA_ORG and not calendar_mod.token():
        print(
            f"Source is {source!r} but {config.CALENDAR_TOKEN_ENV} is not set.\n"
            "Either export a free token from https://www.football-data.org/ or "
            "use --source openfootball, which needs none."
        )
        return 1

    print(f"Fetching {season}/{(season + 1) % 100:02d} from {source} ...")
    try:
        frame = calendar_mod.fetch(
            leagues, season, known, source=source, force=args.refresh
        )
    except calendar_mod.UnresolvedTeams as exc:
        print(f"\n{exc}\n")
        print("Nothing was written. Add the names above to the alias table for")
        print(f"this source in fbedge/calendar.py, then re-run.")
        return 1
    except calendar_mod.CalendarError as exc:
        print(f"\nCalendar fetch failed: {exc}")
        return 1

    write_con = database.connect(args.db)
    calendar_mod.write(write_con, frame)

    print(f"\nWrote {len(frame)} rows to `{calendar_mod.TABLE_NAME}` from {source}.")
    summary = (
        frame.groupby("league")
        .agg(
            fixtures=("home_team", "size"),
            played=("played", "sum"),
            first=("kickoff_utc", "min"),
            last=("kickoff_utc", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    stored = calendar_mod.load(write_con)
    if stored["source"].nunique() > 1:
        print(
            "\nMore than one source is stored. That is supported on purpose - "
            "provenance is a column - but read a fixture list from one source "
            "at a time, because they will disagree about kick-off times."
        )
        print(stored.groupby("source").size().to_string())
    write_con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
