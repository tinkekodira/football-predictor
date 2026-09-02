"""Build the match database from scratch.

    python scripts/build_database.py                 # 10 seasons, all 5 leagues
    python scripts/build_database.py --seasons 4     # faster first run
    python scripts/build_database.py --force         # ignore the cache
    python scripts/build_database.py --local-dir tmp # load CSVs already on disk

Re-running is safe and cheap: finished seasons come from the local cache and
existing rows are replaced rather than duplicated. Run it after each round of
matches to pull in the new results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, ingest, normalize, quality  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seasons", type=int, default=config.DEFAULT_HISTORY_SEASONS,
        help="How many seasons of history to load (default: %(default)s).",
    )
    parser.add_argument(
        "--leagues", nargs="+", choices=sorted(config.LEAGUES),
        help="Restrict to specific league codes (default: all five).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even when a cached copy exists.",
    )
    parser.add_argument(
        "--local-dir", type=Path,
        help="Load CSVs from a local directory (<dir>/<season>/<league>.csv) "
             "instead of downloading. Useful offline and in tests.",
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help=f"Database path (default: {config.DB_PATH}).",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip the data quality report at the end.",
    )
    return parser.parse_args()


def collect_files(args) -> list[tuple[str, int, Path]]:
    """Resolve every (league, season, csv path) to load."""
    leagues = args.leagues or list(config.LEAGUES)
    years = config.season_years(args.seasons)

    if args.local_dir:
        found = []
        for year in years:
            for league in leagues:
                path = args.local_dir / config.season_code(year) / f"{league}.csv"
                if path.exists():
                    found.append((league, year, path))
                else:
                    print(f"  missing {path}")
        return found

    print(f"Downloading {len(leagues)} leagues x {len(years)} seasons "
          f"({config.season_label(years[0])} to {config.season_label(years[-1])})")
    files = ingest.download_all(
        leagues=leagues, years=years, force=args.force, on_progress=print
    )
    return [(f.league, f.season_start_year, f.path) for f in files]


def main() -> int:
    args = parse_args()
    files = collect_files(args)
    if not files:
        print("No source files found. Nothing to do.")
        return 1

    con = database.connect(args.db)
    total_matches = total_odds = 0

    print("\nLoading into DuckDB")
    for league, year, path in files:
        try:
            raw = ingest.read_raw_csv(path)
            matches, odds = normalize.normalize_league_season(raw, league, year)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the build
            print(f"  FAILED {league} {config.season_label(year)}: {exc}")
            continue

        n_matches = database.load_matches(con, matches)
        n_odds = database.load_odds(con, odds)
        database.log_ingest(con, league, year, n_matches, n_odds)
        total_matches += n_matches
        total_odds += n_odds
        print(f"  {league} {config.season_label(year)}: "
              f"{n_matches:>4} matches, {n_odds:>6} odds rows")

    print(f"\nLoaded {total_matches} matches and {total_odds} odds rows.")
    print(f"Database: {args.db or config.DB_PATH}")

    print("\n=== Contents ===")
    print(database.summary(con).to_string(index=False))

    if not args.skip_checks:
        quality.run_all(con)

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
