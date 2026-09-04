"""Load current injuries from the external feed.

    setx FOOTBALL_API_KEY "your-key"        # Windows, once
    python scripts/build_injuries.py
    python scripts/build_injuries.py --probe --league E0   # see the raw shape

Needs a free key from https://www.api-football.com/ in the `FOOTBALL_API_KEY`
environment variable. One request covers a whole league, so a full refresh of
all five costs five of the free plan's hundred daily requests.

**The free plan stops at the 2024 season.** Confirmed by probing: 2024 returns
3,168 Premier League rows, the current season returns none. So a free key is
enough to *study* injuries - 2022 to 2024 is real, per-fixture history - but
not to show today's team news. That needs a paid plan.

**`--probe` prints the raw JSON for one league and writes nothing.** The parser
was originally written against the published schema because the documentation
refused an automated fetch; it has since been confirmed against a real 2024
response, and `test_the_parser_agrees_with_a_real_response` pins that entry
verbatim. Probe again before trusting anything if the provider changes.

**Unmatched club names are printed and the rows dropped.** The feed's team
names are not the project's, and this project has been badly served once
already by a fuzzy matcher; nothing here guesses. Add what it prints to
`injuries.TEAM_ALIASES` after checking each one by eye.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, fixtures, injuries  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--league", default=None,
                        help="shorthand for a single league, used with --probe")
    parser.add_argument("--probe", action="store_true",
                        help="print the raw response for one league and stop")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache even if it is fresh")
    parser.add_argument("--cache", type=Path,
                        default=config.RAW_DIR / injuries.CACHE_DIRNAME)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    season = args.season if args.season is not None else fixtures.current_season()
    leagues = args.leagues or ([args.league] if args.league
                               else fixtures.default_leagues())

    if injuries.api_key() is None:
        print(
            f"No {config.INJURY_API_KEY_ENV} set, so there is nothing to fetch.\n"
            "\n"
            "  1. Sign up free at https://www.api-football.com/ (no card).\n"
            "  2. Copy the key from the dashboard.\n"
            f"  3. setx {config.INJURY_API_KEY_ENV} \"your-key\"  "
            "then open a new terminal.\n"
            "\n"
            "The free plan allows 100 requests a day; a full refresh costs five."
        )
        return 1

    if args.probe:
        return probe(leagues[0], season, args.cache, args.refresh)

    # Team names we already know, so the matcher can compare against reality
    # rather than only against its alias table.
    con = database.connect(args.db, read_only=True)
    known = set(database.known_teams(con))
    con.close()

    frames, all_unmatched = [], {}
    for league in leagues:
        try:
            payload = injuries.fetch_league(
                league, season, args.cache, force=args.refresh
            )
        except injuries.InjuryFeedError as error:
            print(f"{league}: {error}")
            continue

        frame, unmatched = injuries.injury_frame(payload, league, season, known)
        out = int(frame["ruled_out"].sum()) if not frame.empty else 0
        doubtful = int(frame["doubtful"].sum()) if not frame.empty else 0
        print(f"{league}: {len(frame)} entries ({out} out, {doubtful} doubtful)")
        if not payload.get("response") and season > config.INJURY_FREE_PLAN_LAST_SEASON:
            # An out-of-plan season and a genuinely quiet league both come back
            # as an empty list, so say which this probably is.
            print(
                f"  ...and nothing came back. The free plan stops at "
                f"{config.INJURY_FREE_PLAN_LAST_SEASON}; season {season} needs "
                "a paid plan."
            )
        if unmatched:
            all_unmatched[league] = unmatched
        if not frame.empty:
            frames.append(frame)

    if all_unmatched:
        print("\n  UNMATCHED CLUB NAMES - these rows were dropped, not guessed:")
        for league, names in all_unmatched.items():
            for name in names:
                print(f"    {league}: {name!r}  ->  normalises to "
                      f"{injuries.normalise(name)!r}")
        print("  Add each to injuries.TEAM_ALIASES once you have checked which "
              "club it is.")

    if not frames:
        print("\nNothing to write.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    con = database.connect(args.db, read_only=False)
    try:
        written = injuries.write_injuries(con, combined)
    finally:
        con.close()
    print(f"\nWrote {written} rows to `{injuries.TABLE_NAME}` at "
          f"{dt.datetime.now():%Y-%m-%d %H:%M}.")
    return 0


def probe(league: str, season: int, cache: Path, refresh: bool) -> int:
    """Print one league's raw response so the parser can be checked against it."""
    try:
        payload = injuries.fetch_league(league, season, cache, force=refresh)
    except injuries.InjuryFeedError as error:
        print(f"{league}: {error}")
        return 1

    print(f"top-level keys: {sorted(payload)}")
    print(f"results: {payload.get('results')}  paging: {payload.get('paging')}")
    entries = payload.get("response", [])
    print(f"response entries: {len(entries)}")
    if entries:
        print("\nfirst entry, in full:")
        print(json.dumps(entries[0], indent=2)[:2000])
        print("\nblocks present on the first entry:", sorted(entries[0]))
    elif season > config.INJURY_FREE_PLAN_LAST_SEASON:
        # An out-of-plan season and a genuinely quiet league both come back as
        # an empty list, and only one of them is worth investigating.
        print(
            f"\nNothing came back, and season {season} is past the free plan's "
            f"last ({config.INJURY_FREE_PLAN_LAST_SEASON}). On a free key that "
            "is the expected answer, not a fault. Try "
            f"--season {config.INJURY_FREE_PLAN_LAST_SEASON}."
        )
    else:
        print(
            "\nThe feed returned no entries. That is plausible in a quiet week "
            "but is also what a wrong league id looks like; check "
            f"config.INJURY_LEAGUE_IDS[{league!r}] = "
            f"{config.INJURY_LEAGUE_IDS.get(league)}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
