"""Download Understat xG and attach it to the matches already in the database.

    python scripts/build_xg.py                    # all five leagues, 2017 on
    python scripts/build_xg.py --league E0
    python scripts/build_xg.py --refresh          # re-download, ignoring cache

Creates a `match_xg` table keyed on the existing `match_id`, so nothing about
the matches table changes and every model can opt in by joining.

**The join is the part that can go wrong, so it is measured.** Two sources
name teams differently (41 of 156 names differ), disagree about which
competitions belong to a league, and disagree at the edges about dates when a
fixture is rearranged. A row that fails to join is not an error - it is
normal - but a *drop* in the join rate means the alias table has rotted, and
the difference between "94% joined, same as last time" and "62% joined" is the
difference between working data and a silently crippled model. The script
prints the rate per league and refuses to write if it collapses.

Matching is on (league, season, home team, away team) rather than on the date,
because a postponed fixture keeps its teams but moves its date, and the two
sources do not always agree about the new one. Within one league-season a pair
of teams meets exactly once at home, so the pair is the natural key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, understat  # noqa: E402

# Below this share of matched fixtures something is wrong with the mapping
# rather than with a handful of rearranged games.
MINIMUM_JOIN_RATE = 0.80


def collect(league: str, seasons: list[int], cache: Path, refresh: bool) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            payload = understat.fetch_season(league, season, cache, force=refresh)
        except understat.UnderstatError as exc:
            print(f"  {league} {season}: skipped, {exc}")
            continue
        frames.append(understat.match_frame(payload, league, season))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default=None)
    parser.add_argument("--from-season", type=int, default=2017)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download even when a cached copy exists")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/understat"))
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    leagues = [args.league] if args.league else sorted(understat.LEAGUE_SLUGS)
    seasons = understat.season_range(args.from_season)

    # **Read on a read-only connection and close it before fetching anything.**
    # DuckDB allows one writer at a time, so holding a write handle across the
    # downloads below locks the app, the tests and every read-only script for
    # the whole run - which on a cold cache is tens of minutes. The same defect
    # was found in `build_rosters.py`, where it locked the database for about
    # fifty minutes; this is the same fix. The write connection is opened at
    # the bottom, once there is something to write.
    con = database.connect(args.db, read_only=True)
    try:
        fixtures = con.execute(
            """
            SELECT match_id, league, season_start_year, date, home_team, away_team,
                   home_goals, away_goals
            FROM matches
            WHERE home_goals IS NOT NULL
            """
        ).df()
    finally:
        con.close()

    joined_frames = []
    # Which leagues actually produced rows, so the write can replace those
    # and only those. A league that was skipped must keep whatever it had.
    written_leagues: list[str] = []
    print(f"Seasons {seasons[0]}-{seasons[-1]}\n")
    for league in leagues:
        scraped = collect(league, seasons, args.cache, args.refresh)
        if scraped.empty:
            print(f"{league}: nothing downloaded")
            continue

        ours = fixtures[fixtures["league"] == league]
        merged = ours.merge(
            scraped[[
                "league", "season_start_year", "home_team", "away_team",
                "home_xg", "away_xg", "home_goals", "away_goals", "understat_id",
            ]],
            on=["league", "season_start_year", "home_team", "away_team"],
            how="left",
            suffixes=("", "_understat"),
        )
        matched = merged["home_xg"].notna()
        rate = float(matched.mean()) if len(merged) else 0.0
        print(
            f"{league}: {int(matched.sum())}/{len(merged)} fixtures matched "
            f"({rate:.1%}), {len(scraped)} available from Understat"
        )

        if rate < MINIMUM_JOIN_RATE:
            print(
                f"  -> below {MINIMUM_JOIN_RATE:.0%}. Refusing to write this league. "
                "Check understat.TEAM_ALIASES against the unmatched names below."
            )
            unmatched = merged.loc[~matched, ["home_team", "away_team"]]
            names = sorted(set(unmatched["home_team"]) | set(unmatched["away_team"]))
            print(f"     teams in unmatched fixtures: {names[:25]}")
            continue

        # A disagreement about the score means the two sources are describing
        # different matches, whatever the team names say.
        agree = (
            (merged["home_goals"] == merged["home_goals_understat"])
            & (merged["away_goals"] == merged["away_goals_understat"])
        )
        conflicts = int((matched & ~agree).sum())
        if conflicts:
            print(f"  -> {conflicts} matched fixtures disagree on the score; dropping them")
        keep = merged[matched & agree]
        written_leagues.append(league)
        joined_frames.append(
            keep[["match_id", "understat_id", "home_xg", "away_xg"]]
        )

    if not joined_frames:
        print("\nNothing to write.")
        return 1

    combined = pd.concat(joined_frames, ignore_index=True).drop_duplicates("match_id")
    # Only now, with every download done, take the write lock.
    con = database.connect(args.db, read_only=False)
    try:
        # **Replace only the leagues this run actually covered.** The previous
        # version dropped the whole table and re-inserted whatever it had just
        # built, so `--league E0` silently deleted the xG for the other four -
        # found by running it. Same shape as the scoped deletes in
        # `fixtures.write_calendar` and `injuries.write_injuries`, and the same
        # reason: a partial run must not look like a full one.
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS match_xg (
                match_id      VARCHAR PRIMARY KEY,
                understat_id  VARCHAR,
                home_xg       DOUBLE,
                away_xg       DOUBLE
            )
            """
        )
        placeholders = ",".join("?" for _ in written_leagues)
        con.execute(
            f"DELETE FROM match_xg WHERE match_id IN "
            f"(SELECT match_id FROM matches WHERE league IN ({placeholders}))",
            written_leagues,
        )
        con.register("combined_xg", combined)
        con.execute("INSERT INTO match_xg SELECT * FROM combined_xg")
        con.unregister("combined_xg")

        print(f"\nWrote {len(combined)} rows to match_xg for "
              f"{', '.join(written_leagues)}.")
        print(
            con.execute(
                """
                SELECT m.league, COUNT(*) AS matches,
                       ROUND(AVG(x.home_xg), 3) AS mean_home_xg,
                       ROUND(AVG(x.away_xg), 3) AS mean_away_xg,
                       ROUND(AVG(m.home_goals), 3) AS mean_home_goals,
                       ROUND(AVG(m.away_goals), 3) AS mean_away_goals
                FROM match_xg x JOIN matches m USING (match_id)
                GROUP BY m.league ORDER BY m.league
                """
            ).df().to_string(index=False)
        )
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
