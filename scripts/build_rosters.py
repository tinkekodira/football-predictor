"""Download Understat per-match rosters and load them into the database.

    python scripts/build_rosters.py --league E0
    python scripts/build_rosters.py --league E0 --pause 1.0

Creates a `match_lineups` table: one row per player appearance, keyed on the
`match_id` already in `matches`. Everything is cached one file per match, so an
interrupted run resumes rather than restarting.

**This is thousands of requests against somebody else's free service.** One
league of nine seasons is about 3,400 of them. The default pause is deliberately
unhurried, the cache means it only ever happens once, and there is no reason to
run it for a league nobody is going to model.

**What this data may and may not be used for.** It records who actually played.
Confirmed line-ups appear about an hour before kick-off, which is after the
opening price this project bets into and at or after the closing line it
measures against. Using it as a feature of the match it describes would be
lookahead, and it would not look like a bug - availability genuinely matters, so
it would look like a large edge. `fbedge/availability.py` derives the
before-kick-off features and is where that rule is enforced.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, understat  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--pause", type=float, default=0.6,
                        help="Seconds between requests (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many new downloads; for a smoke test")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/understat"))
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    # Read the fixture list through a read-only connection and close it before
    # any network work. An earlier version held a write connection open for the
    # whole download, which locks the database for the best part of an hour and
    # blocks the app, the tests and every other script for no reason: the write
    # takes a second and belongs at the end.
    con = database.connect(args.db, read_only=True)
    fixtures = con.execute(
        """
        SELECT x.match_id, x.understat_id
        FROM match_xg x JOIN matches m USING (match_id)
        WHERE m.league = ? AND x.understat_id IS NOT NULL
        ORDER BY m.date
        """,
        [args.league],
    ).df()
    con.close()
    if fixtures.empty:
        print(f"No {args.league} fixtures with an understat_id. Run build_xg.py first.")
        return 1

    cache = Path(args.cache) / understat.MATCH_CACHE_DIRNAME
    already = {p.stem for p in cache.glob("*.json")} if cache.exists() else set()
    todo = [u for u in fixtures["understat_id"] if u not in already]
    print(f"{args.league}: {len(fixtures)} fixtures, {len(already & set(fixtures['understat_id']))} cached, "
          f"{len(todo)} to download (~{len(todo) * args.pause / 60:.0f} min)")

    frames, failures, downloaded = [], [], 0
    began = time.time()
    for i, row in enumerate(fixtures.itertuples(), start=1):
        if args.limit is not None and downloaded >= args.limit:
            break
        try:
            was_cached = row.understat_id in already
            payload = understat.fetch_match(row.understat_id, args.cache, pause=args.pause)
            if not was_cached:
                downloaded += 1
        except understat.UnderstatError as exc:
            failures.append((row.understat_id, str(exc)))
            continue
        frames.append(understat.roster_frame(payload, row.match_id))
        if i % 250 == 0:
            print(f"  {i}/{len(fixtures)}  ({time.time() - began:.0f}s elapsed)")

    if not frames:
        print("Nothing parsed.")
        return 1
    lineups = pd.concat(frames, ignore_index=True)

    con = database.connect(args.db, read_only=False)
    con.execute("DROP TABLE IF EXISTS match_lineups")
    con.execute(
        """
        CREATE TABLE match_lineups (
            match_id     VARCHAR,
            is_home      BOOLEAN,
            player_id    VARCHAR,
            player       VARCHAR,
            position     VARCHAR,
            started      BOOLEAN,
            minutes      INTEGER,
            yellow_card  INTEGER,
            red_card     INTEGER,
            xg           DOUBLE,
            xa           DOUBLE,
            xgchain      DOUBLE
        )
        """
    )
    con.register("lineups_in", lineups)
    con.execute("INSERT INTO match_lineups SELECT * FROM lineups_in")
    con.unregister("lineups_in")

    print(f"\nWrote {len(lineups)} appearances for {lineups['match_id'].nunique()} matches.")
    if failures:
        print(f"{len(failures)} matches failed; first few: {failures[:3]}")
    print(
        con.execute(
            """
            SELECT m.season_start_year AS season,
                   COUNT(DISTINCT l.match_id) AS matches,
                   COUNT(*) AS appearances,
                   ROUND(AVG(CASE WHEN l.started THEN 1.0 ELSE 0.0 END), 3) AS started_share
            FROM match_lineups l JOIN matches m USING (match_id)
            GROUP BY 1 ORDER BY 1
            """
        ).df().to_string(index=False)
    )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
