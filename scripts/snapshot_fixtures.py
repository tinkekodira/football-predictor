"""Archive the current upcoming-fixtures file. Run it often; it is cheap.

    python scripts/snapshot_fixtures.py
    python scripts/snapshot_fixtures.py --refresh          # force a download
    python scripts/snapshot_fixtures.py --reconcile-only   # no network at all

**Run this on a schedule if you run nothing else in Phase 4.** The source
overwrites `fixtures.csv` every time it rebuilds it and archives nothing, so a
pre-match price that is not captured when it is published is gone permanently.
Every other table in this project can be rebuilt from static files in two
minutes; this one cannot be rebuilt at all.

Twice a week is the useful minimum, because that is how often the source
collects prices: Friday afternoons for weekend fixtures and Tuesday afternoons
for midweek ones. Running it more often costs nothing and stores nothing extra
- identical prices are deduped by content hash - but it does catch the
occasional intra-window revision.

The run also reconciles archived snapshots against matches that have since been
played, which is what will eventually let closing line value be measured on our
own recorded prices rather than only on the source's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, snapshots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="Force a download even if the cached copy is recent")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Join existing snapshots to played matches; no network")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Archive every division in the file, not only the top five. "
                             "The models cannot price the rest, but the prices are "
                             "unrecoverable either way and the file is 60KB.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    con = database.connect(args.db)

    if not args.reconcile_only:
        leagues = None if args.all_leagues else list(config.LEAGUES)
        frame, path = snapshots.download(force=args.refresh)
        snapshot, odds = snapshots.build_snapshot(frame, leagues=leagues)

        # Reported, never enforced here. A stale file is a reason to refuse to
        # *scan*; it is not a reason to refuse to *archive*, because the rows
        # are still the only copy of those prices in existence.
        report = snapshots.staleness(snapshot, path)
        if report["stale"]:
            print("WARNING: this file looks stale.")
            for reason in report["reasons"]:
                print(f"  - {reason}")
            print("  Archived anyway; scans will refuse it. Try --refresh.")

        counts = snapshots.write_snapshot(con, snapshot, odds)
        print(
            f"{counts['fixtures_seen']} fixtures in the file: "
            f"{counts['new_snapshots']} new, {counts['repeat_snapshots']} already "
            f"archived, {counts['new_odds_rows']} new price rows."
        )
        if counts["new_snapshots"] == 0 and counts["fixtures_seen"]:
            print("  Nothing new is the normal result between price collections.")

    result = snapshots.reconcile(con)
    print(
        f"\nReconciliation: {result['matched']} snapshots newly joined to played "
        f"matches, {result['already_matched']} already joined, "
        f"{result['unmatched']} still unmatched."
    )
    unmatched = result["unmatched_rows"]
    if len(unmatched):
        import pandas as pd

        past = unmatched[
            pd.to_datetime(unmatched["fixture_date"]).dt.date < pd.Timestamp.today().date()
        ]
        if len(past):
            print(
                f"\n  {len(past)} unmatched fixture(s) whose date has passed. These "
                "are name-resolution failures, not fixtures still to come:"
            )
            for row in past.head(20).itertuples():
                print(f"    {row.league} {row.fixture_date} "
                      f"{row.home_team} vs {row.away_team}")
            print("    Add the spelling to normalize.TEAM_ALIASES.")

    coverage = snapshots.coverage(con)
    if not coverage.empty:
        print("\nArchive coverage")
        print(coverage.to_string(index=False))

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
