"""Score every market against history, so no price is ever shown unlabelled.

    python scripts/build_evidence.py                       # all five leagues
    python scripts/build_evidence.py --league E0 --from 2022-08-01

**Run this before the scan, and re-run it after a model change.** The scan, the
command line and the app all print a market's track record next to its price,
and they read it from the table this writes. Without it every market reports
"UNTESTED", which is honest but not useful.

It takes a few minutes per league, because it is a real walk-forward: refit
weekly, price the bookmaker's own lines, settle, and separately price the
model's own lines for the markets nobody quotes so their calibration can be
measured too. That cost is why the result is stored rather than computed when
somebody opens a page.

**The two halves of the output are not the same claim.** A market with a CLV
figure was settled against prices somebody actually offered. A market with only
a calibration slope was checked against what happened, which says the model is
about right and says nothing at all about whether there is money in it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from fbedge import backtest, config, database, evidence  # noqa: E402
from fbedge.models import base  # noqa: E402

# Four seasons: long enough for a thin market to accumulate a usable sample,
# short enough that the evidence describes the market as it is now rather than
# as it was before the 2024 benchmark change. See BACKLOG B1 and B10.
DEFAULT_WINDOW_SEASONS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), action="append",
                        help="Repeatable. Defaults to all five.")
    parser.add_argument("--from", dest="start", default=None,
                        help=f"ISO date. Defaults to {DEFAULT_WINDOW_SEASONS} seasons back.")
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--half-life", type=float, default=base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    leagues = args.league or list(config.LEAGUES)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    start = (
        dt.date.fromisoformat(args.start) if args.start
        else dt.date(end.year - DEFAULT_WINDOW_SEASONS, 8, 1)
    )

    # Read-only for the fits, and the write connection is opened afterwards for
    # the insert alone. DuckDB allows one writer, and holding the lock across
    # several minutes of refitting locks out the app and every other script -
    # the defect BACKLOG B2 records twice.
    con = database.connect(args.db, read_only=True)
    frames = []
    for league in leagues:
        started = time.time()
        print(f"{league} {config.LEAGUES[league]}: {start} to {end} ...", flush=True)
        try:
            frame = evidence.compute(
                con, league, start, end,
                half_life_days=args.half_life, ridge=args.ridge,
            )
        except ValueError as exc:
            print(f"  skipped: {exc}")
            continue
        frames.append(frame)
        print(f"  {len(frame)} markets scored in {time.time() - started:.0f}s")
    con.close()

    if not frames:
        print("Nothing was scored.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    write_con = database.connect(args.db)
    evidence.write(write_con, combined)
    write_con.close()
    print(f"\nWrote {len(combined)} rows to `{evidence.TABLE_NAME}`.")

    report = combined[
        ["league", "market", "status", "n", "calibration_slope", "n_bets", "mean_clv"]
    ].copy()
    report["mean_clv"] = (report["mean_clv"] * 100).round(2)
    report["calibration_slope"] = report["calibration_slope"].round(3)
    print()
    print(report.to_string(index=False))

    backtested = combined[combined["status"] == evidence.BACKTESTED]
    print(
        f"\n{len(backtested)} of {len(combined)} market-league pairs are backtested "
        "as bets. The rest are calibration only: the model can be checked "
        "against what happened, but no price ever existed to bet into, so "
        "nothing there is evidence of an edge."
    )
    if not backtested.empty:
        mean = float(backtested["mean_clv"].mean())
        print(
            f"Mean CLV across the backtested markets is {mean * 100:+.2f}%. "
            "Negative is the expected result and the honest one: this model "
            "has never beaten the closing line. Read `HANDOFF.md` before "
            "acting on any scan output."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
