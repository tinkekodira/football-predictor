"""Settle the paper-trading ledger and report what it says so far.

    python scripts/paper_trade.py                  # settle, then report
    python scripts/paper_trade.py --no-settle      # report what is already settled
    python scripts/paper_trade.py --league E0

Claims are put into the ledger by `scripts/scan_fixtures.py --record`. This is
the other half: it joins every open bet to the match that has since been
played, settles it, computes closing line value against the margin-free closing
price, and prints the running record.

**Expect it to say nothing for weeks, and read that as working.** The gate this
ledger exists to serve asks for several weeks of forward claims judged on the
closing line value from that period. A ledger a few days old has a handful of
settled bets, and a mean over a handful of bets is noise whatever the
arithmetic reports. The standard error is printed next to every mean for that
reason, clustered by match, and the summary refuses to editorialise about a
sample this thin.

**Closing line value is the headline; profit is not.** Over a few hundred bets
return on investment is close to noise while CLV is measurable in weeks. That
is the project's own standing rule and it is why the gate is phrased in terms
of CLV. Profit is printed below it as a subordinate check, never above it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from fbedge import config, database, ledger  # noqa: E402

# Below this many settled bets the report declines to characterise the mean at
# all. Not a significance threshold - it is the point below which a reader
# should not be shown a number that invites one.
TOO_THIN_TO_READ = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), action="append")
    parser.add_argument("--no-settle", action="store_true",
                        help="Report only. Settles nothing, so a bet whose "
                             "match has been played stays open.")
    parser.add_argument("--open", action="store_true",
                        help="List the bets still waiting on a result")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db)
    leagues = args.league or None

    if not args.no_settle:
        counts = ledger.settle_open(con)
        print(f"Settled {counts['settled']} of {counts['open']} open bet(s).")
        if counts["unmatched"]:
            print(
                f"  {counts['unmatched']} still open: no played match to join "
                "to yet. That is the normal state of a bet on a fixture that "
                "has not been played."
            )
        if counts["unsettleable"]:
            print(
                f"  {counts['unsettleable']} matched a played fixture but "
                "could not be settled - a market with no result recorded, or "
                "one withdrawn since the claim was filed. Left open."
            )
        if counts["no_closing_price"]:
            print(
                f"  {counts['no_closing_price']} settled with no closing "
                "price, so they carry a result but no closing line value. "
                "This source publishes closing prices for some markets long "
                "before others."
            )
        print()

    _report(con, leagues)

    if args.open:
        _list_open(con, leagues)

    con.close()
    return 0


def _report(con, leagues) -> None:
    summary = ledger.summary(con, leagues=leagues)
    width = 78
    print("=" * width)
    print("  Paper-trading ledger")
    print("=" * width)

    if not summary["bets"]:
        print(
            "  Empty. Record a board with:\n"
            "      python scripts/scan_fixtures.py --record"
        )
        return

    print(
        f"  {summary['bets']} claim(s) on file, "
        f"{summary['distinct_selections']} distinct selection(s)."
    )
    print(
        f"  {summary['staked']} ranked and treated as staked, "
        f"{summary['withheld']} withheld and recorded unstaked."
    )
    print(f"  {summary['settled']} settled, {summary['open']} still open.")
    print(f"  First recorded {summary['first_recorded']}, "
          f"last {summary['last_recorded']}.")
    print()

    if not summary.get("n_clv"):
        print("  No settled bet carries a closing price yet, so there is no")
        print("  closing line value to report. That is the expected state of a")
        print("  ledger younger than the fixtures it has claims on.")
        return

    print("  Closing line value - the headline measure")
    print("  " + "-" * (width - 4))
    mean, se = summary["mean_clv"], summary["clv_se"]
    print(f"  mean CLV            {mean:+.3%}")
    if pd.notna(se) and se > 0:
        print(f"  clustered SE        {se:.3%}   ({mean / se:+.1f} SE)")
    print(f"  bets / matches      {summary['n_clv']} / {summary['n_matches']}")
    print(f"  beat the close      {summary['beat_close_rate']:.1%} of the time")
    print()

    if summary["n_clv"] < TOO_THIN_TO_READ:
        print(
            f"  **{summary['n_clv']} bets is too few to read.** The mean above is\n"
            "  reported because hiding it would be worse, not because it means\n"
            "  anything yet. The backtested figure this is eventually meant to\n"
            "  test is -1.500% over 4,379 bets."
        )
    else:
        print(
            "  For comparison, the backtest over nine seasons of E0 measured\n"
            "  -1.500% (-9.1 SE). This ledger is the forward test of that, and\n"
            "  the gate needs CLV indistinguishable from zero to open."
        )
    print()

    if summary.get("n_profit"):
        print("  Profit - the subordinate check, and it is noise at this size")
        print("  " + "-" * (width - 4))
        print(f"  staked              {summary['total_staked']:.0f} unit(s), flat")
        print(f"  profit              {summary['profit']:+.2f} unit(s)")
        print(f"  ROI                 {summary['roi']:+.2%}")
        print()

    comparison = ledger.withheld_comparison(con, leagues=leagues)
    if not comparison.empty and comparison["n"].sum():
        print("  Ranked against withheld - BACKLOG B17's own homework")
        print("  " + "-" * (width - 4))
        print(f"  {'group':<12}{'n':>6}{'mean EV':>10}{'mean CLV':>11}"
              f"{'SE':>9}{'ROI':>9}")
        for row in comparison.itertuples():
            print(
                f"  {row.group:<12}{row.n:>6}{_pct(row.mean_ev):>10}"
                f"{_pct(row.mean_clv):>11}{_pct(row.clv_se):>9}"
                f"{_pct(row.roi):>9}"
            )
        print(
            "\n  The withheld rows are recorded and settled precisely so this\n"
            "  comparison exists. It needs hundreds of bets in both groups\n"
            "  before it says anything; until then it is a shape, not a result."
        )
    print()
    print(
        "  Nothing here has been staked and nothing here is a recommendation.\n"
        "  A positive expected value means the model disagrees with the price,\n"
        "  and this model's record is that when it disagrees it is usually the\n"
        "  one that is wrong."
    )


def _pct(value) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:+.2%}"


def _list_open(con, leagues) -> None:
    frame = ledger.load_bets(con, leagues=leagues, settled=False)
    print()
    print("  Open bets, waiting on a result")
    print("  " + "-" * 74)
    if frame.empty:
        print("  None.")
        return
    for row in frame.head(50).itertuples():
        flag = "" if row.staked else "  [withheld]"
        print(
            f"  {str(row.fixture_date)[:10]}  {row.league:<4}"
            f"{row.home_team[:14]:<15} v {row.away_team[:14]:<15}"
            f"{row.market:<16}{str(row.selection_label)[:12]:<13}"
            f"{row.price_taken:>6.2f}{row.expected_value:>+8.1%}{flag}"
        )
    if len(frame) > 50:
        print(f"  ... and {len(frame) - 50} more.")


if __name__ == "__main__":
    raise SystemExit(main())
