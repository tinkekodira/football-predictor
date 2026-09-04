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

    # **Only the settling half needs to write.** DuckDB allows one writer, so
    # asking for a write connection to print a report means this refuses to run
    # whenever the app is open - which is exactly when somebody is most likely
    # to want the report. Reporting takes a read-only connection and coexists
    # with the app happily.
    try:
        con = database.connect(args.db, read_only=args.no_settle)
    except Exception as error:
        if "another process" not in str(error):
            raise
        print(
            "The database is open in another process, and settling needs to "
            "write to it.\n\nStop the app (`streamlit run app.py`) and run this "
            "again, or pass --no-settle to read the ledger without settling "
            "anything - that works alongside a running app."
        )
        return 1

    leagues = args.league or None

    if not args.no_settle:
        counts = ledger.settle_open(con)
        print(f"Settled {counts['settled']} of {counts['open']} open bet(s).")
        # **Three different reasons, and only the last is worth acting on.**
        # Reported as one "unmatched" count they are indistinguishable, and
        # somebody running this for a fortnight would see "0 settled" every
        # time with no way to tell a working ledger from a stale database from
        # a broken join.
        if counts["awaiting_kickoff"]:
            print(
                f"  {counts['awaiting_kickoff']} waiting on a kick-off. That "
                "is the normal state of a bet on a fixture in the future, and "
                "nothing needs doing."
            )
        if counts["awaiting_results"]:
            print(
                f"  {counts['awaiting_results']} played, but the result is not "
                "in the database yet. `matches` only advances when the current "
                "season is re-downloaded:\n"
                "      python scripts/build_database.py"
            )
        if counts["unmatched_unexpected"]:
            print(
                f"  {counts['unmatched_unexpected']} PLAYED AND STILL "
                "UNMATCHED. The database holds later matches for those "
                "leagues, so these should have joined and did not - usually a "
                "club name the two sources spell differently, or a "
                "postponement beyond the one-day tolerance. Worth looking at; "
                "run with --open to see them."
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

    report(con, leagues)

    if args.open:
        _list_open(con, leagues)

    con.close()
    return 0


def report(con, leagues) -> None:
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
