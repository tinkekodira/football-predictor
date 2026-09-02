"""Walk-forward backtest.

    python scripts/backtest.py --league E0 --from 2022-08-01
    python scripts/backtest.py --league E0 --from 2022-08-01 --half-life 120 --ridge 10
    python scripts/backtest.py --league SP1 --from 2020-08-01 --csv bets.csv

At each step the models are fitted on matches played before that week, then
used to price every selection the bookmaker actually offered, then settled
against the result. Nothing is ever fitted on a match it later predicts.

Read the output in this order:

  1. **Calibration.** If the model is not calibrated, nothing below it means
     anything.
  2. **Model versus market log loss.** The cleanest test of whether the model
     knows something the closing line does not.
  3. **Closing line value.** Whether the prices taken beat the close. This is
     measurable over hundreds of bets; profit is not.
  4. **Simulated returns**, with a confidence interval. The interval is
     normally wide enough to contain both a good profit and a bad loss, and
     that width is the finding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402
from fbedge.models import base  # noqa: E402


def print_calibration(predictions: pd.DataFrame, market: str) -> None:
    frame = predictions[
        (predictions["market"] == market) & (predictions["push_fraction"] < 0.5)
    ]
    if len(frame) < 50:
        return
    column = "model_conditional" if "model_conditional" in frame.columns else "model_probability"
    frame = frame[frame[column].notna()]
    table = evaluation.calibration_table(
        frame[column].to_numpy(),
        (frame["win_fraction"] > 0.5).astype(float).to_numpy(),
        bins=10,
    )
    if table.empty:
        return
    print(f"\n  Calibration, {market}")
    print("  " + table.to_string(index=False).replace("\n", "\n  "))


def report(result: backtest.BacktestResult) -> None:
    predictions = result.predictions
    if predictions.empty:
        print("No selections were priced. Check that the odds table is populated.")
        return

    print(f"\n{'=' * 74}")
    print(f"  {config.LEAGUES[result.config.league]}  |  "
          f"{result.config.start} to {result.config.end}")
    print(f"  half-life {result.config.half_life_days:g}d, "
          f"shrinkage {result.config.ridge:g}, "
          f"edge threshold {result.config.edge_threshold:.1%}")
    print("=" * 74)
    print(f"  {result.refits} refits, {result.matches} matches, "
          f"{len(predictions)} selections priced")

    print("\n--- Probability quality " + "-" * 50)
    rows = []
    for market in sorted(predictions["market"].unique()):
        score = evaluation.score_market(predictions, market)
        if score.get("n", 0) < 30:
            continue
        rows.append(score)
    if rows:
        frame = pd.DataFrame(rows)
        columns = [c for c in (
            "market", "n", "model_log_loss", "base_rate_log_loss",
            "n_priced", "model_log_loss_priced", "market_log_loss", "log_loss_gap",
        ) if c in frame.columns]
        print(frame[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if "log_loss_gap" in frame.columns:
            print("\n  log_loss_gap is model minus market: negative means the model")
            print("  beat the closing line, positive means the market did.")

    for market in ("1x2", "total_goals"):
        print_calibration(predictions, market)

    bets = result.bets
    print(f"\n--- Betting {'-' * 62}")
    if bets.empty:
        print("  No selection cleared the edge threshold. That is a normal and")
        print("  entirely respectable outcome: it means the model found nothing")
        print("  it believed was mispriced.")
        return

    clv = evaluation.closing_line_value(bets)
    print(f"  bets placed         {len(bets)}")
    print(f"  with a closing line {clv.get('n', 0)}")
    if clv.get("n"):
        print(f"  mean CLV            {clv['mean_clv']:+.3%} "
              f"(standard error {clv['clv_standard_error']:.3%})")
        print(f"  beat the close      {clv['beat_close_rate']:.1%} of the time")
        if "mean_price_movement" in clv:
            print(f"  same-book movement  {clv['mean_price_movement']:+.3%} "
                  f"(on {clv['n_price_movement']} bets)")
        if clv["mean_clv"] > 2 * clv["clv_standard_error"]:
            print("  -> positive closing line value, which is the strongest evidence")
            print("     available that this is a real edge rather than variance.")
        elif clv["mean_clv"] < -2 * clv["clv_standard_error"]:
            print("  -> negative closing line value. Any profit shown below is luck")
            print("     and will not survive. This is the usual first result.")
        else:
            print("  -> closing line value is indistinguishable from zero, so far.")
    else:
        print("  -> no closing prices for these selections, so CLV cannot be")
        print("     measured. This source carries closing 1X2 prices much further")
        print("     back than closing totals or handicaps.")

    ledger = evaluation.simulate_staking(bets, method="flat")
    summary = evaluation.staking_summary(ledger)
    interval = evaluation.bootstrap_roi(bets)
    print(f"\n  flat staking        {summary['bets']} bets, "
          f"turnover {summary['turnover']:.0f}")
    print(f"  profit              {summary['profit']:+.1f} units")
    print(f"  ROI                 {summary['roi']:+.2%}")
    if interval.get("n"):
        print(f"  95% interval        {interval['roi_low']:+.2%} to "
              f"{interval['roi_high']:+.2%}")
        print(f"  chance of profit    {interval['probability_profitable']:.0%} "
              "(resampling whole matchdays)")
    print(f"  worst drawdown      {summary['max_drawdown']:.1%} "
          f"({summary.get('drawdown_basis', 'of turnover staked so far')})")

    books = evaluation.bookmaker_breakdown(result.predictions, min_selections=50)
    if not books.empty:
        print("\n  Where the prices are beatable (every book's own price, all")
        print("  priced selections, not just the ones bet)")
        print("  " + books.to_string(float_format=lambda v: f"{v:+.4f}").replace("\n", "\n  "))
        print("\n  mean_clv is roughly minus that book's margin when the model has no")
        print("  edge. A book near the top is either sharp or genuinely beatable. To")
        print("  actually bet through one book, rerun with --book <name>.")

    by_market = bets.groupby("market").agg(
        bets=("profit_at_taken", "size"),
        roi=("profit_at_taken", "mean"),
        mean_clv=("clv", "mean"),
    )
    print("\n  By market")
    print("  " + by_market.to_string(float_format=lambda v: f"{v:+.4f}").replace("\n", "\n  "))

    if interval.get("n") and interval["roi_low"] < 0 < interval["roi_high"]:
        print("\n  The interval spans zero, so this backtest cannot distinguish")
        print("  the model from a coin flip. More data or a better model, not a")
        print("  bigger stake.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--half-life", type=float, default=base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--ridge", type=float, default=base.DEFAULT_RIDGE)
    parser.add_argument("--edge", type=float, default=0.02,
                        help="Minimum expected value to place a bet (default: %(default)s)")
    parser.add_argument("--margin-method", choices=("multiplicative", "additive"),
                        default="multiplicative")
    parser.add_argument("--markets", nargs="+", default=list(backtest.BETTABLE_MARKETS),
                        choices=list(backtest.BETTABLE_MARKETS))
    parser.add_argument("--book", default=None,
                        help="Only take prices from this bookmaker, e.g. bet365. "
                             "Defaults to the best price available anywhere.")
    parser.add_argument("--margins", action="store_true",
                        help="Print each bookmaker's average overround and exit.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write every priced selection to a CSV file")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)

    if args.margins:
        margins = evaluation.market_margins(con, args.league)
        print(f"Average 1X2 overround, {config.LEAGUES[args.league]}")
        print(margins.to_string(index=False))
        print("\nA sharp book sits near 1.02, a soft one nearer 1.07. The gap is what")
        print("you pay for betting somewhere convenient, and it is usually larger than")
        print("any edge a model like this will find.")
        con.close()
        return 0

    settings = backtest.BacktestConfig(
        league=args.league,
        start=dt.date.fromisoformat(args.start),
        end=dt.date.fromisoformat(args.end) if args.end else dt.date.today(),
        step_days=args.step_days,
        half_life_days=args.half_life,
        ridge=args.ridge,
        markets=tuple(args.markets),
        margin_method=args.margin_method,
        edge_threshold=args.edge,
        price_source=(args.book,) if args.book else backtest.DEFAULT_PRICE_SOURCE,
    )
    result = backtest.run_backtest(con, settings)
    report(result)

    if args.csv:
        # book_prices is a dict per row, kept for in-session analysis; drop it
        # for the CSV so every cell stays plain text.
        result.predictions.drop(columns=["book_prices"], errors="ignore").to_csv(
            args.csv, index=False
        )
        print(f"\nSelections written to {args.csv}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
