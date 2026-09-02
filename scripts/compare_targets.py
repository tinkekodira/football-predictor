"""Does fitting team strength on xG predict better than fitting it on goals?

    python scripts/compare_targets.py --league E0 --from 2018-08-01
    python scripts/compare_targets.py --league E0 --from 2018-08-01 --blend 0.3 0.5 0.7

One walk-forward per target, scored on the same matches. The comparison that
matters is **log loss against the market's**, not against the base rate: the
model already beats the base rate comfortably and still loses money, so "better
than nothing" is not the bar. The bar is whether the gap to the closing line
narrows.

Three things are reported per target, because they can move independently and
each says something different:

- **log loss** - does the model assign higher probability to what happened.
- **the gap to the market** - the same thing measured against the only
  opponent that counts.
- **calibration slope** - whether the probabilities are spread too far apart.
  A model fitted on a less noisy signal should, if the theory behind this
  change is right, be less over-confident about which fixtures are unusual, and
  this is where that would show.

CLV is reported last and deliberately quietly. It is the noisiest number here
and the one most likely to move by chance between two models that differ this
little; read the log loss first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402


def score(result, market: str) -> dict:
    predictions = result.predictions
    stats = evaluation.score_market(predictions, market)
    bets = result.bets
    clv = (
        evaluation.clustered_mean(bets["clv"], bets["match_id"])
        if not bets.empty else {"mean": float("nan"), "se": float("nan"), "n": 0}
    )
    frame = predictions[
        (predictions["market"] == market)
        & (predictions["push_fraction"] < 0.5)
        & predictions["model_conditional"].notna()
    ]
    side = frame[frame["selection"] == ("home" if market == "1x2" else "over")]
    slope = evaluation.calibration_slope(
        side["model_conditional"].to_numpy(),
        (side["win_fraction"] > 0.5).to_numpy(dtype=float),
    )
    return {
        "n": stats.get("n", 0),
        "model_log_loss": stats.get("model_log_loss", float("nan")),
        "market_log_loss": stats.get("market_log_loss", float("nan")),
        "log_loss_gap": stats.get("log_loss_gap", float("nan")),
        "slope": slope.get("slope", float("nan")),
        "clv": clv.get("mean", float("nan")),
        "clv_se": clv.get("se", float("nan")),
        "bets": clv.get("n", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--market", default="1x2", choices=list(backtest.BETTABLE_MARKETS))
    parser.add_argument("--blend", type=float, nargs="*", default=[0.5],
                        help="Blend weights to try (share given to xG)")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--ridge", type=float, nargs="*", default=None,
                        help="Shrinkage values to try against every target. A "
                             "less noisy target should tolerate less shrinkage, "
                             "so holding this fixed across targets is not a fair "
                             "comparison - see the calibration slope column.")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    if not con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'match_xg'"
    ).fetchone():
        print("No match_xg table. Run scripts/build_xg.py first.")
        return 1

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    from fbedge.models import base as model_base

    ridges = args.ridge or [model_base.DEFAULT_RIDGE]
    runs = [("goals", 0.0), ("xg", 1.0)] + [("blend", w) for w in args.blend]
    rows = []
    for ridge in ridges:
        for target, weight in runs:
            label = target if target != "blend" else f"blend {weight:g}"
            if len(ridges) > 1:
                label = f"{label} r{ridge:g}"
            settings = backtest.BacktestConfig(
                league=args.league, start=start, end=end, step_days=args.step_days,
                markets=(args.market,), fit_count_models=False,
                target=target, blend_weight=weight, ridge=ridge,
            )
            result = backtest.run_backtest(con, settings, verbose=False)
            row = {"target": label} | score(result, args.market)
            rows.append(row)
            print(f"  ran {label:<14} n={row['n']}")

    table = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print(f"  {config.LEAGUES[args.league]}  |  {start} to {end}  |  {args.market}")
    print("=" * 78)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = table.loc[table["model_log_loss"].idxmin()]
    baseline = table[table["target"].str.startswith("goals")].iloc[0]
    improvement = baseline["model_log_loss"] - best["model_log_loss"]
    print()
    print(f"  Lowest log loss here: {best['target']} ({best['model_log_loss']:.4f}), "
          f"{improvement:+.4f} against goals.")
    print()
    print("  **That is the winner of a search, not a measurement.** Picking the")
    print("  best of several targets on one league is the same uncorrected")
    print("  comparison the handoff warns about for leagues, and it has already")
    print("  bitten here: run across the top five, the winner is xG in one")
    print("  league, a blend in two, and plain goals in another, with a spread")
    print("  between them wider than any single league's improvement. The")
    print("  honest number is one rule applied to every league and averaged,")
    print("  not the best cell of a grid.")
    print()
    print("  gap is model minus market: negative would mean beating the close.")
    print("  slope above 1 means the probabilities are too tightly bunched;")
    print("  a target that is less noisy should be able to carry less")
    print("  shrinkage, so compare targets at their own best ridge, not at a")
    print("  shared one. Pass --ridge to check that.")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
