"""Tune the half-life and shrinkage by walk-forward validation.

    python scripts/tune_hyperparameters.py --league E0 --from 2019-08-01
    python scripts/tune_hyperparameters.py --league E0 --from 2019-08-01 --holdout 0.3

Two knobs matter: how fast old results stop counting (`half_life`) and how hard
team ratings are pulled towards the league average (`ridge`). The defaults are
conventional values from the literature. This finds better ones for your data.

**The search is scored on log loss, not on profit.** Over a few hundred bets,
return on investment is almost pure noise, so picking the grid point with the
best ROI reliably selects the luckiest setting rather than the best one. Log
loss uses every match rather than only the ones that cleared a betting
threshold, and it rewards being right about probabilities, which is the thing
the model is actually for.

**The window is split.** Everything is searched on the development portion,
then the single winning setting is re-run once on a later holdout it has never
influenced. If the holdout score is much worse than the development score, the
search found noise, and the honest move is to keep the defaults.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402
from fbedge.models import base  # noqa: E402
from fbedge import predict as predict_mod  # noqa: E402

DEFAULT_HALF_LIVES = (60.0, 90.0, 120.0, 180.0, 270.0, 360.0)
DEFAULT_RIDGES = (1.0, 2.0, 5.0, 10.0, 20.0)


def evaluate_setting(
    con, league: str, start: dt.date, end: dt.date, half_life: float, ridge: float,
    step_days: int, scoring_market: str, target: str = "goals",
    blend_weight: float = 0.5,
) -> dict:
    """One grid point: run the walk-forward and return its scores."""
    settings = backtest.BacktestConfig(
        league=league, start=start, end=end, step_days=step_days,
        half_life_days=half_life, ridge=ridge,
        markets=(scoring_market,), fit_count_models=False,
        target=target, blend_weight=blend_weight,
    )
    result = backtest.run_backtest(con, settings, verbose=False)
    score = evaluation.score_market(result.predictions, scoring_market)
    clv = evaluation.closing_line_value(result.bets)
    return {
        "half_life": half_life,
        "ridge": ridge,
        "n": score.get("n", 0),
        "log_loss": score.get("model_log_loss", float("nan")),
        "market_log_loss": score.get("market_log_loss", float("nan")),
        "log_loss_gap": score.get("log_loss_gap", float("nan")),
        "bets": clv.get("n", 0),
        "mean_clv": clv.get("mean_clv", float("nan")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--holdout", type=float, default=0.3,
                        help="Fraction of the window held back (default: %(default)s)")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--market", default="1x2",
                        choices=list(backtest.BETTABLE_MARKETS))
    parser.add_argument("--half-lives", type=float, nargs="+", default=list(DEFAULT_HALF_LIVES))
    parser.add_argument("--ridges", type=float, nargs="+", default=list(DEFAULT_RIDGES))
    parser.add_argument("--target", default="goals",
                        choices=["goals", "xg", "blend"],
                        help="What team strengths are fitted to (default: %(default)s)")
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1
    if not 0.0 <= args.holdout < 0.9:
        print("--holdout must be between 0 and 0.9.")
        return 1

    con = database.connect(args.db, read_only=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    total_days = (end - start).days
    split = start + dt.timedelta(days=int(total_days * (1.0 - args.holdout)))
    grid = list(itertools.product(args.half_lives, args.ridges))

    print(f"Tuning {config.LEAGUES[args.league]} on {args.market}, "
          f"strengths fitted to {args.target}")
    print(f"  development  {start} to {split}")
    print(f"  holdout      {split} to {end}")
    print(f"  grid         {len(grid)} settings\n")

    rows = []
    began = time.time()
    for index, (half_life, ridge) in enumerate(grid, start=1):
        predict_mod.clear_model_cache()
        try:
            row = evaluate_setting(
                con, args.league, start, split, half_life, ridge,
                args.step_days, args.market, args.target, args.blend_weight,
            )
        except ValueError as exc:
            print(f"  [{index}/{len(grid)}] skipped: {exc}")
            continue
        rows.append(row)
        print(f"  [{index}/{len(grid)}] half-life {half_life:>5.0f}d  ridge {ridge:>5.1f}  "
              f"log loss {row['log_loss']:.4f}  n={row['n']}")

    if not rows:
        print("Nothing could be evaluated. Widen the window or load more history.")
        return 1

    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    print(f"\nSearched {len(rows)} settings in {time.time() - began:.0f}s\n")
    print("Best ten by development log loss")
    print(table.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = table.iloc[0]
    spread = table["log_loss"].max() - table["log_loss"].min()
    print(f"\nBest development setting: half-life {best['half_life']:.0f}d, "
          f"ridge {best['ridge']:.1f} (log loss {best['log_loss']:.4f})")
    print(f"Spread across the whole grid: {spread:.4f}")
    if spread < 0.005:
        print("  That spread is small enough that the settings *searched here*")
        print("  are near enough interchangeable with each other.")
        searched_default = (
            (table["ridge"] == base.DEFAULT_RIDGE)
            & (table["half_life"] == base.DEFAULT_HALF_LIFE_DAYS)
        ).any()
        if searched_default and args.target == "goals":
            print("  The current default is among them, so keeping it is a")
            print("  reasonable choice.")
        else:
            # A narrow grid that excludes the default says nothing about the
            # default. Reading "these are all alike" as "so keep what ships"
            # is exactly wrong when what ships was never in the grid - which
            # is how a flat low-ridge sweep once appeared to endorse a ridge
            # five times larger than anything it tested.
            print("  That says nothing about the current default, which was")
            print(f"  {'not in this grid' if not searched_default else 'searched under a different target'}.")
            print("  The holdout comparison below is the one that speaks to it.")

    if args.holdout > 0 and (end - split).days > 30:
        print("\nRe-running the winner on the holdout window it never saw...")
        predict_mod.clear_model_cache()
        holdout = evaluate_setting(
            con, args.league, split, end, float(best["half_life"]),
            float(best["ridge"]), args.step_days, args.market,
            args.target, args.blend_weight,
        )
        predict_mod.clear_model_cache()
        defaults = evaluate_setting(
            con, args.league, split, end,
            base.DEFAULT_HALF_LIFE_DAYS, base.DEFAULT_RIDGE,
            args.step_days, args.market, "goals", args.blend_weight,
        )
        # The baseline is deliberately the *current production default* -
        # goals, stock half-life and ridge - not the same target at stock
        # settings. The question worth answering is whether to change what
        # ships, and that means comparing against what ships today.
        print(f"  tuned setting   log loss {holdout['log_loss']:.4f}  (n={holdout['n']})"
              f"   [{args.target}, half-life {best['half_life']:.0f}d, ridge {best['ridge']:.1f}]")
        print(f"  defaults        log loss {defaults['log_loss']:.4f}  (n={defaults['n']})"
              f"   [goals, half-life {base.DEFAULT_HALF_LIFE_DAYS:.0f}d, "
              f"ridge {base.DEFAULT_RIDGE:.1f}]")
        difference = holdout["log_loss"] - defaults["log_loss"]
        if difference < -0.002:
            print(f"  -> the tuned setting held up out of sample ({difference:+.4f}).")
        elif difference > 0.002:
            print(f"  -> the tuned setting was worse out of sample ({difference:+.4f}).")
            print("     The search fitted noise. Keep the defaults.")
        else:
            print("  -> no meaningful difference out of sample. Keep the defaults.")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nFull grid written to {args.csv}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
