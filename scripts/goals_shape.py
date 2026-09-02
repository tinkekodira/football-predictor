"""Is the goals distribution the right shape, and is the totals table readable?

    python scripts/goals_shape.py --league E0 --from 2017-08-01
    python scripts/goals_shape.py --league E0 --from 2017-08-01 --no-handicap

This exists to settle a specific claim. The previous handoff recorded a
`total_goals` calibration table with two large adjacent bands whose gaps ran in
opposite directions - +0.079 on n=467 and -0.048 on n=1006 - and reasoned that
pooling several over/under lines into one table was hiding the cause, since
over 2.5 and over 3.5 would land in different bands.

**The source quotes exactly one totals line.** football-data.co.uk carries
over/under 2.5 and nothing else, in every league in the database. So that
explanation cannot be the right one, and splitting by line changes nothing for
this market. What the pooled table really does is mix the two sides: over and
under at one line are exact complements, so the 0.30-0.40 band holds the overs
from defensive fixtures *and* the unders from open ones. Two different kinds of
fixture in one row, and every match counted twice. The sign flip is a property
of that mixing, not of the model.

The section below prints the pooled table and the one-sided table together, so
the artifact is demonstrated rather than asserted.

The structural question survives, though, and it is the more interesting one.
`models/counts.py` uses a negative binomial for corners and cards because a
Poisson is too narrow for a real count: it is overconfident about the middle and
underprices the tails, which is where over/under lines sit. `score_matrix_from
_rates` builds goals from two independent Poissons with a four-cell Dixon-Coles
correction, and nobody ever checked whether goals need the same treatment. That
is testable directly against the fitted distributions, and the last section
does it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402


def _print_totals_calibration(predictions) -> None:
    """The pooled table and the one-sided table, side by side."""
    totals = predictions[predictions["market"] == "total_goals"]
    if totals.empty:
        print("  No total_goals selections were priced.")
        return

    lines = sorted(totals["line"].dropna().unique().tolist())
    print(f"  Lines quoted by the source: {lines}")
    if len(lines) == 1:
        print("  One line only, so 'split by line' is a no-op for this market.")
        print("  The mixing below is the two sides, not two lines.")

    pooled = evaluation.calibration_table(
        totals["model_conditional"].to_numpy(),
        (totals["win_fraction"] > 0.5).to_numpy(dtype=float),
    )
    print("\n  Pooled over both sides (the misleading table):")
    print(_indent(pooled.to_string(index=False, float_format=_fmt)))

    one_sided = evaluation.calibration_by_line(
        predictions, market="total_goals", side="over"
    )
    print("\n  Over side only, split by line (the readable table):")
    if one_sided.empty:
        print("    Not enough selections in any band.")
    else:
        print(_indent(one_sided.to_string(index=False, float_format=_fmt)))

    overs = totals[totals["selection"] == "over"]
    outcomes = (overs["win_fraction"] > 0.5).to_numpy(dtype=float)
    slope = evaluation.calibration_slope(
        overs["model_conditional"].to_numpy(), outcomes
    )
    if "slope" not in slope:
        return

    print(f"\n  Calibration slope {slope['slope']:.3f} "
          f"(SE {slope['slope_se']:.3f}, {slope['slope_z']:+.2f} SE from 1.0), "
          f"intercept {slope['intercept']:+.3f}, n={slope['n']}")

    # The control that makes the number above worth reading. The closing line
    # is a forecaster known to be well calibrated, scored on the same matches
    # by the same code, so if it does not come back near 1.0 the finding is
    # about the method rather than about the model.
    priced = overs[overs["market_probability"].notna()]
    if len(priced) >= 30:
        priced_outcomes = (priced["win_fraction"] > 0.5).to_numpy(dtype=float)
        control = evaluation.calibration_slope(
            priced["market_probability"].to_numpy(), priced_outcomes
        )
        if "slope" in control:
            print(f"  Market's own slope on the same matches "
                  f"{control['slope']:.3f} (SE {control['slope_se']:.3f}, "
                  f"{control['slope_z']:+.2f} SE) - the control.")

    if slope["slope_z"] < -2:
        print("  -> Below one: the probabilities are spread too far apart.")
        print("     The model separates high- from low-scoring fixtures more")
        print("     confidently than the results justify. That is a shrinkage")
        print("     problem, and it is what the sign flip above really is.")
    elif slope["slope_z"] > 2:
        print("  -> Above one: the model is hedging toward the base rate and")
        print("     could afford to separate fixtures more aggressively.")
    else:
        print("  -> Indistinguishable from one; no systematic over- or")
        print("     under-spreading of the probabilities.")

    _print_information_content(overs, outcomes)


def _print_information_content(overs, outcomes) -> None:
    """How much the model knows about totals at all, against two benchmarks.

    Printed next to the slope on purpose. A slope well below one can mean the
    model is badly calibrated, or it can mean the model has little real signal
    and most of the spread in its probabilities is noise. The distance from the
    base rate separates those two readings, and they call for very different
    responses.
    """
    priced = overs[overs["market_probability"].notna()]
    if len(priced) < 30:
        return
    priced_outcomes = (priced["win_fraction"] > 0.5).to_numpy(dtype=float)
    model = evaluation.binary_log_loss(
        priced["model_conditional"].to_numpy(), priced_outcomes
    )
    market = evaluation.binary_log_loss(
        priced["market_probability"].to_numpy(), priced_outcomes
    )
    base = evaluation.binary_log_loss(
        np.full_like(priced_outcomes, priced_outcomes.mean()), priced_outcomes
    )
    print(f"\n  Log loss on over 2.5   model {model:.4f}   "
          f"market {market:.4f}   base rate {base:.4f}")
    print(f"  Model beats the base rate by {base - model:+.4f}; "
          f"the market beats it by {base - market:+.4f}.")


def _print_handicap_calibration(predictions) -> None:
    table = evaluation.calibration_by_line(
        predictions, market="asian_handicap", side="home"
    )
    if table.empty:
        print("  Not enough handicap selections in any band.")
        return
    print("  Home side only, split by line. Unlike totals, this market really")
    print("  is quoted at many lines, so this split does something.")
    print(_indent(table.to_string(index=False, float_format=_fmt)))


def _print_shape(fit: dict) -> None:
    if not fit.get("n"):
        print("  No matches to test.")
        return

    print(f"  Matches: {fit['n']}")
    print(f"  Mean total   observed {fit['observed_mean']:.3f}   "
          f"predicted {fit['predicted_mean']:.3f}")
    print(f"  Variance     observed {fit['observed_variance']:.3f}   "
          f"predicted {fit['predicted_variance']:.3f}")
    print(f"               of which within-match {fit['within_match_variance']:.3f}, "
          f"between-match {fit['between_match_variance']:.3f}")
    print("               (within is the Poisson spread a dispersion parameter")
    print("                would change; between is how far the fitted rates")
    print("                move between fixtures, which shrinkage controls)")
    print()
    print(f"  Dispersion ratio {fit['dispersion']:.4f} "
          f"(SE {fit['dispersion_se']:.4f}, {fit['dispersion_z']:+.2f} SE from 1.0)")
    print()
    if fit["dispersion_z"] > 2:
        print("  -> Realised totals scatter wider than the model expects. That is")
        print("     the corners-and-cards failure, and a dispersion parameter on")
        print("     goals is worth trying.")
    elif fit["dispersion_z"] < -2:
        print("  -> Realised totals scatter *less* widely than the model expects.")
        print("     A negative binomial would widen the distribution and make this")
        print("     worse, not better. Do not add one.")
    else:
        print("  -> Indistinguishable from a correctly dispersed distribution.")
        print("     Goals are not corners: the Poisson spread is about right, and")
        print("     switching goals to a negative binomial is not indicated.")

    print("\n  Observed versus predicted frequency of each total:")
    print(_indent(fit["buckets"].to_string(index=False, float_format=_fmt)))
    print()
    print("  se is Poisson-binomial: matches have different rates, so the")
    print("  variance of a bucket count is the sum of p(1-p), not n*p*(1-p).")
    print("  Overdispersion looks like a hollow middle and heavy tails on both")
    print("  sides. A drift running one way across the whole table is a")
    print("  misplaced mean instead, which is a different repair.")


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-bucket", type=int, default=6,
                        help="Totals above this are pooled into one tail bucket")
    parser.add_argument("--no-handicap", action="store_true",
                        help="Skip the handicap calibration section")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    wanted = ("total_goals",) if args.no_handicap else ("total_goals", "asian_handicap")
    settings = backtest.BacktestConfig(
        league=args.league, start=start, end=end, step_days=args.step_days,
        markets=wanted, fit_count_models=False,
    )
    result = backtest.run_backtest(con, settings, verbose=True)
    if result.match_totals.empty:
        print("Nothing was scored. Widen the window or check the database.")
        return 1

    print()
    print("=" * 74)
    print(f"  {config.LEAGUES[args.league]}  |  {start} to {end}")
    print("=" * 74)

    print("\n--- Totals calibration: pooled versus one-sided "
          "--------------------------")
    _print_totals_calibration(result.predictions)

    if not args.no_handicap:
        print("\n--- Handicap calibration by line "
              "-----------------------------------------")
        _print_handicap_calibration(result.predictions)

    print("\n--- Shape of the goals distribution "
          "--------------------------------------")
    fit = evaluation.goal_total_fit(result.match_totals, max_bucket=args.max_bucket)
    _print_shape(fit)

    if args.csv:
        fit["buckets"].to_csv(args.csv, index=False)
        print(f"\nBucket table written to {args.csv}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
