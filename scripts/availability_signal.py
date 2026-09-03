"""Does knowing who is missing tell you anything the model does not already know?

    python scripts/availability_signal.py --league E0 --from 2018-08-01

**This is a measurement, not a feature.** Nothing here changes the model. The
question it answers is whether it is worth changing: if the share of a team's
recent minutes belonging to newly absent players has no relationship to how
that team then performs against the model's own expectation, there is nothing
to build and the honest move is to stop.

**The specification.** For each team in each match the walk-forward already
produces a fitted scoring rate. That rate is used as an offset, so the question
becomes multiplicative and the coefficients are interpretable directly:

    log E[goals] = log(model rate) + alpha
                   + beta_own * missing_share(this team)
                   + beta_opp * missing_share(opponent)

`beta_own` should come out negative if losing players costs a team goals, and
`beta_opp` positive, since a weakened defence concedes more. `alpha` absorbs
any overall bias in the model rates so that neither coefficient can pick it up
by accident. A one-eleventh missing share is roughly one regular starter, so
multiply a coefficient by 0.09 to read it as "the effect of losing one regular".

**Uncertainty is bootstrapped by match, not by row.** Every match contributes
two rows which share a fitted model and an opponent, so treating them as
independent would overstate precision by roughly the square root of two - the
same mistake `clustered_mean` exists to avoid for closing line value.

The availability features are built by `fbedge/availability.py`, which reads
only matches played strictly earlier. See that module for why that matters more
here than anywhere else in the project.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import availability, backtest, config, database  # noqa: E402


def fit_poisson(goals, log_rate, own, opp):
    """Poisson MLE for (alpha, beta_own, beta_opp) with `log_rate` as offset."""

    def negative_log_likelihood(theta):
        eta = log_rate + theta[0] + theta[1] * own + theta[2] * opp
        eta = np.clip(eta, -10.0, 4.0)
        return -float(np.sum(goals * eta - np.exp(eta)))

    result = optimize.minimize(
        negative_log_likelihood, np.zeros(3), method="BFGS"
    )
    return result.x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--feature", nargs="+",
                        default=["missing_xgchain_share",
                                 "missing_starter_share", "missing_share"],
                        choices=["missing_xgchain_share",
                                 "missing_starter_share", "missing_share"],
                        help="Three weightings of the same absence. "
                             "missing_xgchain_share weights by attacking "
                             "involvement, which is the right currency for a "
                             "question about scoring; missing_starter_share by "
                             "minutes among regulars; missing_share by minutes "
                             "among everyone, which is dominated by rotation.")
    parser.add_argument("--absence-window", type=int, nargs="+", default=[2, 1],
                        help="How many recent matches an absence is read from. "
                             "Several are tried because a null result on one "
                             "arbitrary choice would not distinguish no effect "
                             "from a badly shaped proxy.")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    if not con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'match_lineups'"
    ).fetchone():
        print("No match_lineups table. Run scripts/build_rosters.py first.")
        return 1

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    print(f"Refitting {config.LEAGUES[args.league]} week by week...")
    settings = backtest.BacktestConfig(
        league=args.league, start=start, end=end, step_days=args.step_days,
        markets=("1x2",), fit_count_models=False,
    )
    result = backtest.run_backtest(con, settings, verbose=False)
    totals = result.match_totals
    if totals.empty:
        print("Nothing was scored.")
        return 1

    lineups = con.execute("SELECT * FROM match_lineups").df()
    matches = con.execute(
        "SELECT match_id, date, home_team, away_team FROM matches WHERE league = ?",
        [args.league],
    ).df()
    print()
    print("=" * 78)
    print(f"  {config.LEAGUES[args.league]}  |  {start} to {end}")
    print("=" * 78)
    for window in args.absence_window:
        for feature in args.feature:
            _report_window(totals, lineups, matches, window, feature, args)
    con.close()
    return 0


def _report_window(totals, lineups, matches, absence_window, feature, args) -> None:
    """Fit and print the model for one feature and one absence window."""
    features = availability.availability_features(
        lineups, matches, absence_window=absence_window
    )
    wide = availability.attach_to_fixtures(features)

    home_col, away_col = f"{feature}_home", f"{feature}_away"
    frame = totals.merge(wide, on="match_id", how="inner")
    frame = frame.dropna(
        subset=[home_col, away_col, "model_home_rate", "model_away_rate"]
    )
    if frame.empty:
        print("  Nothing left after joining availability onto the backtest.")
        return

    # One row per team per match: the attacking side, its own missing share and
    # its opponent's.
    long = pd.concat(
        [
            pd.DataFrame({
                "match_id": frame["match_id"],
                "goals": frame["observed_home"].astype(float),
                "rate": frame["model_home_rate"].astype(float),
                "own": frame[home_col].astype(float),
                "opp": frame[away_col].astype(float),
            }),
            pd.DataFrame({
                "match_id": frame["match_id"],
                "goals": frame["observed_away"].astype(float),
                "rate": frame["model_away_rate"].astype(float),
                "own": frame[away_col].astype(float),
                "opp": frame[home_col].astype(float),
            }),
        ],
        ignore_index=True,
    )
    long = long[long["rate"] > 0]
    log_rate = np.log(long["rate"].to_numpy())
    goals = long["goals"].to_numpy()
    own = long["own"].to_numpy()
    opp = long["opp"].to_numpy()

    alpha, beta_own, beta_opp = fit_poisson(goals, log_rate, own, opp)

    rng = np.random.default_rng(args.seed)
    match_ids = long["match_id"].to_numpy()
    unique = np.unique(match_ids)
    index_of = {m: np.flatnonzero(match_ids == m) for m in unique}
    draws = np.empty((args.bootstrap, 3))
    for i in range(args.bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_of[m] for m in chosen])
        draws[i] = fit_poisson(goals[rows], log_rate[rows], own[rows], opp[rows])

    print()
    print("-" * 78)
    print(f"  {feature}, absence window {absence_window} match"
          f"{'es' if absence_window != 1 else ''}: "
          f"{len(frame)} matches, {len(long)} team-matches, "
          f"{args.bootstrap} bootstrap resamples")
    print("-" * 78)
    scale = "of the team's recent attacking involvement"         if feature == "missing_xgchain_share" else         f"of its minutes, so roughly {long['own'].mean() * 11:.1f} regulars"
    print(f"  mean {feature} {long['own'].mean():.4f}   {scale}")
    print(f"  share of team-matches with nobody newly missing: "
          f"{(long['own'] == 0).mean():.1%}")
    print()

    names = ["intercept", "beta_own", "beta_opp"]
    values = [alpha, beta_own, beta_opp]
    print(f"  {'':<12}{'estimate':>10}{'boot SE':>10}{'z':>8}   effect of one regular out")
    for name, value, column in zip(names, values, draws.T):
        se = float(column.std(ddof=1))
        z = value / se if se > 0 else float("nan")
        effect = ""
        if name != "intercept":
            effect = f"{(np.exp(value / 11) - 1) * 100:+.2f}% on the rate"
        print(f"  {name:<12}{value:>10.4f}{se:>10.4f}{z:>8.2f}   {effect}")

    own_se = float(draws[:, 1].std(ddof=1))
    own_z = beta_own / own_se if own_se else float("nan")

    # What this run could have found. A null is only worth acting on if the
    # test had the power to detect an effect worth caring about, which is the
    # same standard the distribution-shape work was held to. One regular
    # starter is about one eleventh of the minutes, so the smallest effect
    # separable from noise at two standard errors is stated in those terms.
    detectable = (np.exp(2 * own_se / 11) - 1) * 100
    print()
    print(f"  Smallest effect this run could separate from noise (2 SE): "
          f"{detectable:.2f}% on the rate per regular out.")
    print()
    if own_z < -2 and beta_own < 0:
        print("  -> Teams missing players do score less than the model expects,")
        print("     by more than sampling noise. There is something here worth")
        print("     feeding into the fit.")
    elif own_z > 2:
        print("  -> The sign is backwards: teams missing players score MORE than")
        print("     expected. Suspect the feature before believing the finding;")
        print("     rotation before an easy fixture would look like this.")
    else:
        print("  -> No effect distinguishable from noise. Read that against the")
        print("     detectable size above: a null here rules out an effect")
        print("     larger than that, and says nothing about smaller ones. The")
        print("     likeliest explanations are that the model already prices")
        print("     what this knows, or that the proxy is too blunt - Understat")
        print("     lists only players who appeared, so an unused substitute and")
        print("     an injured first choice look identical.")
    print()
    print("  Read the size, not only the sign. Anything under about 1% on the")
    print("  rate is smaller than the blend/shrinkage change already shipped,")
    print("  and would not be worth thousands of requests per league to keep")
    print("  current.")


if __name__ == "__main__":
    raise SystemExit(main())
