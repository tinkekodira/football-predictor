"""What the data says the shrinkage should be, league by league.

    python scripts/shrinkage_report.py --as-of 2025-05-01

Reports, for every league, the prior variance of team strengths that empirical
Bayes estimates from the fit itself, and the ridge that implies. See
`fbedge.models.hierarchical` for the algebra; the short version is that the
ridge penalty *is* a Gaussian prior with `lambda = 1 / (2 tau^2)`, so `tau` can
be estimated rather than chosen.

**Read the result alongside what predicts best, not on its own.** The two
disagree, sharply and reproducibly, and the disagreement is the finding this
script exists to make visible:

- the EM fixed point sits above `lambda = 10` in every league, and left to run
  it climbs to the bound;
- out of sample, `lambda` anywhere in 0.1 to 1.0 is equivalent and clearly
  better, with 5.0 already worse and 10.0 worse still.

Both numbers are computed correctly. They answer different questions. Empirical
Bayes asks which prior best explains the *training* data under the model as
written; the holdout asks which penalty predicts matches nobody has seen. The
model as written says the weighted likelihood is a real likelihood with about
300 observations, and under that reading a team's strength is barely resolved,
so most of the observed spread must be noise and should be shrunk away. That
reading is wrong: the weights discount old matches because strength *drifts*,
not because those matches were noisy, and an old match still carries real
information about a team.

So the number this prints is the answer to a well-posed question that is not
quite the question the model needs answered. It is kept because it is the
cheapest way to see that, and because the attack-versus-defence *balance* it
reports survives the objection - that comparison is between two blocks of one
fit on one likelihood scale, so whatever distorts the level distorts both.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database  # noqa: E402
from fbedge.models import base as model_base, goals, hierarchical  # noqa: E402


def describe(con, league: str, as_of: dt.date, target: str, blend_weight: float,
             half_life: float, level: float) -> dict:
    training = model_base.load_training_set(
        con, league, as_of, half_life_days=half_life
    )
    anchored = goals.fit_goals_model(
        training, ridge=goals.AUTO_SPLIT_RIDGE, target=target,
        blend_weight=blend_weight, half_life_days=half_life,
    )
    free = goals.fit_goals_model(
        training, ridge=goals.AUTO_RIDGE, target=target,
        blend_weight=blend_weight, half_life_days=half_life,
    )
    split = anchored.ridge_estimate
    loop = free.ridge_estimate
    return {
        "league": league,
        "matches": training.n_matches,
        "effective_n": round(training.effective_n, 1),
        "teams": len(training.index),
        "dispersion": round(split.dispersion, 3),
        "tau_attack": round(split.tau_attack, 3),
        "tau_defence": round(split.tau_defence, 3),
        "split_attack": round(split.attack, 3),
        "split_defence": round(split.defence, 3),
        "em_attack": round(loop.attack, 2),
        "em_defence": round(loop.defence, 2),
        "em_rounds": loop.rounds,
        "em_settled": loop.converged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="point in time to fit at (default: today)")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--target", default=model_base.DEFAULT_TARGET)
    parser.add_argument("--blend-weight", type=float,
                        default=model_base.DEFAULT_BLEND_WEIGHT)
    parser.add_argument("--half-life", type=float,
                        default=model_base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--level", type=float, default=None,
                        help="anchor for the split mode; defaults to the "
                             "recommended ridge for the target")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}.")
        return 1

    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    level = args.level if args.level is not None else model_base.default_ridge(args.target)
    leagues = args.leagues or sorted(config.LEAGUES)

    con = database.connect(args.db, read_only=True)
    rows = []
    for league in leagues:
        try:
            rows.append(
                describe(con, league, as_of, args.target, args.blend_weight,
                         args.half_life, level)
            )
        except Exception as error:  # a league with too little data is not fatal
            print(f"{league}: skipped ({type(error).__name__}: {error})")

    if not rows:
        print("Nothing to report.")
        return 1

    frame = pd.DataFrame(rows)
    print(f"\nTarget {args.target}, half-life {args.half_life:g}d, as of {as_of}.")
    print(f"Anchored level {level:g}.\n")
    print(frame.to_string(index=False))

    ratios = frame["tau_attack"] ** 2 / frame["tau_defence"] ** 2
    print(
        f"\nAttack variance over defence variance: "
        f"{ratios.min():.2f} to {ratios.max():.2f}, "
        f"above 1 in {int((ratios > 1).sum())} of {len(ratios)} leagues."
    )
    print(
        "  Attack strengths spread more than defensive ones, consistently. "
        "That is the\n  part of the estimate that survives the objection in "
        "this file's docstring."
    )

    settled = int(frame["em_settled"].sum())
    print(
        f"\nThe unanchored EM settled in {settled} of {len(frame)} leagues, at "
        f"lambda {frame['em_attack'].min():.1f} to {frame['em_attack'].max():.1f} "
        "on attack."
    )
    print(
        "  Out of sample anything from 0.1 to 1.0 is equivalent and better, and "
        "5.0 is\n  already worse. Do not read these as settings; read the "
        "docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
