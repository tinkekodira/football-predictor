"""Test one candidate setting against the shipping default, league by league.

    python scripts/validate_setting.py --target blend --ridge 1 --from 2018-08-01

A grid search tells you which setting won a search. It does not tell you
whether that setting is better, because the winner of a search over nine
options on one league is partly just the luckiest of nine. This script exists
to answer the second question, and it is deliberately not a search: it takes
**one** candidate, decided in advance, and runs it against the current default.

**The selection league is reported separately from the rest.** The candidate
here was chosen by looking at the Premier League, so the Premier League cannot
also judge it - that is the same number twice. The other four leagues had no
part in choosing it, so their verdict is the honest one. They are printed apart
for that reason, and the pooled figure at the bottom excludes the selection
league unless `--include-selection` is passed.

This is the discipline the handoff asks for in its "additional leagues" note:
decide what counts as success first, run every league, and report all of them
including the ones that disagree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402
from fbedge.models import base as model_base, goals  # noqa: E402


def _ridge(value: str):
    """A number, or one of the names that means "estimate it"."""
    if value in goals.AUTO_MODES:
        return value
    return float(value)


def run_one(con, league, start, end, step_days, market, target, blend_weight,
            ridge, half_life) -> dict:
    settings = backtest.BacktestConfig(
        league=league, start=start, end=end, step_days=step_days,
        markets=(market,), fit_count_models=False,
        target=target, blend_weight=blend_weight,
        ridge=ridge, half_life_days=half_life,
    )
    result = backtest.run_backtest(con, settings, verbose=False)
    stats = evaluation.score_market(result.predictions, market)
    bets = result.bets
    clv = (
        evaluation.clustered_mean(bets["clv"], bets["match_id"])
        if not bets.empty else {"mean": float("nan")}
    )
    frame = result.predictions
    side = frame[
        (frame["market"] == market)
        & (frame["selection"] == ("home" if market == "1x2" else "over"))
        & (frame["push_fraction"] < 0.5)
        & frame["model_conditional"].notna()
    ]
    slope = evaluation.calibration_slope(
        side["model_conditional"].to_numpy(),
        (side["win_fraction"] > 0.5).to_numpy(dtype=float),
    )
    return {
        "log_loss": stats.get("model_log_loss", float("nan")),
        "gap": stats.get("log_loss_gap", float("nan")),
        "slope": slope.get("slope", float("nan")),
        "clv": clv.get("mean", float("nan")),
        "n": stats.get("n", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--market", default="1x2", choices=list(backtest.BETTABLE_MARKETS))
    parser.add_argument("--target", nargs="+", default=["blend"],
                        choices=["goals", "xg", "blend"],
                        help="One or more targets; the cross-product with "
                             "--ridge is run so a combined gain can be "
                             "attributed to the target or to the shrinkage")
    parser.add_argument("--blend-weight", type=float, default=0.5)
    parser.add_argument("--ridge", type=_ridge, nargs="+", default=[1.0],
                        help="numbers, or 'auto' / 'auto-split' to estimate "
                             "shrinkage from the data (models.hierarchical)")
    parser.add_argument("--half-life", type=float, default=model_base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--selection-league", default="E0",
                        help="The league the candidate was chosen on; judged separately")
    parser.add_argument("--include-selection", action="store_true",
                        help="Fold the selection league into the pooled verdict too")
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    leagues = args.leagues or sorted(config.LEAGUES)

    def label_of(target: str, ridge) -> str:
        name = target if target != "blend" else f"blend{args.blend_weight:g}"
        shown = ridge if isinstance(ridge, str) else f"r{ridge:g}"
        return f"{name} {shown}"

    combos = [(t, r) for t in args.target for r in args.ridge]
    baseline_key = ("goals", model_base.DEFAULT_RIDGE)
    if baseline_key not in combos:
        combos.insert(0, baseline_key)

    print("=" * 78)
    print(f"  baseline  : {label_of(*baseline_key)}  (what ships today)")
    print(f"  candidates: {', '.join(label_of(t, r) for t, r in combos if (t, r) != baseline_key)}")
    print(f"  half-life : {args.half_life:g}d for every run")
    print(f"  window    : {start} to {end}   |  market {args.market}")
    print(f"  chosen on : {args.selection_league} - judged separately below")
    print("=" * 78)
    print()

    rows = []
    for league in leagues:
        measured = {}
        for target, ridge in combos:
            try:
                measured[(target, ridge)] = run_one(
                    con, league, start, end, args.step_days, args.market,
                    target, args.blend_weight, ridge, args.half_life,
                )
            except ValueError as exc:
                print(f"  {league}: skipped, {exc}")
                measured = {}
                break
        if not measured:
            continue
        base = measured[baseline_key]
        for (target, ridge), stats in measured.items():
            rows.append({
                "league": league,
                "setting": label_of(target, ridge),
                "held_out": league != args.selection_league,
                "log_loss": stats["log_loss"],
                "delta": stats["log_loss"] - base["log_loss"],
                "gap": stats["gap"],
                "slope": stats["slope"],
                "clv": stats["clv"],
            })
        print(f"  ran {league}")

    if not rows:
        print("Nothing ran.")
        return 1

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    judged = table if args.include_selection else table[table["held_out"]]
    label = "all leagues" if args.include_selection else "held-out leagues only"
    print()
    print("-" * 78)
    print(f"  Verdict on {label}, each setting against what ships today:")
    print()
    for setting, group in judged.groupby("setting", sort=False):
        deltas = group["delta"].to_numpy()
        if setting == label_of(*baseline_key):
            continue
        mean = float(deltas.mean())
        se = (
            float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
            if len(deltas) > 1 else float("nan")
        )
        t = mean / se if se else float("nan")
        verdict = (
            "holds up" if (mean < 0 and np.isfinite(t) and abs(t) > 2)
            else "right way, not significant" if mean < 0
            else "no better"
        )
        print(f"    {setting:<16} mean {mean:+.5f}  SE {se:.5f}  t {t:+.2f}  "
              f"better in {int((deltas < 0).sum())}/{len(deltas)}   {verdict}")

    print()
    print("  A setting that beats the default on leagues which had no part in")
    print("  choosing it is the evidence needed to change a default. Comparing")
    print("  the target-only and shrinkage-only rows against the combined one")
    print("  says which of the two changes is actually doing the work, which")
    print("  matters because they are not equally cheap to keep.")
    print()
    print("  Read the slope column even where the verdict is negative: it says")
    print("  whether the shrinkage change did what it was meant to do,")
    print("  separately from whether that helped.")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
