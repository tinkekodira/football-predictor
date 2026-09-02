"""Break one backtest down by season, and compare two eras.

    python scripts/season_breakdown.py --league E0 --from 2019-08-01
    python scripts/season_breakdown.py --league E0 --from 2019-08-01 --market 1x2

A pooled closing-line-value figure assumes the quantity being measured held
still for the length of the window. That is an assumption, not a finding, and
this checks it.

The reason to check it here specifically: the Phase 3 backtest over 2022-08 to
2026-08 reported roughly -0.5% CLV on 1X2, while the hyperparameter tuner's
development window of 2019-08 to 2024-07 reported roughly +0.9% on the same
league, the same market, the same bet-selection rule and the same code. Both
cannot describe one stable process. Either something differs between the
seasons only one of them covers, or one of the numbers is wrong. Backing the
overlap out of the two aggregates implies the 2019-20 to 2021-22 stretch
carries somewhere near +1.8% CLV on its own, and those are the seasons played
without crowds, when home advantage measurably moved.

If that is what this shows, it is not an edge. It is a regime that ended, and
the practical consequence is that the tuner's default window is majority
contaminated by it: settings chosen there are chosen partly to fit a world
that no longer exists.

**One run, then read the table.** This deliberately does not sweep split
points looking for the largest gap. `--split-season` defaults to 2021 because
crowds returned for 2021-22, which is a reason that existed before the data
was looked at. Moving it needs a reason of the same kind, decided first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, database, evaluation  # noqa: E402


def authoritative_seasons(con, predictions: pd.DataFrame) -> pd.Series | None:
    """The database's own `season_start_year`, aligned to the predictions.

    Preferred over deriving the season from the match date, because
    football-data.co.uk publishes one file per season and that assignment is
    the source's, not a guess reconstructed from a calendar. Falls back to
    None so the caller can use the date rule if the column is missing.
    """
    try:
        lookup = con.execute(
            "SELECT match_id, season_start_year FROM matches"
        ).fetch_df()
    except Exception:
        return None
    if lookup.empty or lookup["season_start_year"].isna().all():
        return None
    mapping = dict(zip(lookup["match_id"], lookup["season_start_year"]))
    mapped = predictions["match_id"].map(mapping)
    if mapped.isna().any():
        return None
    return mapped.astype(int).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--market", default=None,
                        choices=list(backtest.BETTABLE_MARKETS),
                        help="Restrict to one market. Omit to pool all three.")
    parser.add_argument("--split-season", type=int, default=2021,
                        help="Season that starts the later era (default: %(default)s)")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    markets = (args.market,) if args.market else backtest.BETTABLE_MARKETS

    settings = backtest.BacktestConfig(
        league=args.league, start=start, end=end, step_days=args.step_days,
        markets=markets, edge_threshold=args.edge_threshold,
        fit_count_models=False,
    )
    result = backtest.run_backtest(con, settings, verbose=True)
    if result.predictions.empty:
        print("Nothing was priced. Widen the window or check the database.")
        return 1

    seasons = authoritative_seasons(con, result.predictions)
    source = "database season_start_year" if seasons is not None else "match date"

    print()
    print("=" * 74)
    print(f"  {config.LEAGUES[args.league]}  |  {start} to {end}"
          f"  |  {args.market or 'all markets'}")
    print(f"  season assigned from {source}")
    print("=" * 74)

    table = evaluation.season_breakdown(
        result.predictions, market=args.market, season=seasons,
        edge_threshold=args.edge_threshold,
    )
    if table.empty:
        print("  No seasons with priced selections.")
        return 1

    sources = evaluation.fair_line_sources(result.predictions, season=seasons)
    changes = evaluation.benchmark_changed(sources)
    if changes:
        print("\n!!! THE BENCHMARK CHANGED INSIDE THIS WINDOW !!!")
        for line in changes:
            print(f"  {line}")
        print("  Closing line value is measured against this benchmark, so the")
        print("  seasons either side of a change are not on one scale. Re-run")
        print("  pinned to a single book before reading the table below:")
        print("    BacktestConfig(fair_line_preference=('pinnacle',), ...)")
        print("\n  Benchmark by season:")
        print(sources.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n--- Closing line value by season "
          "-----------------------------------------")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("  se_clustered groups bets by match, because selections on one")
    print("  match share a model fit and a closing-line move. se_naive is the")
    print("  number scripts/backtest.py reports, shown alongside so the two")
    print("  are known to be the same estimator under different assumptions.")
    print("  z uses the clustered figure. Treat any single season's z with")
    print("  suspicion: there are as many of them as there are rows.")

    era = evaluation.era_comparison(
        result.predictions, split_season=args.split_season,
        market=args.market, season=seasons, edge_threshold=args.edge_threshold,
    )
    print(f"\n--- Eras, split at {args.split_season} "
          "-------------------------------------------------")
    if "difference" not in era:
        print(f"  Only one era present ({era['n_before']} bets before, "
              f"{era['n_after']} after). Widen the window.")
    else:
        print(f"  before {args.split_season}   mean CLV {era['mean_clv_before']:+.3%} "
              f"(SE {era['se_before']:.3%}, n={era['n_before']})")
        print(f"  from {args.split_season}     mean CLV {era['mean_clv_after']:+.3%} "
              f"(SE {era['se_after']:.3%}, n={era['n_after']})")
        print(f"  difference     {era['difference']:+.3%} "
              f"(SE {era['difference_se']:.3%}, {era['difference_z']:+.1f} SE)")
        print()
        if abs(era["difference_z"]) < 2:
            print("  -> the two eras are not distinguishable. The pooled CLV")
            print("     figure describes one process and can be read as-is.")
        elif era["mean_clv_after"] < era["mean_clv_before"]:
            print("  -> CLV was higher in the earlier era. Whatever produced")
            print("     it is not present now. The pooled figure is an average")
            print("     over two regimes and overstates the current one; the")
            print("     later era is the honest estimate of where things stand.")
            print("     Retune and re-evaluate on the later era only.")
        else:
            print("  -> CLV is higher in the later era. Before treating that as")
            print("     improvement, check it is not driven by one season, and")
            print("     that the split was not chosen after seeing the table.")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nPer-season table written to {args.csv}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
