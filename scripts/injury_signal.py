"""Does *real* injury news tell the model anything? The availability retest.

    python scripts/build_injuries.py --season 2024     # then 2023, 2022
    python scripts/injury_signal.py --league E0 --from 2024-08-01 --to 2025-06-01

**This exists because the availability study had one stated weakness and this
fixes exactly it.** That study built absences from Understat line-ups, which
list only players who *appeared*, so a fit player left on the bench and a
player in hospital were indistinguishable. It found the right signs in all six
specifications and significance in none, and the honest reading was "an effect
under about 2% cannot be resolved by this proxy" rather than "team news does
not matter". See the availability section of HANDOFF.md and BACKLOG B6.

The injury feed states a **reason** - "Ankle Injury" - per player per fixture.
That replaces the proxy with a measurement, and it is free: API-Football's free
plan serves 2022 to 2024, which sits inside the era the original study covered.

**Same specification as `availability_signal.py`, deliberately**, so the two
numbers are comparable:

    log E[goals] = log(model rate) + alpha
                   + beta_own * out(this team) + beta_opp * out(opponent)

with the model's own fitted rate as an offset and uncertainty bootstrapped by
match rather than by row. The one change is the unit: `out` is a **count of
players ruled out**, not a share, so `beta_own` reads directly as "what one
missing player does to the scoring rate" with no eleventh-of-a-squad
conversion in the way.

**What this can and cannot claim.** It measures whether absences move scoring.
It does *not* establish that the information was tradeable: the feed's rows are
attached to fixtures, and how far before kick-off each row existed is not
something the endpoint says. Treat a positive result as "the effect is real",
which is a question the project has open, and not as "here is an edge".
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

from fbedge import backtest, config, database, injuries  # noqa: E402

# Below this share of matches carrying injury rows the study is not worth
# reading: a thin join usually means the seasons fetched do not overlap the
# window asked for, and a quiet answer from half the data is not a null.
MINIMUM_JOIN_RATE = 0.5


def fit_poisson(goals, log_rate, own, opp):
    """Poisson MLE for (alpha, beta_own, beta_opp) with `log_rate` as offset.

    Identical to `availability_signal.fit_poisson`. Copied rather than imported
    because these two scripts must stay comparable even if one is later
    changed, and a shared helper is exactly how that quietly stops being true.
    """

    def negative_log_likelihood(theta):
        eta = log_rate + theta[0] + theta[1] * own + theta[2] * opp
        eta = np.clip(eta, -10.0, 4.0)
        return -float(np.sum(goals * eta - np.exp(eta)))

    result = optimize.minimize(
        negative_log_likelihood, np.zeros(3), method="BFGS"
    )
    return result.x


def out_counts(stored: pd.DataFrame, doubtful: bool) -> pd.DataFrame:
    """Players unavailable per team per fixture date.

    `doubtful` decides whether "Questionable" counts. Both are reported,
    because the feed makes a distinction and collapsing it would be choosing an
    answer rather than measuring one.
    """
    if stored.empty:
        return pd.DataFrame(columns=["team", "fixture_date", "out"])
    mask = stored["ruled_out"].astype(bool)
    if doubtful:
        mask = mask | stored["doubtful"].astype(bool)
    subset = stored[mask]
    if subset.empty:
        return pd.DataFrame(columns=["team", "fixture_date", "out"])
    grouped = (
        subset.groupby(["team", "fixture_date"])["player"]
        .nunique()
        .reset_index(name="out")
    )
    # Both sides of the later `merge_asof` must carry the same datetime
    # resolution; DuckDB hands back seconds on one column and microseconds on
    # the other, and pandas refuses to join across them.
    grouped["fixture_date"] = pd.to_datetime(
        grouped["fixture_date"]
    ).astype("datetime64[ns]")
    return grouped


def new_out_counts(stored: pd.DataFrame, doubtful: bool) -> pd.DataFrame:
    """Players newly unavailable: out for this fixture, available for the last.

    **This is the specification that matters, and the plain count is the one
    that does not.** A player out for the season appears in every fixture, and
    by the third of them the model has already adapted: the team's recent
    results were produced without him, so the fitted rate carries his absence
    already. Counting him again asks the model to subtract the same player
    twice.

    What is genuinely news relative to the ratings is a *change* - somebody who
    played last week and will not play today. That is the same reasoning as the
    absence window in `fbedge/availability.py`, arrived at there for the same
    reason, and it is why that study looked at recent absences rather than at
    who happened to be missing.

    **Known undercount, and it is the safe direction.** The comparison is
    between consecutive *observations* of a team, not consecutive fixtures. The
    feed emits a row only when somebody is unavailable, so a date with no rows
    means either "everyone was fit" or "the team did not play", and injuries
    alone cannot separate those. A player who is out, recovers, and relapses is
    therefore counted once rather than twice. That misses real news and biases
    the study towards finding *less* effect than exists, which is the direction
    to err in for a result this project would like to be true.
    """
    if stored.empty:
        return pd.DataFrame(columns=["team", "fixture_date", "out"])
    mask = stored["ruled_out"].astype(bool)
    if doubtful:
        mask = mask | stored["doubtful"].astype(bool)
    subset = stored[mask].dropna(subset=["team", "player", "fixture_date"]).copy()
    if subset.empty:
        return pd.DataFrame(columns=["team", "fixture_date", "out"])

    subset["fixture_date"] = pd.to_datetime(subset["fixture_date"])
    rows = []
    for team, block in subset.groupby("team"):
        dates = sorted(block["fixture_date"].unique())
        previous: set = set()
        for date in dates:
            current = set(block.loc[block["fixture_date"] == date, "player"])
            rows.append({
                "team": team,
                "fixture_date": date,
                "out": float(len(current - previous)),
            })
            previous = current
    grouped = pd.DataFrame(rows)
    grouped["fixture_date"] = pd.to_datetime(
        grouped["fixture_date"]
    ).astype("datetime64[ns]")
    return grouped


def attach(totals: pd.DataFrame, matches: pd.DataFrame,
           counts: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Join injury counts onto scored matches. Returns (frame, join rate).

    The join is on team and date. Both sources date a match by its kick-off,
    but in different timezones, so a late kick-off can differ by a day; the
    match is therefore made on the *nearest* date within one day rather than on
    equality. Doing it on equality silently drops the late fixtures, which are
    exactly the ones a Saturday evening is full of.
    """
    # `match_totals` already carries the date, so only the team names are taken
    # from `matches`. Pulling `date` from both would give `date_x`/`date_y` and
    # a KeyError that reads as missing data rather than a duplicated column.
    frame = totals.merge(
        matches[["match_id", "home_team", "away_team"]],
        on="match_id", how="inner",
    )
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    if counts.empty:
        frame["home_out"] = np.nan
        frame["away_out"] = np.nan
        return frame, 0.0

    for side in ("home", "away"):
        merged = pd.merge_asof(
            frame.sort_values("date"),
            counts.sort_values("fixture_date").rename(
                columns={"team": f"{side}_team", "out": f"{side}_out"}
            ),
            left_on="date", right_on="fixture_date", by=f"{side}_team",
            tolerance=pd.Timedelta(days=1), direction="nearest",
        )
        frame = frame.sort_values("date").reset_index(drop=True)
        frame[f"{side}_out"] = merged[f"{side}_out"].to_numpy()
        frame = frame.drop(columns=["fixture_date"], errors="ignore")

    joined = frame["home_out"].notna() & frame["away_out"].notna()
    return frame, float(joined.mean()) if len(frame) else 0.0


def report(frame: pd.DataFrame, label: str, bootstrap: int, seed: int) -> dict | None:
    """Fit and print one specification, and hand the estimates back.

    Returned rather than only printed so that several leagues can be pooled
    without anyone retyping numbers off a terminal - which is how a table
    ends up disagreeing with the run that produced it.
    """
    usable = frame.dropna(
        subset=["home_out", "away_out", "model_home_rate", "model_away_rate"]
    )
    if usable.empty:
        print(f"  {label}: nothing left after the join.")
        return None

    # One row per team per match: the attacking side, its own absentees and its
    # opponent's.
    long = pd.concat([
        pd.DataFrame({
            "match_id": usable["match_id"],
            "goals": usable["observed_home"].astype(float),
            "rate": usable["model_home_rate"].astype(float),
            "own": usable["home_out"].astype(float),
            "opp": usable["away_out"].astype(float),
        }),
        pd.DataFrame({
            "match_id": usable["match_id"],
            "goals": usable["observed_away"].astype(float),
            "rate": usable["model_away_rate"].astype(float),
            "own": usable["away_out"].astype(float),
            "opp": usable["home_out"].astype(float),
        }),
    ], ignore_index=True).dropna()

    log_rate = np.log(np.maximum(long["rate"].to_numpy(), 1e-6))
    point = fit_poisson(
        long["goals"].to_numpy(), log_rate,
        long["own"].to_numpy(), long["opp"].to_numpy(),
    )

    # Bootstrap by match: two rows share a fitted model and an opponent, so
    # resampling rows would overstate precision by about root two.
    rng = np.random.default_rng(seed)
    match_ids = usable["match_id"].to_numpy()
    by_match = {m: g for m, g in long.groupby("match_id")}
    draws = []
    for _ in range(bootstrap):
        picked = rng.choice(match_ids, size=len(match_ids), replace=True)
        block = pd.concat([by_match[m] for m in picked if m in by_match],
                          ignore_index=True)
        if block.empty:
            continue
        draws.append(fit_poisson(
            block["goals"].to_numpy(),
            np.log(np.maximum(block["rate"].to_numpy(), 1e-6)),
            block["own"].to_numpy(), block["opp"].to_numpy(),
        ))
    errors = np.std(np.array(draws), axis=0) if draws else np.full(3, np.nan)

    mean_out = float(long["own"].mean())
    print(f"\n  {label}")
    print(f"    matches {len(usable)}, mean players out per side {mean_out:.2f}")
    for name, index in (("beta_own", 1), ("beta_opp", 2)):
        value, error = point[index], errors[index]
        z = value / error if error and np.isfinite(error) and error > 0 else float("nan")
        print(f"    {name:9s} {value:+.4f}  SE {error:.4f}  z {z:+.2f}"
              f"   ({value * 100:+.1f}% on the rate per player)")
    detectable = 2 * errors[1] * 100 if np.isfinite(errors[1]) else float("nan")
    print(f"    detectable at 2 SE: {detectable:.1f}% per missing player")
    return {
        "matches": len(usable),
        "beta_own": float(point[1]), "se_own": float(errors[1]),
        "beta_opp": float(point[2]), "se_opp": float(errors[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="E0")
    parser.add_argument(
        "--leagues", nargs="+", choices=sorted(config.LEAGUES), default=None,
        help="run several leagues and pool them. One league on its own is far "
             "too small to read: the first pass on a single season of E0 gave "
             "-7.7 percent and the same league over three seasons gave -4.6.",
    )
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}.")
        return 1

    con = database.connect(args.db, read_only=True)
    if not con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'injuries'"
    ).fetchone():
        print("No injuries table. Run scripts/build_injuries.py first "
              f"(free plan covers up to season "
              f"{config.INJURY_FREE_PLAN_LAST_SEASON}).")
        return 1

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    leagues = args.leagues or [args.league]

    collected: dict[str, list[dict]] = {}
    for league in leagues:
        for label, values in run_league(con, league, start, end, args):
            collected.setdefault(label, []).append(values)
    con.close()

    if len(leagues) > 1:
        pool(collected)

    print("\n  Reminder: this measures whether absences move scoring. It does "
          "not\n  establish the information was knowable before the price - see "
          "the\n  docstring.")
    return 0


def pool(collected: dict[str, list[dict]]) -> None:
    """Inverse-variance pool across leagues, one row per specification.

    Pooling is not here to manufacture significance out of five small studies.
    It is here because a per-league table invites picking the league that
    agrees with you, which is exactly the multiple-comparisons trap this
    project warns about for this shape of question.

    **Read the `signs` column before the z.** Five leagues agreeing weakly is
    better evidence than one league shouting, and this study has already shown
    why: a single season of E0 gave -7.7 percent, and the same league over
    three seasons gave -4.6.
    """
    print(f"\n{'=' * 78}")
    print("  POOLED ACROSS LEAGUES  (inverse-variance weighted)")
    print(f"{'=' * 78}")
    print(f"  {'specification':32s} {'beta_own':>9s} {'z':>6s} {'signs':>6s} "
          f"{'beta_opp':>9s} {'z':>6s} {'signs':>6s}")
    for label, entries in collected.items():
        pieces = [f"  {label:32s}"]
        for name, expected in (("own", -1.0), ("opp", 1.0)):
            betas = np.array([v[f"beta_{name}"] for v in entries])
            errors = np.array([v[f"se_{name}"] for v in entries])
            good = np.isfinite(betas) & np.isfinite(errors) & (errors > 0)
            if not good.any():
                pieces.append(f" {'n/a':>9s} {'':>6s} {'':>6s}")
                continue
            weights = 1.0 / errors[good] ** 2
            mean = float(np.sum(weights * betas[good]) / np.sum(weights))
            error = float(1.0 / np.sqrt(np.sum(weights)))
            agreeing = int(np.sum(np.sign(betas[good]) == expected))
            pieces.append(
                f" {mean:+9.4f} {mean / error:+6.2f} "
                f"{agreeing:>3d}/{int(good.sum()):<2d}"
            )
        print("".join(pieces))
    print("\n  `signs` counts leagues whose estimate carries the predicted "
          "direction:\n  a side missing players scores less, its opponent more.")


def run_league(con, league: str, start, end, args) -> list[tuple[str, dict]]:
    """Every specification for one league. Returns (label, estimates) pairs."""
    print(f"\nRefitting {config.LEAGUES[league]} week by week...")
    settings = backtest.BacktestConfig(
        league=league, start=start, end=end, step_days=args.step_days,
        markets=("1x2",), fit_count_models=False,
    )
    result = backtest.run_backtest(con, settings, verbose=False)
    totals = result.match_totals
    if totals.empty:
        print("  Nothing was scored in this window.")
        return []

    stored = injuries.load_injuries(con)
    stored = stored[stored["league"] == league] if not stored.empty else stored
    matches = con.execute(
        "SELECT match_id, date, home_team, away_team FROM matches WHERE league = ?",
        [league],
    ).df()

    if stored.empty:
        print(f"  No injuries stored for {league}. Run "
              f"scripts/build_injuries.py --season <year>.")
        return []

    seasons = sorted(stored["season_start_year"].unique())
    print(f"\n{'=' * 78}")
    print(f"  {config.LEAGUES[league]}  |  {start} to {end}")
    print(f"  {len(stored)} injury rows, seasons {seasons}")
    print(f"{'=' * 78}")

    specs = [
        ("newly out", new_out_counts, False),
        ("newly out or doubtful", new_out_counts, True),
        ("all currently out", out_counts, False),
        ("all currently out or doubtful", out_counts, True),
    ]
    out: list[tuple[str, dict]] = []
    for label, builder, doubtful in specs:
        counts = builder(stored, doubtful=doubtful)
        frame, rate = attach(totals, matches, counts)
        print(f"\n  join rate {rate:.1%} of {len(totals)} scored matches")
        if rate < MINIMUM_JOIN_RATE:
            print(f"  -> below {MINIMUM_JOIN_RATE:.0%}. Most likely the seasons "
                  "fetched do not cover this window. Not reporting a result "
                  "from half the data.")
            continue
        values = report(frame, label, args.bootstrap, args.seed)
        if values is not None:
            out.append((label, values))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
