"""Generate synthetic CSVs in football-data.co.uk's exact raw format.

Why this exists: it lets the whole pipeline be exercised - and the test suite
run - without touching the network, and it deliberately reproduces the source's
awkward habits, so that a build which survives this file will survive the real
one:

  * dd/mm/yy dates in old seasons, dd/mm/yyyy in recent ones;
  * corners/shots/referee columns missing entirely from some league-seasons;
  * blank padding rows and an unnamed trailing column;
  * unplayed fixtures sitting in the in-progress season's file;
  * Pinnacle columns named "PH/PD/PA" in old files, "PSH/PSD/PSA" in new ones.

The numbers are drawn from plausible distributions, not from real matches.
Nothing here should ever be used to fit or evaluate a model.

    python scripts/make_sample_data.py --out data/sample --seasons 4
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config  # noqa: E402

TEAMS = {
    "E0": ["Arsenal", "Liverpool", "Man City", "Man United", "Chelsea", "Tottenham",
           "Newcastle", "Aston Villa", "Brighton", "Everton"],
    "SP1": ["Real Madrid", "Barcelona", "Ath Madrid", "Sevilla", "Betis",
            "Sociedad", "Villarreal", "Valencia", "Girona", "Ath Bilbao"],
    "I1": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio",
           "Atalanta", "Fiorentina", "Bologna", "Torino"],
    "D1": ["Bayern Munich", "Dortmund", "Leverkusen", "RB Leipzig", "Stuttgart",
           "Ein Frankfurt", "Freiburg", "Wolfsburg", "M'gladbach", "Werder Bremen"],
    "F1": ["Paris SG", "Marseille", "Monaco", "Lille", "Lyon",
           "Nice", "Lens", "Rennes", "Brest", "Strasbourg"],
}

REFEREES = ["M Oliver", "A Taylor", "P Tierney", "S Attwell", "C Pawson"]


def _team_strengths(teams: list[str], rng: random.Random) -> dict[str, float]:
    """Attack multipliers, strongest first, with noise."""
    return {
        team: max(0.55, 1.45 - 0.09 * i + rng.gauss(0, 0.05))
        for i, team in enumerate(teams)
    }


def _simulate_match(home: str, away: str, strength: dict[str, float], rng, np_rng):
    """One match: goals from a Poisson pair, everything else correlated to it."""
    home_lambda = 1.35 * strength[home] / strength[away] ** 0.5
    away_lambda = 1.05 * strength[away] / strength[home] ** 0.5
    home_goals = int(np_rng.poisson(min(home_lambda, 4.0)))
    away_goals = int(np_rng.poisson(min(away_lambda, 4.0)))

    home_ht = int(np_rng.binomial(home_goals, 0.45))
    away_ht = int(np_rng.binomial(away_goals, 0.45))

    return {
        "FTHG": home_goals,
        "FTAG": away_goals,
        "FTR": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D"),
        "HTHG": home_ht,
        "HTAG": away_ht,
        "HTR": "H" if home_ht > away_ht else ("A" if away_ht > home_ht else "D"),
        "HS": int(np_rng.poisson(6 + 3 * home_goals)),
        "AS": int(np_rng.poisson(5 + 3 * away_goals)),
        "HST": int(np_rng.poisson(2 + 1.4 * home_goals)),
        "AST": int(np_rng.poisson(2 + 1.4 * away_goals)),
        "HC": int(np_rng.poisson(5.6)),
        "AC": int(np_rng.poisson(4.6)),
        "HF": int(np_rng.poisson(11)),
        "AF": int(np_rng.poisson(12)),
        "HY": int(np_rng.poisson(1.7)),
        "AY": int(np_rng.poisson(2.1)),
        "HR": int(np_rng.binomial(1, 0.04)),
        "AR": int(np_rng.binomial(1, 0.05)),
        "Referee": rng.choice(REFEREES),
    }


def _prices(home_goals_lambda: float, away_goals_lambda: float, rng) -> dict:
    """Rough 1X2 and totals prices with a realistic bookmaker margin."""
    home_p = 0.45 * home_goals_lambda / (home_goals_lambda + away_goals_lambda)
    draw_p = 0.25
    away_p = max(0.05, 1 - home_p - draw_p)
    total = home_p + draw_p + away_p
    home_p, draw_p, away_p = (p / total for p in (home_p, draw_p, away_p))

    def price(p, margin):
        return round(1 / (p * (1 + margin)), 2)

    sharp, soft = 0.025, 0.06
    over_p = rng.uniform(0.45, 0.62)
    return {
        "B365H": price(home_p, soft), "B365D": price(draw_p, soft), "B365A": price(away_p, soft),
        "PSH": price(home_p, sharp), "PSD": price(draw_p, sharp), "PSA": price(away_p, sharp),
        "PSCH": price(home_p * rng.uniform(0.96, 1.04), sharp),
        "PSCD": price(draw_p, sharp),
        "PSCA": price(away_p * rng.uniform(0.96, 1.04), sharp),
        "B365>2.5": price(over_p, soft), "B365<2.5": price(1 - over_p, soft),
        "P>2.5": price(over_p, sharp), "P<2.5": price(1 - over_p, sharp),
        "AHh": round(rng.choice([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]), 2),
        "B365AHH": round(rng.uniform(1.8, 2.1), 2),
        "B365AHA": round(rng.uniform(1.8, 2.1), 2),
    }


def build_season(league: str, start_year: int, seed: int) -> pd.DataFrame:
    """A full double round-robin for one league-season."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    teams = TEAMS[league]
    strength = _team_strengths(teams, rng)

    fixtures = [(h, a) for h in teams for a in teams if h != a]
    rng.shuffle(fixtures)

    is_current = start_year == config.CURRENT_SEASON_START_YEAR
    date = pd.Timestamp(year=start_year, month=8, day=15)
    date_format = "%d/%m/%Y" if start_year >= 2019 else "%d/%m/%y"

    rows = []
    for index, (home, away) in enumerate(fixtures):
        match_date = date + pd.Timedelta(days=3 * index // 5)
        row = {
            "Div": league,
            "Date": match_date.strftime(date_format),
            "Time": f"{rng.choice([13, 15, 17, 20])}:00",
            "HomeTeam": home,
            "AwayTeam": away,
        }
        # The in-progress season's file only has results up to today.
        played = not is_current or match_date < pd.Timestamp.today()
        if played:
            row |= _simulate_match(home, away, strength, rng, np_rng)
        row |= _prices(1.35 * strength[home], 1.05 * strength[away], rng)

        # Old files used Pinnacle's original column names.
        if start_year < 2019:
            row["PH"], row["PD"], row["PA"] = row.pop("PSH"), row.pop("PSD"), row.pop("PSA")
            for key in ("PSCH", "PSCD", "PSCA"):
                row.pop(key, None)
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Not every league-season carries every column. Mirror that.
    if league in {"F1", "I1"} and start_year < 2021:
        frame = frame.drop(columns=[c for c in ("Referee", "HC", "AC") if c in frame])
    if start_year < 2018:
        frame = frame.drop(columns=[c for c in ("HST", "AST") if c in frame])

    frame["Unnamed: 104"] = pd.NA                      # trailing junk column
    padding = pd.DataFrame([{c: pd.NA for c in frame.columns}] * 2)  # blank rows
    return pd.concat([frame, padding], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    parser.add_argument("--seasons", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    years = config.season_years(args.seasons)
    for offset, year in enumerate(years):
        directory = args.out / config.season_code(year)
        directory.mkdir(parents=True, exist_ok=True)
        for league_index, league in enumerate(config.LEAGUES):
            frame = build_season(league, year, args.seed + 100 * offset + league_index)
            frame.to_csv(directory / f"{league}.csv", index=False)
        print(f"  wrote {config.season_label(year)} ({len(config.LEAGUES)} leagues)")

    print(f"\nSynthetic data written to {args.out}")
    print("Load it with: python scripts/build_database.py "
          f"--local-dir {args.out} --seasons {args.seasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
