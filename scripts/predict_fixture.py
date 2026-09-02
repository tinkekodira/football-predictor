"""Price a fixture from the command line.

    python scripts/predict_fixture.py Arsenal Liverpool
    python scripts/predict_fixture.py "Man City" Chelsea --as-of 2026-05-01
    python scripts/predict_fixture.py Inter Milan --league I1 --referee "D Massa"
    python scripts/predict_fixture.py Arsenal Everton --ratings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, normalize, predict  # noqa: E402
from fbedge.models import base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home")
    parser.add_argument("away")
    parser.add_argument("--as-of", default=None,
                        help="ISO date; only earlier matches inform the model")
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default=None)
    parser.add_argument("--referee", default=None,
                        help="Applies that referee's fitted card multiplier")
    parser.add_argument("--half-life", type=float, default=base.DEFAULT_HALF_LIFE_DAYS,
                        help="Time-decay half-life in days (default: %(default)s)")
    parser.add_argument("--ridge", type=float, default=base.DEFAULT_RIDGE,
                        help="Shrinkage strength (default: %(default)s)")
    parser.add_argument("--ratings", action="store_true",
                        help="Also print the fitted team ratings for the league")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Write every selection to a CSV file")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    con = database.connect(args.db, read_only=True)
    known = database.known_teams(con, league=args.league)

    resolved = {}
    for role, name in (("home", args.home), ("away", args.away)):
        match = normalize.resolve_team(name, known)
        if match is None:
            print(f"Could not resolve {role} team {name!r}.")
            hints = [t for t in known if name.lower()[:3] in t.lower()][:8]
            if hints:
                print("Did you mean: " + ", ".join(hints))
            return 1
        resolved[role] = match

    try:
        forecast = predict.predict_fixture(
            con, resolved["home"], resolved["away"],
            as_of=args.as_of, league=args.league, referee=args.referee,
            half_life_days=args.half_life, ridge=args.ridge,
        )
    except base.InsufficientData as exc:
        print(f"Not enough history to model this league: {exc}")
        return 1

    print(forecast.render())

    if args.ratings:
        bundle = predict.build_models(
            con, forecast.league, forecast.as_of,
            half_life_days=args.half_life, ridge=args.ridge,
        )
        print("\nTeam ratings (log scale; higher is stronger)")
        print(bundle.goals.ratings().to_string(index=False))
        if bundle.cards and bundle.cards.referee_effects:
            print("\nReferee card multipliers")
            print(bundle.cards.referee_table().to_string(index=False))

    if args.csv:
        forecast.to_frame().to_csv(args.csv, index=False)
        print(f"\nSelections written to {args.csv}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
