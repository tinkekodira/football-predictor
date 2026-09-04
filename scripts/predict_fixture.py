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

from fbedge import config, database, evidence, normalize, predict  # noqa: E402
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
    parser.add_argument("--half-time", action="store_true",
                        help="Also price half-time result and half-time goals. "
                             "Needs a second Dixon-Coles fit on half-time "
                             "scores - they cannot be derived from the "
                             "full-time one - so it roughly doubles the time.")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Hide each market's track record. You are unlikely "
                             "to want this: a fair price with no record beside "
                             "it is the most misleading thing here.")
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
            half_time=args.half_time,
        )
    except base.InsufficientData as exc:
        print(f"Not enough history to model this league: {exc}")
        return 1

    # Attach each market's own track record, on this league, before rendering.
    # `render` prints whatever is in `forecast.evidence`, so a caller that has
    # the evidence cannot forget to show it - but it does have to fetch it.
    if not args.no_evidence:
        stored = evidence.load(con, forecast.league)
        markets_priced = sorted({s.market for s in forecast.selections})
        forecast.evidence = evidence.short_labels(stored, markets_priced)
        if stored.empty:
            print(
                "No evidence has been computed for this database, so every "
                "market below is reported as untested.\n"
                "Run: python scripts/build_evidence.py\n"
            )

    print(forecast.render())

    if not args.no_evidence:
        _print_evidence(con, forecast)

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


def _print_evidence(con, forecast) -> None:
    """The full record behind each market, under the prices it belongs to.

    The short tag on each heading fits a table row; this is the sentence that
    says what it means. Both are shown because the tag alone is enough to spot
    a calibration-only market and not enough to understand one.
    """
    stored = evidence.load(con, forecast.league)
    markets_priced = sorted({s.market for s in forecast.selections})
    width = 66
    print("What the labels mean")
    print("-" * width)
    for market, label in evidence.labels(stored, markets_priced).items():
        print(f"  {market}")
        for line in _wrap(label, width - 6):
            print(f"      {line}")

    if any(m.endswith(("cards", "card_handicap")) for m in markets_priced):
        conditions = evidence.card_conditions(con, forecast.league)
        print()
        print("Card counting, on this league specifically")
        print("-" * width)
        for line in _wrap(conditions["note"], width - 4):
            print(f"  {line}")
    print("=" * width)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


if __name__ == "__main__":
    raise SystemExit(main())
