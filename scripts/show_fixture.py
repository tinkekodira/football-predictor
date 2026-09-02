"""Print a fixture profile as a text card, without starting the web app.

    python scripts/show_fixture.py Arsenal Liverpool
    python scripts/show_fixture.py "Man City" Chelsea --as-of 2026-05-01
    python scripts/show_fixture.py Inter Milan --scope season --league I1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, normalize, profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("home", help="Home team (partial names are resolved)")
    parser.add_argument("away", help="Away team")
    parser.add_argument("--as-of", default=None,
                        help="ISO date; only earlier matches are used (default: today)")
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default=None)
    parser.add_argument("--season", type=int, default=None,
                        help="Season start year (default: %(default)s)")
    parser.add_argument("--scope", choices=profile.SCOPES, default="season_venue")
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
            suggestions = [t for t in known if name.lower()[:3] in t.lower()][:8]
            if suggestions:
                print("Did you mean: " + ", ".join(suggestions))
            return 1
        resolved[role] = match

    card = profile.fixture_profile(
        con, resolved["home"], resolved["away"],
        as_of=args.as_of, league=args.league, season_start_year=args.season,
    )
    print(profile.format_profile(card, scope=args.scope))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
