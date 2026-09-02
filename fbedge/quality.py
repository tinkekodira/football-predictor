"""Data quality checks.

Run after every build. The point is not to prove the data is perfect - it
isn't - but to make its gaps visible before they turn into a model that
silently trains on 60% of the corner data and nobody notices.

The known gaps in this source, which the coverage report will show:
  * corners, shots and fouls thin out in older seasons;
  * referee names are present for some leagues and absent for others;
  * Pinnacle closing prices do not go all the way back.
"""

from __future__ import annotations

import pandas as pd

STAT_COLUMNS = [
    "home_goals", "home_goals_ht", "home_shots", "home_sot",
    "home_corners", "home_fouls", "home_yellows", "home_reds", "referee",
]


def coverage_report(con) -> pd.DataFrame:
    """Percentage of rows with data present, per league-season and column."""
    parts = ", ".join(
        f"ROUND(100.0 * COUNT({c}) / NULLIF(COUNT(*), 0), 1) AS {c}_pct"
        for c in STAT_COLUMNS
    )
    return con.execute(
        f"""
        SELECT league, season, COUNT(*) AS matches, {parts}
        FROM matches
        GROUP BY league, season, season_start_year
        ORDER BY league, season_start_year
        """
    ).df()


def odds_coverage(con) -> pd.DataFrame:
    """How many matches carry prices, by bookmaker, market and phase.

    Closing prices are the ones Phase 3 needs; if `close` coverage is thin for
    a season, that season cannot contribute to a closing-line-value analysis.
    """
    return con.execute(
        """
        SELECT o.bookmaker, o.market, o.phase,
               COUNT(DISTINCT o.match_id) AS matches_priced,
               MIN(m.season) AS earliest_season,
               MAX(m.season) AS latest_season
        FROM odds o
        JOIN matches m USING (match_id)
        GROUP BY o.bookmaker, o.market, o.phase
        ORDER BY o.market, o.bookmaker, o.phase
        """
    ).df()


def integrity_checks(con) -> pd.DataFrame:
    """Assertions that should all return zero rows. Anything else is a bug."""
    checks = {
        "duplicate_match_ids": """
            SELECT COUNT(*) FROM (
                SELECT match_id FROM matches GROUP BY match_id HAVING COUNT(*) > 1
            )
        """,
        "duplicate_fixtures": """
            SELECT COUNT(*) FROM (
                SELECT league, date, home_team, away_team
                FROM matches GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1
            )
        """,
        "team_playing_itself": """
            SELECT COUNT(*) FROM matches WHERE home_team = away_team
        """,
        "negative_goals": """
            SELECT COUNT(*) FROM matches WHERE home_goals < 0 OR away_goals < 0
        """,
        "implausible_goals": """
            SELECT COUNT(*) FROM matches WHERE home_goals > 15 OR away_goals > 15
        """,
        "halftime_exceeds_fulltime": """
            SELECT COUNT(*) FROM matches
            WHERE home_goals_ht > home_goals OR away_goals_ht > away_goals
        """,
        "result_disagrees_with_score": """
            SELECT COUNT(*) FROM matches WHERE result IS NOT NULL AND result <> (
                CASE WHEN home_goals > away_goals THEN 'H'
                     WHEN home_goals < away_goals THEN 'A' ELSE 'D' END)
        """,
        "implausible_corners": """
            SELECT COUNT(*) FROM matches WHERE total_corners > 40
        """,
        "odds_below_evens": """
            SELECT COUNT(*) FROM odds WHERE price <= 1.0
        """,
        "orphan_odds": """
            SELECT COUNT(*) FROM odds o
            LEFT JOIN matches m USING (match_id) WHERE m.match_id IS NULL
        """,
        "dates_in_future": """
            SELECT COUNT(*) FROM matches WHERE date > current_date
        """,
    }
    rows = []
    for name, sql in checks.items():
        count = con.execute(sql).fetchone()[0]
        rows.append({"check": name, "offending_rows": int(count), "passed": count == 0})
    return pd.DataFrame(rows)


def overround_sample(con, bookmaker: str = "pinnacle", limit: int = 5) -> pd.DataFrame:
    """Bookmaker margin on 1X2 closing prices, as a sanity check on the odds.

    The sum of implied probabilities should sit a little above 1. A sharp book
    lands around 1.02-1.04; a soft book is higher. Anything at or below 1.00
    means the prices are wrong, not that free money has been found.
    """
    return con.execute(
        """
        SELECT m.season, m.league, m.home_team, m.away_team,
               ROUND(SUM(1.0 / o.price), 4) AS implied_probability_sum
        FROM odds o
        JOIN matches m USING (match_id)
        WHERE o.market = '1x2' AND o.phase = 'close' AND o.bookmaker = ?
        GROUP BY m.match_id, m.season, m.league, m.home_team, m.away_team
        HAVING COUNT(*) = 3
        ORDER BY random()
        LIMIT ?
        """,
        [bookmaker, limit],
    ).df()


def run_all(con, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Run every check and optionally print a readable report."""
    results = {
        "coverage": coverage_report(con),
        "odds_coverage": odds_coverage(con),
        "integrity": integrity_checks(con),
    }
    if verbose:
        pd.set_option("display.width", 140)
        pd.set_option("display.max_columns", 40)
        print("\n=== Match & statistic coverage (% of rows populated) ===")
        print(results["coverage"].to_string(index=False))
        print("\n=== Odds coverage ===")
        print(results["odds_coverage"].to_string(index=False))
        print("\n=== Integrity checks ===")
        print(results["integrity"].to_string(index=False))
        failed = results["integrity"].query("not passed")
        if len(failed):
            print(f"\n{len(failed)} check(s) FAILED - investigate before modelling.")
        else:
            print("\nAll integrity checks passed.")
    return results
