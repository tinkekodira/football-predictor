"""DuckDB storage layer.

DuckDB rather than Postgres because it is a single file with no server to run,
no password to manage and no monthly cost, and because the workload is almost
entirely analytical scans over a few hundred thousand rows - exactly what a
columnar engine is for. If this ever needs concurrent writers or a hosted API,
the schema ports to Postgres with minimal change.

The one piece of real design here is `team_matches`: a view that turns each
match into two rows, one per team, with the columns relabelled to "for" and
"against". Every statistic in the profile layer is a filter and an average
over that view, which keeps the analytics code short and, more importantly,
keeps home/away handling in exactly one place instead of duplicated across
every query.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id           VARCHAR PRIMARY KEY,
    league             VARCHAR NOT NULL,
    league_name        VARCHAR,
    country            VARCHAR,
    season_start_year  INTEGER NOT NULL,
    season             VARCHAR,
    date               DATE NOT NULL,
    kickoff_time       VARCHAR,
    home_team          VARCHAR NOT NULL,
    away_team          VARCHAR NOT NULL,
    referee            VARCHAR,
    attendance         INTEGER,
    home_goals         INTEGER,
    away_goals         INTEGER,
    result             VARCHAR,
    home_goals_ht      INTEGER,
    away_goals_ht      INTEGER,
    result_ht          VARCHAR,
    home_shots         INTEGER,
    away_shots         INTEGER,
    home_sot           INTEGER,
    away_sot           INTEGER,
    home_corners       INTEGER,
    away_corners       INTEGER,
    home_fouls         INTEGER,
    away_fouls         INTEGER,
    home_offsides      INTEGER,
    away_offsides      INTEGER,
    home_woodwork      INTEGER,
    away_woodwork      INTEGER,
    home_yellows       INTEGER,
    away_yellows       INTEGER,
    home_reds          INTEGER,
    away_reds          INTEGER,
    total_goals        INTEGER,
    goal_difference    INTEGER,
    btts               BOOLEAN,
    over_1_5           BOOLEAN,
    over_2_5           BOOLEAN,
    over_3_5           BOOLEAN,
    goals_ht           INTEGER,
    goals_2h           INTEGER,
    total_corners      INTEGER,
    total_cards        INTEGER,
    total_yellows      INTEGER,
    total_reds         INTEGER,
    booking_points     INTEGER,
    total_shots        INTEGER,
    total_sot          INTEGER,
    total_fouls        INTEGER
);

CREATE TABLE IF NOT EXISTS odds (
    match_id   VARCHAR NOT NULL,
    bookmaker  VARCHAR NOT NULL,
    phase      VARCHAR NOT NULL,   -- 'open' or 'close'
    market     VARCHAR NOT NULL,   -- '1x2' | 'total_goals' | 'asian_handicap'
    selection  VARCHAR NOT NULL,   -- 'home'|'draw'|'away'|'over'|'under'
    line       DOUBLE,             -- NULL for 1x2
    price      DOUBLE NOT NULL     -- decimal odds
);

CREATE TABLE IF NOT EXISTS ingest_log (
    league             VARCHAR,
    season_start_year  INTEGER,
    rows_loaded        INTEGER,
    odds_rows_loaded   INTEGER,
    loaded_at          TIMESTAMP
);
"""

# One row per team per match. `is_home` lets any query split by venue, and the
# for/against framing means "goals conceded away from home" is a filter rather
# than a different column name.
TEAM_MATCHES_VIEW_SQL = """
CREATE OR REPLACE VIEW team_matches AS
SELECT
    match_id, league, league_name, country, season_start_year, season,
    date, referee,
    home_team            AS team,
    away_team            AS opponent,
    TRUE                 AS is_home,
    home_goals           AS goals_for,
    away_goals           AS goals_against,
    home_goals_ht        AS goals_for_ht,
    away_goals_ht        AS goals_against_ht,
    home_shots           AS shots_for,
    away_shots           AS shots_against,
    home_sot             AS sot_for,
    away_sot             AS sot_against,
    home_corners         AS corners_for,
    away_corners         AS corners_against,
    home_fouls           AS fouls_for,
    away_fouls           AS fouls_against,
    home_yellows         AS yellows_for,
    away_yellows         AS yellows_against,
    home_reds            AS reds_for,
    away_reds            AS reds_against,
    CASE result WHEN 'H' THEN 'W' WHEN 'A' THEN 'L' WHEN 'D' THEN 'D' END AS outcome,
    CASE result WHEN 'H' THEN 3  WHEN 'D' THEN 1  ELSE 0 END AS points,
    total_goals, btts, over_1_5, over_2_5, over_3_5,
    goals_ht, goals_2h, total_corners, total_cards, total_yellows,
    total_reds, booking_points, total_shots, total_sot, total_fouls
FROM matches
UNION ALL
SELECT
    match_id, league, league_name, country, season_start_year, season,
    date, referee,
    away_team            AS team,
    home_team            AS opponent,
    FALSE                AS is_home,
    away_goals           AS goals_for,
    home_goals           AS goals_against,
    away_goals_ht        AS goals_for_ht,
    home_goals_ht        AS goals_against_ht,
    away_shots           AS shots_for,
    home_shots           AS shots_against,
    away_sot             AS sot_for,
    home_sot             AS sot_against,
    away_corners         AS corners_for,
    home_corners         AS corners_against,
    away_fouls           AS fouls_for,
    home_fouls           AS fouls_against,
    away_yellows         AS yellows_for,
    home_yellows         AS yellows_against,
    away_reds            AS reds_for,
    home_reds            AS reds_against,
    CASE result WHEN 'A' THEN 'W' WHEN 'H' THEN 'L' WHEN 'D' THEN 'D' END AS outcome,
    CASE result WHEN 'A' THEN 3  WHEN 'D' THEN 1  ELSE 0 END AS points,
    total_goals, btts, over_1_5, over_2_5, over_3_5,
    goals_ht, goals_2h, total_corners, total_cards, total_yellows,
    total_reds, booking_points, total_shots, total_sot, total_fouls
FROM matches;
"""


def connect(db_path: Path | str | None = None, read_only: bool = False):
    """Open (and initialise, if needed) the database."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        initialise(con)
    return con


def initialise(con) -> None:
    """Create tables and views. Safe to call repeatedly."""
    con.execute(SCHEMA_SQL)
    con.execute(TEAM_MATCHES_VIEW_SQL)


def load_matches(con, matches: pd.DataFrame) -> int:
    """Insert matches, replacing any rows already present.

    Delete-then-insert on match_id keeps re-running the build idempotent,
    which matters because the in-progress season's file is re-downloaded and
    reloaded every time new results appear.
    """
    if matches.empty:
        return 0
    con.register("incoming_matches", matches)
    con.execute(
        "DELETE FROM matches WHERE match_id IN (SELECT match_id FROM incoming_matches)"
    )
    columns = [row[1] for row in con.execute("PRAGMA table_info('matches')").fetchall()]
    col_list = ", ".join(f'"{c}"' for c in columns)
    con.execute(
        f"INSERT INTO matches ({col_list}) SELECT {col_list} FROM incoming_matches"
    )
    con.unregister("incoming_matches")
    return len(matches)


def load_odds(con, odds: pd.DataFrame) -> int:
    """Insert odds, replacing all prices for the affected matches."""
    if odds.empty:
        return 0
    con.register("incoming_odds", odds)
    con.execute(
        "DELETE FROM odds WHERE match_id IN (SELECT DISTINCT match_id FROM incoming_odds)"
    )
    con.execute(
        "INSERT INTO odds (match_id, bookmaker, phase, market, selection, line, price) "
        "SELECT match_id, bookmaker, phase, market, selection, line, price "
        "FROM incoming_odds"
    )
    con.unregister("incoming_odds")
    return len(odds)


def log_ingest(con, league: str, season_start_year: int, rows: int, odds_rows: int) -> None:
    con.execute(
        "INSERT INTO ingest_log VALUES (?, ?, ?, ?, now())",
        [league, season_start_year, rows, odds_rows],
    )


def known_teams(con, league: str | None = None, season_start_year: int | None = None) -> list[str]:
    """Distinct team names, optionally scoped to a league and season."""
    sql = "SELECT DISTINCT team FROM team_matches WHERE 1=1"
    params: list = []
    if league:
        sql += " AND league = ?"
        params.append(league)
    if season_start_year is not None:
        sql += " AND season_start_year = ?"
        params.append(season_start_year)
    sql += " ORDER BY team"
    return [row[0] for row in con.execute(sql, params).fetchall()]


def summary(con) -> pd.DataFrame:
    """Row counts and date ranges per league and season."""
    return con.execute(
        """
        SELECT league, league_name, season,
               COUNT(*)                     AS matches,
               MIN(date)                    AS first_match,
               MAX(date)                    AS last_match
        FROM matches
        GROUP BY league, league_name, season, season_start_year
        ORDER BY league, season_start_year
        """
    ).df()


def has_xg(con) -> bool:
    """Whether `scripts/build_xg.py` has been run against this database.

    Callers use this to offer the xG model only when it can actually be fitted,
    rather than letting the fit fail in front of a user who has no idea which
    script they were supposed to run first.
    """
    try:
        found = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'match_xg'"
        ).fetchone()
    except Exception:
        return False
    return found is not None
