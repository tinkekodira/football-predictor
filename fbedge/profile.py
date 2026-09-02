"""The descriptive layer: what has actually happened, with sample sizes.

This is deliberately *not* a prediction. It answers "what do the real matches
say about this fixture" and nothing more. Two rules govern everything here:

**Point-in-time correctness.** Every query filters on `date < as_of`. Nothing
in this module can see a match that had not been played at the moment being
asked about. That is not just tidiness: Phase 3's walk-forward backtest calls
these same functions with historical as-of dates, and a single leak of future
information would make the backtest results meaningless.

**Sample size travels with the number.** Every statistic is a `Stat`, which
carries its own `n`. Two matches into a season, a team's "2.5 goals per game"
rests on two matches, and the interface has to say so. The model layer (Phase
2) is what turns these thin samples into usable estimates, by shrinking them
towards league averages and blending in earlier seasons; this layer's job is
to show the raw truth and be honest about how little of it there is.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config

# name -> (source column, aggregation, printable label)
METRICS: list[tuple[str, str, str, str]] = [
    ("goals_for", "goals_for", "mean", "Goals scored"),
    ("goals_against", "goals_against", "mean", "Goals conceded"),
    ("total_goals", "total_goals", "mean", "Total goals in match"),
    ("goals_ht", "goals_ht", "mean", "Goals, 1st half"),
    ("goals_2h", "goals_2h", "mean", "Goals, 2nd half"),
    ("btts", "btts", "rate", "Both teams scored"),
    ("over_1_5", "over_1_5", "rate", "Over 1.5 goals"),
    ("over_2_5", "over_2_5", "rate", "Over 2.5 goals"),
    ("over_3_5", "over_3_5", "rate", "Over 3.5 goals"),
    ("clean_sheet", "clean_sheet", "rate", "Clean sheet"),
    ("failed_to_score", "failed_to_score", "rate", "Failed to score"),
    ("corners_for", "corners_for", "mean", "Corners won"),
    ("corners_against", "corners_against", "mean", "Corners conceded"),
    ("total_corners", "total_corners", "mean", "Total corners in match"),
    ("cards_for", "cards_for", "mean", "Cards received"),
    ("total_cards", "total_cards", "mean", "Total cards in match"),
    ("shots_for", "shots_for", "mean", "Shots"),
    ("sot_for", "sot_for", "mean", "Shots on target"),
    ("shots_against", "shots_against", "mean", "Shots faced"),
    ("fouls_for", "fouls_for", "mean", "Fouls committed"),
    ("points", "points", "mean", "Points per game"),
]

RATE_METRICS = {name for name, _, agg, _ in METRICS if agg == "rate"}
METRIC_LABELS = {name: label for name, _, _, label in METRICS}


@dataclass(frozen=True)
class Stat:
    """A single statistic and the number of matches it was computed from."""

    value: float | None
    n: int

    @property
    def is_reliable(self) -> bool:
        """Crude flag for the interface. Ten matches is not a lot; it is
        simply the point below which a mean is more noise than signal."""
        return self.n >= 10

    def format(self, as_percent: bool = False, decimals: int = 2) -> str:
        if self.value is None or self.n == 0:
            return "-"
        if as_percent:
            return f"{self.value * 100:.0f}%"
        return f"{self.value:.{decimals}f}"


@dataclass
class TeamBlock:
    """One team's numbers across several scopes."""

    team: str
    scopes: dict[str, dict[str, Stat]] = field(default_factory=dict)
    form: list[dict[str, Any]] = field(default_factory=list)
    rest_days: int | None = None
    matches_last_14_days: int = 0

    def stat(self, scope: str, metric: str) -> Stat:
        return self.scopes.get(scope, {}).get(metric, Stat(None, 0))


@dataclass
class FixtureProfile:
    home_team: str
    away_team: str
    league: str | None
    league_name: str | None
    season: str
    as_of: dt.date
    home: TeamBlock
    away: TeamBlock
    league_baseline: dict[str, Stat]
    h2h: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

def _team_rows(con, team: str, as_of: dt.date, league: str | None = None) -> pd.DataFrame:
    """Every match this team played strictly before `as_of`, newest first."""
    sql = """
        SELECT * FROM team_matches
        WHERE team = ? AND date < ?
    """
    params: list = [team, as_of]
    if league:
        sql += " AND league = ?"
        params.append(league)
    sql += " ORDER BY date DESC"
    return _add_derived(con.execute(sql, params).df())


def _league_rows(
    con, league: str, season_start_year: int, as_of: dt.date
) -> pd.DataFrame:
    """Every team-match in the league this season, before `as_of`.

    Used as the baseline every team number is compared against. Without it,
    "1.4 goals per game" means nothing; against a league average it does.
    """
    return _add_derived(
        con.execute(
            """
            SELECT * FROM team_matches
            WHERE league = ? AND season_start_year = ? AND date < ?
            """,
            [league, season_start_year, as_of],
        ).df()
    )


def _add_derived(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the per-team-match flags that the raw view does not carry."""
    if rows.empty:
        for column in ("cards_for", "clean_sheet", "failed_to_score"):
            rows[column] = pd.Series(dtype="float64")
        return rows

    yellows = pd.to_numeric(rows["yellows_for"], errors="coerce")
    reds = pd.to_numeric(rows["reds_for"], errors="coerce")
    rows["cards_for"] = yellows + reds

    goals_against = pd.to_numeric(rows["goals_against"], errors="coerce")
    goals_for = pd.to_numeric(rows["goals_for"], errors="coerce")
    rows["clean_sheet"] = (goals_against == 0).where(goals_against.notna())
    rows["failed_to_score"] = (goals_for == 0).where(goals_for.notna())
    return rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(rows: pd.DataFrame) -> dict[str, Stat]:
    """Compute every metric over a set of team-match rows.

    `n` is counted per metric rather than per scope, because corners, shots
    and referee names are missing from some leagues and older seasons. A team
    can have 38 matches of goals and zero matches of corners in the same
    scope, and the interface needs to be able to say so.
    """
    result: dict[str, Stat] = {}
    for name, column, how, _label in METRICS:
        if rows.empty or column not in rows.columns:
            result[name] = Stat(None, 0)
            continue
        series = rows[column]
        if how == "rate":
            series = series.astype("object").map(
                lambda v: None if v is None or pd.isna(v) else bool(v)
            )
        result[name] = stat_from_series(series)
    return result


def stat_from_series(series: pd.Series) -> Stat:
    """Build a Stat from a column, tolerating all-null input.

    Needed because a column that exists but is entirely null (corners in the
    leagues that never reported them) has a mean of pd.NA, and float(pd.NA)
    raises rather than returning NaN.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    return Stat(float(values.mean()), int(values.size)) if values.size else Stat(None, 0)


def _form_lines(rows: pd.DataFrame, limit: int = 6) -> list[dict[str, Any]]:
    """The most recent results, as compact display records."""
    lines = []
    for _, row in rows.head(limit).iterrows():
        goals_for = row.get("goals_for")
        goals_against = row.get("goals_against")
        lines.append(
            {
                "date": row["date"],
                "opponent": row["opponent"],
                "venue": "H" if row["is_home"] else "A",
                "score": (
                    f"{int(goals_for)}-{int(goals_against)}"
                    if pd.notna(goals_for) and pd.notna(goals_against)
                    else "-"
                ),
                "outcome": row.get("outcome"),
                "competition": row.get("league_name"),
            }
        )
    return lines


def _rest_profile(rows: pd.DataFrame, as_of: dt.date) -> tuple[int | None, int]:
    """Days since the last match, and how many matches in the last 14 days.

    Schedule density is one of the few contextual effects with a real,
    repeatedly measured impact on performance, and it is free to compute here.
    It becomes a model feature in Phase 2.
    """
    if rows.empty:
        return None, 0
    dates = pd.to_datetime(rows["date"])
    last = dates.max()
    rest = (pd.Timestamp(as_of) - last).days
    recent = int((dates > pd.Timestamp(as_of) - pd.Timedelta(days=14)).sum())
    return int(rest), recent


def _head_to_head(
    con, home: str, away: str, as_of: dt.date, limit: int = 10
) -> dict[str, Any]:
    """Previous meetings between the two clubs, in any of the covered leagues.

    Shown because it is always asked for, and flagged because it is nearly
    always too small to mean anything: two league meetings per season means a
    decade of history is twenty matches, played by squads that have turned
    over several times. The models in Phase 2 do not use it.
    """
    frame = con.execute(
        """
        SELECT date, league_name, home_team, away_team, home_goals, away_goals,
               total_goals, btts, total_corners, total_cards
        FROM matches
        WHERE date < ?
          AND ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))
        ORDER BY date DESC
        LIMIT ?
        """,
        [as_of, home, away, away, home, limit],
    ).df()

    if frame.empty:
        return {"matches": [], "n": 0}

    home_wins = int(
        (((frame["home_team"] == home) & (frame["home_goals"] > frame["away_goals"]))
         | ((frame["away_team"] == home) & (frame["away_goals"] > frame["home_goals"]))).sum()
    )
    draws = int((frame["home_goals"] == frame["away_goals"]).sum())
    return {
        "matches": frame.to_dict("records"),
        "n": len(frame),
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": len(frame) - home_wins - draws,
        "avg_total_goals": stat_from_series(frame["total_goals"]),
        "btts_rate": stat_from_series(frame["btts"]),
        "avg_total_corners": stat_from_series(frame["total_corners"]),
    }


def _infer_league(con, home: str, away: str, as_of: dt.date) -> str | None:
    """The league both teams most recently played in before `as_of`."""
    row = con.execute(
        """
        SELECT league FROM matches
        WHERE date < ? AND (home_team IN (?, ?) OR away_team IN (?, ?))
        ORDER BY date DESC LIMIT 1
        """,
        [as_of, home, away, home, away],
    ).fetchone()
    return row[0] if row else None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

SCOPES = ("season", "season_venue", "last6", "last6_venue")

SCOPE_LABELS = {
    "season": "This season (all)",
    "season_venue": "This season (home/away)",
    "last6": "Last 6 (all)",
    "last6_venue": "Last 6 (home/away)",
}


def _build_team_block(
    con,
    team: str,
    as_of: dt.date,
    league: str | None,
    season_start_year: int,
    is_home: bool,
) -> TeamBlock:
    rows = _team_rows(con, team, as_of, league=league)
    season_rows = rows[rows["season_start_year"] == season_start_year]
    venue_rows = rows[rows["is_home"] == is_home]
    season_venue_rows = season_rows[season_rows["is_home"] == is_home]

    block = TeamBlock(team=team)
    block.scopes = {
        "season": aggregate(season_rows),
        "season_venue": aggregate(season_venue_rows),
        "last6": aggregate(rows.head(6)),
        "last6_venue": aggregate(venue_rows.head(6)),
    }
    block.form = _form_lines(rows, limit=6)
    block.rest_days, block.matches_last_14_days = _rest_profile(rows, as_of)
    return block


def fixture_profile(
    con,
    home_team: str,
    away_team: str,
    as_of: dt.date | str | None = None,
    league: str | None = None,
    season_start_year: int | None = None,
) -> FixtureProfile:
    """Build the descriptive card for one fixture.

    Args:
        con: an open DuckDB connection.
        home_team, away_team: canonical team names (use
            `normalize.resolve_team` on anything typed by a user).
        as_of: only matches strictly before this date are used. Defaults to
            today, which is what you want for an upcoming fixture; pass a past
            date to reconstruct what was knowable back then.
        league: restrict to one league. Inferred when omitted.
        season_start_year: which season counts as "this season".

    Returns:
        A FixtureProfile. Thin samples are reported, not hidden - check
        `warnings` and each Stat's `n`.
    """
    as_of = _coerce_date(as_of)
    season_start_year = season_start_year or config.CURRENT_SEASON_START_YEAR
    league = league or _infer_league(con, home_team, away_team, as_of)

    home_block = _build_team_block(con, home_team, as_of, league, season_start_year, True)
    away_block = _build_team_block(con, away_team, as_of, league, season_start_year, False)

    baseline: dict[str, Stat] = {}
    if league:
        baseline = aggregate(_league_rows(con, league, season_start_year, as_of))

    profile = FixtureProfile(
        home_team=home_team,
        away_team=away_team,
        league=league,
        league_name=config.LEAGUES.get(league) if league else None,
        season=config.season_label(season_start_year),
        as_of=as_of,
        home=home_block,
        away=away_block,
        league_baseline=baseline,
        h2h=_head_to_head(con, home_team, away_team, as_of),
    )
    profile.warnings = _collect_warnings(profile)
    return profile


def _coerce_date(value: dt.date | str | None) -> dt.date:
    if value is None:
        return dt.date.today()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    if isinstance(value, dt.datetime):
        return value.date()
    return value


def _collect_warnings(profile: FixtureProfile) -> list[str]:
    """Say plainly where the numbers are too thin to lean on."""
    warnings: list[str] = []
    for block in (profile.home, profile.away):
        n = block.stat("season", "goals_for").n
        if n == 0:
            warnings.append(f"{block.team} has no matches yet in {profile.season}.")
        elif n < 6:
            warnings.append(
                f"{block.team} has played {n} match{'es' if n != 1 else ''} in "
                f"{profile.season}. Season averages here are mostly noise."
            )
        if block.stat("season", "total_corners").n == 0 and n > 0:
            warnings.append(f"No corner data available for {block.team} this season.")

    if profile.h2h["n"] and profile.h2h["n"] < 6:
        warnings.append(
            f"Only {profile.h2h['n']} previous meeting(s) on record - too few to "
            "read anything into."
        )
    if not profile.league_baseline:
        warnings.append("League baseline unavailable; figures have no comparison point.")
    return warnings


# --------------------------------------------------------------------------
# Text rendering (used by the CLI; the Streamlit app renders its own tables)
# --------------------------------------------------------------------------

HEADLINE_METRICS = [
    "goals_for", "goals_against", "total_goals", "btts", "over_2_5",
    "total_corners", "total_cards", "sot_for", "points",
]


def format_profile(profile: FixtureProfile, scope: str = "season_venue") -> str:
    """Render a profile as a plain-text card."""
    width = 78
    lines = [
        "=" * width,
        f"{profile.home_team}  vs  {profile.away_team}",
        f"{profile.league_name or 'Unknown league'} | {profile.season} | "
        f"as of {profile.as_of.isoformat()}",
        "=" * width,
        "",
        f"Scope: {SCOPE_LABELS.get(scope, scope)}   "
        "(n = matches behind each figure)",
        "",
        f"{'':<26}{profile.home_team[:16]:>18}{profile.away_team[:16]:>18}{'League':>16}",
        "-" * width,
    ]

    for metric in HEADLINE_METRICS:
        as_pct = metric in RATE_METRICS
        home = profile.home.stat(scope, metric)
        away = profile.away.stat(scope, metric)
        base = profile.league_baseline.get(metric, Stat(None, 0))
        lines.append(
            f"{METRIC_LABELS[metric]:<26}"
            f"{home.format(as_pct) + f' (n={home.n})':>18}"
            f"{away.format(as_pct) + f' (n={away.n})':>18}"
            f"{base.format(as_pct):>16}"
        )

    lines += ["", "Recent form (most recent first)", "-" * width]
    for block in (profile.home, profile.away):
        results = " ".join(
            f"{line['outcome'] or '?'}"
            for line in block.form
        ) or "no matches on record"
        lines.append(f"  {block.team:<22} {results}")
        for line in block.form[:6]:
            date = pd.Timestamp(line["date"]).strftime("%d %b %y")
            lines.append(
                f"      {date}  {line['venue']}  vs {str(line['opponent'])[:20]:<20}"
                f" {line['score']:>6}"
            )
        rest = "unknown" if block.rest_days is None else f"{block.rest_days} days"
        lines.append(
            f"      rest: {rest}, {block.matches_last_14_days} match(es) in last 14 days"
        )
        lines.append("")

    h2h = profile.h2h
    lines += ["Head to head", "-" * width]
    if h2h["n"] == 0:
        lines.append("  No previous meetings on record.")
    else:
        lines.append(
            f"  {h2h['n']} meetings: {profile.home_team} {h2h['home_wins']}, "
            f"draws {h2h['draws']}, {profile.away_team} {h2h['away_wins']}"
        )
        lines.append(
            f"  avg total goals {h2h['avg_total_goals'].format()}, "
            f"BTTS {h2h['btts_rate'].format(as_percent=True)}, "
            f"avg corners {h2h['avg_total_corners'].format()}"
        )

    if profile.warnings:
        lines += ["", "Read with care", "-" * width]
        lines += [f"  - {w}" for w in profile.warnings]

    lines += ["", "=" * width]
    return "\n".join(lines)
