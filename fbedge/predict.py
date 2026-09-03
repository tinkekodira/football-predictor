"""The prediction entry point: one fixture in, a full set of prices out.

    from fbedge import database, predict
    con = database.connect()
    forecast = predict.predict_fixture(con, "Arsenal", "Liverpool")
    print(forecast.render())

Three models are fitted per league - goals, corners, cards - and every market
is derived from them. Fitting is fast enough to do on demand, and the results
are cached against a fingerprint of the underlying data so a rebuilt database
never serves stale parameters.

Nothing here can see a match played on or after `as_of`. That constraint runs
all the way down through `models.base.load_training_set`, and it is what will
let Phase 3 replay history honestly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from . import availability as availability_mod
from . import markets
from .models import base, counts, goals

# Fitted models, keyed by league, date, hyperparameters and a fingerprint of
# the data itself.
_MODEL_CACHE: dict[tuple, "ModelBundle"] = {}


@dataclass
class ModelBundle:
    """The three fitted models for one league at one moment."""

    league: str
    as_of: dt.date
    goals: goals.GoalsModel
    corners: counts.CountModel | None = None
    cards: counts.CountModel | None = None
    notes: list[str] = field(default_factory=list)

    def summaries(self) -> list[str]:
        out = [self.goals.summary()]
        for model in (self.corners, self.cards):
            if model is not None:
                out.append(model.summary())
        return out


@dataclass
class FixturePrediction:
    """Model output for one fixture."""

    home_team: str
    away_team: str
    league: str
    as_of: dt.date
    expected_goals: tuple[float, float]
    expected_corners: tuple[float, float] | None
    expected_cards: tuple[float, float] | None
    selections: list[markets.Selection]
    notes: list[str] = field(default_factory=list)
    model_summaries: list[str] = field(default_factory=list)

    def market(self, name: str) -> list[markets.Selection]:
        return [s for s in self.selections if s.market == name]

    def to_frame(self) -> pd.DataFrame:
        """Every selection as a table, with fair prices."""
        return pd.DataFrame(
            [
                {
                    "market": s.market,
                    "selection": s.selection,
                    "line": s.line,
                    "probability": s.probability,
                    "push": s.push_probability or None,
                    "fair_price": round(s.fair_price, 3),
                }
                for s in self.selections
            ]
        )

    def render(self) -> str:
        """A readable summary card for the terminal."""
        width = 66
        lines = [
            "=" * width,
            f"{self.home_team} vs {self.away_team}",
            f"{self.league} | model as of {self.as_of.isoformat()}",
            "=" * width,
            "",
            f"Expected goals    {self.expected_goals[0]:.2f} - {self.expected_goals[1]:.2f}",
        ]
        if self.expected_corners:
            lines.append(
                f"Expected corners  {self.expected_corners[0]:.1f} - "
                f"{self.expected_corners[1]:.1f}"
            )
        if self.expected_cards:
            lines.append(
                f"Expected cards    {self.expected_cards[0]:.1f} - "
                f"{self.expected_cards[1]:.1f}"
            )
        lines.append("")

        groups = [
            ("Match result", "1x2"),
            ("Total goals", "total_goals"),
            ("Both teams to score", "btts"),
            ("Total corners", "total_corners"),
            ("Total cards", "total_cards"),
            ("Asian handicap", "asian_handicap"),
            ("Most likely scores", "correct_score"),
        ]
        for title, market in groups:
            selections = self.market(market)
            if not selections:
                continue
            lines += [title, "-" * width]
            for s in selections:
                lines.append(
                    f"  {s.label:<22}{s.probability * 100:>7.1f}%"
                    f"{'   fair ' + format(s.fair_price, '.2f'):>16}"
                )
            lines.append("")

        if self.notes:
            lines += ["Read with care", "-" * width]
            lines += [f"  - {note}" for note in self.notes]
            lines.append("")
        lines += ["Model detail", "-" * width]
        lines += [f"  {summary}" for summary in self.model_summaries]
        lines.append("=" * width)
        return "\n".join(lines)


def _fingerprint(con, league: str) -> tuple:
    """Cheap signature of the league's data, so the cache invalidates itself."""
    row = con.execute(
        "SELECT COUNT(*), MAX(date) FROM matches WHERE league = ?", [league]
    ).fetchone()
    return (int(row[0]), str(row[1]))


def build_models(
    con,
    league: str,
    as_of: dt.date,
    half_life_days: float = base.DEFAULT_HALF_LIFE_DAYS,
    ridge: float | None = None,
    use_cache: bool = True,
    fit_counts: bool = True,
    target: str | None = None,
    blend_weight: float = base.DEFAULT_BLEND_WEIGHT,
    use_availability: bool = False,
) -> ModelBundle:
    """Fit goals, corners and cards models for one league.

    Corner and card models are optional by design: this source does not report
    those statistics for every league and season, so a missing model is a
    normal outcome that downgrades the output rather than failing it.

    `fit_counts=False` skips them entirely, which roughly triples the speed of
    a hyperparameter sweep that only scores goal markets.
    """
    key = (
        league, as_of, half_life_days, ridge, fit_counts, target, blend_weight,
        use_availability, _fingerprint(con, league),
    )
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    training = base.load_training_set(con, league, as_of, half_life_days=half_life_days)
    notes: list[str] = []
    goals_model = goals.fit_goals_model(
        training, ridge=ridge, half_life_days=half_life_days,
        target=target, blend_weight=blend_weight,
        use_availability=use_availability,
    )
    # `fit_goals_model` downgrades an unspecified target when the database has
    # no xG. It is silent about it by design - it has no idea whether the
    # caller cares - so the telling happens here, where `notes` exists and
    # reaches the user.
    if target is None and goals_model.target != base.DEFAULT_TARGET:
        notes.append(
            f"No expected-goals data for {league} before {as_of}, so team "
            f"strengths were fitted to goals rather than "
            f"{base.DEFAULT_TARGET!r}. Run scripts/build_xg.py to enable it; "
            "the blend scored about 0.003 better on log loss across four "
            "leagues that had no part in choosing it."
        )
    if not goals_model.converged:
        notes.append("The goals model did not fully converge; treat prices as indicative.")

    # The count models take a concrete number, and `ridge` may still be None
    # here meaning "whatever suits the target". The goals fit has already
    # resolved it, so reuse that rather than resolving it twice and risking
    # the two disagreeing after a fallback.
    ridge = goals_model.ridge

    optional: dict[str, counts.CountModel | None] = {"corners": None, "cards": None}
    if fit_counts:
        for kind in ("corners", "cards"):
            try:
                optional[kind] = counts.fit_count_model(training, kind, ridge=ridge)
            except base.InsufficientData as exc:
                notes.append(str(exc))

    bundle = ModelBundle(
        league=league,
        as_of=as_of,
        goals=goals_model,
        corners=optional["corners"],
        cards=optional["cards"],
        notes=notes,
    )
    if use_cache:
        _MODEL_CACHE[key] = bundle
    return bundle


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()


def _infer_league(con, home_team: str, away_team: str, as_of: dt.date) -> str | None:
    row = con.execute(
        """
        SELECT league FROM matches
        WHERE date < ? AND (home_team IN (?, ?) OR away_team IN (?, ?))
        ORDER BY date DESC LIMIT 1
        """,
        [as_of, home_team, away_team, home_team, away_team],
    ).fetchone()
    return row[0] if row else None


def _fixture_availability(con, league, home_team, away_team, as_of):
    """Missing-starter shares for a fixture, or zeros when unavailable."""
    if not base._has_table(con, "match_lineups"):
        return 0.0, 0.0
    lineups = con.execute("SELECT * FROM match_lineups").df()
    matches = con.execute(
        "SELECT match_id, date, home_team, away_team FROM matches WHERE league = ?",
        [league],
    ).df()
    if lineups.empty or matches.empty:
        return 0.0, 0.0
    return availability_mod.for_fixture(
        lineups, matches, home_team, away_team, as_of
    )


def predict_fixture(
    con,
    home_team: str,
    away_team: str,
    as_of: dt.date | str | None = None,
    league: str | None = None,
    referee: str | None = None,
    half_life_days: float = base.DEFAULT_HALF_LIFE_DAYS,
    ridge: float | None = None,
    max_goals: int = 12,
    target: str | None = None,
    blend_weight: float = base.DEFAULT_BLEND_WEIGHT,
    use_availability: bool = False,
) -> FixturePrediction:
    """Price every market for one fixture.

    Args:
        as_of: only matches before this date inform the models. Defaults to
            today, which is what you want for an upcoming fixture.
        referee: if known, applies that referee's fitted card multiplier.
            Usually unknown before kick-off, in which case cards are priced at
            the league-average referee.

    Returns:
        A FixturePrediction. Check `notes` before acting on anything: an
        unknown team or a missing corner model is reported there, not hidden.
    """
    as_of = _coerce_date(as_of)
    league = league or _infer_league(con, home_team, away_team, as_of)
    if league is None:
        raise ValueError(
            f"Cannot work out which league {home_team} vs {away_team} belongs to. "
            "Pass league= explicitly."
        )

    bundle = build_models(
        con, league, as_of, half_life_days=half_life_days, ridge=ridge,
        target=target, blend_weight=blend_weight,
        use_availability=use_availability,
    )
    notes = list(bundle.notes)

    for team in (home_team, away_team):
        if not bundle.goals.is_known(team):
            notes.append(
                f"{team} has no match history before {as_of}. Priced using the "
                "promoted-team prior, which is a guess rather than a measurement."
            )
        elif bundle.goals.sample_size(team) < base.PROMOTED_MATCH_THRESHOLD:
            notes.append(
                f"{team} has only {bundle.goals.sample_size(team)} matches of "
                "history, so its rating is mostly the prior, not its own results."
            )

    # An upcoming fixture has no row in `match_availability` - it has not been
    # played - so its availability is derived from the two sides' earlier
    # matches here. Only done when asked for, because it reads the whole
    # line-up table and most callers do not need it.
    missing_home, missing_away = 0.0, 0.0
    if use_availability:
        missing_home, missing_away = _fixture_availability(
            con, league, home_team, away_team, as_of
        )

    matrix = bundle.goals.score_matrix(
        home_team, away_team, max_goals=max_goals,
        missing_home=missing_home, missing_away=missing_away,
    )
    expected_goals = bundle.goals.rates(
        home_team, away_team, missing_home, missing_away
    )

    selections: list[markets.Selection] = []
    selections += markets.match_odds(matrix)
    selections += markets.double_chance(matrix)
    selections += markets.total_goals(matrix)
    selections += markets.both_teams_to_score(matrix)
    selections += markets.team_totals(matrix)
    for line in _handicap_lines(expected_goals):
        selections += markets.asian_handicap(matrix, line)
    selections += markets.correct_score(matrix)

    expected_corners = _add_count_market(
        selections, bundle.corners, home_team, away_team, referee,
        market="total_corners", default_lines=markets.DEFAULT_CORNER_LINES,
    )
    expected_cards = _add_count_market(
        selections, bundle.cards, home_team, away_team, referee,
        market="total_cards", default_lines=markets.DEFAULT_CARD_LINES,
    )
    if referee and bundle.cards and referee not in bundle.cards.referee_effects:
        notes.append(
            f"No fitted effect for referee {referee}; cards priced at the "
            "league-average referee."
        )

    return FixturePrediction(
        home_team=home_team,
        away_team=away_team,
        league=league,
        as_of=as_of,
        expected_goals=expected_goals,
        expected_corners=expected_corners,
        expected_cards=expected_cards,
        selections=selections,
        notes=notes,
        model_summaries=bundle.summaries(),
    )


def _add_count_market(
    selections: list[markets.Selection],
    model: counts.CountModel | None,
    home_team: str,
    away_team: str,
    referee: str | None,
    market: str,
    default_lines,
) -> tuple[float, float] | None:
    """Append over/under lines for a count market, if the model exists."""
    if model is None:
        return None
    rates = model.rates(home_team, away_team, referee)
    pmf = model.total_distribution(home_team, away_team, referee)
    lines = markets.suggested_lines(markets.count_mean(pmf))
    lines = tuple(sorted(set(lines) | set(default_lines)))
    selections += markets.count_totals(pmf, market, lines)
    return rates


def _handicap_lines(expected_goals: tuple[float, float]) -> list[float]:
    """Handicap lines bracketing the model's expected margin.

    The line is applied to the home team, so a home favourite gets a negative
    line: expecting to win by 1.65 means quoting around -1.75, not +1.75.
    Quoting lines near the model's own margin is more useful than a fixed
    ladder, because that is where a bookmaker will have set the line too.
    """
    margin = expected_goals[0] - expected_goals[1]
    centre = -round(margin * 4) / 4
    return [round(centre + step, 2) for step in (-0.5, -0.25, 0.0, 0.25, 0.5)]


def _coerce_date(value: dt.date | str | None) -> dt.date:
    if value is None:
        return dt.date.today()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    if isinstance(value, dt.datetime):
        return value.date()
    return value
