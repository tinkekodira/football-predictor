"""Shared machinery for every model in the project.

Three ideas live here, and they are the answer to the problem that made the
descriptive layer untrustworthy on its own: two matchweeks into a season, a
team's raw averages are noise.

**Exponential time decay.** A match played 100 days ago tells you less about a
team than one played last week, but it is not worthless. Each match carries a
weight of 0.5 ** (age_in_days / half_life), so several seasons can be fitted at
once without an ancient result counting as much as a recent one. Dixon and
Coles introduced this in 1997 and everything since has kept it.

**Ridge priors.** The penalty term pulls every team's parameters towards a
prior mean, with a strength that is fixed while the evidence is not. A team
with three matches gets pulled most of the way back to the league average; a
team with a hundred barely moves. This is what stops a team that won its first
two games 5-0 from being modelled as the best side in Europe, and it is why the
model can produce a sane number in August at all.

**Prior means for teams without history.** Newly promoted sides have no
top-flight record, so shrinking them to the league average would flatter them.
They get a deliberately pessimistic prior instead.

Everything here is point-in-time: `as_of` is passed down from the caller and
every query filters on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

# Defaults. Phase 3 tunes these by walk-forward validation rather than taste;
# until then they are conventional values from the literature.
DEFAULT_HALF_LIFE_DAYS = 180.0
DEFAULT_RIDGE = 5.0

# What team strengths are fitted to by default. A half-and-half blend of goals
# and expected goals beat plain goals on all four leagues that had no part in
# choosing it (mean -0.0033 log loss, range 0.0006), so it ships. Falls back to
# goals automatically when a database has no xG - see `predict.build_models`.
DEFAULT_TARGET = "blend"
DEFAULT_BLEND_WEIGHT = 0.5

# **Shrinkage has to be chosen per target, not once.** Ridge trades variance
# against bias, so the right amount depends on how noisy the thing being fitted
# is, and xG is markedly less noisy than goals. Measured on a holdout the two
# want different values, and using one number for both understates whichever
# target it does not suit:
#
#   goals   best near 2-5; at 0.5 it is already overfitting, and lowering
#           shrinkage on goals alone bought nothing out of sample (+0.00006
#           across four held-out leagues).
#   blend   best near 0.5-1.0, a broad flat plateau; the same drop here is
#           worth -0.0033, because less shrinkage is what lets the better
#           signal actually show.
#
# That interaction is why two earlier experiments came back negative: the
# previous tuner only varied ridge on goals, and the first xG comparison held
# ridge at 5 for both targets. Neither was wrong; both asked half the question.
RECOMMENDED_RIDGE = {"goals": 5.0, "xg": 1.0, "blend": 1.0}


def default_ridge(target: str = DEFAULT_TARGET) -> float:
    """Shrinkage suited to how noisy `target` is. See `RECOMMENDED_RIDGE`."""
    return RECOMMENDED_RIDGE.get(target, DEFAULT_RIDGE)

# Teams with fewer matches than this in the training window are treated as
# newcomers and given the promoted-team prior instead of the league average.
PROMOTED_MATCH_THRESHOLD = 25

# How much worse a promoted side is assumed to be, on the log-rate scale,
# before any of its own results are taken into account. -0.15 on both attack
# and defence is roughly "scores 14% fewer, concedes 16% more".
PROMOTED_ATTACK_PRIOR = -0.15
PROMOTED_DEFENCE_PRIOR = -0.15


def time_weights(
    dates: pd.Series | np.ndarray,
    as_of: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Exponential decay weights: 1.0 today, 0.5 one half-life ago.

    A half-life of zero or less disables decay, which is occasionally useful
    for diagnostics but is not a sensible way to fit a real model.
    """
    ages = (pd.Timestamp(as_of) - pd.to_datetime(pd.Series(dates))).dt.days.to_numpy()
    ages = np.clip(ages.astype(float), 0.0, None)
    if half_life_days <= 0:
        return np.ones_like(ages)
    return 0.5 ** (ages / float(half_life_days))


@dataclass
class TeamIndex:
    """Maps team names to positions in the parameter vector."""

    teams: list[str]
    position: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.position = {team: i for i, team in enumerate(self.teams)}

    def __len__(self) -> int:
        return len(self.teams)

    def encode(self, names: pd.Series | list[str]) -> np.ndarray:
        return np.array([self.position[name] for name in names], dtype=np.intp)

    def known(self, name: str) -> bool:
        return name in self.position


@dataclass
class TrainingSet:
    """Point-in-time training data for one league.

    `frame` holds only matches played strictly before `as_of`. Nothing
    downstream needs to filter again, and nothing can accidentally see the
    result of the match being predicted.
    """

    league: str
    as_of: dt.date
    frame: pd.DataFrame
    weights: np.ndarray
    index: TeamIndex
    home_idx: np.ndarray
    away_idx: np.ndarray

    @property
    def n_matches(self) -> int:
        return len(self.frame)

    @property
    def effective_n(self) -> float:
        """Sum of weights: how many matches the fit really has to work with."""
        return float(self.weights.sum())

    def match_counts(self) -> np.ndarray:
        """Unweighted matches per team, used to spot newly promoted sides."""
        counts = np.bincount(self.home_idx, minlength=len(self.index))
        counts += np.bincount(self.away_idx, minlength=len(self.index))
        return counts

    def subset(self, mask: np.ndarray) -> "TrainingSet":
        """A new TrainingSet over a row subset, keeping the same team index.

        Used when a model needs a column that is missing from some rows, for
        instance corners in leagues that never reported them: the goals model
        still gets every match, the corner model gets the ones with data.
        """
        return TrainingSet(
            league=self.league,
            as_of=self.as_of,
            frame=self.frame.loc[mask].reset_index(drop=True),
            weights=self.weights[mask.to_numpy() if hasattr(mask, "to_numpy") else mask],
            index=self.index,
            home_idx=self.home_idx[mask.to_numpy() if hasattr(mask, "to_numpy") else mask],
            away_idx=self.away_idx[mask.to_numpy() if hasattr(mask, "to_numpy") else mask],
        )


def _has_table(con, name: str) -> bool:
    """Whether a table exists, so an optional one can be joined conditionally."""
    try:
        found = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
    except Exception:  # pragma: no cover - dialects without information_schema
        return False
    return found is not None


def load_training_set(
    con,
    league: str,
    as_of: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_age_days: int | None = None,
    min_matches: int = 50,
) -> TrainingSet:
    """Pull every match in a league played before `as_of`.

    Args:
        max_age_days: optionally hard-cut old matches. Decay already makes
            them near-weightless, so this is mostly a speed knob.
        min_matches: below this the fit is refused outright rather than
            returning parameters nobody should act on.
    """
    # LEFT JOIN, and the whole join is skipped when match_xg does not exist, so
    # a database built before scripts/build_xg.py was run still fits. A model
    # asked for xG it does not have should fail in `fit_goals_model` with a
    # sentence saying which script to run, not here with a SQL error.
    xg_select, xg_join = "NULL AS home_xg, NULL AS away_xg", ""
    if _has_table(con, "match_xg"):
        xg_select = "x.home_xg, x.away_xg"
        xg_join = "LEFT JOIN match_xg x USING (match_id)"

    # Availability joins the same optional way. It is precomputed by
    # scripts/build_rosters.py rather than derived here, because it is a
    # rolling window over each team's earlier matches and a walk-forward refits
    # thousands of times; deriving it per fit would redo identical work.
    availability_select = (
        "NULL AS missing_starter_share_home, NULL AS missing_starter_share_away, "
        "NULL AS missing_xgchain_share_home, NULL AS missing_xgchain_share_away"
    )
    availability_join = ""
    if _has_table(con, "match_availability"):
        availability_select = (
            "a.missing_starter_share_home, a.missing_starter_share_away, "
            "a.missing_xgchain_share_home, a.missing_xgchain_share_away"
        )
        availability_join = "LEFT JOIN match_availability a USING (match_id)"

    sql = f"""
        SELECT m.match_id, m.date, m.home_team, m.away_team, m.referee,
               m.home_goals, m.away_goals, m.home_corners, m.away_corners,
               m.home_yellows, m.away_yellows, m.home_reds, m.away_reds,
               m.home_shots, m.away_shots, m.home_sot, m.away_sot,
               {xg_select},
               {availability_select}
        FROM matches m
        {xg_join}
        {availability_join}
        WHERE m.league = ? AND m.date < ?
    """
    params: list = [league, as_of]
    if max_age_days is not None:
        sql += " AND m.date >= ?"
        params.append(pd.Timestamp(as_of) - pd.Timedelta(days=max_age_days))
    sql += " ORDER BY m.date"

    frame = con.execute(sql, params).df()
    if len(frame) < min_matches:
        raise InsufficientData(
            f"{league}: only {len(frame)} matches before {as_of}, need {min_matches}. "
            "Load more history with scripts/build_database.py."
        )

    frame["cards_home"] = frame["home_yellows"].fillna(0) + frame["home_reds"].fillna(0)
    frame["cards_away"] = frame["away_yellows"].fillna(0) + frame["away_reds"].fillna(0)
    frame.loc[frame["home_yellows"].isna(), "cards_home"] = np.nan
    frame.loc[frame["away_yellows"].isna(), "cards_away"] = np.nan

    teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    index = TeamIndex(teams)
    return TrainingSet(
        league=league,
        as_of=as_of,
        frame=frame,
        weights=time_weights(frame["date"], as_of, half_life_days),
        index=index,
        home_idx=index.encode(frame["home_team"]),
        away_idx=index.encode(frame["away_team"]),
    )


class InsufficientData(RuntimeError):
    """Raised when there is too little history to fit anything honest."""


class MissingExpectedGoals(InsufficientData):
    """Raised when xG was asked for and the database has none.

    A subclass rather than a flag because the two cases deserve different
    handling and telling them apart matters. "This league has forty matches"
    is a reason to give up; "this database was built before build_xg.py
    existed" is a reason to fit on goals instead and say so. Catching the base
    class for both would silently turn a genuinely unfittable league into a
    successful fallback, which is the failure this project keeps finding.
    """


class ConvergenceWarning(RuntimeWarning):
    """Raised as a warning when the optimiser stops without converging."""


# --------------------------------------------------------------------------
# Parameter packing
# --------------------------------------------------------------------------
# Every team-rate model shares the same core layout:
#
#     theta = [intercept, home_advantage, attack(n_teams), defence(n_teams), ...]
#
# with the model-specific extras (Dixon-Coles rho, negative-binomial
# dispersion, referee effects) appended afterwards.

def unpack_core(theta: np.ndarray, n_teams: int):
    """Split the shared prefix off a parameter vector."""
    intercept = theta[0]
    home_advantage = theta[1]
    attack = theta[2 : 2 + n_teams]
    defence = theta[2 + n_teams : 2 + 2 * n_teams]
    extras = theta[2 + 2 * n_teams :]
    return intercept, home_advantage, attack, defence, extras


def linear_predictors(
    theta: np.ndarray, n_teams: int, home_idx: np.ndarray, away_idx: np.ndarray
):
    """Log rates for both sides of every match.

    log rate(home) = intercept + home_advantage + attack[home] - defence[away]
    log rate(away) = intercept                  + attack[away] - defence[home]

    Attack is "how much this team adds to its own rate"; defence is "how much
    this team subtracts from its opponent's". Higher defence is better.
    """
    intercept, home_advantage, attack, defence, _ = unpack_core(theta, n_teams)
    eta_home = intercept + home_advantage + attack[home_idx] - defence[away_idx]
    eta_away = intercept + attack[away_idx] - defence[home_idx]
    return eta_home, eta_away


def team_gradient(
    residual_home: np.ndarray,
    residual_away: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    n_teams: int,
):
    """Chain the per-match residuals through to the team parameters.

    `residual_*` is d(log-likelihood)/d(linear predictor) for each side. A
    team's attack appears in its own rate whether home or away; its defence
    appears with a negative sign in its opponent's rate.
    """
    d_attack = np.bincount(home_idx, weights=residual_home, minlength=n_teams)
    d_attack += np.bincount(away_idx, weights=residual_away, minlength=n_teams)
    d_defence = -np.bincount(away_idx, weights=residual_home, minlength=n_teams)
    d_defence -= np.bincount(home_idx, weights=residual_away, minlength=n_teams)
    d_intercept = residual_home.sum() + residual_away.sum()
    d_home_advantage = residual_home.sum()
    return d_intercept, d_home_advantage, d_attack, d_defence


def build_priors(
    training: TrainingSet,
    promoted_attack: float = PROMOTED_ATTACK_PRIOR,
    promoted_defence: float = PROMOTED_DEFENCE_PRIOR,
    threshold: int = PROMOTED_MATCH_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior means for attack and defence, one entry per team.

    Established teams are shrunk towards the league average (zero). Teams with
    almost no history in the window - newly promoted sides, mostly - are shrunk
    towards a weaker-than-average prior, because "unknown" and "average" are
    not the same claim and promoted teams are reliably worse.
    """
    counts = training.match_counts()
    newcomers = counts < threshold
    attack_prior = np.where(newcomers, promoted_attack, 0.0)
    defence_prior = np.where(newcomers, promoted_defence, 0.0)
    return attack_prior, defence_prior


def ridge_penalty(
    attack: np.ndarray,
    defence: np.ndarray,
    attack_prior: np.ndarray,
    defence_prior: np.ndarray,
    strength: float | tuple[float, float],
):
    """Penalty value and its gradient. Behaves like `strength` pseudo-matches
    of evidence that every team is exactly at its prior.

    `strength` may be a single number, or a pair applying separately to attack
    and defence. The pair exists because `models.hierarchical` estimates the
    two spreads from the data and they do not come out equal - defensive
    strength is the less variable of the two - and a single value has to split
    the difference. A scalar keeps behaving exactly as it always did.
    """
    if np.isscalar(strength):
        attack_strength = defence_strength = float(strength)
    else:
        attack_strength, defence_strength = (float(v) for v in strength)
    attack_gap = attack - attack_prior
    defence_gap = defence - defence_prior
    value = (
        attack_strength * np.sum(attack_gap**2)
        + defence_strength * np.sum(defence_gap**2)
    )
    return value, 2 * attack_strength * attack_gap, 2 * defence_strength * defence_gap


def minimise(
    objective,
    x0: np.ndarray,
    bounds=None,
    max_iterations: int = 2000,
) -> optimize.OptimizeResult:
    """Run L-BFGS-B on an objective returning (value, gradient)."""
    result = optimize.minimize(
        objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-11, "gtol": 1e-8},
    )
    return result
