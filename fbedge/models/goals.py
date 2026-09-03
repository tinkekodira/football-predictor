"""The Dixon-Coles goals model.

Two independent Poisson distributions, one per side, with rates driven by team
attack and defence strengths and a home advantage. Dixon and Coles' 1997
contribution was to notice that independent Poissons get the low-scoring
results wrong - 0-0, 1-0, 0-1 and 1-1 happen at different rates than
independence predicts - and to add a four-parameter correction for exactly
those scorelines. Everything above 1-1 is left alone.

The model produces two scoring rates for a fixture. Expanding those into a
matrix of every plausible scoreline gives 1X2, over/under at any line, both
teams to score, Asian handicap and correct score from a single fit. That is
the main reason to model goals rather than markets: one coherent object, many
prices, all of them automatically consistent with each other.

What it does not include: xG (not in the free data source), lineups, injuries,
or European fixture congestion. Those are Phase 5.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import base
from .base import (
    ConvergenceWarning,
    InsufficientData,
    MissingExpectedGoals,
    TrainingSet,
    build_priors,
    linear_predictors,
    minimise,
    ridge_penalty,
    team_gradient,
    unpack_core,
)

# Rho is bounded well inside the region where the correction stays positive.
# Empirically it lands around -0.03 to -0.13 for European league football.
RHO_BOUNDS = (-0.25, 0.25)

# Guards against log(0) when the correction is pushed to its limit.
TAU_FLOOR = 1e-8


def tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float):
    """The Dixon-Coles low-score correction, and its partial derivatives.

    Applies only to 0-0, 0-1, 1-0 and 1-1; every other scoreline is untouched.
    Returns the correction alongside its derivatives with respect to the two
    log-rates and rho, so the optimiser can use an analytic gradient.
    """
    value = np.ones_like(lam)
    d_eta_home = np.zeros_like(lam)
    d_eta_away = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    is_00 = (x == 0) & (y == 0)
    is_01 = (x == 0) & (y == 1)
    is_10 = (x == 1) & (y == 0)
    is_11 = (x == 1) & (y == 1)

    value[is_00] = 1.0 - lam[is_00] * mu[is_00] * rho
    value[is_01] = 1.0 + lam[is_01] * rho
    value[is_10] = 1.0 + mu[is_10] * rho
    value[is_11] = 1.0 - rho
    value = np.maximum(value, TAU_FLOOR)

    # d value / d log-rate, i.e. multiplied through by the rate itself.
    d_eta_home[is_00] = -lam[is_00] * mu[is_00] * rho
    d_eta_away[is_00] = -lam[is_00] * mu[is_00] * rho
    d_eta_home[is_01] = lam[is_01] * rho
    d_eta_away[is_10] = mu[is_10] * rho

    d_rho[is_00] = -lam[is_00] * mu[is_00]
    d_rho[is_01] = lam[is_01]
    d_rho[is_10] = mu[is_10]
    d_rho[is_11] = -1.0
    return value, d_eta_home, d_eta_away, d_rho


@dataclass
class GoalsModel:
    """A fitted Dixon-Coles model for one league at one point in time."""

    league: str
    as_of: dt.date
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    half_life_days: float
    ridge: float
    n_matches: int
    effective_n: float
    converged: bool
    match_counts: np.ndarray
    # What the team strengths were fitted to: "goals", "xg" or "blend". Worth
    # carrying on the model rather than only in the caller, because two models
    # that differ only in this are otherwise indistinguishable downstream and
    # the whole point of the comparison is telling them apart.
    target: str = "goals"
    # How much a side's rate moves per unit of missing availability: the first
    # entry is the effect of its own absentees, the second of its opponent's.
    # (0.0, 0.0) means the model was fitted without availability and `rates`
    # then ignores whatever it is passed, so callers need not know either way.
    availability_beta: tuple[float, float] = (0.0, 0.0)

    # ----------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------

    def _position(self, team: str) -> int | None:
        try:
            return self.teams.index(team)
        except ValueError:
            return None

    def rates(
        self,
        home_team: str,
        away_team: str,
        missing_home: float = 0.0,
        missing_away: float = 0.0,
    ) -> tuple[float, float]:
        """Expected goals for each side.

        A team the model has never seen is given the promoted-team prior
        rather than an error: that is the honest answer for a club that has
        just come up, and the caller is told about it through `is_known`.

        `missing_home` and `missing_away` are availability shares from
        `fbedge.availability`, which reads only matches played earlier. They do
        nothing unless the model was fitted with `use_availability=True`, and
        default to zero so that every existing caller keeps its behaviour.
        """
        home_pos = self._position(home_team)
        away_pos = self._position(away_team)

        home_attack = self.attack[home_pos] if home_pos is not None else base.PROMOTED_ATTACK_PRIOR
        home_defence = self.defence[home_pos] if home_pos is not None else base.PROMOTED_DEFENCE_PRIOR
        away_attack = self.attack[away_pos] if away_pos is not None else base.PROMOTED_ATTACK_PRIOR
        away_defence = self.defence[away_pos] if away_pos is not None else base.PROMOTED_DEFENCE_PRIOR

        own, opp = self.availability_beta
        home_shift = own * float(missing_home) + opp * float(missing_away)
        away_shift = own * float(missing_away) + opp * float(missing_home)

        lam = np.exp(
            self.intercept + self.home_advantage + home_attack - away_defence + home_shift
        )
        mu = np.exp(self.intercept + away_attack - home_defence + away_shift)
        return float(lam), float(mu)

    def is_known(self, team: str) -> bool:
        return self._position(team) is not None

    def sample_size(self, team: str) -> int:
        position = self._position(team)
        return int(self.match_counts[position]) if position is not None else 0

    def score_matrix(
        self, home_team: str, away_team: str, max_goals: int = 12,
        missing_home: float = 0.0, missing_away: float = 0.0,
    ) -> np.ndarray:
        """Probability of every scoreline from 0-0 up to max_goals each.

        Truncating at 12 discards well under a thousandth of the mass; the
        matrix is renormalised so the probabilities still sum to one.
        """
        lam, mu = self.rates(home_team, away_team, missing_home, missing_away)
        return score_matrix_from_rates(lam, mu, self.rho, max_goals=max_goals)

    def ratings(self) -> pd.DataFrame:
        """Team strengths, strongest first, with the sample behind each.

        `overall` is attack plus defence: a single number for ranking. It is
        on the log scale, so a gap of 0.1 is roughly a 10% rate difference.
        """
        frame = pd.DataFrame(
            {
                "team": self.teams,
                "attack": self.attack,
                "defence": self.defence,
                "overall": self.attack + self.defence,
                "matches": self.match_counts,
            }
        )
        return frame.sort_values("overall", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        return (
            f"Dixon-Coles | {self.league} | as of {self.as_of} | "
            f"{self.n_matches} matches (effective {self.effective_n:.0f}) | "
            f"home advantage {np.exp(self.home_advantage):.3f}x | "
            f"rho {self.rho:+.3f}"
            + ("" if self.converged else " | DID NOT CONVERGE")
        )


def score_matrix_from_rates(
    lam: float, mu: float, rho: float, max_goals: int = 12
) -> np.ndarray:
    """Build the scoreline matrix for a given pair of rates."""
    from scipy import stats

    goals = np.arange(max_goals + 1)
    home_pmf = stats.poisson.pmf(goals, lam)
    away_pmf = stats.poisson.pmf(goals, mu)
    matrix = np.outer(home_pmf, away_pmf)

    # The low-score correction, applied to the four affected cells only.
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho

    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()
    if total <= 0:  # pragma: no cover - only reachable with absurd parameters
        raise ValueError("Score matrix has no probability mass.")
    return matrix / total


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _has_expected_goals(training: TrainingSet) -> bool:
    """Whether this training set carries usable xG at all."""
    frame = training.frame
    return "home_xg" in frame.columns and not frame["home_xg"].isna().all()


def responses(
    training: TrainingSet, target: str = "goals", blend_weight: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """The pair of quantities the rate model is fitted to.

    `"goals"` is what the model has always used. `"xg"` fits the same team
    strengths to expected goals instead, and `"blend"` to a weighted average of
    the two, with `blend_weight` the share given to xG.

    **Why fitting to a continuous quantity is legitimate here.** The objective
    is a Poisson log-likelihood with the factorial term dropped, because it does
    not depend on the parameters. What remains, `y*log(mu) - mu`, is a perfectly
    good quasi-likelihood for any non-negative response, and its score equation
    still sets the fitted rate to the weighted mean of the response. So xG can
    be substituted for goals without touching the arithmetic. What does have to
    change is the Dixon-Coles correction, which tests for exact 0-0, 1-0, 0-1
    and 1-1 scorelines and would never fire on a continuous input; see
    `fit_goals_model` for how that is handled.

    Raises:
        InsufficientData: when xG was asked for and the database has none.
            Silently falling back to goals would produce a model that quietly
            is not the one that was requested.
    """
    frame = training.frame
    goals = (
        frame["home_goals"].to_numpy(dtype=float),
        frame["away_goals"].to_numpy(dtype=float),
    )
    if target == "goals":
        return goals

    if "home_xg" not in frame.columns or frame["home_xg"].isna().all():
        raise MissingExpectedGoals(
            f"{training.league}: no expected-goals data before {training.as_of}. "
            "Run scripts/build_xg.py to download and attach it."
        )
    xg = (
        frame["home_xg"].to_numpy(dtype=float),
        frame["away_xg"].to_numpy(dtype=float),
    )
    if target == "xg":
        return xg
    if target == "blend":
        weight = float(np.clip(blend_weight, 0.0, 1.0))
        return (
            weight * xg[0] + (1.0 - weight) * goals[0],
            weight * xg[1] + (1.0 - weight) * goals[1],
        )
    raise ValueError(f"Unknown target {target!r}; use 'goals', 'xg' or 'blend'.")


def build_goals_objective(
    training: TrainingSet,
    ridge: float = base.DEFAULT_RIDGE,
    use_dixon_coles_correction: bool = True,
    response: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Build the objective the fitter minimises, and its starting point.

    Exposed rather than buried inside `fit_goals_model` so that the test suite
    can gradient-check the function that actually runs, instead of a copy of
    it that might drift. Returns (objective, start, bounds), where objective
    maps a parameter vector to (value, analytic gradient).

    `response` overrides what is being fitted; it defaults to actual goals.
    """
    frame = training.frame
    n_teams = len(training.index)
    x, y = response if response is not None else (
        frame["home_goals"].to_numpy(dtype=float),
        frame["away_goals"].to_numpy(dtype=float),
    )
    weights = training.weights
    home_idx, away_idx = training.home_idx, training.away_idx
    attack_prior, defence_prior = build_priors(training)

    def objective(theta: np.ndarray):
        intercept, home_advantage, attack, defence, extras = unpack_core(theta, n_teams)
        rho = float(extras[0]) if use_dixon_coles_correction else 0.0

        eta_home, eta_away = linear_predictors(theta, n_teams, home_idx, away_idx)
        eta_home = np.clip(eta_home, -8.0, 3.0)
        eta_away = np.clip(eta_away, -8.0, 3.0)
        lam, mu = np.exp(eta_home), np.exp(eta_away)

        # Poisson log-likelihood, dropping the constant factorial term.
        log_lik = weights * (x * eta_home - lam + y * eta_away - mu)
        residual_home = weights * (x - lam)
        residual_away = weights * (y - mu)
        d_rho = 0.0

        if use_dixon_coles_correction:
            correction, dc_home, dc_away, dc_rho = tau(x, y, lam, mu, rho)
            log_lik = log_lik + weights * np.log(correction)
            residual_home = residual_home + weights * dc_home / correction
            residual_away = residual_away + weights * dc_away / correction
            d_rho = float(np.sum(weights * dc_rho / correction))

        penalty, d_penalty_attack, d_penalty_defence = ridge_penalty(
            attack, defence, attack_prior, defence_prior, ridge
        )
        d_intercept, d_hfa, d_attack, d_defence = team_gradient(
            residual_home, residual_away, home_idx, away_idx, n_teams
        )

        # Negated, because the optimiser minimises.
        gradient = np.concatenate(
            [
                [-d_intercept, -d_hfa],
                -d_attack + d_penalty_attack,
                -d_defence + d_penalty_defence,
                [-d_rho] if use_dixon_coles_correction else [],
            ]
        )
        return -float(log_lik.sum()) + penalty, gradient

    start = np.zeros(2 + 2 * n_teams + (1 if use_dixon_coles_correction else 0))
    start[0] = np.log(max(np.average(np.r_[x, y], weights=np.r_[weights, weights]), 0.2))
    start[1] = 0.2
    if use_dixon_coles_correction:
        start[-1] = -0.05

    bounds = [(-5.0, 5.0), (-2.0, 2.0)] + [(-3.0, 3.0)] * (2 * n_teams)
    if use_dixon_coles_correction:
        bounds.append(RHO_BOUNDS)

    return objective, start, bounds


def recalibrate_on_goals(
    training: TrainingSet,
    attack: np.ndarray,
    defence: np.ndarray,
    use_dixon_coles_correction: bool = True,
) -> tuple[float, float, float]:
    """Refit only the level, the home advantage and rho, on actual goals.

    This is what makes an xG-fitted model usable for pricing goal markets.
    Expected goals and goals do not share a scale: across this database xG runs
    about 4% above goals for home sides, so feeding xG-derived rates straight
    into a scoreline matrix would over-predict every total in the book. Nor is
    home advantage necessarily the same in the two quantities, and rho is a
    correction for the way *goals* clump at 0-0 and 1-1 - a property of the
    discrete outcome that has no counterpart in a continuous expectation.

    So the division of labour is: the attack and defence strengths come from
    xG, because that is where xG is better; and the three parameters that
    describe how those strengths turn into goals are re-estimated on goals,
    because that is what is being predicted. Team strengths are held fixed here
    and not re-penalised, since they were already shrunk in the first stage.

    Returns (intercept, home_advantage, rho).
    """
    frame = training.frame
    x = frame["home_goals"].to_numpy(dtype=float)
    y = frame["away_goals"].to_numpy(dtype=float)
    weights = training.weights
    home_idx, away_idx = training.home_idx, training.away_idx
    attack_home, defence_away = attack[home_idx], defence[away_idx]
    attack_away, defence_home = attack[away_idx], defence[home_idx]

    def negative_log_likelihood(theta: np.ndarray) -> float:
        intercept, home_advantage = float(theta[0]), float(theta[1])
        rho = float(theta[2]) if use_dixon_coles_correction else 0.0
        eta_home = np.clip(intercept + home_advantage + attack_home - defence_away, -8.0, 3.0)
        eta_away = np.clip(intercept + attack_away - defence_home, -8.0, 3.0)
        lam, mu = np.exp(eta_home), np.exp(eta_away)
        log_lik = weights * (x * eta_home - lam + y * eta_away - mu)
        if use_dixon_coles_correction:
            correction, _, _, _ = tau(x, y, lam, mu, rho)
            log_lik = log_lik + weights * np.log(correction)
        return -float(log_lik.sum())

    start = np.array([np.log(max(np.average(np.r_[x, y], weights=np.r_[weights, weights]), 0.2)), 0.2, -0.05])
    bounds = [(-5.0, 5.0), (-2.0, 2.0), RHO_BOUNDS]
    if not use_dixon_coles_correction:
        start, bounds = start[:2], bounds[:2]

    from scipy import optimize

    result = optimize.minimize(
        negative_log_likelihood, start, method="L-BFGS-B", bounds=bounds
    )
    rho = float(result.x[2]) if use_dixon_coles_correction else 0.0
    return float(result.x[0]), float(result.x[1]), rho


AVAILABILITY_COLUMNS = ("missing_starter_share_home", "missing_starter_share_away")

# The effect of absence is bounded well inside anything plausible. Losing a
# whole first eleven is a share of 1.0, so a coefficient of -2 would be a 17%
# cut in the scoring rate for one missing regular; nothing real is larger, and
# a fit that wants to go further has found noise or a broken join.
AVAILABILITY_BOUNDS = (-2.0, 2.0)


def fit_availability_effect(
    training: TrainingSet,
    attack: np.ndarray,
    defence: np.ndarray,
    intercept: float,
    home_advantage: float,
    rho: float,
    columns: tuple[str, str] = AVAILABILITY_COLUMNS,
) -> tuple[float, float]:
    """How much a side's rate moves when players are missing.

    Fitted as a second stage with the team strengths held fixed, the same shape
    as `recalibrate_on_goals` and for the same reason: the strengths are
    already shrunk and re-penalising them here would let two parameters trade
    against twenty. Only two numbers are free, so a plain bounded optimiser is
    enough and no gradient is needed.

    `own` is the effect of a team's own absentees on its scoring rate and
    should come out negative; `opp` is the effect of its opponent's, and should
    come out positive, since a weakened defence concedes more. Both are on the
    log-rate scale, so multiply by a share of about one eleventh to read either
    as "the effect of one regular starter being out".

    Matches with no availability figure - early in a team's history, or before
    `scripts/build_rosters.py` was run - are dropped rather than filled with
    zero, since zero means "everyone fit" and would be a claim, not a gap.

    **How well it recovers a known effect.** Measured on synthetic leagues with
    the effect planted, `own` comes back essentially unbiased - a planted -0.60
    recovers as -0.65, -0.59 and -0.48 at 240, 720 and 2400 matches - but with
    a standard deviation near 0.09 even at 2400. Single-season estimates are
    therefore close to worthless and only a full history says anything. `opp`
    is the weaker of the two and runs high, recovering +0.40 to +0.55 from a
    planted +0.30, because the team-strength stage absorbs some of the
    structure it would otherwise pick up. Trust `own`; treat `opp` as
    directional.
    """
    frame = training.frame
    home_col, away_col = columns
    if home_col not in frame.columns or away_col not in frame.columns:
        return 0.0, 0.0

    usable = frame[home_col].notna() & frame[away_col].notna()
    if usable.sum() < 50:
        return 0.0, 0.0

    subset = training.subset(usable)
    frame = subset.frame
    x = frame["home_goals"].to_numpy(dtype=float)
    y = frame["away_goals"].to_numpy(dtype=float)
    weights = subset.weights
    miss_home = frame[home_col].to_numpy(dtype=float)
    miss_away = frame[away_col].to_numpy(dtype=float)

    base_home = (
        intercept + home_advantage
        + attack[subset.home_idx] - defence[subset.away_idx]
    )
    base_away = intercept + attack[subset.away_idx] - defence[subset.home_idx]

    def negative_log_likelihood(theta: np.ndarray) -> float:
        own, opp = float(theta[0]), float(theta[1])
        eta_home = np.clip(base_home + own * miss_home + opp * miss_away, -8.0, 3.0)
        eta_away = np.clip(base_away + own * miss_away + opp * miss_home, -8.0, 3.0)
        lam, mu = np.exp(eta_home), np.exp(eta_away)
        log_lik = weights * (x * eta_home - lam + y * eta_away - mu)
        correction, _, _, _ = tau(x, y, lam, mu, rho)
        log_lik = log_lik + weights * np.log(correction)
        return -float(log_lik.sum())

    from scipy import optimize

    result = optimize.minimize(
        negative_log_likelihood, np.zeros(2), method="L-BFGS-B",
        bounds=[AVAILABILITY_BOUNDS, AVAILABILITY_BOUNDS],
    )
    return float(result.x[0]), float(result.x[1])


def fit_goals_model(
    training: TrainingSet,
    ridge: float | None = None,
    half_life_days: float = base.DEFAULT_HALF_LIFE_DAYS,
    use_dixon_coles_correction: bool = True,
    target: str | None = None,
    blend_weight: float = base.DEFAULT_BLEND_WEIGHT,
    use_availability: bool = False,
) -> GoalsModel:
    """Fit by weighted, penalised maximum likelihood.

    The objective is the negative time-weighted log-likelihood plus the ridge
    penalty. Gradients are analytic, which makes the fit fast enough for Phase
    3 to refit thousands of times during a walk-forward backtest.

    `target` selects what the team strengths are fitted to: `"goals"` (the
    original behaviour), `"xg"`, or `"blend"`. Anything other than `"goals"`
    runs in two stages - strengths from the chosen target with the Dixon-Coles
    correction switched off, then level, home advantage and rho re-estimated on
    real goals by `recalibrate_on_goals`. See that function for why.
    """
    # `target=None` means "the default, and adapt if the data cannot support
    # it"; naming a target explicitly is a promise the caller wants kept, so
    # that case still raises. The distinction matters: a default quietly
    # degrading is helpful, a specific request quietly degrading is the bug
    # this project has been bitten by repeatedly.
    strict = target is not None
    if target is None:
        target = base.DEFAULT_TARGET
        if target != "goals" and not _has_expected_goals(training):
            target = "goals"
    elif target != "goals" and not _has_expected_goals(training) and strict:
        raise MissingExpectedGoals(
            f"{training.league}: no expected-goals data before {training.as_of}. "
            "Run scripts/build_xg.py to download and attach it."
        )

    # None means "whatever suits this target"; an explicit value always wins.
    if ridge is None:
        ridge = base.default_ridge(target)

    frame = training.frame
    playable = frame["home_goals"].notna() & frame["away_goals"].notna()
    if target != "goals":
        # A match without xG cannot contribute to the first stage, and letting
        # the two stages see different matches would make the recalibration
        # correct for a population the strengths were never fitted on.
        playable = playable & frame["home_xg"].notna() & frame["away_xg"].notna()
    if not playable.all():
        training = training.subset(playable)
    if training.n_matches < 50:
        raise InsufficientData(
            f"{training.league}: {training.n_matches} usable matches is too few to fit"
            + (" on xG. Run scripts/build_xg.py." if target != "goals" else ".")
        )

    n_teams = len(training.index)
    # The correction describes how discrete goals clump, so it is only fitted
    # in the first stage when the first stage is itself about goals.
    first_stage_dc = use_dixon_coles_correction and target == "goals"
    objective, start, bounds = build_goals_objective(
        training,
        ridge=ridge,
        use_dixon_coles_correction=first_stage_dc,
        response=responses(training, target, blend_weight),
    )

    result = minimise(objective, start, bounds=bounds)
    if not result.success:
        warnings.warn(
            f"Goals model for {training.league} did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=2,
        )

    intercept, home_advantage, attack, defence, extras = unpack_core(result.x, n_teams)
    attack = np.asarray(attack, dtype=float)
    defence = np.asarray(defence, dtype=float)

    if target == "goals":
        rho = float(extras[0]) if use_dixon_coles_correction else 0.0
    else:
        intercept, home_advantage, rho = recalibrate_on_goals(
            training, attack, defence, use_dixon_coles_correction
        )

    availability_beta = (0.0, 0.0)
    if use_availability:
        availability_beta = fit_availability_effect(
            training, attack, defence, intercept, home_advantage, rho
        )

    return GoalsModel(
        league=training.league,
        as_of=training.as_of,
        teams=list(training.index.teams),
        attack=np.asarray(attack, dtype=float),
        defence=np.asarray(defence, dtype=float),
        intercept=float(intercept),
        home_advantage=float(home_advantage),
        rho=rho,
        half_life_days=half_life_days,
        ridge=ridge,
        target=target,
        n_matches=training.n_matches,
        effective_n=training.effective_n,
        converged=bool(result.success),
        match_counts=training.match_counts(),
        availability_beta=availability_beta,
    )
