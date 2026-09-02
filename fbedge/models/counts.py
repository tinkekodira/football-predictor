"""Negative binomial models for corners and cards.

Goals are close enough to Poisson that a Poisson model is a reasonable start.
Corners and cards are not: both are **overdispersed**, meaning the variance of
the count exceeds its mean. A Poisson fit would therefore be too confident
about the middle and systematically underprice the tails - which is precisely
where over/under corner and card lines are set, and precisely where the money
is. The negative binomial adds a dispersion parameter that widens the
distribution to match what actually happens.

Two further details this module gets right:

**Referees.** Card counts depend as much on who is refereeing as on who is
playing. Each referee with enough matches gets their own multiplier, heavily
shrunk because even a busy referee handles perhaps thirty matches a season.
Referees below the threshold, and matches where the source records no referee
at all, fall back to a neutral effect rather than being dropped.

**The total is modelled separately from the two teams.** Adding two independent
negative binomials would understate the spread of the match total, because both
teams win more corners in an open game than a closed one - the counts are
positively correlated. So the team rates are fitted first, and then a second,
one-parameter fit estimates how the observed match totals actually scatter
around their predicted means. Team markets use the team distributions; totals
markets use the total distribution.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import base
from .base import (
    ConvergenceWarning,
    InsufficientData,
    TrainingSet,
    build_priors,
    minimise,
    ridge_penalty,
    team_gradient,
)

# A referee needs at least this many matches in the training window before
# being given their own parameter.
MIN_REFEREE_MATCHES = 12

# Referee effects are shrunk much harder than team effects: the samples are
# smaller and the temptation to over-read them is larger.
REFEREE_RIDGE_MULTIPLIER = 4.0

COLUMN_SETS = {
    "corners": ("home_corners", "away_corners"),
    "cards": ("cards_home", "cards_away"),
}


@dataclass
class CountModel:
    """A fitted count model for one league, one metric, one point in time."""

    kind: str
    league: str
    as_of: dt.date
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_advantage: float
    dispersion: float
    total_dispersion: float
    referee_effects: dict[str, float] = field(default_factory=dict)
    n_matches: int = 0
    effective_n: float = 0.0
    converged: bool = True
    match_counts: np.ndarray | None = None

    def _position(self, team: str) -> int | None:
        try:
            return self.teams.index(team)
        except ValueError:
            return None

    def rates(
        self, home_team: str, away_team: str, referee: str | None = None
    ) -> tuple[float, float]:
        """Expected count for each side.

        An unknown team or an unknown referee contributes nothing rather than
        raising: the league average is the right answer when there is no
        information, and pretending otherwise would be worse.
        """
        home_pos, away_pos = self._position(home_team), self._position(away_team)
        home_attack = self.attack[home_pos] if home_pos is not None else 0.0
        home_defence = self.defence[home_pos] if home_pos is not None else 0.0
        away_attack = self.attack[away_pos] if away_pos is not None else 0.0
        away_defence = self.defence[away_pos] if away_pos is not None else 0.0
        referee_effect = self.referee_effects.get(referee or "", 0.0)

        mu_home = np.exp(
            self.intercept + self.home_advantage + home_attack - away_defence + referee_effect
        )
        mu_away = np.exp(self.intercept + away_attack - home_defence + referee_effect)
        return float(mu_home), float(mu_away)

    def team_distribution(
        self, mean: float, max_count: int = 40
    ) -> np.ndarray:
        """Distribution of one team's count, using the team-level dispersion."""
        return _nb_pmf(mean, self.dispersion, max_count)

    def total_distribution(
        self,
        home_team: str,
        away_team: str,
        referee: str | None = None,
        max_count: int = 40,
    ) -> np.ndarray:
        """Distribution of the match total.

        Uses the separately estimated total dispersion, not the convolution of
        the two team distributions, for the correlation reason in the module
        docstring.
        """
        mu_home, mu_away = self.rates(home_team, away_team, referee)
        return _nb_pmf(mu_home + mu_away, self.total_dispersion, max_count)

    def is_known(self, team: str) -> bool:
        return self._position(team) is not None

    def referee_table(self) -> pd.DataFrame:
        """Referee multipliers, strictest first."""
        if not self.referee_effects:
            return pd.DataFrame(columns=["referee", "effect", "multiplier"])
        frame = pd.DataFrame(
            {
                "referee": list(self.referee_effects),
                "effect": list(self.referee_effects.values()),
            }
        )
        frame["multiplier"] = np.exp(frame["effect"])
        return frame.sort_values("multiplier", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        return (
            f"{self.kind.title()} (negative binomial) | {self.league} | "
            f"as of {self.as_of} | {self.n_matches} matches | "
            f"team dispersion {self.dispersion:.1f}, total dispersion "
            f"{self.total_dispersion:.1f} | {len(self.referee_effects)} referees"
            + ("" if self.converged else " | DID NOT CONVERGE")
        )


def _nb_pmf(mean: float, dispersion: float, max_count: int) -> np.ndarray:
    """Negative binomial probabilities for 0..max_count, renormalised.

    Parameterised by mean and dispersion k, so the variance is
    mean + mean^2 / k. Large k approaches Poisson.
    """
    mean = max(float(mean), 1e-6)
    dispersion = max(float(dispersion), 1e-6)
    counts = np.arange(max_count + 1)
    probability = dispersion / (dispersion + mean)
    pmf = stats.nbinom.pmf(counts, dispersion, probability)
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def _negative_binomial_terms(y: np.ndarray, mu: np.ndarray, k: float):
    """Log-likelihood and the derivatives needed for the gradient.

    Returns (log_lik, d/d_eta, d/d_log_k) where eta is the log rate.
    """
    k = max(k, 1e-6)
    log_lik = (
        special.gammaln(y + k)
        - special.gammaln(k)
        - special.gammaln(y + 1.0)
        + k * (np.log(k) - np.log(k + mu))
        + y * (np.log(mu) - np.log(k + mu))
    )
    d_eta = (y - mu) * k / (k + mu)
    d_k = (
        special.digamma(y + k)
        - special.digamma(k)
        + np.log(k)
        - np.log(k + mu)
        + (mu - y) / (k + mu)
    )
    return log_lik, d_eta, d_k * k  # chain rule for log k


def fit_count_model(
    training: TrainingSet,
    kind: str,
    ridge: float = base.DEFAULT_RIDGE,
    use_referee: bool | None = None,
    min_matches: int = 50,
) -> CountModel:
    """Fit a negative binomial team-rate model for corners or cards.

    Args:
        kind: "corners" or "cards".
        use_referee: defaults to True for cards, False for corners. Referees
            plainly affect bookings; the evidence that they affect corners is
            much weaker, so the parameters are not spent there by default.

    Raises:
        InsufficientData: when the league-season has too few matches carrying
            this statistic. Corners and referees are missing from several
            league-seasons in the source, so this is a normal outcome rather
            than an error, and callers should handle it.
    """
    if kind not in COLUMN_SETS:
        raise ValueError(f"kind must be one of {sorted(COLUMN_SETS)}, got {kind!r}")
    if use_referee is None:
        use_referee = kind == "cards"

    home_col, away_col = COLUMN_SETS[kind]
    frame = training.frame
    usable = frame[home_col].notna() & frame[away_col].notna()
    if usable.sum() < min_matches:
        raise InsufficientData(
            f"{training.league}: only {int(usable.sum())} matches carry {kind} data "
            f"before {training.as_of} (need {min_matches}). This source does not "
            "report it for every league and season."
        )

    training = training.subset(usable)
    n_teams = len(training.index)
    objective, start, bounds, referee_names, referee_idx = build_count_objective(
        training, kind, ridge=ridge, use_referee=use_referee
    )
    n_referees = len(referee_names)

    result = minimise(objective, start, bounds=bounds)
    if not result.success:
        warnings.warn(
            f"{kind} model for {training.league} did not converge: {result.message}",
            ConvergenceWarning,
            stacklevel=2,
        )
    return _assemble_count_model(
        result.x, training, kind, n_teams, referee_names, referee_idx, n_referees,
        converged=bool(result.success),
    )


def build_count_objective(
    training: TrainingSet,
    kind: str,
    ridge: float = base.DEFAULT_RIDGE,
    use_referee: bool | None = None,
):
    """Build the objective the count fitter minimises, plus its starting point.

    Exposed for the same reason as the goals equivalent: so the test suite can
    gradient-check the function that actually runs.
    """
    if use_referee is None:
        use_referee = kind == "cards"
    home_col, away_col = COLUMN_SETS[kind]
    frame = training.frame
    n_teams = len(training.index)

    x = frame[home_col].to_numpy(dtype=float)
    y = frame[away_col].to_numpy(dtype=float)
    weights = training.weights
    home_idx, away_idx = training.home_idx, training.away_idx
    attack_prior, defence_prior = build_priors(training)

    referee_names, referee_idx = _referee_index(frame, use_referee)
    n_referees = len(referee_names)

    def objective(theta: np.ndarray):
        intercept = theta[0]
        home_advantage = theta[1]
        attack = theta[2 : 2 + n_teams]
        defence = theta[2 + n_teams : 2 + 2 * n_teams]
        log_k = theta[2 + 2 * n_teams]
        referee = theta[3 + 2 * n_teams :]

        referee_term = referee[referee_idx] if n_referees else 0.0
        eta_home = np.clip(
            intercept + home_advantage + attack[home_idx] - defence[away_idx] + referee_term,
            -6.0, 5.0,
        )
        eta_away = np.clip(
            intercept + attack[away_idx] - defence[home_idx] + referee_term, -6.0, 5.0
        )
        mu_home, mu_away = np.exp(eta_home), np.exp(eta_away)
        k = float(np.exp(log_k))

        ll_home, d_eta_home, d_logk_home = _negative_binomial_terms(x, mu_home, k)
        ll_away, d_eta_away, d_logk_away = _negative_binomial_terms(y, mu_away, k)
        log_lik = float(np.sum(weights * (ll_home + ll_away)))

        residual_home = weights * d_eta_home
        residual_away = weights * d_eta_away
        d_log_k = float(np.sum(weights * (d_logk_home + d_logk_away)))

        penalty, d_penalty_attack, d_penalty_defence = ridge_penalty(
            attack, defence, attack_prior, defence_prior, ridge
        )
        d_intercept, d_hfa, d_attack, d_defence = team_gradient(
            residual_home, residual_away, home_idx, away_idx, n_teams
        )

        pieces = [
            [-d_intercept, -d_hfa],
            -d_attack + d_penalty_attack,
            -d_defence + d_penalty_defence,
            [-d_log_k],
        ]
        if n_referees:
            referee_ridge = ridge * REFEREE_RIDGE_MULTIPLIER
            penalty += referee_ridge * float(np.sum(referee**2))
            d_referee = np.bincount(
                referee_idx, weights=residual_home + residual_away, minlength=n_referees
            )
            pieces.append(-d_referee + 2 * referee_ridge * referee)

        return -log_lik + penalty, np.concatenate(pieces)

    start = np.zeros(3 + 2 * n_teams + n_referees)
    observed_mean = np.average(np.r_[x, y], weights=np.r_[weights, weights])
    start[0] = np.log(max(observed_mean, 0.2))
    start[2 + 2 * n_teams] = np.log(8.0)

    bounds = (
        [(-5.0, 5.0), (-2.0, 2.0)]
        + [(-2.0, 2.0)] * (2 * n_teams)
        + [(np.log(0.5), np.log(500.0))]
        + [(-1.0, 1.0)] * n_referees
    )
    return objective, start, bounds, referee_names, referee_idx


def _assemble_count_model(
    theta, training, kind, n_teams, referee_names, referee_idx, n_referees,
    converged: bool = True,
) -> CountModel:
    """Turn a fitted parameter vector into a CountModel."""
    home_col, away_col = COLUMN_SETS[kind]
    frame = training.frame
    x = frame[home_col].to_numpy(dtype=float)
    y = frame[away_col].to_numpy(dtype=float)

    intercept = float(theta[0])
    home_advantage = float(theta[1])
    attack = np.asarray(theta[2 : 2 + n_teams], dtype=float)
    defence = np.asarray(theta[2 + n_teams : 2 + 2 * n_teams], dtype=float)
    dispersion = float(np.exp(theta[2 + 2 * n_teams]))
    referee_effects = (
        {
            name: float(value)
            for name, value in zip(referee_names, theta[3 + 2 * n_teams :].tolist())
            if name != "(other)"
        }
        if n_referees
        else {}
    )

    total_dispersion = _fit_total_dispersion(
        observed=x + y,
        predicted_mean=_predicted_means(
            theta, n_teams, training.home_idx, training.away_idx, referee_idx, n_referees
        ),
        weights=training.weights,
    )

    return CountModel(
        kind=kind,
        league=training.league,
        as_of=training.as_of,
        teams=list(training.index.teams),
        attack=attack,
        defence=defence,
        intercept=intercept,
        home_advantage=home_advantage,
        dispersion=dispersion,
        total_dispersion=total_dispersion,
        referee_effects=referee_effects,
        n_matches=training.n_matches,
        effective_n=training.effective_n,
        converged=converged,
        match_counts=training.match_counts(),
    )


def _referee_index(frame: pd.DataFrame, use_referee: bool):
    """Referees with enough matches get a parameter; everyone else is neutral.

    Matches with no recorded referee, or a referee below the threshold, are
    mapped onto a shared bucket whose effect is pinned at zero, so they still
    contribute to every other parameter in the fit.
    """
    if not use_referee or "referee" not in frame.columns:
        return [], np.zeros(len(frame), dtype=np.intp)

    counts = frame["referee"].value_counts()
    eligible = sorted(counts[counts >= MIN_REFEREE_MATCHES].index.tolist())
    if not eligible:
        return [], np.zeros(len(frame), dtype=np.intp)

    # Index 0 is the neutral bucket; eligible referees start at 1. The neutral
    # parameter is included but pinned by its own ridge term to stay near zero.
    names = ["(other)"] + eligible
    position = {name: i for i, name in enumerate(names)}
    idx = frame["referee"].map(lambda r: position.get(r, 0)).to_numpy(dtype=np.intp)
    return names, idx


def _predicted_means(theta, n_teams, home_idx, away_idx, referee_idx, n_referees):
    """Predicted match totals from a fitted parameter vector."""
    intercept, home_advantage = theta[0], theta[1]
    attack = theta[2 : 2 + n_teams]
    defence = theta[2 + n_teams : 2 + 2 * n_teams]
    referee = theta[3 + 2 * n_teams :]
    referee_term = referee[referee_idx] if n_referees else 0.0
    mu_home = np.exp(intercept + home_advantage + attack[home_idx] - defence[away_idx] + referee_term)
    mu_away = np.exp(intercept + attack[away_idx] - defence[home_idx] + referee_term)
    return mu_home + mu_away


def _fit_total_dispersion(
    observed: np.ndarray, predicted_mean: np.ndarray, weights: np.ndarray
) -> float:
    """One-parameter fit of how match totals scatter around their means.

    The mean structure is already settled by the team-level fit; this only
    asks how wide the distribution around it needs to be. Fitting it on the
    totals directly captures the within-match correlation that treating the
    two teams as independent would miss.
    """

    def negative_log_likelihood(log_k: np.ndarray) -> float:
        k = float(np.exp(log_k[0]))
        log_lik, _, _ = _negative_binomial_terms(observed, predicted_mean, k)
        return -float(np.sum(weights * log_lik))

    result = optimize.minimize_scalar(
        lambda v: negative_log_likelihood(np.array([v])),
        bounds=(np.log(0.5), np.log(500.0)),
        method="bounded",
    )
    return float(np.exp(result.x))
