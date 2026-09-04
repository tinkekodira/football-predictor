"""Learning how much to shrink, instead of being told.

The two real wins this project has had - fitting team strengths to a goals/xG
blend, and dropping the ridge from 5 to 1 - were both variance reduction, and
both were found by hand on a grid. This module does the same job from the data.

**The ridge is a prior in disguise.** `base.ridge_penalty` adds
`lambda * sum((beta - prior)^2)` to the negative log-likelihood, which is
exactly the log posterior of a model in which each team's strength is drawn
from `N(prior, tau^2)` with

    lambda = 1 / (2 * tau^2)

So the ridge of 5.0 that shipped for most of this project is a claim that team
strengths have a standard deviation of 0.32 in log-rate space, and the 1.0 that
ships now is a claim of 0.71. Nobody ever measured which is true, and there is
no reason the answer should be the same in the Bundesliga as in Serie A.

`tau` is estimable from the same data the strengths are. Estimating it turns a
penalised fit into a genuinely hierarchical one: team strengths at the first
level, the spread of strengths within a league at the second, with the second
level fitted rather than assumed. That is backlog item B5 - the shipped global
ridge knowingly makes D1 and F1 worse in exchange for improving I1 and SP1,
because one number cannot suit five leagues.

## How

Empirical Bayes by EM, with a Laplace approximation at the mode. Given a fit at
some `lambda`, the posterior over strengths is approximately Gaussian with
covariance `H^-1`, where `H` is the Hessian of the penalised objective. The EM
update for the prior variance is then

    tau^2 <- ( sum (beta_hat - prior)^2 + sum diag(H^-1) ) / p

which reads as "the spread we observed, plus the spread we could not resolve".
Dropping the second term is the classic mistake: it makes `tau` too small, the
ridge too large, and the model over-shrunk - which is the exact defect the
calibration-slope diagnostic keeps finding.

Attack and defence get separate variances. They are separate claims about the
world and there is no reason they should match; defensive strength is usually
the less variable of the two, and a single `tau` splits the difference.

## Three things that would make this wrong if left out

- **The fit is a quasi-likelihood, not a likelihood.** With `target="blend"`
  the response is an average of goals and xG, whose variance is well below its
  mean, so the Poisson `H = X'WX` overstates how much the data knows. That
  understates the posterior variance, understates `tau`, and lands on a ridge
  that is too big - which would leave the whole exercise reproducing the bug it
  is meant to fix. `dispersion` estimates the Pearson scale and `H` is divided
  by it.
- **Attack and defence are identified only up to a shift.** Adding a constant
  to every attack and every defence leaves every linear predictor unchanged, so
  the likelihood has an exact null direction and only the penalty makes `H`
  invertible. That is fine and needs no special handling: along a direction the
  data says nothing about, the posterior variance is exactly `tau^2` and the
  fitted gap is zero, so the direction contributes `tau^2` to the numerator and
  `1` to `p`, and cancels out of the update exactly. It neither biases the
  estimate nor has to be removed.
- **The Dixon-Coles term is left out of the Hessian.** It touches four
  scorelines and its curvature contribution is second-order next to the Poisson
  block, and with any target other than `"goals"` the first stage does not fit
  it at all. The approximation is deliberate; the fit itself is unaffected,
  since this module only chooses `lambda`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Wide enough to contain any believable answer and narrow enough that a fit
# that has gone wrong cannot return something absurd. In tau terms this is a
# strength standard deviation between 0.10 and 3.16 in log-rate space, against
# team parameters the fitter bounds at +/-3.0.
RIDGE_BOUNDS = (0.05, 50.0)

# The EM update is a fixed point and converges from either side, so a start in
# the middle of the plausible range costs at most a round or two. This is the
# value the five-league validation settled on for the blend target.
DEFAULT_INITIAL_RIDGE = 1.0

# Five rounds is enough for the update to settle to well inside the tolerance
# on real leagues; the loop sits inside a walk-forward that refits thousands of
# times, so an extra round is not free.
DEFAULT_MAX_ROUNDS = 5

# Relative change in lambda below which another round is not worth its cost.
# Log loss across the whole ridge plateau of 0.1 to 1.0 spans 0.0008, so a few
# percent of lambda is far below anything measurable.
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class RidgeEstimate:
    """What the empirical-Bayes loop settled on, and how it got there.

    Kept as a record rather than a bare pair because the interesting question
    is not only the answer but whether the loop converged and how far it moved
    from where it started - a league that runs to the bounds is telling you
    something about the data, not about the ridge.
    """

    attack: float
    defence: float
    tau_attack: float
    tau_defence: float
    dispersion: float
    rounds: int
    converged: bool
    history: tuple[tuple[float, float], ...] = field(default=())

    @property
    def pair(self) -> tuple[float, float]:
        return (self.attack, self.defence)

    def summary(self) -> str:
        state = "converged" if self.converged else f"stopped at {self.rounds} rounds"
        return (
            f"ridge attack {self.attack:.2f} (tau {self.tau_attack:.3f}), "
            f"defence {self.defence:.2f} (tau {self.tau_defence:.3f}), "
            f"dispersion {self.dispersion:.3f}, {state}"
        )


def design_matrix(training, n_teams: int) -> np.ndarray:
    """The linear predictors as a matrix, home rows then away rows.

    `base.linear_predictors` computes the same thing with fancy indexing, which
    is the right choice inside an objective that runs thousands of times. The
    Hessian needs the actual matrix, and building it here keeps the two
    definitions next to each other so a change to one is visibly a change to
    the other. The column order matches `unpack_core`: intercept, home
    advantage, attack, defence.
    """
    n = training.n_matches
    columns = 2 + 2 * n_teams
    out = np.zeros((2 * n, columns))
    home_rows = np.arange(n)
    away_rows = n + home_rows

    out[:, 0] = 1.0
    out[home_rows, 1] = 1.0
    out[home_rows, 2 + training.home_idx] = 1.0
    out[away_rows, 2 + training.away_idx] = 1.0
    out[home_rows, 2 + n_teams + training.away_idx] = -1.0
    out[away_rows, 2 + n_teams + training.home_idx] = -1.0
    return out


def fitted_rates(
    training,
    attack: np.ndarray,
    defence: np.ndarray,
    intercept: float,
    home_advantage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Home and away scoring rates implied by a fitted parameter set.

    The same clipping the objective applies, so the curvature is evaluated at
    the point the fitter actually stopped at rather than at an extrapolation
    of it.
    """
    eta_home = (
        intercept
        + home_advantage
        + attack[training.home_idx]
        - defence[training.away_idx]
    )
    eta_away = intercept + attack[training.away_idx] - defence[training.home_idx]
    return np.exp(np.clip(eta_home, -8.0, 3.0)), np.exp(np.clip(eta_away, -8.0, 3.0))


def dispersion(
    response: tuple[np.ndarray, np.ndarray],
    rates: tuple[np.ndarray, np.ndarray],
    weights: np.ndarray,
    effective_parameters: float,
) -> float:
    """Pearson scale: how far the response really scatters around its rate.

    One for a true Poisson, below one for anything smoother. The blend target
    is an average of goals and xG and lands well below one, which is the whole
    reason this function exists - treating a smoothed response as Poisson would
    credit the fit with information it does not have, and every downstream
    number would inherit the error.
    """
    observed = np.concatenate(response)
    expected = np.concatenate(rates)
    doubled = np.concatenate([weights, weights])
    safe = np.maximum(expected, 1e-8)
    chi_square = float(np.sum(doubled * (observed - expected) ** 2 / safe))
    residual_df = float(doubled.sum()) - effective_parameters
    if residual_df <= 1.0:
        return 1.0
    return max(chi_square / residual_df, 1e-3)


def posterior_state(
    training,
    attack: np.ndarray,
    defence: np.ndarray,
    intercept: float,
    home_advantage: float,
    ridge: float | tuple[float, float],
    response: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Posterior variances for every strength, plus the dispersion scale.

    Returns `(variance_attack, variance_defence, dispersion)`. The variances
    are the diagonal of the inverse Hessian of the penalised objective, which
    is the Laplace approximation to the posterior that the EM update needs.

    The dispersion is estimated first, against an effective parameter count
    rather than a raw one, and then divides the observed information - so a
    smoother response correctly buys less certainty about every team.
    """
    n_teams = len(training.index)
    lam_attack, lam_defence = as_pair(ridge)

    # A response with a hole in it produces NaN through the chi-square, out of
    # `dispersion`, and on into every variance - silently, because NaN
    # propagates without complaint. `fit_goals_model` filters unplayable rows
    # before it gets here; anything else calling in has to do the same, and
    # should hear about it rather than receive a frame of NaN.
    if not (np.isfinite(response[0]).all() and np.isfinite(response[1]).all()):
        raise ValueError(
            f"{training.league}: the response has missing values, so no variance "
            "can be estimated. Subset the training set to rows the target "
            "actually covers first."
        )

    matrix = design_matrix(training, n_teams)
    rate_home, rate_away = fitted_rates(
        training, attack, defence, intercept, home_advantage
    )
    # Poisson curvature: d2(-loglik)/d(eta)^2 is the rate itself.
    curvature = np.concatenate(
        [training.weights * rate_home, training.weights * rate_away]
    )
    information = matrix.T @ (curvature[:, None] * matrix)

    penalty_diagonal = np.zeros(information.shape[0])
    penalty_diagonal[2 : 2 + n_teams] = 2.0 * lam_attack
    penalty_diagonal[2 + n_teams :] = 2.0 * lam_defence

    # Effective parameters of the penalised fit, for the dispersion's residual
    # degrees of freedom: the trace of the hat matrix rather than the raw
    # parameter count, because a shrunk parameter costs less than a free one.
    penalised = information + np.diag(penalty_diagonal)
    effective_parameters = float(np.trace(_safe_inverse(penalised) @ information))

    scale = dispersion(
        response, (rate_home, rate_away), training.weights, effective_parameters
    )

    # Redo the inverse with the information scaled. The penalty is deliberately
    # left unscaled: the prior is a claim about how much team strengths vary,
    # not about how noisy the response is, and dividing it by the dispersion
    # would quietly turn one into the other.
    posterior = _safe_inverse(information / scale + np.diag(penalty_diagonal))
    variances = np.clip(np.diag(posterior), 0.0, None)
    return variances[2 : 2 + n_teams], variances[2 + n_teams :], scale


def variance_update(
    gaps: np.ndarray, posterior_variance: np.ndarray, floor: float = 1e-4
) -> float:
    """One EM step for a prior variance: observed spread plus unresolved spread.

    `gaps` are the fitted strengths minus their prior means, so a promoted team
    shrunk towards a weaker-than-average prior contributes its distance from
    *that* prior rather than from zero. Measuring from zero instead would read
    the promoted-team offset as evidence that strengths are more spread out
    than they are, and inflate every league that had promotions in its window.
    """
    if gaps.size == 0:
        return floor
    return max(float(np.mean(gaps**2) + np.mean(posterior_variance)), floor)


def ridge_from_variance(
    tau_squared: float, bounds: tuple[float, float] = RIDGE_BOUNDS
) -> float:
    """The penalty strength a prior variance implies, clipped to the bounds."""
    if tau_squared <= 0.0:
        return bounds[1]
    return float(np.clip(1.0 / (2.0 * tau_squared), *bounds))


def empirical_bayes_ridge(
    step,
    initial: float | tuple[float, float] = DEFAULT_INITIAL_RIDGE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    tolerance: float = DEFAULT_TOLERANCE,
    bounds: tuple[float, float] = RIDGE_BOUNDS,
) -> RidgeEstimate:
    """Iterate the EM update to a fixed point.

    `step(ridge_pair)` refits at that penalty and returns
    `(tau2_attack, tau2_defence, dispersion)`. Taking it as a callback keeps
    the loop free of any knowledge of how a model is fitted, which is what lets
    the tests drive it with a constructed fixed point instead of a real league.

    Convergence is judged on the *relative* move in lambda, because the useful
    range spans three orders of magnitude: an absolute tolerance would be far
    too tight at one end and meaningless at the other.
    """
    current = as_pair(initial)
    history: list[tuple[float, float]] = [current]
    tau_attack = tau_defence = float("nan")
    scale = 1.0
    converged = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        tau2_attack, tau2_defence, scale = step(current)
        if not (np.isfinite(tau2_attack) and np.isfinite(tau2_defence)):
            raise ValueError(
                "The empirical-Bayes step returned a non-finite variance at "
                f"ridge {current}. Something upstream produced NaN; do not "
                "treat this as a converged estimate."
            )
        tau_attack, tau_defence = np.sqrt(tau2_attack), np.sqrt(tau2_defence)
        proposal = (
            ridge_from_variance(tau2_attack, bounds),
            ridge_from_variance(tau2_defence, bounds),
        )
        history.append(proposal)
        moved = max(
            abs(proposal[i] - current[i]) / max(current[i], 1e-9) for i in (0, 1)
        )
        current = proposal
        if moved < tolerance:
            converged = True
            break

    return RidgeEstimate(
        attack=current[0],
        defence=current[1],
        tau_attack=float(tau_attack),
        tau_defence=float(tau_defence),
        dispersion=float(scale),
        rounds=rounds,
        converged=converged,
        history=tuple(history),
    )


# How far apart the attack and defence penalties are allowed to be pulled. The
# measured ratio of the two variances sits between 1.05 and 1.55 across five
# leagues and four windows, so this only ever fires on an estimate that has
# gone wrong.
MAX_SPLIT_RATIO = 4.0


def split_ridge(
    tau_squared_attack: float,
    tau_squared_defence: float,
    level: float,
    max_ratio: float = MAX_SPLIT_RATIO,
) -> tuple[float, float]:
    """Keep a known-good overall penalty, take only its split from the data.

    The full empirical-Bayes estimate sets both the *level* of shrinkage and
    the *balance* between attack and defence, and the level is the part that
    cannot be trusted here - see the module docstring's note on the weighted
    likelihood. The balance is a comparison between two parameter blocks
    inside one fit, on one likelihood scale, so whatever distorts the level
    distorts both sides of it and largely cancels.

    So this holds the geometric mean of the two penalties at `level` - the
    value the five-league validation settled on - and moves them apart by the
    ratio the data implies. `level` alone reproduces the shipped behaviour
    exactly when the two variances come out equal.
    """
    if tau_squared_attack <= 0.0 or tau_squared_defence <= 0.0:
        return (float(level), float(level))
    ratio = float(
        np.clip(tau_squared_attack / tau_squared_defence, 1.0 / max_ratio, max_ratio)
    )
    root = np.sqrt(ratio)
    return (float(level) / root, float(level) * root)


def pool_variances(
    per_league: dict[str, float], weight: float = 0.5
) -> dict[str, float]:
    """Shrink each league's fitted variance towards the common one.

    The second level of the hierarchy, and what makes this a multi-league model
    rather than five single-league ones. Each league's `tau^2` is itself an
    estimate from twenty-odd teams and carries real noise; five leagues
    agreeing on roughly one number is evidence that the truth is near that
    number, and a league whose estimate wanders off is more likely to be noisy
    than to be special.

    Pooling is done on the log scale, because a variance is positive and its
    sampling error is roughly multiplicative. `weight` is how far to pull: 0
    keeps every league's own estimate, 1 gives every league the common one.

    With five leagues there is nowhere near enough information to fit `weight`
    from the data as well, so it is a stated choice rather than an estimate,
    and the validation reports what happens at both ends of it.
    """
    if not per_league:
        return {}
    share = float(np.clip(weight, 0.0, 1.0))
    logs = {name: np.log(max(value, 1e-8)) for name, value in per_league.items()}
    common = float(np.mean(list(logs.values())))
    return {
        name: float(np.exp((1.0 - share) * value + share * common))
        for name, value in logs.items()
    }


def as_pair(value: float | tuple[float, float]) -> tuple[float, float]:
    """Accept either a scalar ridge or a separate one for attack and defence."""
    if np.isscalar(value):
        return (float(value), float(value))
    first, second = value
    return (float(first), float(second))


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert, falling back to a pseudo-inverse on a singular matrix.

    The likelihood alone is singular - see the module docstring on the shift
    that leaves every linear predictor unchanged - so this only matters at the
    very bottom of `RIDGE_BOUNDS`, where the penalty may not lift the null
    direction far enough for a plain solve.
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - needs a degenerate fit
        return np.linalg.pinv(matrix)
