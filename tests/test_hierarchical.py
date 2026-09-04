"""Tests for empirical-Bayes shrinkage.

Two of these carry more weight than the rest, for the same reason the goals
shape tests do: the headline result of this feature is a **negative** one, and
a null is only worth acting on if the estimator could have found the effect.

- `test_recovers_a_planted_prior_variance` points the estimator at leagues
  built to have a known `tau`, at a realistic size, and checks it comes back.
  Without it, "empirical Bayes disagrees with the holdout" is indistinguishable
  from "the estimator is broken".
- `test_the_loop_runs_away_on_a_flat_likelihood` pins the failure mode that
  was actually observed on real data, so that if someone later makes the loop
  converge nicely they can tell whether they fixed it or hid it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fbedge.models import base, goals, hierarchical


# ----------------------------------------------------------------------
# Synthetic leagues
# ----------------------------------------------------------------------


def synthetic_league(
    tau: float = 0.25,
    n_teams: int = 20,
    seasons: int = 8,
    seed: int = 0,
    half_life: float = 180.0,
    season_days: int = 280,
):
    """A league whose team strengths really are drawn from `N(0, tau^2)`.

    Fixtures are spread across `season_days` with a summer break, because the
    time decay is the whole difficulty here: compressing eight seasons into one
    year leaves the weights nearly flat, the effective sample far larger than
    any real league's, and an estimator that looks better than it is.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0.0, tau, n_teams)
    defence = rng.normal(0.0, tau, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    fixtures = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
    start = dt.date(2017, 8, 1)
    rows = []
    for season in range(seasons):
        order = list(fixtures)
        rng.shuffle(order)
        for k, (home, away) in enumerate(order):
            date = start + dt.timedelta(
                days=season * 365 + int(k * season_days / len(order))
            )
            lam = np.exp(0.15 + 0.22 + attack[home] - defence[away])
            mu = np.exp(0.15 + attack[away] - defence[home])
            rows.append(
                (date, teams[home], teams[away], rng.poisson(lam), rng.poisson(mu))
            )

    frame = pd.DataFrame(
        rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"]
    ).sort_values("date").reset_index(drop=True)
    as_of = frame["date"].max() + dt.timedelta(days=1)
    index = base.TeamIndex(teams)
    training = base.TrainingSet(
        league="SYN",
        as_of=as_of,
        frame=frame,
        weights=base.time_weights(frame["date"], as_of, half_life),
        index=index,
        home_idx=index.encode(frame["home_team"]),
        away_idx=index.encode(frame["away_team"]),
    )
    return training, attack, defence


# ----------------------------------------------------------------------
# The algebra
# ----------------------------------------------------------------------


def test_ridge_and_prior_variance_are_inverses():
    """The correspondence the whole module rests on: lambda = 1 / (2 tau^2)."""
    assert hierarchical.ridge_from_variance(0.5) == pytest.approx(1.0)
    assert hierarchical.ridge_from_variance(0.05) == pytest.approx(10.0)


def test_ridge_from_variance_respects_the_bounds():
    assert hierarchical.ridge_from_variance(1e-12) == hierarchical.RIDGE_BOUNDS[1]
    assert hierarchical.ridge_from_variance(1e6) == hierarchical.RIDGE_BOUNDS[0]
    assert hierarchical.ridge_from_variance(0.0) == hierarchical.RIDGE_BOUNDS[1]


def test_as_pair_accepts_a_scalar_or_a_pair():
    assert hierarchical.as_pair(3.0) == (3.0, 3.0)
    assert hierarchical.as_pair((1.0, 2.0)) == (1.0, 2.0)


def test_variance_update_adds_the_unresolved_spread():
    """The term that stops the estimate collapsing. Dropping it is the classic
    mistake, so its contribution is pinned rather than assumed."""
    gaps = np.array([0.2, -0.2, 0.2, -0.2])
    without = hierarchical.variance_update(gaps, np.zeros(4))
    with_variance = hierarchical.variance_update(gaps, np.full(4, 0.03))
    assert without == pytest.approx(0.04)
    assert with_variance == pytest.approx(0.07)


def test_variance_update_measures_from_the_prior_not_from_zero():
    """A promoted team's prior is not zero, and measuring from zero would read
    the promoted-team offset as evidence of a wider spread of strengths."""
    strengths = np.array([-0.15, -0.15, -0.15])
    prior = np.full(3, -0.15)
    assert hierarchical.variance_update(strengths - prior, np.zeros(3)) < 1e-3
    assert hierarchical.variance_update(strengths, np.zeros(3)) == pytest.approx(0.0225)


def test_split_ridge_keeps_the_level_and_moves_the_balance():
    low, high = hierarchical.split_ridge(0.04, 0.01, level=1.0)
    # Attack varies more, so attack is penalised less; the geometric mean is
    # exactly the level that was validated.
    assert low < 1.0 < high
    assert np.sqrt(low * high) == pytest.approx(1.0)
    assert high / low == pytest.approx(4.0)


def test_split_ridge_is_a_no_op_when_the_variances_agree():
    assert hierarchical.split_ridge(0.02, 0.02, level=1.0) == pytest.approx((1.0, 1.0))


def test_split_ridge_clamps_an_absurd_ratio():
    low, high = hierarchical.split_ridge(1.0, 1e-6, level=1.0)
    assert high / low == pytest.approx(hierarchical.MAX_SPLIT_RATIO)


def test_pool_variances_shrinks_towards_the_common_value():
    pooled = hierarchical.pool_variances({"a": 0.01, "b": 0.04}, weight=1.0)
    assert pooled["a"] == pytest.approx(pooled["b"])
    assert pooled["a"] == pytest.approx(0.02)  # geometric mean


def test_pool_variances_at_zero_weight_changes_nothing():
    original = {"a": 0.01, "b": 0.04}
    pooled = hierarchical.pool_variances(original, weight=0.0)
    for key, value in original.items():
        assert pooled[key] == pytest.approx(value)


# ----------------------------------------------------------------------
# The design matrix must agree with the objective it approximates
# ----------------------------------------------------------------------


def test_design_matrix_reproduces_the_linear_predictors():
    """`design_matrix` and `base.linear_predictors` are two spellings of one
    model. If they ever drift apart the Hessian describes a different fit from
    the one that ran, and nothing downstream would notice."""
    training, _, _ = synthetic_league(seasons=2, n_teams=8, seed=3)
    n_teams = len(training.index)
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 0.3, 2 + 2 * n_teams)

    eta_home, eta_away = base.linear_predictors(
        theta, n_teams, training.home_idx, training.away_idx
    )
    product = hierarchical.design_matrix(training, n_teams) @ theta
    assert np.allclose(product[: training.n_matches], eta_home)
    assert np.allclose(product[training.n_matches :], eta_away)


def test_posterior_variance_falls_as_shrinkage_rises():
    """More penalty means a tighter posterior. Obvious, and the sign of this is
    what drives the whole fixed point, so it is worth a guard."""
    training, _, _ = synthetic_league(seasons=3, n_teams=10, seed=5)
    response = goals.responses(training, "goals", 0.5)
    model_low = goals.fit_goals_model(training, ridge=0.5, target="goals")
    model_high = goals.fit_goals_model(training, ridge=20.0, target="goals")

    low, _, _ = hierarchical.posterior_state(
        training, model_low.attack, model_low.defence, model_low.intercept,
        model_low.home_advantage, (0.5, 0.5), response,
    )
    high, _, _ = hierarchical.posterior_state(
        training, model_high.attack, model_high.defence, model_high.intercept,
        model_high.home_advantage, (20.0, 20.0), response,
    )
    assert high.mean() < low.mean()


def test_dispersion_is_about_one_for_poisson_goals():
    """The scale correction must not fire when the response really is Poisson,
    or every estimate on the goals target would be quietly wrong."""
    training, _, _ = synthetic_league(seasons=4, n_teams=14, seed=9)
    response = goals.responses(training, "goals", 0.5)
    model = goals.fit_goals_model(training, ridge=1.0, target="goals")
    _, _, scale = hierarchical.posterior_state(
        training, model.attack, model.defence, model.intercept,
        model.home_advantage, (1.0, 1.0), response,
    )
    assert 0.85 < scale < 1.15


def test_posterior_state_refuses_a_response_with_holes():
    """NaN propagates silently all the way to the ridge; the caller hears about
    it instead. This fired for real on a league with partial xG coverage."""
    training, _, _ = synthetic_league(seasons=2, n_teams=8, seed=1)
    response = goals.responses(training, "goals", 0.5)
    holed = (response[0].copy(), response[1].copy())
    holed[0][0] = np.nan
    model = goals.fit_goals_model(training, ridge=1.0, target="goals")
    with pytest.raises(ValueError, match="missing values"):
        hierarchical.posterior_state(
            training, model.attack, model.defence, model.intercept,
            model.home_advantage, (1.0, 1.0), holed,
        )


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


def test_empirical_bayes_finds_a_constructed_fixed_point():
    """Driven by a step function with a known answer, so the loop is tested
    apart from everything that makes a real fit hard."""
    target = 0.05  # tau^2, so lambda = 10

    def step(pair):
        return target, target, 1.0

    estimate = hierarchical.empirical_bayes_ridge(step, initial=1.0)
    assert estimate.converged
    assert estimate.attack == pytest.approx(10.0)
    assert estimate.defence == pytest.approx(10.0)
    assert estimate.rounds == 2  # one to move there, one to see it stayed


def test_empirical_bayes_stops_at_max_rounds_without_claiming_convergence():
    def step(pair):
        # Always asks for twice the current penalty: never settles.
        return 1.0 / (4.0 * max(pair[0], 1e-9)), 1.0 / (4.0 * max(pair[1], 1e-9)), 1.0

    estimate = hierarchical.empirical_bayes_ridge(step, initial=1.0, max_rounds=4)
    assert not estimate.converged
    assert estimate.rounds == 4


def test_empirical_bayes_refuses_a_non_finite_step():
    def step(pair):
        return float("nan"), 0.05, 1.0

    with pytest.raises(ValueError, match="non-finite"):
        hierarchical.empirical_bayes_ridge(step)


def test_recovers_a_planted_prior_variance():
    """**The power test.** The headline result of this feature is that
    empirical Bayes disagrees with the out-of-sample optimum. That is only
    interesting if the estimator works, so here it is pointed at leagues whose
    `tau` is known, at a realistic number of teams and seasons.

    The tolerance is wide on purpose - the estimate carries real sampling noise
    at this size, which is itself part of the finding - but it is far tighter
    than the gap between what this returns on real data (lambda above 10) and
    the value that actually predicts best (lambda near 1).
    """
    for tau, seed in [(0.15, 3), (0.25, 4), (0.40, 5)]:
        training, _, _ = synthetic_league(tau=tau, seed=seed, seasons=6, n_teams=18)
        model = goals.fit_goals_model(training, ridge="auto", target="goals")
        estimate = model.ridge_estimate
        recovered = 0.5 * (estimate.tau_attack + estimate.tau_defence)
        assert 0.6 * tau < recovered < 1.7 * tau, (
            f"planted tau={tau}, recovered {recovered:.3f}"
        )


def test_the_loop_runs_away_on_a_flat_likelihood():
    """Pins the failure mode seen on real leagues, so a later change that makes
    the loop settle can be told apart from one that merely hides it.

    When the data says almost nothing about a team, the observed spread and the
    posterior variance both shrink as the penalty rises, so nothing pushes back
    and the iteration climbs to the bound. Reproduced here with a league of two
    matches per team.
    """
    training, _, _ = synthetic_league(seasons=1, n_teams=30, seed=6, half_life=20.0)
    model = goals.fit_goals_model(training, ridge="auto", target="goals")
    estimate = model.ridge_estimate
    # Still climbing when the round budget ran out, and every round higher than
    # the last. That monotone history is the signature; the value it happens to
    # have reached is just where the budget stopped it.
    assert not estimate.converged
    levels = [np.sqrt(a * b) for a, b in estimate.history]
    assert all(later > earlier for earlier, later in zip(levels, levels[1:]))
    assert levels[-1] > 5.0 * levels[0]


# ----------------------------------------------------------------------
# Wiring into the fitter
# ----------------------------------------------------------------------


def test_a_scalar_ridge_still_behaves_exactly_as_before():
    training, _, _ = synthetic_league(seasons=3, n_teams=12, seed=8)
    model = goals.fit_goals_model(training, ridge=2.0, target="goals")
    assert model.ridge == pytest.approx(2.0)
    assert model.ridge_pair == (2.0, 2.0)
    assert model.ridge_estimate is None


def test_a_pair_penalises_the_two_blocks_differently():
    """Attack shrunk hard and defence left free must produce a narrower spread
    of attack strengths than the reverse, or the pair is not reaching the
    penalty at all."""
    training, _, _ = synthetic_league(seasons=4, n_teams=14, seed=2)
    tight_attack = goals.fit_goals_model(training, ridge=(30.0, 0.5), target="goals")
    tight_defence = goals.fit_goals_model(training, ridge=(0.5, 30.0), target="goals")
    assert tight_attack.attack.std() < tight_attack.defence.std()
    assert tight_defence.defence.std() < tight_defence.attack.std()


def test_scalar_ridge_summary_is_the_geometric_mean():
    training, _, _ = synthetic_league(seasons=2, n_teams=10, seed=12)
    model = goals.fit_goals_model(training, ridge=(1.0, 4.0), target="goals")
    assert model.ridge == pytest.approx(2.0)


def test_auto_split_keeps_the_requested_level():
    """The point of the anchored mode: it moves the balance and leaves the
    level alone, so it cannot inherit the runaway."""
    training, _, _ = synthetic_league(seasons=4, n_teams=14, seed=13)
    model = goals.fit_goals_model(training, ridge="auto-split", target="goals")
    assert model.ridge == pytest.approx(base.default_ridge("goals"))
    assert model.ridge_estimate is not None
    assert model.ridge_estimate.converged


def test_an_unknown_ridge_name_is_rejected():
    training, _, _ = synthetic_league(seasons=2, n_teams=8, seed=14)
    with pytest.raises(ValueError, match="Unknown ridge"):
        goals.fit_goals_model(training, ridge="automatic", target="goals")


def test_ridge_penalty_accepts_a_pair_and_agrees_with_the_scalar():
    attack = np.array([0.2, -0.1])
    defence = np.array([0.3, 0.0])
    zeros = np.zeros(2)
    scalar = base.ridge_penalty(attack, defence, zeros, zeros, 2.0)
    paired = base.ridge_penalty(attack, defence, zeros, zeros, (2.0, 2.0))
    assert scalar[0] == pytest.approx(paired[0])
    assert np.allclose(scalar[1], paired[1])
    assert np.allclose(scalar[2], paired[2])


def test_ridge_penalty_pair_splits_the_two_blocks():
    attack = np.array([1.0])
    defence = np.array([1.0])
    zeros = np.zeros(1)
    value, d_attack, d_defence = base.ridge_penalty(
        attack, defence, zeros, zeros, (1.0, 3.0)
    )
    assert value == pytest.approx(4.0)
    assert d_attack[0] == pytest.approx(2.0)
    assert d_defence[0] == pytest.approx(6.0)
