"""Tests for the distribution-shape diagnostics.

The load-bearing tests here are the two power tests,
`test_dispersion_detects_genuine_overdispersion` and
`test_slope_detects_probabilities_that_are_spread_too_far`.

Both new diagnostics returned "nothing wrong" on the real data: the dispersion
ratio came back at 0.95 against a null of 1.0, and that was used to argue
*against* giving goals the negative binomial treatment that corners and cards
get. A null result is only worth acting on if the test could have found the
effect had it been there, so each one is pointed at data built to contain
exactly the defect it is supposed to catch, at roughly the sample size the real
run had. Without those two, the negative findings are indistinguishable from a
diagnostic that never fires.

`test_pooled_calibration_mirrors_a_two_sided_market` pins the artifact that
started this: it reproduces, on synthetic data, the perfectly mirrored table
that made a real `total_goals` calibration look like it had a shape problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import evaluation, markets  # noqa: E402
from fbedge.models.goals import score_matrix_from_rates  # noqa: E402


# --------------------------------------------------------------------------
# The total distribution
# --------------------------------------------------------------------------

def test_total_distribution_is_a_probability_distribution():
    matrix = score_matrix_from_rates(1.4, 1.1, rho=-0.05)
    pmf = markets.total_distribution(matrix)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf >= 0).all()
    # 13x13 scorelines means totals from 0 to 24.
    assert len(pmf) == 2 * matrix.shape[0] - 1


def test_total_distribution_agrees_with_expected_total():
    matrix = score_matrix_from_rates(1.7, 0.9, rho=-0.08)
    pmf = markets.total_distribution(matrix)
    by_hand = float((pmf * np.arange(len(pmf))).sum())
    assert by_hand == pytest.approx(markets.expected_total(matrix))


def test_total_distribution_sums_the_right_anti_diagonals():
    # A matrix with all its mass on 0-2, 1-1 and 2-0 must put all of it on 2.
    matrix = np.zeros((3, 3))
    matrix[0, 2] = matrix[1, 1] = matrix[2, 0] = 1 / 3
    pmf = markets.total_distribution(matrix)
    assert pmf[2] == pytest.approx(1.0)
    assert pmf[np.arange(len(pmf)) != 2].sum() == pytest.approx(0.0)


def test_total_distribution_matches_the_over_under_split():
    # The same number two ways: summing the pmf above 2.5, and asking
    # markets.total_goals for the over 2.5 price.
    matrix = score_matrix_from_rates(1.5, 1.2, rho=-0.04)
    pmf = markets.total_distribution(matrix)
    from_pmf = float(pmf[3:].sum())
    over = next(
        s for s in markets.total_goals(matrix, (2.5,)) if s.selection == "over"
    )
    assert from_pmf == pytest.approx(over.probability)


# --------------------------------------------------------------------------
# Calibration by line
# --------------------------------------------------------------------------

def _totals_frame(probabilities, outcomes, line=2.5, side="over") -> pd.DataFrame:
    """A predictions frame carrying only what the calibration code reads."""
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(len(probabilities))],
            "market": "total_goals",
            "selection": side,
            "line": line,
            "model_conditional": probabilities,
            "win_fraction": outcomes,
            "push_fraction": 0.0,
        }
    )


def test_calibration_by_line_takes_only_the_requested_side():
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0.35, 0.65, 400)
    outcomes = (rng.random(400) < probabilities).astype(float)
    overs = _totals_frame(probabilities, outcomes)
    unders = _totals_frame(1 - probabilities, 1 - outcomes, side="under")
    both = pd.concat([overs, unders], ignore_index=True)

    table = evaluation.calibration_by_line(both, "total_goals", side="over")
    # Every band must be drawn from the 400 overs, never the 800 rows.
    assert table["n"].sum() <= 400


def test_calibration_by_line_separates_two_lines():
    rng = np.random.default_rng(4)
    low = _totals_frame(rng.uniform(0.4, 0.6, 200), rng.random(200) < 0.5, line=2.5)
    high = _totals_frame(rng.uniform(0.4, 0.6, 200), rng.random(200) < 0.5, line=3.5)
    table = evaluation.calibration_by_line(
        pd.concat([low, high], ignore_index=True), "total_goals", side="over"
    )
    assert set(table["line"]) == {2.5, 3.5}


def test_calibration_by_line_finds_no_gap_when_the_model_is_right():
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0.25, 0.75, 4000)
    outcomes = (rng.random(4000) < probabilities).astype(float)
    table = evaluation.calibration_by_line(
        _totals_frame(probabilities, outcomes), "total_goals", side="over"
    )
    # A correctly calibrated model should not produce a 3-sigma band.
    assert table["z"].abs().max() < 3.0


def test_calibration_by_line_flags_an_overconfident_model():
    rng = np.random.default_rng(6)
    truth = rng.uniform(0.3, 0.7, 3000)
    # Claim probabilities pushed away from the middle: too spread out.
    claimed = np.clip(0.5 + (truth - 0.5) * 1.8, 0.01, 0.99)
    outcomes = (rng.random(3000) < truth).astype(float)
    table = evaluation.calibration_by_line(
        _totals_frame(claimed, outcomes), "total_goals", side="over"
    )
    assert table["z"].abs().max() > 3.0


def test_calibration_by_line_drops_pushes():
    frame = _totals_frame(np.full(100, 0.5), np.ones(100))
    frame.loc[:49, "push_fraction"] = 1.0
    table = evaluation.calibration_by_line(frame, "total_goals", side="over")
    assert table["n"].sum() == 50


def test_calibration_by_line_is_empty_for_an_absent_market():
    frame = _totals_frame(np.full(100, 0.5), np.ones(100))
    assert evaluation.calibration_by_line(frame, "btts", side="yes").empty


def test_pooled_calibration_mirrors_a_two_sided_market():
    # The artifact that made the real total_goals table unreadable: feeding a
    # complementary market to calibration_table produces a table symmetric
    # about 0.5, with every match counted twice and the gaps exactly negated.
    rng = np.random.default_rng(7)
    probabilities = rng.uniform(0.2, 0.8, 2000)
    outcomes = (rng.random(2000) < probabilities).astype(float)
    pooled = evaluation.calibration_table(
        np.r_[probabilities, 1 - probabilities], np.r_[outcomes, 1 - outcomes]
    )
    assert list(pooled["n"]) == list(pooled["n"])[::-1]
    assert pooled["gap"].to_numpy() == pytest.approx(
        -pooled["gap"].to_numpy()[::-1], abs=1e-12
    )


# --------------------------------------------------------------------------
# Calibration slope
# --------------------------------------------------------------------------

def test_slope_is_one_when_the_model_is_calibrated():
    rng = np.random.default_rng(8)
    probabilities = rng.uniform(0.15, 0.85, 6000)
    outcomes = (rng.random(6000) < probabilities).astype(float)
    result = evaluation.calibration_slope(probabilities, outcomes)
    assert abs(result["slope_z"]) < 2.5


def test_slope_detects_probabilities_that_are_spread_too_far():
    # The power test for the real finding: the run reported slope 0.62 at
    # 4.2 SE below one on 3380 matches, so the same defect at the same sample
    # size must be caught here rather than being an artifact of that data.
    rng = np.random.default_rng(9)
    truth = rng.uniform(0.3, 0.7, 3380)
    logit = np.log(truth / (1 - truth))
    claimed = 1.0 / (1.0 + np.exp(-logit / 0.62))  # over-spread by the same factor
    outcomes = (rng.random(3380) < truth).astype(float)

    result = evaluation.calibration_slope(claimed, outcomes)
    assert result["slope"] < 1.0
    assert result["slope_z"] < -2.0


def test_slope_detects_probabilities_that_hedge_toward_the_base_rate():
    rng = np.random.default_rng(10)
    truth = rng.uniform(0.1, 0.9, 4000)
    logit = np.log(truth / (1 - truth))
    claimed = 1.0 / (1.0 + np.exp(-logit * 0.5))  # squashed toward 0.5
    outcomes = (rng.random(4000) < truth).astype(float)

    result = evaluation.calibration_slope(claimed, outcomes)
    assert result["slope"] > 1.0
    assert result["slope_z"] > 2.0


def test_slope_declines_to_answer_on_a_tiny_sample():
    result = evaluation.calibration_slope([0.4, 0.6], [0.0, 1.0])
    assert result == {"n": 2}


def test_slope_declines_when_every_outcome_is_the_same():
    result = evaluation.calibration_slope(np.full(100, 0.5), np.ones(100))
    assert result == {"n": 100}


# --------------------------------------------------------------------------
# Goal total fit
# --------------------------------------------------------------------------

def _match_totals(pmfs, observed) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(len(observed))],
            "observed_total": observed,
            "total_pmf": list(pmfs),
        }
    )


def _poisson_pmfs(means, size=25) -> np.ndarray:
    from scipy import stats

    support = np.arange(size)
    return np.vstack([stats.poisson.pmf(support, m) for m in means])


def test_dispersion_is_one_when_the_model_generated_the_data():
    rng = np.random.default_rng(11)
    means = rng.uniform(2.0, 3.6, 3000)
    pmfs = _poisson_pmfs(means)
    observed = rng.poisson(means)
    fit = evaluation.goal_total_fit(_match_totals(pmfs, observed))
    assert abs(fit["dispersion_z"]) < 2.5


def test_dispersion_detects_genuine_overdispersion():
    # The power test for the headline null. Same means, but the counts are
    # drawn from a negative binomial whose variance is well above its mean -
    # exactly the corners-and-cards failure the goals model was checked for.
    rng = np.random.default_rng(12)
    means = rng.uniform(2.0, 3.6, 3000)
    pmfs = _poisson_pmfs(means)
    dispersion = 4.0
    probability = dispersion / (dispersion + means)
    observed = rng.negative_binomial(dispersion, probability)

    fit = evaluation.goal_total_fit(_match_totals(pmfs, observed))
    assert fit["dispersion"] > 1.0
    assert fit["dispersion_z"] > 3.0


def test_dispersion_detects_a_distribution_that_is_too_wide():
    # The direction the real data actually pointed: realised counts scatter
    # less than the model says they should.
    rng = np.random.default_rng(13)
    means = np.full(3000, 2.8)
    pmfs = _poisson_pmfs(means)
    # Binomial with the same mean has variance below it.
    observed = rng.binomial(10, 0.28, 3000)
    fit = evaluation.goal_total_fit(_match_totals(pmfs, observed))
    assert fit["dispersion"] < 1.0
    assert fit["dispersion_z"] < -3.0


def test_bucket_expectations_sum_to_the_number_of_matches():
    rng = np.random.default_rng(14)
    means = rng.uniform(2.0, 3.6, 500)
    pmfs = _poisson_pmfs(means)
    observed = rng.poisson(means)
    fit = evaluation.goal_total_fit(_match_totals(pmfs, observed))

    buckets = fit["buckets"]
    assert buckets["observed"].sum() == 500
    assert buckets["expected"].sum() == pytest.approx(500.0)


def test_variance_decomposition_obeys_the_law_of_total_variance():
    rng = np.random.default_rng(15)
    means = rng.uniform(2.0, 3.6, 800)
    pmfs = _poisson_pmfs(means)
    fit = evaluation.goal_total_fit(_match_totals(pmfs, rng.poisson(means)))

    assert fit["predicted_variance"] == pytest.approx(
        fit["within_match_variance"] + fit["between_match_variance"]
    )
    # For Poissons the within-match variance is the mean of the rates.
    assert fit["within_match_variance"] == pytest.approx(means.mean(), abs=0.02)
    assert fit["between_match_variance"] == pytest.approx(means.var(ddof=1), abs=0.02)


def test_goal_total_fit_handles_an_empty_frame():
    assert evaluation.goal_total_fit(pd.DataFrame()) == {"n": 0}
