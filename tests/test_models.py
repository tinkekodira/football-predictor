"""Tests for the Phase 2 modelling layer.

The three that matter most:

* `test_*_gradient_is_correct` checks the analytic gradients against numerical
  differentiation. A wrong gradient does not crash; it quietly converges to the
  wrong parameters and every price downstream is subtly off.
* `test_model_recovers_known_strengths` fits the model to synthetic data whose
  true team strengths are known, and checks it finds them back.
* `test_handicap_at_zero_matches_match_odds` is the coherence check: the same
  probabilities have to fall out of two different summations of the same
  matrix, or the market layer is wrong somewhere.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, markets, normalize, predict, pricing  # noqa: E402
from fbedge.models import base, counts, goals  # noqa: E402
from scripts.make_sample_data import TEAMS, build_season  # noqa: E402

AS_OF = dt.date(config.CURRENT_SEASON_START_YEAR, 8, 31)


@pytest.fixture(scope="module")
def model_db(tmp_path_factory):
    """Six synthetic seasons of one league, enough to fit against."""
    path = tmp_path_factory.mktemp("models") / "models.duckdb"
    con = database.connect(path)
    # 2019 and 2020 are included because the generator omits corner and
    # referee columns from F1 before 2021, which is what lets the tests
    # exercise the "this league-season has no such data" path.
    years = [2019, 2020] + list(
        range(config.CURRENT_SEASON_START_YEAR - 5, config.CURRENT_SEASON_START_YEAR + 1)
    )
    for year in years:
        for league in ("E0", "F1"):
            raw = build_season(league, year, seed=year * 17)
            matches, odds = normalize.normalize_league_season(raw, league, year)
            database.load_matches(con, matches)
            database.load_odds(con, odds)
    yield con
    con.close()


@pytest.fixture(scope="module")
def training(model_db) -> base.TrainingSet:
    return base.load_training_set(model_db, "E0", AS_OF, half_life_days=400)


@pytest.fixture(scope="module")
def goals_model(training) -> goals.GoalsModel:
    return goals.fit_goals_model(training)


# --------------------------------------------------------------------------
# Time decay and training data
# --------------------------------------------------------------------------

def test_time_weights_halve_over_a_half_life():
    as_of = dt.date(2026, 8, 31)
    dates = pd.Series(pd.to_datetime(["2026-08-31", "2026-03-04", "2025-09-04"]))
    weights = base.time_weights(dates, as_of, half_life_days=180)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, abs=0.01)
    assert weights[2] == pytest.approx(0.25, abs=0.01)


def test_zero_half_life_disables_decay():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2026-08-31"]))
    weights = base.time_weights(dates, dt.date(2026, 8, 31), half_life_days=0)
    assert np.allclose(weights, 1.0)


def test_training_set_is_point_in_time(model_db):
    """Nothing on or after the cut-off may reach the fit."""
    as_of = dt.date(2026, 8, 20)
    training = base.load_training_set(model_db, "E0", as_of)
    assert (pd.to_datetime(training.frame["date"]).dt.date < as_of).all()


def test_effective_sample_shrinks_with_shorter_half_life(model_db):
    long_memory = base.load_training_set(model_db, "E0", AS_OF, half_life_days=720)
    short_memory = base.load_training_set(model_db, "E0", AS_OF, half_life_days=60)
    assert long_memory.n_matches == short_memory.n_matches
    assert short_memory.effective_n < long_memory.effective_n


def test_too_little_history_is_refused(model_db):
    with pytest.raises(base.InsufficientData):
        base.load_training_set(model_db, "E0", dt.date(2000, 1, 1))


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------

def _relative_gradient_error(objective, theta) -> float:
    error = check_grad(lambda t: objective(t)[0], lambda t: objective(t)[1], theta,
                       epsilon=1e-6)
    return error / np.linalg.norm(objective(theta)[1])


def test_goals_gradient_is_correct(training):
    objective, _, _ = goals.build_goals_objective(training)
    n = len(training.index)
    rng = np.random.default_rng(0)
    for _ in range(3):
        theta = np.concatenate(
            [[0.3, 0.25], rng.normal(0, 0.2, n), rng.normal(0, 0.2, n), [-0.06]]
        )
        assert _relative_gradient_error(objective, theta) < 1e-4


def test_goals_gradient_is_correct_without_correction(training):
    objective, _, _ = goals.build_goals_objective(
        training, use_dixon_coles_correction=False
    )
    n = len(training.index)
    rng = np.random.default_rng(1)
    theta = np.concatenate([[0.3, 0.2], rng.normal(0, 0.2, n), rng.normal(0, 0.2, n)])
    assert _relative_gradient_error(objective, theta) < 1e-4


@pytest.mark.parametrize("kind", ["corners", "cards"])
def test_count_gradient_is_correct(training, kind):
    objective, _, _, referee_names, _ = counts.build_count_objective(training, kind)
    n = len(training.index)
    rng = np.random.default_rng(2)
    theta = np.concatenate(
        [
            [1.6, 0.1],
            rng.normal(0, 0.15, n),
            rng.normal(0, 0.15, n),
            [np.log(9.0)],
            rng.normal(0, 0.1, len(referee_names)),
        ]
    )
    assert _relative_gradient_error(objective, theta) < 1e-4


# --------------------------------------------------------------------------
# Does the model learn anything?
# --------------------------------------------------------------------------

def test_model_recovers_known_strengths(goals_model):
    """The generator ranks teams strongest-first; the fit should agree."""
    truth = TEAMS["E0"]
    fitted = goals_model.ratings()["team"].tolist()
    correlation, p_value = sps.spearmanr(
        [truth.index(team) for team in fitted], range(len(fitted))
    )
    assert correlation > 0.7, f"rank correlation only {correlation:.2f}"
    assert p_value < 0.01


def test_home_advantage_is_positive(goals_model):
    assert 0.0 < goals_model.home_advantage < 0.8


def test_stronger_team_is_favoured_at_home_and_away(goals_model):
    strong, weak = "Arsenal", "Everton"
    home_lam, home_mu = goals_model.rates(strong, weak)
    away_lam, away_mu = goals_model.rates(weak, strong)
    assert home_lam > home_mu
    assert away_mu > away_lam


def test_home_advantage_shifts_the_rates(goals_model):
    """The same pairing must be better for whichever side is at home."""
    a_home, b_away = goals_model.rates("Chelsea", "Newcastle")
    b_home, a_away = goals_model.rates("Newcastle", "Chelsea")
    assert a_home > a_away
    assert b_home > b_away


# --------------------------------------------------------------------------
# Shrinkage and priors: the answer to thin samples
# --------------------------------------------------------------------------

def test_stronger_shrinkage_compresses_ratings(training):
    loose = goals.fit_goals_model(training, ridge=0.5)
    tight = goals.fit_goals_model(training, ridge=50.0)
    assert np.std(tight.attack) < np.std(loose.attack)
    assert np.std(tight.defence) < np.std(loose.defence)


def test_unknown_team_gets_the_promoted_prior(goals_model):
    """A club with no history must be priced, not rejected."""
    assert not goals_model.is_known("Newly Promoted FC")
    lam, mu = goals_model.rates("Newly Promoted FC", "Arsenal")
    assert np.isfinite(lam) and np.isfinite(mu)
    assert mu > lam  # the established side should be favoured


def test_promoted_prior_is_pessimistic(training):
    """A newcomer must be rated below the league average, not level with it."""
    attack_prior, defence_prior = base.build_priors(training)
    counts_per_team = training.match_counts()
    established = counts_per_team >= base.PROMOTED_MATCH_THRESHOLD
    assert established.all(), "sample fixture should have no newcomers"
    assert np.allclose(attack_prior, 0.0)

    # Force a newcomer by raising the threshold, and check the prior turns
    # pessimistic rather than neutral.
    attack_prior, defence_prior = base.build_priors(training, threshold=10_000)
    assert (attack_prior < 0).all()
    assert (defence_prior < 0).all()


# --------------------------------------------------------------------------
# Score matrix and goal markets
# --------------------------------------------------------------------------

def test_score_matrix_is_a_distribution(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    assert matrix.sum() == pytest.approx(1.0)
    assert (matrix >= 0).all()


def test_score_matrix_mean_tracks_the_rates(goals_model):
    """Truncation and the low-score correction must not move the mean much."""
    lam, mu = goals_model.rates("Arsenal", "Everton")
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    goals_axis = np.arange(matrix.shape[0])
    assert (matrix.sum(axis=1) * goals_axis).sum() == pytest.approx(lam, rel=0.05)
    assert (matrix.sum(axis=0) * goals_axis).sum() == pytest.approx(mu, rel=0.05)


def test_match_odds_sum_to_one(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    assert sum(s.probability for s in markets.match_odds(matrix)) == pytest.approx(1.0)


def test_double_chance_agrees_with_match_odds(goals_model):
    matrix = goals_model.score_matrix("Chelsea", "Tottenham")
    home, draw, away = (s.probability for s in markets.match_odds(matrix))
    combos = {s.selection: s.probability for s in markets.double_chance(matrix)}
    assert combos["home_or_draw"] == pytest.approx(home + draw)
    assert combos["home_or_away"] == pytest.approx(home + away)
    assert combos["draw_or_away"] == pytest.approx(draw + away)


@pytest.mark.parametrize("line", [0.5, 1.5, 2.5, 3.5, 4.5])
def test_over_under_pairs_are_complementary(goals_model, line):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    pair = markets.total_goals(matrix, (line,))
    assert sum(s.probability for s in pair) == pytest.approx(1.0)


def test_over_lines_are_monotonic(goals_model):
    """Over 3.5 cannot be more likely than over 2.5."""
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    overs = [
        s.probability
        for s in markets.total_goals(matrix, (0.5, 1.5, 2.5, 3.5, 4.5))
        if s.selection == "over"
    ]
    assert all(earlier >= later for earlier, later in zip(overs, overs[1:]))


def test_btts_is_complementary(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    assert sum(s.probability for s in markets.both_teams_to_score(matrix)) == pytest.approx(1.0)


def test_correct_scores_are_ordered_and_bounded(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    scores = markets.correct_score(matrix, top_n=10)
    probabilities = [s.probability for s in scores]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) < 1.0


# --------------------------------------------------------------------------
# Asian handicap
# --------------------------------------------------------------------------

def test_handicap_at_zero_matches_match_odds(goals_model):
    """The coherence check: AH 0.0 is draw-no-bet, so it must reproduce 1X2."""
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    home, draw, away = (s.probability for s in markets.match_odds(matrix))
    handicap = markets.asian_handicap(matrix, 0.0)
    assert handicap[0].probability == pytest.approx(home)
    assert handicap[1].probability == pytest.approx(away)
    assert handicap[0].push_probability == pytest.approx(draw)


@pytest.mark.parametrize("line", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
def test_handicap_outcomes_are_exhaustive(goals_model, line):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    home, away = markets.asian_handicap(matrix, line)
    total = home.probability + away.probability + home.push_probability
    assert total == pytest.approx(1.0)


def test_quarter_line_is_the_average_of_its_halves(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    quarter = markets.asian_handicap(matrix, -0.25)[0]
    at_zero = markets.asian_handicap(matrix, 0.0)[0]
    at_half = markets.asian_handicap(matrix, -0.5)[0]
    expected = 0.5 * (at_zero.probability + at_half.probability)
    assert quarter.probability == pytest.approx(expected)


def test_handicap_sides_report_opposite_lines(goals_model):
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    for line in (-1.5, -0.25, 0.75):
        home, away = markets.asian_handicap(matrix, line)
        assert home.line == pytest.approx(line)
        assert away.line == pytest.approx(-line)


def test_giving_a_bigger_start_is_less_likely_to_win(goals_model):
    """Home -2.5 must be harder to land than home -1.5."""
    matrix = goals_model.score_matrix("Arsenal", "Everton")
    easier = markets.asian_handicap(matrix, -1.5)[0].probability
    harder = markets.asian_handicap(matrix, -2.5)[0].probability
    assert harder < easier


# --------------------------------------------------------------------------
# Count models
# --------------------------------------------------------------------------

def test_negative_binomial_pmf_is_a_distribution():
    pmf = counts._nb_pmf(mean=10.0, dispersion=8.0, max_count=60)
    assert pmf.sum() == pytest.approx(1.0)
    assert markets.count_mean(pmf) == pytest.approx(10.0, rel=0.01)


def test_large_dispersion_approaches_poisson():
    """Dispersion is what separates the negative binomial from Poisson."""
    nb = counts._nb_pmf(mean=4.0, dispersion=1e5, max_count=40)
    poisson = sps.poisson.pmf(np.arange(41), 4.0)
    assert np.abs(nb - poisson).max() < 1e-4


def test_small_dispersion_widens_the_distribution():
    tight = counts._nb_pmf(mean=10.0, dispersion=1000.0, max_count=80)
    wide = counts._nb_pmf(mean=10.0, dispersion=2.0, max_count=80)
    counts_axis = np.arange(81)
    tight_var = (tight * (counts_axis - 10.0) ** 2).sum()
    wide_var = (wide * (counts_axis - 10.0) ** 2).sum()
    assert wide_var > tight_var


def test_count_model_fits_and_predicts(training):
    model = counts.fit_count_model(training, "corners")
    home, away = model.rates("Arsenal", "Everton")
    assert 1.0 < home < 15.0 and 1.0 < away < 15.0
    pmf = model.total_distribution("Arsenal", "Everton")
    assert pmf.sum() == pytest.approx(1.0)
    assert 5.0 < markets.count_mean(pmf) < 20.0


def test_card_model_fits_referee_effects(training):
    model = counts.fit_count_model(training, "cards")
    table = model.referee_table()
    assert not table.empty
    assert "(other)" not in table["referee"].tolist()
    # Referee effects are heavily shrunk, so multipliers stay near neutral.
    assert table["multiplier"].between(0.5, 2.0).all()


def test_missing_statistic_raises_insufficient_data(model_db):
    """F1 has no corner data before 2021 in the sample generator."""
    training = base.load_training_set(model_db, "F1", dt.date(2021, 1, 1))
    with pytest.raises(base.InsufficientData):
        counts.fit_count_model(training, "corners")


def test_count_totals_are_complementary(training):
    model = counts.fit_count_model(training, "corners")
    pmf = model.total_distribution("Arsenal", "Everton")
    for over, under in zip(*[iter(markets.count_totals(pmf, "total_corners", (9.5, 10.5)))] * 2):
        assert over.probability + under.probability == pytest.approx(1.0)


def test_unknown_kind_is_rejected(training):
    with pytest.raises(ValueError):
        counts.fit_count_model(training, "throw_ins")


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

def test_overround_exceeds_one_for_real_prices():
    assert pricing.overround([2.10, 3.40, 3.80]) > 1.0


@pytest.mark.parametrize("method", ["multiplicative", "additive", "shin"])
def test_margin_removal_yields_a_distribution(method):
    probabilities = pricing.remove_margin([2.10, 3.40, 3.80], method=method)
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities > 0).all()


def test_additive_removal_penalises_longshots_more():
    """The favourite-longshot bias: margin is loaded onto the outsider."""
    prices = [1.30, 5.50, 12.0]
    multiplicative = pricing.remove_margin(prices, "multiplicative")
    additive = pricing.remove_margin(prices, "additive")
    assert additive[-1] < multiplicative[-1]
    assert additive[0] > multiplicative[0]


def test_the_default_method_is_shin():
    """Pinned deliberately: this default changed, and the change matters.

    Multiplicative was the default until it was measured against Betfair
    Exchange closing prices and found to overstate longshots by more than half
    a point - which, because this model bets longshots, inflated closing line
    value by 1.75 points and manufactured an apparent era of edge. Anything
    that flips this back should have to argue with a test.
    """
    prices = [1.30, 5.50, 12.0]
    assert pricing.remove_margin(prices) == pytest.approx(
        pricing.remove_margin(prices, "shin")
    )


def test_shin_lands_between_the_other_two_on_longshots():
    """The bracketing that made Shin worth adding.

    Measured against the exchange, multiplicative leaves longshots too high and
    additive pushes them too low. Shin has a free parameter and lands between
    them, which is why it comes out flat across probability bands where the
    other two show a gradient.
    """
    prices = [1.30, 5.50, 12.0]
    multiplicative = pricing.remove_margin(prices, "multiplicative")
    additive = pricing.remove_margin(prices, "additive")
    shin = pricing.remove_margin(prices, "shin")

    assert additive[-1] < shin[-1] < multiplicative[-1]   # the longshot
    assert multiplicative[0] < shin[0] < additive[0]      # the favourite


def test_shin_is_a_no_op_on_a_market_with_no_margin():
    # Prices that already imply exactly 1.0 leave nothing to remove.
    prices = [2.0, 4.0, 4.0]
    assert pricing.overround(prices) == pytest.approx(1.0)
    assert pricing.remove_margin(prices, "shin") == pytest.approx([0.5, 0.25, 0.25])


def test_shin_handles_a_two_outcome_market():
    probabilities = pricing.remove_margin([1.90, 1.95], "shin")
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities > 0).all()


def test_shin_falls_back_rather_than_failing_on_degenerate_prices():
    # An overround below 1.0 is bad data, not an arbitrage to exploit. The
    # caller gets normalised probabilities instead of an exception.
    probabilities = pricing.remove_margin([3.0, 4.0, 5.0], "shin")
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities > 0).all()


def test_shin_preserves_the_ordering_of_the_prices():
    prices = [1.30, 5.50, 12.0]
    shin = pricing.remove_margin(prices, "shin")
    assert shin[0] > shin[1] > shin[2]


def test_unknown_margin_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        pricing.remove_margin([2.0, 4.0, 4.0], "wishful")


def test_expected_value_is_zero_at_the_fair_price():
    probability = 0.4
    assert pricing.expected_value(probability, pricing.fair_price(probability)) == pytest.approx(0.0)


def test_expected_value_is_positive_above_the_fair_price():
    assert pricing.expected_value(0.5, 2.20) > 0
    assert pricing.expected_value(0.5, 1.80) < 0


def test_fair_price_accounts_for_pushes():
    """A quarter of the stake coming back lowers the price needed to break even."""
    without_push = markets.Selection("asian_handicap", "home", 0.5)
    with_push = markets.Selection("asian_handicap", "home", 0.5, push_probability=0.2)
    assert with_push.fair_price < without_push.fair_price
    assert with_push.fair_price == pytest.approx(0.8 / 0.5)


def test_impossible_selection_has_infinite_fair_price():
    assert markets.Selection("1x2", "home", 0.0).fair_price == float("inf")


# --------------------------------------------------------------------------
# End-to-end prediction
# --------------------------------------------------------------------------

def test_predict_fixture_produces_every_market(model_db):
    forecast = predict.predict_fixture(
        model_db, "Arsenal", "Everton", as_of=AS_OF, league="E0"
    )
    found = {s.market for s in forecast.selections}
    assert {"1x2", "total_goals", "btts", "asian_handicap", "correct_score"} <= found
    assert forecast.expected_goals[0] > 0
    assert isinstance(forecast.render(), str)
    assert not forecast.to_frame().empty


def test_prediction_is_point_in_time(model_db):
    """Predicting an old fixture must not use anything played since."""
    early = predict.predict_fixture(
        model_db, "Arsenal", "Everton", as_of=dt.date(2023, 9, 1), league="E0"
    )
    late = predict.predict_fixture(
        model_db, "Arsenal", "Everton", as_of=AS_OF, league="E0"
    )
    assert early.expected_goals != late.expected_goals


def test_unknown_team_is_flagged_not_hidden(model_db):
    forecast = predict.predict_fixture(
        model_db, "Mystery FC", "Arsenal", as_of=AS_OF, league="E0"
    )
    assert any("no match history" in note for note in forecast.notes)


def test_model_cache_invalidates_when_data_changes(model_db, tmp_path):
    """A rebuilt database must not serve stale parameters."""
    predict.clear_model_cache()
    first = predict.build_models(model_db, "E0", AS_OF)
    again = predict.build_models(model_db, "E0", AS_OF)
    assert first is again  # same data, same object

    raw = build_season("E0", config.CURRENT_SEASON_START_YEAR - 6, seed=999)
    matches, _ = normalize.normalize_league_season(
        raw, "E0", config.CURRENT_SEASON_START_YEAR - 6
    )
    database.load_matches(model_db, matches)
    after = predict.build_models(model_db, "E0", AS_OF)
    assert after is not first


def test_prediction_probabilities_are_all_valid(model_db):
    forecast = predict.predict_fixture(
        model_db, "Chelsea", "Newcastle", as_of=AS_OF, league="E0"
    )
    for selection in forecast.selections:
        assert 0.0 <= selection.probability <= 1.0
        assert 0.0 <= selection.push_probability <= 1.0
        assert selection.fair_price >= 1.0
