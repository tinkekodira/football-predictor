"""Every goal market must come from the one score matrix, and agree with it.

The README's central claim about the model layer is that "because every price
comes from one matrix, they cannot contradict each other. A system that fits
each market separately will happily quote an over 2.5 that is inconsistent with
its own correct-score prices."

That claim is currently true by construction, and it would stay true right up
until somebody adds a market with its own fit, or a caller starts assembling
its own selection list. These tests make the claim checkable instead of merely
intended: each one states an identity that must hold between two markets, so a
refactor that breaks the shared derivation fails here rather than in front of a
reader comparing two prices on one screen.

Half-time markets are the deliberate exception and have their own test saying
so: they are *not* derivable from the full-time matrix, and the thing to guard
there is that nobody quietly starts deriving them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import markets  # noqa: E402
from fbedge.models import goals  # noqa: E402

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def matrix() -> np.ndarray:
    """An ordinary fixture: a home side expected to score 1.7 against 1.1."""
    return goals.score_matrix_from_rates(1.7, 1.1, rho=-0.06, max_goals=12)


@pytest.fixture(scope="module")
def selections(matrix) -> dict:
    """Every market, keyed by (market, selection, line), from one matrix."""
    everything = markets.all_goal_selections(
        matrix, handicap_lines=(-0.5, -0.75, -1.0)
    )
    return {(s.market, s.selection, s.line): s for s in everything}


def probability(selections, market, selection, line=None) -> float:
    return selections[(market, selection, line)].probability


# --------------------------------------------------------------------------
# The matrix itself
# --------------------------------------------------------------------------

def test_the_matrix_is_a_probability_distribution(matrix):
    assert matrix.sum() == pytest.approx(1.0, abs=1e-6)
    assert (matrix >= 0).all()


# --------------------------------------------------------------------------
# Markets that must sum to one
# --------------------------------------------------------------------------

@pytest.mark.parametrize("market", ["1x2", "btts", "winning_margin"])
def test_complete_markets_sum_to_one(selections, market):
    total = sum(
        s.probability for key, s in selections.items() if key[0] == market
    )
    assert total == pytest.approx(1.0, abs=1e-6)


def test_each_over_under_line_sums_to_one_including_its_push(selections):
    for line in markets.DEFAULT_GOAL_LINES:
        over = selections[("total_goals", "over", line)]
        under = selections[("total_goals", "under", line)]
        assert over.probability + under.probability + over.push_probability == (
            pytest.approx(1.0, abs=1e-6)
        )
        assert over.push_probability == pytest.approx(under.push_probability)


# --------------------------------------------------------------------------
# Cross-market identities: the actual consistency claim
# --------------------------------------------------------------------------

def test_double_chance_is_the_sum_of_its_two_1x2_legs(selections):
    home = probability(selections, "1x2", "home")
    draw = probability(selections, "1x2", "draw")
    away = probability(selections, "1x2", "away")
    assert probability(selections, "double_chance", "home_or_draw") == pytest.approx(
        home + draw, abs=TOLERANCE
    )
    assert probability(selections, "double_chance", "home_or_away") == pytest.approx(
        home + away, abs=TOLERANCE
    )
    assert probability(selections, "double_chance", "draw_or_away") == pytest.approx(
        draw + away, abs=TOLERANCE
    )


def test_draw_no_bet_is_the_1x2_with_the_draw_pushed(selections):
    home = probability(selections, "1x2", "home")
    draw = probability(selections, "1x2", "draw")
    dnb = selections[("draw_no_bet", "home", None)]
    assert dnb.probability == pytest.approx(home, abs=TOLERANCE)
    assert dnb.push_probability == pytest.approx(draw, abs=TOLERANCE)
    # And therefore the fair price is the conditional one, not the raw one.
    assert dnb.fair_price == pytest.approx((1.0 - draw) / home, abs=1e-9)


def test_winning_margin_collapses_back_onto_1x2(selections):
    home = sum(
        s.probability for key, s in selections.items()
        if key[0] == "winning_margin" and key[1].startswith("home")
    )
    away = sum(
        s.probability for key, s in selections.items()
        if key[0] == "winning_margin" and key[1].startswith("away")
    )
    assert home == pytest.approx(probability(selections, "1x2", "home"), abs=1e-9)
    assert away == pytest.approx(probability(selections, "1x2", "away"), abs=1e-9)
    assert probability(selections, "winning_margin", "draw") == pytest.approx(
        probability(selections, "1x2", "draw"), abs=1e-9
    )


def test_btts_agrees_with_the_two_team_totals(selections):
    """BTTS yes is exactly "both sides over 0.5", and must equal it.

    This is the identity most likely to break under a refactor, because BTTS
    is computed by slicing the matrix and team totals by summing its rows and
    columns. Two different derivations of the same quantity.
    """
    home_scores = probability(selections, "home_goals", "over", 0.5)
    away_scores = probability(selections, "away_goals", "over", 0.5)
    btts_yes = probability(selections, "btts", "yes")
    assert btts_yes <= min(home_scores, away_scores) + TOLERANCE
    # Under independence they would multiply; Dixon-Coles adds correlation in
    # the low cells, so the exact identity is the marginal one:
    # P(both) = P(home>0) + P(away>0) - P(at least one > 0).
    at_least_one = 1.0 - probability(selections, "total_goals", "under", 0.5)
    assert btts_yes == pytest.approx(
        home_scores + away_scores - at_least_one, abs=1e-9
    )


def test_total_goals_agrees_with_the_sum_of_the_team_totals(selections):
    """Over 0.5 total is "at least one side scored", by inclusion-exclusion."""
    home = probability(selections, "home_goals", "over", 0.5)
    away = probability(selections, "away_goals", "over", 0.5)
    both = probability(selections, "btts", "yes")
    assert probability(selections, "total_goals", "over", 0.5) == pytest.approx(
        home + away - both, abs=1e-9
    )


def test_the_half_goal_handicap_is_the_1x2_home_win(selections):
    """A -0.5 handicap wins on exactly the results a home win does."""
    assert probability(selections, "asian_handicap", "home", -0.5) == pytest.approx(
        probability(selections, "1x2", "home"), abs=1e-9
    )


def test_the_zero_handicap_is_draw_no_bet(selections):
    """Two names for the same bet, and they must price the same."""
    pair = markets.asian_handicap(
        goals.score_matrix_from_rates(1.7, 1.1, rho=-0.06, max_goals=12), 0.0
    )
    handicap_home = next(s for s in pair if s.selection == "home")
    dnb_home = selections[("draw_no_bet", "home", None)]
    assert handicap_home.probability == pytest.approx(dnb_home.probability, abs=1e-9)
    assert handicap_home.push_probability == pytest.approx(
        dnb_home.push_probability, abs=1e-9
    )
    assert handicap_home.fair_price == pytest.approx(dnb_home.fair_price, abs=1e-9)


def test_correct_score_never_exceeds_the_market_it_belongs_to(selections):
    """The most likely single scoreline cannot beat the result it implies."""
    for key, s in selections.items():
        if key[0] != "correct_score":
            continue
        home_goals, away_goals = (int(v) for v in s.selection.split("-"))
        implied = (
            "home" if home_goals > away_goals
            else ("away" if away_goals > home_goals else "draw")
        )
        assert s.probability <= probability(selections, "1x2", implied) + TOLERANCE


def test_price_selection_returns_what_the_bulk_builder_returns(matrix, selections):
    """The backtest prices one selection at a time; the app prices them all.

    Two code paths into the same arithmetic, which is precisely the shape of
    drift this file exists to catch.
    """
    for (market, selection, line), expected in selections.items():
        if market in ("correct_score", "asian_handicap"):
            continue  # priced from the backed side's own line, tested below
        one = markets.price_selection(matrix, market, selection, line)
        assert one is not None, f"{market}/{selection} lost its single-selection path"
        assert one.probability == pytest.approx(expected.probability, abs=1e-12)
        assert one.push_probability == pytest.approx(
            expected.push_probability, abs=1e-12
        )


def test_price_selection_reads_the_handicap_line_from_the_backed_side(matrix):
    away = markets.price_selection(matrix, "asian_handicap", "away", 0.5)
    pair = markets.asian_handicap(matrix, -0.5)
    expected = next(s for s in pair if s.selection == "away")
    assert away.probability == pytest.approx(expected.probability, abs=1e-12)


# --------------------------------------------------------------------------
# The exception, stated as a test
# --------------------------------------------------------------------------

def test_a_withdrawn_market_is_not_priced_and_cannot_be_settled(matrix):
    """odd/even goals was measured, found unrankable, and removed.

    Its calibration slope ran -2.41 to +1.54 across five leagues, so in at
    least one the model's confidence pointed the wrong way. Pinned because the
    market is trivially cheap to derive from the matrix and would otherwise be
    re-added by somebody enumerating what a score matrix can produce.
    """
    from fbedge import settlement

    assert not hasattr(markets, "odd_even_goals")
    assert markets.price_selection(matrix, "odd_even_goals", "odd", None) is None
    assert not any(
        s.market == "odd_even_goals" for s in markets.all_goal_selections(matrix)
    )
    # A stored row from before the removal must fail loudly, not settle.
    with pytest.raises(ValueError, match="was removed on 2026-09-04"):
        settlement.settle("odd_even_goals", "odd", None, 2, 1)


def test_half_time_markets_are_not_derivable_from_the_full_time_matrix(matrix):
    """The guard against the worst of the three options.

    Silently deriving half-time prices by halving full-time rates would look
    plausible and be wrong: about 45% of goals arrive before the interval, not
    50%, and on E0 the fitted half-time home advantage is *larger* than the
    full-time one (0.33 against 0.20). So `price_selection` must refuse rather
    than approximate.
    """
    assert markets.price_selection(matrix, "1x2_ht", "home", None) is None
    assert markets.price_selection(matrix, "total_goals_ht", "over", 0.5) is None
    assert not any(
        s.market.endswith("_ht") for s in markets.all_goal_selections(matrix)
    )


# --------------------------------------------------------------------------
# Count markets
# --------------------------------------------------------------------------

def test_a_count_handicap_sums_to_one_with_its_push():
    home = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    away = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    pair = markets.count_handicap(home, away, "corner_handicap", -1.0)
    assert sum(s.probability for s in pair) + pair[0].push_probability == (
        pytest.approx(1.0, abs=1e-9)
    )


def test_a_half_line_count_handicap_cannot_push():
    home = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    away = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    pair = markets.count_handicap(home, away, "corner_handicap", -1.5)
    assert all(s.push_probability == 0 for s in pair)
    assert sum(s.probability for s in pair) == pytest.approx(1.0, abs=1e-9)


def test_a_quarter_line_count_handicap_is_the_average_of_its_two_halves():
    home = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    away = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    quarter = markets.count_handicap(home, away, "corner_handicap", -0.75)
    lower = markets.count_handicap(home, away, "corner_handicap", -0.5)
    upper = markets.count_handicap(home, away, "corner_handicap", -1.0)
    assert quarter[0].probability == pytest.approx(
        0.5 * (lower[0].probability + upper[0].probability), abs=1e-12
    )


def test_the_count_difference_recovers_the_two_means():
    """A sanity check on the convolution, not on the independence assumption."""
    home = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    away = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    offsets, weights = markets.count_difference(home, away)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    mean = float((offsets * weights).sum())
    expected = markets.count_mean(home) - markets.count_mean(away)
    assert mean == pytest.approx(expected, abs=1e-9)


def test_team_count_totals_read_the_team_distributions_not_the_match_total():
    """The match total is fitted separately and must not be the sum of these.

    Summing two independent team distributions understates how much a total
    scatters, because both teams win more corners in an open game. The models
    fit the total's dispersion separately for exactly that reason, so a test
    that team totals *sum* to the match total would be enforcing a bug.
    """
    home = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    away = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    team = markets.team_count_totals(home, away, "total_corners", (1.5,))
    assert {s.market for s in team} == {"home_total_corners", "away_total_corners"}
    over_home = next(
        s for s in team if s.market == "home_total_corners" and s.selection == "over"
    )
    assert over_home.probability == pytest.approx(home[2:].sum(), abs=1e-12)


# --------------------------------------------------------------------------
# Every market has to be nameable
# --------------------------------------------------------------------------

def test_every_priced_market_has_a_title(matrix):
    """A table of probabilities is unreadable without one.

    Pinned because the failure is silent: a market added without a title still
    renders, as a nameless block of numbers, and the person who added it knows
    what it is.
    """
    priced = {s.market for s in markets.all_goal_selections(matrix)}
    priced |= {
        "total_corners", "home_total_corners", "away_total_corners",
        "corner_handicap", "total_cards", "home_total_cards",
        "away_total_cards", "card_handicap", "1x2_ht", "total_goals_ht",
    }
    missing = sorted(priced - set(markets.MARKET_TITLES))
    assert not missing, f"markets with no title: {missing}"


def test_team_market_titles_name_the_clubs():
    """"Home team goals" still asks the reader to remember which side is which.

    This is the bug that prompted the titles: two team-total tables, identical
    in shape, carrying identical evidence text, with nothing on the page saying
    whose goals they were.
    """
    assert markets.market_title("home_goals", "Arsenal", "Liverpool") == "Arsenal goals"
    assert markets.market_title("away_goals", "Arsenal", "Liverpool") == "Liverpool goals"
    assert markets.market_title(
        "home_total_corners", "Arsenal", "Liverpool"
    ) == "Arsenal corners"


def test_the_two_sides_of_a_team_market_never_share_a_title():
    for home_market, away_market in (
        ("home_goals", "away_goals"),
        ("home_total_corners", "away_total_corners"),
        ("home_total_cards", "away_total_cards"),
    ):
        pair = {
            markets.market_title(home_market, "Arsenal", "Liverpool"),
            markets.market_title(away_market, "Arsenal", "Liverpool"),
        }
        assert len(pair) == 2, f"{home_market} and {away_market} render alike"


def test_titles_fall_back_to_something_visible_rather_than_nothing():
    """A market added without a title is labelled badly, not invisibly."""
    assert markets.market_title("some_new_market") == "Some new market"


def test_a_title_works_without_team_names():
    """The CLI and the app both have them; a future caller might not."""
    assert markets.market_title("home_goals") == "Home team goals"
    assert markets.market_title("away_goals") == "Away team goals"
