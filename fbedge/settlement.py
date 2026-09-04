"""Settling bets against results.

This is short, unglamorous, and the place where a backtest most easily lies to
you. Getting Asian handicap quarter lines subtly wrong, or settling an away
handicap against the home line, produces a backtest that runs cleanly and
reports a profit that does not exist. Every function here is covered by tests
that check the arithmetic by hand.

A settlement is expressed as three fractions of the stake that sum to one:
how much won, how much was returned, how much was lost. Quarter lines are the
reason it needs three numbers instead of a boolean - a bet at -0.75 that wins
by exactly one goal is half won and half returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TOLERANCE = 1e-9


@dataclass(frozen=True)
class Settlement:
    """How a stake was resolved. The three fractions sum to one."""

    win: float
    push: float = 0.0

    @property
    def loss(self) -> float:
        return max(0.0, 1.0 - self.win - self.push)

    def profit(self, price: float) -> float:
        """Profit per unit staked at a given decimal price.

        A full winner at 2.50 returns +1.50. A push returns 0. A half-loss
        returns -0.50.
        """
        return self.win * (price - 1.0) - self.loss

    def __add__(self, other: "Settlement") -> "Settlement":
        return Settlement(self.win + other.win, self.push + other.push)

    def __mul__(self, factor: float) -> "Settlement":
        return Settlement(self.win * factor, self.push * factor)


WON = Settlement(1.0)
LOST = Settlement(0.0)
PUSHED = Settlement(0.0, 1.0)


def settle_1x2(selection: str, home_goals: int, away_goals: int) -> Settlement:
    winner = (
        "home" if home_goals > away_goals
        else ("away" if away_goals > home_goals else "draw")
    )
    return WON if selection == winner else LOST


def settle_double_chance(selection: str, home_goals: int, away_goals: int) -> Settlement:
    winner = (
        "home" if home_goals > away_goals
        else ("away" if away_goals > home_goals else "draw")
    )
    covered = {
        "home_or_draw": {"home", "draw"},
        "home_or_away": {"home", "away"},
        "draw_or_away": {"draw", "away"},
    }[selection]
    return WON if winner in covered else LOST


def settle_draw_no_bet(selection: str, home_goals: int, away_goals: int) -> Settlement:
    """Like 1X2, except the draw returns the stake instead of losing it."""
    if home_goals == away_goals:
        return PUSHED
    winner = "home" if home_goals > away_goals else "away"
    return WON if selection == winner else LOST


def settle_odd_even(selection: str, total: int) -> Settlement:
    """Odd or even total. Zero is even."""
    odd = int(total) % 2 == 1
    if selection == "odd":
        return WON if odd else LOST
    if selection == "even":
        return LOST if odd else WON
    raise ValueError(f"Unknown odd/even selection {selection!r}")


def settle_winning_margin(
    selection: str, home_goals: int, away_goals: int, max_margin: int = 3
) -> Settlement:
    """Who won and by how many, with the tail collapsed at `max_margin`.

    `max_margin` must match the value `markets.winning_margin` priced with, or
    the two disagree about which cell a 4-0 belongs to. It is a default in both
    places rather than a shared constant only because the pricing side takes it
    as an argument; if either ever changes, both must.
    """
    margin = int(home_goals) - int(away_goals)
    if margin == 0:
        return WON if selection == "draw" else LOST
    side = "home" if margin > 0 else "away"
    size = abs(margin)
    expected = (
        f"{side}_{max_margin}_plus" if size >= max_margin else f"{side}_{size}"
    )
    return WON if selection == expected else LOST


def settle_count_handicap(
    selection: str, line: float, home_count: float, away_count: float
) -> Settlement:
    """Handicap on corners or cards, settled exactly like the goals version.

    Delegates to `settle_asian_handicap` because the arithmetic is identical -
    a margin, a line, and the quarter-line split - and two copies of quarter
    handicap logic is one more than this project should own.
    """
    return settle_asian_handicap(
        selection, line, int(round(home_count)), int(round(away_count))
    )


def settle_over_under(selection: str, line: float, total: float) -> Settlement:
    """Over/under on any count: goals, corners, cards.

    Whole-number lines push on an exact total, which is why totals markets are
    sometimes quoted at 3.0 rather than 2.5.
    """
    if abs(total - line) < TOLERANCE:
        return PUSHED
    over = total > line
    if selection == "over":
        return WON if over else LOST
    if selection == "under":
        return LOST if over else WON
    raise ValueError(f"Unknown over/under selection {selection!r}")


def settle_btts(selection: str, home_goals: int, away_goals: int) -> Settlement:
    both_scored = home_goals > 0 and away_goals > 0
    if selection == "yes":
        return WON if both_scored else LOST
    if selection == "no":
        return LOST if both_scored else WON
    raise ValueError(f"Unknown BTTS selection {selection!r}")


def settle_asian_handicap(
    selection: str, line: float, home_goals: int, away_goals: int
) -> Settlement:
    """Asian handicap, with `line` given from the backed team's own side.

    Home -0.5 and away +0.5 are the two halves of the same market, and each is
    passed its own line. Quarter lines are split into the two neighbouring
    half-stakes and settled separately, which is what actually happens at the
    bookmaker rather than an approximation of it.
    """
    if selection == "home":
        margin = home_goals - away_goals + line
    elif selection == "away":
        margin = away_goals - home_goals + line
    else:
        raise ValueError(f"Unknown handicap selection {selection!r}")

    if _is_quarter_line(line):
        lower = _settle_margin(margin - 0.25)
        upper = _settle_margin(margin + 0.25)
        return (lower + upper) * 0.5
    return _settle_margin(margin)


def _settle_margin(margin: float) -> Settlement:
    if margin > TOLERANCE:
        return WON
    if margin < -TOLERANCE:
        return LOST
    return PUSHED


def _is_quarter_line(line: float) -> bool:
    return not math.isclose(round(line * 2) / 2, line, abs_tol=TOLERANCE)


def settle(
    market: str,
    selection: str,
    line: float | None,
    home_goals: int,
    away_goals: int,
    total_corners: float | None = None,
    total_cards: float | None = None,
    home_corners: float | None = None,
    away_corners: float | None = None,
    home_cards: float | None = None,
    away_cards: float | None = None,
    home_goals_ht: int | None = None,
    away_goals_ht: int | None = None,
) -> Settlement | None:
    """Dispatch to the right settlement rule.

    Returns None when the market cannot be settled from the data available -
    a corners bet on a match with no corner record, for instance. Callers must
    drop those rather than treating them as losses.
    """
    if market == "1x2":
        return settle_1x2(selection, home_goals, away_goals)
    if market == "double_chance":
        return settle_double_chance(selection, home_goals, away_goals)
    if market == "draw_no_bet":
        return settle_draw_no_bet(selection, home_goals, away_goals)
    if market == "btts":
        return settle_btts(selection, home_goals, away_goals)
    if market == "odd_even_goals":
        return settle_odd_even(selection, home_goals + away_goals)
    if market == "winning_margin":
        return settle_winning_margin(selection, home_goals, away_goals)
    if market == "total_goals":
        return settle_over_under(selection, float(line), home_goals + away_goals)
    if market == "home_goals":
        return settle_over_under(selection, float(line), home_goals)
    if market == "away_goals":
        return settle_over_under(selection, float(line), away_goals)
    if market == "asian_handicap":
        return settle_asian_handicap(selection, float(line), home_goals, away_goals)
    if market == "total_corners":
        if total_corners is None:
            return None
        return settle_over_under(selection, float(line), total_corners)
    if market == "total_cards":
        if total_cards is None:
            return None
        return settle_over_under(selection, float(line), total_cards)

    # Team-level and handicap variants of the count markets. Each needs both
    # sides separately, which the match total cannot supply, so they return
    # None rather than guessing when only the total was passed.
    if market in ("home_total_corners", "away_total_corners", "corner_handicap"):
        if home_corners is None or away_corners is None:
            return None
        if market == "corner_handicap":
            return settle_count_handicap(
                selection, float(line), home_corners, away_corners
            )
        side = home_corners if market.startswith("home") else away_corners
        return settle_over_under(selection, float(line), side)
    if market in ("home_total_cards", "away_total_cards", "card_handicap"):
        if home_cards is None or away_cards is None:
            return None
        if market == "card_handicap":
            return settle_count_handicap(
                selection, float(line), home_cards, away_cards
            )
        side = home_cards if market.startswith("home") else away_cards
        return settle_over_under(selection, float(line), side)

    # Half-time markets settle on the half-time score, which is a different
    # observation from the full-time one and is passed separately for exactly
    # that reason. It cannot be derived from the full-time score - which is the
    # concrete version of why the *pricing* side needs its own fit too.
    if market in ("1x2_ht", "total_goals_ht"):
        if home_goals_ht is None or away_goals_ht is None:
            return None
        if market == "1x2_ht":
            return settle_1x2(selection, int(home_goals_ht), int(away_goals_ht))
        return settle_over_under(
            selection, float(line), int(home_goals_ht) + int(away_goals_ht)
        )
    if market == "correct_score":
        expected = f"{home_goals}-{away_goals}"
        return WON if selection == expected else LOST
    raise ValueError(f"No settlement rule for market {market!r}")
