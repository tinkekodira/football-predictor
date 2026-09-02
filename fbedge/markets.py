"""From distributions to market prices.

The goals model produces one object - a matrix holding the probability of every
scoreline - and every goal-based market is a different way of summing it.
1X2 is the two triangles and the diagonal. Over 2.5 is the cells where the
totals exceed two. Both teams to score is everything outside the first row and
column. Because they all come from the same matrix, the prices cannot
contradict each other, which is not true of a system that fits each market
separately.

Count markets work the same way in one dimension: a distribution over totals,
summed either side of the line.

Every probability here is a *fair* probability, with no margin. Comparing it to
a bookmaker's price is Phase 4's job; `pricing.py` provides the arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
DEFAULT_CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
DEFAULT_CARD_LINES = (2.5, 3.5, 4.5, 5.5)
DEFAULT_TEAM_GOAL_LINES = (0.5, 1.5, 2.5)


@dataclass(frozen=True)
class Selection:
    """One bettable outcome with its model probability.

    `push_probability` matters for Asian handicaps on whole-goal lines, where
    the stake is returned on an exact-margin result. It changes the fair price,
    so it is carried rather than folded away.
    """

    market: str
    selection: str
    probability: float
    line: float | None = None
    push_probability: float = 0.0

    @property
    def fair_price(self) -> float:
        """Decimal odds at which this bet breaks even.

        With a push, the stake comes back on that outcome, so the price needed
        to break even on the remainder is (1 - push) / win.
        """
        if self.probability <= 0:
            return math.inf
        return (1.0 - self.push_probability) / self.probability

    @property
    def label(self) -> str:
        if self.line is None:
            return self.selection
        return f"{self.selection} {self.line:+g}" if self.market == "asian_handicap" \
            else f"{self.selection} {self.line:g}"


# --------------------------------------------------------------------------
# Goal markets, all derived from the scoreline matrix
# --------------------------------------------------------------------------

def match_odds(matrix: np.ndarray) -> list[Selection]:
    """1X2: home win is the lower triangle, draw the diagonal, away the upper."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return [
        Selection("1x2", "home", home),
        Selection("1x2", "draw", draw),
        Selection("1x2", "away", away),
    ]


def double_chance(matrix: np.ndarray) -> list[Selection]:
    home, draw, away = (s.probability for s in match_odds(matrix))
    return [
        Selection("double_chance", "home_or_draw", home + draw),
        Selection("double_chance", "home_or_away", home + away),
        Selection("double_chance", "draw_or_away", draw + away),
    ]


def total_goals(
    matrix: np.ndarray, lines: tuple[float, ...] = DEFAULT_GOAL_LINES
) -> list[Selection]:
    """Over/under at each line.

    Whole-number lines push on an exact total, which is why a book quoting
    "over 3.0" is offering something different from "over 2.5". Treating the
    exact total as a loss for the over would misprice it by several percent.
    """
    size = matrix.shape[0]
    totals = np.add.outer(np.arange(size), np.arange(size))
    out: list[Selection] = []
    for line in lines:
        out += _over_under(matrix, totals, "total_goals", float(line))
    return out


def _over_under(matrix, totals, market: str, line: float) -> list[Selection]:
    over = float(matrix[totals > line].sum())
    under = float(matrix[totals < line].sum())
    push = float(matrix[np.abs(totals - line) < 1e-9].sum())
    return [
        Selection(market, "over", over, line=line, push_probability=push),
        Selection(market, "under", under, line=line, push_probability=push),
    ]


def both_teams_to_score(matrix: np.ndarray) -> list[Selection]:
    yes = float(matrix[1:, 1:].sum())
    return [
        Selection("btts", "yes", yes),
        Selection("btts", "no", 1.0 - yes),
    ]


def team_totals(
    matrix: np.ndarray, lines: tuple[float, ...] = DEFAULT_TEAM_GOAL_LINES
) -> list[Selection]:
    """Over/under on each side's own goals."""
    goals = np.arange(matrix.shape[0])
    out: list[Selection] = []
    for side, pmf in (("home", matrix.sum(axis=1)), ("away", matrix.sum(axis=0))):
        for line in lines:
            out += _over_under(pmf, goals, f"{side}_goals", float(line))
    return out


def asian_handicap(matrix: np.ndarray, line: float) -> list[Selection]:
    """Asian handicap, with `line` applied to the home team.

    A negative line means the home side gives a start: -1.5 requires them to
    win by two or more. Each selection reports the line from its own side's
    point of view, so home -1.5 and away +1.5 are the two halves of one market.

    A quarter line such as -0.25 is genuinely two half-stake bets, at 0.0 and
    -0.5, so it is computed that way rather than approximated. Whole-number
    lines can push, and the push probability is reported so that the fair price
    accounts for the returned stake.
    """
    if not _is_quarter_line(line):
        return _asian_handicap_simple(matrix, line)

    lower = _asian_handicap_simple(matrix, line - 0.25)
    upper = _asian_handicap_simple(matrix, line + 0.25)
    signs = (1.0, -1.0)  # home reports +line, away reports -line
    return [
        Selection(
            "asian_handicap",
            low.selection,
            0.5 * (low.probability + high.probability),
            line=sign * line,
            push_probability=0.5 * (low.push_probability + high.push_probability),
        )
        for low, high, sign in zip(lower, upper, signs)
    ]


def _is_quarter_line(line: float) -> bool:
    return not math.isclose(round(line * 2) / 2, line, abs_tol=1e-9)


def _asian_handicap_simple(matrix: np.ndarray, line: float) -> list[Selection]:
    size = matrix.shape[0]
    margin = np.subtract.outer(np.arange(size), np.arange(size)) + line
    home_win = float(matrix[margin > 1e-9].sum())
    push = float(matrix[np.abs(margin) < 1e-9].sum())
    away_win = float(matrix[margin < -1e-9].sum())
    return [
        Selection("asian_handicap", "home", home_win, line=line, push_probability=push),
        Selection("asian_handicap", "away", away_win, line=-line, push_probability=push),
    ]


def correct_score(matrix: np.ndarray, top_n: int = 8) -> list[Selection]:
    """The most likely scorelines, highest probability first."""
    flat = matrix.flatten()
    order = np.argsort(flat)[::-1][:top_n]
    out = []
    for position in order:
        home, away = divmod(int(position), matrix.shape[1])
        out.append(
            Selection("correct_score", f"{home}-{away}", float(flat[position]))
        )
    return out


def expected_total(matrix: np.ndarray) -> float:
    size = matrix.shape[0]
    totals = np.add.outer(np.arange(size), np.arange(size))
    return float((matrix * totals).sum())


def total_distribution(matrix: np.ndarray) -> np.ndarray:
    """The whole distribution of the match total, not just its mean.

    The scoreline matrix holds P(home=i, away=j) and the total is i+j, so this
    sums the anti-diagonals.

    Worth having separately from `expected_total` because the mean cannot
    answer the question that matters about shape. `models/counts.py` uses a
    negative binomial for corners and cards precisely because a Poisson is too
    narrow and underprices the tails; nobody ever checked whether the same is
    true of goals. Comparing the spread of realised totals against the spread
    this distribution predicts is that check, and it needs the full
    distribution rather than a point estimate.
    """
    size = matrix.shape[0]
    totals = np.add.outer(np.arange(size), np.arange(size))
    return np.bincount(
        totals.ravel(), weights=matrix.ravel(), minlength=2 * size - 1
    )


# --------------------------------------------------------------------------
# Count markets (corners, cards)
# --------------------------------------------------------------------------

def price_selection(
    matrix: np.ndarray, market: str, selection: str, line: float | None
) -> Selection | None:
    """Price one specific selection at one specific line.

    The backtest needs this because a bookmaker's line is whatever the
    bookmaker chose, not whatever the model would have quoted. Pricing the
    book's own line is the only way to compare like with like; asking the
    model for over 2.5 when the book is offering over 3.0 compares nothing.

    Returns None for a market this module cannot price from a scoreline
    matrix, so the caller can skip it rather than crash.
    """
    if market == "1x2":
        return next((s for s in match_odds(matrix) if s.selection == selection), None)
    if market == "double_chance":
        return next(
            (s for s in double_chance(matrix) if s.selection == selection), None
        )
    if market == "btts":
        return next(
            (s for s in both_teams_to_score(matrix) if s.selection == selection), None
        )
    if line is None:
        return None
    if market == "total_goals":
        return next(
            (s for s in total_goals(matrix, (float(line),)) if s.selection == selection),
            None,
        )
    if market in ("home_goals", "away_goals"):
        return next(
            (
                s
                for s in team_totals(matrix, (float(line),))
                if s.market == market and s.selection == selection
            ),
            None,
        )
    if market == "asian_handicap":
        # `line` arrives from the backed side's own point of view; the helper
        # works in home-team terms, so an away line is flipped back.
        home_line = float(line) if selection == "home" else -float(line)
        pair = asian_handicap(matrix, home_line)
        return next((s for s in pair if s.selection == selection), None)
    return None


def price_count_selection(
    pmf: np.ndarray, market: str, selection: str, line: float | None
) -> Selection | None:
    """Price one over/under selection from a count distribution."""
    if line is None:
        return None
    pair = count_totals(pmf, market, (float(line),))
    return next((s for s in pair if s.selection == selection), None)


def count_totals(pmf: np.ndarray, market: str, lines) -> list[Selection]:
    """Over/under on a count total, from a one-dimensional distribution."""
    counts = np.arange(len(pmf))
    out: list[Selection] = []
    for line in lines:
        out += _over_under(pmf, counts, market, float(line))
    return out


def count_mean(pmf: np.ndarray) -> float:
    return float((pmf * np.arange(len(pmf))).sum())


def suggested_lines(mean: float, spread: float = 2.0, step: float = 1.0) -> tuple[float, ...]:
    """Half-integer lines bracketing an expected count.

    Quoting lines near the model's own mean is more useful than a fixed list,
    because that is where a bookmaker will have set the line too.
    """
    centre = math.floor(mean) + 0.5
    count = int(spread / step)
    return tuple(centre + step * offset for offset in range(-count, count + 1))
