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


def draw_no_bet(matrix: np.ndarray) -> list[Selection]:
    """1X2 with the draw removed and the stake returned on it.

    A pure marginalisation of the same matrix, not a separate view of the
    match: the draw becomes a push rather than a loss, so the fair price is
    (1 - draw) / win. Deriving it any other way would let it disagree with the
    1X2 it came from.
    """
    home, draw, away = (s.probability for s in match_odds(matrix))
    return [
        Selection("draw_no_bet", "home", home, push_probability=draw),
        Selection("draw_no_bet", "away", away, push_probability=draw),
    ]


def winning_margin(matrix: np.ndarray, max_margin: int = 3) -> list[Selection]:
    """Who wins and by how many, with the tail collapsed.

    The tail is collapsed at `max_margin` because the individual cells beyond
    it are thin enough that the model's estimate of them is mostly the shape of
    the Poisson rather than anything it has learned. Collapsing rather than
    truncating keeps the selections summing to one, which is what makes this
    consistent with the 1X2 it is derived from.
    """
    size = matrix.shape[0]
    margin = np.subtract.outer(np.arange(size), np.arange(size))
    out = [Selection("winning_margin", "draw", float(np.trace(matrix)))]
    for side, sign in (("home", 1), ("away", -1)):
        for by in range(1, max_margin):
            out.append(
                Selection(
                    "winning_margin", f"{side}_{by}",
                    float(matrix[margin == sign * by].sum()),
                )
            )
        tail = matrix[sign * margin >= max_margin].sum()
        out.append(
            Selection("winning_margin", f"{side}_{max_margin}_plus", float(tail))
        )
    return out


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
    if market == "draw_no_bet":
        return next(
            (s for s in draw_no_bet(matrix) if s.selection == selection), None
        )
    if market == "btts":
        return next(
            (s for s in both_teams_to_score(matrix) if s.selection == selection), None
        )
    if market == "winning_margin":
        return next(
            (s for s in winning_margin(matrix) if s.selection == selection), None
        )
    if market in ("1x2_ht", "total_goals_ht"):
        # Half-time markets are priced from their own matrix, which this
        # function does not have. Returning None rather than quietly halving
        # the full-time rates is the whole point: see `all_goal_selections`.
        return None
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


def price_count_handicap(
    home_pmf: np.ndarray, away_pmf: np.ndarray,
    market: str, selection: str, line: float | None,
) -> Selection | None:
    """Price one count-handicap selection, the line read from the backed side."""
    if line is None:
        return None
    home_line = float(line) if selection == "home" else -float(line)
    pair = count_handicap(home_pmf, away_pmf, market, home_line)
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


def count_difference(home_pmf: np.ndarray, away_pmf: np.ndarray):
    """Distribution of (home count - away count), assuming independence.

    Returns `(margins, probabilities)` with `margins` running from
    `-(len(away)-1)` to `+(len(home)-1)`.

    **Independence is an approximation here and it is the opposite of the one
    the module makes for totals.** `models/counts.py` fits the match total
    separately from the two team rates precisely because the two are positively
    correlated - both sides win more corners in an open game - so convolving
    two independent team distributions *understates* how much a total scatters.
    A difference runs the other way: positive correlation cancels in a
    subtraction, so independence *overstates* how much the difference scatters,
    and a handicap priced from it is slightly too generous to the outsider.

    That is a real approximation, stated rather than hidden, and it is the best
    available: nothing in this source records the joint distribution, and there
    is no historical corner or card handicap price to check it against.
    """
    joint = np.outer(np.asarray(home_pmf, float), np.asarray(away_pmf, float))
    size_home, size_away = joint.shape
    margins = np.subtract.outer(np.arange(size_home), np.arange(size_away))
    offsets = np.arange(-(size_away - 1), size_home)
    weights = np.bincount(
        (margins - offsets[0]).ravel(), weights=joint.ravel(), minlength=len(offsets)
    )
    return offsets, weights


def count_handicap(
    home_pmf: np.ndarray, away_pmf: np.ndarray, market: str, line: float
) -> list[Selection]:
    """Handicap on a count, with `line` applied to the home team.

    Same convention as the goals handicap: a negative line means the home side
    gives that start, each selection reports the line from its own point of
    view, and a whole line can push. Quarter lines are two half-stake bets and
    are computed that way rather than approximated.

    See `count_difference` for the independence assumption this rests on and
    which way it errs.
    """
    if _is_quarter_line(line):
        lower = count_handicap(home_pmf, away_pmf, market, line - 0.25)
        upper = count_handicap(home_pmf, away_pmf, market, line + 0.25)
        signs = (1.0, -1.0)
        return [
            Selection(
                market, low.selection,
                0.5 * (low.probability + high.probability),
                line=sign * line,
                push_probability=0.5 * (low.push_probability + high.push_probability),
            )
            for low, high, sign in zip(lower, upper, signs)
        ]

    offsets, weights = count_difference(home_pmf, away_pmf)
    adjusted = offsets + line
    return [
        Selection(market, "home", float(weights[adjusted > 1e-9].sum()), line=line,
                  push_probability=float(weights[np.abs(adjusted) < 1e-9].sum())),
        Selection(market, "away", float(weights[adjusted < -1e-9].sum()), line=-line,
                  push_probability=float(weights[np.abs(adjusted) < 1e-9].sum())),
    ]


def team_count_totals(
    home_pmf: np.ndarray, away_pmf: np.ndarray, market: str, lines
) -> list[Selection]:
    """Over/under on each side's own count.

    The count models fit a team rate and a match total separately by design -
    see `models/counts.py` - so a team's own distribution is already there and
    these are a straight read of it rather than a new fit. The match total is
    *not* the sum of these two: summing two independent team distributions
    understates how much totals scatter, because both teams win more corners in
    an open game than a closed one. Use `count_totals` for the match total.
    """
    out: list[Selection] = []
    for side, pmf in (("home", home_pmf), ("away", away_pmf)):
        counts = np.arange(len(pmf))
        for line in lines:
            out += _over_under(pmf, counts, f"{side}_{market}", float(line))
    return out


# --------------------------------------------------------------------------
# The whole set, from one matrix
# --------------------------------------------------------------------------

def all_goal_selections(
    matrix: np.ndarray,
    handicap_lines: tuple[float, ...] = (),
    goal_lines: tuple[float, ...] = DEFAULT_GOAL_LINES,
    team_goal_lines: tuple[float, ...] = DEFAULT_TEAM_GOAL_LINES,
    correct_scores: int = 8,
) -> list[Selection]:
    """Every goal market this module can price, from one scoreline matrix.

    **This function is the internal-consistency guarantee, made operational.**
    The README's claim that "because every price comes from one matrix, they
    cannot contradict each other" is only true while every caller derives its
    prices the same way, and there were three callers doing it by hand before
    this existed - the CLI, the app and the backtest - which is three chances
    for one of them to drift. `test_markets_are_mutually_consistent` asserts
    the arithmetic; this makes there be one place for it to hold.

    Two markets are deliberately absent. **Half-time** is not derivable from a
    full-time matrix, because scoring rates are not uniform across the halves;
    it comes from its own fit. **Odd/even total goals** was priced here until
    2026-09-04 and was removed after it was measured: its calibration slope ran
    from -2.41 to +1.54 across five leagues, so in at least one the model's
    confidence pointed the wrong way. That is the expected result for a market
    that is close to a coin flip by construction, and a market the model cannot
    rank is not worth the row it occupies. See BACKLOG B16.
    """
    out: list[Selection] = []
    out += match_odds(matrix)
    out += double_chance(matrix)
    out += draw_no_bet(matrix)
    out += total_goals(matrix, goal_lines)
    out += both_teams_to_score(matrix)
    out += team_totals(matrix, team_goal_lines)
    out += winning_margin(matrix)
    for line in handicap_lines:
        out += asian_handicap(matrix, line)
    if correct_scores:
        out += correct_score(matrix, correct_scores)
    return out


def suggested_lines(mean: float, spread: float = 2.0, step: float = 1.0) -> tuple[float, ...]:
    """Half-integer lines bracketing an expected count.

    Quoting lines near the model's own mean is more useful than a fixed list,
    because that is where a bookmaker will have set the line too.
    """
    centre = math.floor(mean) + 0.5
    count = int(spread / step)
    return tuple(centre + step * offset for offset in range(-count, count + 1))
