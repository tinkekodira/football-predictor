"""Tests for the closing-benchmark provenance guard.

These exist because of a measurement bug that cost a session's worth of wrong
conclusions. `betfair_exchange` first appears in football-data.co.uk in 2024-25
and it heads `FAIR_LINE_PREFERENCE`, so the benchmark CLV is measured against
switched from Pinnacle to Betfair partway through the window - with nothing in
any output saying so. Measured on the overlap, the exchange scores the same
bets about 1.75 points lower, which accounted for most of an apparent "regime
change" that was read as the market sharpening.

The lesson worth pinning down in tests is narrow and mechanical: the supplying
bookmaker must survive into the predictions frame, and a change in it must
raise a warning without anyone having to think to look.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, evaluation  # noqa: E402


# --------------------------------------------------------------------------
# The fair line records where it came from
# --------------------------------------------------------------------------

def _odds(bookmakers, prices=(2.0, 4.0, 4.0)) -> pd.DataFrame:
    """A complete 1X2 close for each named bookmaker."""
    rows = []
    for bookmaker in bookmakers:
        for selection, price in zip(("home", "draw", "away"), prices):
            rows.append(
                {
                    "match_id": "m1",
                    "bookmaker": bookmaker,
                    "phase": "close",
                    "market": "1x2",
                    "selection": selection,
                    "line": None,
                    "price": price,
                }
            )
    return pd.DataFrame(rows)


def test_fair_line_reports_which_bookmaker_supplied_it():
    out = backtest._market_probabilities(
        _odds(["pinnacle"]), "multiplicative", ("pinnacle",)
    )
    probability, source = out[("1x2", "home", None)]
    assert source == "pinnacle"
    assert probability == pytest.approx(0.5)


def test_preference_order_decides_the_supplier():
    odds = _odds(["pinnacle", "betfair_exchange"])
    betfair_first = backtest._market_probabilities(
        odds, "multiplicative", ("betfair_exchange", "pinnacle")
    )
    pinnacle_first = backtest._market_probabilities(
        odds, "multiplicative", ("pinnacle", "betfair_exchange")
    )
    assert betfair_first[("1x2", "home", None)][1] == "betfair_exchange"
    assert pinnacle_first[("1x2", "home", None)][1] == "pinnacle"


def test_a_book_outside_the_preference_is_still_used_as_a_last_resort():
    # Coverage beats purity: a market nobody preferred is better priced by a
    # soft book than left unpriced. It just has to be visible afterwards.
    out = backtest._market_probabilities(
        _odds(["some_soft_book"]), "multiplicative", ("pinnacle",)
    )
    assert out[("1x2", "home", None)][1] == "some_soft_book"


def test_an_incomplete_market_is_not_priced():
    odds = _odds(["pinnacle"]).iloc[:2]  # two legs of a three-leg market
    assert backtest._market_probabilities(odds, "multiplicative", ("pinnacle",)) == {}


# --------------------------------------------------------------------------
# Detecting a benchmark change
# --------------------------------------------------------------------------

def _predictions(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": date, "fair_line_source": source, "match_id": f"m{i}"}
            for i, (date, source) in enumerate(rows)
        ]
    )


def test_fair_line_sources_reports_shares_by_season():
    frame = _predictions(
        [("2022-10-01", "pinnacle")] * 3 + [("2022-11-01", "bet365")]
    )
    table = evaluation.fair_line_sources(frame)
    pinnacle = table[table["fair_line_source"] == "pinnacle"].iloc[0]
    assert pinnacle["share"] == pytest.approx(0.75)


def test_benchmark_change_is_flagged():
    # The real shape of the bug: Pinnacle throughout, then Betfair appears.
    rows = [(f"{year}-10-01", "pinnacle") for year in (2021, 2022, 2023)] * 10
    rows += [(f"{year}-10-01", "betfair_exchange") for year in (2024, 2025)] * 10
    warnings_out = evaluation.benchmark_changed(
        evaluation.fair_line_sources(_predictions(rows))
    )
    assert len(warnings_out) == 1
    assert "2024" in warnings_out[0]
    assert "pinnacle" in warnings_out[0]
    assert "betfair_exchange" in warnings_out[0]


def test_a_stable_benchmark_raises_nothing():
    rows = [(f"{year}-10-01", "pinnacle") for year in range(2017, 2026)] * 10
    assert evaluation.benchmark_changed(
        evaluation.fair_line_sources(_predictions(rows))
    ) == []


def test_a_minority_book_does_not_count_as_a_change():
    # One season mostly Pinnacle with a handful of fallbacks is still Pinnacle.
    rows = [("2022-10-01", "pinnacle")] * 90 + [("2022-10-01", "bet365")] * 10
    rows += [("2023-10-01", "pinnacle")] * 90 + [("2023-10-01", "market_avg")] * 10
    assert evaluation.benchmark_changed(
        evaluation.fair_line_sources(_predictions(rows))
    ) == []


def test_sources_are_empty_without_the_column():
    frame = pd.DataFrame({"date": ["2022-10-01"], "match_id": ["m1"]})
    assert evaluation.fair_line_sources(frame).empty
    assert evaluation.benchmark_changed(pd.DataFrame()) == []
