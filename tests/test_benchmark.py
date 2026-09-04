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

import datetime as dt
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


def test_a_match_nobody_priced_does_not_break_the_benchmark_table():
    """**Rows without a benchmark are the ordinary case, not an edge case.**

    A match nobody priced reaches here with a null `fair_line_source`, and the
    project's own design keeps those rows: they are fitted, they inform team
    strengths, they count towards calibration, and they can never be settled as
    bets. `BacktestResult.fitted_not_bettable` exists precisely to count them.

    The function used to label seasons *after* filtering those rows out, then
    realign by the survivors' original labels - which asks for positions that
    no longer exist. It raised `KeyError` whenever the dropped rows were not a
    trailing block, which is to say almost always. That made the one diagnostic
    the project tells you to run before trusting a CLV trend unrunnable on the
    frames it is meant for.
    """
    rows = [
        ("2021-09-01", None), ("2021-09-02", None), ("2021-09-03", None),
        ("2024-09-01", "pinnacle"), ("2024-09-02", "pinnacle"),
        ("2024-09-03", "pinnacle"),
    ]
    table = evaluation.fair_line_sources(_predictions(rows))
    assert len(table) == 1
    assert table["season"].iloc[0] == 2024
    assert table["selections"].iloc[0] == 3


def test_a_season_is_read_from_its_own_date_not_its_position():
    """Interleaved gaps must not shift a row into a neighbouring season.

    The failure this guards is silent rather than loud: a row keeping its
    position but taking another row's season would move selections between
    seasons in the table, which is exactly the signal `benchmark_changed` reads.
    """
    rows = [
        ("2021-09-01", "pinnacle"), ("2021-09-02", None),
        ("2021-09-03", "pinnacle"), ("2024-09-01", None),
        ("2024-09-02", "betfair_exchange"), ("2024-09-03", None),
    ]
    table = evaluation.fair_line_sources(_predictions(rows)).set_index("season")
    assert table.loc[2021, "fair_line_source"] == "pinnacle"
    assert table.loc[2021, "selections"] == 2
    assert table.loc[2024, "fair_line_source"] == "betfair_exchange"
    assert table.loc[2024, "selections"] == 1


def test_an_explicit_season_column_survives_the_same_gaps():
    """`season_breakdown` passes seasons in rather than deriving them."""
    rows = [
        ("2021-09-01", None), ("2021-09-02", None),
        ("2024-09-01", "pinnacle"), ("2024-09-02", "pinnacle"),
    ]
    table = evaluation.fair_line_sources(
        _predictions(rows), season=pd.Series([2021, 2021, 2024, 2024])
    )
    assert len(table) == 1
    assert table["season"].iloc[0] == 2024


def test_sources_are_empty_without_the_column():
    frame = pd.DataFrame({"date": ["2022-10-01"], "match_id": ["m1"]})
    assert evaluation.fair_line_sources(frame).empty
    assert evaluation.benchmark_changed(pd.DataFrame()) == []


# ----------------------------------------------------------------------
# Pinning the benchmark (BACKLOG B1)
# ----------------------------------------------------------------------


def _two_book_market() -> pd.DataFrame:
    """One 1X2 market priced by Pinnacle and by a soft book."""
    rows = []
    for bookmaker, prices in (
        ("pinnacle", (2.00, 3.40, 4.00)),
        ("bet365", (1.90, 3.30, 3.80)),
    ):
        for selection, price in zip(("home", "draw", "away"), prices):
            rows.append({
                "market": "1x2", "selection": selection, "line": None,
                "bookmaker": bookmaker, "phase": "close", "price": price,
            })
    return pd.DataFrame(rows)


def test_naming_a_preference_alone_does_not_pin_the_benchmark():
    """The bug, pinned so it cannot come back silently.

    Asking for Pinnacle and getting bet365 wherever Pinnacle is quiet is a
    *preference*, not a pin, and every CLV number compared across seasons on
    that basis is measured against a moving ruler.
    """
    rows = _two_book_market()
    only_soft = rows[rows["bookmaker"] == "bet365"]
    out = backtest._market_probabilities(
        only_soft, "shin", preference=("pinnacle",), fallback=True
    )
    assert out, "with fallback on, a soft book still answers"
    assert {book for _, book in out.values()} == {"bet365"}


def test_turning_the_fallback_off_really_pins_it():
    rows = _two_book_market()
    only_soft = rows[rows["bookmaker"] == "bet365"]
    out = backtest._market_probabilities(
        only_soft, "shin", preference=("pinnacle",), fallback=False
    )
    assert out == {}, "a pinned run must drop what the named book did not price"


def test_a_pinned_run_still_uses_the_named_book_where_it_exists():
    """Pinning must not throw away the matches it is supposed to measure."""
    out = backtest._market_probabilities(
        _two_book_market(), "shin", preference=("pinnacle",), fallback=False
    )
    assert {book for _, book in out.values()} == {"pinnacle"}
    assert len(out) == 3


def test_the_fallback_defaults_to_on_so_nothing_silently_narrows():
    """Existing runs must keep their coverage; pinning is opt-in."""
    assert backtest.BacktestConfig(
        league="E0", start=dt.date(2024, 8, 1), end=dt.date(2025, 6, 1)
    ).fair_line_fallback is True
    out = backtest._market_probabilities(
        _two_book_market()[lambda f: f["bookmaker"] == "bet365"],
        "shin", preference=("pinnacle",),
    )
    assert out, "the default must behave as it always did"
