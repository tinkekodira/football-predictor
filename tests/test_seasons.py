"""Tests for the per-season and per-era CLV breakdown.

Two things here were worked out by hand before they were written down.

`test_season_label_puts_the_covid_restart_in_the_right_season` is the one that
matters most. The 2019-20 season finished in July 2020. Any labelling rule
that files those matches under 2020-21 pools the tail of one behind-closed-
doors season with the whole of the next one, which is precisely the boundary
this breakdown exists to draw.

`test_clustered_se_matches_the_naive_se_without_clusters` pins the variance
formula to the one already used in `closing_line_value`, so the two numbers
are known to be the same estimator seen under different assumptions rather
than two different estimators that happen to disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import evaluation  # noqa: E402


# --------------------------------------------------------------------------
# Season labelling
# --------------------------------------------------------------------------

def test_season_label_splits_at_august_not_january():
    labels = evaluation.season_labels(
        ["2022-08-06", "2022-12-26", "2023-01-02", "2023-05-28"]
    )
    # One season, four dates, two calendar years.
    assert list(labels) == [2022, 2022, 2022, 2022]


def test_season_label_puts_the_covid_restart_in_the_right_season():
    # The 2019-20 Premier League season was suspended in March 2020 and
    # finished on 26 July 2020, behind closed doors.
    labels = evaluation.season_labels(
        ["2020-03-09", "2020-06-17", "2020-07-26", "2020-09-12"]
    )
    assert list(labels) == [2019, 2019, 2019, 2020]


def test_season_label_boundary_is_the_first_of_august():
    labels = evaluation.season_labels(["2021-07-31", "2021-08-01"])
    assert list(labels) == [2020, 2021]


def test_season_label_cutover_is_configurable():
    labels = evaluation.season_labels(["2021-07-15"], cutover_month=7)
    assert list(labels) == [2021]


# --------------------------------------------------------------------------
# Clustered standard error
# --------------------------------------------------------------------------

def test_clustered_se_matches_the_naive_se_without_clusters():
    values = np.array([0.1, -0.2, 0.3, 0.05, -0.15])
    result = evaluation.clustered_mean(values, clusters=np.arange(len(values)))
    naive = values.std(ddof=1) / np.sqrt(len(values))
    assert result["se"] == pytest.approx(naive)
    assert result["mean"] == pytest.approx(values.mean())


def test_clustered_se_is_larger_when_bets_repeat_within_a_match():
    # Same eight numbers, but arranged as four matches of two identical bets.
    values = np.array([0.2, 0.2, -0.1, -0.1, 0.3, 0.3, -0.4, -0.4])
    independent = evaluation.clustered_mean(values, clusters=np.arange(8))
    clustered = evaluation.clustered_mean(
        values, clusters=np.array([0, 0, 1, 1, 2, 2, 3, 3])
    )
    assert clustered["se"] > independent["se"]
    assert clustered["n_clusters"] == 4
    assert clustered["n"] == 8


def test_perfectly_duplicated_bets_inflate_the_se_by_the_duplication_factor():
    # Each match contributes two identical bets, so the effective sample is
    # four, not eight. The clustered SE should equal the SE of the four
    # distinct match-level values.
    per_match = np.array([0.2, -0.1, 0.3, -0.4])
    doubled = np.repeat(per_match, 2)
    clustered = evaluation.clustered_mean(
        doubled, clusters=np.repeat(np.arange(4), 2)
    )
    by_hand = per_match.std(ddof=1) / np.sqrt(len(per_match))
    assert clustered["se"] == pytest.approx(by_hand)


def test_clustered_mean_ignores_missing_values():
    result = evaluation.clustered_mean(
        [0.1, np.nan, 0.3], clusters=[0, 1, 2]
    )
    assert result["n"] == 2
    assert result["mean"] == pytest.approx(0.2)


def test_clustered_mean_handles_an_empty_input():
    result = evaluation.clustered_mean([], clusters=[])
    assert result["n"] == 0
    assert np.isnan(result["mean"])


# --------------------------------------------------------------------------
# Season breakdown
# --------------------------------------------------------------------------

def _predictions(rows) -> pd.DataFrame:
    """Minimal predictions frame: date, match, market, edge, clv."""
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "date": date,
                "market": "1x2",
                "selection": "home",
                "expected_value": edge,
                "clv": clv,
                "push_fraction": 0.0,
                "win_fraction": 1.0,
                "model_conditional": 0.5,
                "market_probability": 0.5,
            }
            for match_id, date, edge, clv in rows
        ]
    )


def test_season_breakdown_separates_two_seasons():
    frame = _predictions(
        [
            ("a", "2020-10-01", 0.05, 0.04),
            ("b", "2021-02-01", 0.05, 0.02),
            ("c", "2022-10-01", 0.05, -0.03),
            ("d", "2023-02-01", 0.05, -0.01),
        ]
    )
    table = evaluation.season_breakdown(frame)
    assert list(table["season"]) == [2020, 2022]
    assert table.loc[0, "mean_clv"] == pytest.approx(0.03)
    assert table.loc[1, "mean_clv"] == pytest.approx(-0.02)
    assert list(table["bets"]) == [2, 2]


def test_season_breakdown_excludes_selections_below_the_edge_threshold():
    frame = _predictions(
        [
            ("a", "2022-10-01", 0.05, 0.10),
            ("b", "2022-11-01", 0.01, -0.90),  # never bet, must not count
        ]
    )
    table = evaluation.season_breakdown(frame, edge_threshold=0.02)
    assert table.loc[0, "bets"] == 1
    assert table.loc[0, "mean_clv"] == pytest.approx(0.10)
    assert table.loc[0, "priced"] == 2


def test_season_breakdown_accepts_an_authoritative_season_column():
    # Deliberately wrong dates: the explicit labels must win, because the
    # database's season_start_year is the source of truth when available.
    frame = _predictions(
        [("a", "2022-10-01", 0.05, 0.04), ("b", "2022-11-01", 0.05, 0.06)]
    )
    table = evaluation.season_breakdown(frame, season=pd.Series([1999, 1999]))
    assert list(table["season"]) == [1999]
    assert table.loc[0, "mean_clv"] == pytest.approx(0.05)


def test_season_breakdown_returns_empty_for_an_empty_frame():
    assert evaluation.season_breakdown(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Era comparison
# --------------------------------------------------------------------------

def test_era_comparison_recovers_a_sign_flip():
    rows = []
    for index in range(20):
        rows.append((f"old{index}", "2019-10-01", 0.05, 0.02))
    for index in range(20):
        rows.append((f"new{index}", "2023-10-01", 0.05, -0.01))
    result = evaluation.era_comparison(_predictions(rows), split_season=2021)

    assert result["n_before"] == 20
    assert result["n_after"] == 20
    assert result["mean_clv_before"] == pytest.approx(0.02)
    assert result["mean_clv_after"] == pytest.approx(-0.01)
    assert result["difference"] == pytest.approx(-0.03)


def test_era_comparison_reports_no_difference_when_there_is_none():
    rng = np.random.default_rng(11)
    rows = []
    for index in range(200):
        year = "2019" if index % 2 else "2023"
        rows.append(
            (f"m{index}", f"{year}-10-01", 0.05, float(rng.normal(0.0, 0.08)))
        )
    result = evaluation.era_comparison(_predictions(rows), split_season=2021)
    assert abs(result["difference_z"]) < 2.5


def test_era_comparison_needs_both_sides():
    frame = _predictions([("a", "2023-10-01", 0.05, 0.01)])
    result = evaluation.era_comparison(frame, split_season=2021)
    assert result["n_before"] == 0
    assert "difference" not in result
