"""Tests for the injury retest's feature construction.

The distinction these pin is the whole finding. A player out for the season
appears in *every* fixture, and the model has already adapted to his absence by
the third one - the team's recent results were produced without him, so the
fitted rate carries it. Counting him again asks the model to subtract the same
player twice, and the measurement says exactly that: the plain count comes back
at zero while newly-missing players show a clear effect.

So `new_out_counts` must count a change and `out_counts` must count a state,
and neither must quietly become the other.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts.injury_signal import new_out_counts, out_counts


def rows(*entries) -> pd.DataFrame:
    """(team, date, player, ruled_out, doubtful) tuples as a stored frame."""
    return pd.DataFrame(
        [
            {
                "team": team,
                "fixture_date": dt.date.fromisoformat(date),
                "player": player,
                "ruled_out": ruled_out,
                "doubtful": doubtful,
            }
            for team, date, player, ruled_out, doubtful in entries
        ]
    )


# ----------------------------------------------------------------------
# The state: who is out
# ----------------------------------------------------------------------


def test_out_counts_counts_everyone_currently_out():
    frame = out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Arsenal", "2024-08-17", "B", True, False),
        ),
        doubtful=False,
    )
    assert float(frame.iloc[0]["out"]) == 2.0


def test_out_counts_excludes_doubtful_unless_asked():
    stored = rows(
        ("Arsenal", "2024-08-17", "A", True, False),
        ("Arsenal", "2024-08-17", "B", False, True),
    )
    assert float(out_counts(stored, doubtful=False).iloc[0]["out"]) == 1.0
    assert float(out_counts(stored, doubtful=True).iloc[0]["out"]) == 2.0


def test_the_same_player_twice_is_counted_once():
    """The feed sends a row per player per fixture and can repeat one."""
    frame = out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Arsenal", "2024-08-17", "A", True, False),
        ),
        doubtful=False,
    )
    assert float(frame.iloc[0]["out"]) == 1.0


# ----------------------------------------------------------------------
# The change: who is newly out
# ----------------------------------------------------------------------


def test_a_long_term_absence_counts_once_not_every_week():
    """The finding, in one test. A player out all season is news on the day it
    happens and nothing afterwards, because by then the ratings have seen the
    team play without him."""
    frame = new_out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Arsenal", "2024-08-24", "A", True, False),
            ("Arsenal", "2024-08-31", "A", True, False),
        ),
        doubtful=False,
    ).sort_values("fixture_date")
    assert list(frame["out"]) == [1.0, 0.0, 0.0]


def test_a_fresh_absence_is_counted_on_the_day_it_appears():
    frame = new_out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Arsenal", "2024-08-24", "A", True, False),
            ("Arsenal", "2024-08-24", "B", True, False),
        ),
        doubtful=False,
    ).sort_values("fixture_date")
    assert list(frame["out"]) == [1.0, 1.0]


def test_a_relapse_across_a_gap_is_undercounted_on_purpose():
    """Out on the 17th, no row on the 24th, out again on the 31st.

    The feed emits a row only when somebody is unavailable, so a missing date
    means either "the team played and everyone was fit" or "the team did not
    play". Injuries alone cannot tell those apart, and guessing would invent
    news that may never have happened.

    So the comparison is between consecutive *observations*, which counts this
    as one absence rather than two. That undercounts real relapses and biases
    the study towards finding **less** signal than there is - the safe
    direction for a result the project would like to believe.
    """
    frame = new_out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Arsenal", "2024-08-31", "A", True, False),
        ),
        doubtful=False,
    ).sort_values("fixture_date")
    assert list(frame["out"]) == [1.0, 0.0]


def test_teams_are_tracked_separately():
    """Arsenal's news must not reset Chelsea's."""
    frame = new_out_counts(
        rows(
            ("Arsenal", "2024-08-17", "A", True, False),
            ("Chelsea", "2024-08-17", "C", True, False),
            ("Arsenal", "2024-08-24", "A", True, False),
            ("Chelsea", "2024-08-24", "C", True, False),
        ),
        doubtful=False,
    )
    arsenal = frame[frame["team"] == "Arsenal"].sort_values("fixture_date")
    chelsea = frame[frame["team"] == "Chelsea"].sort_values("fixture_date")
    assert list(arsenal["out"]) == [1.0, 0.0]
    assert list(chelsea["out"]) == [1.0, 0.0]


def test_new_and_total_disagree_which_is_the_point():
    """If these two ever returned the same thing the study would be measuring
    one specification twice and calling it a replication."""
    stored = rows(
        ("Arsenal", "2024-08-17", "A", True, False),
        ("Arsenal", "2024-08-24", "A", True, False),
    )
    total = out_counts(stored, doubtful=False).sort_values("fixture_date")
    fresh = new_out_counts(stored, doubtful=False).sort_values("fixture_date")
    assert list(total["out"]) == [1, 1]
    assert list(fresh["out"]) == [1.0, 0.0]


# ----------------------------------------------------------------------
# Empty input
# ----------------------------------------------------------------------


@pytest.mark.parametrize("builder", [out_counts, new_out_counts])
def test_no_injuries_gives_an_empty_frame_with_the_right_columns(builder):
    """The caller checks the join rate against this rather than crashing."""
    frame = builder(pd.DataFrame(), doubtful=False)
    assert frame.empty
    assert {"team", "fixture_date", "out"} <= set(frame.columns)


@pytest.mark.parametrize("builder", [out_counts, new_out_counts])
def test_nobody_matching_the_filter_gives_an_empty_frame(builder):
    frame = builder(
        rows(("Arsenal", "2024-08-17", "A", False, True)), doubtful=False
    )
    assert frame.empty
