"""Tests for the availability features.

`test_features_ignore_the_match_they_describe` is the reason this file exists.
Understat records who actually played, and confirmed line-ups appear about an
hour before kick-off - after the price this project bets into. A feature that
leaked that would not present as a bug; availability genuinely matters, so it
would present as a large and convincing edge, and the model would look like it
had finally found something. The test rewrites the match being described into
something absurd and asserts the numbers do not move.

The rest pin the two design decisions that are easy to undo by accident: that
importance and absence are read from windows which do not overlap, and that a
long-term absentee is not counted as missing.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import availability  # noqa: E402


def _season(
    n_matches: int = 14,
    drop_player: str | None = None,
    drop_from: int | None = None,
    team: str = "T",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One team playing `n_matches`, optionally losing a player part-way."""
    lineup_rows, fixtures = [], []
    for index in range(n_matches):
        match_id = f"m{index:02d}"
        fixtures.append(
            {
                "match_id": match_id,
                "date": dt.date(2023, 1, 1) + dt.timedelta(days=7 * index),
                "home_team": team,
                "away_team": "Opponent",
            }
        )
        squad = [f"P{i}" for i in range(11)]
        if drop_player is not None and drop_from is not None and index >= drop_from:
            squad = [p for p in squad if p != drop_player] + ["Reserve"]
        for player in squad:
            lineup_rows.append(
                {
                    "match_id": match_id, "is_home": True, "player_id": player,
                    "minutes": 90, "started": True,
                    # P0 is the side's creative hub; everyone else contributes
                    # equally little. Same minutes, very different importance.
                    "xgchain": 1.0 if player == "P0" else 0.1,
                }
            )
        for player in [f"Q{i}" for i in range(11)]:
            lineup_rows.append(
                {
                    "match_id": match_id, "is_home": False, "player_id": player,
                    "minutes": 90, "started": True, "xgchain": 0.2,
                }
            )
    return pd.DataFrame(lineup_rows), pd.DataFrame(fixtures)


def _for_team(features: pd.DataFrame, team: str = "T") -> pd.DataFrame:
    return features[features["team"] == team].sort_values("n_prior").reset_index(drop=True)


# --------------------------------------------------------------------------
# The point-in-time rule
# --------------------------------------------------------------------------

def test_features_ignore_the_match_they_describe():
    """Rewriting a match must not change that match's own features.

    This is the whole safety property. If it ever fails, every downstream
    number built on these features is measuring the future.
    """
    lineups, matches = _season()
    before = _for_team(availability.availability_features(lineups, matches))

    # Empty out the last match entirely: in reality that would mean nobody
    # played, which should be invisible to that match's own features.
    tampered = lineups[
        ~((lineups["match_id"] == "m13") & (lineups["is_home"]))
    ].copy()
    # ...and put a completely different eleven in its place.
    tampered = pd.concat(
        [
            tampered,
            pd.DataFrame([
                {"match_id": "m13", "is_home": True, "player_id": f"Z{i}",
                 "minutes": 90, "started": True}
                for i in range(11)
            ]),
        ],
        ignore_index=True,
    )
    after = _for_team(availability.availability_features(tampered, matches))

    last_before = before[before["match_id"] == "m13"].iloc[0]
    last_after = after[after["match_id"] == "m13"].iloc[0]
    assert last_before["missing_share"] == pytest.approx(last_after["missing_share"])
    assert last_before["n_missing"] == pytest.approx(last_after["n_missing"])


def test_an_absence_is_invisible_to_the_match_it_happens_in():
    """The match where a player first goes missing cannot know about it."""
    lineups, matches = _season(drop_player="P9", drop_from=12)
    table = _for_team(availability.availability_features(lineups, matches))

    first_absence = table[table["match_id"] == "m12"].iloc[0]
    next_match = table[table["match_id"] == "m13"].iloc[0]
    assert first_absence["missing_share"] == pytest.approx(0.0)
    assert next_match["missing_share"] > 0.0


# --------------------------------------------------------------------------
# What the numbers mean
# --------------------------------------------------------------------------

def test_one_missing_regular_is_one_eleventh_of_the_minutes():
    lineups, matches = _season(drop_player="P9", drop_from=12)
    table = _for_team(availability.availability_features(lineups, matches))
    row = table[table["match_id"] == "m13"].iloc[0]
    assert row["missing_share"] == pytest.approx(1 / 11)
    assert row["missing_starter_share"] == pytest.approx(1 / 11)
    assert row["n_missing"] == 1


def test_a_full_squad_reports_nobody_missing():
    lineups, matches = _season()
    table = _for_team(availability.availability_features(lineups, matches))
    settled = table[table["n_prior"] >= availability.MIN_PRIOR_MATCHES]
    assert (settled["missing_share"] == 0.0).all()


def test_a_long_term_absentee_stops_counting_as_missing():
    """The point of measuring *newly* missing players.

    Someone out for months has already been absent from the results the model
    learned the team's strength from, so their absence is priced in. Counting
    them for ever would double-count it, and would also mean every team looked
    permanently weakened by a player who retired.
    """
    lineups, matches = _season(n_matches=26, drop_player="P9", drop_from=12)
    table = _for_team(availability.availability_features(lineups, matches))
    just_after = table[table["match_id"] == "m13"].iloc[0]["missing_share"]
    long_after = table[table["match_id"] == "m25"].iloc[0]["missing_share"]
    assert just_after > 0.0
    assert long_after == pytest.approx(0.0)


def test_a_fringe_player_does_not_count_as_a_missing_starter():
    lineups, matches = _season()
    # A substitute who plays ten minutes now and then, then disappears.
    extra = []
    for index in range(12):
        if index % 3 == 0:
            extra.append(
                {"match_id": f"m{index:02d}", "is_home": True,
                 "player_id": "Fringe", "minutes": 10, "started": False}
            )
    lineups = pd.concat([lineups, pd.DataFrame(extra)], ignore_index=True)
    table = _for_team(availability.availability_features(lineups, matches))
    row = table[table["match_id"] == "m13"].iloc[0]
    assert row["missing_share"] > 0.0          # they do count as missing
    assert row["missing_starter_share"] == pytest.approx(0.0)   # but not as a starter


def test_features_are_missing_rather_than_zero_without_enough_history():
    lineups, matches = _season()
    table = _for_team(availability.availability_features(lineups, matches))
    early = table[table["n_prior"] < availability.MIN_PRIOR_MATCHES]
    assert len(early) == availability.MIN_PRIOR_MATCHES
    assert early["missing_share"].isna().all()


def test_windows_do_not_overlap():
    """Importance must not be measured over the match the absence is read from.

    If they overlapped, missing the most recent match would itself shrink the
    player's measured importance, and the feature would understate exactly the
    absences it exists to catch.
    """
    lineups, matches = _season(drop_player="P9", drop_from=13)
    wide = availability.availability_features(
        lineups, matches, importance_window=10, absence_window=1
    )
    row = _for_team(wide)
    row = row[row["match_id"] == "m13"].iloc[0]
    # P9 played all ten importance matches and missed none of them, so the
    # share must be exactly one eleventh - not reduced by their own absence.
    assert row["missing_share"] == pytest.approx(0.0)


def test_empty_input_returns_an_empty_frame():
    empty = pd.DataFrame(
        columns=["match_id", "is_home", "player_id", "minutes", "started"]
    )
    matches = pd.DataFrame(columns=["match_id", "date", "home_team", "away_team"])
    result = availability.availability_features(empty, matches)
    assert result.empty
    assert "missing_share" in result.columns


def test_attach_to_fixtures_puts_both_sides_on_one_row():
    lineups, matches = _season(drop_player="P9", drop_from=12)
    features = availability.availability_features(lineups, matches)
    wide = availability.attach_to_fixtures(features)
    assert "missing_share_home" in wide.columns
    assert "missing_share_away" in wide.columns
    row = wide[wide["match_id"] == "m13"].iloc[0]
    assert row["missing_share_home"] == pytest.approx(1 / 11)
    assert row["missing_share_away"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Weighting by attacking involvement rather than minutes
# --------------------------------------------------------------------------

def test_xgchain_share_weights_by_involvement_not_minutes():
    """The reason the third feature exists.

    A full-back and a centre-forward play the same ninety minutes and losing
    them does not cost the same number of goals. Two squads identical in
    minutes must therefore give the same `missing_share` but different
    `missing_xgchain_share` when the player who drops out differs in how much
    of the attack ran through them.
    """
    creative, _ = _season(drop_player="P0", drop_from=12)   # the hub, chain 1.0
    ordinary, matches = _season(drop_player="P5", drop_from=12)  # a squad player

    hub = _for_team(availability.availability_features(creative, matches))
    hub = hub[hub["match_id"] == "m13"].iloc[0]
    other = _for_team(availability.availability_features(ordinary, matches))
    other = other[other["match_id"] == "m13"].iloc[0]

    # Identical by minutes, because both played every minute of every match.
    assert hub["missing_share"] == pytest.approx(other["missing_share"])
    # Very different by attacking involvement.
    assert hub["missing_xgchain_share"] > other["missing_xgchain_share"] * 5


def test_xgchain_share_is_a_share_of_the_teams_own_total():
    lineups, matches = _season(drop_player="P0", drop_from=12)
    table = _for_team(availability.availability_features(lineups, matches))
    row = table[table["match_id"] == "m13"].iloc[0]
    # P0 contributes 1.0 of a team total of 1.0 + 10 * 0.1 = 2.0.
    assert row["missing_xgchain_share"] == pytest.approx(0.5)


def test_xgchain_share_is_missing_when_the_source_has_no_chain_column():
    """Older line-up tables predate the column; that must not fabricate a zero."""
    lineups, matches = _season(drop_player="P0", drop_from=12)
    without = lineups.drop(columns=["xgchain"])
    table = _for_team(availability.availability_features(without, matches))
    row = table[table["match_id"] == "m13"].iloc[0]
    assert np.isnan(row["missing_xgchain_share"])
    assert row["missing_share"] > 0      # the minutes-based features still work


# --------------------------------------------------------------------------
# Feeding availability into the goals model
# --------------------------------------------------------------------------

def _training_with_availability(seed=0, n_teams=12, rounds=16, beta_own=-0.6,
                                beta_opp=0.3):
    """A league where absence really does change the scoring rate.

    Goals are drawn from the model's own form - attack, defence and a home
    advantage - multiplied by exp(beta * missing share). Recovering those two
    betas from the result is the only way to know the fitter works, since on
    real data there is no true value to compare against.
    """
    from fbedge.models import base as model_base

    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0.0, 0.2, n_teams)
    defence = rng.normal(0.0, 0.2, n_teams)
    intercept, hfa = np.log(1.35), 0.22

    rows = []
    day = dt.date(2022, 8, 1)
    for round_index in range(rounds):
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = int(order[i]), int(order[i + 1])
            miss_h = float(rng.choice([0.0, 0.09, 0.18, 0.27]))
            miss_a = float(rng.choice([0.0, 0.09, 0.18, 0.27]))
            lam = np.exp(intercept + hfa + attack[h] - defence[a]
                         + beta_own * miss_h + beta_opp * miss_a)
            mu = np.exp(intercept + attack[a] - defence[h]
                        + beta_own * miss_a + beta_opp * miss_h)
            rows.append({
                "match_id": f"m{len(rows)}",
                "date": day + dt.timedelta(days=round_index * 7),
                "home_team": teams[h], "away_team": teams[a],
                "home_goals": float(rng.poisson(lam)),
                "away_goals": float(rng.poisson(mu)),
                "home_xg": lam, "away_xg": mu,
                "missing_starter_share_home": miss_h,
                "missing_starter_share_away": miss_a,
                "referee": None,
            })
    frame = pd.DataFrame(rows)
    index = model_base.TeamIndex(teams)
    training = model_base.TrainingSet(
        league="TEST", as_of=day + dt.timedelta(days=rounds * 7 + 1), frame=frame,
        weights=np.ones(len(frame)), index=index,
        home_idx=index.encode(frame["home_team"]),
        away_idx=index.encode(frame["away_team"]),
    )
    return training


def test_the_fitter_recovers_a_planted_availability_effect():
    """The power test. A null on real data is only worth acting on if a real
    effect of that size would have been found here."""
    from fbedge.models import goals

    training = _training_with_availability(seed=5, rounds=40)
    model = goals.fit_goals_model(training, target="goals", use_availability=True)
    own, opp = model.availability_beta
    assert own < -0.2, f"planted -0.6, recovered {own:.3f}"
    assert opp > 0.05, f"planted +0.3, recovered {opp:.3f}"


def test_availability_is_off_unless_asked_for():
    from fbedge.models import goals

    training = _training_with_availability(seed=5)
    model = goals.fit_goals_model(training, target="goals")
    assert model.availability_beta == (0.0, 0.0)


def test_rates_ignore_availability_when_the_model_was_fitted_without_it():
    """Callers should not have to know whether the model uses availability."""
    from fbedge.models import goals

    training = _training_with_availability(seed=5)
    model = goals.fit_goals_model(training, target="goals")
    plain = model.rates("T00", "T01")
    with_shares = model.rates("T00", "T01", missing_home=0.5, missing_away=0.2)
    assert plain == pytest.approx(with_shares)


def test_rates_move_the_right_way_when_players_are_missing():
    from fbedge.models import goals

    training = _training_with_availability(seed=5, rounds=40)
    model = goals.fit_goals_model(training, target="goals", use_availability=True)
    base_home, base_away = model.rates("T00", "T01")
    hurt_home, hurt_away = model.rates("T00", "T01", missing_home=0.27)
    assert hurt_home < base_home     # the weakened side scores less
    assert hurt_away > base_away     # and concedes more


def test_the_fitter_declines_without_the_columns():
    from fbedge.models import goals

    training = _training_with_availability(seed=5)
    training.frame.drop(
        columns=["missing_starter_share_home", "missing_starter_share_away"],
        inplace=True,
    )
    model = goals.fit_goals_model(training, target="goals", use_availability=True)
    assert model.availability_beta == (0.0, 0.0)


# --------------------------------------------------------------------------
# Fixtures that have not been played yet
# --------------------------------------------------------------------------

def test_for_fixture_reads_the_same_windows_as_a_played_match():
    """An upcoming fixture must not get a different rule from a past one.

    The danger is two implementations of the windowing, only one of which
    carries the point-in-time guarantee. `for_fixture` therefore appends an
    empty placeholder and asks the same function, so this pins that the answer
    matches what the shared code would give.
    """
    lineups, matches = _season(drop_player="P9", drop_from=12)
    after_last = dt.date(2023, 1, 1) + dt.timedelta(days=7 * 14)
    home, away = availability.for_fixture(
        lineups, matches, "T", "Opponent", after_last
    )
    # P9 played nine of the ten matches in the importance window and missed
    # the most recent one, so the share is nine elevenths of one eleventh.
    assert home == pytest.approx(9 * 90 / (11 * 90 * 10))
    assert away == pytest.approx(0.0)


def test_for_fixture_returns_zero_for_a_team_with_no_history():
    lineups, matches = _season()
    home, away = availability.for_fixture(
        lineups, matches, "Nobody", "Opponent", dt.date(2024, 1, 1)
    )
    assert home == pytest.approx(0.0)


def test_for_fixture_ignores_matches_on_or_after_the_cutoff():
    """The same lookahead rule, at the boundary that matters for live use."""
    lineups, matches = _season(drop_player="P9", drop_from=12)
    # Asking as of m12 must not see m12 or anything later.
    early = availability.for_fixture(
        lineups, matches, "T", "Opponent", dt.date(2023, 1, 1) + dt.timedelta(days=7 * 12)
    )
    later = availability.for_fixture(
        lineups, matches, "T", "Opponent", dt.date(2023, 1, 1) + dt.timedelta(days=7 * 14)
    )
    assert early[0] == pytest.approx(0.0)
    assert later[0] > 0.0
