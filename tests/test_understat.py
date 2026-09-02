"""Tests for the Understat xG source and the xG-fitted goals model.

Two things here are worth more than the rest.

`test_payload_missing_keys_raises` and its neighbours pin the *shape* of the
endpoint. Understat moved its data out of the page HTML and into an AJAX call
at some point, which silently broke every published scraper - including the
`understat` PyPI package - because they look for `var datesData = JSON.parse(`
in HTML that no longer contains it and find nothing. Nothing raises; the result
is simply empty. A source that can fail that way has to fail loudly here
instead, and these tests are what force that.

`test_xg_fit_recovers_strengths_the_goals_fit_cannot` is the reason the feature
exists at all, expressed as a test: on data where scorelines are a noisy draw
around a true rate, strengths fitted to the underlying rate must track the
truth better than strengths fitted to the noisy realisation. If that ever stops
holding, the whole premise is wrong and no amount of tuning will rescue it.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import understat  # noqa: E402
from fbedge.models import base, goals  # noqa: E402


# --------------------------------------------------------------------------
# Parsing and the endpoint contract
# --------------------------------------------------------------------------

def _payload() -> dict:
    return {
        "dates": [
            {
                "id": "1",
                "isResult": True,
                "h": {"title": "Manchester City"},
                "a": {"title": "Nottingham Forest"},
                "goals": {"h": "3", "a": "0"},
                "xG": {"h": "2.5", "a": "0.4"},
                "datetime": "2023-08-11 19:00:00",
            },
            {
                "id": "2",
                "isResult": False,  # not played yet
                "h": {"title": "Arsenal"},
                "a": {"title": "Wolverhampton Wanderers"},
                "goals": {"h": None, "a": None},
                "xG": {"h": "0", "a": "0"},
                "datetime": "2026-12-01 15:00:00",
            },
        ],
        "teams": {
            "1": {
                "id": "1",
                "title": "Manchester City",
                "history": [
                    {"h_a": "h", "date": "2023-08-11 19:00:00", "xG": 2.5,
                     "npxG": 1.74, "xGA": 0.4, "npxGA": 0.4, "deep": 9, "scored": 3},
                ],
            }
        },
    }


def test_match_frame_translates_team_names():
    frame = understat.match_frame(_payload(), "E0", 2023)
    assert frame.loc[0, "home_team"] == "Man City"
    assert frame.loc[0, "away_team"] == "Nott'm Forest"
    # The original is kept so a mapping problem can be diagnosed later.
    assert frame.loc[0, "home_team_understat"] == "Manchester City"


def test_match_frame_drops_fixtures_not_yet_played():
    """An unplayed fixture carries xG of zero, not a missing value.

    Keeping it would feed a real 0.0 into the ratings as though a team had
    created nothing, which is the kind of error that never announces itself.
    """
    frame = understat.match_frame(_payload(), "E0", 2023)
    assert len(frame) == 1
    assert frame.loc[0, "understat_id"] == "1"


def test_match_frame_parses_the_numbers():
    frame = understat.match_frame(_payload(), "E0", 2023)
    assert frame.loc[0, "home_xg"] == pytest.approx(2.5)
    assert frame.loc[0, "home_goals"] == 3
    assert frame.loc[0, "date"] == dt.date(2023, 8, 11)


def test_npxg_frame_keeps_penalties_separate():
    frame = understat.npxg_frame(_payload(), "E0", 2023)
    assert frame.loc[0, "xg"] == pytest.approx(2.5)
    assert frame.loc[0, "npxg"] == pytest.approx(1.74)
    assert frame.loc[0, "team"] == "Man City"


def test_alias_table_is_injective():
    """Two Understat teams must never collapse onto one of ours.

    'Milan' and 'Inter' are the trap: Understat calls one of them 'AC Milan'
    and the other just 'Inter', so a careless rule maps both onto 'Milan' and
    silently merges two clubs' entire histories.
    """
    targets = list(understat.TEAM_ALIASES.values())
    assert len(targets) == len(set(targets))


def test_alias_table_does_not_rename_a_name_that_already_matches():
    # A no-op alias means somebody added an entry that was never needed and
    # may be shadowing a real name.
    for source, target in understat.TEAM_ALIASES.items():
        assert source != target


def test_unknown_team_passes_through_unchanged():
    assert understat.to_football_data_name("Some New Club") == "Some New Club"


def test_fetch_season_rejects_an_unknown_league():
    with pytest.raises(ValueError, match="Unknown league"):
        understat.fetch_season("XX", 2023, Path("."))


def test_fetch_season_reads_the_cache_without_network(tmp_path):
    (tmp_path / "E0_2023.json").write_text(json.dumps(_payload()), encoding="utf-8")
    # No network is available in the test environment; a cache hit must not
    # need one.
    payload = understat.fetch_season("E0", 2023, tmp_path)
    assert len(payload["dates"]) == 2


def test_season_range_starts_no_earlier_than_understat_has_data():
    assert min(understat.season_range(2010, 2020)) == understat.FIRST_SEASON


# --------------------------------------------------------------------------
# Fitting on xG
# --------------------------------------------------------------------------

def _synthetic_training(seed: int = 0, n_teams: int = 12, rounds: int = 12):
    """A league whose true strengths are known, with goals drawn around them.

    xG is recorded as the true rate plus a little measurement noise; goals are
    a Poisson draw from that rate. That is the situation the feature assumes:
    xG is a noisy view of the thing that matters, goals are a noisier one.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0.0, 0.25, n_teams)
    defence = rng.normal(0.0, 0.25, n_teams)
    intercept, hfa = np.log(1.35), 0.22

    rows = []
    day = dt.date(2022, 8, 1)
    for round_index in range(rounds):
        order = rng.permutation(n_teams)
        for i in range(0, n_teams, 2):
            h, a = int(order[i]), int(order[i + 1])
            lam = np.exp(intercept + hfa + attack[h] - defence[a])
            mu = np.exp(intercept + attack[a] - defence[h])
            rows.append(
                {
                    "match_id": f"m{len(rows)}",
                    "date": day + dt.timedelta(days=round_index * 7),
                    "home_team": teams[h], "away_team": teams[a],
                    "home_goals": float(rng.poisson(lam)),
                    "away_goals": float(rng.poisson(mu)),
                    "home_xg": float(lam * rng.lognormal(0.0, 0.15)),
                    "away_xg": float(mu * rng.lognormal(0.0, 0.15)),
                    "referee": None,
                }
            )
    frame = pd.DataFrame(rows)
    index = base.TeamIndex(teams)
    training = base.TrainingSet(
        league="TEST", as_of=day + dt.timedelta(days=rounds * 7 + 1), frame=frame,
        weights=np.ones(len(frame)), index=index,
        home_idx=index.encode(frame["home_team"]),
        away_idx=index.encode(frame["away_team"]),
    )
    return training, attack, defence


def test_responses_returns_goals_by_default():
    training, _, _ = _synthetic_training()
    x, y = goals.responses(training)
    assert x == pytest.approx(training.frame["home_goals"].to_numpy())


def test_responses_blend_sits_between_the_two():
    training, _, _ = _synthetic_training()
    goals_x, _ = goals.responses(training, "goals")
    xg_x, _ = goals.responses(training, "xg")
    blend_x, _ = goals.responses(training, "blend", blend_weight=0.5)
    assert blend_x == pytest.approx(0.5 * goals_x + 0.5 * xg_x)


def test_responses_rejects_an_unknown_target():
    training, _, _ = _synthetic_training()
    with pytest.raises(ValueError, match="Unknown target"):
        goals.responses(training, "vibes")


def test_asking_for_xg_without_xg_raises_rather_than_falling_back():
    """Silently fitting on goals when xG was requested is the worst outcome:
    the caller gets a model that is not the one it asked for and cannot tell."""
    training, _, _ = _synthetic_training()
    training.frame["home_xg"] = np.nan
    training.frame["away_xg"] = np.nan
    with pytest.raises(base.InsufficientData, match="build_xg"):
        goals.responses(training, "xg")


def test_xg_fit_recovers_strengths_the_goals_fit_cannot():
    """The premise of the whole feature, on data where the truth is known."""
    training, true_attack, true_defence = _synthetic_training(seed=7)
    truth = np.r_[true_attack - true_attack.mean(), true_defence - true_defence.mean()]

    def error(model):
        fitted = np.r_[
            model.attack - model.attack.mean(), model.defence - model.defence.mean()
        ]
        return float(np.sqrt(np.mean((fitted - truth) ** 2)))

    from_goals = goals.fit_goals_model(training, ridge=2.0, target="goals")
    from_xg = goals.fit_goals_model(training, ridge=2.0, target="xg")
    assert error(from_xg) < error(from_goals)


def test_xg_model_is_calibrated_to_goals_not_to_xg():
    """The level must come from goals even when the shape comes from xG.

    Here xG averages well above goals, so a model that skipped the second stage
    would over-predict every total. The two fits must land on nearly the same
    expected number of goals per match.
    """
    training, _, _ = _synthetic_training(seed=3)
    training.frame["home_xg"] = training.frame["home_xg"] * 1.4
    training.frame["away_xg"] = training.frame["away_xg"] * 1.4

    from_goals = goals.fit_goals_model(training, target="goals")
    from_xg = goals.fit_goals_model(training, target="xg")

    def mean_total(model):
        return float(
            np.mean([
                sum(model.rates(row.home_team, row.away_team))
                for row in training.frame.itertuples()
            ])
        )

    assert mean_total(from_xg) == pytest.approx(mean_total(from_goals), rel=0.05)


def test_target_is_recorded_on_the_model():
    training, _, _ = _synthetic_training()
    assert goals.fit_goals_model(training, target="xg").target == "xg"
    assert goals.fit_goals_model(training, target="goals").target == "goals"
    # Unspecified means the shipping default, which is the blend.
    assert goals.fit_goals_model(training).target == base.DEFAULT_TARGET


def test_an_unspecified_target_falls_back_when_there_is_no_xg():
    """A default should degrade rather than fail on an older database."""
    training, _, _ = _synthetic_training()
    training.frame["home_xg"] = np.nan
    training.frame["away_xg"] = np.nan
    assert goals.fit_goals_model(training).target == "goals"


def test_an_explicit_target_still_refuses_when_there_is_no_xg():
    """The fallback must not extend to a caller who asked for xG by name."""
    training, _, _ = _synthetic_training()
    training.frame["home_xg"] = np.nan
    training.frame["away_xg"] = np.nan
    with pytest.raises(base.MissingExpectedGoals, match="build_xg"):
        goals.fit_goals_model(training, target="blend")


def test_ridge_defaults_to_something_that_suits_the_target():
    """One shrinkage value cannot serve targets with different noise levels."""
    training, _, _ = _synthetic_training()
    assert goals.fit_goals_model(training, target="goals").ridge == base.default_ridge("goals")
    assert goals.fit_goals_model(training, target="blend").ridge == base.default_ridge("blend")
    assert base.default_ridge("blend") < base.default_ridge("goals")
    # An explicit value always wins over the recommendation.
    assert goals.fit_goals_model(training, ridge=7.5, target="blend").ridge == 7.5


def test_recalibration_leaves_team_strengths_untouched():
    training, _, _ = _synthetic_training()
    model = goals.fit_goals_model(training, target="xg")
    intercept, hfa, rho = goals.recalibrate_on_goals(
        training, model.attack, model.defence
    )
    # Calling it again must be idempotent: it does not move the strengths, so
    # the parameters it returns should not drift either.
    again = goals.recalibrate_on_goals(training, model.attack, model.defence)
    assert (intercept, hfa, rho) == pytest.approx(again)
