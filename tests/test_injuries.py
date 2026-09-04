"""Tests for the external injury feed.

The ones that matter most are about **names**, not about injuries. This project
already paid once for a name-matching mistake: 41 of 156 Understat club names
differed from football-data's, the differences were not systematic, and a fuzzy
matcher would have merged Milan with Inter without a word. So the guarantee
worth pinning is negative - `to_football_data_name` must return **None** rather
than a best guess, and a caller must be able to see what it refused.

The second theme is that a broken feed must not look like a quiet week. An
empty injury list and a dead endpoint render identically on a page, so the
fetch raises on anything unexpected instead of returning nothing.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from fbedge import config, injuries


def response(entries: list[dict]) -> dict:
    return {
        "get": "injuries",
        "parameters": {"league": "39", "season": "2026"},
        "errors": [],
        "results": len(entries),
        "paging": {"current": 1, "total": 1},
        "response": entries,
    }


def entry(
    player: str,
    team: str,
    kind: str = "Missing Fixture",
    reason: str = "Hamstring Injury",
    fixture_id: int = 1,
    date: str = "2026-09-05T14:00:00+00:00",
) -> dict:
    return {
        "player": {"id": 1, "name": player, "type": kind, "reason": reason},
        "team": {"id": 33, "name": team},
        "fixture": {"id": fixture_id, "date": date, "timezone": "UTC"},
        "league": {"id": 39, "season": 2026, "name": "Premier League"},
    }


# ----------------------------------------------------------------------
# Names: the part that must never guess
# ----------------------------------------------------------------------


def test_an_unknown_club_returns_none_rather_than_a_guess():
    """The whole safety property. A wrong club is worse than a missing one:
    it attaches somebody's injury to a team they do not play for."""
    assert injuries.to_football_data_name("Some FC Nobody Knows", {"Arsenal"}) is None


def test_similar_but_different_clubs_are_not_merged():
    """The Understat lesson, restated. `Milan` and `Inter` are both real and
    both short; a similarity score would happily confuse these."""
    known = {"Milan", "Inter"}
    assert injuries.to_football_data_name("AC Milan", known) == "Milan"
    assert injuries.to_football_data_name("Internazionale", known) == "Inter"
    # Nothing maps one onto the other.
    assert injuries.to_football_data_name("AC Milan", known) != "Inter"


def test_an_exact_name_passes_straight_through():
    assert injuries.to_football_data_name("Arsenal", {"Arsenal"}) == "Arsenal"


def test_the_alias_table_handles_the_documented_differences():
    known = {"Man United", "Wolves", "Nott'm Forest", "Ath Madrid", "Paris SG"}
    assert injuries.to_football_data_name("Manchester United", known) == "Man United"
    assert injuries.to_football_data_name(
        "Wolverhampton Wanderers", known
    ) == "Wolves"
    assert injuries.to_football_data_name(
        "Nottingham Forest", known
    ) == "Nott'm Forest"
    assert injuries.to_football_data_name("Atletico Madrid", known) == "Ath Madrid"
    assert injuries.to_football_data_name(
        "Paris Saint Germain", known
    ) == "Paris SG"


def test_normalisation_matches_across_accents_and_corporate_noise():
    """`FC Augsburg` and `Augsburg` are one club; `Köln` and `Koln` are one club.

    This is the only inexact step, and it is exact *after* a documented
    transformation rather than a scored comparison.
    """
    assert injuries.to_football_data_name("FC Augsburg", {"Augsburg"}) == "Augsburg"
    assert injuries.normalise("Köln") == injuries.normalise("Koln")
    assert injuries.normalise("Bayern München") == "bayern munchen"


def test_normalisation_only_strips_noise_from_the_ends():
    """A club genuinely called Milan must survive the rule that drops `AC`."""
    assert injuries.normalise("Milan") == "milan"
    assert injuries.normalise("AC Milan") == "milan"


def test_the_alias_table_is_injective():
    """Two feed names mapping to one club is fine; it is the reverse that would
    be a bug. Guards against an edit that quietly points an alias at the wrong
    club by reusing a target that was meant for another."""
    for source, target in injuries.TEAM_ALIASES.items():
        assert isinstance(target, str) and target, f"{source} maps to nothing"
        assert injuries.normalise(source) == source, (
            f"alias key {source!r} is not in normalised form, so it can never "
            "be hit"
        )


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def test_unmatched_clubs_are_reported_not_silently_dropped():
    """A club that vanishes looks exactly like a club with nobody injured."""
    frame, unmatched = injuries.injury_frame(
        response([
            entry("A Player", "Arsenal"),
            entry("B Player", "Club Nobody Knows"),
        ]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    assert len(frame) == 1
    assert unmatched == ["Club Nobody Knows"]


def test_out_and_doubtful_are_kept_apart():
    """"Ruled out" and "doubtful" are different claims and collapsing them to a
    boolean would be inventing certainty the feed did not offer."""
    frame, _ = injuries.injury_frame(
        response([
            entry("Out Player", "Arsenal", kind="Missing Fixture"),
            entry("Maybe Player", "Arsenal", kind="Questionable", fixture_id=2),
        ]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    out = frame[frame["player"] == "Out Player"].iloc[0]
    maybe = frame[frame["player"] == "Maybe Player"].iloc[0]
    assert out["ruled_out"] and not out["doubtful"]
    assert maybe["doubtful"] and not maybe["ruled_out"]


def test_an_unrecognised_status_is_kept_rather_than_discarded():
    """A word this project has not seen is news about the feed, not a reason to
    drop a player. Neither flag is set, and the page shows the raw status."""
    frame, _ = injuries.injury_frame(
        response([entry("X", "Arsenal", kind="Suspended")]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    row = frame.iloc[0]
    assert not row["ruled_out"] and not row["doubtful"]
    assert row["status"] == "Suspended"


def test_the_reason_is_carried_through():
    frame, _ = injuries.injury_frame(
        response([entry("X", "Arsenal", reason="Broken ankle")]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    assert frame.iloc[0]["reason"] == "Broken ankle"


def test_a_player_listed_against_several_fixtures_appears_once_per_date():
    frame, _ = injuries.injury_frame(
        response([
            entry("X", "Arsenal", fixture_id=1, date="2026-09-05T14:00:00+00:00"),
            entry("X", "Arsenal", fixture_id=1, date="2026-09-05T14:00:00+00:00"),
        ]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    assert len(frame) == 1


def test_an_empty_response_parses_to_an_empty_frame():
    frame, unmatched = injuries.injury_frame(response([]), "E0", 2026, {"Arsenal"})
    assert frame.empty
    assert unmatched == []


def test_every_row_carries_when_it_was_read():
    """Freshness travels with the data. A stale injury list is worse than none
    because it looks current, so the page can always say how old it is."""
    stamp = dt.datetime(2026, 9, 4, 9, 0)
    frame, _ = injuries.injury_frame(
        response([entry("X", "Arsenal")]), "E0", 2026, {"Arsenal"}, fetched_at=stamp
    )
    assert frame.iloc[0]["fetched_at"] == stamp
    assert injuries.freshness(frame) == stamp


def test_freshness_of_nothing_is_none():
    assert injuries.freshness(pd.DataFrame()) is None


# ----------------------------------------------------------------------
# Ordering for display
# ----------------------------------------------------------------------


def test_for_teams_puts_the_worse_news_first():
    frame, _ = injuries.injury_frame(
        response([
            entry("Maybe", "Arsenal", kind="Questionable", fixture_id=1),
            entry("Definitely", "Arsenal", kind="Missing Fixture", fixture_id=2),
        ]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    ordered = injuries.for_teams(frame, ["Arsenal"])
    assert list(ordered["player"]) == ["Definitely", "Maybe"]


def test_for_teams_filters_to_the_teams_asked_for():
    frame, _ = injuries.injury_frame(
        response([entry("A", "Arsenal"), entry("B", "Chelsea", fixture_id=2)]),
        "E0", 2026, known_teams={"Arsenal", "Chelsea"},
    )
    assert set(injuries.for_teams(frame, ["Arsenal"])["team"]) == {"Arsenal"}


# ----------------------------------------------------------------------
# Fetching: a broken feed must not look like a quiet week
# ----------------------------------------------------------------------


def test_no_key_is_reported_as_its_own_condition(tmp_path, monkeypatch):
    """Absence of a key is not a failure, and the page treats the two
    differently: one is a setup step, the other is something being wrong."""
    monkeypatch.delenv(config.INJURY_API_KEY_ENV, raising=False)
    assert injuries.api_key() is None
    with pytest.raises(injuries.MissingApiKey):
        injuries.fetch_league("E0", 2026, tmp_path)


def test_an_unknown_league_is_refused_before_a_request_is_spent(tmp_path):
    with pytest.raises(injuries.InjuryFeedError, match="league id"):
        injuries.fetch_league("XX", 2026, tmp_path)


def test_a_fresh_cache_is_used_without_a_key(tmp_path):
    """Caching is the request budget, not politeness: 100 a day, and the page
    re-renders on every click."""
    path = tmp_path / "E0_2026.json"
    path.write_text(json.dumps(response([entry("X", "Arsenal")])), encoding="utf-8")
    payload = injuries.fetch_league("E0", 2026, tmp_path, max_age_hours=6.0)
    assert payload["results"] == 1


def test_a_stale_cache_is_not_used(tmp_path, monkeypatch):
    monkeypatch.delenv(config.INJURY_API_KEY_ENV, raising=False)
    path = tmp_path / "E0_2026.json"
    path.write_text(json.dumps(response([])), encoding="utf-8")
    with pytest.raises(injuries.MissingApiKey):
        injuries.fetch_league(
            "E0", 2026, tmp_path, max_age_hours=1.0,
            now=dt.datetime.now() + dt.timedelta(hours=3),
        )


def test_every_configured_league_has_a_feed_id():
    """A league in the project with no feed id would return nothing for ever,
    and look like a league where nobody is ever injured."""
    for league in config.LEAGUES:
        assert league in config.INJURY_LEAGUE_IDS


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    frame, _ = injuries.injury_frame(
        response([entry("X", "Arsenal")]), "E0", 2026, {"Arsenal"}
    )
    assert injuries.write_injuries(con, frame) == 1
    assert len(injuries.load_injuries(con)) == 1
    assert len(injuries.load_injuries(con, ["Arsenal"])) == 1
    assert len(injuries.load_injuries(con, ["Chelsea"])) == 0


def test_a_recovered_player_disappears_on_refresh(tmp_path):
    """An injury ends by the player being *absent* from the next payload, not
    by being marked fit in it. Anything additive would keep him on the page
    for ever."""
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    first, _ = injuries.injury_frame(
        response([entry("X", "Arsenal"), entry("Y", "Arsenal", fixture_id=2)]),
        "E0", 2026, {"Arsenal"},
    )
    injuries.write_injuries(con, first)
    second, _ = injuries.injury_frame(
        response([entry("X", "Arsenal")]), "E0", 2026, {"Arsenal"}
    )
    injuries.write_injuries(con, second)

    back = injuries.load_injuries(con)
    assert list(back["player"]) == ["X"]


def test_refreshing_one_league_leaves_another_alone(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    england, _ = injuries.injury_frame(
        response([entry("X", "Arsenal")]), "E0", 2026, {"Arsenal"}
    )
    italy, _ = injuries.injury_frame(
        response([entry("Y", "Inter")]), "I1", 2026, {"Inter"}
    )
    injuries.write_injuries(con, england)
    injuries.write_injuries(con, italy)
    injuries.write_injuries(con, england)

    back = injuries.load_injuries(con)
    assert set(back["league"]) == {"E0", "I1"}
    assert len(back) == 2


def test_writing_nothing_still_creates_the_table(tmp_path):
    """The page reads this table before anything has ever been fetched."""
    import duckdb

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    assert injuries.write_injuries(con, pd.DataFrame()) == 0
    assert injuries.load_injuries(con).empty


# ----------------------------------------------------------------------
# The schema, as confirmed against a live response
# ----------------------------------------------------------------------

# Copied verbatim from `build_injuries.py --probe --league E0 --season 2024`.
# Every other test in this file builds its own payload, which proves the parser
# is self-consistent and nothing more; this one proves it agrees with the
# provider. Do not tidy it - its value is that it was not written by hand.
LIVE_ENTRY = {
    "player": {
        "id": 153434,
        "name": "W. Fish",
        "photo": "https://media.api-sports.io/football/players/153434.png",
        "type": "Missing Fixture",
        "reason": "Ankle Injury",
    },
    "team": {
        "id": 33,
        "name": "Manchester United",
        "logo": "https://media.api-sports.io/football/teams/33.png",
    },
    "fixture": {
        "id": 1208021,
        "timezone": "UTC",
        "date": "2024-08-16T19:00:00+00:00",
        "timestamp": 1723834800,
    },
    "league": {
        "id": 39,
        "season": 2024,
        "name": "Premier League",
        "country": "England",
        "logo": "https://media.api-sports.io/football/leagues/39.png",
        "flag": "https://media.api-sports.io/flags/gb-eng.svg",
    },
}


def test_the_parser_agrees_with_a_real_response():
    """The one test here that could catch the provider changing under us."""
    frame, unmatched = injuries.injury_frame(
        response([LIVE_ENTRY]), "E0", 2024, known_teams={"Man United"}
    )
    assert unmatched == []
    row = frame.iloc[0]
    assert row["player"] == "W. Fish"
    assert row["team"] == "Man United"       # feed says "Manchester United"
    assert row["team_feed"] == "Manchester United"
    assert row["status"] == "Missing Fixture"
    assert row["reason"] == "Ankle Injury"
    assert row["ruled_out"] and not row["doubtful"]
    assert row["fixture_id"] == 1208021
    assert row["fixture_date"] == dt.date(2024, 8, 16)


def test_a_timezone_aware_fixture_date_lands_on_the_right_day():
    """The feed sends `2024-08-16T19:00:00+00:00`. Losing the offset would move
    late kick-offs onto the wrong day of the calendar."""
    frame, _ = injuries.injury_frame(
        response([LIVE_ENTRY]), "E0", 2024, {"Man United"}
    )
    assert frame.iloc[0]["fixture_date"] == dt.date(2024, 8, 16)


# ----------------------------------------------------------------------
# The two defects the live probe exposed
# ----------------------------------------------------------------------


def test_injuries_are_filtered_to_the_day_being_looked_at():
    """A league-season is thousands of rows - 3,168 for the 2024 Premier League.

    Without a date filter every club's panel would list its entire season of
    absences at once and read as a permanent injury crisis.
    """
    frame, _ = injuries.injury_frame(
        response([
            entry("August Player", "Arsenal", fixture_id=1,
                  date="2026-09-05T14:00:00+00:00"),
            entry("Later Player", "Arsenal", fixture_id=2,
                  date="2026-12-20T14:00:00+00:00"),
        ]),
        "E0", 2026, known_teams={"Arsenal"},
    )
    on_day = injuries.for_teams(frame, ["Arsenal"], on_date=dt.date(2026, 9, 5))
    assert list(on_day["player"]) == ["August Player"]

    # And without the filter, both come back - which is the bug, pinned.
    assert len(injuries.for_teams(frame, ["Arsenal"])) == 2


def test_a_date_with_no_entries_returns_nothing_rather_than_everything():
    frame, _ = injuries.injury_frame(
        response([entry("X", "Arsenal", date="2026-09-05T14:00:00+00:00")]),
        "E0", 2026, {"Arsenal"},
    )
    assert injuries.for_teams(
        frame, ["Arsenal"], on_date=dt.date(2026, 9, 6)
    ).empty


def test_a_paged_response_is_refused_rather_than_half_read(tmp_path, monkeypatch):
    """The live response fits on one page today - 3,168 rows, paging total 1.

    If that ever changes, reading only page one would drop most of a league
    without a word, which is the exact shape of failure this project keeps
    getting caught by.
    """
    payload = response([entry("X", "Arsenal")])
    payload["paging"] = {"current": 1, "total": 4}

    class FakeResponse:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setenv(config.INJURY_API_KEY_ENV, "test-key")
    monkeypatch.setattr(
        injuries.urllib.request, "urlopen", lambda *a, **k: FakeResponse()
    )
    monkeypatch.setattr(injuries.time, "sleep", lambda *a: None)
    with pytest.raises(injuries.InjuryFeedError, match="pages"):
        injuries.fetch_league("E0", 2026, tmp_path, force=True)
