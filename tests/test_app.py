"""Smoke tests for the Streamlit app.

The app had no coverage at all until the xG selector was added to it, which
made the gap uncomfortable: it is the only part of the project a person
actually looks at, and a broken sidebar is invisible to every other test here.

These are deliberately shallow. They do not assert anything about the numbers -
that is what the model and evaluation tests are for - only that each branch of
the app runs to completion without raising. That is enough to catch the failure
that actually happens in practice, which is a signature changing underneath a
call site and nobody noticing until the page is opened.

**They skip without a built database.** The app reads the real thing rather
than a fixture, so on a fresh clone these cannot run, and skipping is right:
failing would tell a newcomer their checkout is broken when it is merely empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"

pytest.importorskip("streamlit.testing.v1", reason="streamlit too old for AppTest")

pytestmark = pytest.mark.skipif(
    not config.DB_PATH.exists(),
    reason="no database; run scripts/build_database.py",
)


def _app():
    """The app as a visitor first sees it: the calendar home page."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=600)


def _match_app():
    """The app already opened on a fixture.

    The home page is the landing view now, so every test that wants the model
    controls has to say so. Seeding `view` rather than clicking a Details
    button keeps these tests independent of which fixtures happen to be on
    today - a test that only works on a matchday is worse than no test.
    """
    at = _app()
    at.session_state["view"] = "match"
    return at


def test_the_app_runs_at_all():
    at = _app()
    at.run()
    assert not at.exception


def test_the_rate_teams_on_selector_offers_every_target():
    at = _match_app()
    at.run()
    radios = [r for r in at.sidebar.radio if r.label == "Rate teams on"]
    assert radios, "the target selector is missing from the sidebar"
    assert len(radios[0].options) == 3


def test_shrinkage_default_follows_the_target():
    """Switching target must move the shrinkage slider with it.

    These two settings interact: a blend fitted at the goals-appropriate
    shrinkage of 5 is the worst of both worlds - the better signal, shrunk
    until it cannot show - and that combination was measurably worse than
    either sensible pairing. A user who changes the target and does not think
    to also change a slider three lines below should not land there.
    """
    from fbedge.models import base as model_base

    def offered_shrinkage(choice: str) -> str:
        # A fresh app per choice: selecting "blend" reveals an extra slider, so
        # switching back and forth inside one session changes which widgets
        # exist and confuses Streamlit's state bookkeeping. That is a quirk of
        # the test harness rather than of the app.
        at = _match_app()
        at.run()
        radio = [r for r in at.sidebar.radio if r.label == "Rate teams on"][0]
        radio.set_value(choice).run()
        assert not at.exception
        captions = [c.value for c in at.sidebar.caption if "Shrinkage" in c.value]
        assert captions, "the shrinkage caption is missing"
        return captions[0]

    assert f"Shrinkage {model_base.default_ridge('goals'):g}" in offered_shrinkage("Goals")
    assert f"Shrinkage {model_base.default_ridge('blend'):g}" in offered_shrinkage(
        "Goals + xG blend"
    )
    assert model_base.default_ridge("blend") < model_base.default_ridge("goals")


@pytest.mark.parametrize("choice", ["Expected goals", "Goals + xG blend"])
def test_each_xg_target_renders(choice):
    """The paths that only exist because of the xG work.

    Worth having both: 'blend' additionally reveals a weight slider, so it
    exercises a branch that 'Expected goals' does not.
    """
    if not database.has_xg(database.connect(config.DB_PATH, read_only=True)):
        pytest.skip("no match_xg table; run scripts/build_xg.py")

    at = _match_app()
    at.run()
    radio = [r for r in at.sidebar.radio if r.label == "Rate teams on"][0]
    radio.set_value(choice).run()
    assert not at.exception


# ----------------------------------------------------------------------
# The calendar home page
# ----------------------------------------------------------------------


def test_the_home_page_is_the_landing_view():
    """A visitor lands on the calendar, not on a fixture form.

    Pins the routing decision itself. Everything else about the home page can
    change; that it is what opens should not, silently.
    """
    at = _app()
    at.run()
    assert not at.exception
    # AppTest's session_state proxy has no .get, so ask before reading.
    assert "view" not in at.session_state or at.session_state["view"] == "home"


def test_the_home_page_offers_a_way_into_a_fixture():
    """There is a route from the calendar to the detail view.

    Skips rather than fails on a day with no football: international breaks
    are real and a test that only passes on a matchday would be flaky by
    design.
    """
    from fbedge import fixtures as fixtures_mod

    con = database.connect(config.DB_PATH, read_only=True)
    try:
        calendar = fixtures_mod.load_calendar(con, season=fixtures_mod.current_season())
    except Exception:
        pytest.skip("no fixtures table; run scripts/build_fixtures.py")
    if calendar.empty:
        pytest.skip("fixtures table is empty")

    import datetime as dt

    day = fixtures_mod.nearest_date_with_matches(calendar, dt.date.today())
    at = _app()
    at.session_state["calendar_date"] = day
    at.run()
    assert not at.exception
    assert any(b.label == "Details" for b in at.button), (
        "the calendar showed fixtures but no way to open one"
    )


def test_clicking_into_a_fixture_and_back_out_again():
    """The round trip, which is the one interaction that spans both views."""
    at = _match_app()
    at.run()
    assert not at.exception
    back = [b for b in at.sidebar.button if "Back to calendar" in b.label]
    assert back, "the detail view has no way back to the calendar"
    back[0].click().run()
    assert not at.exception
    assert at.session_state["view"] == "home"
