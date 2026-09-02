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
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=600)


def test_the_app_runs_at_all():
    at = _app()
    at.run()
    assert not at.exception


def test_the_rate_teams_on_selector_offers_every_target():
    at = _app()
    at.run()
    radios = [r for r in at.sidebar.radio if r.label == "Rate teams on"]
    assert radios, "the target selector is missing from the sidebar"
    assert len(radios[0].options) == 3


@pytest.mark.parametrize("choice", ["Expected goals", "Goals + xG blend"])
def test_each_xg_target_renders(choice):
    """The paths that only exist because of the xG work.

    Worth having both: 'blend' additionally reveals a weight slider, so it
    exercises a branch that 'Expected goals' does not.
    """
    if not database.has_xg(database.connect(config.DB_PATH, read_only=True)):
        pytest.skip("no match_xg table; run scripts/build_xg.py")

    at = _app()
    at.run()
    radio = [r for r in at.sidebar.radio if r.label == "Rate teams on"][0]
    radio.set_value(choice).run()
    assert not at.exception
