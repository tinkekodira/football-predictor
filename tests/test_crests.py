"""Tests for club badges.

The property worth pinning is that a **missing** badge is invisible rather than
broken. Promotion happens every summer, so a club with no badge on disk is a
normal state, not a fault, and the page has to look deliberate when it happens
instead of showing a torn-image icon.

The slug tests exist because football-data club names contain apostrophes -
`Nott'm Forest`, `M'gladbach` - which are legal in a filename on Linux and a
nuisance on Windows, where this project runs.
"""

from __future__ import annotations

import base64

import pytest

from fbedge import crests


@pytest.fixture
def badge_dir(tmp_path):
    """A crest directory holding one real, tiny PNG."""
    directory = tmp_path / crests.CREST_DIRNAME
    directory.mkdir(parents=True)
    # A 1x1 transparent PNG, small enough to inline in a test.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUA"
        "AeImBZsAAAAASUVORK5CYII="
    )
    (directory / "liverpool.png").write_bytes(png)
    crests.data_uri.cache_clear()
    yield tmp_path
    crests.data_uri.cache_clear()


# ----------------------------------------------------------------------
# Filenames
# ----------------------------------------------------------------------


def test_slug_survives_apostrophes_and_punctuation():
    """These are real club names in this project's own convention."""
    assert crests.slug("Nott'm Forest") == "nott-m-forest"
    assert crests.slug("M'gladbach") == "m-gladbach"
    assert crests.slug("Man United") == "man-united"
    assert crests.slug("Liverpool") == "liverpool"


def test_slug_is_stable_across_case_and_spacing():
    assert crests.slug("  Real   Madrid ") == crests.slug("real madrid")


def test_slug_never_returns_an_empty_filename():
    """An empty stem would collide with every other empty stem and silently
    give two clubs the same badge."""
    assert crests.slug("!!!") == "unknown"
    assert crests.slug("") == "unknown"


# ----------------------------------------------------------------------
# A missing badge is a normal state
# ----------------------------------------------------------------------


def test_a_club_with_no_badge_renders_as_nothing(badge_dir):
    """Not a placeholder and not a broken image: a row with a missing badge
    should read as a row with a slightly shorter name."""
    assert crests.img_tag("Elversberg", base=str(badge_dir)) == ""
    assert crests.data_uri("Elversberg", str(badge_dir)) is None
    assert crests.path_for("Elversberg", badge_dir) is None


def test_a_club_with_no_badge_still_shows_its_name(badge_dir):
    rendered = crests.labelled("Elversberg", base=str(badge_dir))
    assert "Elversberg" in rendered
    assert "<img" not in rendered


def test_a_club_with_a_badge_gets_an_inline_image(badge_dir):
    tag = crests.img_tag("Liverpool", base=str(badge_dir))
    assert tag.startswith('<img src="data:image/png;base64,')
    assert 'width="20"' in tag


def test_the_badge_is_inlined_rather_than_linked(badge_dir):
    """A hotlinked URL would make every page render depend on somebody else's
    uptime and leak a request per badge per viewer."""
    uri = crests.data_uri("Liverpool", str(badge_dir))
    assert uri.startswith("data:image/png;base64,")
    assert "http" not in uri


def test_the_requested_size_is_honoured(badge_dir):
    assert 'width="34"' in crests.img_tag("Liverpool", 34, base=str(badge_dir))


# ----------------------------------------------------------------------
# Coverage reporting
# ----------------------------------------------------------------------


def test_missing_lists_only_the_clubs_without_a_badge(badge_dir):
    absent = crests.missing(["Liverpool", "Elversberg", "Le Mans"], badge_dir)
    assert absent == ["Elversberg", "Le Mans"]


def test_available_reads_the_directory(badge_dir):
    assert crests.available(badge_dir) == {"liverpool"}


def test_available_on_a_directory_that_does_not_exist(tmp_path):
    """The page asks before anything has ever been downloaded."""
    assert crests.available(tmp_path / "nope") == set()
    assert crests.missing(["Liverpool"], tmp_path / "nope") == ["Liverpool"]


# ----------------------------------------------------------------------
# The URL template
# ----------------------------------------------------------------------


def test_the_crest_url_needs_no_api_key():
    """The badge CDN is public - confirmed by fetching one with no key at all.

    That is the whole reason badges work on a free plan while the injury data
    behind the same provider does not.
    """
    url = crests.CREST_URL.format(team_id=40)
    assert url == "https://media.api-sports.io/football/teams/40.png"
    assert "key" not in url
