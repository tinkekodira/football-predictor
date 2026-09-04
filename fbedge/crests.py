"""Club badges, stored locally and served inline.

**Why local files rather than hotlinked URLs.** The badges come from
API-Football's image CDN, which is public - fetching one sends no key, which is
how this works at all on a free plan. Pointing the page straight at that CDN
would still be wrong: it makes every page render depend on somebody else's
uptime, leaks a request per badge per viewer, and breaks entirely offline. They
are downloaded once instead.

**Why they are resized on the way in.** The originals are around 90KB each. A
Saturday shows 22 fixtures, so 44 badges, which is four megabytes of PNG on one
page. At 48 pixels they are two or three kilobytes, which is small enough to
inline as data URIs and skip the second request entirely.

**Why inline data URIs rather than `st.image`.** A badge belongs *beside* a
team name on one line. Streamlit's image element is a block, so laying a row
out with it means a column per element and a row that wraps unpredictably at
narrow widths. One HTML string with the image inline is both simpler and what
actually looks right.

**A club with no badge gets a generated monogram**, not a gap. That reverses
an earlier decision in this file, and the reversal is explained on `monogram`:
rendering nothing is right when a handful are missing and wrong when most are,
which is where a free API key leaves you. The stand-in needs no network and no
licence, cannot be mistaken for a real crest, and is replaced the moment
`build_crests.py` downloads the genuine one. Promotion means missing badges are
a permanent normal state, not a transient one.
"""

from __future__ import annotations

import base64
import hashlib
import functools
import re
from pathlib import Path

from . import config

CREST_DIRNAME = "crests"

# Big enough to read at a glance on a dense fixture list, small enough that
# forty of them inline without bloating the page.
CREST_PIXELS = 48

# The public image CDN behind API-Football. Requesting one of these sends no
# key, which is why badges work on a free plan while the injury data does not.
CREST_URL = "https://media.api-sports.io/football/teams/{team_id}.png"


def crest_dir(base: Path | None = None) -> Path:
    """Where badges live. Under `data/`, alongside the other downloaded input."""
    return Path(base or config.DATA_DIR) / CREST_DIRNAME


def slug(team: str) -> str:
    """A filename for a club name.

    Football-data names contain apostrophes and full stops - `Nott'm Forest`,
    `M'gladbach` - which are legal in a filename on one platform and awkward on
    another. Lowercased, non-alphanumerics collapsed to a hyphen.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(team).lower()).strip("-")
    return cleaned or "unknown"


def path_for(team: str, base: Path | None = None) -> Path | None:
    """The badge file for a club, or None when there isn't one."""
    candidate = crest_dir(base) / f"{slug(team)}.png"
    return candidate if candidate.exists() else None


@functools.lru_cache(maxsize=512)
def data_uri(team: str, base: str | None = None) -> str | None:
    """A badge as an inline `data:` URI, or None.

    Cached because a fixture list asks for the same handful of clubs on every
    rerun and the encoding is pure work. The cache is keyed on a string rather
    than a Path so that it hashes.
    """
    found = path_for(team, Path(base) if base else None)
    if found is None:
        return None
    encoded = base64.b64encode(found.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# A monogram's colour is picked from this rather than from a hashed hue, so
# that every one is legible against white text. Deliberately muted: a
# placeholder that shouts is worse than one that sits quietly beside a real
# badge.
MONOGRAM_COLOURS = (
    "#3d5a80", "#6b4e71", "#2f6690", "#5f7161", "#8c5e58",
    "#41618a", "#6d597a", "#3f6f6f", "#7a5c3e", "#4a5859",
)


def initials(team: str) -> str:
    """One or two letters standing for a club.

    Two words give two initials; one word gives its first two letters, which
    reads better than a lone capital at 20 pixels.
    """
    # Apostrophes are removed rather than treated as separators, so that
    # "Nott'm Forest" gives NF and "M'gladbach" gives MG. Splitting on them
    # instead yields a one-letter fragment that wins the second slot.
    cleaned = str(team).replace("'", "").replace("’", "")
    words = [w for w in re.split(r"[^A-Za-z0-9]+", cleaned) if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def monogram(team: str, size: int = 20) -> str:
    """A generated stand-in badge, as an inline SVG.

    **This is a deliberate reversal of an earlier decision**, which was that a
    club with no badge should render as nothing at all so that a gap looked
    like a shorter name rather than a broken image. That was right when a
    handful were missing. It is wrong at eighty of ninety-six, where the page
    reads as half-built instead of deliberate.

    A flat disc with initials cannot be mistaken for a club's real crest, needs
    no network and no licence, and is replaced automatically the moment
    `build_crests.py` downloads the real thing. Colour is chosen by a stable
    digest of the name - `hash()` is salted per process and would give a club a
    different colour on every restart.
    """
    digest = hashlib.md5(str(team).encode("utf-8")).hexdigest()
    colour = MONOGRAM_COLOURS[int(digest[:8], 16) % len(MONOGRAM_COLOURS)]
    text = initials(team)
    font = size * (0.42 if len(text) > 1 else 0.5)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
        f'<circle cx="20" cy="20" r="20" fill="{colour}"/>'
        f'<text x="20" y="20" fill="#ffffff" font-size="{40 * font / size:.0f}" '
        f'font-family="Helvetica,Arial,sans-serif" font-weight="600" '
        f'text-anchor="middle" dominant-baseline="central">{text}</text></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def img_tag(
    team: str, size: int = 20, base: str | None = None, fallback: bool = True
) -> str:
    """An `<img>` for a club badge, falling back to a generated monogram.

    `fallback=False` restores the older behaviour of rendering nothing, which
    is still what you want somewhere a stand-in would be misleading.
    `vertical-align: middle` keeps the badge on the text baseline instead of
    sitting the line height on top of it.
    """
    uri = data_uri(team, base)
    # Rounding belongs to the generated disc alone: a real badge is square with
    # transparent corners, and a 50% radius would clip it.
    radius = ""
    if uri is None:
        if not fallback:
            return ""
        uri = monogram(team, size)
        radius = ";border-radius:50%"
    return (
        f'<img src="{uri}" width="{size}" height="{size}" '
        f'style="vertical-align:middle;margin-right:6px{radius}" alt="">'
    )


def labelled(
    team: str, size: int = 20, base: str | None = None, fallback: bool = True
) -> str:
    """Badge and club name as one inline HTML fragment."""
    return f"{img_tag(team, size, base, fallback)}<span>{team}</span>"


def available(base: Path | None = None) -> set[str]:
    """Slugs that have a badge on disk, for reporting coverage."""
    directory = crest_dir(base)
    if not directory.exists():
        return set()
    return {path.stem for path in directory.glob("*.png")}


def missing(teams: list[str], base: Path | None = None) -> list[str]:
    """Clubs with no badge, so a script can say so rather than shrug."""
    have = available(base)
    return sorted({team for team in teams if slug(team) not in have})
