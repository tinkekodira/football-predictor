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

Everything here degrades to nothing: a club with no badge on disk renders as
its name alone, which is exactly what the page did before badges existed. That
matters because promotion happens - a club coming up will be missing from the
catalogue until it is rebuilt.
"""

from __future__ import annotations

import base64
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


def img_tag(team: str, size: int = 20, base: str | None = None) -> str:
    """An `<img>` for a club badge, or an empty string.

    Returning "" rather than a placeholder is deliberate: a row with a missing
    badge should look like a row with a slightly shorter name, not like a
    broken image. `vertical-align: middle` is what keeps the badge on the text
    baseline instead of sitting the line height on top of it.
    """
    uri = data_uri(team, base)
    if uri is None:
        return ""
    return (
        f'<img src="{uri}" width="{size}" height="{size}" '
        f'style="vertical-align:middle;margin-right:6px" alt="">'
    )


def labelled(team: str, size: int = 20, base: str | None = None) -> str:
    """Badge and club name as one inline HTML fragment."""
    return f"{img_tag(team, size, base)}<span>{team}</span>"


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
