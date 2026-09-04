"""Download club badges for every team in the calendar.

    python scripts/build_crests.py            # needs FOOTBALL_API_KEY once
    python scripts/build_crests.py --report   # what is missing, no requests

Two steps, and only the first needs a key:

1. **Find each club's API-Football team id.** Ids come from `/teams`, one
   request per league. The top five leagues are not enough on their own -
   promoted clubs were in a second tier last season - so the second tiers are
   fetched too. About ten requests of the free plan's hundred, once.
2. **Download the badge.** From `media.api-sports.io`, which is a **public**
   image CDN: fetching a badge sends no key. So once the ids are known the
   badges keep working on any plan, and a rebuild for one newly promoted club
   costs a single request.

Ids are cached in `data/raw/crest_ids.json`, so a re-run after a promotion only
fetches what it does not already have. Badges are resized to 48px on the way
in - the originals are ~90KB each and a Saturday would otherwise put four
megabytes of PNG on one page.

**Unmatched club names are reported, never guessed**, the same rule as the
injury feed: the feed's names are not the project's, and a badge on the wrong
club is worse than no badge.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, crests, database, injuries  # noqa: E402

# Second tiers, so that a club promoted this summer still gets a badge. These
# ids are API-Football's own and are checked the only way that works: the
# script reports how many teams each returned, and zero means a wrong id.
SECOND_TIER_IDS = {
    "E0": 40,   # Championship
    "SP1": 141,  # Segunda Division
    "I1": 136,  # Serie B
    "D1": 79,   # 2. Bundesliga
    "F1": 62,   # Ligue 2
}

# The free plan serves this season for /teams as well; see
# config.INJURY_FREE_PLAN_LAST_SEASON for how that limit was established.
ID_SEASON = config.INJURY_FREE_PLAN_LAST_SEASON

ID_CACHE = config.RAW_DIR / "crest_ids.json"


def wanted_teams(con) -> list[str]:
    """Every club appearing in the stored calendar."""
    frame = con.execute(
        "SELECT DISTINCT home_team AS team FROM fixtures "
        "UNION SELECT DISTINCT away_team FROM fixtures"
    ).df()
    return sorted(frame["team"].dropna().astype(str))


def load_ids() -> dict[str, int]:
    if ID_CACHE.exists():
        return {k: int(v) for k, v in json.loads(
            ID_CACHE.read_text(encoding="utf-8")
        ).items()}
    return {}


def save_ids(ids: dict[str, int]) -> None:
    ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ID_CACHE.write_text(json.dumps(ids, indent=2, sort_keys=True), encoding="utf-8")


def fetch_team_ids(league_id: int, season: int, key: str) -> list[dict]:
    """One league's teams from `/teams`. Raises on anything unexpected."""
    url = f"https://v3.football.api-sports.io/teams?league={league_id}&season={season}"
    request = urllib.request.Request(
        url,
        headers={"x-apisports-key": key, "Accept": "application/json",
                 "User-Agent": config.USER_AGENT},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise injuries.InjuryFeedError(
            f"{url} returned HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise injuries.InjuryFeedError(
            f"{url} could not be reached: {error.reason}"
        ) from error

    if payload.get("errors"):
        raise injuries.InjuryFeedError(f"The feed reported: {payload['errors']}")
    return payload.get("response", [])


def download_badge(team_id: int, destination: Path, pixels: int) -> bool:
    """Fetch one badge and write it resized. No key is sent; the CDN is public."""
    url = crests.CREST_URL.format(team_id=team_id)
    request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    try:
        with urllib.request.urlopen(
            request, timeout=config.REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False

    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return False
    image.thumbnail((pixels, pixels), Image.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="say what is missing and make no requests")
    parser.add_argument("--pixels", type=int, default=crests.CREST_PIXELS)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download badges that already exist")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    con = database.connect(args.db, read_only=True)
    try:
        teams = wanted_teams(con)
        known = set(database.known_teams(con)) | set(teams)
    finally:
        con.close()

    if not teams:
        print("No fixtures stored. Run scripts/build_fixtures.py first.")
        return 1

    absent = crests.missing(teams)
    print(f"{len(teams)} clubs in the calendar, "
          f"{len(teams) - len(absent)} with a badge, {len(absent)} without.")
    if args.report:
        for team in absent:
            print(f"  missing: {team}")
        return 0
    if not absent and not args.refresh:
        print("Nothing to do.")
        return 0

    ids = load_ids()
    # Injury payloads already on disk carry a team id and name for every club
    # that had an absentee, which in a full season is all of them. Free ids for
    # any league that has ever been fetched, and no request at all.
    harvested = harvest_cached_ids(known)
    if harvested:
        new = {k: v for k, v in harvested.items() if k not in ids}
        if new:
            print(f"Recovered {len(new)} team id(s) from cached injury "
                  "payloads, at no request cost.")
        ids.update(harvested)
        save_ids(ids)

    todo = teams if args.refresh else absent
    unresolved = [t for t in todo if t not in ids]

    if unresolved:
        key = injuries.api_key()
        if key is None:
            # Do the work that *can* be done rather than nothing. Refusing the
            # whole run because part of it is blocked is the same mistake as a
            # league silently missing: the outcome looks like failure when most
            # of it would have succeeded.
            print(
                f"\n{len(unresolved)} club(s) have no cached id and "
                f"{config.INJURY_API_KEY_ENV} is not set, so those ids cannot "
                "be looked up. Downloading the rest anyway - the image CDN "
                "needs no key, only the id lookup does."
            )
        else:
            ids.update(discover_ids(key, known))
            save_ids(ids)

    downloaded, failed = 0, []
    for team in todo:
        team_id = ids.get(team)
        if team_id is None:
            failed.append(team)
            continue
        target = crests.crest_dir() / f"{crests.slug(team)}.png"
        if download_badge(team_id, target, args.pixels):
            downloaded += 1
        else:
            failed.append(team)
        time.sleep(0.2)

    print(f"\nDownloaded {downloaded} badge(s) to {crests.crest_dir()}.")
    if failed:
        print(f"  no badge for {len(failed)}: {', '.join(sorted(failed)[:12])}")
        print("  Add an alias to injuries.TEAM_ALIASES if the club is really "
              "there under another name.")
    return 0


def harvest_cached_ids(known: set[str]) -> dict[str, int]:
    """Team ids from injury payloads already downloaded.

    Every injury row names its club and carries the club's id, so a league that
    has been fetched once has already paid for its ids. Worth doing first: it
    means the Premier League gets badges from a single earlier probe, with no
    key and no request.
    """
    cache = config.RAW_DIR / injuries.CACHE_DIRNAME
    if not cache.exists():
        return {}
    found: dict[str, int] = {}
    for path in sorted(cache.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("response", []):
            block = item.get("team") or {}
            name, team_id = block.get("name"), block.get("id")
            if not name or not team_id:
                continue
            mapped = injuries.to_football_data_name(name, known)
            if mapped is not None:
                found.setdefault(mapped, int(team_id))
    return found


def discover_ids(key: str, known: set[str]) -> dict[str, int]:
    """Look up team ids across the top five leagues and their second tiers."""
    found: dict[str, int] = {}
    unmatched: list[str] = []
    for league, top_id in config.INJURY_LEAGUE_IDS.items():
        for tier, league_id in (("top", top_id),
                                ("2nd", SECOND_TIER_IDS.get(league))):
            if league_id is None:
                continue
            try:
                entries = fetch_team_ids(league_id, ID_SEASON, key)
            except injuries.InjuryFeedError as error:
                print(f"  {league} {tier}: {error}")
                continue
            print(f"  {league} {tier}: {len(entries)} teams")
            if not entries:
                print(f"    (zero teams usually means league id {league_id} is "
                      "wrong for this provider)")
            for item in entries:
                block = item.get("team") or {}
                name, team_id = block.get("name"), block.get("id")
                if not name or not team_id:
                    continue
                mapped = injuries.to_football_data_name(name, known)
                if mapped is None:
                    unmatched.append(name)
                    continue
                found.setdefault(mapped, int(team_id))
            time.sleep(config.REQUEST_DELAY_SECONDS)

    if unmatched:
        print(f"\n  {len(set(unmatched))} feed name(s) did not map to a club we "
              "know, and were dropped rather than guessed:")
        for name in sorted(set(unmatched))[:20]:
            print(f"    {name!r} -> normalises to {injuries.normalise(name)!r}")
    return found


if __name__ == "__main__":
    raise SystemExit(main())
