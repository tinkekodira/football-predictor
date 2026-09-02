"""Fetch raw CSVs from football-data.co.uk.

The source is a plain static file host: no API key, no quota, one CSV per
league-season. That makes the ingest layer simple, but three details matter.

* **Caching.** Completed seasons never change, so they are downloaded once and
  then read from disk forever. Only the in-progress season is refreshed.
* **Encoding.** The files are Latin-1 in places (accented club names), and
  a few carry trailing blank columns and padding rows.
* **Politeness.** A short delay between requests. The host publishes no rate
  limit, which is not a reason to behave badly.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from . import config


class IngestError(RuntimeError):
    """Raised when a file cannot be fetched or parsed."""


@dataclass(frozen=True)
class SeasonFile:
    league: str
    season_start_year: int
    path: Path
    from_cache: bool


def _cache_path(league: str, season_start_year: int) -> Path:
    return config.RAW_DIR / config.season_code(season_start_year) / f"{league}.csv"


def _is_fresh(path: Path, season_start_year: int) -> bool:
    """Whether a cached file can be reused without re-downloading."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if season_start_year < config.CURRENT_SEASON_START_YEAR:
        return True  # a finished season's results cannot change
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < config.CURRENT_SEASON_CACHE_HOURS


def _get(url: str, attempts: int = 3) -> bytes:
    """GET with linear backoff. Raises IngestError once attempts run out."""
    headers = {"User-Agent": config.USER_AGENT}
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url, headers=headers, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            if not response.content.strip():
                raise IngestError(f"Empty response from {url}")
            return response.content
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last = exc
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise IngestError(f"Failed to fetch {url} after {attempts} attempts: {last}")


def download_season(
    league: str, season_start_year: int, force: bool = False
) -> SeasonFile:
    """Fetch one league-season CSV, using the cache when it is still valid."""
    path = _cache_path(league, season_start_year)
    if not force and _is_fresh(path, season_start_year):
        return SeasonFile(league, season_start_year, path, from_cache=True)

    url = config.season_csv_url(league, season_start_year)
    content = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    time.sleep(config.REQUEST_DELAY_SECONDS)
    return SeasonFile(league, season_start_year, path, from_cache=False)


def download_all(
    leagues: list[str] | None = None,
    years: list[int] | None = None,
    force: bool = False,
    on_progress=None,
) -> list[SeasonFile]:
    """Fetch every league-season combination.

    Missing files are skipped with a warning rather than aborting the run: a
    league occasionally has no file for an old season, and one gap should not
    cost you the other 49 downloads.
    """
    leagues = leagues or list(config.LEAGUES)
    years = years or config.season_years()
    files: list[SeasonFile] = []
    for year in years:
        for league in leagues:
            try:
                season_file = download_season(league, year, force=force)
            except IngestError as exc:
                if on_progress:
                    on_progress(f"  skipped {league} {config.season_label(year)}: {exc}")
                continue
            files.append(season_file)
            if on_progress:
                source = "cached" if season_file.from_cache else "downloaded"
                on_progress(f"  {source} {league} {config.season_label(year)}")
    return files


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a raw CSV defensively.

    Handles the Latin-1 club names, the unnamed trailing columns some files
    carry, and the fully blank padding rows at the end of others.
    """
    raw_bytes = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
                on_bad_lines="skip",
                low_memory=False,
            )
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte sequence
        raise IngestError(f"Could not decode {path}")

    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df.dropna(how="all")


def download_upcoming_fixtures() -> pd.DataFrame:
    """Fetch the site's list of upcoming fixtures with current prices.

    Used by the Phase 4 scanner. Returned unnormalised; the caller decides
    which leagues it cares about.
    """
    content = _get(config.FIXTURES_URL)
    path = config.RAW_DIR / "fixtures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return read_raw_csv(path)
