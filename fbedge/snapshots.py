"""Archiving the upcoming-fixtures file, which the source overwrites weekly.

**This module exists because of one fact.**
`https://www.football-data.co.uk/fixtures.csv` holds the next few days of
fixtures with the prices available when it was built, and it is *replaced* each
time the source rebuilds it. Nothing anywhere publishes last Friday's version.
So a pre-match price that is not written down when it is pulled is gone for
good, and no amount of later work can reconstruct it - unlike every other input
in this project, which can be re-downloaded from a static archive at any time.

Every other table here can be dropped and rebuilt in two minutes. This one
cannot, which is why it is append-only and why it is the first thing Phase 4
builds.

**What it is worth.** The season files carry an "open" and a "close" price per
match, and closing line value is currently measured against the source's own
closing line. An archive of our own pulls eventually measures something the
season files cannot: whether the price *we* recorded at the moment we would
have bet beat the close. That needs `reconcile()` to join these rows to the
played match, which is why that half is here too rather than deferred.

**Dedupe is by content, not by pull.** Pulling twice in an afternoon usually
finds the same prices, and storing that twice would inflate the archive and
make "how often did this price move" unanswerable. Each fixture's identity plus
every price attached to it is hashed, and a hash already present is recorded as
a repeat sighting rather than a new row. A genuine price change produces a
different hash and therefore a new row, which is exactly the event worth
keeping. Hashing per fixture rather than per file matters: one match's price
moving must not force the other 197 to be stored again.

**Odds go in long format**, one row per price, like every other odds table in
this project. Adding a bookmaker is then a data change rather than a migration,
and `normalize.odds_long` is reused verbatim so there is only one reader of the
source's wide column conventions - the away-handicap sign alone is a bug this
project has already paid for once.

**Timing is fixed, not continuous.** The source states on its own fixtures page
that odds for weekend fixtures are collected on Friday afternoons "generally
not later than 17:00 British Standard Time", and for midweek fixtures on
Tuesdays "not later than 13:00". So this is a twice-weekly snapshot, not a live
feed, and `collection_window` records which of the two a row belongs to. That
is inferred from the fixture's weekday, not stated per row by the source; see
`collection_window` for the inference and its one ambiguity.

**Timestamps are UTC.** `pulled_at_utc` is a permanent forensic record of when
a price was seen, and this project already has a `DISPLAY_TIMEZONE` constant
because local time is a display concern. Storing local time here would bake one
machine's clock settings into an archive nobody can rebuild.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, ingest, normalize

SNAPSHOT_TABLE = "fixture_snapshots"
SNAPSHOT_ODDS_TABLE = "snapshot_odds"

# The source's stated collection deadlines, in its own words and its own
# timezone. Weekday numbers are Python's: Monday is 0.
WEEKEND_COLLECTION_WEEKDAY = 4     # Friday
WEEKEND_COLLECTION_HOUR = 17       # 17:00 British time
MIDWEEK_COLLECTION_WEEKDAY = 1     # Tuesday
MIDWEEK_COLLECTION_HOUR = 13       # 13:00 British time
SOURCE_TIMEZONE = "Europe/London"

# Fixture weekdays belonging to each collection round. A Monday-night match is
# the tail of the weekend round and its prices come from the Friday pull, which
# is the one genuinely arguable assignment here: in a midweek programme that
# starts on Monday it would be wrong. The source publishes no per-row marker,
# so this is an inference, and it is recorded as one - `nominal_collected_at`
# is an upper bound on when a price was collected, not a measurement of it.
WEEKEND_FIXTURE_WEEKDAYS = frozenset({4, 5, 6, 0})   # Fri Sat Sun Mon
MIDWEEK_FIXTURE_WEEKDAYS = frozenset({1, 2, 3})      # Tue Wed Thu


class StaleFixtures(RuntimeError):
    """The fixtures file is too old to scan against.

    A separate exception because a stale file is not a download failure and
    must not be handled like one. The source's own page warns that browser
    caching serves people last week's fixtures, and a scan run against last
    week's prices produces confident, wrong, entirely plausible-looking output.
    """


# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------

def download(force: bool = False, cache_hours: float | None = None) -> tuple[pd.DataFrame, Path]:
    """Fetch `fixtures.csv`, reusing a recent cached copy.

    Reuses `ingest`'s User-Agent, retry and polite-delay behaviour rather than
    opening a second HTTP path into the same host.
    """
    path = config.RAW_DIR / "fixtures.csv"
    budget = config.FIXTURES_CACHE_HOURS if cache_hours is None else cache_hours
    if not force and path.exists() and path.stat().st_size > 0:
        age_hours = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours < budget:
            return ingest.read_raw_csv(path), path
    frame = ingest.download_upcoming_fixtures()
    return frame, path


def fixture_frame(raw: pd.DataFrame, leagues: list[str] | None = None) -> pd.DataFrame:
    """The fixtures file's identity columns, canonicalised.

    `Div` is what selects the league here, and it is the one column the season
    files do not need: a season file is one league by construction, while this
    file mixes twenty-two divisions in one table. Anything outside `leagues` is
    dropped rather than stored, because a snapshot of a division the models
    cannot price is dead weight in an append-only archive.
    """
    df = raw.copy().reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    if "Div" not in df.columns:
        raise ValueError(
            "The fixtures file has no `Div` column, so its rows cannot be "
            "assigned to a league. The format has changed; check the source "
            "before storing anything."
        )

    out = pd.DataFrame({"source_row": df.index})
    out["league"] = df["Div"].astype("string").str.strip()
    out["fixture_date"] = normalize.parse_dates(df["Date"])
    out["kickoff_time"] = (
        df["Time"].astype("string").str.strip() if "Time" in df.columns else pd.NA
    )
    out["home_team_raw"] = df["HomeTeam"]
    out["away_team_raw"] = df["AwayTeam"]
    out["home_team"] = df["HomeTeam"].map(normalize.canonical_team)
    out["away_team"] = df["AwayTeam"].map(normalize.canonical_team)
    out["referee"] = (
        df["Referee"].map(normalize.canonical_referee)
        if "Referee" in df.columns else None
    )

    keep = (
        out["fixture_date"].notna()
        & (out["home_team"] != "")
        & (out["away_team"] != "")
    )
    if leagues is not None:
        keep &= out["league"].isin(leagues)
    out = out[keep].reset_index(drop=True)

    if out.empty:
        return out

    windows = [collection_window(d) for d in out["fixture_date"]]
    out["collection_window"] = [w for w, _ in windows]
    out["nominal_collected_at"] = [t for _, t in windows]
    return out


def collection_window(fixture_date) -> tuple[str, pd.Timestamp | None]:
    """Which of the source's two collection rounds a fixture belongs to.

    Returns `(window, nominal_collected_at_utc)`. The timestamp is the *latest*
    moment the source says the prices could have been collected - the Friday
    17:00 or Tuesday 13:00 immediately preceding the fixture, converted from
    British time to UTC - not a claim about when they actually were. Reading it
    as a measurement would overstate what the source publishes.
    """
    day = pd.Timestamp(fixture_date)
    if pd.isna(day):
        return "unknown", None
    weekday = day.weekday()
    if weekday in WEEKEND_FIXTURE_WEEKDAYS:
        window, target, hour = "weekend", WEEKEND_COLLECTION_WEEKDAY, WEEKEND_COLLECTION_HOUR
    elif weekday in MIDWEEK_FIXTURE_WEEKDAYS:
        window, target, hour = "midweek", MIDWEEK_COLLECTION_WEEKDAY, MIDWEEK_COLLECTION_HOUR
    else:  # pragma: no cover - the two sets cover all seven weekdays
        return "unknown", None

    days_back = (weekday - target) % 7
    local = dt.datetime.combine(
        (day - pd.Timedelta(days=days_back)).date(), dt.time(hour=hour)
    ).replace(tzinfo=ZoneInfo(SOURCE_TIMEZONE))
    return window, pd.Timestamp(local.astimezone(dt.timezone.utc)).tz_localize(None)


# --------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------

def content_hash(identity: dict, odds: pd.DataFrame) -> str:
    """A stable fingerprint of one fixture and every price attached to it.

    Identity and prices together, so that a moved kick-off time counts as a
    change just as a moved price does. Prices are rounded to three decimals and
    sorted before hashing: the source writes 2.1 and 2.10 interchangeably, and
    an archive that treated those as different snapshots would fill up with
    formatting noise and report price movement that never happened.
    """
    parts = [f"{key}={identity.get(key)}" for key in sorted(identity)]
    if not odds.empty:
        rows = odds.copy()
        rows["line"] = [
            "" if pd.isna(v) else f"{float(v):.3f}" for v in rows["line"]
        ]
        rows["price"] = [f"{float(v):.3f}" for v in rows["price"]]
        tuples = sorted(
            f"{r.bookmaker}|{r.phase}|{r.market}|{r.selection}|{r.line}|{r.price}"
            for r in rows.itertuples()
        )
        parts += tuples
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def fixture_key(league: str, fixture_date, home: str, away: str) -> str:
    """Identity of a fixture across snapshots, ignoring its prices.

    What groups an archive into "the price history of this match". Deliberately
    not `normalize.make_match_id`: that one is keyed on the date the match was
    *played*, and a postponed fixture snapshotted on its original date would
    then never join to itself. The reconciliation step handles that drift with
    a date tolerance, and needs a key that survives it.
    """
    date_str = "unknown" if pd.isna(fixture_date) else pd.Timestamp(fixture_date).strftime("%Y%m%d")
    return (
        f"{league}_{date_str}_{normalize.team_slug(home)}_{normalize.team_slug(away)}"
    )


# --------------------------------------------------------------------------
# Building a snapshot
# --------------------------------------------------------------------------

def build_snapshot(
    raw: pd.DataFrame,
    leagues: list[str] | None = None,
    pulled_at: dt.datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn one raw fixtures file into (snapshot rows, long odds rows).

    Both frames are keyed on `content_hash`, which is what makes the dedupe in
    `write_snapshot` a plain anti-join rather than a comparison of every price.
    """
    stamp = pulled_at or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    fixtures = fixture_frame(raw, leagues)
    columns = [
        "content_hash", "fixture_key", "league", "fixture_date", "kickoff_time",
        "home_team", "away_team", "home_team_raw", "away_team_raw", "referee",
        "collection_window", "nominal_collected_at",
        "first_pulled_at_utc", "last_pulled_at_utc", "match_id",
    ]
    odds_columns = [
        "content_hash", "bookmaker", "phase", "market", "selection", "line", "price",
    ]
    if fixtures.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=odds_columns)

    wide = raw.reset_index(drop=True).loc[fixtures["source_row"].to_numpy()]
    placeholders = pd.Series(range(len(fixtures)), name="row")
    odds = normalize.odds_long(wide, placeholders, key_name="row")

    rows, odds_frames = [], []
    by_row = dict(tuple(odds.groupby("row"))) if not odds.empty else {}
    for position, fixture in enumerate(fixtures.itertuples()):
        identity = {
            "league": fixture.league,
            "date": pd.Timestamp(fixture.fixture_date).date().isoformat(),
            "kickoff": "" if pd.isna(fixture.kickoff_time) else str(fixture.kickoff_time),
            "home": fixture.home_team,
            "away": fixture.away_team,
        }
        prices = by_row.get(position, odds.iloc[0:0])
        digest = content_hash(identity, prices)
        rows.append(
            {
                "content_hash": digest,
                "fixture_key": fixture_key(
                    fixture.league, fixture.fixture_date,
                    fixture.home_team, fixture.away_team,
                ),
                "league": fixture.league,
                "fixture_date": pd.Timestamp(fixture.fixture_date).date(),
                "kickoff_time": None if pd.isna(fixture.kickoff_time) else str(fixture.kickoff_time),
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "home_team_raw": str(fixture.home_team_raw),
                "away_team_raw": str(fixture.away_team_raw),
                "referee": fixture.referee,
                "collection_window": fixture.collection_window,
                "nominal_collected_at": fixture.nominal_collected_at,
                "first_pulled_at_utc": stamp,
                "last_pulled_at_utc": stamp,
                "match_id": None,
            }
        )
        if not prices.empty:
            block = prices.drop(columns=["row"]).copy()
            block.insert(0, "content_hash", digest)
            odds_frames.append(block)

    snapshot = pd.DataFrame(rows, columns=columns)
    # Two identical rows in one file - the source has shipped duplicate lines
    # before - would violate the primary key. Collapse them here rather than
    # letting the insert fail on a pull nobody is watching.
    snapshot = snapshot.drop_duplicates("content_hash", keep="first").reset_index(drop=True)
    if odds_frames:
        long_odds = pd.concat(odds_frames, ignore_index=True)
        long_odds = long_odds.drop_duplicates(
            subset=["content_hash", "bookmaker", "phase", "market", "selection", "line"]
        ).reset_index(drop=True)
    else:
        long_odds = pd.DataFrame(columns=odds_columns)
    return snapshot, long_odds[odds_columns]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def create_tables(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE} (
            content_hash          VARCHAR PRIMARY KEY,
            fixture_key           VARCHAR NOT NULL,
            league                VARCHAR NOT NULL,
            fixture_date          DATE NOT NULL,
            kickoff_time          VARCHAR,
            home_team             VARCHAR NOT NULL,
            away_team             VARCHAR NOT NULL,
            home_team_raw         VARCHAR,
            away_team_raw         VARCHAR,
            referee               VARCHAR,
            collection_window     VARCHAR,
            nominal_collected_at  TIMESTAMP,
            first_pulled_at_utc   TIMESTAMP NOT NULL,
            last_pulled_at_utc    TIMESTAMP NOT NULL,
            match_id              VARCHAR
        )
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOT_ODDS_TABLE} (
            content_hash  VARCHAR NOT NULL,
            bookmaker     VARCHAR NOT NULL,
            phase         VARCHAR NOT NULL,
            market        VARCHAR NOT NULL,
            selection     VARCHAR NOT NULL,
            line          DOUBLE,
            price         DOUBLE NOT NULL
        )
        """
    )


def write_snapshot(con, snapshot: pd.DataFrame, odds: pd.DataFrame) -> dict:
    """Append what is new; record a repeat sighting for what is not.

    **Nothing here ever updates or deletes a price.** Every other loader in
    this project is delete-then-insert, because its source can be re-downloaded
    and the newest copy is the right one. This source cannot: the file it came
    from no longer exists. The only field that moves is `last_pulled_at_utc`,
    and only forwards, so the archive can answer "how long did this price
    stand" without ever losing a price it once held.

    Returns counts rather than a bare number: "stored 0 new rows, saw 198
    again" is a healthy afternoon and must not read like a failure.
    """
    create_tables(con)
    if snapshot.empty:
        return {"fixtures_seen": 0, "new_snapshots": 0, "repeat_snapshots": 0,
                "new_odds_rows": 0}

    con.register("incoming_snapshot", snapshot)
    existing = set(
        row[0]
        for row in con.execute(
            f"SELECT content_hash FROM {SNAPSHOT_TABLE} WHERE content_hash IN "
            "(SELECT content_hash FROM incoming_snapshot)"
        ).fetchall()
    )

    # A hash already on file means these exact prices were archived before.
    # Move its sighting stamp forward and store nothing else.
    con.execute(
        f"""
        UPDATE {SNAPSHOT_TABLE} AS t
        SET last_pulled_at_utc = GREATEST(
            t.last_pulled_at_utc,
            (SELECT MAX(i.last_pulled_at_utc) FROM incoming_snapshot i
             WHERE i.content_hash = t.content_hash)
        )
        WHERE t.content_hash IN (SELECT content_hash FROM incoming_snapshot)
        """
    )

    fresh = snapshot[~snapshot["content_hash"].isin(existing)]
    con.register("fresh_snapshot", fresh)
    columns = ", ".join(f'"{c}"' for c in snapshot.columns)
    con.execute(
        f"INSERT INTO {SNAPSHOT_TABLE} ({columns}) SELECT {columns} FROM fresh_snapshot"
    )

    fresh_odds = odds[odds["content_hash"].isin(set(fresh["content_hash"]))]
    con.register("fresh_odds", fresh_odds)
    con.execute(
        f"INSERT INTO {SNAPSHOT_ODDS_TABLE} "
        "SELECT content_hash, bookmaker, phase, market, selection, line, price "
        "FROM fresh_odds"
    )
    for name in ("incoming_snapshot", "fresh_snapshot", "fresh_odds"):
        con.unregister(name)

    return {
        "fixtures_seen": int(len(snapshot)),
        "new_snapshots": int(len(fresh)),
        "repeat_snapshots": int(len(snapshot) - len(fresh)),
        "new_odds_rows": int(len(fresh_odds)),
    }


def load_snapshots(
    con,
    leagues: list[str] | None = None,
    unplayed_only: bool = False,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Read the archive back.

    `latest_only` keeps the most recent snapshot per fixture, which is what a
    scan wants; the full history is what a CLV study wants, so both are here.
    """
    if not _has_table(con, SNAPSHOT_TABLE):
        return pd.DataFrame()
    clauses, params = [], []
    if leagues:
        clauses.append(f"league IN ({','.join('?' for _ in leagues)})")
        params.extend(leagues)
    if unplayed_only:
        clauses.append("match_id IS NULL")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    frame = con.execute(
        f"SELECT * FROM {SNAPSHOT_TABLE} {where} ORDER BY fixture_date, "
        "fixture_key, first_pulled_at_utc",
        params,
    ).df()
    if latest_only and not frame.empty:
        frame = (
            frame.sort_values("first_pulled_at_utc")
            .drop_duplicates("fixture_key", keep="last")
            .reset_index(drop=True)
        )
    return frame


def load_snapshot_odds(con, content_hashes: list[str]) -> pd.DataFrame:
    """Every archived price for a set of snapshots."""
    if not content_hashes or not _has_table(con, SNAPSHOT_ODDS_TABLE):
        return pd.DataFrame(
            columns=["content_hash", "bookmaker", "phase", "market",
                     "selection", "line", "price"]
        )
    placeholders = ",".join("?" for _ in content_hashes)
    return con.execute(
        f"SELECT * FROM {SNAPSHOT_ODDS_TABLE} WHERE content_hash IN ({placeholders})",
        list(content_hashes),
    ).df()


def _has_table(con, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone() is not None


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------

def staleness(
    fixtures: pd.DataFrame,
    path: Path | None = None,
    now: dt.datetime | None = None,
    max_age_hours: float | None = None,
) -> dict:
    """Is this file fresh enough to scan against?

    Two independent checks, because they fail differently. **Fixture dates in
    the past** means the file describes a round that has already been played,
    which is what the source's own page warns about: "IF YOU SEE OLDER FIXTURES
    TRY CLEARING YOUR BROWSER CACHE". **File age** catches the subtler case
    where a cached copy is stale but its fixtures have not kicked off yet.

    Reported as a dict rather than raised, so a caller can print the whole
    diagnosis. `check()` is the raising version.
    """
    moment = now or dt.datetime.now()
    budget = config.FIXTURES_MAX_AGE_HOURS if max_age_hours is None else max_age_hours

    age_hours = None
    if path is not None and Path(path).exists():
        age_hours = (moment.timestamp() - Path(path).stat().st_mtime) / 3600

    dates = (
        pd.to_datetime(fixtures["fixture_date"], errors="coerce").dropna()
        if not fixtures.empty and "fixture_date" in fixtures else pd.Series(dtype="datetime64[ns]")
    )
    latest = dates.max().date() if len(dates) else None
    reasons = []
    if latest is None:
        reasons.append("the file holds no dated fixtures at all")
    elif latest < moment.date():
        reasons.append(
            f"every fixture in it has already been played (latest {latest}, "
            f"today {moment.date()})"
        )
    if age_hours is not None and age_hours > budget:
        reasons.append(
            f"the cached copy is {age_hours:.1f}h old, past the "
            f"{budget:g}h threshold in config.FIXTURES_MAX_AGE_HOURS"
        )
    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "latest_fixture_date": latest,
        "age_hours": age_hours,
        "n_fixtures": int(len(fixtures)),
    }


def check(fixtures: pd.DataFrame, path: Path | None = None, **kwargs) -> None:
    """Raise `StaleFixtures` if the file is not fit to scan. Silent if it is."""
    report = staleness(fixtures, path, **kwargs)
    if report["stale"]:
        raise StaleFixtures(
            "Refusing to scan: " + "; and ".join(report["reasons"]) + ". "
            "Re-run with --refresh to force a download. A scan against a stale "
            "file produces confident prices for matches already played."
        )


# --------------------------------------------------------------------------
# Reconciliation with played matches
# --------------------------------------------------------------------------

def reconcile(con, tolerance_days: int | None = None) -> dict:
    """Join archived snapshots to the matches that were eventually played.

    The fixtures file has no results column and no stable identifier, so the
    join is on `(date, canonical home, canonical away)` after both sides have
    been through `normalize.canonical_team` - which is why the snapshot stores
    the canonical name rather than the raw one.

    **The date needs a tolerance.** A fixture snapshotted on Saturday and
    postponed to Sunday is the same match, and a late kick-off can land on a
    different calendar date in the two files. One day either side catches both
    without being loose enough to merge two legs of a double-header, since the
    same pair of teams does not play twice in three days. The exact-date match
    is preferred when several are in range, so the tolerance only ever rescues
    a row that would otherwise be lost.

    Unmatched rows are **left unmatched and counted**, never dropped: a
    snapshot that silently fails to join looks exactly like a fixture nobody
    priced, and this project has been bitten by that shape of bug before.
    Newly promoted clubs and accented names are where it happens.
    """
    create_tables(con)
    window = (
        config.FIXTURE_RECONCILE_TOLERANCE_DAYS
        if tolerance_days is None else tolerance_days
    )
    snapshots = con.execute(
        f"SELECT content_hash, league, fixture_date, home_team, away_team, match_id "
        f"FROM {SNAPSHOT_TABLE}"
    ).df()
    if snapshots.empty:
        return {"snapshots": 0, "matched": 0, "already_matched": 0,
                "unmatched": 0, "unmatched_rows": pd.DataFrame()}

    matches = con.execute(
        "SELECT match_id, league, date AS match_date, home_team, away_team FROM matches"
    ).df()
    snapshots["fixture_date"] = pd.to_datetime(snapshots["fixture_date"])
    already = int(snapshots["match_id"].notna().sum())

    if matches.empty:
        unmatched = snapshots[snapshots["match_id"].isna()]
        return {"snapshots": len(snapshots), "matched": 0,
                "already_matched": already, "unmatched": len(unmatched),
                "unmatched_rows": unmatched}
    matches["match_date"] = pd.to_datetime(matches["match_date"])

    pending = snapshots[snapshots["match_id"].isna()].copy()
    joined = pending.merge(
        matches, on=["league", "home_team", "away_team"], how="left",
        suffixes=("", "_played"),
    )
    drift = (joined["match_date"] - joined["fixture_date"]).dt.days.abs()
    joined = joined[drift.notna() & (drift <= window)].copy()
    joined["drift"] = drift[drift.notna() & (drift <= window)]
    # Exact date first, so the tolerance only ever rescues an otherwise-lost
    # row rather than quietly preferring a neighbouring fixture.
    joined = joined.sort_values("drift").drop_duplicates("content_hash", keep="first")

    if not joined.empty:
        con.register("reconciled", joined[["content_hash", "match_id_played"]]
                     .rename(columns={"match_id_played": "resolved_id"}))
        con.execute(
            f"""
            UPDATE {SNAPSHOT_TABLE} AS t
            SET match_id = (SELECT r.resolved_id FROM reconciled r
                            WHERE r.content_hash = t.content_hash)
            WHERE t.content_hash IN (SELECT content_hash FROM reconciled)
            """
        )
        con.unregister("reconciled")

    matched_hashes = set(joined["content_hash"])
    unmatched = pending[~pending["content_hash"].isin(matched_hashes)]
    return {
        "snapshots": int(len(snapshots)),
        "matched": int(len(joined)),
        "already_matched": already,
        "unmatched": int(len(unmatched)),
        "unmatched_rows": unmatched,
    }


def export(con, directory: Path | None = None) -> dict:
    """Mirror the archive to tracked CSV files.

    **This is the backup, and it is the point.** Everything else in the
    database rebuilds from static files in two minutes, which is why the
    database itself is not tracked. This table does not rebuild from anything:
    the source overwrote the file it came from the moment it published the next
    one. An irreplaceable asset living only in an untracked binary is one
    disk failure away from never having existed.

    Written whole rather than appended, and sorted deterministically, so that a
    commit diff shows the week's new prices and no reordering noise.
    """
    target = Path(directory) if directory else config.SNAPSHOT_EXPORT_DIR
    target.mkdir(parents=True, exist_ok=True)
    if not _has_table(con, SNAPSHOT_TABLE):
        return {"fixtures": 0, "prices": 0, "directory": target}

    fixtures = con.execute(
        f"SELECT * FROM {SNAPSHOT_TABLE} ORDER BY fixture_date, fixture_key, "
        "first_pulled_at_utc, content_hash"
    ).df()
    prices = con.execute(
        f"""
        SELECT o.* FROM {SNAPSHOT_ODDS_TABLE} o
        JOIN {SNAPSHOT_TABLE} f USING (content_hash)
        ORDER BY f.fixture_date, f.fixture_key, o.content_hash, o.bookmaker,
                 o.phase, o.market, o.selection, o.line
        """
    ).df()
    fixtures.to_csv(target / "fixture_snapshots.csv", index=False)
    prices.to_csv(target / "snapshot_odds.csv", index=False)
    return {
        "fixtures": int(len(fixtures)),
        "prices": int(len(prices)),
        "directory": target,
    }


def import_export(con, directory: Path | None = None) -> dict:
    """Load a CSV mirror back into an empty database.

    The other half of `export`, and the half that makes it a backup rather than
    a gesture. A fresh clone rebuilds every other table from the season files;
    this is how it gets the archive, which nothing else can supply.

    Uses the same content-hash dedupe as a live pull, so importing over an
    existing archive merges rather than duplicating.
    """
    source = Path(directory) if directory else config.SNAPSHOT_EXPORT_DIR
    fixtures_path = source / "fixture_snapshots.csv"
    prices_path = source / "snapshot_odds.csv"
    if not fixtures_path.exists():
        return {"fixtures_seen": 0, "new_snapshots": 0, "repeat_snapshots": 0,
                "new_odds_rows": 0}
    fixtures = pd.read_csv(fixtures_path)
    prices = (
        pd.read_csv(prices_path) if prices_path.exists()
        else pd.DataFrame(columns=["content_hash", "bookmaker", "phase",
                                   "market", "selection", "line", "price"])
    )
    for column in ("fixture_date",):
        fixtures[column] = pd.to_datetime(fixtures[column]).dt.date
    for column in ("nominal_collected_at", "first_pulled_at_utc",
                   "last_pulled_at_utc"):
        fixtures[column] = pd.to_datetime(fixtures[column])
    return write_snapshot(con, fixtures, prices)


def coverage(con) -> pd.DataFrame:
    """How much of the archive has been joined to a played match, per league.

    Printed by the build's coverage report. An unmatched row for a fixture that
    has not kicked off yet is normal and expected; an unmatched row for a
    fixture whose date has passed is a name that failed to resolve, and those
    are counted separately because only the second is a defect.
    """
    if not _has_table(con, SNAPSHOT_TABLE):
        return pd.DataFrame()
    return con.execute(
        f"""
        SELECT league,
               COUNT(*)                                        AS snapshots,
               COUNT(DISTINCT fixture_key)                     AS fixtures,
               SUM(CASE WHEN match_id IS NOT NULL THEN 1 END)  AS reconciled,
               SUM(CASE WHEN match_id IS NULL
                         AND fixture_date < CURRENT_DATE THEN 1 END) AS unmatched_past,
               SUM(CASE WHEN match_id IS NULL
                         AND fixture_date >= CURRENT_DATE THEN 1 END) AS awaiting_kickoff,
               MIN(first_pulled_at_utc)                        AS first_pull,
               MAX(last_pulled_at_utc)                         AS last_pull
        FROM {SNAPSHOT_TABLE}
        GROUP BY league
        ORDER BY league
        """
    ).df()
