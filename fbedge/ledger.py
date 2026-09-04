"""A paper-trading ledger: what the scan claimed, and what happened next.

**This module exists because of one sentence in the standing rules.** The gate
that stops this project staking money says: paper-trade for several weeks
recording what would have been bet, and judge it on the closing line value from
*that* period, not the historical backtest. Nothing in this repository recorded
what the scan claimed, so that sentence described a procedure nobody could
carry out. This is the missing half.

**It is the same argument as `snapshots.py`, one level up.** That module exists
because the source overwrites its price file and last Friday's prices are
otherwise gone for ever. This one exists because a scan is transient in the
same way: it prints a table, the table scrolls away, and the claim it made
leaves no trace. Every other measurement in this project can be recomputed from
static inputs; a forward claim cannot be recomputed after the fact, because by
then the result is known and the model has moved on.

**The claim is immutable and the outcome lives in a separate table.** Two
tables rather than one with columns filled in later, because immutability
enforced by a schema is worth more than immutability enforced by a convention
somebody has to remember. `paper_bets` is insert-only. `paper_settlements` is
written once per bet, and `settle_open` will not overwrite a settled row.

**Model provenance is stored per row, and it is the load-bearing part.** A bet
recorded as "+6% EV on Arsenal" is worthless six weeks later if the defaults
have changed underneath it, because there is then no way to say whether the
claim was the model's or the reader's memory of it. So every row carries the
*resolved* target, blend weight, ridge, half-life, margin method and both
withholding thresholds - resolved, not requested, since `ridge=None` means
"whatever suits the target" and storing that would record the question rather
than the answer. This project has twice been misled by an instrument changing
mid-series without saying so (BACKLOG B1, B10) and once by a schema change
masquerading as a price move (B15). Those are the same defect, and this column
set is the fix for it in advance rather than afterwards.

**Withheld selections are recorded too, and that is deliberate.** BACKLOG B17
withholds a fixture from the ranking when a side has too little history or the
claimed edge is too large to believe, on measured grounds. Whether those two
thresholds are set correctly is itself an open question, and it can only be
answered by keeping the rows they suppress and settling them alongside the
rest. A ledger that recorded only the bets it liked could never mark its own
homework. They carry `staked = FALSE`, so the headline never includes them.

**Stakes are flat and there is no Kelly here.** One unit per bet, always. The
project's own rule is that return on investment over a few hundred bets is
close to noise while closing line value is measurable in weeks, so this ledger
reports CLV as the headline and profit as a subordinate check. A stake-sizing
scheme would make the profit column louder and no more informative.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import backtest, config, settlement

BETS_TABLE = "paper_bets"
SETTLEMENTS_TABLE = "paper_settlements"

# One unit per bet, and it is not a parameter. See the module docstring.
FLAT_STAKE = 1.0

BET_COLUMNS = [
    "bet_id", "recorded_at_utc", "last_seen_at_utc", "as_of",
    "fixture_key", "content_hash", "league", "fixture_date", "kickoff_time",
    "home_team", "away_team",
    "market", "selection", "selection_label", "line",
    "price_taken", "book", "market_probability", "market_source",
    "model_probability", "push_probability", "fair_price", "edge",
    "expected_value", "stake", "staked", "withheld_reason",
    "n_home", "n_away", "thin_history",
    "target", "blend_weight", "ridge", "half_life_days", "margin_method",
    "price_source", "min_matches", "max_ev", "code_version",
]

SETTLEMENT_COLUMNS = [
    "bet_id", "settled_at_utc", "match_id", "match_date",
    "home_goals", "away_goals",
    "win_fraction", "push_fraction", "profit_at_taken",
    "price_close", "closing_source", "closing_fair", "clv", "price_movement",
]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """The model settings a claim was made under, resolved rather than asked.

    **`ridge` and `target` must be the values the fit actually used**, which
    is why `from_bundle` exists and why the constructor is rarely called by
    hand. `predict.build_models` accepts `ridge=None` meaning "whatever suits
    the target" and `target=None` meaning "the default, and degrade quietly on
    a database with no xG". Recording either as given would store the question.

    `code_version` is carried but deliberately **not** part of the identity
    hash. Including it would re-record every open claim on every commit, which
    is noise; excluding it entirely would leave a repeat sighting from changed
    code indistinguishable from one from unchanged code, which is BACKLOG B15
    exactly. Stored, first-seen wins, and `record` reports when a repeat
    sighting arrives from a different revision.
    """

    target: str
    blend_weight: float
    ridge: float
    half_life_days: float
    margin_method: str
    price_source: str
    min_matches: int
    max_ev: float
    code_version: str = ""

    @classmethod
    def from_bundle(
        cls,
        bundle,
        margin_method: str,
        price_source: tuple[str, ...] | str,
        min_matches: int,
        max_ev: float,
        blend_weight: float = 0.5,
        code_version: str | None = None,
    ) -> "Provenance":
        """Read the resolved settings off a fitted model rather than a request."""
        source = (
            price_source if isinstance(price_source, str)
            else ",".join(price_source)
        )
        return cls(
            target=bundle.goals.target,
            # Only meaningful for a blended target; stored regardless so the
            # row is readable without knowing that rule.
            blend_weight=float(blend_weight),
            ridge=float(bundle.goals.ridge),
            half_life_days=float(bundle.goals.half_life_days),
            margin_method=str(margin_method),
            price_source=source,
            min_matches=int(min_matches),
            max_ev=float(max_ev),
            code_version=(
                code_version if code_version is not None else git_revision()
            ),
        )

    def key(self) -> str:
        """The part of a claim's identity that is about the model, not the price.

        Two scans that differ in any of these made different claims and both
        belong in the ledger. `code_version` is excluded; see the class
        docstring.
        """
        return "|".join(
            (
                f"target={self.target}",
                f"blend_weight={float(self.blend_weight):.4f}",
                f"ridge={float(self.ridge):.4f}",
                f"half_life={float(self.half_life_days):.2f}",
                f"margin={self.margin_method}",
                f"price_source={self.price_source}",
                f"min_matches={int(self.min_matches)}",
                f"max_ev={float(self.max_ev):.4f}",
            )
        )


def git_revision() -> str:
    """The working tree's revision, or "" when that cannot be determined.

    Best-effort on purpose. A ledger that refused to record a claim because the
    repository was not a git checkout would fail at the one job it has, so an
    unknown revision is stored as an empty string and the row is kept.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

def bet_id(claim: dict, provenance: Provenance) -> str:
    """A stable fingerprint of one claim: this selection, this price, this model.

    **What counts as the same claim is the whole design decision here.**

    In the hash: the fixture, the market, the selection and its line, the price
    and the book it came from, and the model settings. Two scans on consecutive
    days that find the same archived price and the same settings recorded the
    same claim, and must not stake it twice - the second is a repeat sighting,
    handled exactly as `snapshots.write_snapshot` handles one.

    Not in the hash: `as_of`, and the model probability it implies. A scan run
    the next day has seen another day of results and may quote a slightly
    different number for the same bet, but it is still the bet you would have
    struck once. Storing the first sighting and moving `last_seen_at_utc`
    forward keeps one row per bet rather than one per day the bet stayed on the
    board.

    **A moved price is a new claim**, because `price_taken` is in the hash.
    That is intentional: betting into 2.10 and betting into 2.30 are different
    bets and settle differently. It does mean a fixture whose price drifts can
    appear several times, which the summary reports as distinct selections
    alongside rows so the difference is visible rather than silent.

    Prices are rounded to three decimals before hashing, for the reason
    `snapshots.content_hash` gives: the source writes 2.1 and 2.10
    interchangeably and an archive full of formatting noise is worse than
    useless.
    """
    line = claim.get("line")
    line_part = "" if line is None or pd.isna(line) else f"{float(line):.3f}"
    price = float(claim["price_taken"])
    parts = (
        f"fixture={claim['fixture_key']}",
        f"market={claim['market']}",
        f"selection={claim['selection']}",
        f"line={line_part}",
        f"price={price:.3f}",
        f"book={claim.get('book') or ''}",
        provenance.key(),
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def create_tables(con) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BETS_TABLE} (
            bet_id              VARCHAR PRIMARY KEY,
            recorded_at_utc     TIMESTAMP NOT NULL,
            last_seen_at_utc    TIMESTAMP NOT NULL,
            as_of               DATE NOT NULL,
            fixture_key         VARCHAR NOT NULL,
            content_hash        VARCHAR,
            league              VARCHAR NOT NULL,
            fixture_date        DATE NOT NULL,
            kickoff_time        VARCHAR,
            home_team           VARCHAR NOT NULL,
            away_team           VARCHAR NOT NULL,
            market              VARCHAR NOT NULL,
            selection           VARCHAR NOT NULL,
            selection_label     VARCHAR,
            line                DOUBLE,
            price_taken         DOUBLE NOT NULL,
            book                VARCHAR,
            market_probability  DOUBLE,
            market_source       VARCHAR,
            model_probability   DOUBLE NOT NULL,
            push_probability    DOUBLE,
            fair_price          DOUBLE,
            edge                DOUBLE,
            expected_value      DOUBLE,
            stake               DOUBLE NOT NULL,
            staked              BOOLEAN NOT NULL,
            withheld_reason     VARCHAR,
            n_home              INTEGER,
            n_away              INTEGER,
            thin_history        VARCHAR,
            target              VARCHAR NOT NULL,
            blend_weight        DOUBLE,
            ridge               DOUBLE,
            half_life_days      DOUBLE,
            margin_method       VARCHAR,
            price_source        VARCHAR,
            min_matches         INTEGER,
            max_ev              DOUBLE,
            code_version        VARCHAR
        )
        """
    )
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SETTLEMENTS_TABLE} (
            bet_id           VARCHAR PRIMARY KEY,
            settled_at_utc   TIMESTAMP NOT NULL,
            match_id         VARCHAR NOT NULL,
            match_date       DATE,
            home_goals       INTEGER,
            away_goals       INTEGER,
            win_fraction     DOUBLE NOT NULL,
            push_fraction    DOUBLE NOT NULL,
            profit_at_taken  DOUBLE,
            price_close      DOUBLE,
            closing_source   VARCHAR,
            closing_fair     DOUBLE,
            clv              DOUBLE,
            price_movement   DOUBLE
        )
        """
    )


def _has_table(con, name: str) -> bool:
    found = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return found is not None


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

def build_claims(
    scan: pd.DataFrame,
    provenance: Provenance,
    recorded_at_utc: dt.datetime | None = None,
    as_of: dt.date | None = None,
) -> pd.DataFrame:
    """Turn a scan frame into ledger rows, without touching the database.

    Split out from `record` so the shape can be tested, and inspected before
    anything is written, without a connection in the way.

    **Rows the scan could not value are dropped here rather than stored as
    nulls.** A selection with no finite price or no model probability is not a
    bet somebody could have struck, and an unstakeable row in a ledger of
    stakes would be counted by every later query that forgot to exclude it.
    """
    if scan is None or scan.empty:
        return pd.DataFrame(columns=BET_COLUMNS)

    missing = {"fixture_key", "market", "selection", "price_taken"} - set(scan.columns)
    if missing:
        raise ValueError(
            f"The scan frame is missing {sorted(missing)}, which the ledger "
            "needs to identify a claim. `scan_fixtures.scan` supplies them; a "
            "hand-built frame must too."
        )

    stamp = recorded_at_utc or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows: list[dict] = []
    for row in scan.to_dict("records"):
        # **Per-row provenance, because a scan spans several leagues.** The
        # run-level settings come from the `Provenance` argument; the three the
        # fit resolves for itself are read off the row when the scan supplied
        # them. Nothing guarantees five leagues resolve to one ridge, and a
        # ledger that assumed so would file a claim under settings that never
        # priced it.
        prov = provenance
        overrides = {
            field: row[field]
            for field in ("target", "ridge", "half_life_days")
            if row.get(field) is not None and not pd.isna(row.get(field))
        }
        if overrides:
            prov = replace(provenance, **overrides)

        price = row.get("price_taken")
        probability = row.get("model_probability")
        if price is None or not np.isfinite(float(price)) or float(price) <= 1.0:
            continue
        if probability is None or not np.isfinite(float(probability)):
            continue

        # **A claim with no point-in-time boundary has no provenance.** `as_of`
        # is what the fit behind this number was allowed to see, so a row
        # without it cannot be audited later and is refused rather than stored
        # with a guessed or null one.
        boundary = as_of if as_of is not None else row.get("as_of")
        if boundary is None or pd.isna(boundary):
            raise ValueError(
                "Every claim needs an `as_of`: it is the date the model was "
                "fitted up to, and without it the row cannot be checked "
                "against the model that made it. Pass as_of= to build_claims, "
                "or carry an as_of column on the scan frame."
            )

        reason = row.get("withheld_reason") or ""
        claim = {
            "recorded_at_utc": stamp,
            "last_seen_at_utc": stamp,
            "as_of": boundary,
            "fixture_key": row["fixture_key"],
            "content_hash": row.get("content_hash"),
            "league": row["league"],
            "fixture_date": row.get("fixture_date", row.get("date")),
            "kickoff_time": row.get("kickoff_time", row.get("kickoff")),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "market": row["market"],
            "selection": row["selection"],
            "selection_label": row.get("selection_label", row["selection"]),
            "line": row.get("line"),
            "price_taken": float(price),
            "book": row.get("book"),
            "market_probability": row.get("market_probability"),
            "market_source": row.get("market_source"),
            "model_probability": float(probability),
            "push_probability": row.get("push_probability", 0.0),
            "fair_price": row.get("fair_price"),
            "edge": row.get("edge"),
            "expected_value": row.get("expected_value"),
            "stake": FLAT_STAKE,
            # A withheld row is recorded and never counted as a stake. The
            # distinction is the whole reason both are here; see the module
            # docstring and BACKLOG B17.
            "staked": reason == "",
            "withheld_reason": reason,
            "n_home": row.get("n_home"),
            "n_away": row.get("n_away"),
            "thin_history": row.get("thin_history", ""),
            "target": prov.target,
            "blend_weight": prov.blend_weight,
            "ridge": prov.ridge,
            "half_life_days": prov.half_life_days,
            "margin_method": prov.margin_method,
            "price_source": prov.price_source,
            "min_matches": prov.min_matches,
            "max_ev": prov.max_ev,
            "code_version": prov.code_version,
        }
        claim["bet_id"] = bet_id(claim, prov)
        rows.append(claim)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=BET_COLUMNS)
    # One scan can offer the same claim twice only if the archive holds a
    # duplicate price row; keep the first and let the count show it.
    frame = frame.drop_duplicates("bet_id", keep="first")
    return frame[BET_COLUMNS]


def record(con, claims: pd.DataFrame) -> dict:
    """Append what is new; move the sighting stamp on what is not.

    **Nothing here ever updates a claim.** The only column that moves is
    `last_seen_at_utc`, and only forwards, so the ledger can answer "how long
    did this bet stay on the board" without a claim ever being restated. That
    is the same rule `snapshots.write_snapshot` follows and for the same
    reason.

    Returns counts rather than a bare number: "0 new, 47 seen again" is a
    healthy second scan of an afternoon and must not read as a failure. A
    repeat sighting arriving from a different revision of the code is counted
    separately, because that is the one case where an unchanged claim might not
    mean unchanged arithmetic - see BACKLOG B15.
    """
    create_tables(con)
    if claims is None or claims.empty:
        return {"seen": 0, "new": 0, "repeat": 0, "repeat_other_revision": 0}

    con.register("incoming_claims", claims)
    existing = con.execute(
        f"SELECT bet_id, code_version FROM {BETS_TABLE} WHERE bet_id IN "
        "(SELECT bet_id FROM incoming_claims)"
    ).df()
    known = set(existing["bet_id"]) if not existing.empty else set()

    other_revision = 0
    if not existing.empty:
        incoming_versions = dict(zip(claims["bet_id"], claims["code_version"]))
        other_revision = int(
            sum(
                1
                for _, row in existing.iterrows()
                if (row["code_version"] or "")
                != (incoming_versions.get(row["bet_id"]) or "")
            )
        )

    con.execute(
        f"""
        UPDATE {BETS_TABLE} AS t
        SET last_seen_at_utc = GREATEST(
            t.last_seen_at_utc,
            (SELECT MAX(i.last_seen_at_utc) FROM incoming_claims i
             WHERE i.bet_id = t.bet_id)
        )
        WHERE t.bet_id IN (SELECT bet_id FROM incoming_claims)
        """
    )

    fresh = claims[~claims["bet_id"].isin(known)]
    con.register("fresh_claims", fresh)
    columns = ", ".join(f'"{c}"' for c in BET_COLUMNS)
    con.execute(
        f"INSERT INTO {BETS_TABLE} ({columns}) SELECT {columns} FROM fresh_claims"
    )
    for name in ("incoming_claims", "fresh_claims"):
        con.unregister(name)

    return {
        "seen": int(len(claims)),
        "new": int(len(fresh)),
        "repeat": int(len(claims) - len(fresh)),
        "repeat_other_revision": other_revision,
    }


def load_bets(
    con,
    leagues: list[str] | None = None,
    settled: bool | None = None,
    staked_only: bool = False,
) -> pd.DataFrame:
    """The ledger, with each bet's outcome attached when there is one.

    `settled=None` returns everything, `True` only bets with an outcome, and
    `False` only those still open. The outcome columns are NULL on an open bet
    rather than absent, so one frame shape serves every caller.
    """
    # **Never creates a table.** The app reads this through a read-only
    # connection, and DuckDB refuses even `CREATE TABLE IF NOT EXISTS` on one -
    # so a reader that tried to be helpful here would take the whole page down
    # on a database that simply has no ledger yet. Both tables are made
    # together by `create_tables`, so either one missing means neither exists.
    if not _has_table(con, BETS_TABLE) or not _has_table(con, SETTLEMENTS_TABLE):
        return pd.DataFrame(columns=BET_COLUMNS + SETTLEMENT_COLUMNS[1:])

    outcome_columns = ", ".join(f"s.{c}" for c in SETTLEMENT_COLUMNS[1:])
    where, params = [], []
    if leagues:
        where.append(f"b.league IN ({', '.join('?' for _ in leagues)})")
        params += list(leagues)
    if settled is True:
        where.append("s.bet_id IS NOT NULL")
    elif settled is False:
        where.append("s.bet_id IS NULL")
    if staked_only:
        where.append("b.staked")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    return con.execute(
        f"""
        SELECT b.*, {outcome_columns}
        FROM {BETS_TABLE} b
        LEFT JOIN {SETTLEMENTS_TABLE} s USING (bet_id)
        {clause}
        ORDER BY b.fixture_date, b.fixture_key, b.market, b.selection
        """,
        params,
    ).df()


# --------------------------------------------------------------------------
# Settling
# --------------------------------------------------------------------------

def settle_open(
    con,
    tolerance_days: int | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Settle every open bet whose match has since been played.

    **This never restates a settled bet.** A bet already in
    `paper_settlements` is skipped whatever this run would have computed for
    it, which is the schema-level version of the promise in the module
    docstring. If a settlement is ever genuinely wrong the row has to be
    deleted deliberately, and that should be as awkward as it sounds.

    The join to a played match reuses `snapshots.reconcile`'s reasoning rather
    than its code: match on league and both canonical team names, allow the
    fixture date to drift by a day for a postponement or a late kick-off, and
    prefer the exact date when several are in range. A bet that cannot be
    joined stays open and is counted, never dropped - an unsettled bet and a
    bet that silently failed to join look identical otherwise, and only one of
    them is a reason to go looking for a bug.

    Closing line value is computed exactly as `backtest` computes it -
    `price_taken * closing_fair - 1` against the margin-free closing
    probability, using `backtest._market_probabilities` so that one piece of
    code produces both numbers. A ledger whose CLV was computed by a second
    implementation would be comparing itself to the backtest across a
    difference nobody could see.
    """
    create_tables(con)
    window = (
        config.FIXTURE_RECONCILE_TOLERANCE_DAYS
        if tolerance_days is None else tolerance_days
    )
    stamp = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

    # Every caller can rely on the same keys being present whichever path is
    # taken, so a report never has to guard each lookup individually.
    base = {"open": 0, "settled": 0, "unmatched": 0, "unsettleable": 0,
            "no_closing_price": 0, "awaiting_kickoff": 0, "awaiting_results": 0,
            "unmatched_unexpected": 0, "unmatched_rows": pd.DataFrame()}

    open_bets = load_bets(con, settled=False)
    if open_bets.empty:
        return base

    matches = con.execute(
        "SELECT * FROM matches WHERE home_goals IS NOT NULL"
    ).df()
    if matches.empty:
        return {
            **base,
            "open": int(len(open_bets)),
            "unmatched": int(len(open_bets)),
            "unmatched_rows": open_bets,
            **_classify_unmatched(open_bets, matches, stamp.date()),
        }

    joined = _join_to_matches(open_bets, matches, window)
    matched = joined[joined["match_id"].notna()]
    unmatched = joined[joined["match_id"].isna()]

    rows, unsettleable, no_close = [], 0, 0
    for match_id, block in matched.groupby("match_id"):
        match = matches[matches["match_id"] == match_id].iloc[0]
        odds = con.execute(
            "SELECT * FROM odds WHERE match_id = ?", [match_id]
        ).df()
        # Every bet on this match shares one margin-free closing line, so it is
        # built once per match rather than once per bet.
        method = str(block["margin_method"].iloc[0])
        closing = (
            backtest._market_probabilities(odds, method, fallback=True)
            if not odds.empty else {}
        )
        inputs = _settlement_inputs(match)

        for bet in block.to_dict("records"):
            line = None if pd.isna(bet["line"]) else float(bet["line"])
            try:
                outcome = settlement.settle(
                    bet["market"], bet["selection"], line,
                    int(match["home_goals"]), int(match["away_goals"]), **inputs,
                )
            except ValueError:
                # A market withdrawn since the claim was recorded, or one with
                # no settlement rule. Left open and counted; see B16, where a
                # withdrawn market raises rather than failing to match.
                unsettleable += 1
                continue
            if outcome is None:
                unsettleable += 1
                continue

            closing_fair, closing_book = closing.get(
                (bet["market"], bet["selection"], backtest._line_key(line)),
                (np.nan, None),
            )
            price_close, closed_at = (
                backtest._pick_price(
                    odds[
                        (odds["market"] == bet["market"])
                        & (odds["selection"] == bet["selection"])
                        & (_line_matches(odds["line"], line))
                    ],
                    "close",
                    backtest.FAIR_LINE_PREFERENCE,
                )
                if not odds.empty else (np.nan, None)
            )
            same_book_close = _same_book_close(odds, bet, line)
            if not np.isfinite(closing_fair):
                no_close += 1

            price_taken = float(bet["price_taken"])
            rows.append(
                {
                    "bet_id": bet["bet_id"],
                    "settled_at_utc": stamp,
                    "match_id": match_id,
                    "match_date": match["date"],
                    "home_goals": int(match["home_goals"]),
                    "away_goals": int(match["away_goals"]),
                    "win_fraction": outcome.win,
                    "push_fraction": outcome.push,
                    "profit_at_taken": outcome.profit(price_taken),
                    "price_close": price_close,
                    "closing_source": closed_at,
                    "closing_fair": closing_fair,
                    "clv": (
                        price_taken * closing_fair - 1.0
                        if np.isfinite(closing_fair) else np.nan
                    ),
                    "price_movement": (
                        price_taken / same_book_close - 1.0
                        if np.isfinite(same_book_close) and same_book_close > 0
                        else np.nan
                    ),
                }
            )

    written = pd.DataFrame(rows, columns=SETTLEMENT_COLUMNS)
    if not written.empty:
        con.register("new_settlements", written)
        columns = ", ".join(f'"{c}"' for c in SETTLEMENT_COLUMNS)
        con.execute(
            f"INSERT INTO {SETTLEMENTS_TABLE} ({columns}) "
            f"SELECT {columns} FROM new_settlements "
            f"WHERE bet_id NOT IN (SELECT bet_id FROM {SETTLEMENTS_TABLE})"
        )
        con.unregister("new_settlements")

    reasons = _classify_unmatched(unmatched, matches, stamp.date())
    return {
        **base,
        "open": int(len(open_bets)),
        "settled": int(len(written)),
        "unmatched": int(len(unmatched)),
        "unsettleable": int(unsettleable),
        "no_closing_price": int(no_close),
        "unmatched_rows": unmatched,
        **reasons,
    }


def _classify_unmatched(unmatched: pd.DataFrame, matches: pd.DataFrame,
                        today: dt.date) -> dict:
    """Why each unsettled bet did not settle, split into three different things.

    **A bet that did not settle has three possible reasons and only one of them
    is worth acting on.** Reported as a single "unmatched" count they are
    indistinguishable, which is the shape of bug this project keeps paying for:
    the fixture has not kicked off yet; it has been played but the result is
    not in the database, because `matches` only advances when
    `build_database.py` re-downloads the current season; or it was played, the
    database holds later matches for that league, and the bet still found no
    join - which is a real defect, usually a club name or a postponement beyond
    the date tolerance.

    Without this split, somebody running the settle step for a fortnight would
    see "0 settled" every time and have no way to tell a working ledger from a
    stale database from a broken join. The first is normal, the second is one
    command, and the third is a bug hunt.
    """
    empty = {"awaiting_kickoff": 0, "awaiting_results": 0, "unmatched_unexpected": 0}
    if unmatched.empty:
        return empty

    # The latest *result* on file per league. A league absent from this map has
    # no played matches at all, so nothing there can have been ingested yet.
    latest = (
        matches.assign(date=pd.to_datetime(matches["date"]))
        .groupby("league")["date"].max().dt.date.to_dict()
    )

    counts = dict(empty)
    for row in unmatched.itertuples():
        fixture_date = pd.Timestamp(row.fixture_date).date()
        if fixture_date >= today:
            counts["awaiting_kickoff"] += 1
            continue
        newest = latest.get(row.league)
        if newest is None or newest < fixture_date:
            counts["awaiting_results"] += 1
        else:
            counts["unmatched_unexpected"] += 1
    return counts


def _join_to_matches(bets: pd.DataFrame, matches: pd.DataFrame,
                     window: int) -> pd.DataFrame:
    """Attach a played match to each bet, tolerating a day of date drift."""
    left = bets.copy()
    left["fixture_date"] = pd.to_datetime(left["fixture_date"])
    right = matches[["match_id", "league", "date", "home_team", "away_team"]].copy()
    right["date"] = pd.to_datetime(right["date"])

    merged = left.drop(columns=["match_id"], errors="ignore").merge(
        right, on=["league", "home_team", "away_team"], how="left",
        suffixes=("", "_played"),
    )
    drift = (merged["date"] - merged["fixture_date"]).dt.days.abs()
    merged["drift"] = drift
    # Out-of-range candidates lose their match rather than the bet: the row
    # survives with a null match_id and is reported as unmatched.
    merged.loc[drift.isna() | (drift > window), "match_id"] = None
    merged = merged.sort_values("drift", na_position="last")
    return merged.drop_duplicates("bet_id", keep="first")


def _settlement_inputs(match) -> dict:
    """Everything `settlement.settle` might need, from one played match.

    Mirrors `backtest._settlement_inputs`, which takes an itertuples row rather
    than a mapping. Kept as a small duplicate instead of generalising that one:
    the shared piece would be four lines and the coupling would run from a
    ledger into the backtest's hot loop.
    """
    def value(name):
        raw = match.get(name) if hasattr(match, "get") else getattr(match, name, None)
        return None if raw is None or pd.isna(raw) else float(raw)

    return {
        "total_corners": value("total_corners"),
        "total_cards": value("total_cards"),
        "home_corners": value("home_corners"),
        "away_corners": value("away_corners"),
        "home_cards": value("home_cards"),
        "away_cards": value("away_cards"),
        "home_goals_ht": value("home_goals_ht"),
        "away_goals_ht": value("away_goals_ht"),
    }


def _line_matches(column: pd.Series, line: float | None) -> pd.Series:
    if line is None:
        return column.isna()
    return (column - line).abs() < 1e-9


def _same_book_close(odds: pd.DataFrame, bet: dict, line: float | None) -> float:
    """The closing price at the same bookmaker the price was taken from.

    Returns NaN when that book did not price the close, which is common and is
    why raw price movement is a secondary check rather than the headline.
    """
    if odds.empty or not bet.get("book"):
        return np.nan
    subset = odds[
        (odds["market"] == bet["market"])
        & (odds["selection"] == bet["selection"])
        & (odds["phase"] == "close")
        & (odds["bookmaker"] == bet["book"])
        & (_line_matches(odds["line"], line))
    ]
    return float(subset["price"].iloc[0]) if not subset.empty else np.nan


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------

def summary(con, leagues: list[str] | None = None) -> dict:
    """What the ledger currently says, with the honest caveats attached.

    **Closing line value is the headline and profit is not.** Over a few
    hundred bets return on investment is close to noise while CLV is
    measurable in weeks, which is the project's own standing rule and the
    reason the gate is phrased in terms of CLV. Both are returned; the caller
    is expected to print them in that order.

    Standard errors are clustered by match through `evaluation.clustered_mean`,
    because several selections on one fixture share a model fit and one
    closing-line move. On a thin ledger that correction is the difference
    between a two-sigma result and nothing, which is exactly the size of claim
    a young ledger is capable of manufacturing.
    """
    from . import evaluation

    frame = load_bets(con, leagues=leagues)
    if frame.empty:
        return {"bets": 0, "staked": 0, "withheld": 0, "settled": 0}

    staked = frame[frame["staked"]]
    settled = staked[staked["match_id"].notna()]
    out = {
        "bets": int(len(frame)),
        "staked": int(len(staked)),
        "withheld": int((~frame["staked"]).sum()),
        "settled": int(len(settled)),
        "open": int(len(staked) - len(settled)),
        "distinct_selections": int(
            frame.drop_duplicates(
                ["fixture_key", "market", "selection", "line"]
            ).shape[0]
        ),
        "first_recorded": frame["recorded_at_utc"].min(),
        "last_recorded": frame["recorded_at_utc"].max(),
    }
    if settled.empty:
        return out

    clv = settled.dropna(subset=["clv"])
    if not clv.empty:
        stats = evaluation.clustered_mean(clv["clv"], clv["match_id"])
        out.update(
            {
                "n_clv": stats["n"],
                "n_matches": stats["n_clusters"],
                "mean_clv": stats["mean"],
                "clv_se": stats["se"],
                "beat_close_rate": float((clv["clv"] > 0).mean()),
            }
        )
    profit = settled.dropna(subset=["profit_at_taken"])
    if not profit.empty:
        out.update(
            {
                "n_profit": int(len(profit)),
                "total_staked": float(profit["stake"].sum()),
                "profit": float(profit["profit_at_taken"].sum()),
                "roi": float(
                    profit["profit_at_taken"].sum() / profit["stake"].sum()
                ),
            }
        )
    return out


def withheld_comparison(con, leagues: list[str] | None = None) -> pd.DataFrame:
    """Settled results for the rows the scan ranked against the ones it withheld.

    **This is the ledger marking BACKLOG B17's homework.** The two withholding
    thresholds were set from a backtest, and whether they are right going
    forward is answerable only by settling the suppressed rows alongside the
    kept ones. One row per group, with the same clustered standard error, so a
    difference has to survive the correction before it means anything.

    It will say nothing for weeks, and a difference on a few dozen bets means
    nothing whatever the arithmetic reports. That is a property of the
    question, not a defect in the answer.
    """
    from . import evaluation

    frame = load_bets(con, leagues=leagues, settled=True)
    if frame.empty:
        return pd.DataFrame(
            columns=["group", "n", "n_matches", "mean_ev", "mean_clv", "clv_se", "roi"]
        )

    rows = []
    for name, block in (
        ("ranked", frame[frame["staked"]]),
        ("withheld", frame[~frame["staked"]]),
    ):
        clv = block.dropna(subset=["clv"])
        stats = (
            evaluation.clustered_mean(clv["clv"], clv["match_id"])
            if not clv.empty else {"n": 0, "n_clusters": 0, "mean": np.nan, "se": np.nan}
        )
        staked_sum = block["stake"].sum()
        rows.append(
            {
                "group": name,
                "n": int(len(block)),
                "n_matches": stats["n_clusters"],
                "mean_ev": float(block["expected_value"].mean()) if len(block) else np.nan,
                "mean_clv": stats["mean"],
                "clv_se": stats["se"],
                "roi": (
                    float(block["profit_at_taken"].sum() / staked_sum)
                    if staked_sum > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
