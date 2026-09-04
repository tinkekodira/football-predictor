"""The whole weekly routine in one command: archive, record, settle, report.

    python scripts/weekly.py
    python scripts/weekly.py --dry-run        # say what it would do, touch nothing
    python scripts/weekly.py --skip-results   # no results download

**Run this twice a week, and the ledger looks after itself.** The source
collects prices on Friday afternoons no later than 17:00 British time for
weekend fixtures and Tuesdays no later than 13:00 for midweek ones, and
overwrites its file each time. A Friday missed is a week of forward evidence
that cannot be recovered afterwards, because nothing anywhere republishes last
Friday's prices.

Four steps, and **the order is a risk ordering rather than a logical one**:

1. **Archive the price file.** First, always, and its failure never stops the
   rest. It is the only step whose input disappears - every other input here
   can be downloaded again tomorrow.
2. **Refresh this season's results.** Needed twice over: the models that price
   tomorrow's board should have seen last night's matches, and a bet cannot
   settle against a result the database does not hold.
3. **Record the board.** Files every valued selection in the paper ledger,
   withheld rows included and flagged unstaked.
4. **Settle and report.** Joins open bets to matches that have since been
   played, then prints the running record.

**Every step is safe to repeat.** Snapshots dedupe by content hash, claims by
their identity hash, and a settled bet is never restated. Running this twice in
an afternoon stores nothing extra and changes no number, which is what makes it
safe to schedule and forget.

**It refuses to record from a stale file, and archives it anyway.** Those are
different decisions on purpose. A stale copy still holds the only surviving
record of the prices in it, so it is stored; but pricing a board from it would
file confident claims on matches that have already kicked off, and in a ledger
that is a permanent lie rather than a bad afternoon. `--allow-stale` overrides,
and should not be needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import (  # noqa: E402
    backtest, config, database, ingest, ledger, normalize, snapshots,
)
from fbedge.models import base  # noqa: E402

from scripts import paper_trade, scan_fixtures  # noqa: E402


@dataclass
class Step:
    """One stage of the run, and what became of it.

    Carried rather than printed as it goes, because the useful thing at the end
    of an unattended run is a single block saying which stages did what - not
    four hundred lines of scrolled output with one failure somewhere inside.
    """

    name: str
    status: str = "pending"      # ok | failed | skipped | pending
    detail: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")


@dataclass
class Run:
    steps: list[Step] = field(default_factory=list)

    def add(self, name: str) -> Step:
        step = Step(name)
        self.steps.append(step)
        return step

    @property
    def failed(self) -> bool:
        return any(step.status == "failed" for step in self.steps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), action="append",
                        help="Restrict the scan; the archive always covers all five.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen and write nothing.")
    parser.add_argument("--skip-results", action="store_true",
                        help="Do not re-download this season's results. The "
                             "settle step then has nothing new to join to.")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Record a board even from a file that looks stale. "
                             "You are unlikely to want this.")
    parser.add_argument("--min-matches", type=int,
                        default=config.SCAN_MIN_TEAM_MATCHES)
    parser.add_argument("--max-ev", type=float, default=config.SCAN_MAX_TRUSTED_EV)
    parser.add_argument("--half-life", type=float, default=base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    started = dt.datetime.now()
    print("=" * 78)
    print(f"  Weekly routine   |   {started:%Y-%m-%d %H:%M}")
    print("=" * 78)

    if args.dry_run:
        return _dry_run(args)

    # **Checked once, up front.** DuckDB allows one writer, so with the app
    # open every one of the four steps would fail in turn and print its own
    # traceback. One clear refusal is better than four confusing ones.
    try:
        con = database.connect(args.db)
    except Exception as error:
        if "another process" not in str(error):
            raise
        print(
            "\nThe database is open in another process and every step here "
            "needs to write to it.\n\nStop the app (`streamlit run app.py`) "
            "and run this again. To read the ledger without changing "
            "anything, use:\n"
            "    python scripts/paper_trade.py --no-settle"
        )
        return 1

    run = Run()
    try:
        # Resolved, so that a path spelled differently but pointing at the
        # real database still counts as the real database.
        is_real_db = args.db.resolve() == Path(config.DB_PATH).resolve()
        stale = _archive(con, run, export=is_real_db)
        _refresh_results(con, run, skip=args.skip_results)
        _record(con, run, args, stale=stale)
        _settle(con, run)
    finally:
        con.close()

    _summarise(run, started)

    # Reporting reopens read-only so the summary is readable even if the
    # database has since been picked up by the app.
    print()
    read = database.connect(args.db, read_only=True)
    try:
        paper_trade.report(read, args.league or None)
    finally:
        read.close()

    return 1 if run.failed else 0


# --------------------------------------------------------------------------
# The four steps
# --------------------------------------------------------------------------

def _archive(con, run: Run, export: bool = True) -> bool:
    """Step 1: store this pull of the price file. Returns whether it looked stale.

    **Its failure is caught and never stops the run.** The steps after it are
    all recoverable; this one is not, so if it breaks the right response is to
    say so loudly and carry on doing the recoverable work rather than abandon
    the afternoon.
    """
    step = run.add("Archive prices")
    print("\n[1/4] Archiving the upcoming-fixtures file...")
    try:
        frame, path = snapshots.download()
        snapshot, odds = snapshots.build_snapshot(frame, leagues=list(config.LEAGUES))
        report = snapshots.staleness(snapshot, path)

        counts = snapshots.write_snapshot(con, snapshot, odds)
        snapshots.reconcile(con)
        # **Exported only from the real database.** The mirror in
        # `data/snapshots/` is the tracked backup of the one table that cannot
        # be rebuilt, and `snapshots.export` writes to that fixed location
        # whatever database it was handed - so a run against a scratch copy
        # would quietly overwrite the real backup with a test database's
        # contents. Skipped rather than redirected, because a test run has no
        # business producing a backup at all.
        if export:
            snapshots.export(con)
        else:
            print("      Not exporting: --db is not the configured database.")

        step.status = "ok"
        step.detail = (
            f"{counts['fixtures_seen']} fixtures seen, "
            f"{counts['new_snapshots']} new, "
            f"{counts['repeat_snapshots']} already held"
        )
        print(f"      {step.detail}.")
        if counts["new_snapshots"] == 0 and counts["fixtures_seen"]:
            print("      Nothing new is the normal result between collections.")
        if report["stale"]:
            print("      WARNING: this file looks stale.")
            for reason in report["reasons"]:
                print(f"        - {reason}")
            print("      Archived anyway - the rows are still the only copy.")
        return bool(report["stale"])
    except Exception as error:  # noqa: BLE001 - an unrecoverable step, reported
        step.status, step.error = "failed", f"{type(error).__name__}: {error}"
        print(f"      FAILED: {step.error}")
        print("      This is the one step whose input cannot be re-fetched "
              "later. Continuing with the rest.")
        traceback.print_exc(limit=3)
        return False


def _refresh_results(con, run: Run, skip: bool) -> None:
    """Step 2: pull this season's results so fits and settlement are current."""
    step = run.add("Refresh results")
    if skip:
        step.status, step.detail = "skipped", "--skip-results"
        print("\n[2/4] Skipping the results refresh (--skip-results).")
        return

    print("\n[2/4] Refreshing this season's results...")
    try:
        year = config.CURRENT_SEASON_START_YEAR
        files = ingest.download_all(leagues=list(config.LEAGUES), years=[year])
        matches = odds_rows = 0
        for f in files:
            raw = ingest.read_raw_csv(f.path)
            frame, odds = normalize.normalize_league_season(
                raw, f.league, f.season_start_year
            )
            matches += database.load_matches(con, frame)
            odds_rows += database.load_odds(con, odds)
            database.log_ingest(con, f.league, f.season_start_year,
                                len(frame), len(odds))
        step.status = "ok"
        step.detail = f"{matches} matches, {odds_rows} odds rows"
        print(f"      {config.season_label(year)}: {step.detail}.")
    except Exception as error:  # noqa: BLE001
        step.status, step.error = "failed", f"{type(error).__name__}: {error}"
        print(f"      FAILED: {step.error}")
        print("      Settlement will find nothing new, but the archive is safe.")


def _record(con, run: Run, args, stale: bool) -> None:
    """Step 3: price the archived board and file every claim."""
    step = run.add("Record claims")
    if stale and not args.allow_stale:
        step.status = "skipped"
        step.detail = "the price file looked stale"
        print("\n[3/4] NOT recording: the price file looked stale.")
        print("      Pricing a stale board files confident claims on matches "
              "that may already have kicked off, and a ledger entry is "
              "permanent. Pass --allow-stale to override.")
        return

    print("\n[3/4] Pricing the board and recording claims...")
    try:
        leagues = args.league or list(config.LEAGUES)
        as_of = dt.date.today()
        price_source = backtest.DEFAULT_PRICE_SOURCE
        frame, notes = scan_fixtures.scan(
            con, as_of, leagues, price_source,
            half_life_days=args.half_life, ridge=args.ridge,
            min_matches=args.min_matches, max_ev=args.max_ev,
        )
        if frame.empty:
            step.status, step.detail = "ok", "nothing to price"
            print("      Nothing to price.")
            for note in dict.fromkeys(notes):
                print(f"        - {note}")
            return

        provenance = ledger.Provenance(
            target=base.DEFAULT_TARGET,
            blend_weight=base.DEFAULT_BLEND_WEIGHT,
            ridge=float("nan") if args.ridge is None else float(args.ridge),
            half_life_days=float(args.half_life),
            margin_method="shin",
            price_source=",".join(price_source),
            min_matches=int(args.min_matches),
            max_ev=float(args.max_ev),
            code_version=ledger.git_revision(),
        )
        counts = ledger.record(
            con, ledger.build_claims(frame, provenance, as_of=as_of)
        )
        step.status = "ok"
        step.detail = f"{counts['new']} new, {counts['repeat']} already held"
        print(f"      {step.detail}.")
        if counts["repeat_other_revision"]:
            where = ", ".join(counts["prior_revisions"]) or "an unknown revision"
            print(
                f"      {counts['repeat_other_revision']} standing claim(s) "
                f"were first recorded under {where}."
            )
    except Exception as error:  # noqa: BLE001
        step.status, step.error = "failed", f"{type(error).__name__}: {error}"
        print(f"      FAILED: {step.error}")
        traceback.print_exc(limit=3)


def _settle(con, run: Run) -> None:
    """Step 4: join open bets to results and settle what can be settled."""
    step = run.add("Settle")
    print("\n[4/4] Settling bets whose matches have been played...")
    try:
        counts = ledger.settle_open(con)
        step.status = "ok"
        step.detail = f"{counts['settled']} settled of {counts['open']} open"
        print(f"      {step.detail}.")
        if counts["awaiting_kickoff"]:
            print(f"      {counts['awaiting_kickoff']} waiting on a kick-off, "
                  "which needs nothing.")
        if counts["awaiting_results"]:
            print(f"      {counts['awaiting_results']} played but not yet in the "
                  "database. Step 2 is what fills that in.")
        if counts["unmatched_unexpected"]:
            print(f"      {counts['unmatched_unexpected']} PLAYED AND STILL "
                  "UNMATCHED - these should have joined and did not. Usually a "
                  "club name the two sources spell differently.")
    except Exception as error:  # noqa: BLE001
        step.status, step.error = "failed", f"{type(error).__name__}: {error}"
        print(f"      FAILED: {step.error}")
        traceback.print_exc(limit=3)


# --------------------------------------------------------------------------
# Saying what happened
# --------------------------------------------------------------------------

def _summarise(run: Run, started: dt.datetime) -> None:
    """One block at the end saying which steps did what.

    An unattended run is read afterwards, if at all, and a failure buried in
    the middle of four hundred lines is a failure nobody sees.
    """
    elapsed = (dt.datetime.now() - started).total_seconds()
    print()
    print("=" * 78)
    print(f"  Summary   |   {elapsed:.0f}s")
    print("=" * 78)
    marks = {"ok": "ok    ", "failed": "FAILED", "skipped": "skip  ",
             "pending": "?     "}
    for step in run.steps:
        line = f"  {marks.get(step.status, '?'):8}{step.name:<20}{step.detail}"
        print(line.rstrip())
        if step.error:
            print(f"          {step.error}")
    if run.failed:
        print(
            "\n  At least one step failed. Nothing recorded is wrong because of "
            "it - every step here is safe to repeat - so fix the cause and run "
            "this again."
        )


def _dry_run(args) -> int:
    """Say what would happen, touch nothing."""
    print("\nDry run. Nothing will be downloaded, written or recorded.\n")
    steps = [
        ("Archive prices",
         "download fixtures.csv, store new snapshots, reconcile, export to CSV"),
        ("Refresh results",
         "skipped (--skip-results)" if args.skip_results
         else f"re-download {config.season_label(config.CURRENT_SEASON_START_YEAR)} "
              "for all five leagues"),
        ("Record claims",
         f"price the archived board for "
         f"{', '.join(args.league or sorted(config.LEAGUES))} and file every "
         f"selection (withheld ones flagged unstaked)"),
        ("Settle", "join open bets to played matches, then print the record"),
    ]
    for i, (name, what) in enumerate(steps, start=1):
        print(f"  [{i}/4] {name:<18} {what}")
    print(
        "\n  Every step is safe to repeat: snapshots dedupe by content hash, "
        "claims\n  by their identity hash, and a settled bet is never "
        "restated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
