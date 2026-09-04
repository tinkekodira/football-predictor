"""Tests for the weekly routine.

**The point of this file is the failure behaviour, not the happy path.** The
whole reason `weekly.py` exists is to be scheduled and forgotten, and a chained
job that is run unattended has exactly two ways to betray you: it stops early
and silently, or it does something irreversible on bad input. Both are pinned
here.

The steps themselves are covered elsewhere - snapshots in `test_snapshots.py`,
the ledger in `test_ledger.py`, the scan in `test_evidence.py`. What is tested
here is the wiring: the order, what survives a failure, and what refuses.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import config, database, ledger  # noqa: E402
from scripts import weekly  # noqa: E402


@pytest.fixture
def run() -> weekly.Run:
    return weekly.Run()


class _Args:
    """Just enough of the parsed namespace for the step functions."""

    def __init__(self, **kwargs):
        self.league = None
        self.allow_stale = False
        self.skip_results = False
        self.half_life = 180.0
        self.ridge = None
        self.min_matches = config.SCAN_MIN_TEAM_MATCHES
        self.max_ev = config.SCAN_MAX_TRUSTED_EV
        self.__dict__.update(kwargs)


# --------------------------------------------------------------------------
# What a failure does, and does not, stop
# --------------------------------------------------------------------------

def test_a_failed_archive_does_not_stop_the_rest_of_the_run(monkeypatch, run, capsys):
    """The unrecoverable step failing is a reason to shout, not to give up.

    Everything after the archive works on inputs that can be downloaded again
    tomorrow, so abandoning them because the network blipped would turn one
    lost thing into four.
    """
    def explode(*args, **kwargs):
        raise ConnectionError("the source is down")

    monkeypatch.setattr(weekly.snapshots, "download", explode)
    stale = weekly._archive(None, run)

    assert stale is False
    assert run.steps[0].status == "failed"
    assert "ConnectionError" in run.steps[0].error
    assert "Continuing with the rest" in capsys.readouterr().out


def test_a_failed_step_is_reported_and_sets_the_exit_code(run):
    """An unattended job that fails quietly is worse than one that fails."""
    run.add("first").status = "ok"
    step = run.add("second")
    step.status, step.error = "failed", "ValueError: nope"

    assert run.failed
    assert not step.ok


def test_a_run_where_everything_worked_does_not_report_failure(run):
    for name in ("one", "two"):
        run.add(name).status = "ok"
    run.add("three").status = "skipped"
    assert not run.failed


# --------------------------------------------------------------------------
# The refusal that matters
# --------------------------------------------------------------------------

def test_a_stale_price_file_is_archived_but_never_recorded(run, capsys):
    """Two different decisions on the same input, and both are deliberate.

    A stale copy still holds the only surviving record of the prices in it, so
    it is stored. But pricing a board from it would file confident claims on
    matches that may already have kicked off - and unlike a bad scan, which
    scrolls away, a ledger entry is permanent.
    """
    weekly._record(None, run, _Args(), stale=True)

    step = run.steps[0]
    assert step.status == "skipped"
    assert "stale" in step.detail
    out = capsys.readouterr().out
    assert "NOT recording" in out
    assert "--allow-stale" in out


def test_allow_stale_overrides_the_refusal(monkeypatch, run):
    """The override exists, and reaching the scan is what proves it works."""
    def no_board(*args, **kwargs):
        return pd.DataFrame(), ["nothing archived"]

    monkeypatch.setattr(weekly.scan_fixtures, "scan", no_board)
    weekly._record(None, run, _Args(allow_stale=True), stale=True)

    assert run.steps[0].status == "ok"
    assert run.steps[0].detail == "nothing to price"


def test_skipping_the_results_refresh_is_recorded_as_a_skip_not_a_pass(run):
    """A step that did not run must not read as a step that succeeded."""
    weekly._refresh_results(None, run, skip=True)
    step = run.steps[0]
    assert step.status == "skipped"
    assert step.detail == "--skip-results"


# --------------------------------------------------------------------------
# Reading the result
# --------------------------------------------------------------------------

def test_the_summary_names_every_step_and_its_outcome(run, capsys):
    """The block a person actually reads after an unattended run."""
    run.add("Archive prices").status = "ok"
    run.steps[0].detail = "47 fixtures seen"
    broken = run.add("Settle")
    broken.status, broken.error = "failed", "IOError: locked"

    weekly._summarise(run, dt.datetime.now())
    out = capsys.readouterr().out

    assert "Archive prices" in out and "47 fixtures seen" in out
    assert "FAILED" in out and "IOError: locked" in out
    assert "safe to repeat" in out


def test_the_dry_run_writes_nothing_and_says_so(capsys):
    assert weekly._dry_run(_Args()) == 0
    out = capsys.readouterr().out
    assert "Nothing will be downloaded, written or recorded" in out
    assert "[1/4]" in out and "[4/4]" in out


def test_a_missing_database_is_refused_before_anything_runs(monkeypatch, capsys, tmp_path):
    """The one precondition worth checking before touching the network."""
    monkeypatch.setattr(
        sys, "argv",
        ["weekly.py", "--db", str(tmp_path / "absent.duckdb")],
    )
    assert weekly.main() == 1
    assert "No database at" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Repeatability, which is what makes it safe to schedule
# --------------------------------------------------------------------------

def test_settling_twice_changes_nothing_the_second_time(tmp_path, run):
    """Every step here is safe to repeat, and the settle step is the one that
    would be most damaging if it were not."""
    path = tmp_path / "weekly.duckdb"
    con = database.connect(path)
    ledger.create_tables(con)

    weekly._settle(con, run)
    weekly._settle(con, run)

    assert [s.status for s in run.steps] == ["ok", "ok"]
    assert ledger.summary(con)["bets"] == 0
    con.close()


def test_a_run_against_a_scratch_database_does_not_export_over_the_backup(
    monkeypatch, run, capsys
):
    """`snapshots.export` writes to a fixed tracked location, whatever database
    it was handed.

    So a run against a copy - which is how this gets tested - would overwrite
    the real backup of the one table that cannot be rebuilt, with a scratch
    database's contents. Skipped rather than redirected: a test run has no
    business producing a backup at all.
    """
    calls = []
    monkeypatch.setattr(weekly.snapshots, "download",
                        lambda *a, **k: (pd.DataFrame(), Path("fixtures.csv")))
    monkeypatch.setattr(weekly.snapshots, "build_snapshot",
                        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(weekly.snapshots, "staleness",
                        lambda *a, **k: {"stale": False, "reasons": []})
    monkeypatch.setattr(weekly.snapshots, "write_snapshot",
                        lambda *a, **k: {"fixtures_seen": 0, "new_snapshots": 0,
                                         "repeat_snapshots": 0, "new_odds_rows": 0})
    monkeypatch.setattr(weekly.snapshots, "reconcile", lambda *a, **k: {})
    monkeypatch.setattr(weekly.snapshots, "export",
                        lambda *a, **k: calls.append(True))

    weekly._archive(None, run, export=False)
    assert calls == []
    assert "Not exporting" in capsys.readouterr().out

    weekly._archive(None, run, export=True)
    assert calls == [True]


def test_the_standing_claim_notice_names_the_revision_rather_than_alarming():
    """A warning that fires on every run after every commit is one nobody reads.

    `code_version` is outside the identity hash on purpose, so after any commit
    every standing claim is a repeat from a different revision. The count alone
    is therefore never actionable; which revision is.
    """
    source = Path(weekly.__file__).read_text(encoding="utf-8")
    assert "prior_revisions" in source
    assert "were first recorded under" in source
