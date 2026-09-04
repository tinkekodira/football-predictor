"""Tests for the paper-trading ledger.

Three groups, and they guard different kinds of failure.

The **identity** tests are the ones that matter most. A ledger that recorded
the same claim twice would inflate every count it reports, and one that
recorded a genuinely different claim as a repeat would silently lose it. Both
failures look like a working ledger from the outside, which is why the rule
about what is and is not part of a claim's identity is pinned here rather than
left to the docstring.

The **immutability** tests exist because the whole value of this table is that
it cannot be rewritten after the result is known. A claim that could be
restated once the match had been played would be worth nothing at all, and
would still pass every arithmetic test in this file.

The **settlement** tests check the arithmetic against the backtest's own,
deliberately: a ledger whose closing line value was computed by a second
implementation would be comparing itself to the backtest across a difference
nobody could see.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import (  # noqa: E402
    backtest, config, database, ledger, normalize, snapshots,
)
from scripts.make_sample_data import build_season  # noqa: E402


PROVENANCE = ledger.Provenance(
    target="blend",
    blend_weight=0.5,
    ridge=1.0,
    half_life_days=180.0,
    margin_method="shin",
    price_source="market_max",
    min_matches=5,
    max_ev=0.20,
    code_version="abc1234",
)


@pytest.fixture(scope="module")
def ledger_db(tmp_path_factory):
    """Two synthetic seasons of played matches with prices attached."""
    path = tmp_path_factory.mktemp("ledger") / "ledger.duckdb"
    con = database.connect(path)
    for year in (
        config.CURRENT_SEASON_START_YEAR - 2,
        config.CURRENT_SEASON_START_YEAR - 1,
    ):
        raw = build_season("E0", year, seed=year * 17)
        matches, odds = normalize.normalize_league_season(raw, "E0", year)
        database.load_matches(con, matches)
        database.load_odds(con, odds)
    yield con
    con.close()


@pytest.fixture
def clean(ledger_db):
    """A ledger with no claims in it, so tests cannot see each other's rows."""
    ledger.create_tables(ledger_db)
    ledger_db.execute(f"DELETE FROM {ledger.SETTLEMENTS_TABLE}")
    ledger_db.execute(f"DELETE FROM {ledger.BETS_TABLE}")
    return ledger_db


@pytest.fixture(scope="module")
def played(ledger_db):
    """One played match that carries a full 1X2 price, opening and closing."""
    row = ledger_db.execute(
        """
        SELECT m.* FROM matches m
        WHERE m.home_goals IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM odds o
              WHERE o.match_id = m.match_id AND o.market = '1x2'
                AND o.phase = 'close'
          )
        ORDER BY m.date LIMIT 1
        """
    ).df().iloc[0]
    return row


def scan_row(played, **overrides) -> dict:
    """One scan-shaped row for a played match, ready for `build_claims`."""
    row = {
        "league": played["league"],
        "as_of": pd.Timestamp(played["date"]).date() - dt.timedelta(days=1),
        "fixture_date": pd.Timestamp(played["date"]).date(),
        "kickoff_time": played.get("kickoff_time"),
        "fixture_key": snapshots.fixture_key(
            played["league"], played["date"],
            played["home_team"], played["away_team"],
        ),
        "content_hash": "hash0001",
        "home_team": played["home_team"],
        "away_team": played["away_team"],
        "fixture": f"{played['home_team']} v {played['away_team']}",
        "market": "1x2",
        "selection": "home",
        "selection_label": "home",
        "line": None,
        "model_probability": 0.50,
        "push_probability": 0.0,
        "fair_price": 2.0,
        "price_taken": 2.10,
        "book": "market_max",
        "market_probability": 0.48,
        "market_source": "pinnacle",
        "edge": 0.02,
        "expected_value": 0.05,
        "withheld_reason": "",
        "n_home": 40,
        "n_away": 40,
        "thin_history": "",
        "target": "blend",
        "ridge": 1.0,
        "half_life_days": 180.0,
    }
    row.update(overrides)
    return row


def frame_of(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --------------------------------------------------------------------------
# Identity: what counts as the same claim
# --------------------------------------------------------------------------

def test_recording_the_same_board_twice_records_it_once(clean, played):
    """The scan is expected to be run repeatedly. It must not stake twice."""
    claims = ledger.build_claims(frame_of(scan_row(played)), PROVENANCE)
    first = ledger.record(clean, claims)
    second = ledger.record(clean, claims)

    assert first == {"seen": 1, "new": 1, "repeat": 0,
                     "repeat_other_revision": 0, "prior_revisions": []}
    assert second["new"] == 0
    assert second["repeat"] == 1
    assert len(ledger.load_bets(clean)) == 1


def test_a_second_sighting_moves_the_stamp_and_changes_nothing_else(clean, played):
    """`last_seen_at_utc` is the only column a repeat is allowed to touch."""
    monday = dt.datetime(2026, 9, 7, 12, 0)
    tuesday = dt.datetime(2026, 9, 8, 12, 0)
    row = scan_row(played)

    ledger.record(clean, ledger.build_claims(
        frame_of(row), PROVENANCE, recorded_at_utc=monday))
    # The same claim seen again a day later, with the model's opinion drifted:
    # same fixture, same price, same settings, so the same bet.
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, model_probability=0.55, expected_value=0.155)),
        PROVENANCE, recorded_at_utc=tuesday,
    ))

    stored = ledger.load_bets(clean)
    assert len(stored) == 1
    assert pd.Timestamp(stored["recorded_at_utc"].iloc[0]) == monday
    assert pd.Timestamp(stored["last_seen_at_utc"].iloc[0]) == tuesday
    # The first sighting's numbers survive. A ledger whose stored claim drifted
    # with the model would be recording the model's current opinion, which is
    # exactly what it exists not to do.
    assert stored["model_probability"].iloc[0] == pytest.approx(0.50)
    assert stored["expected_value"].iloc[0] == pytest.approx(0.05)


def test_a_moved_price_is_a_different_claim(clean, played):
    """Betting into 2.10 and into 2.30 are different bets and settle differently."""
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, price_taken=2.10)), PROVENANCE))
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, price_taken=2.30)), PROVENANCE))

    stored = ledger.load_bets(clean)
    assert len(stored) == 2
    assert set(stored["price_taken"]) == {2.10, 2.30}


def test_a_different_model_setting_is_a_different_claim(clean, played):
    """Two models disagreeing about one fixture made two claims, not one.

    Without this the ledger would silently keep whichever was recorded first
    and attribute it to both settings, which is the provenance failure the
    whole column set exists to prevent.
    """
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, ridge=1.0)), PROVENANCE))
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, ridge=5.0)), PROVENANCE))

    stored = ledger.load_bets(clean)
    assert len(stored) == 2
    assert set(stored["ridge"]) == {1.0, 5.0}


def test_the_as_of_date_is_not_part_of_a_claims_identity(clean, played):
    """A bet that stays on the board for three days is one bet, not three."""
    for day in (1, 2, 3):
        ledger.record(clean, ledger.build_claims(
            frame_of(scan_row(played)), PROVENANCE,
            as_of=dt.date(2026, 9, day),
        ))
    stored = ledger.load_bets(clean)
    assert len(stored) == 1
    assert pd.Timestamp(stored["as_of"].iloc[0]).date() == dt.date(2026, 9, 1)


def test_a_repeat_from_a_different_revision_is_counted_and_not_re_recorded(
    clean, played
):
    """BACKLOG B15: the claim is unchanged, the arithmetic behind it may not be.

    The revision is deliberately outside the identity hash - including it would
    re-record every open claim on every commit - so the only way a reader can
    learn that the code moved underneath a standing claim is this counter.
    """
    other = ledger.Provenance(**{**PROVENANCE.__dict__, "code_version": "def5678"})
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    counts = ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), other))

    assert counts["new"] == 0
    assert counts["repeat"] == 1
    assert counts["repeat_other_revision"] == 1
    # **Which revision, not merely how many.** After any commit the count is
    # every standing claim, every run, so on its own it is never actionable -
    # it would be a warning that always fires, which is a warning nobody reads.
    # The revision it was recorded under is the part somebody can check.
    assert counts["prior_revisions"] == ["abc1234"]
    # First seen wins, so the stored revision is the one that made the claim.
    assert ledger.load_bets(clean)["code_version"].iloc[0] == "abc1234"


def test_provenance_is_taken_per_row_not_per_run(clean, played):
    """A scan spans five leagues and nothing guarantees they resolve alike."""
    claims = ledger.build_claims(
        frame_of(
            scan_row(played, ridge=1.0, target="blend"),
            scan_row(played, ridge=5.0, target="goals", price_taken=2.20),
        ),
        PROVENANCE,
    )
    assert set(claims["ridge"]) == {1.0, 5.0}
    assert set(claims["target"]) == {"blend", "goals"}


def test_a_resolved_setting_beats_the_one_that_was_asked_for(clean, played):
    """`ridge=None` is a question; the ledger has to store the answer."""
    asked = ledger.Provenance(**{**PROVENANCE.__dict__, "ridge": float("nan")})
    claims = ledger.build_claims(frame_of(scan_row(played, ridge=1.0)), asked)
    assert claims["ridge"].iloc[0] == pytest.approx(1.0)


def test_a_claim_without_a_point_in_time_boundary_is_refused(clean, played):
    """`as_of` is what the fit was allowed to see, so it is not optional.

    Stored as a null it would leave the row unauditable while looking complete,
    which is worse than refusing it.
    """
    row = scan_row(played)
    del row["as_of"]
    with pytest.raises(ValueError, match="as_of"):
        ledger.build_claims(frame_of(row), PROVENANCE)


# --------------------------------------------------------------------------
# Withheld rows
# --------------------------------------------------------------------------

def test_withheld_rows_are_recorded_and_never_counted_as_stakes(clean, played):
    """BACKLOG B17 withholds rows from the ranking. They still get filed.

    Recorded so the thresholds can be marked; unstaked so they can never reach
    the headline. Both halves matter and this pins both.
    """
    ledger.record(clean, ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(
                played, price_taken=9.0, expected_value=0.68,
                withheld_reason="expected value +68% is above the +20% ...",
            ),
        ),
        PROVENANCE,
    ))
    summary = ledger.summary(clean)
    assert summary["bets"] == 2
    assert summary["staked"] == 1
    assert summary["withheld"] == 1

    stored = ledger.load_bets(clean, staked_only=True)
    assert len(stored) == 1
    assert stored["withheld_reason"].iloc[0] == ""


def test_an_unvaluable_row_is_dropped_rather_than_filed_as_a_null(clean, played):
    """A selection with no price is not a bet somebody could have struck."""
    claims = ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(played, price_taken=np.nan, selection="draw"),
            scan_row(played, model_probability=np.nan, selection="away"),
        ),
        PROVENANCE,
    )
    assert len(claims) == 1
    assert claims["selection"].iloc[0] == "home"


# --------------------------------------------------------------------------
# Settlement
# --------------------------------------------------------------------------

def test_a_played_match_settles_with_the_right_result(clean, played):
    """The arithmetic, against a result read straight out of the database."""
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    counts = ledger.settle_open(clean)

    assert counts["settled"] == 1
    settled = ledger.load_bets(clean, settled=True).iloc[0]
    home_won = played["home_goals"] > played["away_goals"]
    assert settled["win_fraction"] == pytest.approx(1.0 if home_won else 0.0)
    assert settled["profit_at_taken"] == pytest.approx(
        1.10 if home_won else -1.0
    )
    assert settled["match_id"] == played["match_id"]


def test_closing_line_value_matches_the_backtests_own_formula(clean, played):
    """One implementation, or the ledger and the backtest are not comparable.

    This recomputes the benchmark the way `backtest` does and asserts the
    ledger landed on the same number. If the two ever diverge, every comparison
    between a forward CLV figure and the -1.500% backtested one is invalid, and
    nothing else in this file would notice.
    """
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    ledger.settle_open(clean)

    odds = clean.execute(
        "SELECT * FROM odds WHERE match_id = ?", [played["match_id"]]
    ).df()
    expected_fair, _ = backtest._market_probabilities(
        odds, "shin", fallback=True
    )[("1x2", "home", None)]

    settled = ledger.load_bets(clean, settled=True).iloc[0]
    assert settled["closing_fair"] == pytest.approx(expected_fair)
    assert settled["clv"] == pytest.approx(2.10 * expected_fair - 1.0)


def test_a_settled_bet_is_never_restated(clean, played):
    """The promise the two-table split exists to keep.

    A second settling run must leave a settled bet exactly as it was, whatever
    it would compute now. Without this the ledger could be re-settled under
    changed code once the results were known, which is the one thing a forward
    record must not permit.
    """
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    ledger.settle_open(clean, now=dt.datetime(2026, 9, 10, 9, 0))
    before = ledger.load_bets(clean, settled=True).iloc[0]

    again = ledger.settle_open(clean, now=dt.datetime(2026, 9, 30, 9, 0))
    after = ledger.load_bets(clean, settled=True).iloc[0]

    assert again["settled"] == 0
    assert again["open"] == 0
    assert pd.Timestamp(after["settled_at_utc"]) == dt.datetime(2026, 9, 10, 9, 0)
    assert after["clv"] == pytest.approx(before["clv"], nan_ok=True)
    assert after["win_fraction"] == pytest.approx(before["win_fraction"])


def test_a_bet_with_no_played_match_stays_open_and_is_counted(clean, played):
    """Unmatched, never dropped.

    An unsettled bet and a bet that silently failed to join look identical from
    the outside, and only one of them is a reason to go looking for a bug.
    """
    future = scan_row(
        played,
        home_team="Nowhere United", away_team="Nobody City",
        fixture_key="E0_20990101_nowhere-united_nobody-city",
    )
    ledger.record(clean, ledger.build_claims(frame_of(future), PROVENANCE))
    counts = ledger.settle_open(clean)

    assert counts["settled"] == 0
    assert counts["unmatched"] == 1
    assert len(ledger.load_bets(clean, settled=False)) == 1


def test_the_three_reasons_a_bet_did_not_settle_are_told_apart(clean, played):
    """One "unmatched" count hides the only case worth acting on.

    A fixture in the future, a fixture played but not yet ingested, and a
    fixture played that should have joined and did not are three different
    situations: the first needs nothing, the second needs one command, and the
    third is a bug hunt. Reported as one number they are indistinguishable, and
    somebody running the settle step for a fortnight would see "0 settled"
    every time with no way to tell which they had.
    """
    # The three cases are defined relative to two moving boundaries: today, and
    # the newest result the database actually holds. Anchor on both rather than
    # on the fixture, or the dates drift into each other's categories.
    newest = pd.Timestamp(
        clean.execute(
            "SELECT MAX(date) FROM matches WHERE home_goals IS NOT NULL"
        ).fetchone()[0]
    ).date()
    today = newest + dt.timedelta(days=100)

    ledger.record(clean, ledger.build_claims(
        frame_of(
            # Has not kicked off: nothing to do.
            scan_row(
                played, fixture_date=today + dt.timedelta(days=7),
                home_team="Future FC", away_team="Later Town",
                fixture_key="E0_20991231_future-fc_later-town",
            ),
            # Played, but after the newest result on file: needs a rebuild.
            scan_row(
                played, fixture_date=newest + dt.timedelta(days=50),
                home_team="Ingest United", away_team="Pending City",
                fixture_key="E0_20990102_ingest-united_pending-city",
            ),
            # Well inside the ingested window and still no join: a real problem.
            scan_row(
                played, fixture_date=pd.Timestamp(played["date"]).date(),
                home_team="Mystery Rovers", away_team="Unknown Athletic",
                fixture_key="E0_19000101_mystery-rovers_unknown-athletic",
            ),
        ),
        PROVENANCE,
    ))
    counts = ledger.settle_open(clean, now=dt.datetime.combine(today, dt.time(12, 0)))

    assert counts["unmatched"] == 3
    assert counts["awaiting_kickoff"] == 1
    assert counts["awaiting_results"] == 1
    assert counts["unmatched_unexpected"] == 1


def test_the_settle_counts_carry_the_same_keys_on_every_path(clean, played):
    """An empty ledger, a resultless database and a normal run agree in shape.

    A report that had to guard each lookup would eventually forget one, and the
    key it forgot would be the one that mattered.
    """
    expected = {
        "open", "settled", "unmatched", "unsettleable", "no_closing_price",
        "awaiting_kickoff", "awaiting_results", "unmatched_unexpected",
        "unmatched_rows",
    }
    assert set(ledger.settle_open(clean)) == expected

    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    assert set(ledger.settle_open(clean)) == expected


def test_a_withdrawn_market_leaves_the_bet_open_rather_than_losing_it(
    clean, played
):
    """BACKLOG B16 made `settle` raise for odd/even rather than fail to match.

    A claim filed before that removal must not be settled as a loser, which is
    what treating the exception as "no result" would do.
    """
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, market="odd_even_goals", selection="odd")),
        PROVENANCE,
    ))
    counts = ledger.settle_open(clean)

    assert counts["settled"] == 0
    assert counts["unsettleable"] == 1
    assert len(ledger.load_bets(clean, settled=False)) == 1


def test_a_postponed_fixture_still_joins_to_its_match(clean, played):
    """One day of drift, the same tolerance `snapshots.reconcile` allows."""
    shifted = pd.Timestamp(played["date"]).date() - dt.timedelta(days=1)
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, fixture_date=shifted)), PROVENANCE))
    counts = ledger.settle_open(clean)

    assert counts["settled"] == 1
    assert ledger.load_bets(clean, settled=True)["match_id"].iloc[0] == played["match_id"]


def test_a_fixture_a_week_out_of_position_does_not_join(clean, played):
    """The tolerance rescues a postponement; it must not merge two fixtures."""
    shifted = pd.Timestamp(played["date"]).date() - dt.timedelta(days=7)
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, fixture_date=shifted)), PROVENANCE))
    counts = ledger.settle_open(clean)

    assert counts["settled"] == 0
    assert counts["unmatched"] == 1


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------

def test_stakes_are_flat_and_the_summary_reports_clv_before_profit(clean, played):
    """One unit per bet. A staking scheme would make profit louder, not truer."""
    ledger.record(clean, ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(played, selection="draw", price_taken=3.40,
                     model_probability=0.28, expected_value=-0.048),
        ),
        PROVENANCE,
    ))
    ledger.settle_open(clean)
    summary = ledger.summary(clean)

    assert set(ledger.load_bets(clean)["stake"]) == {1.0}
    assert summary["total_staked"] == pytest.approx(2.0)
    assert "mean_clv" in summary and "clv_se" in summary


def test_two_model_configurations_are_never_pooled(clean, played):
    """A changed default starts a second experiment; it does not spoil the first.

    Provenance is part of a claim's identity, so nothing already recorded is
    corrupted. What would corrupt the *answer* is averaging the two arms into
    one closing line value and reading it as though one model produced it -
    which is BACKLOG B1 exactly, a benchmark that moved mid-window with the
    pooled figure quietly describing two instruments.
    """
    other = ledger.Provenance(**{**PROVENANCE.__dict__, "ridge": 5.0})
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, ridge=5.0)), other))

    summary = ledger.summary(clean)
    assert summary["provenances"] == 2
    assert summary["mixed"] is True

    arms = ledger.by_provenance(clean)
    assert len(arms) == 2
    assert set(arms["ridge"]) == {1.0, 5.0}
    # Each arm counts only its own claims.
    assert list(arms["bets"]) == [1, 1]


def test_one_configuration_is_not_reported_as_mixed(clean, played):
    """The ordinary case must not trip the guard, or the guard gets ignored."""
    for price in (2.10, 2.30, 2.50):
        ledger.record(clean, ledger.build_claims(
            frame_of(scan_row(played, price_taken=price)), PROVENANCE))

    summary = ledger.summary(clean)
    assert summary["bets"] == 3
    assert summary["provenances"] == 1
    assert summary["mixed"] is False
    assert len(ledger.by_provenance(clean)) == 1


def test_the_report_refuses_to_print_a_pooled_figure_for_a_mixed_ledger(
    clean, played, capsys
):
    """The number a reader would otherwise take away is the one to withhold."""
    from scripts import paper_trade

    other = ledger.Provenance(**{**PROVENANCE.__dict__, "ridge": 5.0})
    ledger.record(clean, ledger.build_claims(frame_of(scan_row(played)), PROVENANCE))
    ledger.record(clean, ledger.build_claims(
        frame_of(scan_row(played, ridge=5.0)), other))
    ledger.settle_open(clean)

    paper_trade.report(clean, None)
    out = capsys.readouterr().out

    assert "MIXED LEDGER" in out
    assert "2 model configurations" in out
    assert "must be read separately" in out
    # The pooled headline is absent, which is the whole point.
    assert "Closing line value - the headline measure" not in out


def test_the_summary_of_an_empty_ledger_says_so_rather_than_dividing_by_zero(clean):
    summary = ledger.summary(clean)
    assert summary["bets"] == 0
    assert "mean_clv" not in summary


def test_the_withheld_comparison_separates_the_two_populations(clean, played):
    """The ledger marking BACKLOG B17's homework, which is why both are kept."""
    ledger.record(clean, ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(played, selection="away", price_taken=3.60,
                     expected_value=0.68, withheld_reason="too large to believe"),
        ),
        PROVENANCE,
    ))
    ledger.settle_open(clean)
    comparison = ledger.withheld_comparison(clean)

    assert list(comparison["group"]) == ["ranked", "withheld"]
    assert comparison.set_index("group").loc["ranked", "n"] == 1
    assert comparison.set_index("group").loc["withheld", "n"] == 1


def test_clv_is_clustered_by_match_not_treated_as_independent(clean, played):
    """Several selections on one fixture share a fit and one closing-line move.

    The naive standard error would treat them as independent draws and
    understate it. This asserts the ledger reports one cluster for two bets on
    the same match, which is what makes the difference visible at all.
    """
    ledger.record(clean, ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(played, selection="draw", price_taken=3.40),
        ),
        PROVENANCE,
    ))
    ledger.settle_open(clean)
    summary = ledger.summary(clean)

    assert summary["n_clv"] == 2
    assert summary["n_matches"] == 1


def test_load_bets_can_ask_for_open_settled_or_everything(clean, played):
    ledger.record(clean, ledger.build_claims(
        frame_of(
            scan_row(played),
            scan_row(
                played, home_team="Nowhere United", away_team="Nobody City",
                fixture_key="E0_20990101_nowhere-united_nobody-city",
            ),
        ),
        PROVENANCE,
    ))
    ledger.settle_open(clean)

    assert len(ledger.load_bets(clean)) == 2
    assert len(ledger.load_bets(clean, settled=True)) == 1
    assert len(ledger.load_bets(clean, settled=False)) == 1


def test_the_report_runs_without_a_write_connection(tmp_path, monkeypatch, capsys):
    """`--no-settle` must work while the app has the database open.

    DuckDB allows one writer, so a report that asked for a write connection
    would refuse to run exactly when somebody most wants it - with the app
    open in front of them. Only the settling half writes.
    """
    from scripts import paper_trade

    path = tmp_path / "report.duckdb"
    con = database.connect(path)
    ledger.create_tables(con)
    con.close()

    monkeypatch.setattr(
        sys, "argv", ["paper_trade.py", "--no-settle", "--db", str(path)]
    )
    # A second, live read-only connection stands in for the running app: if the
    # report needed to write, this would make it fail.
    holder = database.connect(path, read_only=True)
    try:
        assert paper_trade.main() == 0
    finally:
        holder.close()
    assert "Paper-trading ledger" in capsys.readouterr().out


def test_the_ledger_reads_as_empty_before_its_tables_exist(ledger_db, tmp_path):
    """The app and the report both call this before anything has been recorded."""
    fresh = database.connect(tmp_path / "fresh.duckdb")
    assert ledger.load_bets(fresh).empty
    assert ledger.summary(fresh)["bets"] == 0
    fresh.close()
