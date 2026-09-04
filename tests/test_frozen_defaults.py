"""The shipped defaults, pinned for the duration of the forward experiment.

**This file exists to make a change deliberate, not to make it hard.** A paper
ledger is only a measurement while the instrument holds still. Every value
below is part of a recorded claim's provenance, so changing one does not
corrupt the claims already filed - provenance is part of their identity - but
it does end the current run and start a second one beside it. That is a
decision worth making on purpose, with a diff attached, rather than
discovering three weeks later that the forward sample averages two models.

**If you are here because this file failed, that is the test working.** The fix
is not to weaken it. Decide whether the change is worth restarting the forward
measurement for, and if it is, edit the expected value here in the same commit
as the change. The failure is the conversation, not the obstacle.

Two of these are worse than the rest and are called out below: `margin_method`
and the fair-line preference feed into `closing_fair`, which is computed at
*settlement* time rather than at record time. Changing either would rewrite the
closing line value of bets already sitting in the ledger unsettled - the only
way this design can retroactively alter a claim.

The same reasoning is why `BACKLOG.md` records the five fixed bugs rather than
deleting them: a constraint with no trace is one somebody removes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fbedge import backtest, config, ledger  # noqa: E402
from fbedge.models import base  # noqa: E402


# The forward run these defaults belong to. Recorded so that a reader who finds
# a failure knows what is at stake and can check whether the run is still live.
EXPERIMENT_STARTED = "2026-09-04"


def test_the_model_defaults_are_frozen():
    """Target, blend weight, shrinkage and half-life: what the model *is*."""
    assert base.DEFAULT_TARGET == "blend"
    assert base.DEFAULT_BLEND_WEIGHT == 0.5
    assert base.DEFAULT_HALF_LIFE_DAYS == 180.0
    # Shrinkage is per target, and the blend's 1.0 is the value the five-league
    # validation chose. `DEFAULT_RIDGE` stays at 5.0 for anything asking by name.
    assert base.default_ridge("blend") == 1.0
    assert base.RECOMMENDED_RIDGE == {"goals": 5.0, "xg": 1.0, "blend": 1.0}
    assert base.DEFAULT_RIDGE == 5.0


def test_the_benchmark_and_the_price_taken_are_frozen():
    """**The two that can rewrite a claim after it was filed.**

    `closing_fair` is computed when a bet settles, not when it is recorded, so
    changing the margin method or the fair-line preference would change the
    closing line value of bets already in the ledger and waiting on a result.
    Every other value here only affects claims filed after the change.

    BACKLOG B1 and B10 are both instances of this going wrong historically: a
    benchmark that moved mid-window, and a benchmark that differed between
    markets inside one run.
    """
    assert backtest.DEFAULT_PRICE_SOURCE == ("market_max", "bet365", "market_avg")
    assert backtest.FAIR_LINE_PREFERENCE == (
        "betfair_exchange", "pinnacle", "market_avg", "market_max", "bet365",
    )
    defaults = {f.name: f.default for f in __import__("dataclasses").fields(
        backtest.BacktestConfig
    )}
    assert defaults["margin_method"] == "shin"
    # B1: preferring a book is not pinning to it, and the fallback stays on so
    # nothing silently narrows.
    assert defaults["fair_line_fallback"] is True


def test_the_scans_withholding_thresholds_are_frozen():
    """BACKLOG B17's two ceilings, and the ledger is testing them forward.

    These decide which claims are `staked` and which are recorded unstaked, so
    moving one does not merely change what the scan ranks - it changes what the
    ranked-against-withheld comparison is comparing.
    """
    assert config.SCAN_MIN_TEAM_MATCHES == 5
    assert config.SCAN_MAX_TRUSTED_EV == 0.20
    assert base.PROMOTED_MATCH_THRESHOLD == 25


def test_every_frozen_value_is_actually_part_of_a_claims_provenance():
    """The pin and the ledger must agree on what identifies an experiment.

    If a value were frozen here but not recorded on the claim, the freeze would
    be protecting something the ledger cannot see. If it were recorded but not
    frozen, it could drift silently. This asserts the eight provenance columns
    are exactly what `Provenance.key` hashes, so the two lists cannot diverge.
    """
    key = ledger.Provenance(
        target="blend", blend_weight=0.5, ridge=1.0, half_life_days=180.0,
        margin_method="shin", price_source="market_max", min_matches=5,
        max_ev=0.20,
    ).key()
    hashed = {part.split("=")[0] for part in key.split("|")}
    assert hashed == {
        "target", "blend_weight", "ridge", "half_life", "margin",
        "price_source", "min_matches", "max_ev",
    }
    # And the column list the reports group by covers the same eight facts.
    assert len(ledger.PROVENANCE_COLUMNS) == len(hashed)


@pytest.mark.skipif(
    not config.DB_PATH.exists(), reason="no database; nothing to check"
)
def test_the_live_ledger_is_still_a_single_experiment():
    """The check the freeze exists to make unnecessary, run anyway.

    A ledger holding two provenances is not broken and nothing in it is wrong -
    but it is two experiments, and no pooled number across them means anything.
    This fails loudly if that happens, rather than leaving it to be noticed in
    a report weeks later.
    """
    from fbedge import database

    con = database.connect(config.DB_PATH, read_only=True)
    try:
        summary = ledger.summary(con)
        if not summary["bets"]:
            pytest.skip("the ledger is empty, so there is nothing to mix")
        arms = ledger.by_provenance(con)
        assert summary["provenances"] == 1, (
            "The paper ledger holds more than one model configuration, so no "
            "pooled figure across it means anything:\n"
            f"{arms[ledger.PROVENANCE_COLUMNS + ['bets']].to_string(index=False)}"
        )
    finally:
        con.close()
