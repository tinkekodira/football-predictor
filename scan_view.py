"""The pre-match EV scan, as a tab in the app.

The same scan `scripts/scan_fixtures.py` prints, and deliberately the same
function underneath it rather than a second implementation: a page that valued
selections slightly differently from the command line would be a bug nobody
would find for months.

**The naming is not cosmetic.** The roadmap called this a "live EV scanner"
before it existed. It is not live. The source collects prices on Friday
afternoons no later than 17:00 British time for weekend fixtures and Tuesdays
no later than 13:00 for midweek ones, and rebuilds the file around those two
moments. A page that implied a moving market would be the first dishonest thing
in this project.

**Every row carries its own market's track record**, on its own league, with
the sample size - and says explicitly when a market has never been backtested
as a bet because no such price has ever existed in this source. That rule is
the whole reason this tab is worth shipping: a "+6% EV" line on its own is a
tip, and this project does not give tips.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from fbedge import backtest as backtest_mod
from fbedge import config, database, evidence as evidence_mod, snapshots
from fbedge import predict as predict_mod

from scripts import scan_fixtures


@st.cache_resource
def _connection(db_path: str):
    return database.connect(db_path, read_only=True)


@st.cache_data(show_spinner="Fitting models and pricing the board...", ttl=900)
def _scan(db_path: str, as_of: dt.date, leagues: tuple[str, ...],
          book: str | None, half_life: float, ridge: float | None):
    """Cached so that changing a filter does not refit five leagues."""
    con = _connection(db_path)
    predict_mod.clear_model_cache()
    price_source = (book,) if book else backtest_mod.DEFAULT_PRICE_SOURCE
    frame, notes = scan_fixtures.scan(
        con, as_of, list(leagues), price_source,
        half_life_days=half_life, ridge=ridge,
    )
    return frame, notes


def render(db_path: str, half_life: float, ridge: float | None) -> None:
    con = _connection(db_path)

    st.subheader("Pre-match EV scan")
    st.caption(
        "Prices published by football-data.co.uk, collected Friday afternoons "
        "no later than 17:00 British time for weekend fixtures and Tuesdays no "
        "later than 13:00 for midweek ones. Not a live feed, and not advice."
    )

    stored = snapshots.load_snapshots(con, leagues=list(config.LEAGUES))
    if stored.empty:
        st.info(
            "Nothing has been archived yet. Run `python "
            "scripts/snapshot_fixtures.py` - and run it regularly, because the "
            "source overwrites its fixtures file and archives nothing, so a "
            "price not captured when it is published cannot be recovered."
        )
        return

    report = snapshots.staleness(stored)
    if report["stale"]:
        st.error(
            "The archived fixtures look stale: "
            + "; ".join(report["reasons"])
            + ". Re-run `python scripts/snapshot_fixtures.py --refresh` before "
            "reading anything below."
        )
        return

    left, middle, right = st.columns([2, 1, 1])
    leagues = left.multiselect(
        "Leagues", options=list(config.LEAGUES),
        default=sorted(set(stored["league"])),
        format_func=lambda code: config.LEAGUES.get(code, code),
    )
    book = middle.selectbox(
        "Price you would take",
        options=["market maximum"] + sorted(
            snapshots.load_snapshot_odds(
                con, stored["content_hash"].tolist()
            )["bookmaker"].unique()
        ),
        help="The market maximum is the best price anywhere, which is what "
             "shopping across several accounts would get. Pick one book to "
             "model what a single account would really have offered.",
    )
    min_edge = right.slider(
        "Minimum EV", min_value=0.0, max_value=0.20, value=0.02, step=0.01,
        format="%+.0f%%",
    )

    if not leagues:
        st.info("Pick at least one league.")
        return

    frame, notes = _scan(
        db_path, dt.date.today(), tuple(sorted(leagues)),
        None if book == "market maximum" else book, half_life, ridge,
    )
    if frame.empty:
        st.info("Nothing to scan. " + " ".join(notes))
        return

    shown = frame[frame["expected_value"] >= min_edge]
    if shown.empty:
        st.success(
            f"No selection reaches {min_edge:+.0%} expected value. That is the "
            "ordinary result and the reassuring one."
        )
        return

    stored_evidence = evidence_mod.load(con)
    table = shown[
        ["league", "date", "fixture", "market", "selection", "price",
         "model_probability", "expected_value", "thin_history"]
    ].copy()
    table["evidence"] = [
        evidence_mod.short_labels(
            stored_evidence[
                (stored_evidence["league"] == league)
                & (stored_evidence["market"] == market)
            ] if not stored_evidence.empty else stored_evidence,
            [market],
        )[market]
        for league, market in zip(shown["league"], shown["market"])
    ]
    table = table.rename(
        columns={
            "model_probability": "model", "expected_value": "EV",
            "thin_history": "model barely knows",
        }
    )
    st.dataframe(
        table.style.format({"price": "{:.2f}", "model": "{:.1%}", "EV": "{:+.1%}"}),
        width="stretch", hide_index=True,
    )

    if (shown["thin_history"] != "").any():
        st.warning(
            "Some rows involve a club the model barely knows - see the last "
            "column. A rating built mostly from the promoted-team prior is a "
            "guess, not a measurement, so a large disagreement with the market "
            "there is the expected output of a model that does not know the "
            "team, not a mispricing it has found."
        )

    with st.expander("What the evidence column means, market by market"):
        for market in sorted(shown["market"].unique()):
            for league in sorted(shown[shown["market"] == market]["league"].unique()):
                row = (
                    stored_evidence[
                        (stored_evidence["league"] == league)
                        & (stored_evidence["market"] == market)
                    ] if not stored_evidence.empty else stored_evidence
                )
                st.markdown(
                    f"**{config.LEAGUES.get(league, league)} - {market}**  \n"
                    + evidence_mod.labels(row, [market])[market]
                )
        if stored_evidence.empty:
            st.info(
                "No evidence has been computed for this database. Run "
                "`python scripts/build_evidence.py`; until then every market "
                "is reported as untested, which is accurate but not useful."
            )

    st.error(
        "A positive EV means the model disagrees with the price, not that the "
        "model is right. On its own backtested markets this model's closing "
        "line value is negative and has been for nine seasons, which is the "
        "strongest evidence available that the disagreement is the model's "
        "error rather than the market's. Read `HANDOFF.md` before staking "
        "anything, and paper-trade first if you ever do."
    )
