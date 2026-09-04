"""The paper-trading ledger, as a tab in the app.

**This page is a viewer and nothing else.** It records no claims and settles
nothing. Both of those are writes to an append-only table, and a page that
mutated a forward record every time somebody loaded it would be the fastest
possible way to corrupt the one measurement this project cannot reconstruct.
The app's connection is read-only for exactly that reason. Recording is
`scripts/scan_fixtures.py --record`; settling is `scripts/paper_trade.py`.

**It reports nothing for weeks, and that is the design.** Prices in this source
reach about three days ahead, so a young ledger holds claims on fixtures that
have not been played. A mean over a handful of settled bets is noise whatever
the arithmetic says, so the headline carries its clustered standard error and
refuses to characterise itself below `TOO_THIN_TO_READ` bets.

**Withheld picks get their own view rather than a footnote.** They are the
population BACKLOG B17's two thresholds suppress, recorded unstaked precisely
so that those thresholds can eventually be marked against what happened rather
than only against the backtest that set them. A ledger that showed only the
bets it liked could not mark its own homework, and a page that hid them would
undo that on the way to the screen.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fbedge import config, database, ledger

# Below this many settled bets the page declines to characterise the mean. Not
# a significance threshold - it is the point below which a reader should not be
# shown a number that invites one. Matches `scripts/paper_trade.py`.
TOO_THIN_TO_READ = 30

# What the forward measurement is eventually meant to test itself against.
BACKTESTED_CLV = -0.01500
BACKTESTED_N = 4379


@st.cache_resource
def _connection(db_path: str):
    return database.connect(db_path, read_only=True)


@st.cache_data(ttl=300)
def _load(db_path: str, leagues: tuple[str, ...]):
    """Cached briefly, because the ledger only changes when a script runs it."""
    con = _connection(db_path)
    chosen = list(leagues) if leagues else None
    return (
        ledger.load_bets(con, leagues=chosen),
        ledger.summary(con, leagues=chosen),
        ledger.withheld_comparison(con, leagues=chosen),
    )


def render(db_path: str) -> None:
    st.subheader("Paper-trading ledger")
    st.caption(
        "What the scan claimed, and what happened next. Nothing here has been "
        "staked. This page reads the ledger; it never writes to it - record a "
        "board with `scripts/scan_fixtures.py --record` and settle it with "
        "`scripts/paper_trade.py`."
    )

    leagues = tuple(
        st.multiselect(
            "Leagues", sorted(config.LEAGUES), default=[],
            help="Empty means every league on file.",
        )
    )
    frame, summary, comparison = _load(db_path, leagues)

    if not summary["bets"]:
        st.info(
            "The ledger is empty. File today's board with:\n\n"
            "```\npython scripts/scan_fixtures.py --record\n```\n\n"
            "It will report nothing until those fixtures have been played and "
            "`scripts/paper_trade.py` has settled them, which is a matter of "
            "weeks rather than minutes."
        )
        return

    if summary["mixed"]:
        _render_mixed(db_path, leagues, summary)
        return

    _render_headline(summary)
    st.divider()

    open_tab, settled_tab, withheld_tab = st.tabs(
        [
            f"Open ({summary['open']})",
            f"Settled ({summary['settled']})",
            f"Withheld ({summary['withheld']})",
        ]
    )
    with open_tab:
        _render_open(frame)
    with settled_tab:
        _render_settled(frame, comparison)
    with withheld_tab:
        _render_withheld(frame)

    st.divider()
    st.caption(
        "A positive expected value means the model disagreed with the price, "
        "not that it was right. On its own backtested markets this model's "
        "closing line value is negative and has been for nine seasons, which "
        "is the strongest evidence available that the disagreement is the "
        "model's error rather than the market's."
    )


def _render_mixed(db_path: str, leagues: tuple[str, ...], summary: dict) -> None:
    """More than one model configuration on file, so no pooled headline.

    **Two provenances is two experiments.** Provenance is part of a claim's
    identity, so a changed default does not corrupt what is already recorded -
    it starts a second run beside the first. What would corrupt the *answer* is
    averaging them into one closing line value and reading it as though a
    single model produced it, which is BACKLOG B1 exactly: a benchmark that
    moved mid-window, with the pooled figure quietly describing two
    instruments.

    So the page shows the arms and declines to combine them. Not an error - a
    second arm may well be deliberate - but never one number.
    """
    con = _connection(db_path)
    st.error(
        f"**This ledger holds {summary['provenances']} model configurations.** "
        "They are separate experiments and are shown separately: a single "
        "pooled closing line value across them would describe no model that "
        "ever existed. Provenance is part of a claim's identity, so nothing "
        "recorded has been corrupted — but nothing here can be read as one "
        "measurement either."
    )
    st.markdown("#### Model configurations on file")
    arms = ledger.by_provenance(con, leagues=list(leagues) if leagues else None)
    show = [
        "target", "ridge", "half_life_days", "min_matches", "max_ev",
        "bets", "staked", "settled", "n_clv", "mean_clv", "clv_se",
        "first_recorded",
    ]
    st.dataframe(
        arms[show].style.format({"mean_clv": "{:+.3%}", "clv_se": "{:.3%}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "To go back to one experiment, either restore the earlier settings or "
        "start a fresh ledger and treat the arms above as separate records."
    )


# --------------------------------------------------------------------------
# The headline
# --------------------------------------------------------------------------

def _render_headline(summary: dict) -> None:
    """Closing line value first, profit second, and the caveat in between.

    The order is the project's standing rule rather than a layout preference:
    over a few hundred bets return on investment is close to noise while CLV is
    measurable in weeks, which is why the gate is phrased in CLV. Putting
    profit above it would invite exactly the reading the gate exists to stop.
    """
    counted, staked, withheld = summary["bets"], summary["staked"], summary["withheld"]
    row = st.columns(4)
    row[0].metric("Claims on file", f"{counted:,}")
    row[1].metric("Ranked, treated as staked", f"{staked:,}")
    row[2].metric("Withheld, unstaked", f"{withheld:,}")
    row[3].metric("Settled", f"{summary['settled']:,}",
                  help=f"{summary['open']:,} still waiting on a result.")

    if not summary.get("n_clv"):
        st.info(
            "**No settled bet carries a closing price yet, so there is no "
            "closing line value to report.** That is the expected state of a "
            "ledger younger than the fixtures it holds claims on - prices in "
            "this source reach about three days ahead, so the first settled "
            "bets arrive within the week and the first readable number does "
            "not."
        )
        return

    mean, se = summary["mean_clv"], summary["clv_se"]
    st.markdown("##### Closing line value — the headline measure")
    clv_row = st.columns(4)
    clv_row[0].metric("Mean CLV", f"{mean:+.3%}")
    clv_row[1].metric(
        "Clustered SE",
        "—" if pd.isna(se) else f"{se:.3%}",
        help="Clustered by match: several selections on one fixture share a "
             "model fit and one closing-line move, so treating them as "
             "independent draws would overstate the precision.",
    )
    clv_row[2].metric(
        "In standard errors",
        "—" if pd.isna(se) or se == 0 else f"{mean / se:+.1f} SE",
    )
    clv_row[3].metric("Beat the close", f"{summary['beat_close_rate']:.1%}")

    if summary["n_clv"] < TOO_THIN_TO_READ:
        st.warning(
            f"**{summary['n_clv']} settled bets is too few to read.** The mean "
            "above is shown because hiding it would be worse, not because it "
            "means anything yet. For scale, the backtest this is eventually "
            f"meant to test measured {BACKTESTED_CLV:+.3%} over "
            f"{BACKTESTED_N:,} bets."
        )
    else:
        st.info(
            f"The backtest over nine seasons of E0 measured "
            f"{BACKTESTED_CLV:+.3%} (−9.1 SE) on {BACKTESTED_N:,} bets. This "
            "ledger is the forward test of that number, and the decision gate "
            "needs closing line value indistinguishable from zero to open — "
            "and needs it to survive a tuner holdout it never influenced."
        )

    if summary.get("n_profit"):
        with st.expander("Profit — the subordinate check, and it is noise at this size"):
            profit_row = st.columns(3)
            profit_row[0].metric("Staked", f"{summary['total_staked']:,.0f} units",
                                 help="Flat: one unit per bet, never sized.")
            profit_row[1].metric("Profit", f"{summary['profit']:+,.2f} units")
            profit_row[2].metric("ROI", f"{summary['roi']:+.2%}")
            st.caption(
                "Reported below closing line value, not beside it. Over a few "
                "hundred bets ROI is almost pure noise, while whether a price "
                "beat the close is measurable in weeks. A profit alongside "
                "negative CLV means the profit was luck and will not survive."
            )


# --------------------------------------------------------------------------
# The three views
# --------------------------------------------------------------------------

DISPLAY = {
    "fixture_date": "date",
    "selection_label": "selection",
    "price_taken": "price",
    "model_probability": "model",
    "expected_value": "EV",
}


def _fixture_column(frame: pd.DataFrame) -> pd.Series:
    return frame["home_team"] + " v " + frame["away_team"]


def _render_open(frame: pd.DataFrame) -> None:
    """Claims still waiting on a result."""
    open_bets = frame[frame["match_id"].isna() & frame["staked"]]
    if open_bets.empty:
        st.info(
            "No open bets. Every claim on file has been settled, or none has "
            "been recorded since the last settlement run."
        )
        return

    st.markdown("#### Open bets, waiting on a result")
    st.caption(
        f"{len(open_bets):,} claim(s) on fixtures that have not been played, "
        "or that have been played but not yet settled. Settling is "
        "`python scripts/paper_trade.py`."
    )
    table = open_bets.assign(fixture=_fixture_column(open_bets))[
        ["fixture_date", "league", "fixture", "market", "selection_label",
         "price_taken", "model_probability", "expected_value", "book"]
    ].rename(columns=DISPLAY)
    st.dataframe(
        table.style.format(
            {"price": "{:.2f}", "model": "{:.1%}", "EV": "{:+.1%}"}
        ),
        width="stretch", hide_index=True,
    )


def _render_settled(frame: pd.DataFrame, comparison: pd.DataFrame) -> None:
    """Claims with a result, and the CLV each one earned."""
    settled = frame[frame["match_id"].notna() & frame["staked"]]
    if settled.empty:
        st.info(
            "Nothing has settled yet. A claim settles once its fixture has "
            "been played and `scripts/paper_trade.py` has joined it to the "
            "result."
        )
        _render_comparison(comparison)
        return

    st.markdown("#### Settled claims, and the closing line value each earned")
    st.caption(
        f"{len(settled):,} settled claim(s). `CLV` is the price taken against "
        "the margin-free closing line: positive means the price beat the "
        "close, whether or not the bet won."
    )
    table = settled.assign(
        fixture=_fixture_column(settled),
        score=(
            settled["home_goals"].astype("Int64").astype(str)
            + "-" + settled["away_goals"].astype("Int64").astype(str)
        ),
    )[
        ["fixture_date", "league", "fixture", "score", "market",
         "selection_label", "price_taken", "expected_value", "win_fraction",
         "profit_at_taken", "clv"]
    ].rename(columns={**DISPLAY, "win_fraction": "won", "profit_at_taken": "profit"})
    st.dataframe(
        table.style.format(
            {"price": "{:.2f}", "EV": "{:+.1%}", "won": "{:.2f}",
             "profit": "{:+.2f}", "clv": "{:+.2%}"}
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        "`won` is a fraction rather than a flag because a quarter-line Asian "
        "handicap can be half won and half returned."
    )
    _render_comparison(comparison)


def _render_withheld(frame: pd.DataFrame) -> None:
    """The rows the scan refused to rank, with the reason for each.

    **Recorded, not dropped, and shown rather than buried.** These are the
    selections where the model is least entitled to an opinion, and a table
    sorted by expected value puts them at the top - which is why they are kept
    out of the ranking. Keeping them out of the *record* would be a different
    and worse decision, because then nothing could ever say whether the two
    thresholds were set correctly.
    """
    withheld = frame[~frame["staked"]]
    if withheld.empty:
        st.info("No selection has been withheld.")
        return

    st.markdown("#### Withheld selections, and the reason for each")
    st.caption(
        f"{len(withheld):,} selection(s) recorded but never staked. These are "
        "not the best bets on the board — they are the ones where the model "
        f"knows least. Below {config.SCAN_MIN_TEAM_MATCHES} matches of history "
        "a rating is mostly the promoted-team prior, and above "
        f"{config.SCAN_MAX_TRUSTED_EV:+.0%} the claimed edge is larger than "
        "anything this model's own record supports."
    )
    table = withheld.assign(fixture=_fixture_column(withheld))[
        ["fixture_date", "league", "fixture", "market", "selection_label",
         "price_taken", "expected_value", "n_home", "n_away", "withheld_reason"]
    ].rename(
        columns={**DISPLAY, "n_home": "home n", "n_away": "away n",
                 "withheld_reason": "why it is withheld"}
    )
    st.dataframe(
        table.style.format({"price": "{:.2f}", "EV": "{:+.1%}"}),
        width="stretch", hide_index=True,
    )


def _render_comparison(comparison: pd.DataFrame) -> None:
    """Ranked against withheld, once both have settled bets.

    This is the ledger marking BACKLOG B17's homework. It needs hundreds of
    bets in both groups before it says anything, and until then it is a shape
    rather than a result - which the caption says rather than leaving a reader
    to infer from the sample column.
    """
    if comparison.empty or not comparison["n"].sum():
        return
    st.markdown("##### Ranked against withheld")
    table = comparison.rename(
        columns={
            "group": "", "n": "bets", "n_matches": "matches",
            "mean_ev": "mean EV", "mean_clv": "mean CLV", "clv_se": "SE",
        }
    )
    st.dataframe(
        table.style.format(
            {"mean EV": "{:+.2%}", "mean CLV": "{:+.2%}", "SE": "{:.2%}",
             "roi": "{:+.2%}"}
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        "The withheld rows are recorded and settled precisely so this "
        "comparison can exist. The two thresholds were set from a backtest of "
        "19,112 bets; this is the forward check on them, and it needs hundreds "
        "of settled bets in both groups before a difference means anything."
    )
