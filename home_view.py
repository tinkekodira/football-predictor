"""The home page: a date you can scroll, and the matches on it.

Kept out of `app.py` because it is a page rather than a widget, and out of
`fbedge/` because everything under there is deliberately free of Streamlit -
the models and the backtest have to run in scripts and tests with no UI
anywhere near them.

Three panels on the right, and it is worth being precise about what each one
can honestly claim, because two of them are constrained by what this project
actually has:

- **Highlighted matches** is real. The model prices every fixture on the day
  from ratings fitted only on matches played before it, and the ones shown are
  where it disagrees most with a coin-flip prior. Where a closing price exists
  the disagreement with the market is shown too - but football-data.co.uk only
  publishes odds a few days ahead, so for most future dates there is no market
  to disagree with, and the panel says so rather than implying an edge.
- **Availability** is derived, not reported. There is no injury feed in this
  project. What `fbedge.availability` can say is which regular starters have
  been missing from recent line-ups, computed from matches already played.
  That is a real signal and an honest one, and it is *not* news: it cannot
  know about an injury picked up in training this morning.
- **News** has no source at all. The panel says so instead of being filled
  with something that looks like news and is not.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from fbedge import availability as availability_mod
from fbedge import config, database, fixtures as fixtures_mod, markets
from fbedge import predict as predict_mod
from fbedge.models import base as model_base

# How many days either side of the selection the date strip shows. Seven fits
# a wide layout without wrapping and covers a full round of fixtures.
STRIP_RADIUS = 3

# Fixtures whose model probability for the most likely outcome clears this are
# not interesting: everyone agrees Bayern will beat the bottom club. The
# highlight panel wants matches where the model has an opinion that is strong
# but not obvious.
OBVIOUS_THRESHOLD = 0.70

LEAGUE_ORDER = ["E0", "SP1", "I1", "D1", "F1"]


@st.cache_resource
def get_connection(db_path: str):
    """A read-only connection, cached per database path.

    Deliberately *not* imported from `app.py`. That module is a script, so
    importing it re-executes the entire page and Streamlit then rejects the
    second copy of every widget. DuckDB is happy with several readers, so a
    second cached handle costs nothing.
    """
    return database.connect(db_path, read_only=True)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


def selected_date() -> dt.date:
    return st.session_state.setdefault("calendar_date", dt.date.today())


def set_date(day: dt.date) -> None:
    st.session_state["calendar_date"] = day


def open_match(row: pd.Series) -> None:
    """Switch to the detail view for one fixture.

    The whole row is stashed rather than an id, because the detail page needs
    the league, the two team names and the kick-off date, and re-querying for
    them would be a second chance to disagree with what the calendar showed.
    """
    st.session_state["view"] = "match"
    st.session_state["match"] = {
        "league": row["league"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "kickoff_local": row["kickoff_local"],
        "played": bool(row["played"]),
        "home_goals": row["home_goals"],
        "away_goals": row["away_goals"],
    }


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def load_fixtures(db_path: str, season: int) -> pd.DataFrame:
    """The stored calendar for a season, in UTC.

    Cached for ten minutes: the table only changes when `build_fixtures.py`
    runs, and the home page re-reads it on every widget interaction.
    """
    con = get_connection(db_path)
    try:
        return fixtures_mod.load_calendar(con, season=season)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner="Pricing the day's fixtures...")
def price_day(
    db_path: str,
    day: dt.date,
    leagues: tuple[str, ...],
    half_life: float,
    matches: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    """Model 1X2 probabilities for every fixture on one day.

    One model fit per league rather than one per fixture: the fit is the
    expensive part and every match in a league on a given day shares it.
    `as_of` is the match date, so nothing played on the day itself informs its
    own prediction.

    `matches` is passed as a tuple of tuples so the cache key is hashable; it
    carries what the caller already read, which keeps this from re-querying.
    """
    con = get_connection(db_path)
    bundles = {}
    for league in leagues:
        try:
            bundles[league] = predict_mod.build_models(
                con, league, day, half_life_days=half_life,
                fit_counts=False, use_cache=False,
            )
        except Exception:
            bundles[league] = None

    rows = []
    for league, home_team, away_team in matches:
        bundle = bundles.get(league)
        if bundle is None:
            continue
        model = bundle.goals
        matrix = model.score_matrix(home_team, away_team)
        # Key on `selection`, which is the stable "home"/"draw"/"away" field,
        # not on `label`, which is a display string and free to change.
        selections = {s.selection: s.probability for s in markets.match_odds(matrix)}
        expected = markets.expected_total(matrix)
        best = max(selections, key=selections.get)
        rows.append(
            {
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "p_home": selections["home"],
                "p_draw": selections["draw"],
                "p_away": selections["away"],
                "pick": {"home": home_team, "draw": "a draw",
                         "away": away_team}[best],
                "confidence": selections[best],
                "expected_goals": expected,
                "known": model.is_known(home_team) and model.is_known(away_team),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The date strip
# ----------------------------------------------------------------------


def render_date_strip(calendar: pd.DataFrame) -> dt.date:
    """A scrollable row of dates, with the days that have football marked.

    Streamlit has no horizontal scroller, so this is a fixed window of days
    around the selection with arrows that move it. The marker matters more
    than it looks: an international break is a fortnight of empty days, and
    without it you scroll blindly.
    """
    day = selected_date()
    with_football = set(fixtures_mod.dates_with_matches(calendar))

    head = st.columns([1, 1, 6, 1])
    with head[0]:
        if st.button("<", width="stretch", help="One week back"):
            set_date(day - dt.timedelta(days=7))
            st.rerun()
    with head[1]:
        if st.button("Today", width="stretch"):
            set_date(dt.date.today())
            st.rerun()
    with head[3]:
        if st.button("\\>", width="stretch", help="One week forward"):
            set_date(day + dt.timedelta(days=7))
            st.rerun()
    with head[2]:
        st.markdown(
            f"<div style='text-align:center;font-size:1.1rem;padding-top:0.35rem'>"
            f"<b>{day.strftime('%A %d %B %Y')}</b></div>",
            unsafe_allow_html=True,
        )

    span = [day + dt.timedelta(days=offset)
            for offset in range(-STRIP_RADIUS, STRIP_RADIUS + 1)]
    for column, candidate in zip(st.columns(len(span)), span):
        with column:
            count = sum(1 for d in with_football if d == candidate)
            # A dot rather than a count: the number of fixtures is on the page
            # already, and what the strip has to answer is only "is there
            # anything here", so that an international break reads at a glance.
            label = candidate.strftime("%a %d/%m") + ("  *" if count else "")
            st.button(
                label,
                key=f"strip_{candidate.isoformat()}",
                width="stretch",
                type="primary" if candidate == day else "secondary",
                disabled=candidate == day,
                on_click=set_date,
                args=(candidate,),
                help=("no fixtures in these leagues" if not count
                      else f"{count} fixture(s)"),
            )
    return day


# ----------------------------------------------------------------------
# The match list
# ----------------------------------------------------------------------


def render_matches(day_frame: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Every fixture on the selected day, grouped by league."""
    if day_frame.empty:
        st.info(
            "No fixtures in the top five leagues on this date. "
            "International breaks and midweeks between rounds look like this; "
            "the arrows above move a week at a time."
        )
        return

    lookup = {}
    if not prices.empty:
        lookup = {
            (r["league"], r["home_team"], r["away_team"]): r
            for _, r in prices.iterrows()
        }

    ordered = sorted(
        day_frame["league"].unique(),
        key=lambda code: LEAGUE_ORDER.index(code) if code in LEAGUE_ORDER else 99,
    )
    for league in ordered:
        block = day_frame[day_frame["league"] == league]
        st.markdown(f"##### {config.LEAGUES.get(league, league)}")
        for _, row in block.iterrows():
            columns = st.columns([1.1, 4.2, 1.4, 2.6])
            with columns[0]:
                if row["played"]:
                    st.markdown("**FT**")
                else:
                    st.markdown(f"**{row['kickoff_local'].strftime('%H:%M')}**")
            with columns[1]:
                st.write(f"{row['home_team']}  -  {row['away_team']}")
            with columns[2]:
                if row["played"] and pd.notna(row["home_goals"]):
                    st.markdown(
                        f"**{int(row['home_goals'])} - {int(row['away_goals'])}**"
                    )
                else:
                    st.write("")
            with columns[3]:
                priced = lookup.get(
                    (row["league"], row["home_team"], row["away_team"])
                )
                if priced is not None and priced["known"]:
                    st.caption(
                        f"{priced['p_home']:.0%} / {priced['p_draw']:.0%} / "
                        f"{priced['p_away']:.0%}"
                    )
                st.button(
                    "Details",
                    key=f"open_{row['understat_id']}",
                    on_click=open_match,
                    args=(row,),
                )
        st.divider()


# ----------------------------------------------------------------------
# Right-hand panels
# ----------------------------------------------------------------------


def render_highlights(prices: pd.DataFrame, day: dt.date) -> None:
    """Where the model has the strongest non-obvious opinion."""
    st.markdown("#### Highlighted matches")
    if prices.empty:
        st.caption("Nothing to price on this date.")
        return

    usable = prices[prices["known"]]
    if usable.empty:
        st.caption(
            "The model has not seen these teams before - most likely a newly "
            "promoted side early in the season."
        )
        return

    interesting = usable[usable["confidence"] < OBVIOUS_THRESHOLD]
    pool = interesting if not interesting.empty else usable
    top = pool.sort_values("confidence", ascending=False).head(4)

    for _, row in top.iterrows():
        st.markdown(f"**{row['home_team']} v {row['away_team']}**")
        st.caption(
            f"{config.LEAGUES.get(row['league'], row['league'])}  -  model makes "
            f"**{row['pick']}** {row['confidence']:.0%}, with "
            f"{row['expected_goals']:.1f} goals expected"
        )
    st.caption(
        "Model probabilities only. These are not edges: a bet needs a price to "
        "beat, and this source publishes odds only a few days ahead. On nine "
        "seasons of history this model has never beaten the closing line, so "
        "treat these as opinions, not tips."
    )


@st.cache_data(ttl=600, show_spinner=False)
def availability_for_day(
    db_path: str,
    day: dt.date,
    matches: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    """Missing-starter share for both sides of every fixture on a day.

    Uses `availability.for_fixture`, which is the same point-in-time windowing
    the model was measured with, rather than a second version of it written for
    display. That matters: a panel that computed absences differently from the
    study would put a number on screen that no measurement backs.
    """
    con = get_connection(db_path)
    try:
        lineups = con.execute("SELECT * FROM match_lineups").df()
    except Exception:
        return pd.DataFrame()
    if lineups.empty:
        return pd.DataFrame()

    leagues = sorted({league for league, _, _ in matches})
    played = con.execute(
        "SELECT match_id, date, home_team, away_team, league FROM matches "
        f"WHERE league IN ({','.join('?' for _ in leagues)})",
        leagues,
    ).df()

    rows = []
    for league, home_team, away_team in matches:
        home_share, away_share = availability_mod.for_fixture(
            lineups, played[played["league"] == league], home_team, away_team, day
        )
        rows.append(
            {
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "home_share": home_share,
                "away_share": away_share,
            }
        )
    return pd.DataFrame(rows)


def render_availability(day_frame: pd.DataFrame, db_path: str, day: dt.date) -> None:
    """Who has been missing, from line-ups of matches already played."""
    st.markdown("#### Availability watch")
    if day_frame.empty:
        st.caption("No fixtures to check.")
        return

    pairs = tuple(
        (r["league"], r["home_team"], r["away_team"]) for _, r in day_frame.iterrows()
    )
    shares = availability_for_day(db_path, day, pairs)

    if shares.empty or not (shares[["home_share", "away_share"]].to_numpy() > 0).any():
        st.caption(
            "No line-up history for these fixtures, so nothing can be derived. "
            "Populate it with the app stopped:"
        )
        st.code("python scripts/build_rosters.py --league E0", language="bash")
    else:
        # A missing-starter share is a fraction of a regular XI, so 0.09 is
        # about one regular out. Below that is ordinary rotation and not worth
        # a line on the page.
        notable = shares[
            (shares["home_share"] >= 0.09) | (shares["away_share"] >= 0.09)
        ].copy()
        if notable.empty:
            st.caption("Nothing unusual: every side is close to its regular XI.")
        else:
            for _, row in notable.head(5).iterrows():
                st.markdown(f"**{row['home_team']} v {row['away_team']}**")
                st.caption(
                    f"regulars missing - {row['home_team']} "
                    f"{row['home_share']:.0%}, {row['away_team']} "
                    f"{row['away_share']:.0%}"
                )

    st.info(
        "**This is not an injury feed.** It counts regular starters absent from "
        "recent teamsheets, so it cannot tell an injury from a rested player, "
        "and it cannot know about anything that happened this morning. "
        "Confirmed line-ups appear about an hour before kick-off, after the "
        "price a bet would be struck at. Measured on the Premier League this "
        "signal moved log loss by 0.00004, so the model ignores it.",
        icon=":material/info:",
    )


def render_news() -> None:
    st.markdown("#### News")
    st.caption(
        "No news source is wired into this project. Adding one means picking a "
        "feed and accepting its terms; nothing here will invent headlines in "
        "the meantime."
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def render(db_path: str, half_life: float) -> None:
    st.title("football-edge")

    season = fixtures_mod.current_season()
    calendar = load_fixtures(db_path, season)
    if calendar.empty:
        st.warning(
            "No fixture calendar loaded yet. Build it with:\n\n"
            "```bash\npython scripts/build_fixtures.py\n```"
        )
        return

    st.caption(
        f"{config.season_label(season)}  -  {len(calendar)} fixtures across the "
        f"top five leagues.  Times shown in {fixtures_mod.DISPLAY_TIMEZONE.split('/')[-1]}."
    )

    day = render_date_strip(calendar)
    st.divider()

    day_frame = fixtures_mod.matches_on(calendar, day)
    leagues = tuple(sorted(day_frame["league"].unique())) if not day_frame.empty else ()
    pairs = tuple(
        (r["league"], r["home_team"], r["away_team"]) for _, r in day_frame.iterrows()
    )
    prices = (
        price_day(db_path, day, leagues, half_life, pairs)
        if pairs else pd.DataFrame()
    )

    left, right = st.columns([2.4, 1])
    with left:
        render_matches(day_frame, prices)
    with right:
        render_highlights(prices, day)
        st.divider()
        render_availability(day_frame, db_path, day)
        st.divider()
        render_news()
