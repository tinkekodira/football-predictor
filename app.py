"""football-edge - fixture profiles for the top-5 European leagues.

    streamlit run app.py

Two things live here, kept on separate tabs on purpose.

**What happened** is the descriptive layer: real match statistics with the
number of matches behind every figure.

**What the model thinks** is the Phase 2 forecast: probabilities and fair
prices for every market, built from team ratings that are shrunk towards the
league average and weighted by how recently each match was played.

They are separated because mixing them invites you to trust "Arsenal will
score 2.1" as much as "Arsenal have scored 2.0 per game", when the first is a
modelled estimate and the second is a fact about two matches.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from fbedge import backtest as backtest_mod
from fbedge import config, database, evaluation, markets, normalize
from fbedge import predict as predict_mod
from fbedge import profile as profile_mod
from fbedge.models import base as model_base

st.set_page_config(page_title="football-edge", page_icon="*", layout="wide")

SUMMARY_METRICS = [
    "goals_for", "goals_against", "total_goals", "goals_ht", "goals_2h",
    "btts", "over_1_5", "over_2_5", "over_3_5",
    "clean_sheet", "failed_to_score",
    "corners_for", "corners_against", "total_corners",
    "cards_for", "total_cards",
    "shots_for", "sot_for", "shots_against", "fouls_for",
    "points",
]

MARKET_GROUPS = {
    "Goals": ["goals_for", "goals_against", "total_goals", "goals_ht", "goals_2h"],
    "Goal markets": ["btts", "over_1_5", "over_2_5", "over_3_5",
                     "clean_sheet", "failed_to_score"],
    "Corners": ["corners_for", "corners_against", "total_corners"],
    "Cards": ["cards_for", "total_cards", "fouls_for"],
    "Underlying": ["shots_for", "sot_for", "shots_against", "points"],
}


@st.cache_resource
def get_connection(db_path: str):
    """One connection per database path, reused across reruns."""
    return database.connect(db_path, read_only=True)


@st.cache_data(ttl=300)
def load_options(db_path: str) -> pd.DataFrame:
    con = get_connection(db_path)
    return con.execute(
        """
        SELECT DISTINCT league, league_name, season_start_year, season
        FROM matches ORDER BY league, season_start_year DESC
        """
    ).df()


@st.cache_resource(show_spinner="Fitting models...")
def fit_models(db_path: str, league: str, as_of: dt.date, half_life: float, ridge: float,
               target: str | None = None, blend_weight: float = 0.5):
    """Fitted models for one league at one date. Cached: the fit is quick but
    not free, and the sidebar controls trigger a rerun on every change."""
    con = get_connection(db_path)
    return predict_mod.build_models(
        con, league, as_of, half_life_days=half_life, ridge=ridge, use_cache=False,
        target=target, blend_weight=blend_weight,
    )


@st.cache_data(ttl=300)
def load_teams(db_path: str, league: str, season_start_year: int) -> list[str]:
    con = get_connection(db_path)
    teams = database.known_teams(con, league=league, season_start_year=season_start_year)
    if not teams:  # early in a season a team may not have played yet
        teams = database.known_teams(con, league=league)
    return teams


def stat_cell(stat: profile_mod.Stat, as_percent: bool) -> str:
    if stat.n == 0:
        return "no data"
    return f"{stat.format(as_percent)}  (n={stat.n})"


def comparison_table(prof: profile_mod.FixtureProfile, scope: str, metrics: list[str]):
    rows = []
    for metric in metrics:
        as_pct = metric in profile_mod.RATE_METRICS
        home = prof.home.stat(scope, metric)
        away = prof.away.stat(scope, metric)
        base = prof.league_baseline.get(metric, profile_mod.Stat(None, 0))
        rows.append(
            {
                "": profile_mod.METRIC_LABELS[metric],
                prof.home_team: stat_cell(home, as_pct),
                prof.away_team: stat_cell(away, as_pct),
                "League average": base.format(as_pct) if base.n else "no data",
            }
        )
    return pd.DataFrame(rows)


def selection_table(selections, as_percent: bool = True) -> pd.DataFrame:
    """Model probabilities and the price at which each bet breaks even."""
    rows = []
    for s in selections:
        row = {
            "Selection": s.label,
            "Probability": f"{s.probability * 100:.1f}%" if as_percent else s.probability,
            "Fair price": f"{s.fair_price:.2f}",
        }
        if s.push_probability:
            row["Push"] = f"{s.push_probability * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def render_forecast(db_path, prof, league, as_of, half_life, ridge,
                    target=None, blend_weight=0.5) -> None:
    """The Phase 2 model output for the selected fixture."""
    try:
        bundle = fit_models(db_path, league, as_of, half_life, ridge, target, blend_weight)
    except model_base.InsufficientData as exc:
        st.error(f"Not enough history to fit a model: {exc}")
        return

    con = get_connection(db_path)
    forecast = predict_mod.predict_fixture(
        con, prof.home_team, prof.away_team, as_of=as_of, league=league,
        half_life_days=half_life, ridge=ridge,
        target=target, blend_weight=blend_weight,
    )

    for note in forecast.notes:
        st.warning(note)

    left, middle, right = st.columns(3)
    left.metric(
        "Expected goals",
        f"{forecast.expected_goals[0]:.2f} - {forecast.expected_goals[1]:.2f}",
    )
    if forecast.expected_corners:
        middle.metric(
            "Expected corners",
            f"{sum(forecast.expected_corners):.1f}",
            help=f"{forecast.expected_corners[0]:.1f} home, "
                 f"{forecast.expected_corners[1]:.1f} away",
        )
    if forecast.expected_cards:
        right.metric(
            "Expected cards",
            f"{sum(forecast.expected_cards):.1f}",
            help=f"{forecast.expected_cards[0]:.1f} home, "
                 f"{forecast.expected_cards[1]:.1f} away",
        )

    market_tabs = st.tabs(
        ["Result", "Goals", "Corners & cards", "Handicap", "Scores", "Ratings"]
    )
    with market_tabs[0]:
        st.dataframe(selection_table(forecast.market("1x2")),
                     width="stretch", hide_index=True)
        st.dataframe(selection_table(forecast.market("double_chance")),
                     width="stretch", hide_index=True)
    with market_tabs[1]:
        st.dataframe(selection_table(forecast.market("total_goals")),
                     width="stretch", hide_index=True)
        st.dataframe(selection_table(forecast.market("btts")),
                     width="stretch", hide_index=True)
    with market_tabs[2]:
        corners = forecast.market("total_corners")
        cards = forecast.market("total_cards")
        if corners:
            st.dataframe(selection_table(corners), width="stretch", hide_index=True)
        else:
            st.info("No corner model: this league-season has no corner data.")
        if cards:
            st.dataframe(selection_table(cards), width="stretch", hide_index=True)
        else:
            st.info("No card model: this league-season has no booking data.")
    with market_tabs[3]:
        st.dataframe(selection_table(forecast.market("asian_handicap")),
                     width="stretch", hide_index=True)
        st.caption(
            "A negative line means the home side gives that start. Whole lines "
            "can push, which is why the fair price allows for a returned stake."
        )
    with market_tabs[4]:
        st.dataframe(selection_table(forecast.market("correct_score")),
                     width="stretch", hide_index=True)
    with market_tabs[5]:
        ratings = bundle.goals.ratings()
        st.dataframe(
            ratings.style.format(
                {"attack": "{:+.3f}", "defence": "{:+.3f}", "overall": "{:+.3f}"}
            ),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Log scale, so a gap of 0.10 is roughly a 10% difference in rate. "
            "Attack is how much a team adds to its own scoring; defence is how "
            "much it subtracts from its opponent's. Both are shrunk towards the "
            "league average, hard for teams with few matches."
        )
        if bundle.cards and bundle.cards.referee_effects:
            st.subheader("Referee card multipliers")
            st.dataframe(bundle.cards.referee_table(), width="stretch", hide_index=True)

    with st.expander("Model detail"):
        for summary in forecast.model_summaries:
            st.text(summary)


@st.cache_data(show_spinner="Refitting week by week...", ttl=1800)
def run_backtest_cached(
    db_path: str, league: str, start: dt.date, end: dt.date,
    half_life: float, ridge: float, edge: float,
) -> tuple[pd.DataFrame, int, int]:
    """A backtest, cached so that changing a display option does not rerun it."""
    con = get_connection(db_path)
    settings = backtest_mod.BacktestConfig(
        league=league, start=start, end=end,
        half_life_days=half_life, ridge=ridge, edge_threshold=edge,
        fit_count_models=False,
    )
    result = backtest_mod.run_backtest(con, settings, verbose=False)
    return result.predictions, result.refits, result.matches


def render_quality(db_path, league, as_of, half_life, ridge) -> None:
    """Walk-forward backtest for the selected league and settings."""
    st.caption(
        "Each week the models are refitted on matches played before it, then "
        "used to price the selections the bookmaker actually offered. Nothing "
        "is ever fitted on a match it later predicts."
    )

    left, middle, right = st.columns(3)
    years = left.slider("Seasons to test", 1, 8, 3)
    edge = middle.slider("Edge threshold", 0.0, 0.15, 0.02, step=0.01,
                         format="%.2f",
                         help="Minimum expected value before a bet is placed.")
    run = right.button("Run backtest", type="primary")

    if not run:
        st.info("Backtests take a little while. Press the button when ready.")
        return

    start = as_of - dt.timedelta(days=365 * years)
    try:
        predictions, refits, matches = run_backtest_cached(
            db_path, league, start, as_of, half_life, ridge, edge
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    if predictions.empty:
        st.error("Nothing could be priced. Check that the odds table is populated.")
        return

    st.caption(f"{refits} refits, {matches} matches, {len(predictions)} selections priced")

    scores = [
        evaluation.score_market(predictions, market)
        for market in sorted(predictions["market"].unique())
    ]
    scores = [s for s in scores if s.get("n", 0) >= 30]
    if scores:
        st.subheader("Probability quality")
        frame = pd.DataFrame(scores)
        columns = [c for c in (
            "market", "n", "model_log_loss", "base_rate_log_loss",
            "market_log_loss", "log_loss_gap",
        ) if c in frame.columns]
        st.dataframe(frame[columns].round(4), width="stretch", hide_index=True)
        if "log_loss_gap" in frame.columns:
            best = frame["log_loss_gap"].min()
            if best < 0:
                st.success(
                    f"On its best market the model's log loss is {abs(best):.4f} "
                    "below the closing line's. Check the calibration below before "
                    "reading anything into that."
                )
            else:
                st.info(
                    "The closing line scores better than the model on every "
                    "market here. That is the normal first result, and it is "
                    "information rather than failure."
                )

    st.subheader("Calibration")
    market_choice = st.selectbox(
        "Market", sorted(predictions["market"].unique()), key="calibration_market"
    )
    subset = predictions[
        (predictions["market"] == market_choice) & (predictions["push_fraction"] < 0.5)
    ].dropna(subset=["model_conditional"])
    if len(subset) >= 50:
        table = evaluation.calibration_table(
            subset["model_conditional"].to_numpy(),
            (subset["win_fraction"] > 0.5).astype(float).to_numpy(),
        )
        if not table.empty:
            st.dataframe(table.round(4), width="stretch", hide_index=True)
            chart = table.set_index("band")[["predicted", "observed"]]
            st.bar_chart(chart)
            st.caption(
                "Read the gap column. Consistently negative means the model is "
                "overconfident, and a few percent of overconfidence is enough to "
                "turn an apparent edge into a real loss."
            )
    else:
        st.info("Not enough settled selections in this market to show calibration.")

    st.subheader("Closing line value")
    bets = predictions[
        predictions["expected_value"].notna()
        & (predictions["expected_value"] >= edge)
        & predictions["price_taken"].between(1.2, 15.0)
    ]
    if bets.empty:
        st.info(
            "No selection cleared the edge threshold. That is a respectable "
            "outcome: the model found nothing it believed was mispriced."
        )
        return

    clv = evaluation.closing_line_value(bets)
    interval = evaluation.bootstrap_roi(bets)
    ledger = evaluation.simulate_staking(bets, method="flat")
    summary = evaluation.staking_summary(ledger)

    left, middle, right = st.columns(3)
    left.metric("Bets", f"{len(bets)}")
    if clv.get("n"):
        middle.metric(
            "Mean closing line value", f"{clv['mean_clv']:+.2%}",
            help="Expected value of each bet at the margin-free closing price. "
                 "The single most reliable sign of a real edge.",
        )
        right.metric("Beat the close", f"{clv['beat_close_rate']:.0%}")

    if interval.get("n"):
        st.write(
            f"Flat staking returned **{summary['roi']:+.2%}**, with a 95% interval of "
            f"**{interval['roi_low']:+.2%} to {interval['roi_high']:+.2%}** "
            f"and a worst drawdown of {summary['max_drawdown']:.1%} of turnover staked."
        )
        if interval["roi_low"] < 0 < interval["roi_high"]:
            st.warning(
                "The interval spans zero, so this backtest cannot tell the model "
                "apart from a coin flip. Over a few hundred bets that is the "
                "expected result even for a model with a genuine edge, which is "
                "why closing line value is the metric to watch instead."
            )
        st.line_chart(ledger.set_index("date")["bankroll"])
        st.caption("Cumulative profit in stake units, not a dollar bankroll.")

    if clv.get("n") and clv["mean_clv"] < -2 * clv["clv_standard_error"]:
        st.error(
            "Closing line value is negative. Whatever the returns above say, "
            "these bets were taking worse prices than the closing line, and that "
            "does not survive contact with a longer sample."
        )


def form_table(block: profile_mod.TeamBlock) -> pd.DataFrame:
    if not block.form:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp(line["date"]).strftime("%d %b %Y"),
                "": line["venue"],
                "Opponent": line["opponent"],
                "Score": line["score"],
                "Result": line["outcome"],
            }
            for line in block.form
        ]
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.header("Fixture")

default_db = str(config.DB_PATH)
db_path = st.sidebar.text_input("Database", value=default_db)

if not Path(db_path).exists():
    st.title("football-edge")
    st.error(f"No database at `{db_path}`.")
    st.markdown(
        "Build one first:\n\n"
        "```bash\n"
        "python scripts/build_database.py --seasons 10\n"
        "```\n\n"
        "Or, to try the app without downloading anything:\n\n"
        "```bash\n"
        "python scripts/make_sample_data.py --out data/sample --seasons 4\n"
        "python scripts/build_database.py --local-dir data/sample --seasons 4\n"
        "```"
    )
    st.stop()

options = load_options(db_path)
if options.empty:
    st.error("The database exists but has no matches in it. Re-run the build script.")
    st.stop()

league_names = options.drop_duplicates("league").set_index("league")["league_name"]
league = st.sidebar.selectbox(
    "League", options=league_names.index.tolist(),
    format_func=lambda code: league_names[code],
)

seasons = (
    options[options["league"] == league]
    .drop_duplicates("season_start_year")
    .sort_values("season_start_year", ascending=False)
)
season_start_year = st.sidebar.selectbox(
    "Season", options=seasons["season_start_year"].tolist(),
    format_func=config.season_label,
)

teams = load_teams(db_path, league, int(season_start_year))
if len(teams) < 2:
    st.warning(f"Only {len(teams)} team(s) on record for this league and season.")
    st.stop()

home_team = st.sidebar.selectbox("Home team", teams, index=0)
away_options = [t for t in teams if t != home_team]
away_team = st.sidebar.selectbox("Away team", away_options, index=0)

as_of = st.sidebar.date_input(
    "Knowledge cut-off", value=dt.date.today(),
    help="Only matches played strictly before this date are used. Move it back "
         "to see what the picture looked like at any point in the past.",
)

st.sidebar.header("Model")
show_model = st.sidebar.toggle(
    "Show model forecast", value=True,
    help="Fits the goals, corners and card models using only matches played "
         "before the cut-off date.",
)
half_life = st.sidebar.slider(
    "Form half-life (days)", 60, 540, int(model_base.DEFAULT_HALF_LIFE_DAYS), step=30,
    help="How fast old matches stop counting. A match one half-life ago carries "
         "half the weight of one played today.",
)
TARGET_LABELS = {
    "blend": "Goals + xG blend",
    "goals": "Goals",
    "xg": "Expected goals",
}
target = st.sidebar.radio(
    "Rate teams on", options=list(TARGET_LABELS),
    format_func=lambda t: TARGET_LABELS[t], index=0,
    help="What team attack and defence strengths are fitted to. Goals are what "
         "you get paid on but they are a noisy sample of how a match went; xG "
         "sums chance quality instead and settles down over fewer matches. "
         "The blend of the two is the default because it beat goals on four "
         "leagues that had no part in choosing it. Whichever is chosen, the "
         "overall goal level, home advantage and the low-score correction are "
         "always re-estimated on real goals, so prices stay on the right scale.",
)
blend_weight = 0.5
if target == "blend":
    blend_weight = st.sidebar.slider(
        "Weight on xG", 0.0, 1.0, 0.5, step=0.1,
        help="0 is the goals model, 1 is the pure xG model.",
    )
if target != "goals" and not database.has_xg(get_connection(db_path)):
    st.sidebar.warning(
        "No xG in this database. Run `python scripts/build_xg.py` to download it; "
        "until then the forecast falls back to goals."
    )
    target = "goals"

# Shrinkage is tied to the target, because the right amount depends on how
# noisy the thing being fitted is - about 5 for goals, about 1 for a blend.
# A blend fitted at the goals value is the worst of both: the better signal,
# shrunk until it cannot show. Rather than silently moving a slider under the
# user, the coupling is made explicit and is on by default.
# The label is kept free of the number on purpose: Streamlit derives a
# widget's identity from its label, so a label that changed with the target
# would make this a different checkbox on every switch and lose its state.
recommended = model_base.default_ridge(target)
use_recommended = st.sidebar.checkbox(
    "Use recommended shrinkage", value=True, key="use_recommended_ridge",
    help="The right amount of shrinkage depends on how noisy the target is, so "
         "it moves with the choice above. Uncheck to set it by hand.",
)
ridge = recommended
if use_recommended:
    st.sidebar.caption(f"Shrinkage {recommended:g}, suited to {TARGET_LABELS[target].lower()}")
else:
    ridge = st.sidebar.slider(
        "Shrinkage", 0.5, 20.0, float(recommended), step=0.5,
        help="How hard team ratings are pulled towards the league average. "
             "Higher means a team needs more matches before the model believes "
             "its results.",
    )

scope = st.sidebar.radio(
    "Sample", options=list(profile_mod.SCOPES),
    format_func=lambda s: profile_mod.SCOPE_LABELS[s],
    index=1,
    help="Home/away scopes use each team's own venue: the home side's home "
         "matches, the away side's away matches.",
)

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

con = get_connection(db_path)
prof = profile_mod.fixture_profile(
    con, home_team, away_team,
    as_of=as_of, league=league, season_start_year=int(season_start_year),
)

st.title(f"{prof.home_team} vs {prof.away_team}")
st.caption(
    f"{prof.league_name} | {prof.season} | using matches played before "
    f"{prof.as_of.strftime('%d %B %Y')}"
)

for warning in prof.warnings:
    st.warning(warning)

left, middle, right = st.columns(3)
for column, block, label in (
    (left, prof.home, "Home"),
    (right, prof.away, "Away"),
):
    with column:
        st.metric(
            f"{block.team} ({label.lower()})",
            f"{block.stat(scope, 'points').format()} pts/game",
            help="Points per game over the selected sample.",
        )
        rest = (
            "unknown" if block.rest_days is None
            else f"{block.rest_days} day{'' if block.rest_days == 1 else 's'}"
        )
        st.caption(f"Rest: {rest} | {block.matches_last_14_days} match(es) in 14 days")

with middle:
    baseline = prof.league_baseline.get("total_goals", profile_mod.Stat(None, 0))
    st.metric("League goals per match", baseline.format(),
              help=f"All {prof.league_name} matches this season.")
    st.caption(f"Based on {baseline.n} team-matches")

st.divider()

forecast_tab, history_tab, quality_tab = st.tabs(
    ["Model forecast", "What happened", "Model quality"]
)

with history_tab:
    tabs = st.tabs(list(MARKET_GROUPS) + ["Form", "Head to head"])

for tab, (group, metrics) in zip(tabs, MARKET_GROUPS.items()):
    with tab:
        st.dataframe(
            comparison_table(prof, scope, metrics),
            width="stretch", hide_index=True,
        )
        st.caption(
            "n is the number of matches behind each figure. Anything under "
            "about ten is thin enough that the league average is often the "
            "better guess."
        )

with tabs[-2]:
    for block in (prof.home, prof.away):
        st.subheader(block.team)
        table = form_table(block)
        if table.empty:
            st.write("No matches on record before the cut-off date.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)

with tabs[-1]:
    h2h = prof.h2h
    if h2h["n"] == 0:
        st.write("No previous meetings on record.")
    else:
        st.write(
            f"**{h2h['n']} meetings** - {prof.home_team} {h2h['home_wins']}, "
            f"drawn {h2h['draws']}, {prof.away_team} {h2h['away_wins']}"
        )
        summary = pd.DataFrame(
            [
                {"": "Average total goals", "Value": h2h["avg_total_goals"].format()},
                {"": "Both teams scored", "Value": h2h["btts_rate"].format(True)},
                {"": "Average total corners", "Value": h2h["avg_total_corners"].format()},
            ]
        )
        st.dataframe(summary, width="stretch", hide_index=True)
        st.dataframe(
            pd.DataFrame(h2h["matches"])[
                ["date", "league_name", "home_team", "away_team",
                 "home_goals", "away_goals", "total_corners", "total_cards"]
            ],
            width="stretch", hide_index=True,
        )
        st.info(
            "Two league meetings a season means even a decade of history is "
            "about twenty matches, played by squads that have turned over "
            "several times. Head-to-head is shown because it gets asked for, "
            "not because the models use it."
        )


with forecast_tab:
    if not show_model:
        st.info("Model forecast is switched off in the sidebar.")
    else:
        render_forecast(db_path, prof, league, as_of, half_life, ridge,
                        target, blend_weight)

with quality_tab:
    render_quality(db_path, league, as_of, half_life, ridge)

st.divider()
st.caption(
    "Fair prices carry no bookmaker margin. Comparing them against real prices "
    "to find value is Phase 4."
)
