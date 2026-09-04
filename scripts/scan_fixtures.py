"""Price the upcoming fixtures against the prices the source published.

    python scripts/scan_fixtures.py
    python scripts/scan_fixtures.py --league E0 --book bet365
    python scripts/scan_fixtures.py --no-fetch --min-edge 0.03

**This is a pre-match scan, not a live scanner.** The prices in it are
collected twice a week - Friday afternoons no later than 17:00 British time for
weekend fixtures, Tuesdays no later than 13:00 for midweek ones - and the file
is rebuilt around those. Nothing here watches a market move. Calling it live
would be the first dishonest thing in this project.

**A scan is a backtest with the results column missing**, so it reuses the same
engine: the same point-in-time fits, the same margin removal, the same expected
value arithmetic, and it prices the selections the source actually published
rather than a fixed ladder of lines. Asking the model for over 2.5 when the
book is offering over 3.0 compares nothing.

**Read the evidence column before the EV column.** Every row carries the track
record of its own market on its own league: what the backtest measured, on how
many bets, and whether it was ever backtested as a bet at all. A "+6% EV" line
with no track record attached is the single most misleading thing this project
could print, and the project's own record says this model has never beaten the
closing line. See `HANDOFF.md`. Nothing here is a recommendation to stake
money.

**The scan refuses to rank its own biggest numbers**, on two measured grounds.
A fixture where either side has fewer than `config.SCAN_MIN_TEAM_MATCHES`
matches is priced mostly from the promoted-team prior; across five leagues and
19,112 settled bets those carried a mean expected value of +17.5% against
+12.8% elsewhere without doing any better. And anything claiming more than
`config.SCAN_MAX_TRUSTED_EV` is withheld whatever the sample, because a model
with negative closing line value is not finding edges that size. Both are
printed under their own heading with the reason - withheld from the sort, never
dropped from the output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fbedge import (  # noqa: E402
    backtest, config, database, evidence, markets, pricing, snapshots,
)
from fbedge import predict as predict_mod  # noqa: E402
from fbedge.models import base  # noqa: E402


def scan(
    con,
    as_of: dt.date,
    leagues: list[str],
    price_source: tuple[str, ...],
    margin_method: str = "shin",
    half_life_days: float = base.DEFAULT_HALF_LIFE_DAYS,
    ridge: float | None = None,
    fair_line_preference: tuple[str, ...] = backtest.FAIR_LINE_PREFERENCE,
    min_matches: int | None = None,
    max_ev: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Every archived selection for an unplayed fixture, priced and valued.

    **`as_of` is the point-in-time boundary and it is a hard one.** Models are
    fitted on `date < as_of` through `models.base.load_training_set`, exactly as
    in the backtest, so a scan run today cannot see a match played today.
    `test_the_scan_cannot_see_a_match_on_its_own_as_of_date` asserts it.
    """
    stored = snapshots.load_snapshots(con, leagues=leagues, latest_only=True)
    notes: list[str] = []
    if stored.empty:
        return pd.DataFrame(), ["No archived fixtures. Run scripts/snapshot_fixtures.py."]

    stored = stored[pd.to_datetime(stored["fixture_date"]).dt.date >= as_of]
    if stored.empty:
        return pd.DataFrame(), ["Every archived fixture has already kicked off."]

    odds = snapshots.load_snapshot_odds(con, stored["content_hash"].tolist())
    odds_by_hash = dict(tuple(odds.groupby("content_hash"))) if not odds.empty else {}

    rows: list[dict] = []
    for league, block in stored.groupby("league"):
        try:
            bundle = predict_mod.build_models(
                con, league, as_of, half_life_days=half_life_days, ridge=ridge,
                fit_counts=False,
            )
        except base.InsufficientData as exc:
            notes.append(f"{league}: skipped, {exc}")
            continue
        notes += [f"{league}: {note}" for note in bundle.notes]

        for fixture in block.itertuples():
            fixture_odds = odds_by_hash.get(fixture.content_hash)
            if fixture_odds is None or fixture_odds.empty:
                notes.append(
                    f"{league} {fixture.home_team} vs {fixture.away_team}: "
                    "archived with no prices attached, so nothing to value."
                )
                continue
            rows += _value_fixture(
                bundle, fixture, fixture_odds, price_source, margin_method,
                fair_line_preference, min_matches, max_ev,
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["withheld"] = frame["withheld_reason"] != ""
        frame = frame.sort_values("expected_value", ascending=False).reset_index(drop=True)
    return frame, notes


def _sample_sizes(bundle, fixture) -> tuple[int, int]:
    """How many matches the model has seen of each side, home then away."""
    return tuple(
        bundle.goals.sample_size(team) if bundle.goals.is_known(team) else 0
        for team in (fixture.home_team, fixture.away_team)
    )


def _thin_history(bundle, fixture) -> str:
    """Which side of this fixture the model barely knows, if either.

    **The single most important caveat a scan can carry.** A newly promoted
    club is priced from the promoted-team prior, which is a deliberately
    pessimistic guess rather than a measurement, so the model disagrees with
    the market about it enormously and confidently. Left unlabelled, those
    fixtures sort straight to the top of an EV table and look like the best
    bets on the board when they are the ones the model knows least about.

    Confirmed on the first live run of this scan: the top four rows were two
    newly promoted clubs with one match of history each. Below
    `config.SCAN_MIN_TEAM_MATCHES` this stops being a note and becomes a
    refusal to rank - see `_withhold`.
    """
    thin = []
    for team in (fixture.home_team, fixture.away_team):
        if not bundle.goals.is_known(team):
            thin.append(f"{team} has no history")
        else:
            played = bundle.goals.sample_size(team)
            if played < base.PROMOTED_MATCH_THRESHOLD:
                thin.append(f"{team} has {played} matches")
    return "; ".join(thin)


def _withhold(
    n_home: int,
    n_away: int,
    expected_value: float,
    min_matches: int | None = None,
    max_ev: float | None = None,
) -> str:
    """Why this selection should not be ranked, or "" if it should.

    **Withheld, not dropped.** These rows stay in the returned frame and are
    printed under their own heading with the reason; what they lose is their
    place in a table sorted by expected value. Silently discarding them would
    be the worse failure - a fixture missing from a scan looks exactly like a
    fixture nobody priced - and this project has been bitten by that shape of
    bug before.

    Two independent grounds, and a row can trip both.

    **Too little history.** Below `config.SCAN_MIN_TEAM_MATCHES` a team's
    rating is mostly the promoted-team prior. Across five leagues and 19,112
    settled bets, selections where the thinner side had fewer than five matches
    carried a mean expected value of +17.5% against +12.8% for everything else,
    and did not go on to do better. The extra edge is the model describing how
    little it knows.

    **Too large to believe.** Above `config.SCAN_MAX_TRUSTED_EV`, regardless of
    sample. A model whose closing line value is -1.5% over nine seasons is not
    finding +40% edges; it is being wrong, and a table sorted on the number
    puts its largest errors at the top. On E0 2022-2026 the bets above +20%
    were the top fifth of the model's claimed edges and returned -5.2% against
    +2.1% for the rest - which over a few hundred bets is not proof they are
    worse, but is a clear absence of evidence that they are better.
    """
    floor = config.SCAN_MIN_TEAM_MATCHES if min_matches is None else min_matches
    ceiling = config.SCAN_MAX_TRUSTED_EV if max_ev is None else max_ev
    reasons = []
    thinner = min(n_home, n_away)
    if thinner < floor:
        reasons.append(
            f"one side has {thinner} match{'' if thinner == 1 else 'es'} of "
            f"history, under the {floor} this scan requires"
        )
    if np.isfinite(expected_value) and expected_value > ceiling:
        reasons.append(
            f"expected value {expected_value:+.0%} is above the "
            f"{ceiling:+.0%} this model has earned the right to claim"
        )
    return "; and ".join(reasons)


def _value_fixture(
    bundle, fixture, fixture_odds, price_source, margin_method,
    fair_line_preference, min_matches=None, max_ev=None,
) -> list[dict]:
    """One fixture's selections, valued against the model."""
    matrix = bundle.goals.score_matrix(fixture.home_team, fixture.away_team)
    thin = _thin_history(bundle, fixture)
    n_home, n_away = _sample_sizes(bundle, fixture)

    # The same function the backtest uses, so the margin-free probability a
    # scan quotes and the one a CLV figure was measured against are produced by
    # one piece of code rather than two that agree today.
    fair = backtest._market_probabilities(
        fixture_odds, margin_method, fair_line_preference, fallback=True
    )

    out = []
    grouped = fixture_odds.groupby(["market", "selection", "line"], dropna=False)
    for (market, selection, line), group in grouped:
        line_value = backtest._line_key(line)
        priced = markets.price_selection(matrix, market, selection, line_value)
        if priced is None or priced.probability <= 0:
            continue
        price, book = backtest._pick_price(group, "open", price_source)
        if not np.isfinite(price):
            continue
        market_probability, fair_source = fair.get(
            (market, selection, line_value), (np.nan, None)
        )
        out.append(
            {
                "league": fixture.league,
                "date": fixture.fixture_date,
                "kickoff": fixture.kickoff_time,
                "fixture": f"{fixture.home_team} v {fixture.away_team}",
                "market": market,
                "selection": priced.label,
                "model_probability": priced.probability,
                "fair_price": priced.fair_price,
                "price": price,
                "book": book,
                "market_probability": market_probability,
                "market_source": fair_source,
                "edge": (
                    pricing.edge_over_market(priced.probability, market_probability)
                    if np.isfinite(market_probability) else np.nan
                ),
                "expected_value": pricing.expected_value(
                    priced.probability, price, priced.push_probability
                ),
                "collection_window": fixture.collection_window,
                "pulled_at_utc": fixture.first_pulled_at_utc,
                "thin_history": thin,
                "n_home": n_home,
                "n_away": n_away,
                "withheld_reason": _withhold(
                    n_home, n_away,
                    pricing.expected_value(
                        priced.probability, price, priced.push_probability
                    ),
                    min_matches, max_ev,
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--league", choices=sorted(config.LEAGUES), action="append")
    parser.add_argument("--as-of", default=None,
                        help="ISO date; only earlier matches inform the models")
    parser.add_argument("--book", default=None,
                        help="The bookmaker whose price you would take. Defaults "
                             "to the same market-maximum benchmark the backtest "
                             "assumes, so scan and backtest output are comparable.")
    parser.add_argument("--min-edge", type=float, default=0.0,
                        help="Hide selections below this expected value")
    parser.add_argument("--min-matches", type=int,
                        default=config.SCAN_MIN_TEAM_MATCHES,
                        help="Withhold a fixture from the ranked table when "
                             "either side has fewer matches than this "
                             "(default: %(default)s). Below it a rating is "
                             "mostly the promoted-team prior, and the claimed "
                             "edge is the model describing its own ignorance.")
    parser.add_argument("--max-ev", type=float,
                        default=config.SCAN_MAX_TRUSTED_EV,
                        help="Withhold selections claiming more than this "
                             "(default: %(default)s). This model's closing "
                             "line value is negative; it does not find edges "
                             "this large.")
    parser.add_argument("--include-withheld", action="store_true",
                        help="Rank the withheld rows alongside the rest. They "
                             "are shown either way; this puts them back in the "
                             "sort, which is what the measurement says not to do.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--half-life", type=float, default=base.DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Scan what is already archived; touch no network")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Scan anyway when the file looks out of date. "
                             "You are unlikely to want this.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db}. Run scripts/build_database.py first.")
        return 1

    leagues = args.league or list(config.LEAGUES)
    as_of = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()

    con = database.connect(args.db)

    if not args.no_fetch:
        # Archiving first, always. The prices are unrecoverable and the scan is
        # not; if the scan then refuses on staleness, the rows are still saved.
        frame, path = snapshots.download()
        snapshot, odds = snapshots.build_snapshot(frame, leagues=list(config.LEAGUES))
        counts = snapshots.write_snapshot(con, snapshot, odds)
        print(
            f"Archived: {counts['new_snapshots']} new snapshots, "
            f"{counts['repeat_snapshots']} already held."
        )
        report = snapshots.staleness(snapshot, path)
        if report["stale"]:
            print()
            if not args.allow_stale:
                for reason in report["reasons"]:
                    print(f"  stale: {reason}")
                print(
                    "\nRefusing to scan. The source warns on its own page that a "
                    "browser cache serves last week's fixtures, and a scan of "
                    "last week's prices produces confident numbers for matches "
                    "already played. Pass --allow-stale to override."
                )
                con.close()
                return 1
            print("  WARNING: scanning a stale file because --allow-stale was given.")

    price_source = (args.book,) if args.book else backtest.DEFAULT_PRICE_SOURCE
    frame, notes = scan(
        con, as_of, leagues, price_source,
        half_life_days=args.half_life, ridge=args.ridge,
        min_matches=args.min_matches, max_ev=args.max_ev,
    )

    if frame.empty:
        print("\nNothing to scan.")
        for note in notes:
            print(f"  {note}")
        con.close()
        return 1

    stored_evidence = evidence.load(con)
    _render(frame, stored_evidence, args, as_of, con)

    for note in dict.fromkeys(notes):
        print(f"  - {note}")

    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"\nEvery valued selection written to {args.csv}")
    con.close()
    return 0


def _render(frame, stored_evidence, args, as_of, con) -> None:
    eligible = frame[frame["expected_value"] >= args.min_edge]
    withheld = eligible[eligible["withheld"]]
    if not args.include_withheld:
        eligible = eligible[~eligible["withheld"]]
    shown = eligible.head(args.top)
    width = 100
    print()
    print("=" * width)
    print(f"  Pre-match EV scan   |   models fitted on matches before {as_of}")
    print("=" * width)
    print(
        "  Prices are the source's twice-weekly collection: Friday afternoons "
        "no later than\n  17:00 British time for weekend fixtures, Tuesdays no "
        "later than 13:00 for midweek.\n  Nothing here is live, and nothing "
        "here is a recommendation."
    )
    print()

    if shown.empty:
        print(f"  No selection reaches {args.min_edge:+.1%} expected value.")
        print("  That is the ordinary result and the reassuring one.")
        _render_withheld(withheld, width, args)
        return

    print(
        f"  {'fixture':<26}{'market':<17}{'sel':<13}{'price':>7}"
        f"{'model':>8}{'EV':>8}   evidence"
    )
    print("  " + "-" * (width - 2))
    for row in shown.itertuples():
        row_evidence = stored_evidence[
            (stored_evidence["league"] == row.league)
            & (stored_evidence["market"] == row.market)
        ] if not stored_evidence.empty else stored_evidence
        tag = evidence.short_labels(row_evidence, [row.market])[row.market]
        flag = " (*)" if row.thin_history else ""
        print(
            f"  {(row.fixture + flag)[:25]:<26}{row.market:<17}"
            f"{row.selection[:12]:<13}"
            f"{row.price:>7.2f}{row.model_probability:>8.1%}"
            f"{row.expected_value:>+8.1%}   {tag}"
        )

    thin_rows = shown[shown["thin_history"] != ""]
    if len(thin_rows):
        print()
        print("  (*) The model has limited history on one of these sides")
        print("  " + "-" * (width - 2))
        for fixture_name, note in sorted(
            set(zip(thin_rows["fixture"], thin_rows["thin_history"]))
        ):
            print(f"  {fixture_name}: {note}.")
        print(
            "  These clear the hard floor below, but a rating built substantially\n"
            "  from the promoted-team prior is still part guess, so read the edge\n"
            "  as less certain than the same number on an established side."
        )

    _render_withheld(withheld, width, args)

    print()
    print("  What each evidence tag means")
    print("  " + "-" * (width - 2))
    for market in sorted(shown["market"].unique()):
        for league in sorted(shown[shown["market"] == market]["league"].unique()):
            row_evidence = stored_evidence[
                (stored_evidence["league"] == league)
                & (stored_evidence["market"] == market)
            ] if not stored_evidence.empty else stored_evidence
            label = evidence.labels(row_evidence, [market])[market]
            print(f"  {league} {market}: {label}")
            if market.endswith(("cards", "card_handicap")):
                print(f"      {evidence.card_conditions(con, league)['note']}")
    print()
    print("  " + "-" * (width - 2))
    print(
        "  A positive EV here means the model disagrees with the price, not that\n"
        "  the model is right. On its own backtested markets this model's closing\n"
        "  line value is negative and has been for nine seasons, which is the\n"
        "  strongest evidence available that the disagreement is the model's\n"
        "  error rather than the market's. Read HANDOFF.md before staking\n"
        "  anything, and paper-trade first if you ever do."
    )


def _render_withheld(withheld, width: int, args) -> None:
    """Print what was kept out of the ranking, and why.

    **Shown, never dropped.** A fixture missing from a scan looks exactly like
    a fixture nobody priced, and the two must stay distinguishable. What these
    rows lose is their place in a table sorted by expected value - which is the
    whole problem with them, because that sort puts the model's largest errors
    at the top.
    """
    if not len(withheld):
        return
    print()
    if args.include_withheld:
        print(f"  {len(withheld)} of the rows above are normally withheld "
              "(--include-withheld is on)")
    else:
        print(f"  WITHHELD FROM THE RANKING: {len(withheld)} selection(s)")
    print("  " + "-" * (width - 2))
    for row in withheld.head(args.top).itertuples():
        print(
            f"  {row.fixture[:25]:<26}{row.market:<17}{row.selection[:12]:<13}"
            f"{row.price:>7.2f}{row.model_probability:>8.1%}"
            f"{row.expected_value:>+8.1%}"
        )
        print(f"      {row.withheld_reason}.")
    if len(withheld) > args.top:
        print(f"  ... and {len(withheld) - args.top} more.")
    print(
        "\n  These are not the best bets on the board. They are the selections\n"
        "  where the model is least entitled to an opinion - a side it has\n"
        "  barely seen, or an edge larger than anything its own track record\n"
        "  supports. Measured across five leagues and 19,112 settled bets, a\n"
        "  fixture with a side under five matches carried a mean expected value\n"
        "  of +17.5% against +12.8% elsewhere, and did not go on to do better."
    )


if __name__ == "__main__":
    raise SystemExit(main())
