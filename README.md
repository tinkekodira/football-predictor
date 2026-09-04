# football-edge

A football analytics platform for the top five European leagues: Premier League,
La Liga, Serie A, Bundesliga and Ligue 1.

**Phases 1 to 4 are built.** Phase 1 is the data spine: it downloads a decade
of real match data, normalises it, stores it and checks it. Phase 2 is the
model: fitted team ratings that produce probabilities and fair prices for
nineteen markets from a single fit. Phase 3 is the part that decides whether any
of it is worth believing: a walk-forward backtest measuring calibration,
closing line value and returns with honest confidence intervals. Phase 4 points
the same machinery forwards, at fixtures that have not been played — and
attaches to every price the track record that price actually has.

**Total cost to run: nothing.** No API key and no subscription are needed for
anything in the quick start. Two qualifications, because the earlier version of
this line said "no scraping" and that was not true:

- **`understat.py` scrapes Understat.** It reads a JSON endpoint the site's own
  page calls, caches every response to disk, and is what supplies the expected
  goals the shipping model is fitted to. It is free and needs no key, but it is
  somebody else's server and it is not a published API.
- **Two features take an optional key**, and degrade to absent without one
  rather than breaking: club badges and injuries need a free API-Football key,
  and the football-data.org calendar takes a free token. The openfootball
  calendar needs none.

---

## Quick start

```bash
git clone <your-repo> football-edge && cd football-edge
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_database.py --seasons 10        # ~2 min, from the CSVs in the repo
python scripts/build_xg.py                           # expected goals (the model default)
python scripts/build_fixtures.py                     # the season calendar the home page needs
python scripts/build_evidence.py                     # ~6 min; what each market's record is
streamlit run app.py
```

`build_evidence.py` is the one that is easy to skip and should not be. It runs
a walk-forward per league and stores what each market's calibration and closing
line value actually are, and everything that shows a price reads it. Without it
every market is labelled UNTESTED — which is accurate, and useless.

**Then, for the forward-looking half:**

```bash
python scripts/snapshot_fixtures.py                  # archive this week's prices
python scripts/scan_fixtures.py --league E0          # what the model disagrees with
python scripts/build_calendar.py                     # the rest of the season's fixtures
```

**Run `snapshot_fixtures.py` on a schedule if you run nothing else.** The
source overwrites `fixtures.csv` every time it rebuilds it and archives
nothing, so a pre-match price not captured when it is published is gone
permanently. Every other table here rebuilds from static files in two minutes;
that one cannot be rebuilt at all.

For the same reason it is the one table mirrored to **tracked** CSV, in
`data/snapshots/`. The database is not in the repository because it rebuilds
itself; this does not, so commit that directory after each run. It is the only
copy of those prices, and `snapshots.import_export` restores it into a fresh
clone.

**The database is not in the repository** - it is a 22MB binary that changes
most sessions, and it is rebuilt by the commands above. The season CSVs it is
built from *are* committed, so the first step needs no network. See
`.gitignore` for what else is left out and why.

The browser opens on the calendar. Pick a date, then a fixture for the detail
page.

Two optional extras, both needing a free API-Football key in
`FOOTBALL_API_KEY`:

```bash
python scripts/build_crests.py                       # club badges
python scripts/build_injuries.py --season 2024       # injuries (free plan stops at 2024)
```

Without them the app still works: clubs fall back to a generated monogram and
the injury panel says what is missing.

**No internet, or want to see it working before downloading anything?** The
project ships a generator that produces synthetic CSVs in the source's exact
format, awkward edge cases included:

```bash
python scripts/make_sample_data.py --out data/sample --seasons 4
python scripts/build_database.py --local-dir data/sample --seasons 4
streamlit run app.py
```

There are terminal versions too, which are faster for quick checks:

```bash
python scripts/show_fixture.py Arsenal Liverpool          # what has happened
python scripts/predict_fixture.py Arsenal Liverpool       # what the model thinks
python scripts/predict_fixture.py Inter Milan --ratings --league I1
python scripts/backtest.py --league E0 --from 2022-08-01  # is it any good?
python scripts/tune_hyperparameters.py --league E0 --from 2019-08-01
```

Re-run `build_database.py` after each round of matches. It is idempotent:
finished seasons come from the local cache, existing rows are replaced rather
than duplicated, and only the in-progress season is re-downloaded.

---

## Where the data comes from

[football-data.co.uk](https://www.football-data.co.uk), maintained by Joseph
Buchdahl. Plain CSVs over HTTP, one per league-season, free, no key, no quota.

Per match it carries goals (full time and half time), shots, shots on target,
corners, fouls, yellows, reds, the referee in some leagues, and — the part that
makes it unusual — historical odds from around twenty bookmakers, both an
earlier price and a closing price, in the same row. That combination is what
lets the whole project be backtested honestly without paying anyone.

The odds columns are mapped from the source's own key at
[notes.txt](https://www.football-data.co.uk/notes.txt), and the loader picks up
every book listed there. Two are worth knowing about:

**Betfair Exchange** charges commission rather than building a margin into the
price, so its overround is a fraction of a bookmaker's. Its closing price is
the most accurate probability estimate in the file, and it is the default
benchmark the backtest measures against — worth using even if you never bet
there.

**The market maximum** is the best price available anywhere at the time, which
is what someone shopping across several accounts would actually get. It is the
default price the backtest assumes you took. Pass `--book bet365` to model what
one account would really have offered you instead.

```bash
python scripts/backtest.py --league E0 --from 2022-08-01 --margins
```

prints each bookmaker's average overround. A sharp book sits near 1.02 and a
soft one nearer 1.07, and that gap is usually larger than any edge a model of
this kind will find — which is the single most useful thing to know before
betting anywhere.

Two quirks in the data, both from the source's own notes. Prices are collected
on Friday afternoons for weekend fixtures and Tuesday afternoons for midweek
ones, so the "earlier" price is not the market open. And English and Scottish
yellow cards exclude the first yellow when a second turns it into a red, while
European competitions count both — so English card totals run slightly below
continental ones for the same events. Models are fitted per league, so the
difference is absorbed, but the raw numbers are not comparable across borders.

**What it does not have**, and what that means:

| Gap | Consequence |
|---|---|
| No UEFA competitions | Top-5 domestic only. FBref was the plan and is gone — see the roadmap |
| No xG | Supplied by Understat instead, and the blend is the shipping default |
| No corner or card prices | Those models can be calibrated but never backtested as bets |
| No BTTS, double-chance or half-time prices | Same: modelled and calibrated, never bettable here |
| Referee missing in some leagues | Card models are league-dependent, and the app says which applies |
| Corners/shots thin before ~2005 | Ten seasons of history is the practical depth |
| One round of upcoming fixtures, overwritten weekly | Archived on every pull, because it cannot be recovered |

The build prints a coverage report showing exactly which columns are populated
for which league-seasons, so none of this is a surprise later.

---

## How it is put together

```
fbedge/
  config.py      leagues, seasons, paths, URLs - every tunable in one place
  ingest.py      download with caching, retries and a polite delay
  normalize.py   column mapping, team-name canonicalisation, derived metrics
  database.py    DuckDB schema, idempotent loading, the team_matches view
  quality.py     coverage report and integrity assertions
  profile.py     the descriptive fixture card
  fixtures.py    the season calendar, played and unplayed
  calendar.py    the same, from three sources behind one interface
  snapshots.py   the append-only archive of the weekly fixtures file
  evidence.py    what track record each market on each league actually has
  injuries.py    the external injury feed (optional key, daily budget enforced)
  crests.py      club badges, downloaded once and inlined
  models/
    base.py      time decay, ridge priors, point-in-time training sets
    goals.py     Dixon-Coles: attack, defence, home advantage, low-score fix
    hierarchical.py  estimating the shrinkage instead of choosing it
    counts.py    negative binomial corners and cards, with referee effects
  markets.py     score matrix -> every goal market; count totals
  pricing.py     implied probability, margin removal, expected value
  predict.py     one fixture in, a full set of fair prices out
  settlement.py  what a bet actually returned, quarter lines included
  backtest.py    walk-forward: refit, price the book's lines, settle
  evaluation.py  calibration, closing line value, staking, bootstrap
  understat.py   expected goals and rosters: fetch, cache, name mapping
  availability.py who is missing, from strictly earlier matches only
scripts/
  build_database.py    one command: download, normalise, load, verify
  snapshot_fixtures.py archive this week's prices before they are overwritten
  scan_fixtures.py     the pre-match EV scan, every row labelled with its record
  build_evidence.py    walk-forward per league, stored so a scan can read it
  build_calendar.py    the remaining season, from openfootball or football-data.org
  make_sample_data.py  synthetic CSVs for offline work and tests
  show_fixture.py      the text version of the descriptive card
  predict_fixture.py   the text version of the model forecast
  backtest.py          walk-forward backtest with a full report
  tune_hyperparameters.py  grid search with a held-out window
  build_xg.py          download Understat xG and attach it to matches
  build_rosters.py     download per-match line-ups into match_lineups
  availability_signal.py  does knowing who is missing predict anything
  validate_setting.py  one candidate setting against the shipping default
  compare_targets.py   goals vs xG vs blend, scored on the same matches
  season_breakdown.py  CLV by season, with a benchmark-change warning
  goals_shape.py       is the goals distribution the right shape
app.py                 the Streamlit web app
scan_view.py           the pre-match scan, as a tab in it
tests/                 543 tests, run with: python -m pytest
```

Four design decisions worth knowing about:

**DuckDB, not Postgres.** A single file, no server, no password, no monthly
cost, and a columnar engine is exactly right for a workload that is almost all
analytical scans. The schema ports to Postgres later if this ever needs
concurrent writers.

**The `team_matches` view.** Every match becomes two rows, one per team, with
columns relabelled "for" and "against". Every statistic in the project is then
a filter and an average over that view, so home/away logic lives in exactly one
place instead of being re-derived in every query.

**Odds in long format.** One row per price, rather than a hundred wide columns.
This paid off exactly as intended in Phase 4: the archive of upcoming prices is
the same shape as the historical table and reuses the same reader, so adding it
was a data change rather than a schema migration.

**Point-in-time correctness, enforced by tests.** Every query filters on
`date < as_of`. Nothing can see a match that had not been played at the moment
being asked about. This is not tidiness: Phase 3's walk-forward backtest calls
these same functions with historical dates, and one leak of future information
would produce a backtest showing a spectacular edge that does not exist. Two of
the tests exist solely to guard this.

---

## What this can actually tell you

Every market below comes from one fit. What differs is the evidence behind it,
and there are only three possibilities:

- **Backtested.** The source carries historical prices, so the model's
  selections were settled as bets against a real market and closing line value
  is measurable. Three markets qualify. That is not a shortlist, it is the
  complete list of prices football-data.co.uk publishes.
- **Calibration only.** No price exists to bet into, but the outcome is
  recorded, so "when the model says 30%, does it happen 30% of the time" is
  answerable and has been answered. **"The model is well calibrated on corners"
  and "there is an edge in corner markets" are different claims, and only the
  first is supported.** Confirming the second needs a paid odds feed.
- **Untested.** Priced, and no calibration run has scored it yet.

Figures below are from `scripts/build_evidence.py` over 2022-08-01 onward, all
five leagues, ranges across leagues rather than a pooled average — a card model
in France and one in England are not the same measurement. The app and the scan
show the per-league figure, never this table's range.

| Market | Status | Calibration slope | n | CLV |
|---|---|---|---|---|
| 1X2 | backtested | 0.98–1.20 | 21,543 | −1.49% over 8,881 bets |
| Over/under goals | backtested | 0.98–1.04 | 71,810 | −1.71% over 4,869 bets |
| Asian handicap | backtested | 1.01–1.26 | 57,712 | −1.39% over 5,368 bets |
| Double chance | calibration only | 0.98–1.20 | 21,543 | no price exists |
| Draw no bet | calibration only | 1.06–1.26 | 10,740 | no price exists |
| Winning margin | calibration only | 0.98–1.24 | 50,267 | no price exists |
| Both teams to score | calibration only | **0.33–0.93** | 14,362 | no price exists |
| Team goals (home/away) | calibration only | 0.95–1.08 | 43,086 | no price exists |
| Half-time result | calibration only | 0.82–0.86 | 21,540 | no price exists |
| Half-time goals | calibration only | 0.91–1.01 | 43,080 | no price exists |
| Total corners | calibration only | 0.81–0.89 | 92,694 | no price exists |
| Team corners | calibration only | 0.81–0.94 | 43,080 | no price exists |
| Corner handicap | calibration only | 0.90–1.10 | 40,996 | no price exists |
| Total cards | calibration only | 0.91–1.03 | 76,006 | no price exists |
| Team cards | calibration only | 0.89–1.09 | 43,039 | no price exists |
| Card handicap | calibration only | 1.11–1.26 | 38,434 | no price exists |
| Correct score | untested | — | — | no price exists |

**Fifteen of ninety-five market-league pairs are backtested.** The other eighty
are modelled and checked and have never been bet into, and the app says so next
to every one of them.

Read the slope as: 1.0 is right, below 1 means the probabilities are spread too
far apart, above 1 means they hedge towards the base rate. Two rows are worth
singling out because the numbers are unflattering and the table is not here to
flatter:

**Both teams to score is over-confident**, at 0.33 to 0.93. The model separates
fixtures on BTTS more sharply than the results justify, in every league.

**The card handicap is the opposite**, at 1.11 to 1.26, and the reason is
written down in advance: it is derived by treating the two teams' card counts
as independent, which overstates how much their *difference* scatters, so the
prices sit too close to even money. `markets.count_difference` says so in its
docstring, and the measurement agrees with the prediction.

**One market was measured and then removed.** Odd/even total goals is trivially
derivable from the score matrix, so it was priced. Its calibration slope came
back between −2.41 and +1.54 across the five leagues — in at least one, the
model's confidence pointed the wrong way — which is what a market close to a
coin flip by construction looks like when a model has nothing to say about it.
It was withdrawn on 2026-09-04 rather than kept with a warning label, because a
market the model cannot rank is not worth the row it occupies. `evidence.py`
still knows the name and reports it as withdrawn, so an old stored row is
distinguishable from a market that simply has no data. That is the point of
measuring: finding out what to stop doing counts as a result.

---

## How the model works

**Goals: Dixon-Coles.** Each team gets an attack rating and a defence rating,
fitted alongside a home advantage. Two ratings and the venue give two scoring
rates for any fixture, and expanding those into a matrix of every plausible
scoreline gives 1X2, over/under at any line, both teams to score, Asian
handicap and correct score from a single fit. Dixon and Coles' 1997
contribution was noticing that two independent Poissons get the low scores
wrong — 0-0, 1-0, 0-1 and 1-1 do not occur at the rate independence predicts —
and adding a correction for exactly those four cells.

Because every price comes from one matrix, they cannot contradict each other.
A system that fits each market separately will happily quote an over 2.5 that
is inconsistent with its own correct-score prices. That used to be true by
construction and is now *checked*: `tests/test_market_consistency.py` asserts
twelve identities between markets, so a refactor that breaks the shared
derivation fails there rather than in front of a reader comparing two prices on
one screen.

**Half-time markets are the exception, and they get their own fit.** They are
not derivable from the full-time matrix, and halving the full-time rates would
be wrong in two directions at once. On the Premier League, 44.6% of goals
arrive before the interval rather than 50%, and the fitted half-time home
advantage is *larger* than the full-time one — 0.33 against 0.20. So
`TrainingSet.half_time()` hands the same Dixon-Coles machinery the half-time
columns and gets an independent fit with its own ratings, its own home
advantage and its own low-score correction. It is off by default because it
doubles the fitting time, and `markets.price_selection` returns None for a
half-time market rather than approximating one.

**Corners and cards: negative binomial.** Both are overdispersed — their
variance exceeds their mean — so a Poisson fit would be too confident about the
middle and would systematically underprice the tails, which is exactly where
over/under lines sit. The negative binomial adds a dispersion parameter that
widens the distribution to match reality.

Two details this gets right. Referees get their own card multiplier, heavily
shrunk because even a busy referee handles about thirty matches a season. And
the match total is fitted separately from the two team rates: adding two
independent distributions would understate how much totals actually scatter,
because both teams win more corners in an open game than a closed one.

**Time decay and shrinkage.** These are the answer to the problem that made the
descriptive layer untrustworthy on its own. Every match carries a weight of
`0.5 ** (age_in_days / half_life)`, so several seasons can be fitted at once
without an ancient result counting as much as a recent one. And a ridge penalty
pulls each team's rating towards the league average with a fixed strength, so a
team with three matches is pulled most of the way back while a team with a
hundred barely moves.

That is what stops a side that won its first two games 5-0 from being modelled
as the best team in Europe, and it is why the model produces a sane number in
August at all. Newly promoted clubs are shrunk towards a deliberately
pessimistic prior rather than the league average, because "unknown" and
"average" are not the same claim.

The strength of that penalty is *chosen*, not fitted, and `models/hierarchical.py`
exists because that looked like an obvious gap. It is not: the ridge is
algebraically a Gaussian prior on team strengths, so the prior variance can be
estimated by empirical Bayes, and doing so gives an answer that is decisively
worse out of sample - 0/4 held-out leagues, with the calibration slope blowing
out to 1.7-2.2. The estimator is right and the question it answers is subtly
the wrong one; the module docstring and the hierarchical section of HANDOFF.md
explain why. `ridge="auto"` is kept so the disagreement can be reproduced, and
should stay off.

Both knobs are exposed in the app's sidebar and on the command line. The
defaults started as conventional values from the literature, and Phase 3's
tuner has since searched them on a walk-forward split — and reported that the
search fitted noise, so they stayed. See "Tuning, without fooling yourself".

**How it is checked.** The analytic gradients are verified against numerical
differentiation in the test suite, because a wrong gradient does not crash — it
quietly converges to the wrong parameters and every price downstream is subtly
off. The model is also fitted to synthetic data whose true team strengths are
known, and has to recover them. Whether it beats the market is a separate
question, answered by the backtest below.

---

## Judging the model

```bash
python scripts/backtest.py --league E0 --from 2022-08-01
```

The engine steps through history a week at a time. At each step it fits the
models on matches played before that week, prices every selection the
bookmaker actually offered, and settles the result. Nothing is ever fitted on
a match it later predicts, and two tests exist purely to enforce that.

Pricing the book's own line matters more than it sounds. Asking the model for
over 2.5 when the bookmaker was offering over 3.0 compares nothing, so the
backtest is driven by the odds table rather than by a fixed ladder of lines.

Read the output in this order.

**Calibration first.** When the model says 30%, does it happen 30% of the time?
If it does not, nothing below matters. A few percent of overconfidence is
enough on its own to turn an apparent edge into a real loss.

**Then model versus market log loss.** A direct comparison against the closing
prices with the margin stripped out, on identical matches. This is the cleanest
single answer to "does the model know something the market does not".

**Then closing line value.** This is the headline. It measures whether the
price you took was better than the margin-free closing line said it was worth:
taking 2.10 on something the close valued at 2.30 is closing line value whether
or not the bet won. Positive CLV over a few hundred bets is the strongest
evidence available that an edge is real. Negative CLV alongside a profit means
the profit was luck and will not survive.

The free data carries both an earlier price and a closing price for each match,
which is exactly what makes this measurable without paying a data vendor.

**Returns last, and with an interval.** ROI is reported with a 95% confidence
interval from a bootstrap that resamples whole matchdays rather than individual
bets, because several bets on one day share a model fit and often a match.
The interval is usually wide enough to contain both a healthy profit and a
serious loss, and that width is the finding. A backtest that reports "+7% ROI"
without saying "give or take fifteen points" is not telling you anything.

### Tuning, without fooling yourself

```bash
python scripts/tune_hyperparameters.py --league E0 --from 2019-08-01
```

This searches the half-life and shrinkage over a grid. Two decisions keep it
honest. It scores on log loss rather than profit, because over a few hundred
bets picking the highest-ROI grid point reliably selects the luckiest setting
rather than the best one. And it splits the window: everything is searched on
the development portion, then the single winner is re-run once on a later
holdout it never influenced. If the holdout score is worse, the search found
noise and the script says so.

### What this data cannot tell you

The source has no historical corner or card prices — and no BTTS,
double-chance, team-total or half-time prices either. Those models can be
checked for calibration against what actually happened, but they cannot be
backtested as bets, because there is nothing to bet into. If the corner markets
turn out to be where the edge is, confirming that needs a paid odds feed —
which is the first thing worth spending money on, and only once the goal models
have earned it.

The same limitation has a forward-looking twin. A match with results and no
odds is fitted, priced, and counted towards calibration, and can never be
settled as a bet: the record carries no price, so it has no expected value, and
`BacktestResult.bets` requires one. Every run prints how many matches were
fitted but not bettable. That is the guard the roadmap needs before UEFA
results ever enter the database.

---

## Sample sizes are part of the answer

Every number the app shows carries its `n`. Two matches into a season, a team's
"2.5 goals per game" rests on two matches, and the interface says so rather than
quietly presenting it next to a figure built on 380.

This is the whole reason the two layers sit on separate tabs. The card shows the
raw truth about a small sample; the model turns thin samples into usable
estimates by shrinking them towards league averages and blending in earlier
seasons. Mixing them on one screen would invite you to trust "Arsenal will
score 2.1" as much as "Arsenal have scored 2.0 per game", when the first is a
modelled estimate and the second is a fact about two matches.

Head-to-head is shown for the same reason it is flagged: two league meetings a
season means a decade of history is about twenty matches played by squads that
have completely turned over. The models do not use it.

---

## Roadmap

| Phase | What it adds | Status |
|---|---|---|
| 1 | Data spine, quality checks, fixture profiles, web app | **done** |
| 2 | Dixon-Coles goals model, negative-binomial corners and cards | **done** |
| 3 | Walk-forward backtest, calibration, closing line value, tuning | **done** |
| 4 | Fixture archive, pre-match EV scan, forward calendar, market coverage | **done** |
| 4b | Paper-trading ledger: record every claim, settle it, measure CLV forward | **done** |
| 5a | xG from Understat | **mostly done** |
| 5b | UEFA competitions, **results only** — see below | blocked on a source |
| 5c | Fatigue and congestion, line-ups | |

Phase 4 points the same machinery at fixtures that have not been played yet.
The engine already existed — a pre-match scan is a backtest with the results
column missing — so the work was the parts around it: an append-only archive of
the source's twice-weekly price file, which is overwritten weekly and cannot be
reconstructed afterwards; a whole-season calendar; and the evidence labelling
that stops a price ever reaching a reader without its track record.

**It is a pre-match scan, not a live scanner**, which is what the roadmap used
to call it. Prices in this source are collected on Friday afternoons no later
than 17:00 British time for weekend fixtures and Tuesdays no later than 13:00
for midweek ones. Nothing here watches a market move.

### The paper-trading ledger, and why it comes before any model work

The rule that stops this project staking money says: paper-trade for several
weeks recording what would have been bet, and judge it on the closing line
value from *that* period rather than on the historical backtest. Nothing
recorded what the scan claimed, so that rule described a procedure nobody could
carry out.

```
python scripts/scan_fixtures.py --record     # file today's board
python scripts/paper_trade.py                # settle it, and read the record
```

The app carries the same record as a **Paper ledger** tab — open bets, settled
claims and withheld picks, each in its own view. That page only ever reads:
recording and settling are writes to an append-only record, and a page that
mutated one every time it loaded would corrupt the one measurement here that
cannot be rebuilt from static inputs.

Every claim is stored with the model settings that produced it — resolved, not
requested — because a "+6% EV" recorded six weeks ago is worthless if the
defaults moved underneath it. The claim is immutable and its outcome lives in a
separate table, so a bet cannot be re-settled under changed code once the
result is known. Withheld selections are recorded too, flagged unstaked, so the
two thresholds in the table above can eventually be marked against what
actually happened rather than only against the backtest that set them.

**It reports nothing for weeks, by design.** Prices reach about three days
ahead, so a young ledger has a handful of settled bets and a mean over a
handful of bets is noise. The report prints a clustered standard error beside
every figure and refuses to characterise a mean under thirty bets.

The honest expectation is that this eventually confirms the backtest at around
−1.5%, measured forward instead of backward. That would be a success for the
ledger and the second independent confirmation that the model has no edge.

### Phase 5 was reordered, and half of it was rescoped

**Understat xG comes first**, because it is nearly delivered and is the
higher-value half: `understat.py` and `build_xg.py` exist, `compare_targets.py`
scores goals against xG against a blend on identical matches, and the blend is
already the shipping default.

**The FBref half is gone.** On 20 January 2026 Stats Perform (Opta) terminated
FBref's data agreement and required the removal of all advanced statistics;
Sports Reference announced it on 23 January and did not contest it, citing
legal costs. Results, schedules, squads and basic match statistics remain on
the site. xG and every Opta-derived advanced metric do not, and are not coming
back.

Two things follow, and the second was found while checking the first:

1. **UEFA work is rescoped to results only.** There is no free xG for European
   competition at any quality. Understat covers the big five domestic leagues
   plus the RFPL from 2014/15 and has no UEFA coverage at all, so a European
   fixture cannot be priced on the same footing as a domestic one however the
   results arrive.
2. **FBref is no longer usable as a source here even for results.** Verified
   2026-09-04: `fbref.com` returns HTTP 403 to a scripted request, with a
   browser user-agent and through two separate network paths. The results-only
   rescope therefore needs a different source, and the free tier of
   football-data.org — already wired up in `fbedge/calendar.py` — includes the
   Champions League. openfootball has a Champions League file for 2024/25 but
   not for 2025/26 or 2026/27, so it is not the one to rely on.

**The backtest is guarded for this already.** UEFA matches would enter the
database with results and no odds, and a bet cannot be settled against a price
that does not exist. Matches with no odds are fitted and priced — they inform
team strengths and count towards calibration — and can never become bets,
because a record with no price has no expected value and `BacktestResult.bets`
requires one. Every run prints how many matches were fitted but not bettable.
This is the same limitation already documented for corners and cards, and it is
the same mechanism: no price, no bet, no closing line value.

---

## An honest note on what this is for

**Phase 3 said the models are worse than the closing line, and they still are.**
That was the expected result and it is the measured one: closing line value on
the backtested markets runs about −1.5% across five leagues, and the model's own
log loss has trailed the market throughout. The closing price at a sharp
bookmaker aggregates an enormous amount of information, and beating it is
genuinely difficult.

That is why the backtest's primary measure is closing line value rather than
profit: over a few hundred bets, return on investment is almost pure noise,
while whether you consistently beat the closing price is measurable in weeks.

**Phase 4 makes this more important, not less.** A scan output is the first
thing this project produces that looks like a betting tip, and it is not one. A
positive expected value means the model disagrees with the price — and the
evidence above is that when this model disagrees with a closing line, the model
is usually the one that is wrong.

**So the scan refuses to rank its own biggest numbers.** The first live run put
two newly promoted clubs at the top of the table at +68% and +50%, on two
matches of history each, which is what a model disagreeing out of ignorance
looks like. Two ceilings now stop that, both set from measurement:

| Rule | Threshold | What it is based on |
|---|---|---|
| Too little history | either side under 5 matches | Across five leagues and 19,112 settled bets, fixtures where the thinner side had 0–4 matches carried a mean expected value of **+17.5% against +12.8%** elsewhere — and did not go on to do better |
| Too large to believe | expected value over +20% | On E0 2022–26, +20% is the 78th percentile of the model's own claimed edges. Bets above it returned −5.2% against +2.1% for the rest |

Read the second row carefully: over a few hundred bets ROI is close to noise,
so that is *an absence of evidence that the big numbers are better*, not proof
they are worse. It is enough. A model whose closing line value is −1.5% over
nine seasons is not finding +40% edges, and a table sorted on expected value
puts its largest errors at the top.

Withheld selections are **printed under their own heading with the reason, not
dropped** — a fixture missing from a scan looks exactly like a fixture nobody
priced, and those must stay distinguishable. `--include-withheld` puts them
back in the sort, and `--min-matches` / `--max-ev` move the thresholds.

Treat this as a software and statistics project, which is where its real value
is. Nothing should be staked until a full backtest plus a stretch of
paper-trading says otherwise, and whatever is eventually staked should be money
you would have spent on a hobby.

---

## Licence and attribution

This is the one section that has to be exactly right, so it lists every
external source the project touches, including the optional ones.

**[football-data.co.uk](https://www.football-data.co.uk)**, maintained by
Joseph Buchdahl. Match results and historical odds, and the upcoming-fixtures
file the pre-match scan reads. © football-data.co.uk and used under their terms
for personal, non-commercial analysis.

**[Understat](https://understat.com)**. Expected goals, per-match line-ups and
the season calendar the home page runs on. `fbedge/understat.py` reads the
`getLeagueData` JSON endpoint the site's own page calls — this is scraping, not
a published API, and calling it anything else would be dishonest. Every
response is cached to disk so a backfill happens once: five leagues and nine
seasons is forty-five requests. It is a free service run by somebody else and
this project has no claim on it; if you fork this, keep the cache and the
delay.

**[openfootball/football.json](https://github.com/openfootball/football.json)**
(optional). Season calendars as plain JSON, public domain, no key. Community
maintained, so expect occasional gaps — it has a Champions League file for
2024/25 and none for 2025/26.

**[football-data.org](https://www.football-data.org)** (optional, free token).
The alternative calendar source. Free tier: twelve competitions, ten calls a
minute, delayed — the limit is enforced client-side here rather than left to
the server to reject.

**[API-Football](https://www.api-football.com)** (optional, free key). Club
badges and injury news. Free tier: 100 requests a day resetting at 00:00 UTC,
10 a minute, recent seasons only. Both limits are enforced locally with a
persistent counter and a hard stop, because a spent budget at this endpoint
does not always fail loudly.

**Club badges are not redistributed.** They are third-party crests from a
provider's CDN and `.gitignore` keeps them out of the repository for that
reason, not for size. `crests.monogram` is why the app still looks finished
without them.

**The synthetic data generator** produces numbers from plausible distributions,
not real matches; never fit or evaluate a model on it.
