# football-edge

A football analytics platform for the top five European leagues: Premier League,
La Liga, Serie A, Bundesliga and Ligue 1.

**Phases 1 to 3 are built.** Phase 1 is the data spine: it downloads a decade
of real match data, normalises it, stores it and checks it. Phase 2 is the
model: fitted team ratings that produce probabilities and fair prices for every
market — 1X2, over/under, both teams to score, Asian handicap, corners, cards,
correct score. Phase 3 is the part that decides whether any of it is worth
believing: a walk-forward backtest measuring calibration, closing line value
and returns with honest confidence intervals.

Total cost to run: nothing. No API key, no subscription, no scraping.

---

## Quick start

```bash
git clone <your-repo> football-edge && cd football-edge
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_database.py --seasons 10        # ~5 min, ~50 CSV downloads
streamlit run app.py
```

The browser opens on the fixture profile page. Pick a league, two teams, and a
knowledge cut-off date.

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
| No UEFA competitions | Top-5 domestic only until Phase 5 adds FBref |
| No xG | Goal models run on actual goals for now, which is noisier |
| Referee missing in some leagues | Card models are league-dependent |
| Corners/shots thin before ~2005 | Ten seasons of history is the practical depth |

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
  models/
    base.py      time decay, ridge priors, point-in-time training sets
    goals.py     Dixon-Coles: attack, defence, home advantage, low-score fix
    counts.py    negative binomial corners and cards, with referee effects
  markets.py     score matrix -> every goal market; count totals
  pricing.py     implied probability, margin removal, expected value
  predict.py     one fixture in, a full set of fair prices out
  settlement.py  what a bet actually returned, quarter lines included
  backtest.py    walk-forward: refit, price the book's lines, settle
  evaluation.py  calibration, closing line value, staking, bootstrap
scripts/
  build_database.py    one command: download, normalise, load, verify
  make_sample_data.py  synthetic CSVs for offline work and tests
  show_fixture.py      the text version of the descriptive card
  predict_fixture.py   the text version of the model forecast
  backtest.py          walk-forward backtest with a full report
  tune_hyperparameters.py  grid search with a held-out window
app.py                 the Streamlit web app
tests/                 184 tests, run with: python -m pytest
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
Adding a bookmaker or a market in Phase 4 becomes a data change, not a schema
migration.

**Point-in-time correctness, enforced by tests.** Every query filters on
`date < as_of`. Nothing can see a match that had not been played at the moment
being asked about. This is not tidiness: Phase 3's walk-forward backtest calls
these same functions with historical dates, and one leak of future information
would produce a backtest showing a spectacular edge that does not exist. Two of
the tests exist solely to guard this.

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
is inconsistent with its own correct-score prices.

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

Both knobs are exposed in the app's sidebar and on the command line. The
defaults are conventional values from the literature; Phase 3 will tune them by
walk-forward validation rather than by taste.

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

The source has no historical corner or card prices. Those models can be checked
for calibration against what actually happened, but they cannot be backtested
as bets, because there is nothing to bet into. If the corner markets turn out to
be where the edge is, confirming that needs a paid odds feed — which is the
first thing worth spending money on, and only once the goal models have earned
it.

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
| 4 | Upcoming fixtures, live EV scanner against book prices | next |
| 5 | xG from Understat, UEFA via FBref, fatigue, lineups | |

Phase 4 points the same machinery at fixtures that have not been played yet:
pulling the upcoming list and current prices, running the model against them,
and surfacing anything it thinks is mispriced. The engine for that already
exists — a live scan is a backtest with the results column missing.

---

## An honest note on what this is for

Phase 3 will probably tell you the first models are worse than the closing line.
That is the normal result. The closing price at a sharp bookmaker aggregates an
enormous amount of information, and beating it is genuinely difficult.

That is why the backtest's primary measure is closing line value rather than
profit: over a few hundred bets, return on investment is almost pure noise,
while whether you consistently beat the closing price is measurable in weeks.

Treat this as a software and statistics project, which is where its real value
is. Nothing should be staked until a full backtest plus a stretch of
paper-trading says otherwise, and whatever is eventually staked should be money
you would have spent on a hobby.

---

## Licence and attribution

Match data is © football-data.co.uk and used under their terms for personal,
non-commercial analysis. The synthetic data generator produces numbers from
plausible distributions, not real matches; never fit or evaluate a model on it.
