# Project handoff: football-edge (supersedes the previous handoff)

Read this whole document before writing or changing any code. It replaces the
earlier handoff. Phases 1–3 are complete and the Phase 4 decision gate has now
been evaluated. **The gate is closed.** Do not build a live betting scanner.

Everything in the previous handoff's "What already exists", "Bugs already found
and fixed", and "Working style to maintain" sections still applies unchanged.
The short version of the working style, because it matters more than anything
else here: ask before big decisions rather than surprising the person with
them; work in phases and confirm the previous one works before building on it;
write docstrings that explain *why*, not what; new code meets the existing test
bar, not a lower one. Be honest and calibrated, not encouraging. This is
gambling-adjacent, and nobody should stake real money on it.

---

## Current status

- 244 tests passing (`python -m pytest`). Was 205; 39 added this session.
- **The CLV measurement was broken, and is now fixed and re-baselined.** A
  benchmark change in 2024 plus favourite-longshot bias in multiplicative
  margin removal inflated every historical CLV figure. `margin_method` now
  defaults to Shin, which measures unbiased against the exchange. The gate
  verdict and Finding 1 have both been restated on the new scale in place; the
  gate is **−1.500% (−9.1 SE)**, worse than the −0.944% it replaces.
- **Model log loss and the tuner are unaffected**, and verified so: they never
  touch `remove_margin`. Only CLV and `market_log_loss` moved.
- Phase 3 backtest and hyperparameter tuner have both been run against real
  data on the Premier League.
- **No demonstrated edge**, and the current era is measurably *negative*. See
  the gate verdict below and the RESOLVED section that supersedes it.
- Phase 4 not started, and should not be started.
- Findings 1 and 2 have both now been tested and both were wrong about their
  proposed cause. Read the RESOLVED section before the two findings below it.

---

## The decision gate: closed

**Restated on the Shin scale.** The figures in this section were recomputed
after the margin-removal fix; the superseded multiplicative numbers are shown
alongside so the size of the correction is visible.

Backtest, E0, 2022-08-01 to 2026-08-31, defaults (half-life 180d, ridge 5),
n=4379 bets:

| | multiplicative (old) | **Shin (current)** |
|---|---|---|
| mean CLV | −0.944% | **−1.500%** |
| clustered SE | 0.163% | 0.165% |
| z | −5.8 | **−9.1** |

Two corrections, not one. The scale changed, and the old headline also quoted
**7.5 SE using the naive standard error**, which the project's own
`clustered_mean` shows overstates precision here. Clustered by match it was
−5.8 SE then and is −9.1 SE now. The verdict moves in the same direction
either way: further from zero, not closer.

- Model log loss is **unaffected** at 0.5767 against a base rate of 0.6365, and
  still trails the market (0.5626). The model knows something real about match
  outcomes; the closing line knows more. Only the *market* side of that
  comparison depends on margin removal, and it barely moves (0.5628 → 0.5626).

Tuner, E0, dev 2019-08-01 to 2024-07-15, holdout 2024-07-15 onward:

**The tuner needs no re-run.** It sorts the grid on `model_log_loss`, which is
computed from the model's probabilities and the settled outcomes and never
touches `remove_margin`. Verified directly: dev-window model log loss is
0.571182 under both methods, identical to six decimal places. Everything below
therefore stands as written.

- Best development setting: half-life 360d, ridge 2.0 (log loss 0.5691).
- On the holdout it scored 0.5885 against 0.5880 for the defaults. The tuned
  winner was very slightly *worse* out of sample.
- The tuner worked correctly and reported that the search fitted noise. **Keep
  the defaults.** Do not re-run the grid hoping for a different answer.

Minor note: 360d was the longest half-life on the grid and it won, which
normally means extend the grid. Don't. The holdout says the gain isn't real,
and log loss across the entire grid spans 0.028 while the dev-to-holdout jump
is 0.019 — the difference between eras is comparable to the difference between
every setting tried.

---

## Finding 1: CLV may be era-dependent, and this is the live question

> **DEAD. The premise was a measurement artifact.** The two figures below
> disagreed in sign, and that disagreement is what launched the entire era
> investigation. Restated on the Shin scale, on the same windows and the same
> bets, they agree:
>
> | 1X2 only | multiplicative (old) | **Shin (current)** |
> |---|---|---|
> | backtest 2022-08 → 2026-08, n=2012 | −0.52% (−1.9 SE) | **−1.46% (−5.3 SE)** |
> | tuner dev 2019-08 → 2024-07, n=2553 | +1.21% (+5.0 SE) | **−0.24% (−1.0 SE)** |
>
> The "+0.89%" that looked like a 5-sigma positive edge is −0.24% and
> indistinguishable from zero once the longshot bias is removed. Nothing needed
> reconciling; the ruler was bent. The reasoning below is kept for audit only —
> it is a careful chain of inference from two numbers that were both wrong.
>
> (The old tuner-window figure reproduces here at +1.21% on 2553 bets rather
> than the recorded +0.89% on 2446; the dev window's end date is computed from
> a holdout fraction and this re-run approximates it as 2024-07-15. The sign
> and magnitude match, which is all that mattered.)

The two runs above disagree about the sign of CLV, and the code says they
shouldn't. Both call the same `evaluation.closing_line_value`, both use
`edge_threshold=0.02`, both settle through the same `_score_match`. Verified by
reading, not assumed.

- Backtest, 1X2 only, 2022-08 to 2026-08: **−0.52%** on 2012 bets.
- Tuner development window, 1X2 only, 2019-08 to 2024-07: **+0.89%** on 2446 bets.

Using the per-bet spread implied by the backtest's own SE (~8.3% per bet),
+0.89% on 2446 bets is roughly 5 SEs above zero, and still ~3.5 SEs even if
1X2-only bets are half again as noisy as the pooled figure.

Backing the overlap (2022-23 and 2023-24) out of the two aggregates implies the
2019-20 through 2021-22 stretch carries roughly **+1.8% mean CLV** on its own.
Those are the seasons played without crowds, when home advantage measurably
moved. A model that re-estimates home advantage from recent results with time
decay would track that faster than a price that has to be moved by hand.

**If that's what it is, it is not an edge.** It's a regime that ended. But it
has two consequences that matter:

1. The negative verdict on the current regime gets cleaner, not muddier.
2. The tuner's default window (`--from 2019-08-01`) is three-fifths
   contaminated by a dead regime. Settings chosen there are chosen partly to
   fit a world that no longer exists.

This is an inference from two aggregate numbers, not a measurement. The tool to
confirm or kill it now exists — see "Immediate next step".

---

## RESOLVED: Finding 1 was wrong about the cause, and Finding 2 was wrong about the mechanism

Both were run. Neither survived contact with the data in the form written
below. The original text is kept underneath so the reasoning can be audited,
but **read this section first and treat the two below as superseded.**

### THE MEASUREMENT WAS BROKEN. Read this before any CLV number in this file.

Most of the "CLV regime change" is an artifact of the benchmark changing, not
of the market changing. This supersedes the season table immediately below it
as well as the original gate verdict.

**`betfair_exchange` first appears in the source in 2024-25, and it heads
`FAIR_LINE_PREFERENCE`.** So the margin-free closing line that CLV is measured
against switched from Pinnacle to Betfair in exactly the season CLV appeared to
collapse. Nothing in any output said so.

Holding the benchmark fixed at Pinnacle across the whole window:

| season | 2023 | 2024 | 2025 |
|---|---|---|---|
| as published (benchmark switches) | +0.76% | **−1.56%** | **−1.96%** |
| pinned to Pinnacle | +0.76% | **+0.22%** | **+0.08%** |

2017-2023 are unchanged, because Pinnacle was already supplying them.

**Which benchmark is right? Betfair.** It closes at a 0.56% overround against
Pinnacle's ~2.9%, so the exchange is the closest thing to margin-free truth in
the data. On the 785 bets where both exist, Betfair scores the same bets
**−1.75pp** lower than Pinnacle (SE 0.16, 11 SE), and the gap is stable across
both seasons (−1.78, −1.70).

**Why:** `remove_margin` defaults to `multiplicative`, which has exactly the
favourite-longshot bias `pricing.py`'s own docstring predicted — tested against
Betfair as ground truth, it overstates longshots by +0.60pp and understates
favourites by −0.64pp, monotonically across bands. **The model bets longshots**
(median backed-side closing probability 0.224, 90th percentile 0.406), so the
bias lands almost entirely on the selections that become bets and flatters CLV.

That test had never been run. `pricing.py` says the choice "should be tested
rather than made on aesthetics"; Betfair finally makes it testable, and
**additive wins** — errors roughly 5x flatter across bands, better |error|
(0.00535 vs 0.00555) and RMSE. But additive over-corrects in the other
direction: on the same bets it lands −0.77pp *below* Betfair (−5.1 SE). Neither
method is unbiased; they bracket the truth, with additive about 2.3x closer.

**Consequences, in order of how much they hurt:**

1. **There was probably never a +2% edge.** Under Pinnacle-additive there is no
   positive era anywhere in the series (2017 −0.34%, 2020 +0.43%, 2022 +0.03%,
   2023 −2.67%). Under Pinnacle-multiplicative there is a +2% era. Betfair,
   where it exists, sits between them. Best estimate on a consistent scale: the
   model hovered within about a point of zero through 2022, then drifted
   negative. The "era" story was mostly the ruler.
2. **The COVID/crowds story and my own "the market sharpened" story are both
   dead.** The 2024-25 discontinuity is the instrument.
3. **The gate stays shut,** and for a better reason than before: not "the edge
   went away" but "the edge was never measured on a stable scale, and on the
   most trustworthy scale available it is negative."

### The fix: Shin's method, and the corrected history

Neither existing method is unbiased, and they *bracket* the truth — which is the
signature of a one-parameter family sitting between them. **Shin's method** is
exactly that: it models the margin as the bookmaker's defence against better-
informed traders and solves for the insider share `z` per market, so it can land
between multiplicative and additive instead of being stuck wherever a fixed rule
puts it.

Tested the same way, on 1770 selections where both books priced the same match:

| band | multiplicative | additive | **Shin** |
|---|---|---|---|
| 0.00–0.15 | +0.0060 | −0.0012 | **+0.0007** |
| 0.15–0.25 | +0.0023 | −0.0015 | **−0.0006** |
| 0.25–0.35 | +0.0005 | −0.0009 | **−0.0005** |
| 0.35–0.50 | −0.0011 | +0.0014 | **+0.0008** |
| 0.50–1.00 | −0.0064 | +0.0030 | **+0.0006** |

Flat across the whole range. And on the model's own bets, measured against the
exchange: multiplicative **+1.75pp** (11 SE), additive **−0.77pp** (5.1 SE),
Shin **−0.12pp (0.9 SE)** — indistinguishable from unbiased. Pinnacle-Shin and
Betfair now agree to 0.13pp on 2024, where they disagreed by 1.75pp before.

**`margin_method` now defaults to `"shin"`.** That was a deliberate change, not
a quiet one, and `test_the_default_method_is_shin` pins it.

**The corrected series** (`--from 2017-08-01 --market 1x2`, Shin):

| season | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| mean CLV | +0.26% | −1.30% | −0.39% | +0.96% | +0.13% | +0.62% | −1.78% | −1.92% | −2.42% |
| z | 0.5 | −2.6 | −0.7 | 1.6 | 0.2 | 1.0 | **−3.1** | **−3.7** | **−4.9** |

Pre-registered 2021 split: before −0.135% (SE 0.269), from 2021 −1.034%
(SE 0.248), difference −0.90% (−2.5 SE).

**What this actually says:**

1. **There was never a positive-CLV era.** 2017–2022 pools to −0.135% ± 0.269%
   — indistinguishable from zero. The +2% was the multiplicative bias.
2. **From 2023 the model is consistently and significantly negative**, around
   −2%, three seasons running at −3.1, −3.7 and −4.9 SE. That part is real and
   is not the benchmark: 2023 has no Betfair data at all.
3. **The gate is shut, and now for a properly measured reason.** Not "the edge
   faded" but "there was no edge, and the model now loses to the close by about
   what a model with no edge and some adverse selection would."

**Also fixed:** `fair_line_preference` is a `BacktestConfig` field;
`predictions` carries `fair_line_source`; `evaluation.fair_line_sources` and
`evaluation.benchmark_changed` report and flag a switch; and
`scripts/season_breakdown.py` prints a loud warning before the table. It fires
correctly on the real window.

### Finding 1: the era split is 2023-24, not COVID (SUPERSEDED — see above)

`python scripts/season_breakdown.py --league E0 --from 2017-08-01 --market 1x2`
(the database starts at 2017/18, so nine full seasons, not ten):

| season | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| mean CLV | +1.99% | +1.18% | +1.70% | +2.51% | +2.13% | +2.31% | +0.76% | −1.56% | −1.96% |
| z (clustered) | 3.7 | 2.1 | 3.3 | 4.1 | 4.0 | 3.4 | 1.3 | −3.0 | −3.9 |

**The crowds explanation is dead.** The empty-stadium seasons (2020, 2021) are
indistinguishable from the healthy pre-COVID ones, and CLV stayed strongly
positive through 2022 with crowds back for a full season. The decline starts in
2023 and turns significantly negative in 2024 and 2025 — two seasons *after*
crowds returned. `beat_close` falls to ~0.40 in both.

The model's raw prediction quality held steady throughout (`log_loss_gap` vs
market ~0.01–0.02 in every season). What changed is that the closing line got
better at pricing whatever this model was catching, not that the model got
worse. **The gate stays shut**, and more firmly: the current era is
significantly negative, which is worse than the "indistinguishable from zero"
bar, not better.

The fixed 2021 split in `era_comparison` now straddles the real boundary — it
pools strongly-positive 2021–22 with strongly-negative 2024–25 and reports a
muddy +0.37% for "from 2021". That is the split doing its job (it was chosen in
advance and must not be swept), not a defect, but the per-season table is the
honest read now.

**A retune was deliberately not run.** It cannot reopen the gate — that needs
CLV near zero, and the current era is negative — a 2023-onward dev window is
only ~2 seasons against a 30-point grid, and the tuner already reported fitting
noise on a five-year window. The open question is *why* 2024–25 turned
negative: market sharpening, or something changing in the recent-season data
pipeline (odds source, bookmaker mix, settlement). That is a code and data
investigation, not a hyperparameter one.

### Finding 2: one line, a mirroring artifact, and goals are not overdispersed

`python scripts/goals_shape.py --league E0 --from 2017-08-01`

Three corrections, in order of how badly they change the conclusion.

**The source quotes exactly one totals line — 2.5 — in every league.** The
proposed explanation below ("over 3.5 and over 2.5 land in different bands")
is not merely unsupported, it is impossible. Splitting `total_goals` by line is
a no-op. Check the data before building the explanation.

**The sign flip was mostly the mirroring.** Over and under at one line are
exact complements, so the pooled band 0.30–0.40 holds the *overs* from
defensive fixtures and the *unders* from open ones — two kinds of fixture in
one row, every match counted twice. `calibration_by_line` takes one side and
kills it. A real gap does survive on the over side alone, but it is a
**calibration slope** problem, not a shape problem: slope **0.623** (SE 0.090,
4.2 SE below 1.0). The probabilities are spread too far apart.

The control that makes that trustworthy: the closing line scored by the same
code on the same 3380 matches comes back at **0.971** (−0.30 SE). The method
correctly certifies a calibrated forecaster.

**Goals are not overdispersed, and a negative binomial would be backwards.**
Dispersion ratio **0.953** (SE 0.024, −1.94 SE) — if anything slightly *under*
dispersed. The bucket table shows lighter tails than predicted (total 0 at
−2.2 SE, 6 at −2.2 SE, 7+ at −1.7 SE) and a heavier middle, which is the
opposite signature from corners and cards. Variance decomposition: predicted
3.005 (within-match 2.838 + between-match 0.167) against observed 2.726. The
predictive distribution is ~10% too wide, and the excess is in how far the
fitted *rates* move between fixtures, which is shrinkage, not dispersion.

**And it is barely worth fixing.** Fitting the slope on 2017–2021 and applying
it to 2022 onward improves holdout log loss from 0.6782 to 0.6772 — 0.0010,
against a market gap of 0.0098 — and the correction overshoots (holdout slope
goes 0.660 → 1.183), so the slope is not stable across eras either.

The reason it buys so little is the finding worth carrying forward: **the model
knows almost nothing about match totals.** On over 2.5 it beats the base rate
by 0.0043 while the market beats it by 0.0147. On 1X2 the model beats the base
rate by ~0.060. Its total-goals probabilities have spread, but most of that
spread is noise, which is exactly what a 0.62 slope means. Do not spend more
effort on the goals-totals market.

---

## Finding 2 (original text, superseded by the section above)

## Finding 2: the total_goals calibration is misshapen

`calibration_table` bins every selection, and over/under are complements, so
the `total_goals` table is perfectly mirrored (n=55/55, 467/467, 1006/1006,
gaps exactly negated). Half of it is decoration. Log loss is unaffected; only
the apparent independence of `n` is.

Reading the informative half of the last run:

| band | n | predicted | observed | gap | approx |
|---|---|---|---|---|---|
| 0.30–0.40 | 467 | 0.362 | 0.441 | +0.079 | 3.5 SE |
| 0.40–0.50 | 1006 | 0.451 | 0.404 | −0.048 | 3.1 SE |

Adjacent bands, both large, opposite directions. `calibration_table`'s own
docstring names this case: a sign that flips across bands means badly shaped
rather than merely biased.

It cannot be diagnosed from that table, because it pools every line together —
over 3.5 and over 2.5 land in different bands. The structural question worth
asking: `score_matrix_from_rates` builds the scoreline matrix from two
independent Poissons with the Dixon-Coles correction on four cells, while
`models/counts.py` deliberately uses negative binomial for corners and cards
because Poisson underprices the tails. Nobody applied that reasoning to goals.
Whether goal totals are overdispersed conditional on the fitted rates is
testable directly.

The 1X2 calibration wrinkle flagged in the previous handoff (0.70–0.90 band)
is still too thin to act on: ~2.1 SE in one of nine bands, n=87 and n=19.

---

## Finding 3: the count models cannot be validated against this data source

`extract_odds` only produces `1x2`, `total_goals` and `asian_handicap` rows.
football-data.co.uk carries no corner or card prices, and
`BacktestConfig.fit_count_models` defaults to `False`.

The negative binomial models are a substantial part of Phase 2 and there is
currently **no way to measure whether they have an edge against any market**.
Not a bug, but it should shape where effort goes: count markets are usually
softer than goals markets, and right now you can't see whether yours are
beatable.

---

## What was added in the xG session (most recent)

The first genuinely new *capability* in a while, rather than a correction to an
existing one: expected goals from Understat, and the option to fit team
strengths to them.

### The result: a small real improvement, and it is the blend that does it

`scripts/compare_targets.py --from 2018-08-01`, all five leagues, 1X2:

| league | goals | xg | blend 0.5 | gap to market: goals → blend |
|---|---|---|---|---|
| E0 | 0.5709 | 0.5708 | **0.5697** | 0.0118 → 0.0106 |
| SP1 | 0.5823 | 0.5822 | **0.5812** | 0.0132 → 0.0121 |
| I1 | **0.5760** | 0.5783 | 0.5762 | 0.0136 → 0.0137 |
| D1 | 0.5818 | **0.5788** | 0.5792 | 0.0125 → 0.0098 |
| F1 | 0.5898 | 0.5888 | **0.5882** | 0.0123 → 0.0108 |

Treated as one fixed rule applied to every league, which is the only honest way
to read a five-league table:

- **blend 0.5: mean −0.00126 log loss, SE 0.00045, t = −2.79, better in 4 of 5.**
- pure xG: mean −0.00038, SE 0.00085, t = −0.44. Not distinguishable from zero.

So **the blend helps and pure xG does not**, which is not what the theory
predicted. The mechanism looks like variance reduction rather than new
information: averaging two noisy views of the same rate beats either. Read
per-league the answer is different in every league — xG wins in D1, goals wins
in I1, the blend wins in three — and that spread is wider than any single
league's improvement, which is exactly the multiple-comparisons trap the
handoff warns about for leagues. The script now says so instead of reporting a
per-league winner.

Size of the win, stated plainly: it closes about **10% of the gap to the
closing line** (mean gap 0.0127 → 0.0114). CLV stays clearly negative in all
five leagues. **This does not reopen the gate** and was never going to.

One suggestive detail worth following up: xG helps most in D1, which has the
fewest matches (306 a season, not 380) and the best-calibrated goals model
(slope 1.02 against 1.12-1.30 elsewhere). If xG's advantage really is variance
reduction, it should pay most where the goals estimate is thinnest, and that is
what the one clean case shows.

### The bigger finding, which is not about xG

**Every model in every league has a calibration slope above 1** — 1.02 to 1.30
on goals, rising to 1.28-1.62 on xG. Above 1 means the probabilities are packed
too tightly together: the model is systematically *under*-confident about which
fixtures are unusual, and shrinkage is the cause. Fitting on a less noisy target
makes it worse because the ridge penalty was never re-tuned for a less noisy
target.

That is a larger and cheaper win than xG if it holds, and it was found by the
calibration-slope diagnostic built in the previous session rather than by
anything in this one.

`scripts/compare_targets.py --league E0 --from 2018-08-01 --ridge 1 2 5`:

| ridge | goals | xg | blend 0.5 | goals slope |
|---|---|---|---|---|
| 5 (ships today) | 0.5709 | 0.5708 | 0.5697 | 1.201 |
| 2 | 0.5693 | 0.5677 | 0.5668 | 1.039 |
| 1 | 0.5692 | 0.5669 | **0.5661** | **0.977** |

Two things fall out. **The slope diagnostic was right**: it said the model was
over-shrunk, and dropping ridge from 5 to 1 moves the slope from 1.201 to 0.977
— near-perfect calibration — with log loss improving in step. And **xG was
being penalised by a shrinkage value calibrated for noisier data**: at ridge 5
xG and goals are a dead heat (0.5708 vs 0.5709), at ridge 1 xG wins by 0.0023.
Comparing targets at a shared ridge was never a fair test.

The best cell closes **41% of the gap to the market** (0.0118 → 0.0070) against
10% for xG alone.

**Do not act on that table.** It is nine settings searched on one league with
no holdout — precisely the error this document warns about everywhere else, and
the tuner has already rejected a lower ridge once: it found ridge 2 best on its
development window and *worse* out of sample, which is why the default is still
5. What is genuinely new here is the mechanism, not the grid point: a slope
above 1 is a reason to expect less shrinkage to help, decided before the search
rather than after it. That is worth a holdout test, which is what
`scripts/tune_hyperparameters.py --target blend` now exists to run. Believe the
holdout, not the table above.

- `fbedge/understat.py` — fetch, cache and parse. Includes `TEAM_ALIASES`.
- `scripts/build_xg.py` — joins xG onto existing matches, writes `match_xg`.
- `fbedge/models/goals.py` — `responses`, `recalibrate_on_goals`, and a
  `target` argument on `fit_goals_model` taking `"goals"`, `"xg"` or `"blend"`.
- `fbedge/models/base.py` — `load_training_set` LEFT JOINs `match_xg` when the
  table exists, so an older database still fits.
- `fbedge/predict.py`, `fbedge/backtest.py`, `app.py` — `target` and
  `blend_weight` plumbed through; the app gets a "Rate teams on" selector.
- `scripts/compare_targets.py` — the head-to-head.
- `tests/test_understat.py` — 18 tests.

Four things worth not undoing:

- **Understat moved its data out of the page HTML.** Every published scraper,
  including the `understat` PyPI package, looks for
  `var datesData = JSON.parse(` in the league page. That is no longer there,
  and those scrapers now return *nothing* without raising. The data comes from
  `getLeagueData/<league>/<season>`, gzipped JSON, found by reading
  `js/league.min.js`. `fetch_season` raises `UnderstatError` rather than
  returning empty on any shape change, and the tests pin that.
- **The 41 team aliases were derived by diffing, not guessed.** 41 of 156 names
  differ between the two sources and the differences are not systematic —
  Understat's "Milan" is AC Milan while its "Inter" matches ours exactly, so a
  fuzzy matcher would silently merge two clubs. `test_alias_table_is_injective`
  guards exactly that. The join currently lands at 100% in all five leagues,
  and `build_xg.py` refuses to write a league below 80%.
- **Fitting to a continuous target needed no new maths.** The Poisson objective
  already drops the factorial, so `y*log(mu) - mu` is a valid quasi-likelihood
  for xG and its score equation still sets the rate to the weighted mean. What
  did need care is the Dixon-Coles correction, which tests for exact 0-0 and
  1-1 scorelines and can never fire on a continuous input; it is switched off
  in the first stage and re-estimated on real goals in the second.
- **The two-stage fit is the load-bearing design.** Strengths come from xG;
  level, home advantage and rho are re-estimated on actual goals by
  `recalibrate_on_goals`. Skipping that would over-predict every total in the
  book, because xG runs ~4% above goals for home sides across this database.
  `test_xg_model_is_calibrated_to_goals_not_to_xg` pins it.

## What was added in the shape session

- `fbedge/markets.py` — `total_distribution`, the match-total pmf from a
  scoreline matrix.
- `fbedge/evaluation.py` — `calibration_by_line`, `calibration_slope`,
  `goal_total_fit`, `_bucket_row`. `calibration_table` gained a docstring
  warning about pointing it at a two-sided market.
- `fbedge/backtest.py` — `BacktestResult.match_totals`, one row per match with
  the predicted total distribution and the realised total. `_score_match` now
  returns `(records, total_record)` and builds the scoreline matrix before
  checking odds, so matches nobody priced still contribute to the shape test.
- `scripts/goals_shape.py` — new CLI.
- `tests/test_goals_shape.py` — 22 new tests.

Two design decisions worth not reversing:

- **The two power tests are the point of the test file.** Both headline results
  were *nulls* ("not overdispersed", "the market's slope is fine"), and a null
  is only worth acting on if the diagnostic could have detected the effect.
  `test_dispersion_detects_genuine_overdispersion` and
  `test_slope_detects_probabilities_that_are_spread_too_far` point each test at
  data built to contain exactly the defect, at the real sample size. Delete
  those and the negative findings become indistinguishable from a broken test.
- **The market is used as a live control, not just a benchmark.** Scoring the
  closing line with the same `calibration_slope` code on the same matches is
  what separates "the model is miscalibrated" from "my slope estimator is
  broken". Any future calibration claim should carry the same control.

## What was added in the season-breakdown session

Three files. `fbedge/evaluation.py` is the existing file with 213 lines
appended (448 → 661); nothing above line 448 changed.

- `fbedge/evaluation.py` — four new functions: `season_labels`,
  `clustered_mean`, `season_breakdown`, `era_comparison`.
- `scripts/season_breakdown.py` — new CLI.
- `tests/test_seasons.py` — 16 new tests.

Three design decisions, with reasons, so they don't get quietly reversed:

- **Season assignment prefers the database's `season_start_year`.** The date
  rule is only a fallback, and it cuts at 1 August specifically so the
  COVID-delayed finish of 2019-20 (matches into late July 2020) stays in the
  2019 season. Getting that boundary wrong blurs the exact line the analysis
  is trying to draw.
- **Standard errors are clustered by match.** Selections on one match share a
  model fit and one closing-line move, so `closing_line_value`'s naive
  `std/sqrt(n)` overstates precision — about 1.12x on this bet density.
  Irrelevant to a 7.5 SE headline, easily enough to invent a 2 SE result in a
  thin single-season row. The formula reduces exactly to the naive one when
  every bet is its own match, and a test pins that.
- **The era split point is fixed and the script will not sweep it.** Crowds
  returned for 2021-22, which is a reason that existed before the data was
  looked at. Scanning candidate splits and reporting the largest gap is the
  same uncorrected multiple-comparisons error as tuning five leagues and
  reporting the best one.

Also: the zip contained a stray directory named `{fbedge,scripts,tests,data`
holding an empty `raw}`. A brace expansion that ran under `sh`. Safe to delete.

---

## Immediate next step

**The re-baseline is done.** The gate verdict and Finding 1 have been restated
in place, and the tuner was checked and needs no re-run. Every CLV figure in
this document is now on one scale.

What remains, honestly assessed:

1. **Why 2023.** The decline from ≈0% to ≈−2% survives the benchmark fix, is
   three seasons deep (−3.1, −3.7, −4.9 SE), and 2023 carries no Betfair data
   so it is not the instrument. Worth understanding whether it is the market or
   the data pipeline. Note the framing though: this is a decline from *no edge*
   to *a small measured loss*, not the loss of something that was ever worth
   having.
2. **Other leagues, as a genuine test rather than a search.** The margin fix
   applies to all five. Four more leagues would say whether "≈0% through 2022,
   ≈−2% after" is a property of this model or of the Premier League. Write the
   plan down first and report all five outcomes, including the dull ones.
3. **Better inputs** — the list further down is unchanged by any of this, and
   is still the only route to a real edge rather than a better-measured
   absence of one.

Consider seriously, though, that the project has now answered its own question.
Three sessions of investigation into an apparent edge ended with the edge being
a defect in the measuring instrument. On a correct scale this model has never
beaten the closing line in nine seasons of data, and the model's own log loss
has trailed the market throughout. "Well-built, well-tested, no edge" is a
legitimate place to stop, and it is better evidenced now than the alternative
ever was.

---

## Previous immediate next step — DONE, results in the RESOLVED section above

Both runs below have been carried out. Kept for the reasoning; do not re-run
expecting news.

First confirm how far back the database goes — do **not** rebuild it:

```
python -c "from fbedge import database, config; con = database.connect(config.DB_PATH, read_only=True); s = database.summary(con); print(s[s.league=='E0'].to_string(index=False))"
```

Then, starting from the earliest full season available (2016-08-01 if you have
10 seasons; three clean pre-COVID seasons are worth having, because they
distinguish "the COVID seasons were special" from "the early window was
special for some other reason"):

```
python scripts/season_breakdown.py --league E0 --from 2016-08-01 --market 1x2
```

`--market 1x2` because that makes it directly comparable to the tuner run that
produced the +0.89%. One backtest, sliced by season. Roughly one to two minutes.

### Reading the result

- **Pre-2021 clearly positive, 2022 onward negative** → the pooled −0.94%
  averages two regimes and the later era is the honest number. Retune with
  `--from 2021-08-01` and see whether the holdout verdict changes. The gate
  still stays shut unless the later era's CLV is indistinguishable from zero
  *and* that survives the tuner's holdout.
- **Every season near −0.5%** → the era hypothesis is dead, and the tuner's
  +0.89% needs a different explanation. Most likely something differs in how
  bets are selected between the two code paths; go looking in the code rather
  than the data.

Either result is useful. Neither reopens the gate on its own.

---

## After that, in rough order of expected value

1. ~~**Per-line calibration for `total_goals`**~~ — **done**, see the RESOLVED
   section above. One line only, the anomaly was mostly a mirroring artifact,
   goals are not overdispersed, and the residual slope problem is not worth
   fixing. Closed. The successor question, if anything: why the 2024–25 CLV
   collapse, which is a data-pipeline and market-sharpness question.
2. **Additional leagues, with the plan written down first.** La Liga, Serie A,
   Bundesliga, Ligue 1. Decide which leagues, on which window, and what counts
   as success, *before* running any of them, and report all outcomes including
   the unflattering ones. Five independent tests will produce a spurious
   winner by chance; that is expected, not surprising. Do not run this until
   the season breakdown has settled which window is the right one.
3. **Better inputs, not a better model.** Zero team-news signal — no injuries,
   no rotation, no resting starters. That is a structural disadvantage no
   hyperparameter fixes. Phase 5 in the previous handoff covers xG from
   Understat, FBref for UEFA competitions, and schedule congestion (`rest_days`
   and `matches_last_14_days` are computed but never fed into the fit).
4. Consider that the honest conclusion may be: this is a well-built,
   well-tested learning project, and that is a fine place for it to stay.

---

## Standing rules

- **Do not build Phase 4** unless the gate opens, and the gate needs CLV
  indistinguishable from zero *and* holding up on a tuner holdout window it
  never influenced.
- **Do not stake real money.** Even if the gate opens, the sequence is:
  paper-trade for several weeks recording what would have been bet, and judge
  it on the CLV from *that* period, not the historical backtest number.
- **Do not re-derive the five fixed bugs.** Away handicap line sign, CLV
  across mismatched bookmaker margins, push probability in market comparisons,
  flat staking going negative, the collapsed bookmaker table. The reasoning is
  in the docstrings of `backtest.py`, `settlement.py`, `evaluation.py`. Read
  those three files before changing anything in them.
- **Do not tune on ROI.** The tuner searches log loss on purpose; the best-ROI
  grid point is the luckiest setting, not the best one.
