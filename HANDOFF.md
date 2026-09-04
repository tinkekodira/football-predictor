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

## Known bugs live in BACKLOG.md, not here

`BACKLOG.md` in the repo root registers defects that were confirmed by reading
the code and then deliberately left unfixed. Check it before investigating
anything that looks like a bug, and add to it rather than to this file when you
shelve one. Two of its entries silently produce a wrong number rather than an
error, so they are worth knowing about before trusting a result:
`fair_line_preference` does not actually pin the benchmark (B1), and
`build_xg.py` locks the database for the length of a download (B2).

---

## Phase 4 was built, and the gate it was blocked on is still shut

**Read this before the status list.** The standing rules below say "Do not
build Phase 4 unless the gate opens". The gate has not opened - CLV is still
-1.500% (-9.1 SE) on E0 - and Phase 4 was built anyway, on 2026-09-04, at the
explicit instruction of a brief that superseded that rule.

That is a decision, not an oversight, and it was made with the disagreement on
the table. What the build does about it:

- Every EV figure the scan prints carries the backtested CLV and calibration
  for **that market on that league**, with the sample size, from
  `fbedge/evidence.py`. There is no code path that produces a bare number.
- A market with no historical price says so explicitly rather than showing a
  blank. Corners, cards, BTTS, double chance, the team totals and the half-time
  markets are all in that category: measurable for calibration, never
  backtestable as bets.
- The scan's footer states that this model has never beaten the closing line
  and that a positive EV means the model disagrees with the price, not that the
  model is right.
- Fixtures involving a club the model barely knows are flagged inline. This was
  not theoretical: the first live run put two newly promoted clubs in the top
  four rows at +68% and +50% EV, on two matches of history each.

**The gate rule stands.** Nothing about building the scan is evidence for
staking money, and the sequence in the standing rules - paper-trade for several
weeks, judge on the CLV from *that* period - is unchanged.

---

## Current status

- 402 tests passing (`python -m pytest`).
- **The CLV measurement was broken, and is now fixed and re-baselined.** A
  benchmark change in 2024 plus favourite-longshot bias in multiplicative
  margin removal inflated every historical CLV figure. `margin_method` now
  defaults to Shin, which measures unbiased against the exchange. The gate
  verdict and Finding 1 have both been restated on the new scale in place; the
  gate is **−1.500% (−9.1 SE)**, worse than the −0.944% it replaces.
- **Model log loss and the tuner are unaffected** by the margin fix, and
  verified so: they never touch `remove_margin`. Only CLV and
  `market_log_loss` moved.
- **The model defaults changed at the end of the day**: team strengths are now
  fitted to a goals/xG blend with per-target shrinkage. Validated on four
  leagues that had no part in choosing it (-0.0033 log loss, 4/4). Every
  number recorded before that describes the old model.
- **Player availability was built, measured and left switched off.** Both
  coefficients carry the predicted sign in all six specifications and none
  reaches two standard errors; turning it on moves log loss by 0.00004. See the
  availability section for what that does and does not rule out.
- **That null half-survived a retest on real injury data.** Across five
  leagues, a team's *own* absentees are worth -1.8% per player (z = -1.87, not
  significant), but its *opponent's* are worth +1.5% to +2.7% (z = 2.6 to 3.0,
  right sign in 4-5 leagues of 5). The -7.7% first reported from one season of
  E0 did not replicate. See the retest section.
- **Fitting the shrinkage instead of choosing it was built, measured and
  rejected.** Empirical Bayes says the ridge should be 11-13, not 1; applied, it
  costs 0.008 log loss and loses on 4 of 4 held-out leagues, with the
  calibration slope blowing out to 1.7-2.2 exactly as that diagnostic predicted
  in advance. The estimator is correct; the question it answers is not the one
  the model needs. See the hierarchical section.
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

## The availability retest: a small, consistent opponent effect

`scripts/injury_signal.py`, on real injury data rather than the line-up proxy.
**This is one season of one league and must not be believed yet**, but it is
the most promising number this project has produced, and the reason it appeared
is a mechanism decided before the run rather than found in the output.

### What changed, and why it was predicted in advance

The first pass counted **everyone currently unavailable** and found nothing:
`beta_own` +0.011 and -0.003, both indistinguishable from zero. The mean was
**3.9 players out per side**, which is the clue. A player out for the season
appears in every fixture, and by the third of them the model has already
adapted - the team's recent results were produced without him, so the fitted
rate carries his absence already. Counting him again asks the model to subtract
the same player twice.

So the feature was changed to **newly** unavailable - out today, playing last
time - which is the same reasoning as the absence window in
`fbedge/availability.py`. That was decided from the 3.9 figure, before the
result was seen.

### E0, 2024/25, 343-359 matches

| specification | mean out/side | beta_own | z | beta_opp | z |
|---|---|---|---|---|---|
| **newly out** | 0.88 | **-0.057** | **-1.81** | +0.012 | +0.35 |
| **newly out or doubtful** | 0.97 | **-0.077** | **-2.67** | +0.002 | +0.06 |
| all currently out | 3.90 | +0.011 | +0.62 | +0.027 | +1.48 |
| all currently out or doubtful | 4.33 | -0.003 | -0.21 | +0.022 | +1.36 |

**-7.7% on the scoring rate per newly missing player**, clearing two standard
errors, on one season. For scale, that is larger than the goals/xG blend and
the shrinkage change put together - those bought 0.0033 of log loss between
them. The prediction split the two specifications exactly as the mechanism said
it would: the state is noise, the change is signal.

`beta_opp` is flat in both new specifications, which is *different* from the
original availability study and worth understanding rather than glossing: a
weakened opponent does not appear to concede more, only to score less.

### REPLICATED ON FIVE LEAGUES, AND THE HEADLINE SHRANK

**The -7.7% did not hold up.** Read this before the single-season section
above, which is kept only so the shrinkage is visible.

`scripts/injury_signal.py --leagues E0 SP1 I1 D1 F1 --from 2022-08-01
--to 2024-06-01`, 2,472 to 2,921 matches per specification:

| specification | beta_own | z | signs | beta_opp | z | signs |
|---|---|---|---|---|---|---|
| newly out | -0.018 | -1.87 | 4/5 | **+0.027** | **+2.86** | 4/5 |
| newly out or doubtful | -0.018 | -1.87 | 4/5 | **+0.024** | **+2.69** | 4/5 |
| all currently out | -0.009 | -1.68 | 4/5 | +0.015 | +2.64 | **5/5** |
| all currently out or doubtful | -0.007 | -1.25 | 4/5 | +0.016 | **+3.03** | **5/5** |

Three things follow, and the first is a correction.

1. **`beta_own` is not significant and the first number was inflated.** One
   season of E0 gave -7.7%; three seasons of E0 gave -4.6%; five leagues pool
   to **-1.8%, z = -1.87**. That is the ordinary fate of a first result, and it
   is why nothing was built on it.
2. **The consistent finding is the opponent, not the team itself.** A side
   whose *opponent* is missing players scores about 1.5% to 2.7% more per
   player, at z = 2.6 to 3.0, with the right sign in 4 or 5 leagues of 5 in
   every specification. That is the opposite emphasis from the single-season
   read, where `beta_opp` looked flat.
3. **The mechanism argument was half right.** "Newly out" really does carry
   about twice the coefficient of "everyone currently out" on both sides
   (-0.018 against -0.009; +0.027 against +0.015), which is what "the model has
   already adapted to a long-term absentee" predicts. The prediction was about
   the *ratio* and it held; it was not a prediction that either would be large.

**Net: the availability null survives for a team's own absentees, and does not
survive for its opponent's.** An effect of 1.5-2.7% per missing player is real
but small - for scale, the goals/xG blend and shrinkage change together bought
0.0033 of log loss - and it is still an effect rather than an edge, for the
timing reason below.

### Why this is still not something to build on

- **One season, one league.** 343 matches. The free plan covers 2022-2024 for
  all five leagues, so this can be roughly fifteen times larger for fourteen
  more requests. Nothing should be built until it is.
- **Four specifications were run and the best is quoted.** The ordering
  protects it - the mechanism was argued from the 3.9 mean before the split was
  measured - but two of the four are close relatives, so treat z = -2.67 as
  weaker than it looks.
- **It is an effect, not an edge.** The feed's rows are attached to fixtures
  and the endpoint does not say when each row came into existence. A "Missing
  Fixture" row is confirmed by the line-up, which is published about an hour
  before kick-off and after the price. This measures whether absences move
  scoring, which is a question the project had open; it does not establish the
  information was ever tradeable.
- **Relapses are undercounted on purpose.** The feed only emits rows when
  somebody is unavailable, so a date with none is either "everyone fit" or "no
  fixture", and those cannot be separated from injuries alone. The undercount
  biases towards finding less signal, which is the right way to err here.

### The next run, and it is cheap

```
python scripts/build_injuries.py --season 2023
python scripts/build_injuries.py --season 2022
python scripts/injury_signal.py --league E0 --from 2022-08-01 --to 2025-06-01
```

Ten requests, then the same for the other four leagues. If -7% survives on
16,000 matches across five leagues, it is the first thing in this project worth
building on. If it does not, it was one season of noise and the availability
null stands.

## What was added in the home-page session (most recent)

A calendar home page, in the shape of a live-scores site: scroll a date, see
every fixture in the top five leagues on it, click one for the detail page that
already existed.

- `fbedge/fixtures.py` - the season calendar, played and unplayed, from
  Understat. UTC in, local dates out.
- `scripts/build_fixtures.py` - populates the new `fixtures` table.
- `home_view.py` - the page itself, at the repo root rather than under
  `fbedge/` so that nothing importable by a model or a test pulls in Streamlit.
- `app.py` - now a two-view router. The calendar is what opens; the fixture
  page is what a click on it opens, seeded from the fixture that was clicked.
- `tests/test_fixtures.py` (21) and three more in `tests/test_app.py`.
  **341 tests**, from 317.

### The source question, which was the whole difficulty

**football-data.co.uk cannot supply a calendar.** It is the project's primary
source and it does publish `fixtures.csv` - `ingest.download_upcoming_fixtures`
already fetched it - but that file is a *price* feed. On the day this was
built it held 48 rows spanning three days, **two of them** in the top five
leagues. It can never answer "what is on in October".

**Understat can, and was already integrated.** Its league payload carries the
whole season in `dates`, played and unplayed alike: 1,752 fixtures across the
five leagues for 2026/27, running to 30 May 2027. No new source, no new
dependency, and the team-name aliases were already derived and tested.

Three things that would be wrong if reversed:

- **`understat.match_frame` drops unplayed fixtures and must keep doing so.**
  An unplayed match carries a scoreline of null and an xG of **zero**, not a
  missing one. `fixtures.fixture_frame` is its mirror image and writes NULL for
  both. `test_unplayed_fixtures_carry_null_not_zero` is the guard; without it,
  twenty teams of "failed to have a shot" would flow into the ratings.
- **Kick-offs are stored in UTC and converted on the way out.** Understat gives
  19:00 for a match that starts at 21:00 in Zagreb. Storing local time would
  bake one reader's timezone into the database; grouping on the *UTC* date
  would scatter one matchday across two and look like missing fixtures.
- **The calendar is re-fetched, not merged.** A result arrives by *changing* a
  row from unplayed to played, so an insert-only load would keep the stale row
  and show a finished match as still to come, next to itself. The delete is
  scoped to the seasons and leagues actually supplied.

### What the right-hand panels can and cannot honestly claim

The brief asked for injury news. **There is no news source in this project, and
none was invented.**

- **Highlighted matches** is real: the model prices every fixture on the day
  from ratings fitted only on earlier matches, and shows where it has a strong
  but non-obvious opinion. It is labelled as opinion, not edge - odds exist
  only a few days ahead, and this model has never beaten a closing line.
- **Availability watch** is derived, not reported. It reuses
  `availability.for_fixture`, the same point-in-time windowing the null result
  was measured with, rather than a second version written for display. It says
  plainly that it cannot tell an injury from a rested player and cannot know
  about this morning.
- **News** says it has no source. Filling it with something that looked like
  news would have been the worst available outcome.

### Real injury news, added after the first pass

The first version of this page had no injury panel worth the name, because no
source existed. One does now: `fbedge/injuries.py`, and it is **the only keyed
external dependency in the project**.

- **Provider: API-Football** (`v3.football.api-sports.io/injuries`). Chosen
  because it is addressed by *league and season*, so five requests refresh all
  five leagues against a free allowance of a hundred a day. The alternative
  looked at, Big Balls Sports Data, offers a larger quota but is addressed per
  player id, which would mean building a player-id mapping first.
- `FOOTBALL_API_KEY` in the environment, never in a tracked file - this repo
  still has no `.gitignore` (BACKLOG B3).
- `scripts/build_injuries.py` loads it; `--probe` prints one league's raw
  response and writes nothing.

**The schema is now confirmed against a live response**, and
`test_the_parser_agrees_with_a_real_response` pins that entry verbatim - the
only test in the file that could catch the provider changing rather than merely
proving the parser self-consistent.

**But the free plan stops at the 2024 season, and that is the headline.**
Probing 2024 returns 3,168 Premier League rows; the current season returns
none. Both look identical - an empty list - which is why
`config.INJURY_FREE_PLAN_LAST_SEASON` exists and why the script now says which
case it is. So a free key **cannot show today's team news**. It can do
something else valuable, below.

**Two real defects the live probe exposed, both now fixed:**

1. **No date filter.** A league-season is thousands of rows because the feed
   sends one per player *per fixture*. The panel would have listed a club's
   entire season of absences at once and read as a permanent injury crisis.
   `for_teams` now takes `on_date` and the page passes the day it is showing.
2. **`paging` was ignored.** It is `{"current": 1, "total": 1}` today, so
   everything fits on one page - but had that changed, the reader would have
   silently taken a fraction of a league. It now refuses rather than
   half-reads.

Neither was findable without a real response, which is the argument for
`--probe` over trusting a published schema.

### What a free key is actually good for

**Retesting the availability null with real injuries.** 2022 to 2024 is exactly
the era the availability study covered, and the study's stated weakness was
that Understat line-ups cannot separate an injury from a rested player
(BACKLOG B6). This feed states the reason - "Ankle Injury" - per player per
fixture. That turns a proxy into a measurement, and it is free.

That is a genuine open question this now makes answerable, and it is worth more
than the panel: the panel is decoration on a model with no edge, whereas the
availability question is about whether a real signal was missed.

**Nothing guesses a club name.** Matching is exact, then exact after a
documented normalisation, then an explicit alias; anything left is printed and
the row dropped. `test_the_alias_table_is_injective` earned its place
immediately by catching twelve alias keys - `'fc barcelona'`, `'ac milan'`,
`'as roma'` and others - that normalise to something else and could therefore
never have been hit.

### Club badges, and why they work when the injury data does not

`fbedge/crests.py` plus `scripts/build_crests.py`. The useful discovery:
**`media.api-sports.io` is a public image CDN** - fetching a badge sends no
key, verified directly. So badges keep working on a free plan even though the
same provider's injury data is capped at 2024. Only the *id lookup* needs a
key, and the ids are cached in `data/raw/crest_ids.json` so it is paid once.

- **16 badges already exist with no key at all**, harvested from the team ids
  sitting inside the cached `E0_2024.json` injury probe. Every injury row names
  its club and carries its id.
- The remaining 80 need one run of `scripts/build_crests.py` with the key.
  It fetches the five leagues *and their second tiers*, because a club promoted
  this summer was in a second tier last season.
- Badges are resized to 48px on download and inlined as data URIs. The
  originals are ~90KB; a Saturday shows 22 fixtures, so hotlinking or storing
  full size would put four megabytes of PNG on one page.

Alternatives ruled out: **TheSportsDB** free tier returns the same 24 dummy
teams for every league id, and caps `search_all_teams` at 10 - unusable.

**A club with no badge gets a generated monogram**, a flat disc of initials
drawn as inline SVG. That reverses the first version of this file, which
rendered nothing so a gap would read as a shorter name rather than a broken
image. Correct at a handful missing; wrong at **80 of 96**, which is where a
free key leaves you - the page read as half-built. The stand-in needs no
network and no licence, cannot be mistaken for a real crest, and is replaced
automatically when `build_crests.py` fetches the genuine one. `crests.missing`
still reports those clubs, so the cosmetic fix does not hide the real gap.

### Still needed for the page to be fully populated

1. **A paid API-Football plan** if the injury panel is to show current team
   news at all; a free key stops at 2024. The code needs no change either way -
   only the plan does.
2. **One run of `scripts/build_crests.py`** with a free key, to get the other
   80 club badges. Costs about ten requests, once, and never again for a club
   whose id is cached.
2. `python scripts/build_rosters.py --league E0` with the app stopped -
   `match_lineups` holds **83 rows**, so the availability panel currently has
   nothing to derive from. The earlier session's backfill cached 3,430 matches
   but the write never landed.
3. Odds for upcoming fixtures stop at the football-data horizon of about three
   days, so most future dates carry no market comparison at all.

## What was added in the hierarchical session

An attempt to stop *choosing* the shrinkage and start fitting it. **Built,
measured, and rejected** - and the way it fails is more useful than the feature
would have been.

- `fbedge/models/hierarchical.py` - empirical Bayes for the ridge, by EM with a
  Laplace approximation. Separate variances for attack and defence, a
  quasi-Poisson dispersion correction, and `pool_variances` for the second
  level across leagues.
- `fbedge/models/base.py` - `ridge_penalty` now takes a scalar *or* an
  `(attack, defence)` pair. A scalar behaves exactly as before.
- `fbedge/models/goals.py` - `ridge="auto"` and `ridge="auto-split"`;
  `GoalsModel.ridge_pair` and `.ridge_estimate`.
- `scripts/shrinkage_report.py` - what the data says the shrinkage should be.
- `scripts/validate_setting.py` - `--ridge` now takes the automatic modes.
- `tests/test_hierarchical.py` - 26 tests. 317 in total, from 291.

### The premise, which is correct

The ridge penalty is a Gaussian prior in disguise. `ridge_penalty` adds
`lambda * sum((beta - prior)^2)` to the negative log-likelihood, which is the
log posterior of a model where each team's strength is drawn from
`N(prior, tau^2)` with `lambda = 1 / (2 * tau^2)`. So the shipped ridge of 1.0
is a *claim*: that team strengths have a standard deviation of 0.71 in log-rate
space. Nobody had checked it, and `tau` is estimable from the same data the
strengths are.

### The result: decisively worse, in the direction the slope predicted

`scripts/validate_setting.py --from 2018-08-01 --target blend --ridge 1
auto-split auto`, against the shipped `goals r5` baseline:

| setting | mean delta log loss | SE | t | better in | slope range |
|---|---|---|---|---|---|
| blend0.5 r1 (ships today) | -0.00325 | 0.00014 | -23.7 | 4/4 | 0.96-1.23 |
| blend0.5 auto-split | -0.00326 | 0.00013 | -24.8 | 4/4 | 0.96-1.22 |
| **blend0.5 auto** | **+0.00824** | 0.00179 | **+4.6** | **0/4** | **1.67-2.23** |

Empirical Bayes estimates `lambda` between 11 and 13 in every league, against a
shipped 1.0, and left to iterate it climbs to the upper bound. Applied, it
costs 0.008 log loss and loses in every league including the one it was
developed on.

**The calibration slope called it in advance, which is the part worth
believing.** A slope above 1 means probabilities packed too tightly - the model
under-confident. That diagnostic is what found the original shrinkage win. Here
it says `auto` should be badly over-shrunk, and out of sample the slope lands
at 1.67 to 2.23 against 0.96 to 1.23 for the setting that ships. Same
mechanism, same direction, opposite sign of outcome. Two independent diagnostics
agreeing about a *rejection* is as good as this project's evidence gets.

### Why it fails, which is the actual finding

**The estimator is not broken.** `test_recovers_a_planted_prior_variance`
builds leagues whose `tau` is known, at realistic size, and recovers it. Without
that test "empirical Bayes disagrees with the holdout" would be
indistinguishable from a coding error, and it is the same power-test discipline
the goals-shape nulls needed.

The two calculations answer different questions. Empirical Bayes asks which
prior best explains the *training* data under the model as written. The model as
written says the time-weighted likelihood is a real likelihood over about 300
observations - `effective_n` is 248 to 301 across the five leagues, against 2400
to 3000 actual matches. Under that reading a team's strength is barely resolved,
most of the observed spread must be noise, and the right move is to shrink it
away. Hence `lambda` above 10.

**That reading is wrong, and specifically wrong.** The time weights discount old
matches because strength *drifts*, not because those matches were noisy. A match
from two years ago carries real information about a team today; the weight says
how much to trust it as a description of the present, not how much signal it
contains. Treat the discount as a statement about noise and you conclude the
model knows far less than it does.

There is no interior fixed point, either: both terms of the EM update shrink as
the penalty rises, so nothing pushes back. Run for 30 rounds it reaches the
bound in three leagues of five. `test_the_loop_runs_away_on_a_flat_likelihood`
pins that, so a later change that makes the loop settle can be told from one
that hides the problem. See BACKLOG.md B9.

### The one part that survived, and it is too small to matter

The *level* of shrinkage is what the misspecification corrupts. The *balance*
between attack and defence is a comparison of two parameter blocks inside one
fit on one likelihood scale, so whatever distorts the level distorts both sides
and largely cancels. That was worth testing separately, and it is what
`auto-split` does: keep the validated level, take only the split from the data.

It is real and consistent - attack strengths spread more than defensive ones in
5 of 5 leagues and in 19 of 19 league-windows tested. It is also tiny: the
implied split is 0.92/1.09 around 1.0, and the ridge plateau from 0.1 to 1.0
spans 0.0008 log loss. So it lands where the size predicted, at **-0.00326
against -0.00325** - a difference of 0.00001, identical to four decimals in
every league.

That is a no-op, not a win. It stays available and off, because an extra fit per
refit for nothing is not worth defaulting to.

### Nothing about the defaults changed

`blend 0.5` at `ridge 1.0` still ships. What this session bought is the
knowledge that the shipped value cannot be justified as a prior on team
strengths, and that this does not matter because the ridge is not functioning as
one. **What would actually address it** is modelling strength as a state that
evolves - a random walk, with the innovation variance estimated - rather than a
fixed effect fitted to discounted data. That would replace the half-life with
something fitted too. It is a rebuild of `models/goals.py`, and it is now the
best-motivated large change on the list.

## What was added in the availability session

Player availability from Understat line-ups, built, measured, and **left
switched off**, because it does not earn its place.

- `fbedge/availability.py` - three availability features, all computed from
  matches played strictly earlier, plus `for_fixture` for a match not yet
  played.
- `fbedge/understat.py` - `fetch_match` and `roster_frame`.
- `scripts/build_rosters.py` - resumable backfill into `match_lineups` and
  `match_availability`. 3,430 E0 matches, 98,515 appearances, ~45 min once.
- `fbedge/models/goals.py` - `fit_availability_effect` and
  `GoalsModel.availability_beta`, applied by `rates`.
- `scripts/availability_signal.py` - the measurement.
- 21 tests in `tests/test_availability.py`.

### The result: a consistent, well-behaved null

Poisson regression with the model's own fitted rate as an offset, so the
coefficients read directly as "what one missing regular does to the scoring
rate". E0, 2018-08 onward, 2,976 matches, bootstrapped by match:

| feature | window | beta_own | z | beta_opp | z | detectable at 2 SE |
|---|---|---|---|---|---|---|
| missing_starter_share | 2 | -0.102 | -0.66 | +0.164 | 1.20 | 2.86% |
| missing_xgchain_share | 2 | -0.094 | -0.86 | +0.164 | 1.53 | 2.00% |
| missing_share | 2 | -0.119 | -0.98 | +0.116 | 1.09 | 2.24% |
| missing_share | 1 | -0.013 | -0.11 | +0.165 | 1.58 | 2.08% |

**Both coefficients carry the predicted sign in every specification** - a side
missing players scores less, its opponent scores more - and not one of them
reaches two standard errors. The implied effects are around 1% on the scoring
rate per missing regular, against a study that could only resolve 2 to 2.9%.

The model-level test agrees. Backtesting E0 with the adjustment on and off:

```
use_availability=False   log loss 0.56611   gap +0.00703   CLV -0.0057
use_availability=True    log loss 0.56615   gap +0.00708   CLV -0.0061
```

A difference of 0.00004, which is nothing. Compare the blend and shrinkage
change, worth 0.0033. **`use_availability` therefore defaults to False.**

### What this does and does not rule out

It rules out an effect larger than about 2%. It says nothing about a 1% one,
and the consistent signs are weak evidence that a small real effect is there.

Two reasons not to read this as "team news does not matter":

- **The proxy cannot tell an injury from a substitute who was not used.**
  Understat lists only players who appeared, so a fit player left on the bench
  and a player in a hospital look identical. Roughly a third of team-matches
  show nobody newly missing, which is already implausible.
- **Confirmed line-ups are not available when the bet is placed.** They appear
  about an hour before kick-off, after the opening price and at or after the
  close. `fbedge/availability.py` is built around never reading them, and that
  restriction is the honest one; it is also why this can only ever be a proxy
  for what a real injury feed would say.

**If it is worth pursuing, the cheap next step is the other four leagues.**
16,000 matches instead of 3,000 would take the detectable effect to about 1.2%,
which is where the point estimates sit. That is roughly 2.7 hours of polite
requests and no new code - `build_rosters.py` takes `--league`. Do that before
considering a paid injury feed, and do not turn the flag on until something
clears the bar the blend cleared: better on leagues that had no part in
choosing it.

## What was added in the xG session

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

That table on its own proves nothing — it is nine settings searched on one
league with no holdout, and the tuner had already rejected a lower ridge once
on exactly that basis. What was genuinely new was the *mechanism*: a slope
above 1 is a reason to expect less shrinkage to help, and it was decided before
the search rather than read off it. So it got a holdout test.

### The holdout agrees, and this one is different from last time

`scripts/tune_hyperparameters.py --league E0 --from 2018-08-01 --target blend
--half-lives 180 360 --ridges 1 2 5 10` (the tuner now takes `--target`):

Development window, ridge is cleanly monotone at both half-lives —
1.0 → 0.5622, 2.0 → 0.5628, 5.0 → 0.5657, 10.0 → 0.5708.

```
Re-running the winner on the holdout window it never saw...
  tuned setting   log loss 0.5760   [blend, half-life 180d, ridge 1.0]
  defaults        log loss 0.5820   [goals, half-life 180d, ridge 5.0]
  -> the tuned setting held up out of sample (-0.0060).
```

**−0.0060 out of sample**, against the +0.0005 *worse* that the previous tuning
attempt produced. Five times the effect of xG on its own. The previous session's
"keep the defaults" verdict was correct for what it tested and is now
superseded for what it did not: it never varied the target, and at ridge 5 the
target barely matters.

Two cautions carried forward rather than waved away:

- **Ridge 1.0 sat at the edge of the grid.** Last session's note says a winner
  at the edge normally means extend the grid, and then says don't — but that
  was because the holdout had rejected the gain. Here the holdout accepts it,
  so the reasoning flipped and the grid was extended downward. **It is a broad
  plateau, not an edge**: 0.1 → 0.5623, 0.25 → 0.5622, 0.5 → 0.5621,
  1.0 → 0.5623, 2.0 → 0.5629, a spread of 0.0008 across the whole low range.
  Holdout at ridge 0.5 is −0.0063, consistent with the −0.0060 at ridge 1.0.
  Anything in 0.1–1.0 is equivalent; ridge 5 is the outlier. A flat optimum is
  much more trustworthy than a spike, because it cannot be an artifact of one
  lucky grid point.

  That run also exposed a **misleading message in the tuner**, now fixed: it
  reported "spread is small, keeping the defaults is a reasonable choice" for a
  grid whose largest ridge was 2 and whose default is 5. The settings searched
  were interchangeable *with each other*; the default was never in the grid and
  the sentence said nothing about it. It now says so, and points at the holdout
  instead.
- **The −0.0060 confounds two changes**: goals → blend, and ridge 5 → 1. That
  has now been decomposed on the holdout, and the answer is the interesting
  part of this whole session.

### The two changes are complementary, and that explains two earlier nulls

Same tuner, same windows, **goals** target, ridges 0.5/1/2/5:

```
dev:      ridge 2.0 -> 0.5651   1.0 -> 0.5652   0.5 -> 0.5655   5.0 -> 0.5666
holdout:  [goals, 180d, ridge 2.0] 0.5800  vs  defaults 0.5820   -> -0.0020
          "no meaningful difference out of sample. Keep the defaults."
```

So lowering shrinkage **on goals alone buys about −0.002 and does not clear the
bar**, while blend at low ridge buys **−0.0063**. The shrinkage fix is not
independent of the target; the two only pay off together.

The optimal ridge differs by target in exactly the direction the theory
predicts:

| target | best ridge on dev | reading |
|---|---|---|
| goals (noisier) | ~2.0 | needs more shrinkage; at 0.5 it is already overfitting (0.5655) |
| blend (less noisy) | ~0.5 | tolerates far less shrinkage, which is what lets the better signal show |

**This retrospectively explains two earlier negative findings, both of which
were correct for what they tested:**

1. The previous session's tuner said "keep the defaults". It only ever varied
   half-life and ridge on the *goals* target, and on goals alone low ridge
   really is worth almost nothing. Correct answer, incomplete question.
2. `compare_targets.py` at the shipped ridge of 5 found xG and goals a dead
   heat. At ridge 5 everything is shrunk hard towards the league average, which
   masks how good the underlying signal is. Comparing targets at a shared
   shrinkage was never a fair test, and the fair version is a cross-product.

The general lesson, worth keeping: **a hyperparameter tuned for one input is
not tuned for a better one.** Changing the data and holding the regularisation
fixed will systematically understate the new data's value.

### Validated across five leagues, and the defaults have changed

`scripts/validate_setting.py --from 2018-08-01 --target goals blend
--ridge 1 5`. E0 chose the setting, so it is reported but excluded from the
verdict; the other four had no part in choosing it.

| setting | mean Δ log loss | SE | t | better in |
|---|---|---|---|---|
| goals, ridge 1 | +0.00006 | 0.00085 | +0.07 | 2/4 — no better |
| blend 0.5, ridge 5 | −0.00128 | 0.00058 | −2.22 | 3/4 |
| **blend 0.5, ridge 1** | **−0.00325** | 0.00014 | **−23.7** | **4/4** |

The four held-out leagues give −0.0031, −0.0030, −0.0032, −0.0036. That
agreement matters more than the t-statistic, which is flattered by n=4 and by
leagues that are not fully independent.

**The calibration slope predicts the outcome, which is the part worth
believing.** Leagues whose goals model was already near slope 1 (D1 at 1.02,
F1 at 1.12) get *worse* with less shrinkage — their slopes overshoot to 0.82
and 0.91. Leagues that were over-shrunk (I1 at 1.30, SP1 at 1.29) improve, and
their slopes move to 1.04 and 1.03. `blend r5` is the most over-shrunk of all
(1.17–1.51), which is exactly why it needs the lower ridge, and `blend r1`
lands nearest 1 and wins. A mechanism decided in advance predicting
out-of-sample results is a different kind of evidence from a grid winner.

**So the defaults changed.** `models.base.DEFAULT_TARGET = "blend"`,
`DEFAULT_BLEND_WEIGHT = 0.5`, and shrinkage is now per-target through
`RECOMMENDED_RIDGE` / `default_ridge()` — 5.0 for goals, 1.0 for xG or blend.
`DEFAULT_RIDGE` still exists at 5.0 for anything that asks for it by name.

Three things that made this more than a one-line change, all of them tested:

- **`target=None` means "the default, and adapt"; a named target is a
  promise.** An unspecified target silently degrades to goals on a database
  with no xG, because a default that hard-fails on an older database is
  useless. An explicitly requested `"xg"` or `"blend"` still raises
  `MissingExpectedGoals`. `build_models` turns the silent downgrade into a
  note, since it is the layer that has somewhere to put one.
- **`ridge=None` means "whatever suits the target".** An explicit value always
  wins. The count models need a real number, so they take the value the goals
  fit resolved rather than resolving it again and risking disagreement after a
  fallback.
- **The app couples the two controls.** Choosing a blend and leaving shrinkage
  at the goals value is the worst available combination, so the recommended
  value follows the target and is on by default.

**Everything measured before this change was measured under the old defaults**
— including the whole RESOLVED section above and the gate verdict. Those
numbers are not wrong, but they describe a model that is no longer what ships.

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

0. **The other four leagues for availability**, if that thread is worth
   pulling: 16,000 matches instead of 3,000 takes the detectable effect from
   about 2.5% to about 1.2%, which is where the point estimates sit. No new
   code, roughly 2.7 hours of polite requests.
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
   hyperparameter fixes. Phase 5 covers xG from Understat (mostly delivered;
   the blend is the shipping default) and schedule congestion (`rest_days` and
   `matches_last_14_days` are computed but never fed into the fit).

   **FBref for UEFA is off the list.** Stats Perform terminated FBref's data
   agreement on 20 January 2026 and required removal of every advanced
   statistic; Sports Reference announced it on 23 January. Results remain on
   the site but `fbref.com` now answers HTTP 403 to a scripted request
   (verified 2026-09-04, browser user-agent, two network paths), so it is not a
   source here at all. UEFA is rescoped to results only, and the honest note is
   that there is no free UEFA xG at any quality — Understat covers the big five
   plus the RFPL from 2014/15 and nothing European — so a European fixture
   could only be fitted on goals, which is a different model from the one every
   number in this document describes. See BACKLOG B13.
4. Consider that the honest conclusion may be: this is a well-built,
   well-tested learning project, and that is a fine place for it to stay.

---

## Standing rules

- **Phase 4 is built (2026-09-04) and the gate is still shut.** The rule was
  "do not build Phase 4 unless the gate opens", and it was overridden by an
  explicit instruction rather than by evidence. CLV is unchanged at -1.500%
  (-9.1 SE). See the section at the top of this document for what the build
  does to keep that visible on every screen it produces. **Building a scanner
  is not the same as having something to act on**, and nothing here has
  reopened the gate: it still needs CLV indistinguishable from zero *and*
  holding up on a tuner holdout window it never influenced.
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
