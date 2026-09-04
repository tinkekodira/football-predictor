# Known bugs and deferred work

Things found and deliberately not fixed, kept so they are not re-discovered
from scratch. Entries marked FIXED are kept rather than deleted: the reasoning
is the useful part, and a fixed bug that leaves no trace is one somebody
reintroduces. Each entry says what is wrong, how it was confirmed, and what
fixing it would involve. Ordered by how much damage it can do if forgotten.

Nothing here blocks the current model. Every one of them was found while doing
something else, and shelved on purpose.

---

## B1. `fair_line_preference` cannot actually pin the benchmark — FIXED

**Severity: high — it made a tool lie to you.**

`_fill_market` (`fbedge/backtest.py:257`) builds its bookmaker order from the
preference list and then appends everything else:

```python
ordered = [b for b in preference if b in available]
ordered += sorted(available - set(ordered))
```

So `BacktestConfig(fair_line_preference=("pinnacle",))` does **not** restrict
the benchmark to Pinnacle. It prefers Pinnacle and silently falls back to any
other book on matches Pinnacle did not price. The fallback is deliberate and
documented in the comment above it — it maximises coverage — but there is no
way to turn it off.

That matters because `scripts/season_breakdown.py:130` prints exactly that
constructor as the remedy when it detects a mid-window benchmark change. The
script detects a real problem and then hands out a fix that does nothing. The
Pinnacle-pinned numbers in `HANDOFF.md` were produced by computing fair lines
directly from the odds table instead, working around this.

**FIXED 2026-09-04.** `BacktestConfig.fair_line_fallback` (default `True`, so
nothing silently narrows) is threaded to `_fill_market`, and
`season_breakdown.py --pin-benchmark pinnacle` sets both halves - naming the
preference alone was never enough. Four tests in `tests/test_benchmark.py` pin
the distinction, including one that reproduces the old behaviour so it cannot
come back quietly.

**It mattered.** E0 from 2023, 1X2, same window and code:

| season | unpinned (benchmark switches) | pinned to Pinnacle | bets |
|---|---|---|---|
| 2023 | -1.54% (pinnacle) | -1.54% | 480 / 480 |
| 2024 | -1.71% (betfair) | -1.50% | 487 / 487 |
| 2025 | -1.90% (betfair) | **-1.36%** | 494 / **253** |

The apparent deterioration across those three seasons is mostly the instrument
again. Note the cost, which the fix reports rather than hides: pinning drops
half of 2025, because Pinnacle did not price those matches, so the two columns
are not the same measurement. That is the honest trade - you cannot compare
across seasons without pinning, and pinning costs coverage.

## B2. `build_xg.py` holds a write lock across the whole network fetch — FIXED

**Severity: medium — locked the database for the length of a download.**

`scripts/build_xg.py:70` opens `database.connect(..., read_only=False)` before
calling `collect()`, which does the Understat requests. DuckDB allows one
writer, so on a cold cache the app, the tests and every read-only script are
locked out for the entire run.

This is the same defect that was found and fixed in `scripts/build_rosters.py`
during the availability session, where it locked the database for ~50 minutes.
`build_xg.py` was never given the same treatment.

**FIXED 2026-09-04.** The fixture list is read on a read-only connection which
is closed before any download; the write connection is opened at the end, for
the insert alone. Mirrors `build_rosters.py`.

**Running it to check the fix exposed a worse bug in the same file, also
fixed.** The write did `DROP TABLE match_xg` and then inserted only the leagues
that run had processed, so `python scripts/build_xg.py --league E0` **silently
deleted the xG for the other four leagues** - which is exactly what happened
while testing. It now creates the table if absent and deletes only the rows
belonging to leagues the run actually covered, the same scoped-delete shape as
`fixtures.write_calendar` and `injuries.write_injuries`. Verified: a single-
league run leaves the other four intact.

## B3. No `.gitignore`, and build artifacts are tracked — FIXED

**Severity: medium — every commit needed hand-listed paths.**

**FIXED 2026-09-04.** There is now a `.gitignore`, and the sixteen `.pyc` files
plus `data/football.duckdb` were untracked with `git rm --cached`, so nothing
left anyone's disk and no history was rewritten.

**The database is no longer tracked**, and the deciding fact was not its size.
The committed copy came from the *initial commit* and had never been updated:
four tables against the working copy's eight, missing `fixtures`, `injuries`,
`match_xg` and `match_lineups`. A fresh clone therefore got a database the home
page could not open — repository weight *and* a misleading artifact. It is
rebuilt by `scripts/build_database.py` in about two minutes.

**The season CSVs stay tracked**, which is what makes that rebuild work with no
network. They are ~8MB, effectively immutable once a season ends, and the one
input that could not be regenerated if football-data.co.uk disappeared.
`data/raw/crest_ids.json` stays for the same reason at a smaller scale: two
kilobytes that otherwise cost requests from a 100-a-day allowance.

**Club badges are ignored for a licensing reason, not a size one.** They are
third-party club crests from a provider's CDN, and a public repository should
not redistribute them. `crests.monogram` is why the app still looks finished
without them.

**The 13.5MB database blob from the initial commit remains in history.** Purging
it needs `git filter-repo`, a rewrite of every commit hash and a force-push,
which is disproportionate for a single blob on a solo repository — a deliberate
choice, not an oversight.

## B4. Congestion features are computed and thrown away

**Severity: low — dead signal, not a defect.**

`fbedge/profile.py` computes `rest_days` and `matches_last_14_days`
(`_rest_profile`, line 331) and they are only ever *displayed* — in the profile
text and in `app.py:566`. Nothing feeds them to a fit.

They are already point-in-time by construction, which is the expensive part.
Wiring them into `GoalsModel.rates` as a rate shift would reuse the machinery
`fit_availability_effect` already established, and could be measured the same
way `scripts/availability_signal.py` measures availability.

## B5. One ridge for five leagues, when the evidence says it should differ

**Severity: low — a known, quantified compromise.**

The five-league validation (`HANDOFF.md`, "Validated across five leagues")
found that `blend, ridge 1` wins overall while making **D1 and F1 worse**:
their goals models were already near-calibrated (slope 1.02 and 1.12), so less
shrinkage overshoots them to 0.82 and 0.91. I1 and SP1 were over-shrunk and
improve.

The shipped default therefore knowingly hurts two leagues out of five. Choosing
ridge per league from the calibration slope is the obvious move and is exactly
the kind of per-league tuning `HANDOFF.md` warns produces spurious winners, so
it needs the same held-out discipline: pick the *rule* on some leagues, test it
on others.

**One principled attempt has already been made and failed.** Empirical Bayes
(`models/hierarchical.py`, `ridge="auto"`) derives a per-league ridge from the
data with no search at all, which is exactly the kind of rule that should be
allowed to differ between leagues. It lands at 11-13 against a shipped 1.0 and
loses on 4 of 4 held-out leagues. See B9 and the hierarchical section of
`HANDOFF.md`. So this entry stays open, but a second attempt should not repeat
that one.

## B6. The availability proxy cannot separate injury from rotation

**Severity: known limitation, measured, documented.**

Understat lists only players who appeared, so a fit player left on the bench is
indistinguishable from one in hospital. See the availability section of
`HANDOFF.md` for the full null result. Two ways forward, both blocked on data
rather than code: a real injury feed, or a squad-list source that says who was
available but unused.

Confirmed line-ups arrive about an hour before kick-off, which is after the
price the bet is struck at, so even a perfect version of this feature cannot
use them. `fbedge/availability.py` is built around that constraint.

**Retested, and the null may not survive.** See the retest section of
HANDOFF.md: with real injuries, *newly* absent players are worth about -7% on
the scoring rate each on one season of E0, while the count of everyone
currently out is worth nothing. Needs 2022-2023 and the other four leagues -
fourteen free requests - before it is a result.

**A source now exists for the first of those.** `fbedge/injuries.py` reads
API-Football, which states a reason per player per fixture ("Ankle Injury"),
and a **free** key covers 2022 to 2024 - the same era the availability study
used. That makes the null retestable with real injuries instead of a rotation-
contaminated proxy, at no cost. It is the cheapest open question in this file.

## B7. The count models cannot be validated against any market

**Severity: structural — no code fix exists.**

`extract_odds` only produces `1x2`, `total_goals` and `asian_handicap`.
football-data.co.uk carries no corner or card prices, so the negative binomial
models in `fbedge/models/counts.py` have no market to be scored against.
`BacktestConfig.fit_count_models` defaults to `False`. This is Finding 3 in
`HANDOFF.md` and it needs a different odds source, not a change here.

## B8. `calibration_table` still mirrors two-sided markets

**Severity: low — correctly documented, wrong function still callable.**

Pointing `calibration_table` at `total_goals` counts every match twice, once
per side, and produces a perfectly mirrored table. The docstring now warns
about it and `calibration_by_line` is the correct function, but nothing stops
the wrong call. Cost a session to diagnose once already.

## B9. The empirical-Bayes ridge loop runs away instead of settling

**Severity: known limitation of a feature that ships switched off.**

`models.hierarchical.empirical_bayes_ridge` has no interior fixed point on
real league data. Both terms of the EM update - the observed spread of fitted
strengths and the posterior variance - shrink as the penalty rises, so nothing
pushes back and the iteration climbs monotonically. Left to run for 30 rounds
it reaches the upper bound of `RIDGE_BOUNDS` in three leagues out of five.

It is not a coding error: `test_recovers_a_planted_prior_variance` shows the
estimator recovers a known `tau` on synthetic leagues at realistic size, and
`test_the_loop_runs_away_on_a_flat_likelihood` pins the runaway so a later
change that hides it can be told from one that fixes it.

The cause is in the model the estimator assumes, and it is written up in the
module docstring and in the hierarchical section of `HANDOFF.md`: the weighted
likelihood is treated as a real likelihood over roughly 300 observations, under
which team strengths are barely resolved and almost all their observed spread
looks like noise. The time weights discount old matches because strength
*drifts*, not because those matches were noisy.

**What would actually fix it** is a model in which strength is a state that
evolves - a Kalman filter or a random walk over team strengths, with the
innovation variance estimated - rather than a fixed effect fitted to
exponentially discounted data. That is a substantial rebuild of
`models/goals.py`, and the honest reason to consider it is not the ridge but
the fact that it would replace the half-life with something estimated too.

`ridge="auto"` is kept because reproducing the disagreement is the whole
evidence for this entry. Do not turn it on.

## B10. The Betfair Exchange benchmark was silently missing for two of three markets — FIXED

**Severity: high — one backtest measured two markets against two different
instruments.**

`normalize.BOOKMAKERS` has carried `"BFE": "betfair_exchange"` since the
exchange first appeared in the source, and `backtest.FAIR_LINE_PREFERENCE`
puts it first precisely because an exchange charges commission rather than
building a margin into its price. But `TOTALS_PREFIXES` and
`HANDICAP_PREFIXES` were never updated, so `BFE>2.5`, `BFE<2.5`, `BFEAHH` and
`BFEAHA` — present in every file from 2024/25 — were never extracted.

The consequence was not a missing column, which would have been visible. It
was that 1X2 CLV from 2024/25 onwards was measured against Betfair while
totals and handicap CLV silently fell through to Pinnacle, inside the same
run. B1 established that a benchmark change rewrites CLV; this was a permanent
benchmark *split* along market lines.

**Found 2026-09-04** by diffing the `fixtures.csv` header against the mapping
during the Phase 4 work — the check the brief asked for rather than assumed.
The same diff found `SKB*`: the source's SkyBet prefix is `SKB`, the mapping
said `SK`, so SkyBet prices were dropped everywhere. New in 2026/27, so the
loss was 594 rows rather than a decade of them.

**FIXED 2026-09-04.** Both added. Measured cost of the correction, E0
2024-08-01 to 2026-08-31, same code and window:

| market | benchmark Pinnacle (old) | benchmark Betfair (fixed) | bets |
|---|---|---|---|
| total_goals | -1.745% (-4.9 SE) | **-1.993% (-5.5 SE)** | 495 |
| asian_handicap | -1.968% (-9.4 SE) | **-1.871% (-8.7 SE)** | 600 |

Both move by about two tenths of a point and in opposite directions, so no
conclusion in `HANDOFF.md` changes. That it is *small* is the finding: the
instrument was wrong and the answer was not, which is the opposite of B1.

## B11. Handicap prices were stored with no line to settle them against — FIXED

**Severity: medium — a third of the handicap table was unusable.**

`_build_handicap_specs` emits three open-phase specs per bookmaker, one for
each place the line might live (`B365AH`, `AHh`, `BbAHh`). `extract_odds` only
skipped a spec when the *price* column was missing, so a spec whose *line*
column was absent still wrote rows — with `line` NULL.

98,498 rows, about 30% of `asian_handicap`. They were never a wrong number:
`markets.price_selection` returns None for a handicap with no line, so the
backtest skipped them silently. They were dead weight that made every
`SELECT ... WHERE market = 'asian_handicap'` count wrong by a third.

**FIXED 2026-09-04.** `normalize.odds_long` now returns early when a requested
line column is absent. A price with no line is not a price: nothing can settle
"home at 1.95" without knowing the start. After a rebuild, `asian_handicap`
holds 225,604 rows of which 30 still carry a NULL line — those are genuinely
blank `AHh` cells in the source, not a spec mismatch.

## B12. The source now publishes its own xG, and nothing reads it

**Severity: low — an unused input, not a defect.**

`HxG` and `AxG` appear in the 2026/27 season files and are in no mapping table.
The project already has xG from Understat, which covers 2014/15 onward, so this
is not a gap in coverage — it is a second opinion on one season.

Wiring it in is a model change, not an ingest change: `models/base.py` fits
team strengths to a goals/xG blend and swapping the xG source mid-history would
mean the blend is computed from Understat before 2026 and from
football-data.co.uk after, which is precisely the kind of silent instrument
change B1 and B10 are both about. If it is ever done, it should be done as a
*comparison* — `scripts/compare_targets.py` already scores two targets on
identical matches — and not as a substitution.

---

## Not bugs — open questions, kept here so they stay visible

- **Why 2023.** CLV falls from about 0% to about -2% and stays there for three
  seasons (-3.1, -3.7, -4.9 SE). It survives the margin fix and 2023 carries no
  Betfair data, so it is not the benchmark artifact. Unexplained.
- **The other four leagues for availability.** 16,000 matches instead of 3,000
  takes the detectable effect from ~2.5% to ~1.2%, where the point estimates
  sit. No new code, roughly 2.7 hours of requests.
