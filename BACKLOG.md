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

## B13. FBref is gone as a source, and not only for xG

**Severity: structural — it removes a roadmap item, and no code fixes it.**

The roadmap said Phase 5 would add "xG from Understat, UEFA via FBref". The
second half no longer exists.

On 20 January 2026 Stats Perform (Opta) terminated FBref's data agreement and
required immediate removal of every advanced statistic. Sports Reference
announced it on 23 January and chose not to contest the claimed breach, citing
the cost of litigation. Results, schedules, squads and basic match statistics
remain. xG and every Opta-derived metric are gone indefinitely.

**Checked on 2026-09-04 rather than assumed, and the check found something
worse.** `fbref.com/en/comps/9/Premier-League-Stats` and `fbref.com/en/` both
return **HTTP 403** to a scripted request — with a realistic browser
user-agent, and through two separate network paths. So FBref is unusable here
even for the results-only rescope, not merely stripped of the advanced data.

**What to use instead**, if UEFA is ever wanted: football-data.org's free tier
includes the Champions League and `fbedge/calendar.py` already speaks to it.
openfootball has a Champions League file for 2024/25 and **not** for 2025/26 or
2026/27, which is a concrete instance of the "occasional gaps mid-season" that
module's docstring warns about — do not build on it for this.

**The consequence for the model is not the source, it is the xG.** Understat
covers the big five domestic leagues plus the RFPL from 2014/15 and has no UEFA
coverage. The shipping model fits team strengths to a goals/xG blend, so a
European fixture could only ever be fitted on goals — a different and worse
model than the one every domestic number in `HANDOFF.md` describes. Mixing the
two in one fit would be the same class of silent instrument change as B1 and
B10.

**Backtest side: already guarded.** A UEFA match arrives with a result and no
odds. `BacktestConfig.calibration_markets` and `BacktestResult.fitted_not_bettable`
make that explicit: odds-less matches are fitted and priced, contribute to
calibration and to team strengths, and can never be settled as bets, because a
record with no price has no expected value. Every run prints the count.

## B14. The two "does team news matter" scripts get mistaken for each other

**Severity: low — a documentation defect that has already misdirected work.**

They answer the same question from different data and have opposite
constraints, and the names do not make that obvious.

`scripts/availability_signal.py` reads `match_lineups`, which
`scripts/build_rosters.py` downloads from Understat. **No key, no quota, every
season already in the database.** It has been run across five leagues and the
result is the documented null. It is also the free availability proxy itself,
implemented in `fbedge/availability.py`, with the honest caveat that Understat
lists only players who *appeared* — so it conflates injury, suspension,
rotation and transfer.

`scripts/injury_signal.py` reads API-Football. **Keyed, 100 requests a day, and
capped at recent seasons.** It is the one with a data ceiling: roughly 2022 to
2024, which is three seasons of a question that wants ten. It has been run and
produced the retest result, and `--status` now prints exactly what the
unanswerable half needs and what it would cost.

**This has cost time once.** A brief written on 2026-09-04 asked for
`availability_signal.py` to be marked blocked and excluded from CI on the
grounds that it needed a historical injury record, and then asked for the
line-up proxy to be implemented as the honest alternative — which is what that
script already is. Neither was done, because both were already false. Both
docstrings now open by saying which of the two they are.

## B15. A schema change can look like a price move in the snapshot archive

**Severity: low now, and permanent if it is not noticed at the time.**

The archive hashes each fixture's identity plus every price attached to it, so
any change to *what is extracted* produces a new content hash and a new
snapshot - which reads, later, as the market having moved.

It happened immediately. The first pull on 2026-09-04 ran before the B11 fix
and carried eight NULL-line Asian handicap rows per fixture. The pull 25
minutes later did not. All 47 fixtures therefore appeared to change price, and
none had: the 1X2, totals and handicap prices were byte-identical.

**Cleaned, and this is the one deletion the archive will ever accept.** Before
deleting, every non-NULL-line price in the superseded snapshots was checked to
exist in the later one - zero real observations would be lost - and then the
pre-fix pull was removed. The archive is append-only *for price observations*;
a row that records a parser version rather than a market is not one.

**What to do next time.** If the odds extraction changes again, either accept a
one-off spurious change and note the date here, or clean it the same way within
the same session. Do not clean it later: after real price movement has
accumulated, "identical apart from the schema" stops being checkable.

The alternative - versioning the hash so a parser change is visible as a
parser change - is the proper fix and is not worth it for a solo archive that
has changed schema once.

## B16. Odd/even total goals was priced, measured and withdrawn — DONE

**Severity: none. A result, recorded so it is not undone.**

Odd/even is two lines of arithmetic off the score matrix, so it was added in
Phase 4 along with the other cheap derivations. `scripts/build_evidence.py`
then scored it, and the answer was unambiguous:

| league | calibration slope |
|---|---|
| E0 | 0.18 |
| SP1 | -2.41 |
| I1 | -0.05 |
| D1 | 1.54 |
| F1 | 0.69 |

A slope of 1.0 is correct calibration. **A negative slope means the model's
confidence pointed the wrong way**: in La Liga and Serie A, fixtures it called
likelier to be odd came in even slightly more often. Across five leagues the
estimates do not agree on a sign, let alone a size.

That is not a defect and there is nothing to fix. The parity of a total is
close to a coin flip by construction: it flips on any single goal, so it
depends on the exact shape of the scoring distribution rather than on anything
team strength determines. The model has no reason to know it, and the
measurement says it does not.

**Removed on 2026-09-04**, rather than kept behind a warning label. A market the
model cannot rank is not worth the row it occupies on a page, and the project's
own standard is that a number and its evidence appear together — the evidence
here says "do not read this", which is a reason to delete the number.

**What survives.** `evidence.WITHDRAWN_MARKETS` keeps the name and the reason,
so an evidence row stored before the removal reports as *withdrawn* rather than
as a market with no data — those look identical otherwise, and only one of them
is a reason to go bug-hunting. `settlement.settle` raises with the same
explanation rather than silently failing to match. And
`test_a_withdrawn_market_is_not_priced_and_cannot_be_settled` pins it, because
this is exactly the kind of two-line derivation somebody re-adds while
enumerating what a score matrix can produce.

## B17. The scan's largest numbers were its least trustworthy — FIXED

**Severity: high, and of a kind this project cares about more than most. It
was not a wrong number; it was a correct number presented so that the worst
rows sorted to the top.**

A pre-match scan sorted by expected value ranks the model's confidence. Its
confidence is highest where it knows least, so the first live run led with two
newly promoted clubs at +68% and +50% EV on two matches of history each.
Nothing was miscomputed — the promoted-team prior is deliberately pessimistic,
the market disagrees, and the arithmetic follows. The presentation was the
defect.

**Measured before fixing, five leagues, 2022-2026, 19,112 settled bets**,
grouping every bet by the thinner side's match count at the time:

| thinner side had | bets | mean EV | median EV | realised ROI |
|---|---|---|---|---|
| 0-4 matches | 290 | **+17.5%** | +12.4% | -8.0% |
| 5-9 matches | 263 | +16.2% | +12.1% | +4.0% |
| 10-24 matches | 704 | +12.5% | +8.9% | -10.3% |
| 25 or more | 17,855 | **+12.8%** | +9.1% | -4.9% |

The claimed edge on a barely-known side runs about a third higher and does not
pay for it. The ROI column is a few hundred bets deep in the thin buckets and
is not evidence on its own — the mean-EV column is the finding.

**And separately, by EV band** (E0, 4,175 settled bets): +20% is the 78th
percentile of the model's own claimed edges, and the bets above it returned
-5.2% against +2.1% for everything at or below. Over a few hundred bets that is
an absence of evidence that they are better, not proof they are worse. It is
enough to stop ranking on them.

**FIXED 2026-09-04.** `config.SCAN_MIN_TEAM_MATCHES` (5) and
`config.SCAN_MAX_TRUSTED_EV` (0.20), applied in `scan_fixtures._withhold`. A
row tripping either is withheld from the ranked table and printed under its own
heading with the reason. **Withheld, never dropped** — a fixture absent from a
scan looks exactly like a fixture nobody priced, and this project has been
bitten by that shape of bug before. `--include-withheld` restores the old
behaviour for anyone who wants it, and both thresholds are flags.

**What this does not fix.** The ranking is still a ranking of the model's
disagreement with the market, and the model's closing line value is still
negative. Removing the rows where it is most obviously wrong does not make the
remaining ones right. See the gate section at the top of `HANDOFF.md`.

---

## Not bugs — open questions, kept here so they stay visible

- **Why 2023.** CLV falls from about 0% to about -2% and stays there for three
  seasons (-3.1, -3.7, -4.9 SE). It survives the margin fix and 2023 carries no
  Betfair data, so it is not the benchmark artifact. Unexplained.
- **The other four leagues for availability.** 16,000 matches instead of 3,000
  takes the detectable effect from ~2.5% to ~1.2%, where the point estimates
  sit. No new code, roughly 2.7 hours of requests.
