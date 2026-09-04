# Known bugs and deferred work

Things found and deliberately not fixed, kept so they are not re-discovered
from scratch. Each entry says what is wrong, how it was confirmed, and what
fixing it would involve. Ordered by how much damage it can do if forgotten.

Nothing here blocks the current model. Every one of them was found while doing
something else, and shelved on purpose.

---

## B1. `fair_line_preference` cannot actually pin the benchmark

**Severity: high — it makes a tool lie to you.**

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

**Fix:** a `fair_line_fallback: bool = True` field on `BacktestConfig`, threaded
to `_fill_market`, that drops the `ordered +=` line when false. Report the
coverage lost, because pinning will drop matches and a silently smaller sample
is the next version of this same bug. Then make `season_breakdown.py` print an
instruction that works.

## B2. `build_xg.py` holds a write lock across the whole network fetch

**Severity: medium — locks the database for the length of a download.**

`scripts/build_xg.py:70` opens `database.connect(..., read_only=False)` before
calling `collect()`, which does the Understat requests. DuckDB allows one
writer, so on a cold cache the app, the tests and every read-only script are
locked out for the entire run.

This is the same defect that was found and fixed in `scripts/build_rosters.py`
during the availability session, where it locked the database for ~50 minutes.
`build_xg.py` was never given the same treatment.

**Fix:** read the fixture list on a read-only connection, close it, do the
fetching, then open a write connection only for the final `CREATE`/`INSERT`.
Mirror `build_rosters.py`, which already has the correct shape.

## B3. No `.gitignore`, and build artifacts are tracked

**Severity: medium — every commit needs hand-listed paths.**

There is no `.gitignore`. Sixteen `.pyc` files and the 14 MB
`data/football.duckdb` are tracked, so running anything at all dirties the tree
and `git add -A` sweeps in bytecode churn plus a binary database. Every commit
in this project is made with explicit paths for that reason.
`data/football.duckdb.wal` now shows up untracked as well.

**Fix, and it needs a decision rather than a default:** ignore `__pycache__/`,
`*.pyc`, `*.wal` and `data/raw/`, then `git rm --cached` the bytecode. Whether
`data/football.duckdb` stays tracked is a real choice — it makes the repo
self-contained and reproducible, at 14 MB a commit whenever it changes.

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

---

## Not bugs — open questions, kept here so they stay visible

- **Why 2023.** CLV falls from about 0% to about -2% and stays there for three
  seasons (-3.1, -3.7, -4.9 SE). It survives the margin fix and 2023 carries no
  Betfair data, so it is not the benchmark artifact. Unexplained.
- **The other four leagues for availability.** 16,000 matches instead of 3,000
  takes the detectable effect from ~2.5% to ~1.2%, where the point estimates
  sit. No new code, roughly 2.7 hours of requests.
