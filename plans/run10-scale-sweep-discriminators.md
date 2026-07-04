# Plan: Run 10 — Scale-Sweep Discriminators (deciding bounded-J vs log-blowup)

## Status: IMPLEMENTED and RUN. Experiment 1 result: INCONCLUSIVE, not a
## verdict for either hypothesis -- see "Doubled-budget rerun" section below
## for the final numbers. Getting to even this honest inconclusive result
## required finding and fixing FOUR distinct methodological issues in a row
## (search-noise contamination, wrong statistic, a resolution floor, and a
## budget that didn't port from Run 9's setup) -- full history below.
## Experiment 2 (window_ratio) has NOT had the same fixes applied yet and
## should be read as suggestive only. Experiments 4-5 not run/built.

## New finding: search-noise contamination in `generation_step` (found here,
## applies retroactively to Run 9 too)

While running Experiment 1 (`scale_sweep` on Run 6's grown base), a sweep
point showed **both** new kinks at `jump_mean=0.000` (dead at convergence,
i.e. the LP gave the inserted generation literally zero weight everywhere)
yet `dJ` was still `+0.037` — nonzero, not the ~0 you'd expect if the
insertion contributed nothing. A direct control test confirmed the cause:
re-running the identical regridded base through `_alternate` with **no
insertion at all** (same budget, one deterministic call) gained exactly
`+0.00000` — ruling out "more optimizer iterations alone explain it". But
re-running `generation_step`'s full multi-seed insertion machinery with the
new kinks' alive masks **forced permanently dead** (so the LP can never give
them weight, regardless of window) still gained `+0.058` at `w=0.5` and
`+0.021` at `w=0.03125` — comparable to, and at `w=0.5` actually *larger*
than, the real windowed arm's measured `dJ` at those same points.

Mechanism: `generation_step`'s multi-seed loop uses a different random
insertion jitter per seed to perturb the position-NLP's starting vector (and,
while the new kink is transiently alive during early outer iterations, the
GAP spacing penalty against same-family kinks). Even when the new kink ends
up pruned to zero weight at convergence, the perturbed search path can settle
the OLD kinks into a different — sometimes better — local optimum than the
un-perturbed base. This is exactly the "multistart on the insertion jitter"
mechanism the `generation_step` docstring already documented as intentional
— but its side effect (inflating `dJ` even when the "generation" itself
contributes nothing) had not been isolated or measured before.

**This is not exclusive to Run 10.** `generation_step` is the same function
Run 9's `generation_ladder` calls every generation, so Run 9's `dJk` numbers
carry an unknown amount of this same contamination (likely partially real
signal, partially search-noise — Run 9's diagnostics showed mostly nonzero
`jump_mean`, unlike Run 10's Experiment-1 case, so the contamination
fraction there is probably smaller, but has never been measured).

**Fix implemented:** `generation_step(..., force_dead=True)` is a new
null-control arm — identical seeded search, new columns permanently masked
dead so the LP can never weight them. `scale_sweep` now runs this
alongside the windowed and guard arms and reports `null_dJ` and
`corrected_dJ = dJ - null_dJ` per point. `corrected_dJ(w)`, not raw `dJ(w)`,
is the number that actually answers Experiment 1's question.

**Not yet fixed:** `generation_ladder` (and therefore Experiment 2's
window_ratio discriminator) does not yet run a null-control arm — its
`dJk`/`decay_ratios` numbers should be read as suggestive, not settled,
until that's added. See "What to run next" below.

Continuation of `run9-generation-gain-ladder.md`. Run 9's trustworthy ceiling
is n_gen=3 on two bases, both showing geometric-ish dJk decay — but three
points per base, and decay is exactly the signature that optimizer starvation
fakes (three separate budget artifacts already caught in Run 9). This plan
decouples "gain truly vanishes at small scale" from "optimizer starved",
using experiments that run at already-proven budgets wherever possible.

## The question (unchanged from STRATEGY.md Section 5)

Does the certified gain from adding a generation at lifetime-window scale w
stay O(1) as w -> 0 (log law, sup J = +infinity), or does it vanish with w
(J bounded, mesh log growth was a discretization transient)?

## Why the ladder alone can't answer it

`generation_ladder` confounds three things that all grow together:

1. window scale w_k (the variable of interest),
2. accumulated kink count (position-NLP dimensionality),
3. shared graded-grid size (precomputed once for the whole ladder).

Both n_gen=5 attempts died on (2) and (3), not (1). The experiments below
isolate (1).

## Experiment 1 — single-generation scale sweep dJ(w) (cheapest decisive test)

From a fixed base (Run 6's grown topology, J_certified 2.4114, budget already
proven adequate at outer=40/pos_iters=100/seeds=5), insert ONE windowed
generation at each of

    w in {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625}

with each point fully independent: its own `graded_grid` sized for that
single window, same kink count every time (base + 1 f-kink + 1 g-kink via
`generation_step`), same budget. No accumulation anywhere — one verified
budget covers all points. Record dJ(w) = Jc(w) - Jc(base), certified, with
per-seed spread, plus the guard arm (free insertion) at each point as the
usual comparison.

Read the semilog plot of dJ vs w:

- **dJ(w) -> nonzero constant as w -> 0**: gain per scale is genuinely O(1);
  log law survives, and pushing the ladder deeper (Experiment 4) is
  justified.
- **dJ(w) ~ w^alpha, alpha > 0**: generation sums converge geometrically,
  J bounded. Done, modulo Experiment 2 confirming.

Key property: if even the FIRST kink inserted at scale w gains nothing, no
amount of hierarchical stacking rescues log growth — this bounds the ladder
from above without ever running it.

## Experiment 2 — window_ratio discriminator (no budget escalation needed)

Run 9's hi-budget data already hints dJk decay ratio tracks window_ratio:
Run3-base ratios 0.31, 0.48 against window_ratio 0.5. The two hypotheses
split sharply on this:

- **Log law**: per-generation gain is scale-free, so dJk should be roughly
  independent of window_ratio.
- **Bounded, dJ proportional to w**: dJk ratio should approximately equal
  window_ratio.

Rerun the n_gen=3 ladders (both bases) at window_ratio 0.7 and 0.3, keeping
n_gen=3 so grids stay in the 41-node class where outer=40/pos_iters=100 is
proven adequate. If the ratio-0.7 ladder decays at ~0.7 and the ratio-0.3
ladder at ~0.3, boundedness is strongly supported without touching gen4+.
If both decay at the same rate regardless of window_ratio, that's evidence
the decay is NOT scale-driven — rethink before concluding.

## Experiment 3 — automated budget-adequacy gate (prerequisite, wraps 1-2)

Run 9 caught three budget artifacts by hand. Formalize the acceptance test
so no point enters a plot ungated. A data point is accepted only if:

1. majority of seeds feasible (`constraints_ok`),
2. every inserted kink has `jump_mean > 0` (not starved),
3. **budget-doubling stability**: rerun the point at 2x outer/pos_iters;
   accept only if dJ moves by < max(10%, 0.005).

Points failing the gate are reported grey/excluded, not data. Implement as a
small wrapper around `generation_step` results (the diagnostics needed —
feasibility, jump_mean — are already computed and persisted by
`kink_opt/persist.py`).

## Experiment 4 — falsifiable extrapolation (only if 1-2 ambiguous)

Fit a geometric to each base's dJ1-3, and **record the predicted dJ4 in this
file before running it**. Then measure gen4 once, with per-generation
escalating budget (Run 9 "What to run next" item 1: outer/pos_iters scaling
with accumulated kink count and `n_live_nodes`, e.g. outer_k = outer0 *
(1 + 0.3k)), gated by Experiment 3's criteria. Landing on the prediction
confirms bounded; overshooting says the transient isn't over. One expensive
point, maximal information.

## Experiment 5 — constructive arm (the only path to PROVING blowup)

The optimizer can only ever fail to find gain; it can never certify gain is
absent, and it can never prove blowup. The prove-side needs a construction:
hand-build an explicit self-similar hierarchy (analytic kink trajectories
and weights per generation, no optimization at all) and evaluate it directly
with `certify()`. Budget artifacts are impossible by construction.

- If a family with per-generation certified gain >= c > 0 exists, the log
  law is established constructively (numerically certified, and the
  construction is then a proof sketch).
- If every attempted construction saturates, the *mechanism* of saturation
  (per-x rise budget: int f_t dt = -f(x,0) <= Lipschitz bound? g_xx mass
  budget per slice?) is the pointer to a boundedness proof.

Note the crude bound J <= int ||f_t(.,t)||_inf * mass(g_xx) dt is NOT
finite a priori — mass concentrating at a moving point is exactly the
loophole — so the question is genuinely open and worth the attempt.

## Order of execution

1. Experiment 3 gate first (small, everything else depends on it).
2. Experiments 1 and 2 (both run at proven budgets, both discriminate
   directly, no escalation treadmill).
3. Experiment 4 only if 1-2 disagree or come back ambiguous.
4. Experiment 5 in parallel, effort-permitting — it is the only experiment
   that can return "proved" rather than "supported".

## Prior expectation (recorded for honesty)

Current lean from Run 9's clean data: dJ tracks w (bounded; log growth was a
discretization transient). Experiments 1-2 are designed to be able to
overturn this, not to confirm it — a flat dJ(w) or ratio-independent decay
would falsify the lean immediately, and per the honesty requirements that
result gets reported just as loudly.

## Conventions carried over

- Every reported J is J_certified; multistart spread reported per point;
  guard arm reported even when it embarrasses the windowed arm.
- New driver code goes into `kink_opt/demos.py` as "Run 10" (and the Run-9
  deeper-base scratchpad script gets ported in as part of this, per
  run9-generation-gain-ladder.md's own note), keeping prior runs intact as
  the experimental record.
- Results persist to `results/run10_*/` via `kink_opt/persist.py`.

## Corrected Experiment 1 results (real run, Run 6 base, 5 seeds) — INCONCLUSIVE

```
       w  nodes        dJ       null_dJ    corrected   guard_dJ
  0.50000     25  +0.00215     +0.05802     -0.05587    +0.08659
  0.25000     21  +0.04277     +0.07333     -0.03057    +0.08108
  0.12500     19  +0.03584     +0.04619     -0.01035    +0.04852
  0.06250     18  +0.03146     +0.02361     +0.00785    +0.09494
  0.03125     19  +0.03683     +0.02051     +0.01632    +0.08239
  0.01562     19  +0.02556     +0.03178     -0.00622    +0.08302
```

`corrected_dJ` is **negative at 3 of 6 points** (including the two widest
windows), and the positive points (`+0.008`, `+0.016`) are the same order of
magnitude as the negative ones. There is no visible trend with `w` at all —
not decaying, not constant, just scattered around zero. Compare against
`null_dJ` itself: it ranges `0.02` to `0.073` across points that should, if
it measured a stable search-noise floor, be roughly constant — it isn't,
it's itself noisy by a factor of ~3.5x.

**Root cause: `generation_step` picks best-of-5-by-max independently for the
windowed and null-control arms, then subtracts.** Best-of-max over a small
sample is a high-variance order statistic (not a mean), and both arms'
maxima are noisy draws from a local-search distribution with standard
deviation comparable to or larger than any real generation-gain signal at
this budget/seed count. Subtracting two independent noisy maxima roughly
adds their variances — exactly the wrong statistic for isolating a small
true effect. **Experiment 1's `corrected_dJ(w)` cannot currently be read
as evidence for EITHER hypothesis** (bounded-J or log-growth) — the honest
statement is "inconclusive: signal-to-noise too low at 5 seeds with a
best-of-max statistic," not "flat, therefore no real gain."

This is the same shape of problem as Run 9's budget artifacts and this
plan's own search-noise finding above — a third, distinct failure mode in
the same measurement pipeline, in a row. Per the project's own precedent
(run9-generation-gain-ladder.md: "Not chasing this further without checking
in" after its third consecutive escalation), this is a checkpoint, not
something to silently push through with a fourth rerun.

## Second, independent confound found: resolution floor at narrow w

Before rerunning with the paired statistic, a second bug was found (by a
targeted investigation, not a run-time symptom): `scale_sweep` regridded
each point with a **flat** `base_fine_sub` (default 4) regardless of `w`,
unlike `generation_ladder`, which scales `fine_sub_k = base_fine_sub *
window0/w_k` precisely so narrow windows don't lose local resolution.
`graded_grid`'s node-count formula, `n_loc = max(2, round(fine_sub *
coarse_N * span / (t1-t0)))`, with a flat `fine_sub=4` and `coarse_N=8`,
gives `round(32*w)` local subintervals — which floors at its `2`-subinterval
minimum (3 nodes spanning the whole window) for every `w <= 0.0625` in the
default sweep. The first run's `n_nodes` plateau at 18-19 for the three
narrowest points looked like the intended "roughly flat problem size" but
was actually this floor engaging, not deliberate density — those three
points could not show real kink development at any window scale, confounded
independent of, and in addition to, the search-noise issue above.

**Fixed**: `scale_sweep` now scales `fine_sub_w = base_fine_sub * max(ws) /
w` per point (same formula as the ladder, anchored at the sweep's own
widest window). Verified directly: per-window node count now holds at 17
across all six `w` in `[0.5, 0.015625]` (previously collapsed to ~3 at the
narrow end). Both fixes (paired statistic + fine_sub scaling) are combined
in the next rerun.

## Doubled-budget rerun (outer=80/pos_iters=200, same 16 seeds) — feasibility
## recovered, result: INCONCLUSIVE, no trend, consistent with zero

```
       w  nodes        dJ    corrected_dJ   +/- se    n    guard_dJ
  0.50000     25  +0.09944       -0.01302  0.00684   10    +0.10413
  0.25000     29  +0.08164       +0.02076  0.01081    9    +0.11452
  0.12500     31  +0.02945       -0.08302      nan    1    +0.09750
  0.06250     32  +0.11562       +0.00213  0.01915    5    +0.14702
  0.03125     33  +0.11930       +0.01400  0.01069    4    +0.15023
  0.01562     33  +0.09800       +0.03539  0.02369    2    +0.11107
```

Doubling the budget recovered usable paired samples at most points (`n` =
2-10, up from 0-7) — the bookend check (`_budget_stable` at the narrowest
window) confirms stability (`delta=+0.00371`, well inside tolerance).
Confidence in the mechanics is now reasonable; `w=0.125` and `w=0.01562`
still have too few paired seeds (1, 2) to trust individually.

**Reading: no point shows `corrected_dJ` clearly separated from zero by
more than ~2 SE, and there is no visible trend with `w`** — the values
bounce between slightly negative (`w=0.5`: `-0.013 ± 0.007`, the only point
with a reasonably tight SE, and it sits *below* zero) and mildly positive
(`w=0.25, 0.0625, 0.03125, 0.01562`, all within 1-1.5 SE of zero). Compare
against `guard_dJ`, which is robustly positive and large (`+0.10` to
`+0.15`) at every single point with no ambiguity — the FREE (unconstrained)
insertion clearly finds real value; the IMPOSED-WINDOW insertion's real
marginal value, once search-noise is subtracted, does not.

**This is genuinely inconclusive, not a null result and not support for
either hypothesis.** It does NOT show `corrected_dJ(w) -> 0` as a
power-law trend (which would cleanly support bounded J) — there's no trend
at all, just noise around zero. It does NOT show a nonzero constant
`corrected_dJ(w)` either (which would support log-growth) — nothing here
is significantly nonzero. The honest reading: at the achieved noise level
(SE ~0.01-0.02, itself larger than most of the point estimates), this
measurement cannot currently distinguish "real per-generation gain is
exactly zero on this base" from "real gain exists but is smaller than we
can resolve at 16 seeds." The one mildly interesting, weakly-powered
observation is that the sign is not consistently positive (unlike every
prior uncorrected measurement in this file and Run 9) — if anything this
leans toward "the windowed-insertion mechanism itself isn't reliably
finding real structure on this base," which is closer to the bounded-J
side than log-growth, but far too weak (given the SEs) to call a finding.

**Five consecutive fixes/escalations were needed to reach even this
inconclusive-but-honest result** (Run 9's budget-starvation class before
this file existed; then in this file: search-noise contamination,
best-of-max being the wrong statistic, the `fine_sub` resolution floor, and
this budget-doubling). Consistent with the project's own precedent of
stopping after repeated escalations, no further automatic reruns were done
after this one — see "What to run next" for the options going forward.

## What to run next (added after the search-noise finding above)

0. **Fix the statistic before spending any more compute on this pipeline.**
   Switch `scale_sweep` (and any future null-corrected `generation_ladder`)
   from "best-of-N-by-max on each arm, then subtract" to a PAIRED per-seed
   comparison: for each seed `s`, compute `paired_dJ_s = windowed_Jc(s) -
   null_Jc(s)` (both already computed per seed in `generation_step`'s
   `spread`-equivalent data, just not currently paired/kept), then report
   `mean(paired_dJ_s)` and its standard error across seeds, not a single
   best-of-max number. This is a real statistical fix, not just "more
   seeds" — best-of-max on two independent noisy quantities is the wrong
   estimator regardless of sample size; paired-per-seed differencing is the
   right one (much of the shared search-path randomness up to the point the
   alive-mask diverges cancels in the subtraction). Increasing seeds (e.g.
   10-20) on top of this is still worth doing to shrink the standard error,
   but do the statistic fix first — it's free and likely matters more than
   sample size here.
1. **Null-correct `generation_ladder`** the same way `scale_sweep` was
   fixed: add a `force_dead=True` guard-style third arm per generation
   (reusing `generation_step`'s new parameter), report `null_dJk` and
   `corrected_dJk = dJk - null_dJk`, and add a `corrected_decay_ratios`
   helper alongside `decay_ratios`. Until this exists, Experiment 2's
   `window_ratio` numbers (both bases, both ratios) are suggestive only.
2. **Re-read Experiment 1's `corrected_dJ(w)`** (not raw `dJ(w)`) once the
   null-corrected sweep is in hand — this is the actual answer to "does a
   single generation's real gain vanish at small scale", the raw numbers
   from the first run are contaminated and should not be used to draw a
   conclusion on their own.
3. Once both discriminators are null-corrected, re-apply this file's
   Experiment 4/5 gating logic (only escalate if corrected results disagree
   or stay ambiguous).
4. Consider whether Run 9's original `dJk` sequences (both bases,
   `window_ratio=0.5`) are worth a retroactive null-correction pass too, or
   whether the mostly-nonzero `jump_mean` diagnostics already recorded there
   are enough to bound the contamination as small — this hasn't been
   checked, only assumed, and should be either confirmed or corrected before
   citing Run 9's numbers as clean going forward.
