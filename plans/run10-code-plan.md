# Code plan: Run 10 — scale-sweep discriminators

Companion to `plans/run10-scale-sweep-discriminators.md` (strategy). This one
is implementation-level: exact functions, signatures, file locations, data
flow. Mirrors `plans/run9-code-plan.md`'s format.

## Status: Items 1-4 and 7-8 IMPLEMENTED and RUN (`kink_opt/topology.py`,
## `kink_opt/persist.py`, `kink_opt/demos.py`). Item 6 (`kink_opt/construct.py`,
## Experiment 5) NOT implemented -- exploratory math, deferred. FOUR
## significant deviations from the plan below, all found and fixed after
## the first real run, not anticipated in the original design: (1) a new
## `force_dead=True` null-control arm added to `generation_step`, (2) a
## `_paired_dJ` per-seed statistic replacing best-of-max subtraction, (3)
## `scale_sweep`'s `fine_sub` scaled per point instead of held flat, (4) a
## doubled outer/pos_iters budget after the first two fixes still left most
## sweep points' paired sample size too small (Run 9's budget didn't port to
## this setup). Final Experiment 1 result: INCONCLUSIVE (no significant
## trend, everything within ~1-2 SE of zero) -- see
## `run10-scale-sweep-discriminators.md`'s findings sections for the full
## story of each. Details below, per piece.

### Deviation: `generation_step(..., force_dead=False)` (not in original plan)

**Location:** `kink_opt/topology.py`, in `generation_step` itself (extra
parameter, not a new function). When `True`, both newly-inserted columns'
alive masks are forced all-`False` immediately after `add_kink`, so the LP
can never give them weight regardless of `window`. Used by `scale_sweep` as
a third ("null-control") arm alongside windowed/guard. Exists because a
real sweep point showed both new kinks at `jump_mean=0` (dead at
convergence) yet nonzero `dJ` -- a direct control run (`force_dead=True`
through the identical multi-seed search) reproduced gains of the same order
as the real windowed arm, confirming the insertion-jitter multistart alone
(independent of any new-kink value) can move the OLD kinks to a better
local optimum. `scale_sweep` reports `null_dJ` and
`corrected_dJ = dJ - null_dJ` per point; `corrected_dJ(w)`, not raw `dJ(w)`,
is what actually answers Experiment 1. `generation_ladder` does **not** yet
have an equivalent null-control arm -- Experiment 2's numbers are
unconnected to this correction and should be read as suggestive only until
that's added (tracked in the strategy plan's "What to run next").

### Deviation: `_paired_dJ` (not in original plan) — best-of-max was the wrong statistic

**Location:** `kink_opt/topology.py`, right before `scale_sweep`. A first
version of `scale_sweep` computed `corrected_dJ = (windowed["Jc"] -
regrid_Jc) - (null["Jc"] - regrid_Jc)`, i.e. subtracted two independent
best-of-5-seeds maxima. The real run's numbers had no discernible trend in
`w` and were negative at 3 of 6 points -- diagnosed as best-of-max being a
high-variance order statistic, unsuited to differencing two noisy arms:
subtracting two independent maxima adds their variances, and a real
per-generation gain (if any) was swamped by that combined variance at 5
seeds. `_paired_dJ(spread_a, spread_b)` fixes this by matching arms
per-seed (both `generation_step` calls use identical `np.random.default_rng
(s)` draws up to the point their alive masks diverge, so most shared search
randomness cancels in the per-seed subtraction) and reporting `mean`/`se`/`n`
over the seeds feasible in both arms, not a single order statistic.
`scale_sweep`'s seed count was also bumped from 5 to 16 (in the `demos.py`
Run 10 driver call, not the function default) to give the paired mean a
usable standard error.

### Deviation: `scale_sweep`'s `fine_sub` scaled per point (not flat)

**Location:** `kink_opt/topology.py`, inside `scale_sweep`'s loop. The
original plan (and first implementation) called `_regrid_onto_windows(base,
[window], [base_fine_sub], ...)` with a flat `base_fine_sub` for every `w`.
`graded_grid`'s node-count formula floors at 2 local subintervals per
window; with a flat `fine_sub=4` and `coarse_N=8` this floor engaged for
every `w <= 0.0625` in the default sweep (3 nodes spanning the whole
window) -- an independent under-resolution confound, found by checking
`graded_grid`'s formula directly against the sweep's `ws`, not by any
runtime symptom. Fixed by scaling `fine_sub_w = base_fine_sub * max(ws) / w`
per point (mirrors `generation_ladder`'s own `base_fine_sub * window0/w_k`,
anchored here at the sweep's widest window instead of a separate
`window0`). Verified: per-window node count now holds flat at 17 across all
six `w` in `[0.5, 0.015625]` (previously collapsed to ~3 at the narrow
end).

## New code, in order

### 1. Refactor: extract `_regrid_onto_windows` out of `generation_ladder`

**Location:** `kink_opt/topology.py`, currently inlined in
`generation_ladder` (the block building `grid`/`t_new`, migrating
`A,XI,B,ETA`, re-solving the weight LPs, and the 1%-relative-gap
`RuntimeError` check — see `kink_opt/topology.py:276-307`).

```
_regrid_onto_windows(cur, windows, fine_subs, coarse_N=8, sub=8, tol_rel=0.01)
```

Behavior-preserving extraction: same union-with-`cur["t"]` fix (never
coarsen below the base grid), same weight-LP repair, same `certify`-based
1%-bar guard. Returns `(cur_regridded, regrid_Jc)`. `generation_ladder`
calls this instead of its inline block — pure refactor, verify Run 9's
recorded numbers reproduce byte-for-byte before touching anything else.
This is the prerequisite for Experiment 1, which needs the *same* per-window
regrid machinery but applied once per sweep point instead of once per
ladder.

### 2. `scale_sweep` — Experiment 1 driver

**Location:** `kink_opt/topology.py`, after `generation_ladder`.

```
scale_sweep(base, ws, seeds=range(5), dx=0.05, base_fine_sub=4, coarse_N=8,
           outer=40, pos_iters=100, sub=8, verbose=True)
```

For each `w` in `ws` (fully independent — no accumulation across points,
unlike the ladder):

1. `window = (t1 - w, t1)`.
2. `cur, regrid_Jc = _regrid_onto_windows(prune(base), [window],
   [base_fine_sub], coarse_N, sub=sub)` — fresh regrid from `base` every
   time, not carried forward from the previous point, so problem size stays
   in the same class at every `w` and one verified budget covers the whole
   sweep.
3. `windowed = generation_step(cur, window, seeds=seeds, dx=dx, outer=outer,
   pos_iters=pos_iters, sub=sub)`.
4. `guard = generation_step(cur, (t0, t1), seeds=seeds, dx=dx, outer=outer,
   pos_iters=pos_iters, sub=sub)`.
5. `null = generation_step(cur, window, seeds=seeds, dx=dx, outer=outer,
   pos_iters=pos_iters, sub=sub, force_dead=True)` — the null-control arm
   (deviation from the original plan; see above).
6. `dJ = windowed["Jc"] - regrid_Jc`, `guard_dJ = guard["Jc"] - regrid_Jc`,
   `null_dJ = null["Jc"] - regrid_Jc`, `corrected_dJ = dJ - null_dJ`.

Collects `dict(w=w, n_nodes=len(cur["t"]), dJ=dJ, feasible=windowed["feasible"],
guard_dJ=guard_dJ, guard_feasible=guard["feasible"], null_dJ=null_dJ,
null_feasible=null["feasible"], corrected_dJ=corrected_dJ,
diagnostics=windowed["diagnostics"], spread=windowed["spread"],
sol=windowed["sol"])` per point.

Returns `dict(points=[...], base_Jc=<certify(prune(base))["Jc"]>)`.

Semilog-plot-ready: `[(p["w"], p["dJ"]) for p in points]`.

### 3. Budget-adequacy gate — Experiment 3

**Location:** `kink_opt/topology.py`, near `generation_step`.

```
_budget_stable(step_fn, *args, factor=2.0, tol_rel=0.10, tol_abs=0.005, **kwargs)
```

Calls `step_fn(*args, **kwargs)` at the given `outer`/`pos_iters`, then again
at `outer*factor`/`pos_iters*factor` (both keys must be present in
`kwargs`); accepts if `|Jc_hi - Jc_lo| < max(tol_abs, tol_rel * |Jc_lo|)`.
Returns `dict(accepted, result=<the higher-budget result, always returned
so a rejected point can still be inspected>, delta=Jc_hi - Jc_lo)`. Generic
over `generation_step` and `generation_ladder`'s per-generation calls (both
take `outer=`/`pos_iters=` kwargs already).

```
_gate_report(point)
```

Cheap companion check (no rerun) using fields `generation_step` already
returns: `feasible_frac = mean(f for _, _, f in point["spread"]) >= 0.5`
and both `point["diagnostics"]["f"]["jump_mean"] > 0` and `[...]["g"][...] > 0`.
Combine as `dict(feasible_ok, jump_ok, ready_for_budget_check)` — cheap
filter to decide which points are even worth the expensive `_budget_stable`
rerun (doubling every sweep point is wasteful; gate the obviously-fine ones
by the cheap check, escalate only borderline ones to the doubled rerun).

Not applied blanket to every point in Experiment 1/2 by default — driver
code (`demos.py` Run 10 block) calls `_gate_report` on all points, and
`_budget_stable` only on points that fail it or sit at the extremes (widest
window, narrowest window) as a sanity bookend.

### 4. window_ratio discriminator — Experiment 2

No new machinery. Reuses `generation_ladder` unchanged, called with
`window_ratio=0.7` and `window_ratio=0.3`, `n_gen=3`,
`outer=40, pos_iters=100, seeds=range(5)` (Run 9's already-proven budget at
this grid size) on both Run 3's G0 and Run 6's grown topology as `base`.

One small analysis helper, since eyeballing ratios by hand is how Run 9's
noise got missed initially:

```
decay_ratios(generations)   # kink_opt/topology.py, near generation_ladder
```

`-> [generations[i+1]["dJk"] / generations[i]["dJk"] for i in
range(len(generations)-1)]`. Driver prints this list next to the
`window_ratio` it was run at, for direct comparison (ratio list should
cluster near `window_ratio` under the bounded hypothesis).

### 5. Falsifiable extrapolation — Experiment 4 (conditional)

**Location:** `kink_opt/topology.py`.

```
fit_geometric(dJk)   # dJk: list of floats, len >= 2
```

Least-squares fit of `log(dJk[k])` vs `k` (guard against `dJk <= 0` by
raising `ValueError` — a non-positive dJk means the hypothesis test is
already moot, don't silently fit garbage). Returns `(r, dJ1_fit)` for
`dJk_fit(k) = dJ1_fit * r**(k-1)`.

`generation_ladder` gains one optional argument to support per-generation
budget scaling, backward-compatible (`None` reproduces current flat
behavior exactly):

```
generation_ladder(..., budget_fn=None)
```

If given, `outer_k, pos_iters_k = budget_fn(k, n_live_nodes)` overrides the
flat `outer`/`pos_iters` for generation `k`'s `generation_step` calls (both
windowed and guard arms); `n_live_nodes` is `cur["XI"].shape[1] +
cur["ETA"].shape[1]` at that point in the loop. Default caller in the Run 10
driver: `budget_fn = lambda k, n: (int(outer0 * (1 + 0.3*n)), int(pos0 *
(1 + 0.3*n)))`, per `run9-generation-gain-ladder.md`'s "What to run next"
item 1.

Driver flow: `fit_geometric` on the existing n_gen=3 dJk (both bases) ->
print predicted `dJk[4]` -> extend a *loaded* ladder
(`kink_opt.persist.load_ladder`, whose docstring already documents this use
case) by one more `generation_step` call at the escalated budget, gated by
`_budget_stable` -> compare actual vs predicted.

### 6. Constructive arm — Experiment 5

**New module:** `kink_opt/construct.py`.

```
hierarchical_construction(n_gen, w0=0.5, ratio=0.5, amp0=1.0, amp_ratio=None,
                          x_center=0.0, t=None)
```

Hand-built, zero-optimizer solution: `n_gen` nested kink pairs (one f, one g
each), generation `k` alive on `window_k = (1 - w0*ratio**(k-1), 1)`
(same anchoring as the ladder), at a fixed or slowly-varying `x_center`
(construction, not search — position trajectories are explicit functions of
`t`, not NLP output), weight amplitude `amp0 * amp_ratio**(k-1)` (default
`amp_ratio=None` means "solve the two amplitude free parameters per
generation by imposing the harvest-sum stationarity condition analytically"
— needs the module docstring's harvest-sum formula worked out on paper
first; if that's not tractable, fall back to `amp_ratio` as an explicit
sweep parameter and report `Jc(amp_ratio)` instead of assuming a single
right answer). Must satisfy `f_t>=0`, `g_t<=0`, boundary conditions
*by construction*, not by solver convergence — this is what makes the arm
immune to the budget artifacts that hit Experiments 1 and 4.

Returns a `sol` dict shape-compatible with the rest of the package (`A, XI,
B, ETA, t, alive_f, alive_g, J`) so it plugs straight into `certify()`.

```
construction_ladder(n_gens=8, **kwargs)
```

Truncate the construction at `1..n_gens` generations, `certify()` each
truncation, return `[(k, Jc_k)]`. Since there's no NLP, `n_gens` can go
well past the optimizer ladder's proven ceiling of 3 — this arm is the one
place in the whole plan where pushing deep is cheap.

This module is genuinely exploratory (the amplitude-selection rule needs
math worked out, not just code) — treat the signature above as a starting
point, not a contract; expect it to change once the first construction
attempt shows what actually keeps `f_t>=0`/`g_t<=0` satisfied.

### 7. `persist.py` — sweep save/load

**Location:** `kink_opt/persist.py`, mirroring `save_ladder`/`load_ladder`.

```
save_sweep(tag, sweep, meta=None, root="results", nx=401)
```

`results/<tag>/w{i}/sol.npz, fields.png` per point (same per-point layout as
`save_ladder`'s `gen{k}/`), `results/<tag>/sweep.json` holding
`{base_Jc, points (sol stripped), meta}`.

```
load_sweep(tag, root="results")
```

Symmetric reload, `sol` reattached per point from `w{i}/sol.npz`.

### 8. `demos.py` Run 10 block

**Location:** `kink_opt/demos.py`, after Run 9.

Narrated header pointing at `run9-generation-gain-ladder.md`'s open
question and `run10-scale-sweep-discriminators.md`'s experiment list.
Sequence:

1. `scale_sweep` on Run 6's base, `ws = [0.5, 0.25, 0.125, 0.0625, 0.03125,
   0.015625]`, `seeds=range(5), outer=40, pos_iters=100` -> table `(w, dJ,
   ok, guard_dJ, ok)`, `_gate_report` per point, save via `save_sweep`.
2. Two `generation_ladder` calls per base (`window_ratio` 0.7 and 0.3,
   `n_gen=3`, same budget as Run 9's hi-budget reruns) -> `decay_ratios`
   printed against the driving `window_ratio`.
3. If the sweep + ratio results agree (both point the same direction),
   print the reading and stop — skip Experiment 4/5 escalation, per the
   strategy plan's "only if ambiguous" gating.
4. Otherwise: `fit_geometric` + escalated-budget gen4 (Experiment 4).
5. `construction_ladder` run independently of 1-4 (no shared state,
   effort-permitting per the strategy plan) -> printed `(k, Jc_k)` table
   compared visually against the optimizer-based `dJ(w)`/`dJk` curves.

## Verification to perform before trusting results

- `_regrid_onto_windows` refactor: rerun Run 9's exact recorded ladder call
  and confirm byte-identical `dJk`/`guard_dJk` to the numbers already in
  `run9-generation-gain-ladder.md`.
- `scale_sweep`: smoke test on a toy 2+2 base at 2-3 `w` values, confirm
  `n_nodes` stays roughly flat across points (the thing Experiment 1 is
  designed to guarantee) and `dJ`/`guard_dJ` are computed against each
  point's own `regrid_Jc`, not a shared one.
- `_budget_stable`: run once on a *known*-starved case from Run 9's history
  (the n_gen=5, outer=40 run — `jump_mean=0.000`, `dJk` collapsed to ~0) and
  confirm it flags `accepted=False`; run once on a known-clean case (n_gen=3
  hi-budget) and confirm `accepted=True`. If either check fails, the
  tolerance constants need adjusting before the gate can be trusted on new
  data.
- `fit_geometric`: unit-check against a hand-built exact geometric sequence
  (recovers `r` and `dJ1_fit` to float precision) and confirm it raises on
  a sequence containing a non-positive value.
- `hierarchical_construction`: verify `f_t>=0`/`g_t<=0`/boundary conditions
  hold by construction at `n_gen=1` (a single generation should reduce to
  something checkable by hand) before trusting `certify()`'s output at
  higher `n_gen`.
- No changes to `total_J`, `grad_total_J`, `penalty`, `grad_penalty`,
  `_step_diff_grad`, or `optimize_positions` — everything here is
  measurement/construction code around the existing solver, same scope
  discipline as Run 9.
