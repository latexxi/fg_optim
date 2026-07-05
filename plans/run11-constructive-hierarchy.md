# Plan: Run 11 — Constructive self-similar hierarchy (the only PROVE-side arm)

## Status: IMPLEMENTED and RUN (`kink_opt/construct.py`). Result: BOUNDED J
under this construction (dJk decays faster than geometric, collapses across
every (scale_t, scale_x) tested), subject to the shared-anchor caveat (every
generation collapses toward the same point (p_end, t1) by construction — see
STRATEGY.md Section 5 status update for the full numbers and the caveat).
All validation checks (insertion-neutral, grid-convergence, travel-sanity)
PASS. Wired into `kink_opt/__init__.py`, `persist.py`
(`save_construct`/`load_construct`), and `demos.py` (Run 11).

Continuation of `run10-scale-sweep-discriminators.md` (Experiment 5). Run 10
exhausted the optimizer route: 5 consecutive methodological fixes
(Run 9 budget-starvation; then search-noise contamination, best-of-max wrong
statistic, `fine_sub` resolution floor, budget-didn't-port) all landed
INCONCLUSIVE, and the route is dead-ended *structurally*, not at budget — a
local-search optimizer can only ever FAIL to find gain; it can never certify
gain absent, and never prove blowup. This run takes the one path that can
return "proved" rather than "supported": build an explicit self-similar
hierarchy with analytic kink trajectories, zero nonconvex optimization, and
certify it directly.

## The question (unchanged, STRATEGY.md Section 5)

Does certified gain from adding generation k at lifetime-window scale w_k stay
`>= c > 0` as k -> infinity (log law, sup J = +inf), or vanish (J bounded)?
Crude bound `J <= int ||f_t(.,t)||_inf * mass(g_xx) dt` is NOT finite a priori
— mass concentrating at a moving point is the loophole — so genuinely open.

## Why constructive beats the optimizer here

- **No budget artifact possible by construction.** Positions are set
  analytically; the only numerical solve is the *convex* weight LP inside
  `certify()` (`lp_weights_f`/`lp_weights_g`, HiGHS, exact global optimum).
  The entire failure class that ate Runs 9-10 (nonconvex position NLP starved
  by low `outer`/`pos_iters`/seeds) cannot occur — there is no position NLP.
- **Decisive both ways:**
  - A family with per-generation certified gain `>= c > 0` establishes the log
    law numerically (and the construction becomes a proof sketch).
  - If every attempted construction *saturates*, the saturation MECHANISM
    (per-x rise budget `int f_t dt = -f(x,0) <= Lip`? g_xx mass budget per
    slice?) is the pointer to a boundedness proof.
- **Run 10's one clean signal motivates it:** `guard_dJ` (free, full-lifetime
  insertion) was robustly positive everywhere while the corrected
  windowed-insertion signal was not — i.e. the value is real but the
  *imposed-window optimizer mechanism* failed to express it. A hand-built
  construction sidesteps that mechanism entirely.

## Key simplification (do NOT hand-set weights)

`spawn_generation` and Experiment 5's original note both imagined
hand-building "weights per generation". Unnecessary and weaker. `certify()`
already REPAIRS weights by re-solving the convex weight LPs with positions
frozen and lifetime masks respected (`verify.py:167-168`). So the construction
need only specify, per generation:

- kink POSITIONS `XI`/`ETA` (analytic trajectories), and
- lifetime MASKS `alive_f`/`alive_g` (the window per generation),

then let `certify()` pick the exact optimal feasible weights for those
positions. This is STRONGER than hand-set weights: it reports the best `Jc`
achievable at the constructed geometry, so a saturation result can't be blamed
on a bad hand-chosen weighting. Deterministic, exact, no seeds.

## New file: `kink_opt/construct.py` (reserved in CLAUDE.md, not yet built)

### `build_hierarchy(base_family, n_gen, scale_t=0.5, scale_x=0.5, t=None) -> sol`

Build a stacked self-similar solution dict (same schema as `run()`'s output:
keys `A, XI, B, ETA, t, alive_f, alive_g, J`, no `rng_seed`).

Construction (analytic analogue of `spawn_generation`, but STACKED and
jitter-free):

1. **Generation 0 carrier** = a single travelling f-kink + g-kink pair with
   real travel (Run 8 lesson: a *static* base makes each contracted copy
   co-located with its parent, and two hats at one point are redundant in a
   convex-hat sum, so the LP zeroes the copy — the Run 8 null). Reuse `run()`'s
   `seed="travel"` path `p(t) = -0.5 + 1.0*t` for gen 0, full lifetime.
2. **Generation k (k>=1)** = affine-contracted copy of gen k-1's trajectory:
   - spatial: `col_k = p_end + scale_x**k * (p - p_end)`, contract about the
     travel end `p_end` (mirror `spawn_generation` topology.py:141-142).
   - temporal: lifetime window of length `w_k = scale_t**k * (td-tb)` anchored
     at the travel end `td` (mirror topology.py:137-140). Encode as
     `alive_*` mask columns; dead outside `[td - w_k, td]`.
   - insert one f-column and one g-column per generation via the existing
     `_insert_column` (topology.py:29) — same machinery, J-neutral at
     insertion, so masks/ordering stay consistent with the rest of the code.
3. Seed `A`/`B` to any feasible ramp (`certify()` overwrites them via LP);
   set `sol["J"] = total_J(...)` on the coarse grid for the schema.

Parameters `scale_t`, `scale_x` are the self-similarity ratios under test —
sweeping them is the experiment.

### Grid: the construction must be certified on a grid that resolves w_n

The narrowest generation has lifetime `w_n = scale_t**n`. `certify()`'s
`refine_time(sub=8)` only subdivides the grid it's handed; it does not add
nodes where the base grid is coarse. So `build_hierarchy` must place `t` on a
graded grid resolving every window, exactly like the ladder/sweep do —
reuse `graded_grid(windows, coarse_N, fine_sub)` (Task C) or
`_regrid_onto_windows` (topology.py:249), sizing `fine_sub_k` per window with
the same `base_fine_sub * w0/w_k` scaling Run 10 fix #3 established
(`run10-...md` "resolution floor"). Windows = the n generation lifetime
intervals, all anchored at `td`.

### `constructive_ladder(n_gen, scale_t, scale_x, ...) -> list[dict]`

Driver. For `k = 0..n_gen`:

1. `sol_k = build_hierarchy(..., n_gen=k, ...)`.
2. `c_k = certify(sol_k)` — read `Jc` and `rep["ALL CONSTRAINTS OK"]`.
3. `dJk = Jc_k - Jc_{k-1}`.

Report `Jc_k`, `dJk`, `constraints_ok` per generation, and the ratio
`dJk / dJ_{k-1}` (constructive analogue of `decay_ratios`). No seeds, no
paired statistic, no null arm — none needed, the measurement is deterministic
and artifact-free. THIS is what Run 10's whole apparatus was trying and
failing to approximate.

Read:
- `dJk -> nonzero constant` (ratio -> 1) as k grows: log law, sup J = +inf.
- `dJk` decaying geometrically (ratio ~ `scale_x` or `scale_t`, consistent
  across k): J bounded; the sum `sum_k dJk` converges — report the limit.

### Sweep the self-similarity ratios

The single most informative output: run `constructive_ladder` across a small
grid of `(scale_t, scale_x)` and report the decay ratio of `dJk` vs those
inputs. If `dJk` ratio tracks `scale_x` (or `scale_t`), that's the clean
`window_ratio`-discriminator signal Experiment 2 could never get from the
optimizer (collapsed to `n=0-1` paired seeds). Ratio-independence of `dJk`
would instead support scale-free / log growth.

## Saturation-mechanism instrumentation (the boundedness-proof pointer)

If `dJk` decays, before concluding "bounded" record WHY, per generation, so
the result points at a proof rather than just asserting a number:

- per-x rise budget: `int_0^1 f_t(x,t) dt = -f(x,0)` at each generation's kink
  x — is the added generation running into the Lipschitz cap on `f`'s rise?
- g_xx mass per slice: total `sum_m b_m(t)` the generation contributes — is the
  convex-hat sum's curvature budget saturating?

These are cheap reads off the certified fields (`viz.field_grid` /
`verify_dense` already rebuild `f`/`g` densely). A clean "generation k+1 can't
add rise because generation k already used the `-f(x,0) <= Lip` budget at that
x" is the boundedness proof sketch.

## Falsifiable prediction (record BEFORE running, per honesty rule)

Following Run 10's `prior expectation` (lean: bounded, dJ tracks w) and its one
clean signal (free insertion finds real value): **predict `dJk` decays with a
ratio approximately equal to `scale_x`** (spatial contraction is the binding
one, since g_xx mass at the harvest point scales with hat width ~ `scale_x`).
Fit a geometric to `dJ_1..dJ_3`, WRITE the predicted `dJ_4` into this file,
then compute gen 4 once. Landing on prediction confirms bounded; a `dJk` floor
that refuses to decay (ratio -> 1) falsifies it toward log growth. Either way
report loudly.

## Validation before trusting any number

1. **J-neutral insertion check:** inserting a zero-weight generation must not
   change `Jc` (same invariant `add_kink`/`spawn_generation` rely on) — assert
   `certify(build_hierarchy(n_gen=k))` with the new columns forced dead equals
   `certify(n_gen=k-1)` within LP roundoff. Reuses the `force_dead` idea from
   Run 10 as a sanity gate, not a statistical arm.
2. **Grid convergence:** re-certify the deepest ladder at `sub=16` (double the
   `refine_time` subdivision); `Jc` must move `< 1%` or the graded grid under-
   resolves `w_n` — mirrors `generation_ladder`'s 1% regrid-gap guard
   (`_regrid_onto_windows`).
3. **Travel sanity:** confirm gen-0 carrier actually travels (`p_end != p[0]`),
   else every copy is co-located and the whole ladder reproduces the Run 8
   null for a trivial reason.

## Driver: "Run 11" in `kink_opt/demos.py`

Follow the numbered-demo convention (CLAUDE.md `demos.py` section): add
`Run 11` narrating "optimizer route (Runs 9-10) hit its structural ceiling;
this is the constructive arm — deterministic, artifact-free, the only one that
can PROVE either side." Print the `constructive_ladder` table (`Jc`, `dJk`,
ratio, `constraints_ok`) for the `(scale_t, scale_x)` sweep, plus the
saturation-budget diagnostics. Persist to `results/run11_*/` via
`kink_opt/persist.py`.

## Order of work

1. `build_hierarchy` + J-neutral insertion check (validation #1) — smallest
   thing that can be wrong; get it green first.
2. Grid sizing (`graded_grid`/`_regrid_onto_windows` for the n windows) +
   grid-convergence check (validation #2).
3. `constructive_ladder`, single `(scale_t, scale_x)`, `n_gen=3`. Read `dJk`.
4. Sweep `(scale_t, scale_x)`; read ratio-vs-input.
5. Saturation instrumentation (only if `dJk` decays).
6. Gen-4 falsifiable prediction (only if 1-4 leave it ambiguous).
7. Run 11 driver + persist.

## Conventions carried over

- Every reported J is `J_certified` (`certify()`'s `Jc`), constraints_ok read
  from `rep["ALL CONSTRAINTS OK"]`.
- Prior runs stay intact as the experimental record; add Run 11, don't replace.
- No nonconvex optimization anywhere in this arm — that's the whole point.
