# Briefing: Hierarchical Kink Optimization for a Coupled PDE Functional

This document is self-contained. You are given one code package, `kink_opt/`
(a working prototype), and this briefing. No other context is needed.

---

## 1. The optimization problem

Maximize over two functions f, g : [-1,1] x [0,1] -> R

    J[f,g] = int_0^1 int_{-1}^{1} f_t(x,t) * g_xx(x,t) dx dt

subject to (a.e., for all t):

    f: convex in x (f_xx >= 0), increasing in t (f_t >= 0),
       Lipschitz |f_x| <= 1, f(+-1, t) = 0, f(x, 1) = 0.
    g: convex in x (g_xx >= 0), decreasing in t (g_t <= 0),
       Lipschitz |g_x| <= 1, g(+-1, t) = 0, g(x, 0) = 0.

Both functions are trapped between the "tent floor" -(1-|x|) and 0.
f starts somewhere below 0 and must rise to 0 by t=1; g starts at 0 and
deepens over time.

**Known reference values.** Static "tent" solutions give exactly J = 2.
Mesh-based numerical optimization (grids up to 65 x 257) reaches J ~ 2.4-3.05,
and — the key empirical fact — J grows like ln(Nx) as the spatial mesh is
refined (measured over 3 doublings of Nx). The goal of this project is to
study that growth without a mesh.

---

## 2. Kink coordinates (what the prototype implements)

Because the constraints saturate |f_x| = 1 almost everywhere at optimum,
solutions are essentially piecewise linear in x. So we parameterize directly
by kinks instead of grid values:

    f(x,t) = - sum_i a_i(t) * hat(x; xi_i(t)),   a_i >= 0
    g(x,t) = - sum_m b_m(t) * hat(x; eta_m(t)),  b_m >= 0

where hat(x; c) is the tent basis function with hat(+-1) = 0, hat(c) = 1.
In these coordinates all constraints become simple and EXACT (see the
docstring of kink_opt/__init__.py): convexity = nonnegative weights, Lipschitz = two
linear inequalities per time node, monotonicity = piecewise-linear
differences checked at the union of kink positions (provably sufficient).

The curvature of g is a sum of point masses ("atoms"):

    g_xx(., t) = sum_m j_m(t) * delta(x - eta_m(t)),
    j_m = 2 b_m / (1 - eta_m^2)     ("jump" of the kink)

so the objective collapses to the **harvest sum**:

    J = sum_m  int_0^1  j_m(t) * f_t(eta_m(t), t)  dt

**Read this formula as physics: J is the total rise of f, harvested exactly
at the locations of g's kinks, weighted by the jump sizes.** Everything
below follows from this reading.

Time is discretized into N steps; decision variables are the weight and
position trajectories at the time nodes. The solver in kink_opt/solver.py
alternates:

  1. LP over f-weights a (positions frozen) — globally optimal, HiGHS
  2. LP over g-weights b — globally optimal
  3. L-BFGS-B over all kink positions (nonconvex block, penalty-enforced
     feasibility), then re-run 1-2 to restore exact feasibility.

A `report()` function certifies results: it interpolates the solution to an
8x finer time grid, re-solves the weight LPs there (repairing any
between-node constraint cheating by the position optimizer), and verifies
every constraint on a dense grid. **Only J_certified counts.**

Baseline results you should be able to reproduce (regression targets):

    Run 1: 1 f-kink + 1 g-kink, positions frozen  -> J_certified ~ 1.916
           (= 2 - O(1/N); the static tent optimum)
    Run 2: same, positions free                   -> J_certified ~ 2.219
    Run 3: 3 f-kinks + 2 g-kinks, positions free  -> J_certified ~ 2.297

> **Status (current `kink_opt/` package):** Runs 1-3 reproduce at J_certified =
> 1.916 / 2.220 / 2.309 respectively (Run 3 ticked up from 2.297 to 2.309 as
> a side effect of an unrelated bugfix — `run()`'s position step was not
> monotone in J and could silently drift below its own best point; it now
> keeps the best feasible state and reverts on regression via a `patience`
> parameter). Two further, ad hoc runs were added beyond this briefing's
> scope: Run 4 (multistart over `rng_seed`, Kf=5/Kg=4, J_certified ~ 2.420)
> and Run 5 (Kf=6/Kg=5, J_certified ~ 2.544). These are brute-force
> "more kinks + more restarts" experiments, not the hierarchical
> generation-spawning protocol in Section 5 below — they don't exercise
> Tasks B/C/D and shouldn't be mistaken for that experiment having been run.

---

## 3. The mechanism you are exploiting (why travel beats 2)

**Static schedules cap at 2.** If g's kinks never move, every unit of f-rise
at a kink is harvested once, the accounting telescopes, and J <= 2 exactly.
This is a theorem-grade observation, and Run 1 confirms it numerically.

**Traveling kinks break the cap.** Suppose a g-kink with jump j moves along
a path of length L, while f rises in a narrow co-moving front of width w and
per-point height h just at the kink's current location. One pass earns

    Delta J  ~  j * h * (L / w).

The factor L/w is an amplification: the same kink harvests fresh f-rise at
every point it visits. The costs are only local budgets: f can spend total
rise <= 1-|x| at each point over its whole lifetime, and g spends descent
~ j*w at each swept point. Run 2 (J = 2.22 > 2 with ONE kink pair) is the
minimal demonstration of this mechanism.

**Why doesn't J then grow linearly in 1/w?** Naive accounting with M passes
of ever-narrower fronts gives J ~ M, i.e., J ~ Nx on a mesh. That is NOT
observed; the mesh data shows J ~ ln(Nx). The proposed explanation — the
central hypothesis this project tests — is a **hierarchical maintenance
cost**:

  A co-moving rising front of f is itself a traveling f-kink. Sustaining it
  consumes f's local convex structure, which must be re-armed. Re-arming a
  kink at scale w taxes the structure at scale 2w, which taxes scale 4w, and
  so on: a recursion with log2(1/w) levels. Each dyadic scale (each
  "generation" of finer, faster, shorter-lived kinks) can therefore
  contribute only a roughly CONSTANT increment to J, and the total over
  log-many generations is

      J ~ c * ln(1/w_min) ~ c * ln(Nx)   on a mesh with w_min ~ 1/Nx.

  Consistent with this, the mesh solutions show "snaking inside snaking":
  the bright filament of f_t * g_xx (the kink trajectory) exhibits
  self-similar sub-structure at each refinement level — the fine-generation
  kink path rides on the coarser generation's path.

**The decisive advantage of kink coordinates:** a mesh cannot distinguish
"J grows forever like ln(Nx)" from "J saturates" because the mesh itself
floors the front width at Delta x. Kink positions are continuous; w -> 0 is
representable. In kink coordinates the question becomes directly measurable:
**add one generation of kinks at a time and measure the gain per
generation.** Constant gain per generation = the log law is real (and
sup J = +infinity); decaying gain = J is bounded. This is a renormalization-
group style computation: instead of resolving 8 levels at once on a giant
grid, compute one level, measure what it passes to the next, iterate.

---

## 4. Your tasks (extension points, in priority order)

The prototype's `# EXT` comments mark where these plug in.

### Task A — Analytic gradients for the position block (enabler, do first)  [DONE]

The position NLP currently uses 2-point finite differences over all
position variables — the runtime bottleneck and a precision limit. But J is
closed-form in positions: hat(x; c) is piecewise linear in both arguments,
so dJ/d(xi_i^k) and dJ/d(eta_m^k) are explicit (careful: hat is
nondifferentiable where an evaluation point crosses a node; use a
subgradient or smooth the tip over epsilon ~ 1e-6). Supply `jac=` to
L-BFGS-B, likewise for the penalty terms.
Acceptance: gradient check against finite differences at random feasible
points (away from crossings) to 1e-6 relative; wall time for Run 3 down by
>= 10x; J_certified for Runs 1-3 not worse than baselines.

> **Status: implemented.** `grad_total_J` gives the closed-form objective
> gradient; `grad_penalty` (via the `_step_diff_grad` helper) gives the
> penalty gradient, including the ordering/Lipschitz terms and the
> self-referential monotonicity terms (kink positions are simultaneously
> the hat-function node *and* the checkpoint the constraint is evaluated
> at — `_step_diff_grad`'s docstring explains the node-role/eval-role
> split this requires). Both are wired into `optimize_positions` via
> `jac=`; there is no finite-difference fallback path.
> Acceptance results: gradients match finite differences to ~1e-9 absolute
> error across randomized `(Np1, Kf, Kg)` shapes (tighter than the ~1e-6
> relative target); full-script wall time dropped from minutes to ~14s for
> Runs 1-3 (well over the >=10x target — closer to 20-100x on the position
> solves themselves); J_certified for Runs 1-3 is at or above baseline (see
> the Status note under Section 2). The derivation is error-prone — it
> caught two real sign bugs during development, both found by finite-
> difference checks rather than by inspection — so re-verify with
> finite differences before trusting any future change to these functions.

### Task B — Topology moves: add/prune kinks between outer iterations  [DONE]

The alternation can only optimize a FIXED number of kinks; the hierarchy
needs new kinks to be born. Implement:
  - `add_kink(family, x_path, t_birth, t_death)`: insert a new trajectory
    with zero initial weight (so J is unchanged at insertion — feasibility
    trivially preserved), positioned as a small perturbation of an existing
    kink's path, alive only on [t_birth, t_death] (weight bounds forced to
    0 outside its lifetime).
  - `prune`: remove kinks whose weight stays < 1e-8 for all t.
After each insertion, re-run the LP/position alternation; keep the change
only if J_certified improves by more than a tolerance.
Acceptance: starting from Run 3's solution, at least one accepted insertion
that raises J_certified.

> **Status: implemented.** Lifetime windows are imposed as per-time-node
> weight upper bounds: `lp_weights_f`/`lp_weights_g` gained an optional `ub`
> argument (built by `_wbounds`), 0 = dead there. A kink's lifetime is a
> boolean mask `alive_f`/`alive_g` (Np1 x K); `_ub()` turns it into those
> bounds. `ub=None` reproduces the original LP bounds EXACTLY, so Runs 1-5
> are bit-for-bit unchanged (verified: all-alive vs ub=None gives
> `max|diff| = 0.0`; full-driver J_certified still 1.916 / 2.220 / 2.309 /
> 2.420 / 2.544). `add_kink` inserts a perturbed copy of a parent column at
> zero weight (confirmed `dJ = 0` at insertion); `prune` drops all-t
> zero-weight columns (keeps >=1 per family). Column identity survives the
> per-row position sort because `optimize_positions` now permutes the masks
> by the same argsort. Certification is window-aware: `certify()` (factored
> out of `report()`, which now wraps it) and `_refine_mask` carry the
> birth/death window onto the fine grid so honest J respects the imposed
> lifetimes; windowless solutions certify identically to before. The block
> alternation was extracted into `_alternate` (mask-aware), shared by `run()`
> and the new topology driver `grow_topology()`.
> Acceptance results: from Run 3's converged 3+2 (J_certified 2.309),
> `grow_topology` accepted two insertions -> 2.393 (+f) -> 2.411 (+g), both
> passing `verify_dense`. In isolated tests a short-lived windowed g-kink on
> [0.50, 1.00] was accepted, exercising the lifetime machinery (the
> full-driver greedy prefers full-lifetime windows because for pure
> J-maximization more DOF wins; deliberately restricting lifetimes is Task
> D's job). Driver "Run 6" narrates this.

### Task C — Per-kink adaptive time nodes  [DONE (graded-grid route)]

Fine-generation kinks live fast and short: a kink alive on a window of
length 0.05 needs dense time nodes there and none elsewhere. Uniform global
time grids waste ~90% of variables. Give each kink its own node set (or use
a global graded grid refined inside each kink's lifetime). The harvest sum
and constraint checks must then be assembled per-interval on the union of
the relevant node sets.
Acceptance: reproduce Run 3's J_certified within 1% using <= half the total
time-node count.

> **Status: implemented (global graded-grid route, the second option above).**
> The key enabling fact: `total_J`, the weight-LPs, and the monotonicity
> checks never read node SPACING -- the `dt` cancels analytically in the
> harvest sum -- so an arbitrary NON-uniform time grid is transparent to
> them. Only seeding and certification consult `t`, and both handle
> non-uniform grids. So the whole graded-grid capability is: `run(..., t=)`
> to inject any node set; `graded_grid(windows, coarse_N, fine_sub)` to build
> a coarse background + per-window refinement; `refine_time` generalized to
> subdivide EACH interval (so grading survives certification -- on a uniform
> grid this is bit-for-bit the old behaviour, Runs 1-6 unchanged); and
> `n_live_nodes` as the cost metric.
> Acceptance results (Run 7): Run 3 (17 nodes, 85 live, J_certified 2.309)
> reproduces at 8 nodes / 40 live vars (47% of baseline) J_certified 2.2925
> via multistart -- within 1% at under half the nodes. Note: all-alive Run 3
> has no short lifetimes, so a graded grid is provably no better than uniform
> there (identical optima) -- the win is purely the halved count. Grading's
> real leverage is scale separation: Run 7 Part B resolves a width-0.10
> lifetime window to 6 local steps with a graded grid at 92 live vars
> (J_certified 2.3065, within 1% of baseline) vs 312 live vars for a uniform
> grid at the same local resolution (3.4x fewer, and the ratio grows as the
> window narrows). NOT the true per-kink-node-set route: kinks still share
> one global (graded) grid, dead nodes pinned to zero weight via Task B's
> masks; that is sufficient for the generation experiment and far less
> invasive than rewriting the LP/harvest assembly per-kink.

### Task D — The renormalization warm start  [DONE (machinery); acceptance NOT met at gen 0 — HONEST NULL, see below]

If the hierarchy is self-similar, generation k+1 is approximately an
affinely rescaled copy of generation k (shorter lifetime, narrower spatial
extent, riding on top of generation k's path). Implement
`spawn_generation(sol, scale_t, scale_x)`: copy the finest existing
generation's kink trajectories, contract them in time and space around the
end of the parent's travel path, insert with zero weight (Task B machinery),
and re-optimize.
Acceptance: spawning finds improvements faster (fewer outer iterations to a
given J) than inserting randomly perturbed kinks.

**Status (Run 8).** `spawn_generation(sol, scale_t, scale_x, families, rng)`
is implemented: it picks each family's most-active kink (the current finest
carrier), contracts that trajectory spatially by `scale_x` about the end of
its travel path and temporally to a `scale_t`-fraction window at that end, and
inserts the contracted copy at zero weight via `_insert_column` (verified
J-neutral at insertion: J at insertion = G0's J exactly). The acceptance test
is **not met on Run 3's gen-0 optimum**, reported straight rather than tuned:

  - spawn: J_certified 2.3089 -> 2.3251 (dJ +0.016), feasible, converges in 2
    outer iterations.
  - random insertion (add_kink, best feasible of 4 seeds): 2.3089 -> 2.3700
    (dJ +0.061), feasible, converges in 8 outers.

The warm start converges *faster* but into a *shallower* basin, and does not
beat random on J. **Root cause is structural, not a bug:** Run 3's gen-0
optimum is not a self-similar travel hierarchy — its kinks barely travel, so
contracting about the travel end yields a copy nearly co-located with its
parent, and two hats at one point are redundant in a convex-hat sum (the LP
gives it ~no weight). Probed across static/travel seeds, sparse/dense (up to
6+5) bases, and wide/narrow windows: the null holds everywhere, and on the
dense base spawn additionally lands infeasible. So Task D ships correct,
reusable machinery, but the *renormalization premise it presupposes is
unvalidated at k=0* — which is itself a finding (see Section 5 caveat).

---

## 5. The experiment that everything serves  [RUNNING — machinery done (Run 9), first result INCONCLUSIVE at n_gen=3]

**Protocol (generation-gain measurement):**

1. G0 := Run 3's converged solution (record J0 ~ 2.30).
2. For k = 1, 2, 3, ...:
   a. Spawn generation k: add 2 short-lived kinks (1 for f, 1 for g, or
      2+1) near the end of the current finest travel path, with lifetimes
      and spatial extents ~ half of generation k-1's (Tasks B + D).
   b. Re-optimize (Tasks A + C make this affordable).
   c. Record Jk := J_certified and the increment dJk := Jk - J_{k-1}.
3. Plot dJk versus k.

**Interpretation — this is the whole point:**

  - dJk roughly CONSTANT in k  =>  J grows without bound, ~ c * (number of
    generations) ~ c * ln(1/w_min). This is the kink-coordinate analogue of
    the mesh observation "J ~ ln(Nx)", now with no mesh ceiling: it would
    support the conjecture sup J = +infinity (approached, never attained).
  - dJk DECAYING geometrically  =>  J is bounded; the mesh's log growth was
    a transient and the supremum is finite.

**Premise caveat (added after the Task D null, Run 8).** This whole protocol
leans on self-similarity in TWO distinct ways, and only one of them is load-
bearing for the *conclusion*:
  1. *As a warm-start convenience (Task D).* Step 2a spawns generation k as a
     rescaled copy of k-1. If the optimum is self-similar this is a good
     initial guess; if not, it is merely a bad guess and the re-optimization
     (2b) still finds whatever the true generation-k improvement is — you just
     pay more optimizer effort (or fall back to fresh multistart insertion,
     which Run 8 shows already works). So the *measurement* dJk does NOT
     depend on self-similarity holding; **the warm start is an accelerator,
     not an assumption.** Run 8's null means the accelerator doesn't yet help
     at k=0, not that the experiment is invalid — replace `spawn_generation`
     with `add_kink`/`grow_topology` multistart insertion and the protocol
     runs unchanged.
  2. *As the physical interpretation (Section: "snaking inside snaking").* The
     mental model — each generation rides on the previous one's path with a
     fixed contraction ratio — is a *hypothesis being tested*, precisely by
     the secondary measurements below. If generations do NOT settle to a fixed
     ratio, that is a real result (the growth, if any, is not a clean
     renormalization ladder), not a failure of the experiment.
The dangerous, load-bearing assumption would be to *impose* a fixed
contraction ratio and only ever spawn rescaled copies — then a constant dJk
could be an artifact of the imposed geometry rather than a discovered law.
Guard: at each generation ALSO try a from-scratch multistart insertion (Run 8
random arm) and keep the honest best; only trust a constant-dJk law if it
survives that free search. Run 8's finding (random beats the rescaled copy at
k=0) means the free search is currently *stronger* than the self-similar warm
start, so this guard is not optional — it is the actual workhorse for now.

Secondary measurements to log per generation: lifetime and spatial extent
of the new kinks after optimization (test self-similarity: do they settle
at a fixed contraction ratio?), their jump sizes (hypothesis: O(1),
Lipschitz-saturated, scale-invariant), and where they sit relative to the
parent path (hypothesis: riding on it — "snaking inside snaking").

**Honesty requirements (non-negotiable):**
  - Every reported J is J_certified: time-refined by >= 8x, weight-LPs
    re-solved on the fine grid, all constraints verified on a dense grid
    (the `report()` pipeline already does this — keep it).
  - As generations get faster, the certification refinement factor must
    grow so that the finest kink's lifetime always spans >= 8 fine steps.
  - The position optimizer is nonconvex: re-run each generation from >= 3
    seeds (rescaled copy, jittered copy, random) and keep the best; report
    the spread, not just the max.

**Status (Run 9).** The protocol is implemented per the Premise Caveat's
guard: `generation_step`/`generation_ladder` insert generation k's kinks via
`add_kink` multistart (the free-search workhorse, not `spawn_generation`)
under an IMPOSED lifetime window `w_k = window0 * window_ratio**(k-1)`
anchored at the shared travel-path end, alongside a guard arm (the same call
with a full-lifetime window, compared but never adopted). One graded time
grid (Task C) is built up front for the whole ladder, sized so each
generation's window keeps roughly constant local node density instead of
collapsing below the >= 8-fine-step floor as it narrows.

First run, from Run 3's G0 (J_certified 2.3089), n_gen=3, window0=0.5,
window_ratio=0.5, 3 seeds/generation:

    k    w_k      Jc       dJk      guard_Jc  guard_dJk
    1   0.500   2.3636   +0.0546    2.4015    +0.0926
    2   0.250   2.4684   +0.1048    2.4864    +0.1229
    3   0.125   2.4728   +0.0045    2.4993    +0.0309

All feasible (after fixing two real bugs during verification: a regrid step
that silently lost resolution outside the imposed windows — caught by the
"regrid must reproduce base J within 1%" guard — and an under-converged
optimizer budget that landed every generation marginally infeasible until
`outer`/`pos_iters` were raised; see `plans/run9-code-plan.md` for detail).

**Reading: inconclusive, not a null.** The guard arm beats the windowed arm
at every generation (same qualitative finding as Run 8's Task D null, now
shown to persist under a shrinking window too), and dJk itself is
non-monotone over only 3 points (+0.055, +0.105, +0.004) — too short a
ladder to distinguish "constant" from "decaying" from "hasn't reached the
relevant scale yet." This is an honest intermediate result, not evidence
either way on sup J. Next steps (more generations, more seeds, reading the
already-collected per-generation diagnostics for a self-similarity signal,
and possibly a different/deeper base if guard keeps winning) are tracked in
`plans/run9-generation-gain-ladder.md`.

## 6. Files

The prototype was originally one file (`kink_opt.py`); it was split into a
`kink_opt/` package by dependency layer (verified byte-identical Run 1-9
output before/after). Read `kink_opt/__init__.py`'s module docstring first;
it documents the exact constraint translations and the LP structure.

  - kink_opt/geometry.py : hat_matrix, conv_eval, _wbounds (geometry +
    weight-bound builder); MARGIN, GAP, PEN_W constants.
  - kink_opt/lp.py : lp_weights_f, lp_weights_g (convex blocks, `ub=` for
    lifetime windows, Task B).
  - kink_opt/objective.py : total_J, grad_total_J (harvest objective +
    analytic gradient, Task A); penalty, grad_penalty, _step_diff_grad,
    optimize_positions (nonconvex block, gradient-driven, Task A; mask-aware
    sort, Task B).
  - kink_opt/verify.py : refine_time (per-interval subdivision so graded
    grids survive certification, Task C), _refine_mask, verify_dense,
    certify, report (window-aware certification); graded_grid, n_live_nodes
    (non-uniform time grid builder + node-count cost metric, Task C);
    _interp_to_grid (linear-in-time interpolation of weights/positions onto
    an arbitrary grid, shared by refine_time and the Section 5 ladder's
    one-off migration); _ub (lifetime mask -> LP upper bounds, shared by
    solver.py and topology.py).
  - kink_opt/solver.py : _alternate (shared mask-aware block alternation);
    run (driver, accept/reject safeguard on the position step, `t=` to
    inject a non-uniform grid, Task C); multistart (reruns `run` over
    several `rng_seed` kink-jitters, parallel across processes, keeps the
    best).
  - kink_opt/topology.py : add_kink, _insert_column, prune, grow_topology
    (topology moves, Task B); _lifetime_window, spawn_generation
    (renormalization warm start, Task D); _seed_grown (feasible-weight
    bootstrap after a family grows by one column, shared by Task D and the
    Section 5 ladder); _kink_diagnostics (post-optimization
    lifetime/extent/jump/offset-from-parent for one kink column, with a
    fully-pruned fallback); generation_step, generation_ladder (Section 5
    driver: one windowed + one guard-arm insertion per generation, dJk
    measurement).
  - kink_opt/demos.py : the narrated Run 1-9 `main()` (was the `__main__`
    block); also runnable as `python3 -m kink_opt` via `kink_opt/__main__.py`.
  - kink_opt/__init__.py : module docstring (source of truth for the math)
    + re-exports of the public API, so `from kink_opt import run, conv_eval,
    report` (used by visualize.py) is unaffected by the internal split.
  - visualize.py : imports run/conv_eval/report from the kink_opt package,
    plots surfaces/heatmaps/slices/kink-trajectories for one solution. Not
    part of the task list above.
  - CLAUDE.md : repo-orientation notes for Claude Code sessions (commands,
    architecture summary, the "verify gradients numerically" lesson).
