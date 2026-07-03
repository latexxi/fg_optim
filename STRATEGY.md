# Briefing: Hierarchical Kink Optimization for a Coupled PDE Functional

This document is self-contained. You are given one code file, `kink_opt.py`
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
docstring of kink_opt.py): convexity = nonnegative weights, Lipschitz = two
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
position trajectories at the time nodes. The solver in kink_opt.py
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

> **Status (current `kink_opt.py`):** Runs 1-3 reproduce at J_certified =
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

### Task C — Per-kink adaptive time nodes  [NOT STARTED]

Fine-generation kinks live fast and short: a kink alive on a window of
length 0.05 needs dense time nodes there and none elsewhere. Uniform global
time grids waste ~90% of variables. Give each kink its own node set (or use
a global graded grid refined inside each kink's lifetime). The harvest sum
and constraint checks must then be assembled per-interval on the union of
the relevant node sets.
Acceptance: reproduce Run 3's J_certified within 1% using <= half the total
time-node count.

### Task D — The renormalization warm start  [NOT STARTED]

If the hierarchy is self-similar, generation k+1 is approximately an
affinely rescaled copy of generation k (shorter lifetime, narrower spatial
extent, riding on top of generation k's path). Implement
`spawn_generation(sol, scale_t, scale_x)`: copy the finest existing
generation's kink trajectories, contract them in time and space around the
end of the parent's travel path, insert with zero weight (Task B machinery),
and re-optimize.
Acceptance: spawning finds improvements faster (fewer outer iterations to a
given J) than inserting randomly perturbed kinks.

---

## 5. The experiment that everything serves  [NOT STARTED — Task B done, blocked on Task D]

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

## 6. Files

  - kink_opt.py : the prototype. Read its module docstring first; it
    documents the exact constraint translations and the LP structure.
    Functions: hat_matrix, conv_eval, _wbounds (geometry + weight-bound
    builder); lp_weights_f, lp_weights_g (convex blocks, `ub=` for lifetime
    windows, Task B); total_J, grad_total_J (harvest objective + analytic
    gradient, Task A); penalty, grad_penalty, _step_diff_grad,
    optimize_positions (nonconvex block, now gradient-driven, Task A;
    mask-aware sort, Task B); refine_time, _refine_mask, verify_dense,
    certify, report (window-aware certification); _alternate (shared
    mask-aware block alternation); run (driver, accept/reject safeguard on
    the position step); multistart (reruns `run` over several `rng_seed`
    kink-jitters, parallel across processes, keeps the best); add_kink,
    prune, grow_topology (topology moves, Task B).
  - visualize.py : imports run/conv_eval/report from kink_opt.py, plots
    surfaces/heatmaps/slices/kink-trajectories for one solution. Not part
    of the task list above.
  - CLAUDE.md : repo-orientation notes for Claude Code sessions (commands,
    architecture summary, the "verify gradients numerically" lesson).
