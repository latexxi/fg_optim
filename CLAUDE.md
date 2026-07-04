# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small numerical-optimization research prototype: the solver is a plain-import Python package (`kink_opt/`, no build system, no `setup.py`/`pyproject.toml`), plus a standalone `visualize.py`. It solves

    max J[f,g] = int_0^1 int_{-1}^1 f_t(x,t) * g_xx(x,t) dx dt

over pairs of time-varying, x-convex, Lipschitz functions `f(x,t)` and `g(x,t)` on `x in [-1,1]`, `t in [0,1]`, subject to `f(+-1,t)=g(+-1,t)=0`, `f(x,1)=0`, `g(x,0)=0`, `f_t>=0`, `g_t<=0`. There is no formal spec beyond the module docstring in `kink_opt/__init__.py` — that docstring is the source of truth for the math (constraint dictionary, objective derivation, basis definition) and should be read before making changes to the solver.

## Files

- `kink_opt/__init__.py` — module docstring (source of truth for the math) + re-exports of the public API (so `from kink_opt import run, conv_eval, report` keeps working).
- `kink_opt/geometry.py` — hat basis (`hat_matrix`, `conv_eval`), weight-LP box bounds (`_wbounds`), the tunable constants `MARGIN`/`GAP`/`PEN_W`.
- `kink_opt/lp.py` — the convex weight blocks, `lp_weights_f` / `lp_weights_g` (scipy `linprog`, HiGHS).
- `kink_opt/objective.py` — `total_J`, analytic gradients (`grad_total_J`, `grad_penalty`, `_step_diff_grad`), `penalty`, and the nonconvex position block `optimize_positions`.
- `kink_opt/verify.py` — `refine_time`, `graded_grid`, `verify_dense`, and the `certify()`/`report()` verification pipeline; also `_ub` (lifetime mask -> LP upper bounds) since both the driver and topology moves need it.
- `kink_opt/solver.py` — the block-coordinate driver: `_alternate`, `run`, `multistart`.
- `kink_opt/topology.py` — topology moves (Task B: `add_kink`/`prune`/`grow_topology`), the renormalization warm start (Task D: `spawn_generation`), the Run 9 generation-gain ladder (`generation_step`/`generation_ladder`, and the shared regrid helper `_regrid_onto_windows`), and the Run 10 scale-sweep discriminators (`scale_sweep`, `decay_ratios`, `fit_geometric`, the budget-adequacy gate `_gate_report`/`_budget_stable`).
- `kink_opt/persist.py` — `save_run`/`load_run`, `save_ladder`/`load_ladder`, `save_sweep`/`load_sweep`: persist a solution's kink coordinates, dense evaluated `f(x,t)`/`g(x,t)` fields, and a heatmap PNG to `results/<tag>/` (gitignored, regenerable via `python3 -m kink_opt`), so later analysis (diagnostics, dJk/dJ(w) trends) works from disk instead of rerunning the optimization.
- `kink_opt/viz.py` — `field_grid` (dense `f`/`g` evaluation from a solution dict) and `plot_heatmaps`, factored out of `visualize.py` so `persist.py` can render the same heatmap PNGs without importing the standalone script.
- `kink_opt/demos.py` — the narrated `__main__` driver, `main()`, running the numbered demos ("Run 1" ... "Run 10"). Also runnable as `python3 -m kink_opt` via `kink_opt/__main__.py`.
- `visualize.py` — imports `run`, `conv_eval`, `report` from the `kink_opt` package, runs one optimization, and saves `surfaces.png`, `heatmaps.png`, `slices.png`, `kink_trajectories.png` to the repo root.

## Commands

```
python3 -m kink_opt      # runs the full Run 1-10 demo sequence, prints J at each stage
python3 visualize.py     # runs one optimization and regenerates the four PNGs (also calls plt.show())
```

No test suite, linter, or build system is present. Dependencies (numpy, scipy, matplotlib) are assumed already installed in the environment; there is no requirements file.

When validating a change to any gradient function, check it against finite differences before trusting it — see "Analytic gradients" below.

## Architecture

### Representation

Both `f` and `g` are represented in every time slice as a negative sum of "hat" (tent) basis functions:

    f(x,t) = -sum_i a_i(t) * hat(x; xi_i(t)),   a_i >= 0
    g(x,t) = -sum_m b_m(t) * hat(x; eta_m(t)),  b_m >= 0

`hat(x;c)` is the piecewise-linear tent with `hat(+-1)=0`, `hat(c)=1`. This basis makes convexity in `x` exact and automatic (`a_i, b_m >= 0`), and makes the boundary condition `f(+-1,t)=0` built-in. `xi_i(t)` / `eta_m(t)` are the "kink" positions (trajectories over time); `a_i(t)` / `b_m(t)` are the weights.

The objective is computed in "harvest" form (`total_J`), not by quadrature: `g_xx` is a sum of Dirac deltas at `g`'s kinks, so `J` reduces to a discrete sum of `f`'s rise sampled at those kink locations, avoiding any near-singular integration.

### Block-coordinate optimization (in `_alternate()`, driven by `run()`)

Positions and weights are optimized in alternating blocks because the problem is only piecewise-convex, not jointly convex:

1. **Positions frozen -> `J` is linear in `a`** -> LP via `lp_weights_f` (scipy `linprog`, HiGHS, exact global optimum).
2. **Positions frozen -> `J` is linear in `b`** -> LP via `lp_weights_g` (same).
3. **Weights frozen -> nonconvex NLP in positions** -> `optimize_positions` (L-BFGS-B), then LPs 1-2 are re-run to restore exact feasibility (the position step only satisfies constraints via a soft penalty, not exactly).

Only step 3 is nonconvex; steps 1-2 are solved to certified global optimality every time. The cycle lives in `_alternate()` (mask-aware; shared by `run()` and `grow_topology()`); it loops the block-coordinate cycle (`outer` iterations), and because step 3 is not guaranteed to improve `J` monotonically, it tracks the best feasible state seen and reverts to it on regression (`patience` controls how many non-improving iterations are tolerated before stopping). `run()` is a thin wrapper that builds the seed geometry + all-alive lifetime masks and calls `_alternate()`.

### Topology moves (Task B — `add_kink` / `prune` / `grow_topology`)

The alternation optimizes a **fixed** kink count; the hierarchy needs kinks to be born and die. A kink's lifetime is a boolean mask (`alive_f`/`alive_g`, shape `(Np1, K)`) turned into per-time-node weight upper bounds (`_ub` -> the LPs' `ub=` argument, built by `_wbounds`): a "dead" node pins that weight to 0. `ub=None` reproduces the original LP bounds **exactly**, so all-alive solutions (Runs 1-5) are bit-for-bit unchanged.

- `add_kink(family, ...)` inserts a perturbed copy of an existing kink's trajectory at **zero weight**, alive only on `[t_birth, t_death]` — J and feasibility are unchanged at insertion, then the LPs decide its weight.
- `prune` drops columns whose weight is `< 1e-8` at every `t` (keeps ≥1 per family).
- `grow_topology` (driver, "Run 6") greedily births one kink per generation over a few families/windows/jitters, re-alternates, prunes, and keeps the insertion only if `J_certified` strictly improves.

Column identity survives the per-row position sort because `optimize_positions` permutes the masks by the same argsort (the math only needs the *set* of positions per node, but the mask must track which of those are dead). Certification is window-aware (see below).

Because `optimize_positions` is only a local search, its outcome depends on the initial kink jitter (`rng_seed`, previously hardcoded to 0). `multistart()` sweeps several seeds through `run()` (in parallel across processes) and keeps the best by coarse `J` — note it does **not** check feasibility when picking a winner, so at high kink counts it can occasionally pick a result that fails verification (see the EXT note at the bottom of the file and Run 5's Kf=7/8 exploration).

### Analytic gradients

`optimize_positions` always uses analytic gradients (`grad_total_J` + `grad_penalty`, passed as `jac=`) — there is no finite-difference fallback path. This matters for performance (~20-100x fewer objective evaluations per solve) and is what makes the wider multistart sweeps in the driver affordable.

The gradient derivation is the trickiest part of this file: in `penalty()`'s monotonicity terms (`f_t>=0`, `g_t<=0`), the kink positions being differentiated are simultaneously (a) the hat-function *node* argument and (b) the *evaluation point* the constraint is checked at (checkpoints are the union of both slices' own kink positions). `_step_diff_grad` handles this by separating each contribution into a "node-role" term (full sum over checkpoints) and an "eval-role" term (only ever lands on the matching diagonal checkpoint). This derivation has already produced two real sign bugs once; any change to `_step_diff_grad`, `grad_total_J`, or `grad_penalty` should be re-verified with a finite-difference check (compare against numerically-differentiated `total_J`/`penalty` across a few random `(Np1, Kf, Kg)` shapes) before being trusted — don't rely on visual inspection of the algebra alone.

### Verification pipeline (`certify()` / `report()`)

`run()`'s returned `J` is computed on the coarse time grid used for optimization and can be inflated by exploiting the midpoint-quadrature discretization. `certify()` is the "honest" check: it interpolates the solution onto a finer time grid (`refine_time`), re-solves the weight LPs there with positions frozen to restore exact feasibility (`lp_weights_f`/`lp_weights_g` again — this repair step only works because those blocks are LPs), and then calls `verify_dense()` to rebuild `f,g` on a dense `x` grid and check every constraint directly plus cross-check `J` via an independent integral identity. `report()` just prints `certify()`'s numbers. Always read `J_certified` / `constraints_ok` from that output rather than the raw `J` from `run()`.

The repair is **window-aware**: if the solution carries lifetime masks, `_refine_mask` maps each kink's `[birth, death]` window onto the fine grid and the fine-grid LPs respect it (via `ub=`), so measured J reflects the imposed lifetimes. Windowless (all-alive) solutions certify identically to before — the mask path is a no-op there.

### Non-uniform (graded) time grids (Task C — `graded_grid` / `run(t=)`)

The time grid need not be uniform. The pivotal fact: `total_J`, the weight-LPs, and the monotonicity checks never read node **spacing** — `dt` cancels analytically in the harvest sum, so `J` is a pure sum over consecutive node pairs. An arbitrary non-uniform `t` is therefore transparent to the whole solver; only seeding and certification consult `t`, and both handle it. So Task C is small: `run(..., t=)` injects any node set; `graded_grid(windows, coarse_N, fine_sub)` builds a coarse background plus per-window refinement; `n_live_nodes(r)` counts live `(kink, node)` decision variables (the cost metric). `refine_time` was generalized to subdivide **each** interval into `sub` pieces (rather than a global `linspace`) so grading survives certification — on a uniform grid this is bit-for-bit the old behaviour, so Runs 1-6 certify unchanged. Kinks still share one global (graded) grid — dead nodes pinned to zero weight via Task B's masks — so this is the "graded-grid" route, **not** true per-kink node sets. All-alive problems gain nothing from grading (no short lifetimes to exploit); the leverage is scale separation, where a narrow lifetime window costs `O(1)` extra nodes instead of forcing an `O(1/w)` global uniform grid.

### Renormalization warm start (Task D — `spawn_generation`)

`spawn_generation(sol, scale_t, scale_x, families, rng)` builds the next hierarchical "generation" as an affine copy of the current finest carrier: for each family it takes the most-active kink, contracts that trajectory spatially by `scale_x` about the end of its travel path and temporally to a `scale_t`-fraction lifetime window at that end, and inserts the contracted copy at **zero weight** via `_insert_column` (the same Task B machinery `add_kink` uses) — so J and feasibility are unchanged at insertion and the caller re-optimizes. It exists to test the STRATEGY.md self-similarity conjecture ("generation k+1 ≈ rescaled copy of k, riding its path").

**Honest null result (Run 8):** on Run 3's gen-0 optimum the warm start does **not** beat a random insertion. It converges faster (2 outers vs ~8) but into a shallower basin (J_certified 2.3251 vs random's 2.3700). Cause is structural, not a bug: Run 3's kinks barely travel, so contracting about the travel end produces a copy nearly **co-located** with its parent, and two hats at one point are redundant in a convex-hat sum (the LP assigns it ~no weight). The null holds across static/travel seeds, sparse/dense bases, wide/narrow windows. Takeaway for anyone continuing this: the Section-5 generation-gain experiment does **not** require the warm start to work — `spawn_generation` is an accelerator, and the load-bearing insertion path is still `add_kink`/`grow_topology` multistart (Run 8's random arm is the current workhorse). See STRATEGY.md Task D + the Section 5 "Premise caveat".

### The generation-gain ladder and its budget-artifact history (Run 9)

`generation_step(cur, window, seeds, outer, pos_iters, ...)` inserts one new f-kink and one new g-kink (via `add_kink`, alive only on `window`), multistarts over `seeds`, and keeps the best certified-feasible candidate. `generation_ladder(base, n_gen, window0, window_ratio, ...)` chains this: `w_k = window0 * window_ratio**(k-1)`, all windows anchored at the shared right endpoint `t1`, tracking `dJk = Jk - J_{k-1}`. It regrids `base` once up front onto a graded grid (Task C) sized to resolve the narrowest window, via `_regrid_onto_windows` (shared with `scale_sweep`, Run 10) — this migration is checked against `certify(base)` and raises past a 1% relative gap, since a silent large mismatch would corrupt every downstream `dJk`. A "guard arm" (the identical call with a full-lifetime window) runs alongside every generation for comparison but is never adopted into the ladder.

This measurement has repeatedly turned out to be budget-sensitive in ways that mimic the signal being measured: under-converged `outer`/`pos_iters` produces a starved new kink (`jump_mean=0`) whose `dJk` collapses toward zero, which looks identical to genuine decay unless caught. Three separate instances of this were found and fixed while producing Run 9's numbers (see `plans/run9-generation-gain-ladder.md`), and the position-NLP dimensionality grows with *accumulated* kink count across a ladder, not just grid size, so a flat per-ladder budget that was adequate at `n_gen=3` silently starves `n_gen=5`. `generation_ladder(..., budget_fn=lambda k, n_live: (outer_k, pos_iters_k))` exists to let a driver scale the budget with accumulated kink count instead of using one flat value for the whole ladder.

### Run 10 — scale-sweep discriminators (decoupling "gain vanishes" from "optimizer starved") — result: INCONCLUSIVE

Run 9's `n_gen=3` result (dJk decaying on two independent bases) is suggestive but not conclusive: three points, and decay is exactly what budget starvation produces. Run 10 (`plans/run10-scale-sweep-discriminators.md`, `plans/run10-code-plan.md`) adds `scale_sweep(base, ws, ...)`, which measures a *single* generation's certified gain at each window scale `w` independently (own fresh regrid per point via `_regrid_onto_windows`, so problem size stays constant across the sweep and nothing accumulates like it does in a ladder).

Getting a trustworthy number out of `scale_sweep` took four rounds of finding and fixing a real methodological issue, each only visible once the previous one was fixed — a cautionary instance of the same "budget-sensitive measurement" pattern Run 9 already hit, showing up in new forms:

1. **Search-noise contamination**: a windowed insertion whose new kinks end up at `jump_mean=0` (dead at convergence, contributing nothing) can still show nonzero `dJ`, because the insertion-jitter multistart alone can push the *existing* kinks to a different local optimum. Fixed with `generation_step(..., force_dead=True)`, a null-control arm that forces the new columns permanently dead so the LP can never weight them, isolating this search-noise floor.
2. **Wrong statistic**: `windowed["Jc"] - null["Jc"]` subtracts two independent best-of-N-seeds maxima — a high-variance order statistic, not a mean, and the wrong tool for isolating a small real effect. Fixed with `_paired_dJ`, which differences `Jc` per matching seed (both arms share the same `rng` draws up to the point their alive masks diverge) and reports a mean and standard error over seeds feasible in both arms.
3. **Resolution floor**: `scale_sweep` originally regridded every point with the same flat `fine_sub`, which silently hit `graded_grid`'s 2-subinterval floor (~3 nodes spanning the whole window) for every `w` below a threshold — fixed by scaling `fine_sub` with `w`, mirroring what `generation_ladder` already did.
4. **Budget didn't port**: Run 9's `outer=40/pos_iters=100`, "proven adequate" for its own ladder setup, gave low seed-feasibility and near-zero search diversity in `scale_sweep`'s different (single-window) regrid — doubled to `outer=80/pos_iters=200`.

Even after all four fixes, the result is genuinely **inconclusive**: `corrected_dJ(w)` sits within roughly 1-2 standard errors of zero at every scale tested, with no visible trend in either direction — it does not show the nonzero-constant signature log-growth would predict, nor a clean power-law decay to zero that would confirm bounded `J`. The one unambiguous signal is that `guard_dJ` (an unconstrained, full-lifetime insertion) is robustly positive at every point while the corrected windowed-insertion signal is not — i.e. free search finds real value here that the imposed-window mechanism, once search-noise is subtracted out, does not detectably provide. `decay_ratios`/`fit_geometric` (Experiment 2/4) and a constructive zero-optimizer arm (Experiment 5, `kink_opt/construct.py`, not yet implemented) remain as options if this is pursued further — see the strategy plan for the full numbers and reasoning, and for why the constructive arm is the only path that could *prove* rather than merely *support* either side.

### Tunable constants (`kink_opt/geometry.py`)

- `MARGIN` — keeps kink positions inside `(-1+MARGIN, 1-MARGIN)`, bounds for the position NLP.
- `GAP` — minimum spacing enforced between same-family kinks (soft penalty).
- `PEN_W` — weight on the soft penalty term relative to `-J` in the position NLP objective.

### `demos.py` (the `__main__` sequence)

Not a test suite — it's a sequence of numbered, narrated demos ("Run 1" through "Run 10") where each run's configuration and print-statement commentary explains what the previous run left on the table and why the next one's hyperparameters were chosen. Runs 1-5 are fixed-K weight/position optimization (Run 5's Kf=6/Kg=5 multistart is the best brute-force feasible frontier); Run 6 is the Task B topology-move demo (`grow_topology` growing Run 3's 3+2 by accepted insertions); Run 7 is the Task C graded-grid demo (Part A reproduces Run 3 at half the time nodes; Part B shows a narrow lifetime window costing far fewer variables graded than uniform); Run 8 is the Task D renormalization-warm-start demo — a deliberately **honest null**: it verifies `spawn_generation`'s insertion is J-neutral, then shows the warm start converging faster but to a worse J than random insertion, and narrates why (no self-similar travel structure at gen 0); Run 9 is the Section-5 generation-gain ladder (`generation_ladder`); Run 10 is the scale-sweep discriminator (`scale_sweep`) that isolates the window-scale variable from the ladder's confounds — also a deliberately **honest inconclusive**: four distinct methodological fixes (search-noise contamination, wrong statistic, a resolution floor, a budget that didn't port from Run 9) were needed before the result could be trusted at all, and even then `corrected_dJ(w)` shows no significant trend either direction. When extending this file, follow that pattern (add a new "Run N" with an explanatory header) rather than replacing prior runs, since they double as a record of what's already been tried.
