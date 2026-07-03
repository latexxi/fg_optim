# Plan: Run 9 — Generation-Gain Ladder (the Section 5 experiment)

## Goal

Decide the central question of STRATEGY.md: does J grow like log(1/w) as ever-finer
kink generations are added (sup J = +infinity), or does the per-generation gain decay
(J bounded, mesh log growth was a transient)?

The measurement is dJk — the certified J increment from adding generation k — plotted
against k. Roughly constant dJk over 3-4 generations supports the log law; geometric
decay refutes it.

## Why this is unblocked now

Run 8's null result killed only the `spawn_generation` warm start (an accelerator),
not the measurement itself. The workhorse insertion path is `add_kink` multistart
(Run 8's random arm), which already works. Tasks A-D provide everything needed:
analytic gradients (affordable re-optimization), lifetime windows (Task B), graded
time grids (Task C), and honest window-aware certification.

## Core design decisions (already made, recorded here)

1. **Base G0 = Run 3's converged optimum** (J_certified 2.309), not Run 5's
   brute-force frontier. Controlled starting point; each generation's contribution
   must be attributable.

2. **Imposed lifetime windows are the experiment.** Each generation k gets a lifetime
   window w_k = w_{k-1}/2, anchored near the end of the current finest travel path.
   Run 6 showed an unrestricted greedy always picks full lifetimes (more DOF wins),
   which just repeats Runs 4-5 and measures nothing about scale. The window constraint
   is what makes dJk a per-scale gain.

3. **Random-jitter seeding around the whole parent path, not a rescaled copy at the
   path end.** Run 8 showed the rescaled copy co-locates with its parent and starves
   in the LP. Let the optimizer discover the geometry; do not impose a contraction
   ratio (STRATEGY.md's own warning about imposed-geometry artifacts).

4. **Guard arm every generation:** alongside the windowed insertion, also try a free
   full-lifetime insertion and report both. A constant-dJk law is only trustworthy if
   it survives this free search.

## Coding steps (in order)

1. **Ladder driver** — a `generation_ladder(...)` function: takes G0, number of
   generations, window-halving schedule; per generation runs the windowed multistart
   insertion (>= 3 seeds), re-optimizes, certifies, records dJk. Reuses `add_kink`,
   `_alternate`, `certify`; no new solver machinery.

2. **Graded-grid integration** — each generation's window gets local time refinement
   via `graded_grid`; grid grows with the ladder instead of a global uniform grid.
   Certification refinement factor must scale so the finest lifetime spans >= 8 fine
   steps (Section 5 honesty rule).

3. **Secondary measurements logger** — per generation record: post-optimization
   effective lifetime and spatial extent of the new kinks (do they shrink from the
   imposed window?), jump sizes, spatial offset from the parent path. These test the
   self-similarity hypothesis as a *finding*, not an assumption.

4. **Guard arm** — free (full-lifetime) insertion alternative per generation, same
   seed budget; keep and report both results.

5. **"Run 9" demo block** — narrated `__main__` entry following the existing Run 1-8
   pattern: header explaining what Run 8 left open, 3-4 generations, print dJk table
   and the secondary measurements. Do not replace prior runs.

6. **Readout** — dJk vs k table (and optionally a plot via visualize.py). Interpret
   per Section 5: constant = log law, decaying = bounded.

## Honesty requirements (carried over from STRATEGY.md, non-negotiable)

- Every reported J is J_certified (fine-grid LP repair + dense verification).
- Multistart spread reported per generation, not just the max.
- Report the guard-arm comparison even when it embarrasses the ladder.
- If dJk collapses because the optimizer fails (infeasible, non-converged) rather
  than because the gain is truly absent, say so — a null from optimizer weakness is
  not evidence for bounded J.

## Out of scope

- True per-kink node sets (graded global grid suffices; Task C note).
- Fixing `spawn_generation` (keep as-is; revisit only if a later generation *does*
  show travel structure worth copying).
- Theory work on the maintenance-cost recursion; this is measurement only.
