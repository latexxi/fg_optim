# Plan: Run 9 — Generation-Gain Ladder (the Section 5 experiment)

## Status: IMPLEMENTED and RUN, result INCONCLUSIVE at n_gen=3 — see below

The machinery (`generation_step`, `generation_ladder`, `_kink_diagnostics`,
`graded_grid` per-window density, the regrid migration) is built, verified,
and committed (`d91fc20`). Details of the build are in
`plans/run9-code-plan.md`. This file now also records what the first real
run found and what to do next.

## Goal

Decide the central question of STRATEGY.md: does J grow like log(1/w) as
ever-finer kink generations are added (sup J = +infinity), or does the
per-generation gain decay (J bounded, mesh log growth was a transient)?

The measurement is dJk — the certified J increment from adding generation k
— plotted against k. Roughly constant dJk over 3-4 generations supports the
log law; geometric decay refutes it.

## Why this was unblocked

Run 8's null result killed only the `spawn_generation` warm start (an
accelerator), not the measurement itself. The workhorse insertion path is
`add_kink` multistart (Run 8's random arm), which already worked. Tasks A-D
provide everything needed: analytic gradients, lifetime windows, graded time
grids, and honest window-aware certification.

## Core design decisions (as implemented)

1. **Base G0 = Run 3's converged optimum** (J_certified 2.309).
2. **Imposed lifetime windows are the experiment.** w_k = window0 *
   window_ratio^(k-1), anchored at the shared right endpoint (the travel
   path all generations ride toward), so window_k = (t1 - w_k, t1). An
   unrestricted greedy (Run 6) always prefers full lifetime, so the window
   constraint is what makes dJk a per-scale measurement instead of a repeat
   of Runs 4-5.
3. **Random-jitter seeding around the whole parent path** via `add_kink`,
   not a rescaled copy at the path end (Run 8 showed that co-locates and
   starves).
4. **Guard arm every generation:** a free (full-lifetime) `generation_step`
   call alongside the windowed one, compared but never adopted into the
   ladder.

## What the first real run found (Run 3's G0, n_gen=3, seeds=range(3))

```
  base J_certified = 2.30893

  k     w_k       Jc      dJk    ok   guard_Jc guard_dJk    ok
  1  0.5000   2.3636  +0.0546  True     2.4015   +0.0926  True
  2  0.2500   2.4684  +0.1048  True     2.4864   +0.1229  True
  3  0.1250   2.4728  +0.0045  True     2.4993   +0.0309  True
```

Observations:
- All feasible (after fixing a budget issue — see below).
- **Guard beats windowed at every generation.** Free insertion still finds
  more J than the imposed-window insertion at every scale tried so far —
  the same qualitative finding as Run 8 (self-similar/constrained insertion
  underperforms free search), now shown to persist under a *shrinking*
  window, not just Run 8's single fixed one.
- **dJk is not monotone over these 3 points** (+0.055, +0.105, +0.004) — it
  rose then fell sharply. Three points cannot distinguish "constant" from
  "decaying" from "not yet at the relevant scale" — this run does not
  answer Section 5's question yet, it only proves the machinery works and
  gives one honest, unflattering data point.
- Per-generation diagnostics (lifetime/extent/jump/offset) are being
  recorded machine-computed per generation but haven't been read closely
  yet for a self-similarity signal — that's next-step work, not something
  concluded here.

**Two real bugs were caught and fixed while producing this result** (full
detail in run9-code-plan.md's Status section): a regrid that silently lost
resolution outside the imposed windows (found via the "regrid must be
J-neutral" guard catching a 2% drift), and an under-converged optimizer
budget that landed all three generations marginally infeasible (found by
raising outer/pos_iters and watching the violation collapse to float
noise). Both are fixed; the numbers above are from the corrected code.

## What to run next (not yet done)

1. **More generations.** n_gen=3 is too short to read a trend. Push to
   n_gen=5-6 (w_k down to ~0.03) and see whether dJk settles into either a
   plateau (log law) or a visible decay, watching the Section 5 honesty
   rule that the finest lifetime must still span >= 8 fine steps (the
   graded_grid fine_sub scaling already targets this, but confirm at the
   smallest w_k actually reached).
2. **More seeds per generation.** range(3) is thin for a nonconvex local
   search; STRATEGY.md's own honesty requirement asks for >= 3 seeds and to
   report the spread (already done) — worth trying range(5) or more once
   per-generation cost is measured, to make sure the "best of seeds" isn't
   noise-dominated, especially since generation 2's spread (2.3708 / 2.4684
   / 2.4218) already shows real seed-to-seed variance.
3. **Read the diagnostics for self-similarity**, not just dJk: has the
   post-optimization lifetime/extent of each generation's kinks contracted
   from the imposed window on its own, and by roughly what ratio? Compare
   across generations 1-3 (already computed, sitting in `ladder["generations"]
   [k]["diagnostics"]`) before running more generations — this is free,
   already-collected data.
4. **Try a different base / window schedule** if G0 keeps showing "guard
   beats windowed": Section 5's Premise Caveat already anticipated this —
   a persistent guard-win could mean Run 3's G0 genuinely isn't a
   self-similar-hierarchy starting point (same conclusion Run 8 reached),
   in which case the productive move is trying a *deeper* base (e.g. Run
   6's grown-topology solution, or a seed from Run 5's wider multistart)
   rather than pushing this same G0 further.

## Honesty requirements (STRATEGY.md, carried over, upheld so far)

- Every reported J is J_certified — done.
- Multistart spread reported per generation, not just the max — done.
- Guard-arm comparison reported even though it currently loses (embarrassing
  is fine) — done.
- If dJk collapses because the optimizer failed rather than the gain being
  truly absent, say so — done: the budget bug above is exactly this case
  and is recorded rather than papered over.

## Out of scope (unchanged)

- True per-kink node sets (graded global grid suffices).
- Fixing `spawn_generation` (unless a later generation shows travel
  structure worth copying).
- Theory work on the maintenance-cost recursion; this is measurement only.
