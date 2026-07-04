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

## Result persistence (infra added, unblocks all of the below)

Every run's setup (kwargs) and result (kink coordinates A/XI/B/ETA/t, dense
evaluated f(x,t)/g(x,t) fields, a heatmap PNG, certified J, and — for
`generation_ladder` — per-generation diagnostics/spread) is now saved to
`./results/<tag>/` by `kink_opt/persist.py` (`save_run`/`load_run`,
`save_ladder`/`load_ladder`), hooked into every Run 1-9 call in
`kink_opt/demos.py`. `results/` is gitignored (regenerable via
`python3 -m kink_opt`). This means items 3 and 4 below no longer require a
fresh optimization run to investigate — `load_ladder("run9")` pulls the
already-computed diagnostics straight from disk.

## What was run next (items 3 and 4, now done)

**Item 3 (diagnostics read) — no self-similarity signal on Run 3's G0.**
Read `load_ladder("run9")`'s per-generation diagnostics: extent/lifetime
don't contract at a consistent ratio across generations (g extent
0.112 -> 0.032 -> 0.033; f extent 0.092 -> 0.000 -> 0.086), offset_from_parent
stays roughly flat (~0.02-0.07) instead of shrinking with the halving window,
and jump_mean bounces (g: 0.455 -> 0.227 -> 0.562) rather than holding
near-constant. **New honesty-relevant finding:** generation 3's f-kink has
`jump_mean = 0.000` — it was starved to ~zero weight, so its reported
lifetime/extent are `_kink_diagnostics`'s fallback (the imposed window, not
real activity) and gen 3's tiny dJk (+0.0045) is partly an optimizer/LP
starvation artifact on the f-family, not purely "true gain is small."

**Item 4 (deeper base) — tried Run 6's grown-topology solution (4 f-kinks +
3 g-kinks, J_certified 2.4114) instead of Run 3's G0, per the Premise
Caveat's trigger** (persistent guard-win + no self-similarity signal on the
shallow base). First pass at the *same* budget as the Run-3-base ladder
(`outer=25, pos_iters=60, seeds=range(3)`) hit the exact under-convergence
failure mode already caught once before in this file: 2 of 3 seeds
infeasible at every generation, the same seed winning every time with an
identical Jc, and dJk collapsing to exactly `[+0.0062, +0.0000, +0.0000]`
(both new kinks starved, `jump_mean=0.000`, beyond generation 1). Raising the
budget (`outer=40, pos_iters=100, seeds=range(5)`) fixed the infeasibility
(1/3 -> 4/5 feasible per generation) and produced a real, non-zero,
**smoothly decaying** dJk:

```
Run3-base (3+2):  dJk = [+0.0546, +0.1048, +0.0045]   guard always ahead, non-monotone
Run6-base (4+3):  dJk = [+0.0636, +0.0131, +0.0039]   guard always ahead, cleanly decaying
                  guard_dJk = [+0.0727, +0.0490, +0.0361]                  also decaying
```

Reading: guard still beats windowed at every generation on both bases (no
self-similar warm-start advantage anywhere yet), but the deeper base's dJk
sequence is a much cleaner monotone decay (ratio ~0.2-0.3 per generation)
than the shallow base's noisy non-monotone one. This leans toward the
"J bounded, gain decays" reading of STRATEGY.md Section 5 rather than the
constant-gain log-growth conjecture — but it is one base, three generations,
not a verdict. Saved to `results/run9_run6base/` (script:
`/tmp/.../scratchpad/run9_deeper_base.py`, not checked in — should be
ported into `demos.py` as a proper "Run 10" if this direction is pursued
further, per the file's own convention of keeping prior runs as a record).

## Item 5 done — Run-3-base ladder rerun at bumped budget, non-monotone spike was ALSO an artifact

Reran the original Run-3-base ladder (n_gen=3, window0=0.5, window_ratio=0.5)
at the Run-6-base's proven-adequate budget (`outer=40, pos_iters=100,
seeds=range(5)`, up from `outer=25, pos_iters=60, seeds=range(3)`):

```
Run3-base, outer=25/pos_iters=60/seeds=3 (original): dJk = [+0.0546, +0.1048, +0.0045]  non-monotone
Run3-base, outer=40/pos_iters=100/seeds=5 (rerun):    dJk = [+0.0806, +0.0246, +0.0117]  smoothly decaying
                                          guard_dJk = [+0.0926, +0.0609*, +0.0543]  (*guard infeasible at gen2)
```

The original's spike-then-drop (gen2 > gen1) is gone at proper budget — the
rerun decays monotonically, same qualitative shape as the Run-6-base ladder.
So the original Run-3-base sequence was itself partly a budget artifact, not
a real non-monotonicity. **Both independent bases now show clean, monotone,
geometric-ish dJk decay** once budget is adequate:

```
Run3-base (3+2 -> hi-budget):  dJk = [+0.0806, +0.0246, +0.0117]   ratio ~0.31, ~0.48
Run6-base (4+3 -> hi-budget):  dJk = [+0.0636, +0.0131, +0.0039]   ratio ~0.21, ~0.30
```

Guard still beats windowed at every generation on both bases (no
self-similar warm-start advantage anywhere). Gen3's f-kink on the Run-3-base
rerun still shows `jump_mean=0.000`, but this time it looks like a genuine
"this particular insertion doesn't help" result rather than budget
starvation — the g-kink in the same generation is fully active
(`jump_mean=0.396`), and cross-seed spread is otherwise tight and consistent
(not identical-across-all-seeds the way the artifact looked before).

Saved to `results/run9_run3base_hibudget/`. Two independent bases decaying
cleanly is stronger evidence for "J bounded, log-growth was a discretization
transient" than either base alone — still only 3 generations each, so not a
final verdict, but the artifact-driven noise that made both prior results
ambiguous is now gone.

## n_gen=5 extension (Run-6 base) hit the SAME artifact, new trigger — budget must scale with grid size

Extended the Run-6-base ladder from n_gen=3 to n_gen=5 at the *same*
outer=40/pos_iters=100/seeds=range(5) budget that gave the clean n_gen=3
result. Result collapsed straight back into the starvation signature:

```
dJk sequence:       ['+0.0016', '+0.0000', '+0.0000', '+0.0000', '+0.0000']
guard dJk sequence: ['+0.0819', '+0.0889', '+0.1247', '+0.1178', '+0.1247']
```

4/5 seeds infeasible every generation, both new kinks `jump_mean=0.000`
from gen 1 onward, gen1's own dJk (+0.0016) far below the n_gen=3 run's
gen1 (+0.0636) despite starting from the identical base. Root cause is
**not** the same "outer too low" bug in isolation — it's that
`generation_ladder` precomputes ONE graded grid for the whole ladder up
front (see its docstring), sized to resolve `w_5=0.0312`'s window. That
pushed the shared grid from 41 nodes (n_gen=3 sizing) to 57 nodes (n_gen=5
sizing). More time nodes means a bigger position-NLP for *every*
generation, including gens 1-3 that previously converged fine — so
outer=40/pos_iters=100 is adequate at 41 nodes and inadequate at 57. Same
class of bug (optimizer budget too low for problem size), triggered by grid
growth instead of generation count directly.

**Consequence: n_gen can't be pushed higher without re-scaling budget with
it.** The n_gen=3 results already recorded above (both bases, clean
monotone decay) stand — they were run at a grid size where the budget was
verified adequate. The n_gen=5 numbers here are not a finding, they're
another under-convergence artifact and should not be read as "decay
flattens" or anything else substantive.

## n_gen=5 retry at outer=80/pos_iters=200/seeds=6 — feasibility fixed at gen1-2, NEW failure at gen3-5

Bumped outer 40->80, pos_iters 100->200, seeds 5->6 (same 57-node grid).
Result:

```
dJk:       [+0.1078, +0.0105, +0.0358, +0.0112, +0.0009]
guard_dJk: [+0.0412, +0.0379, +0.0403, +0.0554, +0.0477]
```

Gen1-2 now mostly feasible (up from 4/5 seeds infeasible before). But gen3,
4, 5 report `feasible=False` for **both** the windowed and guard arms,
across **all 6 seeds** — a different failure mode from the earlier
starvation pattern (jump_mean is often nonzero here, e.g. gen2 g-kink
jump_mean=0.338), so this isn't the same "kink got zero weight" bug. Also
notable: gen1's windowed dJk now exceeds the guard's (+0.1078 vs +0.0412) —
inverted from every prior run, where guard always won. Something shifted
structurally, not just noise.

**Reading:** raising outer/pos_iters fixed the grid-size problem (41->57
nodes) but not a second, distinct problem: each generation appends a kink,
so accumulated kink count grows every generation (base 4f+3g -> up to
9f+8g by gen5) while `outer`/`pos_iters` stayed fixed for the whole ladder.
The position-NLP dimensionality grows with accumulated kink count, not just
grid nodes, so the same per-generation budget likely becomes marginal again
by gen3+. Can't rule out an unrelated resolution floor at very tiny windows
(w_5=0.03) without another escalation. **Not chasing this further without
checking in** — cost is compounding (already 2x the prior budget and still
failing past gen2) and this is the third consecutive escalation.

**Verdict for now:** the n_gen=3 clean results on both bases (this file,
above) are the trustworthy ceiling of this measurement so far. n_gen=4-5
needs either escalating budget with the generation count (not a flat value
for the whole ladder) or a different structural fix; gen3+ numbers from
both n_gen=5 attempts should be treated as noise, not signal.

## What to run next (not yet done)

1. **Push past n_gen=3 correctly, with per-generation-scaling budget**: try
   `outer`/`pos_iters` that grow with `k` (accumulated kink count), rather
   than one flat value for the whole ladder — e.g. `outer_k = outer0 * (1 +
   0.3*k)`. Check feasibility rate and jump_mean recover at gen3-5 before
   trusting any dJk from it.
2. **Same for the Run-3 base** if pushed past n_gen=3.
3. ~~Read the diagnostics for self-similarity~~ — done, no signal.
4. ~~Try a different base~~ — done (Run 6's grown topology); Run 5's
   wider-multistart frontier (6+5) remains an option for a third base if
   more confirmation is wanted.
5. ~~Re-check the Run-3-base ladder's own budget~~ — done above, fixed the
   same artifact pattern, result now clean.

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
